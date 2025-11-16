#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4: Multi-Task Joint Learning and Interpretable Diagnosis
=============================================================

哥，本脚本完成多任务联合训练（疾病61类 + 作物10类 + 疾病类型(含None) + 严重程度4类），
并支持生成可读诊断报告、以及“多任务 vs 严重度单任务”的协同效应对比。

要点：
- 严重度4类：0=Healthy，1=Mild，2=Moderate，3=Severe
  数据原始只有0/1/2(健康/一般/严重)。我们采用确定性的哈希拆分把General拆成 Mild/Moderate。
  不引入人工噪声；映射稳定可复现。
- 联合训练：共享backbone，四个head，权重可调（--task-weights）。
- 报告生成：每张图输出 61类Top3、Crop、Disease、Severity 的预测与置信度；可选Grad-CAM。
- 协同效应对比：可选启用 --compare-synergy，在同一脚本中再跑一轮“仅严重度任务”的训练，
  比较严重度的精度与F1，生成 synergy_comparison.json。

依赖：
- 复用本仓库已有模块：dataset.py（变换参考）、models.py（MultiTaskModel）、losses.py（MultiTaskLoss）、trainer.py（Trainer）
- 可选 grad-cam 库用于可视化（pyproject 已包含）

示例：
  python task4train.py \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --epochs 30 \
    --batch-size 64 \
    --lr 3e-4 \
    --out-dir checkpoints/task4_multitask \
    --use-class-weights \
    --task-weights 1.0,0.3,0.3,0.6 \
    --compare-synergy \
    --compare-epochs 10 \
    --report-samples 30 \
    --cam-samples 12

"""

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 减少 Albumentations 版本检查网络请求噪音
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

# Project modules
# Removed unused get_train_transform, get_val_transform imports
from losses import MultiTaskLoss
from models import MultiTaskModel
from trainer import Trainer

# Grad-CAM (optional)
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    GradCAM = None
    show_cam_on_image = None

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


# =========================
# Severity 4-class mapping
# =========================
def map_severity_to_4class(original_severity: int, image_name: str) -> int:
    """
    0(H) -> 0, 2(Serious) -> 3, 1(General) -> [Mild(1) or Moderate(2)] by deterministic hashing.
    """
    if original_severity == 0:
        return 0
    if original_severity == 2:
        return 3
    h = hashlib.md5(image_name.encode("utf-8")).hexdigest()
    last_hex_digit = h[-1]
    parity = int(last_hex_digit, 16) % 2
    return 1 if parity == 0 else 2


# =========================
# Dataset (multi-task, sev4)
# =========================
class MultiTaskSeverity4Dataset(Dataset):
    """
    读取元数据CSV，图像组织为 data_dir/class_XX/image_name
    返回：
      image: Tensor [3,H,W]
      labels: dict(label_61, crop, disease, severity[4-class])
      meta: dict 保留 image_name 供报告/可视化
    """

    def __init__(
        self,
        data_dir: str,
        metadata_csv: str,
        augment: bool,
        image_size: int = 224,
    ):
        import pandas as pd

        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(metadata_csv)
        if len(self.df) == 0:
            raise ValueError(f"Empty metadata: {metadata_csv}")

        self.df["severity4"] = self.df.apply(
            lambda r: map_severity_to_4class(int(r["severity"]), r["image_name"]), axis=1
        )

        self.augment = augment
        self.image_size = image_size
        self.transform = self._build_transform(augment, image_size)

        # 分布统计
        self.class_counts = self.df["severity4"].value_counts().sort_index().to_dict()

    def _build_transform(self, augment: bool, image_size: int) -> A.Compose:
        if augment:
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.RandomResizedCrop(size=(image_size, image_size), scale=(0.85, 1.0), p=0.7),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=25, p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.6),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        label_61 = int(row["label_61"])
        crop_id = int(row["crop_id"])
        disease_id = int(row["disease_id"])
        severity4 = int(row["severity4"])

        class_folder = f"class_{label_61:02d}"
        image_path = self.data_dir / class_folder / row["image_name"]

        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)["image"]

        labels = {
            "label_61": label_61,
            "crop": crop_id,
            "disease": disease_id,
            "severity": severity4,  # already 4-class
        }
        meta = {
            "image_name": row["image_name"],
            "image_path": str(image_path),
        }
        return img, labels, meta


def multitask_collate(batch):
    images = torch.stack([b[0] for b in batch])
    labels = {}
    for key in ["label_61", "crop", "disease", "severity"]:
        labels[key] = torch.tensor([int(b[1][key]) for b in batch], dtype=torch.long)
    meta = [b[2] for b in batch]
    return images, labels, meta


# =========================
# Diagnostic report
# =========================
@dataclass
class DiagnosticSample:
    image_name: str
    label_61_pred: int
    label_61_conf: float
    label_61_top3: List[Tuple[int, float]]
    crop_pred: int
    crop_conf: float
    disease_pred: int
    disease_conf: float
    severity_pred: int
    severity_conf: float
    correct_61: Optional[bool] = None
    correct_sev: Optional[bool] = None


def softmax_confidence(logits: torch.Tensor) -> Tuple[int, float]:
    """Return (pred_class, confidence) with explicit casting."""
    probs = F.softmax(logits, dim=-1)
    conf, pred = torch.max(probs, dim=-1)
    return int(pred.item()), float(conf.item())


def topk_from_logits(logits: torch.Tensor, k: int = 3) -> List[Tuple[int, float]]:
    """Return top-k (class_id, confidence) tuples."""
    probs = F.softmax(logits, dim=-1)
    confs, preds = torch.topk(probs, k=min(k, probs.size(-1)), dim=-1)
    return [(int(preds[0, i].item()), float(confs[0, i].item())) for i in range(confs.size(1))]


@torch.no_grad()
def generate_diagnostic_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_csv: Path,
    max_samples: Optional[int] = None,
    cam_dir: Optional[Path] = None,
    cam_samples: int = 0,
):
    model.eval()
    rows = []
    # Removed unused 'samples' variable

    # Optional Grad-CAM
    cam = None
    if (
        cam_dir is not None
        and cam_samples > 0
        and GradCAM is not None
        and hasattr(model, "get_last_conv_layer")
    ):
        cam_dir.mkdir(parents=True, exist_ok=True)
        try:
            target_layer = model.get_last_conv_layer()
            cam = GradCAM(model=model, target_layers=[target_layer])
        except Exception:
            cam = None

    saved_cam = 0
    n_seen = 0

    for images, labels, metas in loader:
        images = images.to(device)
        outputs = model(images)
        # outputs: dict with keys label_61, crop, disease, severity

        for i in range(images.size(0)):
            n_seen += 1
            l61_logits = outputs["label_61"][i : i + 1]
            crop_logits = outputs["crop"][i : i + 1]
            dis_logits = outputs["disease"][i : i + 1]
            sev_logits = outputs["severity"][i : i + 1]

            l61_pred, l61_conf = softmax_confidence(l61_logits[0])
            crop_pred, crop_conf = softmax_confidence(crop_logits[0])
            dis_pred, dis_conf = softmax_confidence(dis_logits[0])
            sev_pred, sev_conf = softmax_confidence(sev_logits[0])
            l61_top3 = topk_from_logits(l61_logits, k=3)

            row = {
                "image_name": metas[i]["image_name"],
                "pred_label_61": l61_pred,
                "conf_label_61": f"{l61_conf:.6f}",
                "top3_label_61": json.dumps(l61_top3),
                "pred_crop": crop_pred,
                "conf_crop": f"{crop_conf:.6f}",
                "pred_disease": dis_pred,
                "conf_disease": f"{dis_conf:.6f}",
                "pred_severity4": sev_pred,
                "conf_severity4": f"{sev_conf:.6f}",
            }

            # If true labels available in loader, append correctness
            if "label_61" in labels and "severity" in labels:
                true_l61 = int(labels["label_61"][i].item())
                true_sev = int(labels["severity"][i].item())
                row["true_label_61"] = true_l61
                row["true_severity4"] = true_sev
                row["correct_61"] = int(true_l61 == l61_pred)
                row["correct_severity4"] = int(true_sev == sev_pred)

            rows.append(row)

            # Grad-CAM visualization for a few samples
            if cam is not None and saved_cam < cam_samples:
                img = images[i].detach().cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img_np = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

                # Grad-CAM requires gradients; enable explicitly
                grayscale_cam = cam(input_tensor=images[i].unsqueeze(0))[0]
                if callable(show_cam_on_image):
                    overlay = show_cam_on_image(
                        img_np, grayscale_cam, use_rgb=True, image_weight=0.55
                    )
                    vis = (overlay[:, :, ::-1] * 255).astype(np.uint8)  # to BGR uint8
                else:
                    vis = (img_np[:, :, ::-1] * 255).astype(np.uint8)

                vis_name = f"{metas[i]['image_name']}_sev={sev_pred}_conf={sev_conf:.2f}.jpg"
                cv2.imwrite(str(cam_dir / vis_name), vis)
                saved_cam += 1

            if max_samples is not None and len(rows) >= max_samples:
                break

        if max_samples is not None and len(rows) >= max_samples:
            break

    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        fieldnames = (
            list(rows[0].keys())
            if rows
            else [
                "image_name",
                "pred_label_61",
                "conf_label_61",
                "top3_label_61",
                "pred_crop",
                "conf_crop",
                "pred_disease",
                "conf_disease",
                "pred_severity4",
                "conf_severity4",
            ]
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Diagnostic report saved to: {out_csv} (rows={len(rows)})")


# =========================
# Severity evaluation utils
# =========================
@torch.no_grad()
def evaluate_severity_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    model.eval()
    all_t, all_p = [], []
    for images, labels, _ in loader:
        images = images.to(device)
        logits = model(images)["severity"]
        probs = F.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).cpu().tolist()
        true = labels["severity"].cpu().tolist()
        all_t.extend(true)
        all_p.extend(pred)
    if len(all_t) == 0:
        # Guard empty loader case
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": [[0, 0, 0, 0] for _ in range(4)],
        }
    acc = (np.array(all_t) == np.array(all_p)).mean()
    macro = f1_score(all_t, all_p, average="macro")
    cm = confusion_matrix(all_t, all_p, labels=[0, 1, 2, 3]).tolist()
    return {"accuracy": acc, "macro_f1": macro, "confusion_matrix": cm}


# =========================
# Main
# =========================
def parse_task_weights(s: str) -> Dict[str, float]:
    parts = [float(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("--task-weights must be 'w61,wcrop,wdisease,wseverity'")
    return {
        "label_61": parts[0],
        "crop": parts[1],
        "disease": parts[2],
        "severity": parts[3],
    }


def main():
    parser = argparse.ArgumentParser(description="Task 4: Multi-Task Joint Training + Diagnosis")

    # Data
    parser.add_argument("--train-meta", type=str, required=True)
    parser.add_argument("--val-meta", type=str, required=True)
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)

    # Train hyperparams
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Loss / weights
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Apply class weights for 61-class and severity",
    )
    parser.add_argument(
        "--class-weights-61",
        type=str,
        default=None,
        help="CSV path with 'class_id,weight' for 61 classes",
    )
    parser.add_argument(
        "--class-weights-severity",
        type=str,
        default=None,
        help="CSV path with 'class_id,weight' for severity 4-class",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument(
        "--task-weights", type=str, default="1.0,0.3,0.3,0.6", help="w61,wcrop,wdisease,wseverity"
    )

    # Output
    parser.add_argument("--out-dir", type=str, default="checkpoints/task4_multitask")
    parser.add_argument("--save-freq", type=int, default=5)

    # Diagnosis
    parser.add_argument(
        "--report-samples",
        type=int,
        default=50,
        help="Max samples to include in diagnostic report CSV",
    )
    parser.add_argument(
        "--cam-samples",
        type=int,
        default=12,
        help="Number of Grad-CAM overlays to save (0 to disable)",
    )

    # Synergy comparison
    parser.add_argument(
        "--compare-synergy",
        action="store_true",
        help="Run a second training phase with severity-only task and compare",
    )
    parser.add_argument("--compare-epochs", type=int, default=10)

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets / loaders
    train_ds = MultiTaskSeverity4Dataset(
        data_dir=args.train_dir,
        metadata_csv=args.train_meta,
        augment=True,
        image_size=args.image_size,
    )
    val_ds = MultiTaskSeverity4Dataset(
        data_dir=args.val_dir, metadata_csv=args.val_meta, augment=False, image_size=args.image_size
    )
    print("\nSeverity(4) distribution:")
    print(f"  Train: {train_ds.class_counts}")
    print(f"  Val  : {val_ds.class_counts}")

    num_workers = min(8, os.cpu_count() or 4)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=multitask_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
        collate_fn=multitask_collate,
    )

    # Model
    model = MultiTaskModel(
        backbone=args.backbone,
        pretrained=True,
        dropout=args.dropout,
        num_classes_61=61,
        num_crops=10,
        num_diseases=28,
        num_severity=4,
    ).to(device)

    # Class weights (optional)
    class_weights_61: Optional[torch.Tensor] = None
    class_weights_sev: Optional[torch.Tensor] = None
    # Removed stray placeholder referencing non-existent variable.

    if args.use_class_weights:
        import pandas as pd

        if args.class_weights_61 and Path(args.class_weights_61).exists():
            df = pd.read_csv(args.class_weights_61)
            w = torch.tensor(df["weight"].values, dtype=torch.float32, device=device)
            if w.numel() == 61:
                class_weights_61 = w
                print(f"Loaded class weights for 61-class: shape={tuple(w.shape)}")
            else:
                print("Warning: class_weights_61 length mismatch; ignored.")
        if args.class_weights_severity and Path(args.class_weights_severity).exists():
            df = pd.read_csv(args.class_weights_severity)
            w = torch.tensor(df["weight"].values, dtype=torch.float32, device=device)
            if w.numel() == 4:
                class_weights_sev = w
                print(f"Loaded class weights for severity: shape={tuple(w.shape)}")
            else:
                print("Warning: class_weights_severity length mismatch; ignored.")

    # Loss
    task_w = parse_task_weights(args.task_weights)
    criterion = MultiTaskLoss(
        task_weights=task_w,
        class_weights_61=class_weights_61,
        class_weights_severity=class_weights_sev,
        label_smoothing=args.label_smoothing,
    )

    # Optimizer / Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        save_dir=str(out_dir / "multitask"),
        use_amp=(args.amp and device.type == "cuda"),
        log_interval=20,
        use_tensorboard=True,
        multi_task=True,
    )

    # Train multitask
    trainer.train(num_epochs=args.epochs, save_freq=args.save_freq)

    # Evaluate severity metrics (post-train) and write report
    print("\nEvaluating severity metrics (multitask)...")
    sev_stats = evaluate_severity_metrics(model, val_loader, device)
    with open(out_dir / "severity_metrics_multitask.json", "w") as f:
        json.dump(sev_stats, f, indent=2)
    print(f"Saved severity metrics: {out_dir / 'severity_metrics_multitask.json'}")

    # Diagnostic report (CSV) + optional CAM
    generate_diagnostic_report(
        model=model,
        loader=val_loader,
        device=device,
        out_csv=out_dir / "diagnostic_report_multitask.csv",
        max_samples=args.report_samples,
        cam_dir=(out_dir / "cam_multitask") if args.cam_samples > 0 else None,
        cam_samples=args.cam_samples,
    )

    # =========================
    # Synergy comparison (optional)
    # =========================
    if args.compare_synergy:
        print("\n" + "=" * 60)
        print("Running severity-only training for synergy comparison")
        print("=" * 60)

        # New model (same arch), but loss weights focus on severity
        model2 = MultiTaskModel(
            backbone=args.backbone,
            pretrained=True,
            dropout=args.dropout,
            num_classes_61=61,
            num_crops=10,
            num_diseases=28,
            num_severity=4,
        ).to(device)

        optimizer2 = torch.optim.AdamW(
            model2.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=args.compare_epochs, eta_min=args.lr * 0.1
        )
        # severity-only weights
        sev_only_weights = {"label_61": 0.0, "crop": 0.0, "disease": 0.0, "severity": 1.0}
        criterion2 = MultiTaskLoss(
            task_weights=sev_only_weights,
            class_weights_61=None,
            class_weights_severity=class_weights_sev,
            label_smoothing=args.label_smoothing,
        )

        trainer2 = Trainer(
            model=model2,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer2,
            scheduler=scheduler2,
            criterion=criterion2,
            device=device,
            save_dir=str(out_dir / "severity_only"),
            use_amp=(args.amp and device.type == "cuda"),
            log_interval=20,
            use_tensorboard=True,
            multi_task=True,
        )

        trainer2.train(num_epochs=args.compare_epochs, save_freq=max(1, args.compare_epochs // 2))

        # Evaluate severity metrics
        sev_stats2 = evaluate_severity_metrics(model2, val_loader, device)
        with open(out_dir / "severity_metrics_severity_only.json", "w") as f:
            json.dump(sev_stats2, f, indent=2)

        # Comparison summary
        comparison = {
            "multitask": sev_stats,
            "severity_only": sev_stats2,
            "delta_accuracy": (sev_stats["accuracy"] - sev_stats2["accuracy"]),
            "delta_macro_f1": (sev_stats["macro_f1"] - sev_stats2["macro_f1"]),
        }
        with open(out_dir / "synergy_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        print(f"Synergy comparison saved: {out_dir / 'synergy_comparison.json'}")


if __name__ == "__main__":
    main()
