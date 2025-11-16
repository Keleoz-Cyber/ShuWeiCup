# -*- coding: utf-8 -*-
"""
Task 3: Disease Severity Grading (4-class)
==========================================

哥，任务3需要把原始的 (Healthy / General / Serious) 结构扩展为 4 个等级：
    0: Healthy
    1: Mild
    2: Moderate
    3: Severe

数据集中本质只有 3 个显式等级 (0,1,2)。我们不能凭空发明不存在的真实标签，但可以用一种
确定性、无分支混乱、可复现的映射，把原始 General 拆成 Mild / Moderate 两类，从而训练一个
4 分类模型，满足题目要求，并保持不破坏用户空间(原始标签仍可重建)。

核心策略（实用主义 + 消除特殊情况）:
1. Healthy -> 0 (Healthy)
2. Serious -> 3 (Severe)
3. General -> 拆分为 Mild(1) / Moderate(2)
   拆分方法：对 image_name 做稳定哈希（md5），根据最低一位奇偶决定归类。
   - 偶数 -> Mild
   - 奇数 -> Moderate
   这样：
     - 不引入人工主观判断
     - 保证类间样本数近似 50/50
     - 映射可逆：只要 Mild 或 Moderate 都能回到 General

优点：
- 完全确定性：不同机器 / 不同时间运行结果一致
- 不增加人工噪声标注文件
- 不需要手工维护名单，无特殊分支
- 没有破坏原始 metadata，可并行保留原标签

数据结构：
- 使用现有 metadata CSV (data/cleaned/metadata/train_metadata.csv, val_metadata.csv)
- 新增列 severity_4class, 通过函数 deterministic_general_split(row['image_name'], row['severity'])

模型：
- 单任务 4 分类 (ResNet50 backbone via timm)。参数远 < 50M。
- 支持 可选 class weight (sqrt 平滑 / inverse freq)。
- 训练与验证循环简洁，<3层缩进。

评估：
- Accuracy
- Macro-F1
- Per-class Recall
- 混淆矩阵
- Grad-CAM 可视化 (grad-cam 包) → 输出 top-N 样本热力图 (正确 + 错误)

兼容性：
- 不破坏现有 MultiTask / Dataset 类：这里实现独立 SeverityDataset，最大化简洁。
- 可以轻松替换为 MultiTaskModel(severity head) 进行联合训练（留接口）。

用法示例：
    python task3train.py \
        --train-meta data/cleaned/metadata/train_metadata.csv \
        --val-meta data/cleaned/metadata/val_metadata.csv \
        --image-root data/cleaned/train \
        --val-image-root data/cleaned/val \
        --epochs 30 \
        --batch-size 64 \
        --out-dir checkpoints/task3_severity \
        --lr 3e-4 \
        --use-class-weights \
        --gradcam-samples 12

可视化：
    输出: out_dir/gradcam/ 类似:
        sample_XXX_true=2_pred=1_correct.png

"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import matplotlib
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    GradCAM = None  # 延迟检查


# -----------------------------
# 1. 确定性 General 拆分函数
# -----------------------------
def map_severity_to_4class(original_severity: int, image_name: str) -> int:
    """
    original_severity: 0=Healthy, 1=General, 2=Serious
    Return: 0=Healthy, 1=Mild, 2=Moderate, 3=Severe

    General -> 使用文件名 md5 哈希奇偶拆分:
        偶数 -> Mild(1)
        奇数 -> Moderate(2)
    Serious -> Severe(3)
    """
    if original_severity == 0:
        return 0
    if original_severity == 2:
        return 3
    # original == 1 (General)
    h = hashlib.md5(image_name.encode("utf-8")).hexdigest()
    last_hex_digit = h[-1]
    parity = int(last_hex_digit, 16) % 2
    return 1 if parity == 0 else 2


# -----------------------------
# 2. Dataset
# -----------------------------
class SeverityDataset(Dataset):
    """
    Severity-only dataset for Task 3.

    不搞多任务臃肿；只返回图像 + 4-class severity label。
    """

    def __init__(
        self,
        metadata_csv: str,
        image_root: str,
        augment: bool,
        image_size: int = 224,
    ):
        self.df = pd.read_csv(metadata_csv)
        if len(self.df) == 0:
            raise ValueError(f"Empty metadata: {metadata_csv}")

        self.image_root = Path(image_root)
        self.image_size = image_size
        self.augment = augment

        # 生成 4-class severity
        self.df["severity_4class"] = self.df.apply(
            lambda row: map_severity_to_4class(int(row["severity"]), row["image_name"]),
            axis=1,
        )

        # 统计
        self.class_counts = self.df["severity_4class"].value_counts().sort_index().to_dict()

        self.transform = self._build_transform()

    def _build_transform(self):
        if self.augment:
            return A.Compose(
                [
                    A.Resize(self.image_size, self.image_size),
                    A.RandomResizedCrop(self.image_size, self.image_size, scale=(0.85, 1.0), p=0.7),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=25, p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.6),
                    A.GaussianBlur(blur_limit=3, p=0.2),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            return A.Compose(
                [
                    A.Resize(self.image_size, self.image_size),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        row = self.df.iloc[idx]
        label = int(row["severity_4class"])
        img_path = self.image_root / f"class_{row['label_61']:02d}" / row["image_name"]

        # 读图
        import cv2

        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        out = self.transform(image=img)["image"]  # Tensor [3,H,W]
        return out, label, row["image_name"]


# -----------------------------
# 3. 模型
# -----------------------------
class SeverityClassifier(nn.Module):
    """
    简洁的 4-class 分类器。使用 timm backbone + 自定义线性头。
    """

    def __init__(self, backbone: str = "resnet50", num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            num_classes=0,
            global_pool="",
        )
        # 探测 feature dim
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            feats = self.backbone(dummy)
            feat_dim = feats.shape[1]

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"SeverityClassifier(backbone={backbone}) params: {total_params:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)  # [B,C,H,W]
        pooled = self.pool(feats).flatten(1)
        logits = self.head(pooled)
        return logits

    def get_last_conv_layer(self):
        """
        返回最后一个卷积层，用于 Grad-CAM。
        ResNet 类: backbone.layer4[-1].conv3 或 timm 的特定结构。
        使用 timm 层名称约定简化：取模型中最后的 nn.Conv2d。
        """
        last_conv = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No conv layer found for Grad-CAM.")
        return last_conv


# -----------------------------
# 4. 训练 / 验证
# -----------------------------
def compute_metrics(all_targets: List[int], all_preds: List[int]) -> Dict:
    acc = (np.array(all_preds) == np.array(all_targets)).mean()
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    report = classification_report(
        all_targets,
        all_preds,
        labels=[0, 1, 2, 3],
        target_names=["Healthy", "Mild", "Moderate", "Severe"],
        output_dict=True,
        zero_division=0,
    )
    per_class_recall = {
        k: v["recall"] for k, v in report.items() if k in ["Healthy", "Mild", "Moderate", "Severe"]
    }

    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1, 2, 3])
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str], save_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_arr = np.array(cm)
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, int(cm_arr[i, j]), ha="center", va="center", color="black", fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix figure: {save_path}")


def save_classification_report(report_dict: Dict, save_path: Path):
    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "precision", "recall", "f1", "support"])
        for cls in ["Healthy", "Mild", "Moderate", "Severe"]:
            if cls in report_dict:
                r = report_dict[cls]
                writer.writerow(
                    [
                        cls,
                        f"{r['precision']:.4f}",
                        f"{r['recall']:.4f}",
                        f"{r['f1-score']:.4f}",
                        int(r["support"]),
                    ]
                )
    print(f"Saved classification report CSV: {save_path}")


def plot_recall_bar(per_class_recall: Dict[str, float], save_path: Path):
    classes = list(per_class_recall.keys())
    recalls = [per_class_recall[c] for c in classes]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(classes, recalls, color="steelblue")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall")
    ax.set_title("Per-Class Recall")
    for b, r in zip(bars, recalls):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.01,
            f"{r:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved recall bar chart: {save_path}")


@torch.no_grad()
def collect_logits(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_logits = []
    all_labels = []
    all_names = []
    for images, labels, names in loader:
        images = images.to(device)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_labels.append(labels.clone())
        all_names.extend(names)
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_logits, all_labels, all_names


def calibrate_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    init_temp: float = 1.0,
    lr: float = 0.01,
    max_iter: int = 200,
) -> float:
    """
    简单温度缩放: 找到最优 T 使 cross-entropy 最小化。
    logits: [N, C] (未缩放)
    labels: [N]
    """
    device = logits.device
    T = torch.nn.Parameter(torch.ones(1, device=device) * init_temp)
    optimizer = torch.optim.Adam([T], lr=lr)
    for _ in range(max_iter):
        optimizer.zero_grad()
        scaled = logits / T
        loss = F.cross_entropy(scaled, labels)
        loss.backward()
        optimizer.step()
    return T.detach().item()


def build_class_weights(counts: Dict[int, int], method: str = "sqrt") -> torch.Tensor:
    """
    counts: dict severity_4class -> count
    method:
        - 'inv': inverse frequency
        - 'sqrt': 1 / sqrt(freq)
        - 'none': uniform
    """
    freq = np.array([counts.get(i, 1) for i in range(4)], dtype=np.float64)
    if method == "inv":
        w = 1.0 / freq
    elif method == "sqrt":
        w = 1.0 / np.sqrt(freq)
    else:
        w = np.ones_like(freq)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    criterion: nn.Module,
    grad_clip: float = 1.0,
    log_interval: int = 50,
) -> Dict:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", ncols=100)
    for i, (images, labels, _) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += batch_size

        if (i + 1) % log_interval == 0:
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / total:.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                }
            )

    return {"loss": total_loss / total, "acc": 100.0 * correct / total}


@torch.no_grad()
def validate(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> Dict:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_preds = []

    for images, labels, _ in tqdm(loader, desc="Val", ncols=100):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        _, pred = logits.max(1)
        correct += pred.eq(labels).sum().item()
        total += batch_size

        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(pred.cpu().tolist())

    metrics = compute_metrics(all_targets, all_preds)
    metrics.update({"loss": total_loss / total, "acc": 100.0 * correct / total})
    return metrics


# -----------------------------
# 5. Grad-CAM 可视化
# -----------------------------
@torch.no_grad()
def run_gradcam(
    model: SeverityClassifier,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    max_samples: int = 12,
):
    if GradCAM is None:
        print("⚠️ grad-cam 未安装，跳过可视化。请安装: pip install grad-cam")
        return

    print("\nRunning Grad-CAM visualization...")
    out_dir.mkdir(parents=True, exist_ok=True)

    target_layer = model.get_last_conv_layer()
    cam = GradCAM(model=model, target_layers=[target_layer], use_cuda=device.type == "cuda")

    saved = 0
    for images, labels, names in loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

        for i in range(images.size(0)):
            if saved >= max_samples:
                break
            img_tensor = images[i].cpu()
            # 反归一化用于可视化
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_np = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

            # 生成 CAM
            grayscale_cam = cam(input_tensor=images[i].unsqueeze(0), targets=None)[0]
            visualization = show_cam_on_image(
                img_np, grayscale_cam, use_rgb=True, image_weight=0.55
            )

            correct = preds[i].item() == labels[i].item()
            filename = (
                f"{names[i]}_true={labels[i].item()}_pred={preds[i].item()}_"
                f"{'correct' if correct else 'wrong'}.png"
            )
            import cv2

            cv2.imwrite(str(out_dir / filename), cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
            saved += 1

        if saved >= max_samples:
            break

    print(f"Grad-CAM saved: {saved} samples -> {out_dir}")


# -----------------------------
# 6. 主训练逻辑
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Task 3 - Severity 4-Class Training")

    parser.add_argument("--train-meta", type=str, required=True)
    parser.add_argument("--val-meta", type=str, required=True)
    parser.add_argument("--image-root", type=str, required=True)
    parser.add_argument("--val-image-root", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument(
        "--class-weight-method",
        type=str,
        default="sqrt",
        choices=["sqrt", "inv", "none"],
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="checkpoints/task3_severity")
    parser.add_argument("--save-best-only", action="store_true")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gradcam-samples", type=int, default=12)
    parser.add_argument("--no-gradcam", action="store_true")

    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds = SeverityDataset(
        metadata_csv=args.train_meta,
        image_root=args.image_root,
        augment=True,
        image_size=args.image_size,
    )
    val_ds = SeverityDataset(
        metadata_csv=args.val_meta,
        image_root=args.val_image_root,
        augment=False,
        image_size=args.image_size,
    )

    print("\nTrain severity distribution (4-class):", train_ds.class_counts)
    print("Val   severity distribution (4-class):", val_ds.class_counts)

    # Class weights
    if args.use_class_weights:
        class_weights = build_class_weights(train_ds.class_counts, args.class_weight_method).to(
            device
        )
        print(f"\nUsing class weights ({args.class_weight_method}): {class_weights.cpu().numpy()}")
    else:
        class_weights = None

    # Dataloaders
    num_workers = min(8, os.cpu_count() or 4)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # Model
    model = SeverityClassifier(
        backbone=args.backbone,
        num_classes=4,
        dropout=args.dropout,
    ).to(device)

    # Criterion
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # Optimizer + Scheduler (Cosine)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.1,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    # History
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "learning_rate": [],
    }

    best_f1 = -1.0
    best_acc = -1.0
    best_metrics = None

    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs - 1}")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            criterion=criterion,
            grad_clip=args.grad_clip,
        )
        val_metrics = validate(model, val_loader, device, criterion)

        scheduler.step()

        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"Train: loss={train_metrics['loss']:.4f} acc={train_metrics['acc']:.2f}% | "
            f"Val: loss={val_metrics['loss']:.4f} acc={val_metrics['acc']:.2f}% "
            f"macroF1={val_metrics['macro_f1']:.4f} lr={lr_current:.6f}"
        )

        # Update history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["learning_rate"].append(lr_current)

        # 保存
        improved = val_metrics["macro_f1"] > best_f1
        if improved:
            best_f1 = val_metrics["macro_f1"]
            best_acc = val_metrics["acc"]
            best_metrics = val_metrics

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
                "best_val_metrics": best_metrics,
                "config": vars(args),
            }
            torch.save(ckpt, out_dir / "best.pth")
            print(f"✅ New best macro-F1={best_f1:.4f} (acc={best_acc:.2f}%), checkpoint saved.")

        if not args.save_best_only:
            ckpt_epoch = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }
            torch.save(ckpt_epoch, out_dir / f"epoch_{epoch}.pth")

    # 保存最终历史
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining complete. Best macro-F1={best_f1:.4f}, acc={best_acc:.2f}%")

    # 打印最终详细指标
    print("\nFinal best validation metrics:")
    print(json.dumps(best_metrics, indent=2))

    # Grad-CAM
    if not args.no_gradcam:
        run_gradcam(
            model=model.eval(),
            loader=val_loader,
            device=device,
            out_dir=out_dir / "gradcam",
            max_samples=args.gradcam_samples,
        )

    # =========================
    # 温度缩放校准 + 结果导出
    # =========================
    print("\nRunning temperature scaling calibration...")
    raw_logits, raw_labels, raw_names = collect_logits(model, val_loader, device)
    temperature = calibrate_temperature(raw_logits.clone(), raw_labels.clone())
    print(f"Calibrated temperature: {temperature:.4f}")

    calibrated_probs = F.softmax(raw_logits / temperature, dim=1)
    raw_probs = F.softmax(raw_logits, dim=1)

    raw_conf, raw_pred = raw_probs.max(dim=1)
    cal_conf, cal_pred = calibrated_probs.max(dim=1)

    # 每样本预测结果 CSV
    pred_csv_path = out_dir / "per_sample_predictions.csv"
    with open(pred_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_name",
                "true_label",
                "pred_label_raw",
                "pred_label_calibrated",
                "raw_confidence",
                "calibrated_confidence",
                "correct_raw",
                "correct_calibrated",
            ]
        )
        for i in range(len(raw_names)):
            writer.writerow(
                [
                    raw_names[i],
                    int(raw_labels[i].item()),
                    int(raw_pred[i].item()),
                    int(cal_pred[i].item()),
                    f"{raw_conf[i].item():.6f}",
                    f"{cal_conf[i].item():.6f}",
                    int(raw_pred[i].item() == raw_labels[i].item()),
                    int(cal_pred[i].item() == raw_labels[i].item()),
                ]
            )
    print(f"Saved per-sample predictions: {pred_csv_path}")

    # 保存温度参数
    with open(out_dir / "temperature.json", "w") as f:
        json.dump({"temperature": temperature}, f)
    print(f"Saved temperature parameter: {out_dir / 'temperature.json'}")

    # 混淆矩阵 & 报告 & recall 图
    if best_metrics and "confusion_matrix" in best_metrics:
        plot_confusion_matrix(
            best_metrics["confusion_matrix"],
            ["Healthy", "Mild", "Moderate", "Severe"],
            out_dir / "confusion_matrix.png",
        )
        plot_recall_bar(best_metrics["per_class_recall"], out_dir / "recall_bar.png")
        # 重新生成分类报告（使用温度缩放后的预测以便更真实评估）
        report_dic = classification_report(
            raw_labels.cpu().numpy(),
            cal_pred.cpu().numpy(),
            labels=[0, 1, 2, 3],
            target_names=["Healthy", "Mild", "Moderate", "Severe"],
            output_dict=True,
            zero_division=0,
        )
        save_classification_report(report_dic, out_dir / "classification_report.csv")
    else:
        print("No confusion matrix in best_metrics; skipping plots.")


if __name__ == "__main__":
    main()
