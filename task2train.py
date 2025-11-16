#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 Few-Shot Training (ArcFace + Prototype EMA)
==================================================

Clean, robust, few-shot oriented trainer with:
- CUDA/MPS-safe AMP (AMP only on CUDA; MPS/CPU use FP32)
- Picklable collate function (no lambdas)
- EMA prototypes with configurable refresh interval
- Delayed, selective unfreeze of backbone layer4 (Strategy F)
- Defaults implementing strategies A–F:
  A. head-lr-scale = 3.0
  B. dropout = 0.3
  C. mixup-disable-epoch = 8
  D. label-smoothing = 0.05
  E. proto-weight = 0.4
  F. delayed unfreeze of layer4 at epoch 3; layer3 remains frozen
- Stable Albumentations transforms (Resize + Flip + mild ColorJitter + Normalize)
- Adaptive DataLoader workers (2 on CUDA; 0 on MPS/CPU) and pin_memory only on CUDA
- Full logging, TensorBoard, PNG curves, confusion matrix CSV/PNG, and history CSV

Usage Example:
  python task2train.py \
    --train-meta data/cleaned/metadata/train_metadata_fewshot_10.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --backbone resnet50 \
    --epochs 30 \
    --batch-size 8 \
    --lr 3e-4 \
    --head-lr-scale 3.0 \
    --proto-weight 0.4 \
    --arcface-margin 0.30 \
    --arcface-scale 30.0 \
    --image-size 256 \
    --mixup-alpha 0.2 \
    --mixup-disable-epoch 8 \
    --label-smoothing 0.05 \
    --proto-ema 0.7 \
    --proto-refresh-interval 1 \
    --unfreeze-epoch 3 \
    --freeze-stage12 \
    --save-dir checkpoints/task2_fewshot
"""

import argparse
import contextlib
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

# Suppress albumentations version check warnings (no network needed)
os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from albumentations import ColorJitter, Compose, HorizontalFlip, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Project dataset
from dataset import AgriDiseaseDataset


# =========================
# Utilities
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic flags (can slow a bit but stable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Set random seed = {seed}")


# =========================
# Transforms (stable)
# =========================
def get_fewshot_train_transform(image_size: int = 256) -> Compose:
    # Robust and version-safe: fixed Resize + mild jitter + flip + normalize
    return Compose(
        [
            Resize(image_size, image_size),
            HorizontalFlip(p=0.5),
            ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.6),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


def get_fewshot_val_transform(image_size: int = 256) -> Compose:
    return Compose(
        [
            Resize(image_size, image_size),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )


# =========================
# Collate (picklable)
# =========================
def fewshot_collate(batch):
    """
    Batch: list of (image_tensor, label_dict)
    Returns:
      images: [B, C, H, W]
      labels: {"label_61": LongTensor[B]}
    """
    images = torch.stack([x[0] for x in batch])
    labels = torch.tensor([int(x[1]["label_61"]) for x in batch], dtype=torch.long)
    return images, {"label_61": labels}


# =========================
# ArcFace Head
# =========================
class ArcFaceHead(nn.Module):
    def __init__(
        self, in_features: int, num_classes: int, scale: float = 30.0, margin: float = 0.30
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(
        self, features: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Normalize features and weights
        x = F.normalize(features, dim=1)
        w = F.normalize(self.weight, dim=1)

        logits = torch.matmul(x, w.t())  # [B, C]
        if labels is not None:
            # Apply angular margin to target logits only
            theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
            idx = torch.arange(logits.size(0), device=logits.device)
            target_theta = theta[idx, labels] + self.margin
            logits = logits.clone()
            logits[idx, labels] = torch.cos(target_theta).to(logits.dtype)
        return self.scale * logits


# =========================
# Prototype Loss
# =========================
class PrototypeLoss(nn.Module):
    def forward(
        self, features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor
    ) -> torch.Tensor:
        target_proto = prototypes[labels]  # [B, D]
        return ((features - target_proto) ** 2).mean()


@torch.no_grad()
def compute_prototypes(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> torch.Tensor:
    model.eval()
    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}
    for images, label_dict in dataloader:
        labels = label_dict["label_61"].to(device)
        images = images.to(device)
        feats = model.extract_features(images)
        for f, l in zip(feats, labels):
            li = int(l.item())
            if li not in sums:
                sums[li] = f.clone()
                counts[li] = 1
            else:
                sums[li] += f
                counts[li] += 1
    if not counts:
        raise RuntimeError("No samples found to compute prototypes.")
    num_classes = max(counts.keys()) + 1
    feat_dim = next(iter(sums.values())).size(0)
    prototypes = torch.zeros(num_classes, feat_dim, device=device)
    for k, v in sums.items():
        prototypes[k] = v / counts[k]
    return prototypes


# =========================
# Model Wrapper
# =========================
class FewShotArcFaceModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet50",
        num_classes: int = 61,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Backbone without classifier
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool=""
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            fm = self.backbone(dummy)  # [1, C, H, W]
            feat_dim = int(fm.shape[1])
        self.feature_dim = feat_dim
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.arcface_head = ArcFaceHead(in_features=self.feature_dim, num_classes=num_classes)

        # Start with fully frozen backbone; we'll unfreeze layer4 later (delayed)
        for p in self.backbone.parameters():
            p.requires_grad = False

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Backbone={backbone_name} feat_dim={self.feature_dim}")
        print(
            f"[Model] Total params: {total / 1e6:.2f}M | Trainable params: {trainable / 1e6:.2f}M (start fully frozen)"
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        fm = self.backbone(x)
        vec = self.global_pool(fm).flatten(1)
        vec = self.dropout(vec)
        return vec

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        feats = self.extract_features(x)
        logits = self.arcface_head(feats, labels)
        return logits, feats


# =========================
# Metrics
# =========================
def macro_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    _, preds = torch.max(logits, dim=1)
    f1_vals = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_vals.append(f1)
    return float(sum(f1_vals) / max(len(f1_vals), 1))


# =========================
# Mixup
# =========================
def apply_mixup(
    images: torch.Tensor, labels: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[index]
    return mixed, labels, labels[index], float(lam)


# =========================
# Train / Validate
# =========================
def train_one_epoch(
    model: FewShotArcFaceModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    epochs: int,
    ce_loss_fn,
    proto_loss_fn,
    prototypes: torch.Tensor,
    proto_weight: float,
    mixup_alpha: float,
    mixup_disable_epoch: int,
    amp: bool,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    use_amp = amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    use_mixup_now = mixup_alpha > 0 and epoch < mixup_disable_epoch

    for images, label_dict in loader:
        labels = label_dict["label_61"].to(device)
        images = images.to(device)

        images_aug, y_a, y_b, lam = apply_mixup(
            images, labels, mixup_alpha if use_mixup_now else 0.0
        )

        optimizer.zero_grad(set_to_none=True)
        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=use_amp) if use_amp else contextlib.nullcontext()
        )
        with autocast_ctx:
            logits_a, feats = model(images_aug, labels=y_a)  # margin applied to y_a
            ce_a = ce_loss_fn(logits_a, y_a)
            if use_mixup_now:
                logits_b, _ = model(images_aug, labels=y_b)
                ce_b = ce_loss_fn(logits_b, y_b)
                ce_loss = lam * ce_a + (1 - lam) * ce_b
            else:
                ce_loss = ce_a

            ploss = proto_loss_fn(feats, y_a, prototypes)
            loss = ce_loss + proto_weight * ploss

        if use_amp:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += float(loss.item()) * batch_size

        # Train accuracy: eval without margin bias
        with torch.no_grad():
            eval_logits, _ = model(images, labels=None)
            preds = torch.argmax(eval_logits, dim=1)
            total_correct += int(preds.eq(labels).sum().item())

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": 100.0 * total_correct / max(total_samples, 1),
    }


@torch.inference_mode()
def validate(
    model: FewShotArcFaceModel, loader: DataLoader, device: torch.device, num_classes: int
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    ce_fn = nn.CrossEntropyLoss()

    all_logits = []
    all_labels = []

    for images, label_dict in loader:
        labels = label_dict["label_61"].to(device)
        images = images.to(device)

        logits, _ = model(images, labels=None)  # no margin in eval
        loss = ce_fn(logits, labels)

        bs = images.size(0)
        total_samples += bs
        total_loss += float(loss.item()) * bs

        preds = torch.argmax(logits, dim=1)
        total_correct += int(preds.eq(labels).sum().item())

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)

    # Top-5
    _, pred_top5 = logits_cat.topk(5, dim=1)
    top5_correct = sum(1 for i in range(labels_cat.size(0)) if labels_cat[i] in pred_top5[i])

    m_f1 = macro_f1(logits_cat, labels_cat, num_classes=num_classes)

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": 100.0 * total_correct / max(total_samples, 1),
        "top5": 100.0 * top5_correct / max(total_samples, 1),
        "macro_f1": m_f1,
    }


# =========================
# Args
# =========================
def parse_args():
    p = argparse.ArgumentParser(description="Few-Shot Task2 Training (ArcFace + Prototype EMA)")
    # Data
    p.add_argument("--train-dir", type=str, default="data/cleaned/train", help="Train images root")
    p.add_argument("--val-dir", type=str, default="data/cleaned/val", help="Val images root")
    p.add_argument("--train-meta", type=str, required=True, help="Few-shot train metadata CSV")
    p.add_argument("--val-meta", type=str, required=True, help="Validation metadata CSV")
    # Model
    p.add_argument(
        "--backbone", type=str, default="resnet50", help="Backbone architecture from timm"
    )
    p.add_argument("--pretrained", action="store_true", default=True, help="Use pretrained weights")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout before head (Strategy B)")
    p.add_argument(
        "--freeze-stage12",
        action="store_true",
        help="(Kept for compatibility; training starts fully frozen)",
    )
    # Training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4, help="Base backbone LR")
    p.add_argument("--head-lr-scale", type=float, default=3.0, help="Head LR scale (Strategy A)")
    p.add_argument(
        "--proto-weight", type=float, default=0.4, help="Prototype loss weight (Strategy E)"
    )
    p.add_argument("--arcface-margin", type=float, default=0.30)
    p.add_argument("--arcface-scale", type=float, default=30.0)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--mixup-alpha", type=float, default=0.2, help="Mixup alpha (0 disables)")
    p.add_argument(
        "--mixup-disable-epoch", type=int, default=8, help="Epoch to disable mixup (Strategy C)"
    )
    p.add_argument("--no-mixup", action="store_true", help="Disable Mixup from start")
    p.add_argument(
        "--amp", action="store_true", default=True, help="Use mixed precision (CUDA only)"
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--label-smoothing", type=float, default=0.05, help="Label smoothing for CE (Strategy D)"
    )
    p.add_argument(
        "--proto-ema", type=float, default=0.7, help="EMA momentum for prototypes (0 disables EMA)"
    )
    p.add_argument(
        "--proto-refresh-interval",
        type=int,
        default=1,
        help="Epoch interval to recompute raw prototypes",
    )
    p.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=3,
        help="Epoch to unfreeze backbone layer4 (Strategy F)",
    )
    # IO
    p.add_argument("--save-dir", type=str, default="checkpoints/task2_fewshot")
    p.add_argument("--save-best-only", action="store_true", default=True)
    p.add_argument(
        "--early-stop-patience", type=int, default=10, help="Early stop patience on val acc"
    )
    # Optimizer
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    return p.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()
    set_seed(args.seed)

    # Select device: prefer CUDA, then MPS, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[Device] {device}")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    train_tf = get_fewshot_train_transform(args.image_size)
    val_tf = get_fewshot_val_transform(args.image_size)

    # Datasets
    train_dataset = AgriDiseaseDataset(
        data_dir=args.train_dir,
        metadata_path=args.train_meta,
        transform=train_tf,
        return_multitask=False,
    )
    val_dataset = AgriDiseaseDataset(
        data_dir=args.val_dir,
        metadata_path=args.val_meta,
        transform=val_tf,
        return_multitask=False,
    )
    print(f"[Data] Few-shot train samples: {len(train_dataset)}")
    print(f"[Data] Validation samples: {len(val_dataset)}")

    # DataLoaders (adaptive workers)
    adaptive_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=adaptive_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=fewshot_collate,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=adaptive_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=fewshot_collate,
    )

    # Model
    model = FewShotArcFaceModel(
        backbone_name=args.backbone,
        num_classes=61,
        pretrained=args.pretrained,
        dropout=args.dropout,
    ).to(device)
    # Set ArcFace hyper-params
    model.arcface_head.margin = args.arcface_margin
    model.arcface_head.scale = args.arcface_scale

    # Parameter groups with distinct weight decay
    head_params = [p for p in model.arcface_head.parameters() if p.requires_grad] + [
        p for p in model.dropout.parameters() if p.requires_grad
    ]
    backbone_params = list(model.backbone.parameters())

    param_groups = [
        {"params": head_params, "lr": args.lr * args.head_lr_scale, "weight_decay": 5e-4},
        {"params": backbone_params, "lr": args.lr, "weight_decay": 1e-4},
    ]

    if args.optimizer == "adamw":
        optimizer = AdamW(param_groups)
    else:
        optimizer = SGD(param_groups, momentum=0.9, nesterov=True)

    steps_per_epoch = max(len(train_loader), 1)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=[args.lr * args.head_lr_scale, args.lr],
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        div_factor=1.0,
        final_div_factor=20.0,
        anneal_strategy="cos",
    )

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    proto_loss_fn = PrototypeLoss()

    # Initial prototypes + EMA state
    raw_prototypes = compute_prototypes(model, train_loader, device)
    prototypes = raw_prototypes
    prev_prototypes = raw_prototypes.clone()
    print("[Proto] Initialized prototypes.")

    best_val_acc = 0.0
    best_macro_f1 = 0.0
    epochs_no_improve = 0

    # History + TensorBoard
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_top5": [],
        "val_macro_f1": [],
        "head_lr": [],
        "backbone_lr": [],
    }
    tb_writer = SummaryWriter(str(save_dir / "logs"))
    print(
        f"[Log] TensorBoard -> {save_dir / 'logs'} (run: tensorboard --logdir {save_dir / 'logs'})"
    )

    mixup_alpha = 0.0 if args.no_mixup else args.mixup_alpha

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Prototype refresh + EMA
        if epoch % args.proto_refresh_interval == 0:
            raw_prototypes = compute_prototypes(model, train_loader, device)
            if 0.0 <= args.proto_ema < 1.0:
                prototypes = (
                    args.proto_ema * prev_prototypes + (1 - args.proto_ema) * raw_prototypes
                )
            else:
                prototypes = raw_prototypes
            prev_prototypes = prototypes.clone()

        # Delayed selective unfreeze (only layer4)
        if args.unfreeze_epoch == epoch:
            print(f"[Unfreeze] Epoch {epoch}: enabling layer4 parameters for training.")
            for name, p in model.backbone.named_parameters():
                if "layer4" in name:
                    p.requires_grad = True

        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            epoch=epoch,
            epochs=args.epochs,
            ce_loss_fn=ce_loss_fn,
            proto_loss_fn=proto_loss_fn,
            prototypes=prototypes,
            proto_weight=args.proto_weight,
            mixup_alpha=mixup_alpha,
            mixup_disable_epoch=args.mixup_disable_epoch,
            amp=args.amp,
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            device=device,
            num_classes=61,
        )

        print(
            f"[Train] Loss={train_metrics['loss']:.4f} Acc={train_metrics['acc']:.2f}% | "
            f"[Val] Loss={val_metrics['loss']:.4f} Acc={val_metrics['acc']:.2f}% "
            f"Top5={val_metrics['top5']:.2f}% MacroF1={val_metrics['macro_f1']:.3f}"
        )

        # Update history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        history["val_top5"].append(val_metrics["top5"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["head_lr"].append(optimizer.param_groups[0]["lr"])
        history["backbone_lr"].append(optimizer.param_groups[1]["lr"])

        # TensorBoard logging
        tb_writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        tb_writer.add_scalar("train/acc", train_metrics["acc"], epoch)
        tb_writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        tb_writer.add_scalar("val/acc", val_metrics["acc"], epoch)
        tb_writer.add_scalar("val/top5", val_metrics["top5"], epoch)
        tb_writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
        tb_writer.add_scalar("lr/head", optimizer.param_groups[0]["lr"], epoch)
        tb_writer.add_scalar("lr/backbone", optimizer.param_groups[1]["lr"], epoch)

        # Save best and confusion matrix
        improved = False
        if val_metrics["acc"] > best_val_acc:
            improved = True
            best_val_acc = val_metrics["acc"]
            best_macro_f1 = val_metrics["macro_f1"]

            # Save confusion matrix and checkpoint
            with torch.no_grad():
                all_preds = []
                all_trues = []
                for images_cm, labels_cm_dict in val_loader:
                    labels_cm = labels_cm_dict["label_61"].to(device)
                    images_cm = images_cm.to(device)
                    logits_cm, _ = model(images_cm, labels=None)
                    preds_cm = torch.argmax(logits_cm, dim=1)
                    all_preds.append(preds_cm.cpu())
                    all_trues.append(labels_cm.cpu())
                preds_cat = torch.cat(all_preds)
                trues_cat = torch.cat(all_trues)

                num_classes_cm = 61
                cm = torch.zeros(num_classes_cm, num_classes_cm, dtype=torch.int32)
                for t, p in zip(trues_cat.tolist(), preds_cat.tolist()):
                    cm[t, p] += 1

                # Save confusion matrix CSV and PNG
                try:
                    import pandas as pd

                    cm_df = pd.DataFrame(cm.numpy())
                    cm_df.to_csv(save_dir / "confusion_matrix_best.csv", index=False)
                except Exception as e:
                    print(f"[Warn] Failed to save CM CSV: {e}")

                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                im = ax_cm.imshow(cm.numpy(), cmap="Blues", aspect="auto")
                ax_cm.set_title("Confusion Matrix (Best)")
                ax_cm.set_xlabel("Predicted")
                ax_cm.set_ylabel("True")
                plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
                plt.tight_layout()
                fig_cm.savefig(save_dir / "confusion_matrix_best.png", dpi=120)
                plt.close(fig_cm)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "best_macro_f1": best_macro_f1,
                    "prototypes": prototypes.detach().cpu(),
                },
                save_dir / "best.pth",
            )
            print(
                f"[Checkpoint] New best Acc={best_val_acc:.2f}% MacroF1={best_macro_f1:.3f} saved. CM written."
            )

        # Early stopping
        if not improved:
            epochs_no_improve += 1
        else:
            epochs_no_improve = 0

        if epochs_no_improve >= args.early_stop_patience:
            print(f"[EarlyStop] No improvement for {epochs_no_improve} epochs. Stopping early.")
            break

        # Disable Mixup when scheduled
        if epoch == args.mixup_disable_epoch and mixup_alpha > 0:
            mixup_alpha = 0.0
            print(f"[Regularization] Mixup disabled at epoch {epoch}")

    # Plot curves and history
    if history["epoch"]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Few-Shot Training Progress", fontsize=16, fontweight="bold")

        axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", color="tab:blue")
        axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", color="tab:red")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc", color="tab:blue")
        axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc", color="tab:red")
        axes[1].plot(
            history["epoch"], history["val_macro_f1"], label="Val MacroF1", color="tab:green"
        )
        axes[1].set_title("Accuracy / MacroF1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Percent / Score")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        axes[2].plot(history["epoch"], history["head_lr"], label="Head LR", color="tab:purple")
        axes[2].plot(
            history["epoch"], history["backbone_lr"], label="Backbone LR", color="tab:orange"
        )
        axes[2].set_title("Learning Rates")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("LR")
        axes[2].set_yscale("log")
        axes[2].grid(alpha=0.3)
        axes[2].legend()

        plot_path = save_dir / "training_curves.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[Plot] Saved training curves -> {plot_path}")

        # Save CSV history
        try:
            import pandas as pd

            hist_df = pd.DataFrame(history)
            hist_df.to_csv(save_dir / "training_history.csv", index=False)
            print(f"[History] Saved training history CSV -> {save_dir / 'training_history.csv'}")
        except Exception as e:
            print(f"[Warn] Failed to save training history CSV: {e}")

    tb_writer.close()

    print("\n" + "=" * 60)
    print(
        f"Training complete. Best Val Acc: {best_val_acc:.2f}% | Best MacroF1: {best_macro_f1:.3f}"
    )
    print(f"Checkpoint directory: {save_dir}")
    print("=" * 60)
    print("Next steps:")
    print("  - Evaluate best checkpoint and inspect confusion matrix")
    print("  - Try convnext_tiny / efficientnetv2_s with same regime")
    print("  - Tune proto-ema or mixup schedule if curves suggest over/under-regularization")


if __name__ == "__main__":
    main()
