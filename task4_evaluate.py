#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4: Post-Training Evaluation and Diagnostic Report Generation
==================================================================

Use this script after training to:
1. Load trained multitask model
2. Generate comprehensive diagnostic reports
3. Create Grad-CAM visualizations
4. Evaluate severity metrics with confusion matrix
5. Export per-sample predictions

Usage:
    python task4_evaluate.py \
        --checkpoint checkpoints/task4_multitask/multitask/best.pth \
        --val-meta data/cleaned/metadata/val_metadata.csv \
        --val-dir data/cleaned/val \
        --out-dir checkpoints/task4_multitask/evaluation \
        --batch-size 64 \
        --cam-samples 50 \
        --report-samples 500
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Project modules
from models import MultiTaskModel
from task4train import (
    MultiTaskSeverity3Dataset,
    SeverityCAMWrapper,
    report_collate,
)

# Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except Exception:
    GradCAM = None
    show_cam_on_image = None


# ============================================================
# Evaluation Functions
# ============================================================


@torch.no_grad()
def evaluate_all_tasks(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict:
    """Evaluate all tasks and return comprehensive metrics."""
    model.eval()

    all_preds = {
        "label_61": [],
        "crop": [],
        "disease": [],
        "severity": [],
    }
    all_targets = {
        "label_61": [],
        "crop": [],
        "disease": [],
        "severity": [],
    }

    for batch in tqdm(loader, desc="Evaluating"):
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device)
        outputs = model(images)

        for task in ["label_61", "crop", "disease", "severity"]:
            preds = outputs[task].argmax(dim=1).cpu().tolist()
            targets = labels[task].cpu().tolist()
            all_preds[task].extend(preds)
            all_targets[task].extend(targets)

    # Compute metrics for each task
    results = {}
    for task in ["label_61", "crop", "disease", "severity"]:
        acc = (np.array(all_preds[task]) == np.array(all_targets[task])).mean()
        macro_f1 = f1_score(all_targets[task], all_preds[task], average="macro")

        # Determine number of classes
        n_classes = max(max(all_targets[task]), max(all_preds[task])) + 1
        cm = confusion_matrix(
            all_targets[task],
            all_preds[task],
            labels=list(range(n_classes)),
        ).tolist()

        results[task] = {
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "confusion_matrix": cm,
        }

    return results


@torch.no_grad()
def generate_detailed_report(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_csv: Path,
    cam_dir: Optional[Path] = None,
    cam_samples: int = 0,
    max_samples: Optional[int] = None,
):
    """Generate detailed diagnostic report with optional CAM visualizations."""
    model.eval()
    rows = []

    # Setup Grad-CAM
    cam = None
    if (
        cam_dir is not None
        and cam_samples > 0
        and GradCAM is not None
        and hasattr(model, "get_last_conv_layer")
    ):
        cam_dir.mkdir(parents=True, exist_ok=True)
        try:
            cam_model = SeverityCAMWrapper(model)
            target_layer = cam_model.get_last_conv_layer()
            cam = GradCAM(model=cam_model, target_layers=[target_layer])
            print(f"✓ Grad-CAM initialized for severity visualization")
        except Exception as e:
            print(f"⚠️  Grad-CAM initialization failed: {e}")
            cam = None

    saved_cam = 0

    for images, labels, metas in tqdm(loader, desc="Generating report"):
        images = images.to(device)
        outputs = model(images)

        for i in range(images.size(0)):
            # Get predictions and confidences
            l61_logits = outputs["label_61"][i]
            crop_logits = outputs["crop"][i]
            dis_logits = outputs["disease"][i]
            sev_logits = outputs["severity"][i]

            l61_probs = F.softmax(l61_logits, dim=0)
            crop_probs = F.softmax(crop_logits, dim=0)
            dis_probs = F.softmax(dis_logits, dim=0)
            sev_probs = F.softmax(sev_logits, dim=0)

            l61_conf, l61_pred = l61_probs.max(0)
            crop_conf, crop_pred = crop_probs.max(0)
            dis_conf, dis_pred = dis_probs.max(0)
            sev_conf, sev_pred = sev_probs.max(0)

            # Top-3 for label_61
            top3_confs, top3_preds = torch.topk(l61_probs, k=min(3, len(l61_probs)))
            top3_list = [
                [int(top3_preds[j].item()), float(top3_confs[j].item())]
                for j in range(len(top3_preds))
            ]

            # Ground truth
            true_l61 = int(labels["label_61"][i].item())
            true_crop = int(labels["crop"][i].item())
            true_dis = int(labels["disease"][i].item())
            true_sev = int(labels["severity"][i].item())

            row = {
                "image_name": metas[i]["image_name"],
                # Label 61
                "pred_label_61": int(l61_pred.item()),
                "conf_label_61": float(l61_conf.item()),
                "true_label_61": true_l61,
                "correct_61": int(int(l61_pred.item()) == true_l61),
                "top3_label_61": json.dumps(top3_list),
                # Crop
                "pred_crop": int(crop_pred.item()),
                "conf_crop": float(crop_conf.item()),
                "true_crop": true_crop,
                "correct_crop": int(int(crop_pred.item()) == true_crop),
                # Disease
                "pred_disease": int(dis_pred.item()),
                "conf_disease": float(dis_conf.item()),
                "true_disease": true_dis,
                "correct_disease": int(int(dis_pred.item()) == true_dis),
                # Severity (3-class)
                "pred_severity3": int(sev_pred.item()),
                "conf_severity3": float(sev_conf.item()),
                "true_severity3": true_sev,
                "correct_severity3": int(int(sev_pred.item()) == true_sev),
            }
            rows.append(row)

            # Grad-CAM visualization
            if cam is not None and saved_cam < cam_samples:
                try:
                    img_tensor = images[i].detach().cpu()
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_np = (img_tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

                    with torch.enable_grad():
                        grayscale_cam = cam(input_tensor=images[i : i + 1])[0]

                    if callable(show_cam_on_image):
                        overlay = show_cam_on_image(
                            img_np, grayscale_cam, use_rgb=True, image_weight=0.55
                        )
                        vis = (overlay[:, :, ::-1] * 255).astype(np.uint8)
                    else:
                        vis = (img_np[:, :, ::-1] * 255).astype(np.uint8)

                    correct_str = "correct" if row["correct_severity3"] else "wrong"
                    vis_name = (
                        f"{metas[i]['image_name']}_"
                        f"pred={int(sev_pred.item())}_"
                        f"true={true_sev}_{correct_str}_"
                        f"conf={float(sev_conf.item()):.3f}.jpg"
                    )
                    cv2.imwrite(str(cam_dir / vis_name), vis)
                    saved_cam += 1
                except Exception as e:
                    print(f"⚠️  CAM failed for {metas[i]['image_name']}: {e}")

            if max_samples is not None and len(rows) >= max_samples:
                break

        if max_samples is not None and len(rows) >= max_samples:
            break

    # Save CSV
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        if rows:
            fieldnames = list(rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"✅ Diagnostic report saved: {out_csv} ({len(rows)} samples)")
    if saved_cam > 0:
        print(f"✅ Grad-CAM visualizations saved: {cam_dir} ({saved_cam} images)")


def plot_confusion_matrix(cm, labels, title, save_path):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )

    # Rotate tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Confusion matrix saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Task 4: Post-Training Evaluation")

    # Model & Data
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth)",
    )
    parser.add_argument("--val-meta", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)

    # Model architecture (must match training)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)

    # Output
    parser.add_argument("--out-dir", type=str, default="checkpoints/task4_multitask/evaluation")
    parser.add_argument(
        "--report-samples",
        type=int,
        default=None,
        help="Max samples in diagnostic report (None = all)",
    )
    parser.add_argument(
        "--cam-samples", type=int, default=50, help="Number of CAM images to generate"
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    model = MultiTaskModel(
        backbone=args.backbone,
        pretrained=False,  # We're loading trained weights
        dropout=args.dropout,
        num_classes_61=61,
        num_crops=10,
        num_diseases=28,
        num_severity=3,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")

    # Load validation data
    print(f"\nLoading validation data...")
    val_ds = MultiTaskSeverity3Dataset(
        data_dir=args.val_dir,
        metadata_csv=args.val_meta,
        augment=False,
        image_size=args.image_size,
    )
    print(f"Validation set: {len(val_ds)} samples")
    print(f"Severity distribution: {val_ds.class_counts}")

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(8, os.cpu_count() or 4),
        pin_memory=(device.type == "cuda"),
        collate_fn=report_collate,
    )

    # ========================================
    # 1. Evaluate all tasks
    # ========================================
    print("\n" + "=" * 60)
    print("Evaluating all tasks...")
    print("=" * 60)

    all_metrics = evaluate_all_tasks(model, val_loader, device)

    # Save metrics
    metrics_path = out_dir / "all_task_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✅ Metrics saved: {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    for task, metrics in all_metrics.items():
        print(f"\n{task.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Macro F1:  {metrics['macro_f1']:.4f}")

    # ========================================
    # 2. Plot confusion matrices
    # ========================================
    print("\n" + "=" * 60)
    print("Generating confusion matrices...")
    print("=" * 60)

    severity_cm = np.array(all_metrics["severity"]["confusion_matrix"])
    plot_confusion_matrix(
        severity_cm,
        labels=["Healthy", "General", "Serious"],
        title="Severity Classification Confusion Matrix (3-class)",
        save_path=out_dir / "confusion_matrix_severity.png",
    )

    # ========================================
    # 3. Generate diagnostic report
    # ========================================
    print("\n" + "=" * 60)
    print("Generating diagnostic report...")
    print("=" * 60)

    generate_detailed_report(
        model=model,
        loader=val_loader,
        device=device,
        out_csv=out_dir / "diagnostic_report.csv",
        cam_dir=(out_dir / "grad_cam") if args.cam_samples > 0 else None,
        cam_samples=args.cam_samples,
        max_samples=args.report_samples,
    )

    # ========================================
    # 4. Analyze errors
    # ========================================
    print("\n" + "=" * 60)
    print("Analyzing prediction errors...")
    print("=" * 60)

    import pandas as pd

    df = pd.read_csv(out_dir / "diagnostic_report.csv")

    # Overall accuracy per task
    for task in ["61", "crop", "disease", "severity3"]:
        correct_col = f"correct_{task}"
        if correct_col in df.columns:
            acc = df[correct_col].mean()
            print(f"  {task:12s}: {acc:.4f} ({int(df[correct_col].sum())}/{len(df)})")

    # Severity error breakdown
    if "correct_severity3" in df.columns:
        errors = df[df["correct_severity3"] == 0]
        print(f"\nSeverity errors: {len(errors)}/{len(df)}")
        if len(errors) > 0:
            print("  Error distribution by true class:")
            for true_class in sorted(df["true_severity3"].unique()):
                class_errors = errors[errors["true_severity3"] == true_class]
                class_total = len(df[df["true_severity3"] == true_class])
                if class_total > 0:
                    print(
                        f"    Class {true_class}: {len(class_errors)}/{class_total} "
                        f"({len(class_errors) / class_total:.2%})"
                    )

    print("\n" + "=" * 60)
    print("✅ Evaluation complete!")
    print("=" * 60)
    print(f"\nResults saved to: {out_dir}")
    print(f"  - Metrics:           {metrics_path.name}")
    print(f"  - Confusion matrix:  confusion_matrix_severity.png")
    print(f"  - Diagnostic report: diagnostic_report.csv")
    if args.cam_samples > 0:
        print(f"  - CAM images:        grad_cam/")


if __name__ == "__main__":
    main()
