"""
Agricultural Disease Recognition - Main Training Script
========================================================

"Talk is cheap. Show me the code." - Linus Torvalds

This is the main training script for Task 1: 61-class disease classification.
Keep it simple. Make it work. Then optimize.

Usage:
    python train.py --config config_task1.yaml
    python train.py --backbone resnet50 --epochs 50 --batch-size 64
"""

import argparse
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Import our modules
from dataset import AgriDiseaseDataset, collate_fn, get_train_transform, get_val_transform
from losses import FocalLoss, create_loss_function
from models import create_model
from trainer import Trainer


# Cosine / Center loss moved to module scope (clean activation flags)
class CosineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int, scale: float = 30.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        f = torch.nn.functional.normalize(features, dim=1)
        w = torch.nn.functional.normalize(self.weight, dim=1)
        return self.scale * torch.matmul(f, w.t())


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feat_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.centers)

    def forward(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        centers_batch = self.centers[targets]
        return ((features - centers_batch) ** 2).sum(dim=1).mean()


def build_weighted_sampler(metadata_df, label_col: str = "label_61", power: float = 0.5):
    """
    Build a WeightedRandomSampler with inverse freq^power weights.
    power=0.5 => inverse sqrt frequency (stabilizes extreme long-tail).
    """
    counts = Counter(metadata_df[label_col].tolist())
    weights = [1.0 / (counts[label] ** power) for label in metadata_df[label_col]]

    return torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )


class ModelEMA:
    """
    Exponential Moving Average of model parameters.
    Keeps a shadow copy updated each step for more stable validation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(d).add_(param.data, alpha=(1.0 - d))

    def apply_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    cutmix_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Decide between Mixup / CutMix / None each batch.
    Returns (images, targets_a, targets_b, lambda)
    """
    lam = 1.0
    use_mixup = mixup_alpha > 0 and np.random.rand() < mixup_prob
    use_cutmix = (not use_mixup) and cutmix_alpha > 0 and np.random.rand() < cutmix_prob
    batch_size = images.size(0)
    device = images.device
    index = torch.randperm(batch_size, device=device)

    if use_mixup:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed = lam * images + (1 - lam) * images[index]
        return mixed, targets, targets[index], lam
    elif use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        # Generate random bounding box
        h, w = images.size(2), images.size(3)
        cut_rat = math.sqrt(1.0 - lam)
        cut_h = int(h * cut_rat)
        cut_w = int(w * cut_rat)
        cy = np.random.randint(0, h)
        cx = np.random.randint(0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (h * w))
        return images, targets, targets[index], lam
    else:
        return images, targets, targets, lam


def custom_train_loop(
    model: nn.Module,
    optimizer,
    scheduler,
    criterion,
    device,
    train_loader,
    val_loader,
    epochs: int,
    use_amp: bool,
    multi_task: bool,
    mixup_alpha: float,
    cutmix_alpha: float,
    mixup_prob: float,
    cutmix_prob: float,
    save_dir: str,
    ema: Optional[ModelEMA] = None,
    # Phase scheduling params
    stage2_epoch: int = 10,
    stage3_epoch: int = 20,
    smoothing_decay_epoch1: int = 12,
    smoothing_decay_epoch2: int = 18,
    disable_sampler_epoch: int = 15,
    lr_restart_epoch: int = 25,
    tail_focal_epoch: int = 18,
    progressive_resize_epoch: int = 30,
    progressive_image_size: int = 256,
    final_clean_epoch: int = 40,
    # Dynamic regularization decay params
    mixup_decay_epoch1: int = 10,
    mixup_decay_epoch2: int = 12,
    mixup_disable_epoch: int = 15,
    cutmix_disable_epoch: int = 10,
    # Margin sharpening phase
    use_cosine_classifier: bool = False,
    use_center_loss: bool = False,
    center_loss_weight: float = 0.01,
    center_update_epoch: int = 35,
    model_type: str = "baseline",
):
    """
    Minimal custom training loop supporting sampler, Mixup/CutMix, EMA.
    Prints epoch metrics similar to Trainer.
    Now includes real-time monitoring with TensorBoard, CSV history, and PNG plots.
    """
    # Type assertions
    assert isinstance(model, nn.Module), "model must be nn.Module"
    assert scheduler is None or hasattr(scheduler, "step"), (
        "scheduler must implement step() or be None"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    # Initialize monitoring components
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # TensorBoard writer
    tb_writer = SummaryWriter(str(save_path / "logs"))
    print(
        f"[TensorBoard] Logging to {save_path / 'logs'} (run: tensorboard --logdir {save_path / 'logs'})"
    )

    # History tracking
    history = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "tail_acc": [],
        "macro_f1": [],
        "lr": [],
    }

    # Precompute tail class set (bottom 25% frequency) for focal application
    label_col_values = train_loader.dataset.metadata["label_61"].tolist()
    freq_counter = Counter(label_col_values)
    sorted_ids = sorted(freq_counter.keys(), key=lambda k: freq_counter[k])
    tail_cut = max(1, int(len(sorted_ids) * 0.25))
    tail_set = set(sorted_ids[:tail_cut])

    # Optional focal loss object (will be triggered later)
    focal_loss_obj = FocalLoss(alpha=None, gamma=1.5) if tail_focal_epoch < epochs else None

    def adaptive_loss(outputs_logits: torch.Tensor, targets_tensor: torch.Tensor, epoch_idx: int):
        # Epoch-based switching: focal only for tail samples after tail_focal_epoch
        if focal_loss_obj is not None and epoch_idx >= tail_focal_epoch:
            tail_mask = torch.tensor(
                [t.item() in tail_set for t in targets_tensor], device=targets_tensor.device
            )
            if tail_mask.any():
                tail_indices = tail_mask.nonzero(as_tuple=True)[0]
                non_tail_indices = (~tail_mask).nonzero(as_tuple=True)[0]
                tail_loss = focal_loss_obj(
                    outputs_logits[tail_indices], targets_tensor[tail_indices]
                )
                if non_tail_indices.numel() > 0:
                    ce_loss = criterion(
                        outputs_logits[non_tail_indices], targets_tensor[non_tail_indices]
                    )
                    return tail_loss + ce_loss
                return tail_loss
        return criterion(outputs_logits, targets_tensor)

    def run_val(current_epoch: int):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        per_class_total = {}
        per_class_correct = {}
        with torch.inference_mode():
            predicted_total = {}
            for images, labels in val_loader:
                images = images.to(device)
                if device.type == "cuda":
                    images = images.to(memory_format=torch.channels_last)
                targets = labels["label_61"].to(device)
                outputs_raw = model(images)
                outputs = outputs_raw["label_61"] if isinstance(outputs_raw, dict) else outputs_raw
                loss = criterion(outputs, targets)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                correct += preds.eq(targets).sum().item()
                total += images.size(0)
                # Per-class stats
                for t, p in zip(targets.tolist(), preds.tolist()):
                    per_class_total[t] = per_class_total.get(t, 0) + 1
                    predicted_total[p] = predicted_total.get(p, 0) + 1
                    if p == t:
                        per_class_correct[t] = per_class_correct.get(t, 0) + 1
        acc = 100.0 * correct / total
        # Tail mean acc
        tail_acc_vals = []
        for cid in tail_set:
            if per_class_total.get(cid, 0) > 0:
                tail_acc_vals.append(per_class_correct.get(cid, 0) / per_class_total[cid])
        tail_mean_acc = (sum(tail_acc_vals) / len(tail_acc_vals) * 100.0) if tail_acc_vals else 0.0
        # Macro-F1 (one-vs-rest approximation from counts)
        f1_vals = []
        for cid in per_class_total.keys():
            tp = per_class_correct.get(cid, 0)
            support = per_class_total.get(cid, 0)
            pred_cnt = predicted_total.get(cid, 0)
            fp = pred_cnt - tp
            fn = support - tp
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            )
            f1_vals.append(f1)
        macro_f1 = sum(f1_vals) / len(f1_vals) if f1_vals else 0.0
        print(
            f"  [Val] Epoch {current_epoch} | Loss {total_loss / total:.4f} | Acc {acc:.2f}% | TailAcc {tail_mean_acc:.2f}% | MacroF1 {macro_f1:.3f}"
        )
        return acc, total_loss / total, tail_mean_acc, macro_f1

    best_acc = 0.0
    for epoch in range(epochs):
        # Phase transitions
        if epoch == stage2_epoch:
            # Slight LR bump for partial fine-tune
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * 1.2
            print(f"[Phase] Stage2 start @ epoch {epoch} | LR bump applied")
        if epoch == stage3_epoch:
            for p in model.parameters():
                p.requires_grad = True
            print(f"[Phase] Stage3 start @ epoch {epoch} | Backbone unfrozen")
        if epoch == smoothing_decay_epoch1 and hasattr(criterion, "label_smoothing"):
            criterion.label_smoothing = 0.02
            print(f"[Phase] Smoothing -> 0.02 @ epoch {epoch}")
        if epoch == smoothing_decay_epoch2 and hasattr(criterion, "label_smoothing"):
            criterion.label_smoothing = 0.0
            print(f"[Phase] Smoothing -> 0.00 @ epoch {epoch}")
        if epoch == lr_restart_epoch:
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * 0.4
            print(f"[Phase] LR restart @ epoch {epoch}")
        if epoch == progressive_resize_epoch:
            from dataset import get_train_transform

            new_transform = get_train_transform(progressive_image_size)
            train_loader.dataset.transform = new_transform
            print(f"[Phase] Progressive resize -> {progressive_image_size} @ epoch {epoch}")
        if epoch == final_clean_epoch:
            from dataset import get_light_train_transform

            train_loader.dataset.transform = get_light_train_transform(progressive_image_size)
            print(f"[Phase] Final clean augmentation @ epoch {epoch}")
        if (
            epoch == disable_sampler_epoch
            and hasattr(train_loader, "sampler")
            and isinstance(train_loader.sampler, torch.utils.data.WeightedRandomSampler)
        ):
            print(f"[Phase] Disabling sampler @ epoch {epoch} (switch to natural distribution)")

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            # Dynamic Mixup/CutMix decay schedule (epochs passed as function params)
            if epoch == mixup_decay_epoch1 and mixup_alpha > 0:
                mixup_alpha = max(mixup_alpha * 0.5, 0.2)
            if epoch == mixup_decay_epoch2 and mixup_alpha > 0:
                mixup_alpha = max(mixup_alpha * 0.5, 0.1)
            if epoch == mixup_disable_epoch and mixup_alpha > 0:
                mixup_alpha = 0.0
            if epoch == cutmix_disable_epoch and cutmix_alpha > 0:
                cutmix_alpha = 0.0
            images = images.to(device)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            targets = labels["label_61"].to(device)
            images_aug, ta, tb, lam = apply_mixup_cutmix(
                images, targets, mixup_alpha, cutmix_alpha, mixup_prob, cutmix_prob
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                outputs_all = model(images_aug)
                outputs = outputs_all["label_61"] if isinstance(outputs_all, dict) else outputs_all
                primary_loss = lam * adaptive_loss(outputs, ta, epoch) + (1 - lam) * adaptive_loss(
                    outputs, tb, epoch
                )
                loss = primary_loss
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            if ema:
                ema.update(model)
            batch_size = images.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            preds = outputs.argmax(1)
            correct += preds.eq(targets).sum().item()
        if scheduler:
            scheduler.step()
        train_acc = 100.0 * correct / total
        train_loss = total_loss / total
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | LR {current_lr:.6f}"
        )
        # Validation (EMA shadow if enabled)
        if ema:
            saved = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            ema.apply_to(model)
            val_acc, val_loss, tail_mean_acc, macro_f1 = run_val(epoch)
            for n, p in model.named_parameters():
                if p.requires_grad and n in saved:
                    p.data.copy_(saved[n])
        else:
            val_acc, val_loss, tail_mean_acc, macro_f1 = run_val(epoch)

        # Update history
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["tail_acc"].append(tail_mean_acc)
        history["macro_f1"].append(macro_f1)
        history["lr"].append(current_lr)

        # TensorBoard logging
        tb_writer.add_scalar("train/loss", train_loss, epoch)
        tb_writer.add_scalar("train/acc", train_acc, epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/acc", val_acc, epoch)
        tb_writer.add_scalar("val/tail_acc", tail_mean_acc, epoch)
        tb_writer.add_scalar("val/macro_f1", macro_f1, epoch)
        tb_writer.add_scalar("lr/learning_rate", current_lr, epoch)

        if use_cosine_classifier and epoch == center_update_epoch and model_type == "baseline":
            if not hasattr(model, "cosine_head"):
                with torch.no_grad():
                    dummy = torch.zeros(
                        1, 3, progressive_image_size, progressive_image_size, device=device
                    )
                    feat_dim = model.get_features(dummy).shape[1]
                model.cosine_head = CosineClassifier(feat_dim, 61).to(device)
                # Flag removed to avoid assigning non-Tensor/Module attribute; use presence of cosine_head instead
                print(f"[Phase] Cosine classifier activated @ epoch {epoch} (feat_dim={feat_dim})")
        if use_center_loss and epoch == center_update_epoch and model_type == "baseline":
            if not hasattr(model, "center_loss_mod"):
                with torch.no_grad():
                    dummy = torch.zeros(
                        1, 3, progressive_image_size, progressive_image_size, device=device
                    )
                    feat_dim = model.get_features(dummy).shape[1]
                model.center_loss_mod = CenterLoss(61, feat_dim).to(device)
                # Flag removed; presence of center_loss_mod implies activation
                print(f"[Phase] CenterLoss activated @ epoch {epoch} (weight={center_loss_weight})")
        if val_acc > best_acc:
            best_acc = val_acc

            # Generate confusion matrix for best model
            model.eval()
            all_preds = []
            all_trues = []
            with torch.no_grad():
                for images_cm, labels_cm in val_loader:
                    images_cm = images_cm.to(device)
                    if device.type == "cuda":
                        images_cm = images_cm.to(memory_format=torch.channels_last)
                    targets_cm = labels_cm["label_61"].to(device)
                    outputs_cm = model(images_cm)
                    logits_cm = (
                        outputs_cm["label_61"] if isinstance(outputs_cm, dict) else outputs_cm
                    )
                    preds_cm = torch.argmax(logits_cm, dim=1)
                    all_preds.append(preds_cm.cpu())
                    all_trues.append(targets_cm.cpu())

                preds_cat = torch.cat(all_preds)
                trues_cat = torch.cat(all_trues)

                # Build confusion matrix
                num_classes_cm = 61
                cm = torch.zeros(num_classes_cm, num_classes_cm, dtype=torch.int32)
                for t, p in zip(trues_cat.tolist(), preds_cat.tolist()):
                    cm[t, p] += 1

                # Save confusion matrix CSV
                try:
                    import pandas as pd

                    cm_df = pd.DataFrame(cm.numpy())
                    cm_df.to_csv(save_path / "confusion_matrix_best.csv", index=False)
                    print(f"  [CM] Saved confusion matrix CSV")
                except Exception as e:
                    print(f"  [Warn] Failed to save CM CSV: {e}")

                # Save confusion matrix PNG
                fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
                im = ax_cm.imshow(cm.numpy(), cmap="Blues", aspect="auto")
                ax_cm.set_title("Confusion Matrix (Best Model)", fontsize=14, fontweight="bold")
                ax_cm.set_xlabel("Predicted Label")
                ax_cm.set_ylabel("True Label")
                plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
                plt.tight_layout()
                fig_cm.savefig(save_path / "confusion_matrix_best.png", dpi=120)
                plt.close(fig_cm)
                print(f"  [CM] Saved confusion matrix PNG")

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_acc,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "tail_mean_acc": tail_mean_acc,
                    "macro_f1": macro_f1,
                },
                Path(save_dir) / "best_custom.pth",
            )
            print(
                f"  ✅ New best (custom loop) val acc: {best_acc:.2f}% | TailAcc {tail_mean_acc:.2f}% | MacroF1 {macro_f1:.3f}"
            )

    # Close TensorBoard writer
    tb_writer.close()

    # Plot training curves
    if history["epoch"]:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Task 1 Training Progress", fontsize=16, fontweight="bold")

        # Loss curves
        axes[0, 0].plot(
            history["epoch"],
            history["train_loss"],
            label="Train Loss",
            color="tab:blue",
            linewidth=2,
        )
        axes[0, 0].plot(
            history["epoch"], history["val_loss"], label="Val Loss", color="tab:red", linewidth=2
        )
        axes[0, 0].set_title("Loss Curves", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].grid(alpha=0.3)
        axes[0, 0].legend()

        # Accuracy curves
        axes[0, 1].plot(
            history["epoch"], history["train_acc"], label="Train Acc", color="tab:blue", linewidth=2
        )
        axes[0, 1].plot(
            history["epoch"], history["val_acc"], label="Val Acc", color="tab:red", linewidth=2
        )
        axes[0, 1].plot(
            history["epoch"], history["tail_acc"], label="Tail Acc", color="tab:orange", linewidth=2
        )
        axes[0, 1].set_title("Accuracy Curves", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].grid(alpha=0.3)
        axes[0, 1].legend()

        # Macro F1 curve
        axes[1, 0].plot(
            history["epoch"], history["macro_f1"], label="Macro F1", color="tab:green", linewidth=2
        )
        axes[1, 0].set_title("Macro F1 Score", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Macro F1")
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()

        # Learning rate curve
        axes[1, 1].plot(
            history["epoch"], history["lr"], label="Learning Rate", color="tab:purple", linewidth=2
        )
        axes[1, 1].set_title("Learning Rate Schedule", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(alpha=0.3)
        axes[1, 1].legend()

        plot_path = save_path / "training_curves.png"
        fig.tight_layout()
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"\n[Plot] Saved training curves -> {plot_path}")

        # Save CSV history
        try:
            import pandas as pd

            hist_df = pd.DataFrame(history)
            hist_df.to_csv(save_path / "training_history.csv", index=False)
            print(f"[History] Saved training history CSV -> {save_path / 'training_history.csv'}")
        except Exception as e:
            print(f"[Warn] Failed to save training history CSV: {e}")

    print(f"\nCustom training finished. Best Val Acc: {best_acc:.2f}%")
    print(f"Checkpoints and logs saved to: {save_path}")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.

    Good taste: deterministic behavior for debugging.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train agricultural disease recognition model")

    # Data
    parser.add_argument(
        "--train-dir",
        type=str,
        default="data/cleaned/train",
        help="Training data directory",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="data/cleaned/val",
        help="Validation data directory",
    )
    parser.add_argument(
        "--train-meta",
        type=str,
        default="data/cleaned/metadata/train_metadata.csv",
        help="Training metadata CSV",
    )
    parser.add_argument(
        "--val-meta",
        type=str,
        default="data/cleaned/metadata/val_metadata.csv",
        help="Validation metadata CSV",
    )
    parser.add_argument(
        "--class-weights",
        type=str,
        default="data/cleaned/metadata/class_weights.csv",
        help="Class weights CSV",
    )

    # Model
    parser.add_argument(
        "--model-type",
        type=str,
        default="baseline",
        choices=["baseline", "multitask", "fewshot"],
        help="Model type ('fewshot' freezes backbone and trains lightweight head)",
    )
    parser.add_argument(
        "--progressive-resize",
        action="store_true",
        help="Enable progressive image size increase during training (improves later fine-grained accuracy)",
    )
    parser.add_argument(
        "--progressive-sizes",
        type=str,
        default="224,256",
        help="Comma-separated target sizes to switch to (start uses initial --image-size)",
    )
    parser.add_argument(
        "--resize-epochs",
        type=str,
        default="10",
        help="Comma-separated epochs at which to switch to next size (len must match progressive-sizes minus 1)",
    )
    parser.add_argument(
        "--sampler-switch-epoch",
        type=int,
        default=15,
        help="Epoch at which to disable balanced sampler and revert to standard shuffle",
    )
    parser.add_argument(
        "--focal-start-epoch",
        type=int,
        default=20,
        help="Epoch to switch from CE to Focal loss for hard classes",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=1.5,
        help="Gamma value for focal loss after switch",
    )
    parser.add_argument(
        "--smooth-reduce-epoch",
        type=int,
        default=12,
        help="Epoch to reduce label smoothing (e.g. 0.05 -> 0.02) for sharper decision boundaries",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        help="Backbone architecture (from timm)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate",
    )

    # Training
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
        help="Label smoothing factor (reduced for long-tail stability)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Input image size",
    )

    # Loss
    parser.add_argument(
        "--loss-type",
        type=str,
        default="weighted_ce",
        choices=["weighted_ce", "multitask", "focal"],
        help="Loss function type",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        default=True,
        help="Use class weights for imbalanced data",
    )

    # Optimization
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adam", "adamw", "sgd"],
        help="Optimizer",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "none"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )
    # Removed --compile option (torch.compile disabled to simplify training and avoid type issues)

    # Misc
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="checkpoints/task1_baseline",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Log every N batches",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )

    # Advanced sampling & augmentation & EMA arguments
    parser.add_argument(
        "--balance-sampler",
        action="store_true",
        help="Use inverse sqrt frequency WeightedRandomSampler for long-tail balancing",
    )
    parser.add_argument(
        "--mixup-alpha",
        type=float,
        default=0.4,
        help="Mixup alpha (0 to disable Mixup)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=0.7,
        help="Probability to apply Mixup each batch",
    )
    parser.add_argument(
        "--cutmix-alpha",
        type=float,
        default=0.6,
        help="CutMix alpha (0 to disable CutMix)",
    )
    parser.add_argument(
        "--cutmix-prob",
        type=float,
        default=0.5,
        help="Probability to apply CutMix if Mixup not selected",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Enable EMA of model weights for more stable evaluation",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.999,
        help="EMA decay (0.999 typical, larger = slower updates)",
    )

    # Few-shot specific arguments
    parser.add_argument(
        "--fewshot-freeze-backbone",
        action="store_true",
        default=True,
        help="Freeze backbone parameters for few-shot mode",
    )
    parser.add_argument(
        "--fewshot-hidden",
        type=int,
        default=512,
        help="Hidden dim of intermediate FC layer in few-shot classifier",
    )
    parser.add_argument(
        "--fewshot-dropout",
        type=float,
        default=0.5,
        help="Dropout for few-shot classifier",
    )
    parser.add_argument(
        "--fewshot-head-lr-scale",
        type=float,
        default=5.0,
        help="Scale factor applied to base LR for few-shot classifier params",
    )

    # Phase scheduling arguments (Stage 2/3 fine-tune, loss & augmentation transitions)
    parser.add_argument(
        "--stage2-epoch", type=int, default=10, help="Epoch to start Stage 2 partial fine-tune"
    )
    parser.add_argument(
        "--stage3-epoch",
        type=int,
        default=20,
        help="Epoch to start Stage 3 full backbone fine-tune",
    )
    parser.add_argument(
        "--smoothing-decay-epoch1", type=int, default=12, help="Epoch to reduce label smoothing"
    )
    parser.add_argument(
        "--smoothing-decay-epoch2", type=int, default=18, help="Epoch to set label smoothing to 0"
    )
    parser.add_argument(
        "--disable-sampler-epoch",
        type=int,
        default=15,
        help="Epoch to disable weighted sampler (switch to shuffle)",
    )
    parser.add_argument(
        "--lr-restart-epoch", type=int, default=25, help="Epoch for LR warm restart (minor boost)"
    )
    parser.add_argument(
        "--tail-focal-epoch",
        type=int,
        default=18,
        help="Epoch to enable focal loss for tail classes",
    )
    parser.add_argument(
        "--progressive-resize-epoch",
        type=int,
        default=30,
        help="Epoch to increase input resolution for fine detail",
    )
    parser.add_argument(
        "--progressive-image-size",
        type=int,
        default=256,
        help="Image size after progressive resize phase starts",
    )
    parser.add_argument(
        "--final-clean-epoch",
        type=int,
        default=40,
        help="Epoch to switch to minimal augmentation for final convergence",
    )
    # Dynamic regularization decay (Mixup/CutMix)
    parser.add_argument(
        "--mixup-decay-epoch1",
        type=int,
        default=10,
        help="Epoch to reduce Mixup alpha (e.g. 0.4 -> 0.2)",
    )
    parser.add_argument(
        "--mixup-decay-epoch2",
        type=int,
        default=12,
        help="Epoch to further reduce Mixup alpha (e.g. 0.2 -> 0.1)",
    )
    parser.add_argument(
        "--mixup-disable-epoch",
        type=int,
        default=15,
        help="Epoch to disable Mixup (alpha -> 0)",
    )
    parser.add_argument(
        "--cutmix-disable-epoch",
        type=int,
        default=10,
        help="Epoch to disable CutMix earlier than Mixup",
    )
    # Cosine classifier & CenterLoss for final margin sharpening
    parser.add_argument(
        "--use-cosine-classifier",
        action="store_true",
        help="Replace linear head with cosine classifier in final phase",
    )
    parser.add_argument(
        "--use-center-loss",
        action="store_true",
        help="Enable CenterLoss in final convergence stage",
    )
    parser.add_argument(
        "--center-loss-weight",
        type=float,
        default=0.01,
        help="Weight for CenterLoss term",
    )
    parser.add_argument(
        "--center-update-epoch",
        type=int,
        default=35,
        help="Epoch to enable cosine classifier / center loss adaptations",
    )

    args = parser.parse_args()
    return args


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()

    # Set seed
    set_seed(args.seed)

    # Device - MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nUsing device: {device} (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nUsing device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print(f"\nUsing device: {device}")

    # Optimize DataLoader settings per device
    # pin_memory only works on CUDA
    use_pin_memory = device.type == "cuda"
    # Optimal workers: MPS needs fewer due to memory transfer overhead
    if device.type == "mps":
        optimal_workers = min(4, args.num_workers)
    elif device.type == "cuda":
        optimal_workers = args.num_workers
    else:
        optimal_workers = min(8, args.num_workers)
    use_persistent = optimal_workers > 0

    print("\n" + "=" * 60)
    print("Agricultural Disease Recognition - Training")
    print("=" * 60)
    print(f"Task: Task 1 - 61-class classification")
    print(f"Model: {args.backbone} ({args.model_type})")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Image size: {args.image_size}")
    print("=" * 60)

    # Create datasets
    print("\nLoading datasets...")
    train_transform = get_train_transform(args.image_size)
    val_transform = get_val_transform(args.image_size)

    train_dataset = AgriDiseaseDataset(
        data_dir=args.train_dir,
        metadata_path=args.train_meta,
        transform=train_transform,
        return_multitask=(args.model_type == "multitask"),
    )

    val_dataset = AgriDiseaseDataset(
        data_dir=args.val_dir,
        metadata_path=args.val_meta,
        transform=val_transform,
        return_multitask=(args.model_type == "multitask"),
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders with device-optimized settings (with optional balanced sampler)
    sampler = None
    if args.balance_sampler:
        try:
            sampler = build_weighted_sampler(train_dataset.metadata, label_col="label_61")
            print("✅ Using WeightedRandomSampler (inverse sqrt frequency)")
        except Exception as e:
            print(f"⚠️ Failed to build sampler ({e}), falling back to shuffle.")
            sampler = None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=optimal_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
        sampler=sampler,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=optimal_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
    )

    print(f"DataLoader workers: {optimal_workers}")
    print(f"Pin memory: {use_pin_memory}")
    print(f"Persistent workers: {use_persistent}")

    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")

    # Parse progressive resize schedule
    progressive_resize_enabled = args.progressive_resize
    if progressive_resize_enabled:
        target_sizes = [int(s) for s in args.progressive_sizes.split(",") if s.strip()]
        resize_epochs = [int(e) for e in args.resize_epochs.split(",") if e.strip()]
        if len(target_sizes) < 2 or len(resize_epochs) != len(target_sizes) - 1:
            raise ValueError("Progressive resize configuration invalid: sizes or epochs mismatch")
        print(f"Progressive resize enabled: sizes={target_sizes}, switch_epochs={resize_epochs}")
    if args.model_type == "fewshot":
        # Build a pretrained baseline backbone first
        print("Initializing baseline backbone for few-shot adaptation...")
        backbone_model = create_model(
            model_type="baseline",
            backbone=args.backbone,
            pretrained=args.pretrained,
            dropout=args.dropout,
            num_classes=61,
        )
        backbone_model = backbone_model.to(device)
        # Wrap with FewShotModel
        print("Wrapping backbone with FewShotModel head...")
        model = create_model(
            model_type="fewshot",
            backbone=args.backbone,
            pretrained=False,  # already loaded
            pretrained_model=backbone_model,
            num_classes=61,
            freeze_backbone=args.fewshot_freeze_backbone
            if hasattr(args, "fewshot_freeze_backbone")
            else True,
        )
        model = model.to(device)
        # Adjust optimizer later: head gets higher LR
    else:
        model = create_model(
            model_type=args.model_type,
            backbone=args.backbone,
            pretrained=args.pretrained,
            dropout=args.dropout,
            num_classes=61,
        )
        model = model.to(device)
    # Preserve reference for training
    raw_model = model

    # Optimize memory layout for better performance (CUDA only - MPS has issues)
    if device.type == "cuda":
        # Removed invalid model.to(memory_format=...) call; channels_last will be applied to input tensors in the loop
        print("CUDA device detected - will use channels_last on batch tensors")

    # torch.compile removed: keep raw_model for all training to reduce complexity
    # Removed compiled_model (torch.compile disabled); using raw_model directly

    # Create loss function
    print("\nCreating loss function...")
    class_weights_path = args.class_weights if args.use_class_weights else None
    criterion = create_loss_function(
        loss_type=args.loss_type,
        class_weights_path=class_weights_path,
        device=device,
        label_smoothing=args.label_smoothing,
    )

    # Create optimizer
    print("\nCreating optimizer...")
    # Save initial label smoothing to allow mid-training reduction
    initial_label_smoothing = args.label_smoothing
    reduced_label_smoothing = 0.02 if initial_label_smoothing > 0.02 else initial_label_smoothing
    if args.model_type == "fewshot":
        # Scale LR for few-shot classifier head with safe getattr fallback
        base_lr = args.lr
        head_lr = base_lr * args.fewshot_head_lr_scale
        head_module = getattr(model, "classifier", None)
        if head_module is not None and hasattr(head_module, "parameters"):
            head_params = [p for p in head_module.parameters() if p.requires_grad]
            param_groups = [{"params": head_params, "lr": head_lr}]
            head_lr_display = head_lr
        else:
            # Fallback: no distinct classifier module; use all trainable params
            fallback_params = [p for p in model.parameters() if p.requires_grad]
            param_groups = [{"params": fallback_params, "lr": head_lr}]
            head_lr_display = head_lr
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                param_groups,
                lr=base_lr,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=base_lr,
                weight_decay=args.weight_decay,
            )
        else:  # sgd
            optimizer = torch.optim.SGD(
                param_groups,
                lr=base_lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
    else:
        if args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        elif args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
            )
        else:  # sgd
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay,
            )
        head_lr_display = args.lr
    print(f"Optimizer: {args.optimizer} (effective head lr={head_lr_display:.6f})")

    # Create scheduler with warmup
    # Good taste: warmup prevents pretrained model from destroying learned features
    scheduler = None
    if args.scheduler == "cosine":
        # Warmup for first 5 epochs, then cosine decay
        warmup_epochs = 5
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.02,  # Start from lr * 0.02 = 1e-5
            end_factor=1.0,  # Reach full lr = 5e-4
            total_iters=warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - warmup_epochs,  # Decay after warmup
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
        print(f"Scheduler: Warmup ({warmup_epochs} epochs) + Cosine Annealing")
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.1,
        )
        print(f"Scheduler: Step LR")

    # Create trainer
    print("\nInitializing trainer...")
    # Cast scheduler only if it's an _LRScheduler, otherwise pass None (custom loop handles advanced schedulers)
    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
        print(
            "Scheduler is not _LRScheduler; passing None to Trainer (will still be used in custom loop if enabled)."
        )
        trainer_scheduler = None
    else:
        trainer_scheduler = scheduler
    trainer = Trainer(
        model=raw_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=trainer_scheduler,
        criterion=criterion,
        device=device,
        save_dir=args.save_dir,
        use_amp=args.use_amp,
        log_interval=args.log_interval,
        use_tensorboard=True,
        multi_task=(args.model_type == "multitask"),
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Start training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    # Decide training path: if advanced features requested, use custom loop
    advanced_path = (
        args.balance_sampler or (args.mixup_alpha > 0) or (args.cutmix_alpha > 0) or args.use_ema
    )
    ema_obj = None
    if advanced_path and args.use_ema:
        ema_obj = ModelEMA(model, decay=args.ema_decay)
        print(f"✅ EMA enabled (decay={args.ema_decay})")
    if advanced_path:
        print("\n=== Using custom training loop (EMA / Mixup / CutMix / Sampler) ===")
        # Use compiled_model only for the forward/backward path; EMA shadows raw_model parameters
        custom_train_loop(
            model=raw_model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            use_amp=args.use_amp,
            multi_task=(args.model_type == "multitask"),
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mixup_prob=args.mixup_prob,
            cutmix_prob=args.cutmix_prob,
            save_dir=args.save_dir,
            ema=ema_obj,
            stage2_epoch=args.stage2_epoch,
            stage3_epoch=args.stage3_epoch,
            smoothing_decay_epoch1=args.smoothing_decay_epoch1,
            smoothing_decay_epoch2=args.smoothing_decay_epoch2,
            disable_sampler_epoch=args.disable_sampler_epoch,
            lr_restart_epoch=args.lr_restart_epoch,
            tail_focal_epoch=args.tail_focal_epoch,
            progressive_resize_epoch=args.progressive_resize_epoch,
            progressive_image_size=args.progressive_image_size,
            final_clean_epoch=args.final_clean_epoch,
            mixup_decay_epoch1=args.mixup_decay_epoch1,
            mixup_decay_epoch2=args.mixup_decay_epoch2,
            mixup_disable_epoch=args.mixup_disable_epoch,
            cutmix_disable_epoch=args.cutmix_disable_epoch,
        )
        print("\nCustom loop finished. Skipping Trainer.train().")
    else:
        try:
            trainer.train(num_epochs=args.epochs, save_freq=args.save_freq)
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            print("Saving current state...")
            # Provide an empty metrics dict (Dict[str, float]) to satisfy type expectations
            trainer.save_checkpoint("interrupted.pth", {})
        except Exception as e:
            print(f"\n\nTraining failed with error: {e}")
            raise

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Checkpoints saved to: {args.save_dir}")
    print("\nNext steps:")
    print("  1. Check TensorBoard logs: tensorboard --logdir {}/logs".format(args.save_dir))
    print("  2. Evaluate on test set")
    print("  3. Analyze error cases")


if __name__ == "__main__":
    main()
