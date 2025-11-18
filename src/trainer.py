"""
Training Engine
===============

"Simplicity is the ultimate sophistication." - da Vinci

This module implements the training loop for agricultural disease recognition.

Design principles:
1. Simple, clean training loop
2. No magic - explicit is better than implicit
3. Proper error handling
4. Efficient but readable
"""

import time
from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for server
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class Trainer:
    """
    Training engine for agricultural disease recognition.

    Good taste: one trainer handles both single-task and multi-task training.
    No need for separate trainer classes.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        criterion: nn.Module,
        device: torch.device,
        save_dir: str = "checkpoints",
        use_amp: bool = True,
        log_interval: int = 10,
        use_tensorboard: bool = True,
        multi_task: bool = False,
    ):
        """
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler (optional)
            criterion: Loss function
            device: Device to train on
            save_dir: Directory to save checkpoints
            use_amp: Use automatic mixed precision
            log_interval: Log every N batches
            use_tensorboard: Log to tensorboard
            multi_task: If True, handle multi-task training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.use_amp = use_amp
        self.log_interval = log_interval
        self.multi_task = multi_task

        # Create save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # AMP scaler - only for CUDA (MPS/CPU don't support it well)
        device_type = device.type
        self.use_amp = use_amp and device_type == "cuda"
        self.scaler = GradScaler(device_type) if self.use_amp else None

        # Tensorboard
        self.writer = None
        if use_tensorboard:
            log_dir = self.save_dir / "logs"
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard logging to {log_dir}")

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0

        # Training history for plotting
        self.history = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
        }

        # Multi-task attributes (set externally if needed)
        self.primary_task = "label_61"  # Default primary task for accuracy
        self.dynamic_task_weights = False  # Enable dynamic task weight updating
        self.task_weight_ema = 0.6  # EMA smoothing factor for dynamic weights

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Multi-task: {multi_task}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Supports:
          - Batches optionally containing meta: (images, labels) or (images, labels, meta)
          - Criterion returning either a scalar Tensor OR a dict with per-task losses
          - Dynamic task weights (inverse validation loss EMA) updated in validate()
          - Primary task selection (default 'label_61') for overall accuracy
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        primary_task = getattr(self, "primary_task", "label_61")

        # Multi-task metrics
        if self.multi_task:
            task_losses_accum = {
                "label_61": 0.0,
                "crop": 0.0,
                "disease": 0.0,
                "severity": 0.0,
            }
            task_correct = {
                "label_61": 0,
                "crop": 0,
                "disease": 0,
                "severity": 0,
            }

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Unpack flexible batch
            if len(batch) == 3:
                images, labels, _meta = batch
            else:
                images, labels = batch

            # Move to device - use non_blocking for GPU to overlap transfer
            non_blocking = self.device.type in ["cuda", "mps"]
            images = images.to(self.device, non_blocking=non_blocking)

            # Convert to channels_last for better performance on GPU (CUDA only - MPS has issues)
            if self.device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            if self.multi_task:
                labels = {
                    k: v.to(self.device, non_blocking=non_blocking) for k, v in labels.items()
                }
            else:
                labels = labels["label_61"].to(self.device, non_blocking=non_blocking)

            # Forward pass with AMP
            if self.use_amp:
                with autocast(device_type=self.device.type):
                    outputs = self.model(images)
                    raw_loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                raw_loss = self.criterion(outputs, labels)

            # Normalize loss object
            if isinstance(raw_loss, dict):
                loss = raw_loss.get("total", None)
                if loss is None:
                    # Aggregate manually if 'total' not provided
                    loss = sum(
                        w * raw_loss[k]
                        for k, w in getattr(self.criterion, "task_weights", {}).items()
                        if k in raw_loss
                    )
            else:
                loss = raw_loss  # tensor

            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Metrics
            batch_size = images.size(0)
            total_samples += batch_size

            if self.multi_task:
                total_loss += loss.item() * batch_size

                # Primary task accuracy
                _, predicted_primary = outputs[primary_task].max(1)
                total_correct += predicted_primary.eq(labels[primary_task]).sum().item()

                # Per-task accuracy
                for key in task_correct.keys():
                    _, pred = outputs[key].max(1)
                    task_correct[key] += pred.eq(labels[key]).sum().item()

                # Per-task loss accumulation if dict provided
                if isinstance(raw_loss, dict):
                    for k in task_losses_accum.keys():
                        if k in raw_loss:
                            task_losses_accum[k] += raw_loss[k].item() * batch_size
            else:
                total_loss += loss.item() * batch_size
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()

            # Update progress bar
            current_loss = total_loss / total_samples
            current_acc = 100.0 * total_correct / total_samples
            pbar.set_postfix(
                {
                    "loss": f"{current_loss:.4f}",
                    "acc": f"{current_acc:.2f}%",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            # Log to tensorboard
            if self.writer and batch_idx % self.log_interval == 0:
                self.writer.add_scalar("train/loss", current_loss, self.global_step)
                self.writer.add_scalar("train/acc", current_acc, self.global_step)
                self.writer.add_scalar(
                    "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                )

            self.global_step += 1

        # Epoch metrics
        metrics = {
            "loss": total_loss / total_samples,
            "accuracy": 100.0 * total_correct / total_samples,
            "primary_task": primary_task,
        }

        if self.multi_task:
            for key in task_correct.keys():
                metrics[f"acc_{key}"] = 100.0 * task_correct[key] / total_samples
            if isinstance(raw_loss, dict):
                for k, v in task_losses_accum.items():
                    metrics[f"train_loss_{k}"] = v / total_samples

        return metrics

    @torch.inference_mode()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Supports:
          - Optional meta in batch
          - Criterion dict output
          - Dynamic task weight update (inverse loss EMA)
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        primary_task = getattr(self, "primary_task", "label_61")

        # Multi-task metrics
        if self.multi_task:
            task_correct = {
                "label_61": 0,
                "crop": 0,
                "disease": 0,
                "severity": 0,
            }
            task_losses_accum = {
                "label_61": 0.0,
                "crop": 0.0,
                "disease": 0.0,
                "severity": 0.0,
            }

        for batch in tqdm(self.val_loader, desc="Validating"):
            # Unpack flexible batch
            if len(batch) == 3:
                images, labels, _meta = batch
            else:
                images, labels = batch

            non_blocking = self.device.type in ["cuda", "mps"]
            images = images.to(self.device, non_blocking=non_blocking)

            if self.device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            if self.multi_task:
                labels = {
                    k: v.to(self.device, non_blocking=non_blocking) for k, v in labels.items()
                }
            else:
                labels = labels["label_61"].to(self.device, non_blocking=non_blocking)

            # Forward pass
            outputs = self.model(images)
            raw_loss = self.criterion(outputs, labels)

            if isinstance(raw_loss, dict):
                loss = raw_loss.get("total", None)
                if loss is None:
                    loss = sum(
                        w * raw_loss[k]
                        for k, w in getattr(self.criterion, "task_weights", {}).items()
                        if k in raw_loss
                    )
            else:
                loss = raw_loss

            # Metrics
            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            if self.multi_task:
                # Primary task accuracy
                _, predicted_primary = outputs[primary_task].max(1)
                total_correct += predicted_primary.eq(labels[primary_task]).sum().item()

                for key in task_correct.keys():
                    _, pred = outputs[key].max(1)
                    task_correct[key] += pred.eq(labels[key]).sum().item()

                if isinstance(raw_loss, dict):
                    for k in task_losses_accum.keys():
                        if k in raw_loss:
                            task_losses_accum[k] += raw_loss[k].item() * batch_size
            else:
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()

        # Compute metrics
        metrics = {
            "loss": total_loss / total_samples if total_samples else 0.0,
            "accuracy": 100.0 * total_correct / total_samples if total_samples else 0.0,
            "primary_task": primary_task,
        }

        if self.multi_task:
            for key in task_correct.keys():
                metrics[f"acc_{key}"] = (
                    100.0 * task_correct[key] / total_samples if total_samples else 0.0
                )
            if isinstance(raw_loss, dict):
                for k, v in task_losses_accum.items():
                    metrics[f"val_loss_{k}"] = v / total_samples if total_samples else 0.0

            # Dynamic task weight update
            if getattr(self, "dynamic_task_weights", False) and isinstance(raw_loss, dict):
                eps = 1e-6
                # Average losses per task
                avg_losses = {
                    k: (task_losses_accum[k] / total_samples + eps) for k in task_losses_accum
                }
                inv = {k: 1.0 / avg_losses[k] for k in avg_losses}
                inv_sum = sum(inv.values())
                if inv_sum > 0:
                    # Preserve original total scale
                    current_weights = getattr(self.criterion, "task_weights", {})
                    scale = sum(current_weights.values()) if current_weights else 1.0
                    norm_weights = {k: (inv[k] / inv_sum) * scale for k in inv}
                    alpha = getattr(self, "task_weight_ema", 0.6)
                    new_weights = {}
                    for k in norm_weights:
                        old = current_weights.get(k, norm_weights[k])
                        new_w = alpha * old + (1 - alpha) * norm_weights[k]
                        new_weights[k] = new_w
                    # Apply update
                    self.criterion.task_weights = new_weights
                    metrics["dynamic_task_weights"] = new_weights

        return metrics

    def plot_training_curves(self):
        """
        Plot training curves and save to disk.

        Good taste: one plot shows everything you need.
        No fancy dashboard - just clear, simple visualization.
        """
        if len(self.history["epoch"]) == 0:
            return

        # Create figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Training Progress", fontsize=16, fontweight="bold")

        epochs = self.history["epoch"]

        # Plot 1: Loss curves
        ax1 = axes[0]
        ax1.plot(epochs, self.history["train_loss"], "b-", label="Train Loss", linewidth=2)
        ax1.plot(epochs, self.history["val_loss"], "r-", label="Val Loss", linewidth=2)
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss", fontsize=12)
        ax1.set_title("Loss Curves", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Plot 2: Accuracy curves
        ax2 = axes[1]
        ax2.plot(epochs, self.history["train_acc"], "b-", label="Train Acc", linewidth=2)
        ax2.plot(epochs, self.history["val_acc"], "r-", label="Val Acc", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Accuracy (%)", fontsize=12)
        ax2.set_title("Accuracy Curves", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Add best accuracy marker
        best_idx = self.history["val_acc"].index(max(self.history["val_acc"]))
        best_epoch = self.history["epoch"][best_idx]
        best_acc = self.history["val_acc"][best_idx]
        ax2.plot(
            best_epoch,
            best_acc,
            "g*",
            markersize=15,
            label=f"Best: {best_acc:.2f}% @ Epoch {best_epoch}",
        )
        ax2.legend(fontsize=10)

        # Plot 3: Learning rate schedule
        ax3 = axes[2]
        ax3.plot(epochs, self.history["learning_rate"], "g-", linewidth=2)
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Learning Rate", fontsize=12)
        ax3.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
        ax3.set_yscale("log")  # Log scale for LR
        ax3.grid(True, alpha=0.3)

        # Add text box with current stats
        current_stats = (
            f"Current Epoch: {self.epoch}\n"
            f"Best Val Acc: {self.best_val_acc:.2f}%\n"
            f"Train Acc: {self.history['train_acc'][-1]:.2f}%\n"
            f"Val Acc: {self.history['val_acc'][-1]:.2f}%"
        )
        fig.text(
            0.02,
            0.02,
            current_stats,
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

        # Save figure
        plot_path = self.save_dir / "training_curves.png"
        plt.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close()

        print(f"  📊 Training curves saved to {plot_path}")

    def train(self, num_epochs: int, save_freq: int = 5):
        """
        Full training loop.

        Args:
            num_epochs: Number of epochs to train
            save_freq: Save checkpoint every N epochs
        """
        print(f"\nStarting training for {num_epochs} epochs")
        print("=" * 60)

        start_time = time.time()

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Train
            train_metrics = self.train_epoch()

            # Validate
            val_metrics = self.validate()

            # Get current learning rate before stepping
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Update history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            self.history["learning_rate"].append(current_lr)

            # Log metrics
            print(f"\nEpoch {epoch}/{num_epochs} (LR: {current_lr:.6f}):")
            print(
                f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.2f}%"
            )
            print(f"  Val   Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.2f}%")

            if self.multi_task:
                print(f"  Multi-task accuracies:")
                for key in ["label_61", "crop", "disease", "severity"]:
                    print(f"    {key:12s}: {val_metrics[f'acc_{key}']:.2f}%")

            # Tensorboard logging
            if self.writer:
                self.writer.add_scalar("val/loss", val_metrics["loss"], epoch)
                self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("train/epoch_lr", current_lr, epoch)

                if self.multi_task:
                    for key in ["label_61", "crop", "disease", "severity"]:
                        self.writer.add_scalar(f"val/acc_{key}", val_metrics[f"acc_{key}"], epoch)

            # Save checkpoint
            is_best = val_metrics["accuracy"] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics["accuracy"]
                print(f"  ✅ New best accuracy: {self.best_val_acc:.2f}%")
                self.save_checkpoint("best.pth", val_metrics)

            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pth", val_metrics)

            # Plot training curves after each epoch
            self.plot_training_curves()

        # Training complete
        elapsed_time = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"Training completed in {elapsed_time / 3600:.2f} hours")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print("=" * 60)

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "metrics": metrics,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        save_path = self.save_dir / filename
        torch.save(checkpoint, save_path)
        print(f"  Checkpoint saved: {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_acc = checkpoint["best_val_acc"]

        # Restore training history if available
        if "history" in checkpoint:
            self.history = checkpoint["history"]
            print(f"  Training history restored ({len(self.history['epoch'])} epochs)")

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Best val acc: {self.best_val_acc:.2f}%")
