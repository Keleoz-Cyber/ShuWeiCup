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

        print(f"Trainer initialized:")
        print(f"  Device: {device}")
        print(f"  AMP: {self.use_amp}")
        print(f"  Multi-task: {multi_task}")

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dict of metrics (loss, accuracy, etc.)
        """
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Multi-task metrics
        if self.multi_task:
            task_losses = {
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

        for batch_idx, (images, labels) in enumerate(pbar):
            # Move to device - use non_blocking for GPU to overlap transfer
            non_blocking = self.device.type in ["cuda", "mps"]
            images = images.to(self.device, non_blocking=non_blocking)

            # Convert to channels_last for better performance on GPU (CUDA only - MPS has issues)
            if self.device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)

            if self.multi_task:
                # Multi-task labels
                labels = {
                    k: v.to(self.device, non_blocking=non_blocking) for k, v in labels.items()
                }
            else:
                # Single task
                labels = labels["label_61"].to(self.device, non_blocking=non_blocking)

            # Forward pass with AMP
            if self.use_amp:
                with autocast(device_type=self.device.type):
                    if self.multi_task:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    else:
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
            else:
                if self.multi_task:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

            # Backward pass - zero_grad with set_to_none for better performance
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                self.scaler.scale(loss).backward()
                # Gradient clipping for stability
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            # Metrics
            batch_size = images.size(0)
            total_samples += batch_size

            if self.multi_task:
                # Multi-task metrics
                total_loss += loss.item() * batch_size

                # Primary task accuracy (label_61)
                _, predicted = outputs["label_61"].max(1)
                total_correct += predicted.eq(labels["label_61"]).sum().item()

                # Per-task metrics (if criterion returns dict)
                if isinstance(loss, dict):
                    for key in task_losses.keys():
                        task_losses[key] += loss[key].item() * batch_size

                # Per-task accuracy
                for key in task_correct.keys():
                    _, pred = outputs[key].max(1)
                    task_correct[key] += pred.eq(labels[key]).sum().item()
            else:
                # Single task
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
        }

        if self.multi_task:
            for key in task_correct.keys():
                metrics[f"acc_{key}"] = 100.0 * task_correct[key] / total_samples

        return metrics

    @torch.inference_mode()
    def validate(self) -> Dict[str, float]:
        """
        Validate the model.

        Returns:
            Dict of validation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Multi-task metrics
        if self.multi_task:
            task_correct = {
                "label_61": 0,
                "crop": 0,
                "disease": 0,
                "severity": 0,
            }

        for images, labels in tqdm(self.val_loader, desc="Validating"):
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

            # Forward pass
            if self.multi_task:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Metrics
            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            if self.multi_task:
                # Primary task accuracy
                _, predicted = outputs["label_61"].max(1)
                total_correct += predicted.eq(labels["label_61"]).sum().item()

                # Per-task accuracy
                for key in task_correct.keys():
                    _, pred = outputs[key].max(1)
                    task_correct[key] += pred.eq(labels[key]).sum().item()
            else:
                _, predicted = outputs.max(1)
                total_correct += predicted.eq(labels).sum().item()

        # Compute metrics
        metrics = {
            "loss": total_loss / total_samples,
            "accuracy": 100.0 * total_correct / total_samples,
        }

        if self.multi_task:
            for key in task_correct.keys():
                metrics[f"acc_{key}"] = 100.0 * task_correct[key] / total_samples

        return metrics

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

            # Step scheduler
            if self.scheduler is not None:
                self.scheduler.step()

            # Log metrics
            print(f"\nEpoch {epoch}/{num_epochs}:")
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

        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"  Epoch: {self.epoch}")
        print(f"  Best val acc: {self.best_val_acc:.2f}%")
