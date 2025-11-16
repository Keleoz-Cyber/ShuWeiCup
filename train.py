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

# ===============================
# Helper Utilities (Sampler / EMA / Mixup-CutMix / Custom Loop)
# ===============================
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import our modules
from dataset import AgriDiseaseDataset, collate_fn, get_train_transform, get_val_transform
from losses import create_loss_function
from models import create_model
from trainer import Trainer


def build_weighted_sampler(metadata_df, label_col: str = "label_61", power: float = 0.5):
    """
    Build a WeightedRandomSampler with inverse freq^power weights.
    power=0.5 => inverse sqrt frequency (stabilizes extreme long-tail).
    """
    from collections import Counter

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
        H, W = images.size(2), images.size(3)
        cut_rat = math.sqrt(1.0 - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)
        cy = np.random.randint(0, H)
        cx = np.random.randint(0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (H * W))
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
):
    """
    Minimal custom training loop supporting sampler, Mixup/CutMix, EMA.
    Prints epoch metrics similar to Trainer.
    """
    # Type assertions to satisfy static analyzer
    assert isinstance(model, nn.Module), "model must be nn.Module"
    assert scheduler is None or hasattr(scheduler, "step"), (
        "scheduler must implement step() or be None"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    def run_val(current_epoch: int):
        model.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.inference_mode():
            for images, labels in val_loader:
                images = images.to(device)
                if device.type == "cuda":
                    images = images.to(memory_format=torch.channels_last)
                targets = labels["label_61"] if multi_task else labels["label_61"]
                targets = targets.to(device)
                outputs = model(images) if not multi_task else model(images)["label_61"]
                loss = criterion(outputs, targets)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                correct += preds.eq(targets).sum().item()
                total += images.size(0)
        acc = 100.0 * correct / total
        print(f"  [Val] Epoch {current_epoch} | Loss {total_loss / total:.4f} | Acc {acc:.2f}%")
        return acc, total_loss / total

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            if device.type == "cuda":
                images = images.to(memory_format=torch.channels_last)
            targets = labels["label_61"] if multi_task else labels["label_61"]
            targets = targets.to(device)
            images_aug, ta, tb, lam = apply_mixup_cutmix(
                images, targets, mixup_alpha, cutmix_alpha, mixup_prob, cutmix_prob
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                outputs_all = model(images_aug)
                outputs = outputs_all if not multi_task else outputs_all["label_61"]
                loss = lam * criterion(outputs, ta) + (1 - lam) * criterion(outputs, tb)
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
        print(
            f"Epoch {epoch}/{epochs} | Train Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | LR {optimizer.param_groups[0]['lr']:.6f}"
        )
        # Validation with EMA shadow if enabled
        if ema:
            saved = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
            ema.apply_to(model)
            val_acc, val_loss = run_val(epoch)
            # Restore params
            for n, p in model.named_parameters():
                if p.requires_grad and n in saved:
                    p.data.copy_(saved[n])
        else:
            val_acc, val_loss = run_val(epoch)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_acc,
                },
                Path(save_dir) / "best_custom.pth",
            )
            print(f"  ✅ New best (custom loop) val acc: {best_acc:.2f}%")
    print(f"\nCustom training finished. Best Val Acc: {best_acc:.2f}%")


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
        choices=["baseline", "multitask"],
        help="Model type",
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
    model = create_model(
        model_type=args.model_type,
        backbone=args.backbone,
        pretrained=args.pretrained,
        dropout=args.dropout,
        num_classes=61,
    )
    # Type safety assertion
    assert isinstance(model, nn.Module), "create_model must return nn.Module"
    model = model.to(device)
    # Preserve an uncompiled reference for EMA / Trainer (raw_model) before any optional compilation
    raw_model = model

    # Optimize memory layout for better performance (CUDA only - MPS has issues)
    if device.type == "cuda":
        # Removed invalid model.to(memory_format=...) call; channels_last will be applied to input tensors in the loop
        print("CUDA device detected - will use channels_last on batch tensors")

    # torch.compile removed: keep raw_model for all training to reduce complexity
    compiled_model = raw_model

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
    print(f"Optimizer: {args.optimizer}")

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
