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
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import our modules
from dataset import AgriDiseaseDataset, collate_fn, get_train_transform, get_val_transform
from losses import create_loss_function
from models import create_model
from trainer import Trainer


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
        default=64,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
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
        default=0.1,
        help="Label smoothing factor",
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
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Use torch.compile for 20-50%% speedup",
    )

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

    # Create dataloaders with device-optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=optimal_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
        collate_fn=collate_fn,
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
    model = model.to(device)

    # Optimize memory layout for better performance (CUDA only - MPS has issues)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        print("Using channels_last memory format for better performance")

    # torch.compile for 20-50% speedup (PyTorch 2.0+)
    # MPS: disabled due to stride mismatch bugs in convolution_backward
    if args.compile and device.type != "mps":
        try:
            print("\nCompiling model with torch.compile...")
            if device.type == "cuda":
                # CUDA: use max-autotune for best performance
                model = torch.compile(model, mode="max-autotune")
            else:
                # CPU: use default mode
                model = torch.compile(model)
            print("✅ Model compiled successfully")
        except Exception as e:
            print(f"⚠️  torch.compile failed ({e}), continuing without compilation")
    elif args.compile and device.type == "mps":
        print("\n⚠️  torch.compile disabled on MPS (known bugs with convolution_backward)")
        print("    See: https://github.com/pytorch/pytorch/issues")
    else:
        print("\ntorch.compile disabled (use --compile to enable)")

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

    # Create scheduler
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=1e-6,
        )
        print(f"Scheduler: Cosine Annealing")
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=20,
            gamma=0.1,
        )
        print(f"Scheduler: Step LR")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
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

    try:
        trainer.train(num_epochs=args.epochs, save_freq=args.save_freq)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving current state...")
        trainer.save_checkpoint("interrupted.pth", {"status": "interrupted"})
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
