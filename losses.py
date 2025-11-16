"""
Loss Functions for Agricultural Disease Recognition
====================================================

"Good code is simple code." - Linus Torvalds

This module implements loss functions for the project:
1. Weighted Cross-Entropy for class imbalance
2. Multi-task loss for joint learning
3. Focal Loss (optional, for hard samples)

Design principles:
- Handle class imbalance properly
- Clean multi-task loss aggregation
- No unnecessary complexity
"""

from typing import Dict, Optional

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross-Entropy Loss for imbalanced datasets.

    The dataset has severe imbalance (ratio 2445:1).
    We MUST use class weights, not optional.
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        """
        Args:
            class_weights: Weight for each class [num_classes]
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
        """
        super().__init__()

        self.class_weights = class_weights
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            print(
                f"Using class weights (min={class_weights.min():.3f}, max={class_weights.max():.3f})"
            )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] predicted logits
            targets: [B] ground truth labels

        Returns:
            loss: scalar loss value
        """
        return F.cross_entropy(
            logits,
            targets,
            weight=self.class_weights,
            label_smoothing=self.label_smoothing,
        )


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for joint learning.

    Good taste: simple weighted sum of task losses.
    No fancy uncertainty weighting or learned weights.
    """

    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        class_weights_61: Optional[torch.Tensor] = None,
        class_weights_severity: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.1,
    ):
        """
        Args:
            task_weights: Weight for each task
            class_weights_61: Class weights for 61-class task
            class_weights_severity: Class weights for severity task
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        # Default task weights
        if task_weights is None:
            task_weights = {
                "label_61": 1.0,  # Primary task
                "crop": 0.3,  # Auxiliary task
                "disease": 0.3,  # Auxiliary task
                "severity": 0.5,  # Secondary task (Task 3)
            }

        self.task_weights = task_weights

        # Task-specific loss functions
        self.criterion_61 = nn.CrossEntropyLoss(
            weight=class_weights_61,
            label_smoothing=label_smoothing,
        )

        self.criterion_crop = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

        self.criterion_disease = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
        )

        self.criterion_severity = nn.CrossEntropyLoss(
            weight=class_weights_severity,
            label_smoothing=label_smoothing,
        )

        print(f"MultiTaskLoss initialized with weights: {task_weights}")

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Dict with keys 'label_61', 'crop', 'disease', 'severity'
            targets: Dict with same keys as outputs

        Returns:
            Dict with per-task losses and 'total' weighted sum:
              {
                'total': ...,
                'label_61': ...,
                'crop': ...,
                'disease': ...,
                'severity': ...
              }
        """
        # Compute individual task losses
        loss_61 = self.criterion_61(outputs["label_61"], targets["label_61"])
        loss_crop = self.criterion_crop(outputs["crop"], targets["crop"])
        loss_disease = self.criterion_disease(outputs["disease"], targets["disease"])
        loss_severity = self.criterion_severity(outputs["severity"], targets["severity"])

        # Weighted sum
        total_loss = (
            self.task_weights["label_61"] * loss_61
            + self.task_weights["crop"] * loss_crop
            + self.task_weights["disease"] * loss_disease
            + self.task_weights["severity"] * loss_severity
        )

        return {
            "total": total_loss,
            "label_61": loss_61,
            "crop": loss_crop,
            "disease": loss_disease,
            "severity": loss_severity,
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for hard example mining.

    Optional - use this if standard CE doesn't work well.
    From: "Focal Loss for Dense Object Detection" (Lin et al.)
    """

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        """
        Args:
            alpha: Class weights [num_classes]
            gamma: Focusing parameter (higher = focus more on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C] predicted logits
            targets: [B] ground truth labels

        Returns:
            loss: Focal loss
        """
        # Compute cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Get probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # Apply class weights
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # Reduce
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


def load_class_weights(
    weights_path: str,
    device: torch.device,
) -> torch.Tensor:
    """
    Load class weights from CSV file.

    Args:
        weights_path: Path to class_weights.csv
        device: Device to load weights to

    Returns:
        class_weights: [num_classes] tensor
    """
    df = pd.read_csv(weights_path)
    weights = torch.tensor(df["weight"].values, dtype=torch.float32)
    weights = weights.to(device)

    print(f"Loaded class weights from {weights_path}")
    print(f"  Shape: {weights.shape}")
    print(f"  Range: [{weights.min():.3f}, {weights.max():.3f}]")

    return weights


def create_loss_function(
    loss_type: str = "weighted_ce",
    class_weights_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    **kwargs,
) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: 'weighted_ce', 'multitask', or 'focal'
        class_weights_path: Path to class weights CSV
        device: Device to load weights to
        **kwargs: Additional loss-specific arguments

    Returns:
        Loss function

    Good taste: one entry point for all loss functions.
    """
    # Load class weights if provided
    class_weights = None
    if class_weights_path is not None:
        class_weights = load_class_weights(class_weights_path, device)

    if loss_type == "weighted_ce":
        loss_fn = WeightedCrossEntropyLoss(
            class_weights=class_weights,
            **kwargs,
        )
    elif loss_type == "multitask":
        loss_fn = MultiTaskLoss(
            class_weights_61=class_weights,
            **kwargs,
        )
    elif loss_type == "focal":
        loss_fn = FocalLoss(
            alpha=class_weights,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss_fn


if __name__ == "__main__":
    # Test loss functions
    print("Testing Loss Functions")
    print("=" * 60)

    # Test weighted CE
    print("\n1. Weighted Cross-Entropy Loss")
    print("-" * 60)

    class_weights = torch.rand(61)
    loss_fn = WeightedCrossEntropyLoss(class_weights=class_weights)

    logits = torch.randn(4, 61)
    targets = torch.randint(0, 61, (4,))

    loss = loss_fn(logits, targets)
    print(f"Logits shape: {logits.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Loss: {loss.item():.4f}")

    # Test multi-task loss
    print("\n2. Multi-Task Loss")
    print("-" * 60)

    mt_loss_fn = MultiTaskLoss(class_weights_61=class_weights)

    outputs = {
        "label_61": torch.randn(4, 61),
        "crop": torch.randn(4, 10),
        "disease": torch.randn(4, 28),
        "severity": torch.randn(4, 4),
    }

    targets = {
        "label_61": torch.randint(0, 61, (4,)),
        "crop": torch.randint(0, 10, (4,)),
        "disease": torch.randint(0, 28, (4,)),
        "severity": torch.randint(0, 4, (4,)),
    }

    loss = mt_loss_fn(outputs, targets)
    print(f"Multi-task loss: {loss.item():.4f}")

    # Test focal loss
    print("\n3. Focal Loss")
    print("-" * 60)

    focal_loss_fn = FocalLoss(alpha=class_weights, gamma=2.0)

    logits = torch.randn(4, 61)
    targets = torch.randint(0, 61, (4,))

    loss = focal_loss_fn(logits, targets)
    print(f"Focal loss: {loss.item():.4f}")

    print("\n✅ All loss function tests passed!")
