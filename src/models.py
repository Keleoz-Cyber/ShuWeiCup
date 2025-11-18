"""
Agricultural Disease Recognition Models
========================================

"Simplicity is prerequisite for reliability." - Dijkstra

This module implements models for agricultural disease recognition:
1. BaselineModel: Simple 61-class classifier (Task 1)
2. MultiTaskModel: Joint learning of crop/disease/severity (Task 4)

Design principles:
- Start simple (ResNet50), optimize later if needed
- No premature optimization
- Clean, maintainable code
"""

from typing import Dict, Optional

import timm
import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    """
    Baseline model for 61-class disease classification.

    Simple, proven architecture. No fancy tricks.
    "Make it work, make it right, make it fast" - in that order.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 61,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        """
        Args:
            backbone: Model architecture from timm
            num_classes: Number of output classes
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate before final classifier
        """
        super().__init__()

        self.num_classes = num_classes

        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool="",  # We'll add our own pooling
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, num_classes),
        )

        print(f"Initialized {backbone} with {self.feature_dim}D features")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, 3, H, W] input images

        Returns:
            logits: [B, num_classes] classification logits
        """
        # Extract features
        features = self.backbone(x)  # [B, C, H, W]

        # Global pooling
        pooled = self.global_pool(features)  # [B, C, 1, 1]
        pooled = pooled.flatten(1)  # [B, C]

        # Classify
        logits = self.classifier(pooled)  # [B, num_classes]

        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification (for visualization)"""
        features = self.backbone(x)
        pooled = self.global_pool(features)
        return pooled.flatten(1)


class MultiTaskModel(nn.Module):
    """
    Multi-task model for joint learning of crop/disease/severity.

    Good taste: shared backbone, separate task heads.
    One model, multiple tasks - no special cases.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout: float = 0.3,
        num_classes_61: int = 61,
        num_crops: int = 10,
        num_diseases: int = 28,
        num_severity: int = 4,
    ):
        """
        Args:
            backbone: Model architecture from timm
            pretrained: Use ImageNet pretrained weights
            dropout: Dropout rate
            num_classes_61: Number of 61-class labels
            num_crops: Number of crop types
            num_diseases: Number of disease types (including "None")
            num_severity: Number of severity levels (Task 3: 4 classes)
        """
        super().__init__()

        # Shared backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
        )

        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]
            # Also save spatial features for Grad-CAM
            self.feature_map_shape = features.shape[2:]

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Task-specific heads
        self.head_61class = self._make_head(self.feature_dim, num_classes_61, dropout)
        self.head_crop = self._make_head(self.feature_dim, num_crops, dropout)
        self.head_disease = self._make_head(self.feature_dim, num_diseases, dropout)
        self.head_severity = self._make_head(self.feature_dim, num_severity, dropout)

        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"Initialized MultiTaskModel with {backbone}")
        print(f"  Feature dim: {self.feature_dim}")
        print(f"  Feature map shape: {self.feature_map_shape}")
        print(f"  Total parameters: {total_params:.2f}M")

        if total_params > 50:
            print(f"  ⚠️  Warning: Model has {total_params:.2f}M parameters (limit: 50M)")

    def _make_head(self, in_features: int, out_features: int, dropout: float) -> nn.Module:
        """
        Create a task-specific classification head.

        Keep it simple: dropout + linear.
        """
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, out_features),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for all tasks.

        Args:
            x: [B, 3, H, W] input images
            return_features: If True, also return feature maps (for Grad-CAM)

        Returns:
            Dict with keys: 'label_61', 'crop', 'disease', 'severity'
            Optionally 'features' if return_features=True
        """
        # Extract features
        feature_maps = self.backbone(x)  # [B, C, H, W]

        # Global pooling
        pooled = self.global_pool(feature_maps)  # [B, C, 1, 1]
        pooled = pooled.flatten(1)  # [B, C]

        # Task-specific predictions
        outputs = {
            "label_61": self.head_61class(pooled),
            "crop": self.head_crop(pooled),
            "disease": self.head_disease(pooled),
            "severity": self.head_severity(pooled),
        }

        if return_features:
            outputs["features"] = feature_maps

        return outputs

    def get_last_conv_layer(self):
        """
        Return the last convolutional layer for Grad-CAM.
        We iterate modules of the backbone and grab the last nn.Conv2d.
        """
        last_conv = None
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            raise RuntimeError("No conv layer found for Grad-CAM.")
        return last_conv

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled features"""
        feature_maps = self.backbone(x)
        pooled = self.global_pool(feature_maps)
        return pooled.flatten(1)


class FewShotModel(nn.Module):
    """
    Few-shot learning model (Task 2).

    Strategy: Freeze pretrained backbone, train only classifier.
    With 10 samples per class, we can't afford to fine-tune the whole network.
    """

    def __init__(
        self,
        backbone_model: nn.Module,
        num_classes: int = 61,
        freeze_backbone: bool = True,
    ):
        """
        Args:
            backbone_model: Pretrained model (BaselineModel or MultiTaskModel)
            num_classes: Number of classes
            freeze_backbone: If True, freeze backbone weights
        """
        super().__init__()

        # Extract backbone
        if isinstance(backbone_model, BaselineModel):
            self.backbone = backbone_model.backbone
            self.global_pool = backbone_model.global_pool
            feature_dim = backbone_model.feature_dim
        elif isinstance(backbone_model, MultiTaskModel):
            self.backbone = backbone_model.backbone
            self.global_pool = backbone_model.global_pool
            feature_dim = backbone_model.feature_dim
        else:
            raise ValueError(f"Unsupported model type: {type(backbone_model)}")

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.global_pool.parameters():
                param.requires_grad = False
            print("Backbone frozen. Only training classifier.")

        # New classifier for few-shot learning
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Higher dropout for few-shot
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        total_params = sum(p.numel() for p in self.parameters()) / 1e6
        print(f"FewShotModel initialized:")
        print(f"  Trainable: {trainable_params:.2f}M / {total_params:.2f}M parameters")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Extract features (no gradient if frozen)
        if not self.training:
            with torch.no_grad():
                features = self.backbone(x)
                pooled = self.global_pool(features).flatten(1)
        else:
            features = self.backbone(x)
            pooled = self.global_pool(features).flatten(1)

        # Classify
        logits = self.classifier(pooled)
        return logits


def create_model(
    model_type: str = "baseline",
    backbone: str = "resnet50",
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: 'baseline', 'multitask', or 'fewshot'
        backbone: Backbone architecture
        pretrained: Use pretrained weights
        **kwargs: Additional model-specific arguments

    Returns:
        Model instance

    Good taste: one entry point, no scattered model creation code.
    """
    if model_type == "baseline":
        model = BaselineModel(
            backbone=backbone,
            pretrained=pretrained,
            **kwargs,
        )
    elif model_type == "multitask":
        model = MultiTaskModel(
            backbone=backbone,
            pretrained=pretrained,
            **kwargs,
        )
    elif model_type == "fewshot":
        # Requires a pretrained model

        pretrained_model = kwargs.pop("pretrained_model", None)
        if pretrained_model is None:
            raise ValueError("fewshot model requires 'pretrained_model' argument")
        model = FewShotModel(
            backbone_model=pretrained_model,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


if __name__ == "__main__":
    # Test models
    print("Testing Agricultural Disease Models")
    print("=" * 60)

    # Test baseline model
    print("\n1. Testing BaselineModel")
    print("-" * 60)
    model = BaselineModel(backbone="resnet50", num_classes=61, pretrained=False)

    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (2, 61), f"Expected (2, 61), got {out.shape}"

    # Test multi-task model
    print("\n2. Testing MultiTaskModel")
    print("-" * 60)
    mt_model = MultiTaskModel(backbone="resnet50", pretrained=False)

    outputs = mt_model(x)
    print(f"Input shape: {x.shape}")
    for key, value in outputs.items():
        print(f"  {key:12s}: {value.shape}")

    # Test with feature return
    outputs_with_feat = mt_model(x, return_features=True)
    print(f"\nWith features:")
    for key, value in outputs_with_feat.items():
        print(f"  {key:12s}: {value.shape}")

    # Test few-shot model
    print("\n3. Testing FewShotModel")
    print("-" * 60)
    fs_model = FewShotModel(model, num_classes=61, freeze_backbone=True)

    out = fs_model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")

    print("\n✅ All model tests passed!")
