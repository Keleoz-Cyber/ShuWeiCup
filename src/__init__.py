"""
ShuWeiCamp Source Package
=========================

Core modules for crop disease classification tasks.

Modules:
    - data_structures: Label hierarchies and data structures
    - dataset: PyTorch dataset implementations
    - models: Neural network architectures
    - losses: Custom loss functions
    - trainer: Training utilities
"""

from .data_structures import (
    LABEL_61_TO_NAME,
    LABEL_HIERARCHY,
    get_multitask_labels,
    get_severity_4class_mapping,
)
from .dataset import AgriDiseaseDataset
from .losses import FocalLoss, MultiTaskLoss, WeightedCrossEntropyLoss
from .models import BaselineModel, MultiTaskModel
from .trainer import Trainer

__version__ = "1.0.0"

__all__ = [
    # Data structures
    "LABEL_61_TO_NAME",
    "LABEL_HIERARCHY",
    "get_multitask_labels",
    "get_severity_4class_mapping",
    # Dataset
    "AgriDiseaseDataset",
    # Models
    "BaselineModel",
    "MultiTaskModel",
    # Losses
    "FocalLoss",
    "MultiTaskLoss",
    "WeightedCrossEntropyLoss",
    # Trainer
    "Trainer",
]
