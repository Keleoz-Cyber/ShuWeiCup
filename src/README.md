# Source Modules (`src/`)

This directory contains the core modules for the Agricultural Disease Classification project.

## 📁 Module Structure

```
src/
├── __init__.py           # Package initialization and exports
├── data_structures.py    # Label hierarchies and data structures
├── dataset.py           # PyTorch dataset implementations
├── models.py            # Neural network architectures
├── losses.py            # Custom loss functions
└── trainer.py           # Training utilities and loop
```

---

## 📦 Modules Overview

### 1. `data_structures.py`

Defines the label hierarchy and data structures for the 61-class agricultural disease dataset.

**Key Components:**

- **`DiseaseLabel`**: Dataclass representing a disease label with crop type, disease name, severity, and health status
- **`LABEL_HIERARCHY`**: Dictionary mapping 61 class IDs to `DiseaseLabel` objects
- **`LABEL_61_TO_NAME`**: Simple mapping from class ID to full label name
- **`get_multitask_labels()`**: Convert 61-class label to multitask labels (crop, disease, severity)
- **`get_severity_4class_mapping()`**: Map 61 classes to 4-class severity categories

**Example Usage:**

```python
from src.data_structures import LABEL_HIERARCHY, get_multitask_labels

# Get disease info for class 15
disease = LABEL_HIERARCHY[15]
print(f"Crop: {disease.crop_type}, Disease: {disease.disease}, Severity: {disease.severity}")

# Get multitask labels
labels = get_multitask_labels(15)
# Returns: {'crop': 1, 'disease': 5, 'severity': 1}
```

---

### 2. `dataset.py`

PyTorch dataset classes and data transformations.

**Key Components:**

- **`AgriDiseaseDataset`**: Main dataset class for loading agricultural disease images
- **`DiseaseDataset`**: Alternative dataset implementation
- **`get_train_transform()`**: Strong augmentation pipeline for training
- **`get_val_transform()`**: Minimal transforms for validation
- **`get_light_train_transform()`**: Light augmentation for final training epochs
- **`collate_fn()`**: Custom collate function for DataLoader

**Example Usage:**

```python
from src.dataset import AgriDiseaseDataset, get_train_transform, get_val_transform

# Create dataset
train_dataset = AgriDiseaseDataset(
    data_dir="data/cleaned/train",
    metadata_path="data/cleaned/metadata/train_metadata.csv",
    transform=get_train_transform(image_size=224)
)

# Use with DataLoader
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

**Augmentation Pipeline:**

Training augmentations include:
- Random horizontal/vertical flips
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation, hue)
- Random erasing
- Normalization with ImageNet statistics

---

### 3. `models.py`

Neural network architectures for disease classification.

**Key Components:**

- **`BaselineModel`**: Single-task model for 61-class classification
- **`MultiTaskModel`**: Multi-task model predicting crop, disease, and severity simultaneously
- **`create_model()`**: Factory function to create models with specified backbones

**Supported Backbones:**
- ResNet (18, 34, 50, 101, 152)
- EfficientNet (B0-B7)
- ConvNeXt
- Vision Transformer (ViT)
- Swin Transformer

**Example Usage:**

```python
from src.models import create_model, MultiTaskModel

# Create baseline model
model = create_model(
    model_name="resnet50",
    num_classes=61,
    pretrained=True
)

# Create multitask model
multitask_model = MultiTaskModel(
    backbone="efficientnet_b3",
    num_crops=10,
    num_diseases=43,
    num_severity=3
)
```

**Architecture:**
```
Input Image (224x224x3)
    ↓
Backbone (pretrained CNN/Transformer)
    ↓
Feature Extraction
    ↓
Task-specific Heads
    ↓
Outputs (crop, disease, severity)
```

---

### 4. `losses.py`

Custom loss functions for improved training.

**Key Components:**

- **`FocalLoss`**: Focal loss for handling class imbalance
- **`LabelSmoothingCrossEntropy`**: Cross-entropy with label smoothing
- **`MultiTaskLoss`**: Combined loss for multi-task learning
- **`create_loss_function()`**: Factory function to create loss functions

**Example Usage:**

```python
from src.losses import FocalLoss, MultiTaskLoss, create_loss_function

# Focal loss for imbalanced classes
focal_loss = FocalLoss(alpha=0.25, gamma=2.0)

# Label smoothing
smooth_loss = create_loss_function("label_smoothing", smoothing=0.1)

# Multitask loss
multitask_loss = MultiTaskLoss(
    task_weights={'crop': 0.3, 'disease': 0.4, 'severity': 0.3}
)
```

**Loss Functions:**

1. **Focal Loss**: `FL(p_t) = -α(1-p_t)^γ log(p_t)`
   - Focuses on hard examples
   - Reduces weight of well-classified examples
   
2. **Label Smoothing**: Prevents overconfident predictions
   - Smooths one-hot labels: `(1-ε)·y + ε/K`
   
3. **Multi-Task Loss**: Weighted combination of task losses
   - `L = w1·L_crop + w2·L_disease + w3·L_severity`

---

### 5. `trainer.py`

Training loop and utilities.

**Key Components:**

- **`Trainer`**: Main training class with automatic mixed precision (AMP)
- Training loop with validation
- Checkpointing (best model, latest model)
- TensorBoard logging
- Learning rate scheduling
- Early stopping support

**Example Usage:**

```python
from src.trainer import Trainer
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    criterion=criterion,
    device="cuda",
    save_dir="checkpoints/experiment1",
    use_amp=True
)

# Train
trainer.train(num_epochs=50, save_freq=5)
```

**Features:**

- ✅ Automatic mixed precision (AMP) for faster training
- ✅ Gradient clipping
- ✅ Best model checkpoint saving
- ✅ Resume from checkpoint
- ✅ TensorBoard integration
- ✅ Progress bars with tqdm
- ✅ Comprehensive metrics logging

---

## 🔧 Installation

The source modules are designed to be imported as a package. Make sure the project root is in your Python path:

```python
# From project root
from src.models import create_model
from src.dataset import AgriDiseaseDataset
from src.losses import FocalLoss
```

Or install in development mode:
```bash
pip install -e .
```

---

## 🎯 Design Principles

1. **Simplicity**: Clear, readable code without over-engineering
2. **Modularity**: Each module has a single, well-defined purpose
3. **Reusability**: Components can be mixed and matched
4. **Type Safety**: Type hints throughout for better IDE support
5. **Documentation**: Comprehensive docstrings and comments

---

## 📊 Data Flow

```
Raw Data
    ↓
data_structures.py (label hierarchy)
    ↓
dataset.py (load and transform)
    ↓
DataLoader (batching)
    ↓
models.py (forward pass)
    ↓
losses.py (compute loss)
    ↓
trainer.py (optimize and validate)
    ↓
Trained Model
```

---

## 🧪 Testing

Each module includes inline documentation and can be tested independently:

```python
# Test data structures
python -c "from src.data_structures import LABEL_HIERARCHY; print(len(LABEL_HIERARCHY))"

# Test dataset
python -c "from src.dataset import AgriDiseaseDataset; print('Dataset OK')"

# Test model creation
python -c "from src.models import create_model; m = create_model('resnet50', 61); print(m)"
```

---

## 🔄 Version History

- **v1.0.0**: Initial release with core modules
  - 61-class classification support
  - Multi-task learning support
  - Comprehensive augmentation pipeline
  - Focal loss and label smoothing
  - Mixed precision training

---

## 📚 References

- **Label Hierarchy**: Based on agricultural disease taxonomy
- **Augmentations**: Inspired by timm library best practices
- **Focal Loss**: Lin et al. "Focal Loss for Dense Object Detection" (2017)
- **Multi-Task Learning**: Ruder "An Overview of Multi-Task Learning" (2017)

---

## 🤝 Contributing

When adding new modules:

1. Follow the existing code style
2. Add comprehensive docstrings
3. Include type hints
4. Update `__init__.py` exports
5. Add usage examples in this README

---

For training scripts and usage examples, see the main [README.md](../README.md).