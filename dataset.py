"""
Agricultural Disease Dataset
=============================

"Bad programmers worry about the code. Good programmers worry about data structures."

This module implements the dataset class for agricultural disease recognition.
Key design principles:
1. Simple, clean interface
2. Multi-task label support (61-class, crop, disease, severity)
3. Efficient data loading
4. No special cases - uniform handling of all samples
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


class AgriDiseaseDataset(Dataset):
    """
    Agricultural disease dataset with multi-task labels.

    Good taste: one dataset class handles all tasks.
    No need for separate classes for Task 1/2/3/4.
    """

    def __init__(
        self,
        data_dir: str,
        metadata_path: str,
        transform: Optional[A.Compose] = None,
        return_multitask: bool = True,
    ):
        """
        Args:
            data_dir: Root directory containing class_XX folders
            metadata_path: Path to metadata CSV file
            transform: Albumentations transform pipeline
            return_multitask: If True, return multi-task labels
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.return_multitask = return_multitask

        # Load metadata
        self.metadata = pd.read_csv(metadata_path)

        # Simple sanity check
        if len(self.metadata) == 0:
            raise ValueError(f"Empty metadata file: {metadata_path}")

        print(f"Loaded {len(self.metadata)} samples from {metadata_path}")

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, int]]:
        """
        Get a sample.

        Returns:
            image: [3, H, W] tensor
            labels: Dict with keys 'label_61', 'crop', 'disease', 'severity'

        Good taste: return dict instead of tuple for extensibility.
        """
        # Get metadata
        row = self.metadata.iloc[idx]

        # Construct image path
        class_folder = f"class_{row['label_61']:02d}"
        image_path = self.data_dir / class_folder / row["image_name"]

        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        # Prepare labels
        if self.return_multitask:
            labels = {
                "label_61": int(row["label_61"]),
                "crop": int(row["crop_id"]),
                "disease": int(row["disease_id"]),
                "severity": int(row["severity"]),
            }
        else:
            # Single task (Task 1)
            labels = {"label_61": int(row["label_61"])}

        return image, labels


def get_train_transform(image_size: int = 224) -> A.Compose:
    """
    Training data augmentation (moderated).

    Previous version was overly aggressive (elastic/optical/shadow/fog + heavy dropout)
    which can destroy subtle lesion textures. Good taste: keep only transforms
    that preserve pathological signal while adding diversity.
    """
    return A.Compose(
        [
            # Random resized crop (scale keeps texture, adds position variance)
            A.RandomResizedCrop(image_size, image_size, scale=(0.85, 1.0), ratio=(0.9, 1.1), p=1.0),
            # Orientation variance (fields can flip)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            # Mild geometric perturbation (smaller than before)
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.10,
                rotate_limit=25,
                p=0.5,
            ),
            # Color jitter (reduced strength to avoid masking chlorosis patterns)
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.08,
                p=0.7,
            ),
            # Light blur/noise (avoid motion blur; keep Gaussian + slight noise)
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 5)),
                    A.GaussNoise(var_limit=(10.0, 40.0)),
                ],
                p=0.3,
            ),
            # Reduced coarse dropout (fewer holes; avoid erasing all lesions)
            A.CoarseDropout(
                max_holes=6,
                max_height=int(image_size * 0.12),
                max_width=int(image_size * 0.12),
                min_holes=2,
                fill_value=0,
                p=0.3,
            ),
            # Normalize
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_light_train_transform(image_size: int = 224) -> A.Compose:
    """
    Extra-light augmentation variant for fine-tuning late stages or EMA evaluation.

    Use when model starts overfitting with standard train transform.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.05,
                p=0.5,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_val_transform(image_size: int = 224) -> A.Compose:
    """
    Validation/test transform - no augmentation.

    Good taste: keep validation deterministic.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_strong_augment_transform(image_size: int = 224) -> A.Compose:
    """
    Strong augmentation for few-shot learning (Task 2).

    When you have only 10 samples per class, data augmentation is critical.
    """
    return A.Compose(
        [
            A.Resize(int(image_size * 1.14), int(image_size * 1.14)),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.7),
            A.OneOf(
                [
                    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                    A.HueSaturationValue(
                        hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                ],
                p=0.8,
            ),
            A.OneOf(
                [
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.GaussNoise(var_limit=(10.0, 80.0)),
                    A.ISONoise(),
                ],
                p=0.5,
            ),
            A.CoarseDropout(
                max_holes=12,
                max_height=int(image_size * 0.15),
                max_width=int(image_size * 0.15),
                min_holes=3,
                fill_value=0,
                p=0.5,
            ),
            A.GridDistortion(p=0.3),
            A.ElasticTransform(p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def collate_fn(batch):
    """
    Custom collate function for multi-task labels.

    Good taste: handle dict labels cleanly.
    """
    images = torch.stack([item[0] for item in batch])

    # Collect labels
    labels = {}
    first_label_dict = batch[0][1]

    for key in first_label_dict.keys():
        labels[key] = torch.tensor([item[1][key] for item in batch], dtype=torch.long)

    return images, labels


if __name__ == "__main__":
    # Test the dataset
    print("Testing Agricultural Disease Dataset")
    print("=" * 60)

    # Paths
    train_dir = "data/cleaned/train"
    train_meta = "data/cleaned/metadata/train_metadata.csv"

    # Create dataset
    train_transform = get_train_transform(224)
    dataset = AgriDiseaseDataset(
        data_dir=train_dir,
        metadata_path=train_meta,
        transform=train_transform,
        return_multitask=True,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test loading a sample
    image, labels = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {image.shape}")
    print(f"  Image dtype: {image.dtype}")
    print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Labels: {labels}")

    # Test dataloader
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Use 0 for testing
        collate_fn=collate_fn,
    )

    print(f"\nTesting DataLoader...")
    images_batch, labels_batch = next(iter(dataloader))
    print(f"  Batch images shape: {images_batch.shape}")
    print(f"  Batch labels keys: {labels_batch.keys()}")
    print(f"  Batch label_61 shape: {labels_batch['label_61'].shape}")

    print("\n✅ Dataset test passed!")
