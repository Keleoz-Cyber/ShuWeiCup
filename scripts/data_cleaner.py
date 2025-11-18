"""
Data Cleaning and Preprocessing Script
========================================

"Good code is its own best documentation." - Steve McConnell

This script cleans the raw agricultural disease dataset:
1. Parse train_list.txt and validation list files
2. Verify image integrity
3. Parse and structure labels using the hierarchy
4. Create train/val splits with stratification
5. Generate metadata for downstream tasks

Usage:
    python cleaner.py --src data/raw --dst data/cleaned
"""

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import our data structures
from src.data_structures import (
    LABEL_61_TO_NAME,
    LABEL_HIERARCHY,
    get_multitask_labels,
    get_severity_4class_mapping,
)


class DatasetCleaner:
    """Clean and prepare agricultural disease dataset"""

    def __init__(self, src_dir: Path, dst_dir: Path, val_split: float = 0.15):
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.val_split = val_split

        # Statistics
        self.stats = {
            "total_samples": 0,
            "duplicates_removed": 0,
            "corrupted_removed": 0,
            "valid_samples": 0,
            "class_distribution": Counter(),
        }

    def load_list_file(self, list_path: Path) -> List[Tuple[str, int]]:
        """
        Load labels from train_list.txt or test_list.txt

        Format: path/to/image.jpg label_id
        Example: AgriculturalDisease_trainingset/images\1_0.jpg 1
        """
        samples = []
        with open(list_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 2:
                    continue

                # Parse path and label
                image_path = parts[0].replace("\\", "/")  # Windows path fix
                label_id = int(parts[1])

                # Extract just the filename
                filename = Path(image_path).name

                samples.append((filename, label_id))

        return samples

    def is_valid_image(self, image_path: Path) -> bool:
        """
        Check if image is valid and readable.

        Good taste: fail fast, don't try to recover corrupted data.
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False

            # Basic sanity checks
            if img.shape[0] < 32 or img.shape[1] < 32:
                return False

            if len(img.shape) != 3 or img.shape[2] != 3:
                return False

            return True
        except Exception:
            return False

    def clean_dataset(self, subset: str = "train") -> Tuple[pd.DataFrame, Dict]:
        """
        Clean a dataset subset (train or validation).

        Returns:
            DataFrame with cleaned samples and their labels
            Statistics dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"Cleaning {subset} dataset")
        print(f"{'=' * 60}")

        # Determine source directory
        if subset == "train":
            src_images_dir = self.src_dir / "AgriculturalDisease_trainingset" / "images"
            src_list = self.src_dir / "AgriculturalDisease_trainingset" / "train_list.txt"
        else:
            src_images_dir = self.src_dir / "AgriculturalDisease_validationset" / "images"
            src_list = self.src_dir / "AgriculturalDisease_validationset" / "ttest_list.txt"

        if not src_list.exists():
            print(f"Warning: {src_list} not found. Skipping.")
            return pd.DataFrame(), {}

        # Load labels
        print(f"\nLoading labels from {src_list}")
        samples = self.load_list_file(src_list)
        self.stats["total_samples"] += len(samples)
        print(f"Found {len(samples)} samples")

        # Process each sample
        cleaned_samples = []

        for filename, label_id in tqdm(samples, desc="Processing images"):
            image_path = src_images_dir / filename

            # Verify image exists and is valid
            if not image_path.exists():
                # print(f"\nWarning: Image not found: {filename}")
                continue

            if not self.is_valid_image(image_path):
                self.stats["corrupted_removed"] += 1
                print(f"\nWarning: Corrupted image: {filename}")
                continue

            # Get hierarchical labels
            if label_id not in LABEL_HIERARCHY:
                print(f"\nWarning: Unknown label ID: {label_id}")
                continue

            disease_label = LABEL_HIERARCHY[label_id]
            mt_labels = get_multitask_labels(label_id)

            # Create cleaned sample record
            sample = {
                "image_name": filename,
                "image_path": str(image_path),
                "label_61": label_id,
                "label_name": LABEL_61_TO_NAME[label_id],
                "crop_type": disease_label.crop_type,
                "crop_id": mt_labels["crop"],
                "disease": disease_label.disease or "None",
                "disease_id": mt_labels["disease"],
                "severity": int(disease_label.severity),
                "is_healthy": disease_label.is_healthy,
            }

            cleaned_samples.append(sample)
            self.stats["class_distribution"][label_id] += 1

        self.stats["valid_samples"] += len(cleaned_samples)

        # Create DataFrame
        df = pd.DataFrame(cleaned_samples)

        print(f"\nCleaning completed:")
        print(f"  Valid samples: {len(df)}")
        print(f"  Duplicates removed: {self.stats['duplicates_removed']}")
        print(f"  Corrupted removed: {self.stats['corrupted_removed']}")

        return df, self.stats

    def stratified_split(
        self, df: pd.DataFrame, val_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset with stratification by class.

        Good taste: maintain class distribution in both splits.
        """
        from sklearn.model_selection import train_test_split

        if len(df) == 0:
            return df, df

        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            stratify=df["label_61"],
            random_state=42,
        )

        print(f"\nDataset split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")

        return train_df, val_df

    def copy_and_organize(self, df: pd.DataFrame, split: str):
        """
        Copy cleaned images to destination directory.

        Organize by: dst_dir/split/class_name/image.jpg
        """
        print(f"\nOrganizing {split} images...")

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split}"):
            src_path = Path(row["image_path"])
            dst_path = self.dst_dir / split / f"class_{row['label_61']:02d}" / row["image_name"]

            # Create directory
            dst_path.parent.mkdir(parents=True, exist_ok=True)

            # Copy image
            if not dst_path.exists():
                shutil.copy2(src_path, dst_path)

    def save_metadata(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Save metadata files for later use"""
        metadata_dir = self.dst_dir / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)

        # Save DataFrames
        train_df.to_csv(metadata_dir / "train_metadata.csv", index=False)
        val_df.to_csv(metadata_dir / "val_metadata.csv", index=False)

        print(f"\nMetadata saved to {metadata_dir}")

        # Save class distribution
        class_dist = train_df["label_61"].value_counts().sort_index()
        class_dist.to_csv(metadata_dir / "class_distribution.csv")

        # Compute class weights (for weighted loss)
        total_samples = len(train_df)
        num_classes = 61
        class_weights = {}

        for class_id in range(num_classes):
            count = (train_df["label_61"] == class_id).sum()
            if count > 0:
                # Inverse frequency weighting
                weight = total_samples / (num_classes * count)
                class_weights[class_id] = weight
            else:
                class_weights[class_id] = 0.0

        # Save class weights
        weights_df = pd.DataFrame(list(class_weights.items()), columns=["class_id", "weight"])
        weights_df.to_csv(metadata_dir / "class_weights.csv", index=False)

        # Save severity mapping (for Task 3)
        severity_map = get_severity_4class_mapping()
        severity_df = pd.DataFrame(
            list(severity_map.items()), columns=["label_61", "severity_4class"]
        )
        severity_df.to_csv(metadata_dir / "severity_mapping.csv", index=False)

        print("Class weights and mappings saved")

    def print_statistics(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Print dataset statistics"""
        print(f"\n{'=' * 60}")
        print("Dataset Statistics")
        print(f"{'=' * 60}")

        print(f"\nTotal samples: {len(train_df) + len(val_df)}")
        print(f"Training samples: {len(train_df)}")
        print(f"Validation samples: {len(val_df)}")

        print(f"\nClass distribution (training set):")
        class_counts = train_df["label_61"].value_counts().sort_index()
        print(f"  Min samples per class: {class_counts.min()}")
        print(f"  Max samples per class: {class_counts.max()}")
        print(f"  Mean samples per class: {class_counts.mean():.1f}")
        print(f"  Median samples per class: {class_counts.median():.1f}")

        # Check for severely imbalanced classes
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
        if imbalance_ratio > 10:
            print("  ⚠️  Warning: Severe class imbalance detected!")

        # Crop distribution
        print(f"\nCrop distribution:")
        crop_counts = train_df["crop_type"].value_counts()
        for crop, count in crop_counts.items():
            print(f"  {crop:12s}: {count:4d}")

        # Severity distribution
        print(f"\nSeverity distribution:")
        severity_counts = train_df["severity"].value_counts().sort_index()
        severity_names = ["Healthy", "General", "Serious"]
        for sev, count in severity_counts.items():
            print(f"  {severity_names[sev]:8s}: {count:4d}")

    def run(self):
        """Main cleaning pipeline"""
        print("Agricultural Disease Dataset Cleaner")
        print("=" * 60)
        print(f"Source: {self.src_dir}")
        print(f"Destination: {self.dst_dir}")

        # Clean training set
        train_df, _ = self.clean_dataset("train")

        if len(train_df) == 0:
            print("\n❌ No valid training samples found. Check your data directory.")
            return

        # If validation set exists, clean it too
        # Otherwise, split from training set
        val_df_external, _ = self.clean_dataset("validation")

        if len(val_df_external) > 0:
            # Use external validation set
            print("\nUsing external validation set")
            train_split_df = train_df
            val_split_df = val_df_external
        else:
            # Split training set
            print("\nNo external validation set. Splitting training set...")
            train_split_df, val_split_df = self.stratified_split(train_df, self.val_split)

        # Print statistics
        self.print_statistics(train_split_df, val_split_df)

        # Copy and organize images
        self.copy_and_organize(train_split_df, "train")
        self.copy_and_organize(val_split_df, "val")

        # Save metadata
        self.save_metadata(train_split_df, val_split_df)

        print(f"\n{'=' * 60}")
        print("✅ Dataset cleaning completed successfully!")
        print(f"{'=' * 60}")
        print(f"Cleaned data saved to: {self.dst_dir}")
        print("\nNext steps:")
        print("  1. Review the class distribution")
        print("  2. Check metadata files in data/cleaned/metadata/")
        print("  3. Start training with: python train.py --config config_task1.yaml")


def main():
    parser = argparse.ArgumentParser(description="Clean agricultural disease dataset")
    parser.add_argument(
        "--src",
        type=str,
        default="data/raw",
        help="Source directory containing raw dataset",
    )
    parser.add_argument(
        "--dst",
        type=str,
        default="data/cleaned",
        help="Destination directory for cleaned dataset",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio if no external validation set",
    )

    args = parser.parse_args()

    cleaner = DatasetCleaner(
        src_dir=args.src,
        dst_dir=args.dst,
        val_split=args.val_split,
    )

    try:
        cleaner.run()
    except KeyboardInterrupt:
        print("\n\nCleaning interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during cleaning: {e}")
        raise


if __name__ == "__main__":
    main()
