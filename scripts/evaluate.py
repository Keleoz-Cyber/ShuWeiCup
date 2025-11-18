#!/usr/bin/env python3
"""
Model Evaluation Tool
=====================

"Simplicity is prerequisite for reliability."

Automatically evaluate PT model on dataset with known labels.
No bullshit. No over-engineering. Just load, infer, compare, report.
Usage: python evaluate.py --model best.pth --data data/cleaned/train python evaluate.py --model best.pth --data data/cleaned/val --model-type multitask
"""

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models import BaselineModel, MultiTaskModel


def load_model(
    model_path: str, model_type: str = "baseline", device: str = "cuda"
) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Early failure if something is wrong.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading {model_type} model from {model_path}")

    # Load weights first to infer architecture
    checkpoint = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint

    # Remove _orig_mod prefix if present (from torch.compile)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v  # Remove _orig_mod. prefix
        else:
            new_state_dict[k] = v

    # Create model with correct architecture
    if model_type == "baseline":
        model = BaselineModel(backbone="resnet50", num_classes=61, pretrained=False)
    elif model_type == "multitask":
        # Infer architecture from checkpoint
        num_classes_61 = 61
        num_crops = 10
        num_diseases = 28
        num_severity = 4

        # Try to infer from state dict
        for k, v in new_state_dict.items():
            if k == "head_61class.1.weight":
                num_classes_61 = v.shape[0]
            elif k == "head_crop.1.weight":
                num_crops = v.shape[0]
            elif k == "head_disease.1.weight":
                num_diseases = v.shape[0]
            elif k == "head_severity.1.weight":
                num_severity = v.shape[0]

        print(
            f"Inferred architecture: {num_classes_61} classes, {num_crops} crops, {num_diseases} diseases, {num_severity} severity levels"
        )
        model = MultiTaskModel(
            backbone="resnet50",
            pretrained=False,
            num_classes_61=num_classes_61,
            num_crops=num_crops,
            num_diseases=num_diseases,
            num_severity=num_severity,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    return model


def get_image_transform():
    """
    Image preprocessing.

    Same as training - no special cases.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def scan_dataset(data_dir: str) -> List[Tuple[Path, int]]:
    """
    Scan directory structure for images.

    Expects: data_dir/class_XX/*.jpg
    Returns: [(image_path, true_label), ...]
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    images = []

    # Find all class directories
    class_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("class_")]
    )

    if not class_dirs:
        raise ValueError(f"No class_XX directories found in {data_dir}")

    print(f"Found {len(class_dirs)} classes")

    # Scan each class directory
    for class_dir in class_dirs:
        # Extract class number
        try:
            class_num = int(class_dir.name.split("_")[1])
        except (IndexError, ValueError):
            print(f"Warning: Skipping invalid directory name: {class_dir.name}")
            continue

        # Find all images
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            for img_path in class_dir.glob(ext):
                images.append((img_path, class_num))

    if not images:
        raise ValueError(f"No images found in {data_dir}")

    print(f"Found {len(images)} images")
    return images


def predict(model, image_path: Path, transform, device: str, model_type: str) -> Tuple[int, float]:
    """
    Run inference on single image.

    Returns: (predicted_class, confidence)
    """
    # Load and preprocess image
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return -1, 0.0

    # Infer
    with torch.no_grad():
        if model_type == "baseline":
            logits = model(image_tensor)
        elif model_type == "multitask":
            outputs = model(image_tensor)
            logits = outputs["label_61"]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Get prediction
        probs = F.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    return pred_class.item(), confidence.item()


def visualize_sample(
    image_path: Path,
    true_label: int,
    pred_label: int,
    confidence: float,
    is_correct: bool,
    output_dir: Path,
) -> None:
    """
    Visualize a sample with bounding box and labels.

    Simple, clean visualization. Save to output directory.
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    h, w = img.shape[:2]

    # Draw border rectangle
    color = (0, 255, 0) if is_correct else (0, 0, 255)  # Green if correct, Red if wrong
    thickness = 3
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), color, thickness)

    # Prepare text
    status = "✓ CORRECT" if is_correct else "✗ WRONG"
    text_lines = [
        f"{status}",
        f"True: class_{true_label}",
        f"Pred: class_{pred_label}",
        f"Conf: {confidence:.2%}",
    ]

    # Draw text background and text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    text_thickness = 2
    y_offset = 40

    for i, text in enumerate(text_lines):
        y_pos = y_offset + i * 30
        text_size = cv2.getTextSize(text, font, font_scale, text_thickness)[0]

        # Background rectangle
        cv2.rectangle(
            img,
            (15, y_pos - 25),
            (15 + text_size[0] + 10, y_pos + 5),
            (0, 0, 0),
            -1,
        )

        # Text
        text_color = (0, 255, 0) if is_correct else (0, 0, 255)
        if i == 0:  # Status line
            text_color = (255, 255, 255)
        cv2.putText(img, text, (20, y_pos), font, font_scale, text_color, text_thickness)

    # Save
    output_path = output_dir / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_path), img)
    print(f"  Saved visualization: {output_path.name}")


def evaluate(
    model,
    images: List[Tuple[Path, int]],
    transform,
    device: str,
    model_type: str,
    output_file: str = "evaluation_results.csv",
    sample_count: Optional[int] = None,
    visualize_dir: Optional[Path] = None,
) -> Dict:
    """
    Evaluate model on all images.

    Args:
        sample_count: If set, randomly sample N images for visualization
        visualize_dir: Directory to save visualizations

    Returns: statistics dict
    """
    results = []
    correct = 0
    total = 0

    # Random sampling for visualization
    if sample_count and visualize_dir:
        sample_indices = set(random.sample(range(len(images)), min(sample_count, len(images))))
        visualize_dir.mkdir(exist_ok=True)
        print(f"\nWill visualize {len(sample_indices)} random samples")
    else:
        sample_indices = set()

    print(f"\nEvaluating {len(images)} images...")
    print("-" * 80)

    start_time = time.time()

    for i, (img_path, true_label) in enumerate(images):
        # Predict
        pred_label, confidence = predict(model, img_path, transform, device, model_type)

        # Skip if prediction failed
        if pred_label == -1:
            continue

        # Check correctness
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        total += 1

        # Record result
        results.append(
            {
                "image_path": str(img_path.relative_to(img_path.parent.parent.parent)),
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": f"{confidence:.4f}",
                "correct": is_correct,
            }
        )

        # Visualize if in sample set
        if i in sample_indices:
            visualize_sample(
                img_path,
                true_label,
                pred_label,
                confidence,
                is_correct,
                visualize_dir,
            )

        # Progress
        if (i + 1) % 100 == 0:
            acc = correct / total if total > 0 else 0
            print(f"Processed {i + 1}/{len(images)} | Accuracy: {acc:.2%}")

    elapsed = time.time() - start_time

    # Save results
    if results:
        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["image_path", "true_label", "pred_label", "confidence", "correct"]
            )
            writer.writeheader()
            writer.writerows(results)
        print(f"\n✓ Detailed results saved to: {output_file}")

    # Calculate statistics
    accuracy = correct / total if total > 0 else 0
    stats = {
        "total": total,
        "correct": correct,
        "wrong": total - correct,
        "accuracy": accuracy,
        "time_elapsed": elapsed,
        "images_per_second": total / elapsed if elapsed > 0 else 0,
    }

    return stats


def print_statistics(stats: Dict):
    """
    Print evaluation statistics.

    Clean, readable output.
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Total images:     {stats['total']}")
    print(f"Correct:          {stats['correct']}")
    print(f"Wrong:            {stats['wrong']}")
    print(f"Accuracy:         {stats['accuracy']:.2%}")
    print(f"Time elapsed:     {stats['time_elapsed']:.2f}s")
    print(f"Speed:            {stats['images_per_second']:.1f} images/s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PT model on dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py --model best.pth --data data/cleaned/train
  python evaluate.py --model best.pth --data data/cleaned/val --model-type multitask
  python evaluate.py --model best.pth --data data/cleaned/val --sample 20
  python evaluate.py --model checkpoint.pth --data data/cleaned/train --output results.csv
        """,
    )

    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--data", type=str, required=True, help="Path to data directory")
    parser.add_argument(
        "--model-type",
        type=str,
        default="baseline",
        choices=["baseline", "multitask"],
        help="Model type",
    )
    parser.add_argument(
        "--output", type=str, default="evaluation_results.csv", help="Output CSV file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of random samples to visualize (default: None)",
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="evaluation_samples",
        help="Directory to save visualizations (default: evaluation_samples)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print(f"{'MODEL EVALUATION':^80}")
    print("=" * 80)
    print(f"Checkpoint:  {args.model}")
    print(f"Data dir:    {args.data}")
    print(f"Model type:  {args.model_type}")
    print(f"Device:      {args.device}")
    print(f"Output:      {args.output}")
    print("=" * 80)

    # Load model
    model = load_model(args.model, args.model_type, args.device)

    # Get image transform
    transform = get_image_transform()

    # Collect images
    images = scan_dataset(args.data)

    # Prepare visualization directory
    viz_dir = Path(args.viz_dir) if args.sample else None

    # Run evaluation
    stats = evaluate(
        model,
        images,
        transform,
        args.device,
        args.model_type,
        args.output,
        sample_count=args.sample,
        visualize_dir=viz_dir,
    )

    # Print results
    print_statistics(stats)


if __name__ == "__main__":
    main()
