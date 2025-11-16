#!/usr/bin/env python3
"""
Sample Visualization Tool
=========================

Randomly sample images and visualize predictions vs ground truth.
Simple, clear, informative.
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from models import BaselineModel, MultiTaskModel


def load_model(model_path: str, model_type: str = "baseline", device: str = "cpu"):
    """Load model from checkpoint with auto-detection of architecture."""
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Remove _orig_mod prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    # Create model
    if model_type == "baseline":
        model = BaselineModel(backbone="resnet50", num_classes=61, pretrained=False)
    elif model_type == "multitask":
        # Infer architecture from checkpoint
        num_classes_61 = 61
        num_crops = 10
        num_diseases = 28
        num_severity = 4

        for k, v in new_state_dict.items():
            if k == "head_61class.1.weight":
                num_classes_61 = v.shape[0]
            elif k == "head_crop.1.weight":
                num_crops = v.shape[0]
            elif k == "head_disease.1.weight":
                num_diseases = v.shape[0]
            elif k == "head_severity.1.weight":
                num_severity = v.shape[0]

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

    return model


def get_transform():
    """Standard preprocessing transform."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def scan_dataset(data_dir: str):
    """Scan dataset and return list of (image_path, label)."""
    data_path = Path(data_dir)
    samples = []

    for class_dir in sorted(data_path.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.startswith("class_"):
            continue

        label = int(class_dir.name.split("_")[1])

        for img_path in class_dir.glob("*.jpg"):
            samples.append((img_path, label))

    return samples


def predict(model, image_path: Path, transform, device: str = "cpu"):
    """Run inference on single image."""
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)

        # Handle different output formats
        if isinstance(output, dict):
            # MultiTaskModel returns dict
            logits = output["label_61"]
        else:
            # BaselineModel returns tensor
            logits = output

        probs = torch.softmax(logits, dim=1)
        confidence, pred_label = torch.max(probs, dim=1)

    return pred_label.item(), confidence.item()


def draw_text_with_background(
    draw, text, position, font, text_color=(255, 255, 255), bg_color=(0, 0, 0)
):
    """Draw text with background rectangle for better visibility."""
    # Get text bounding box
    bbox = draw.textbbox(position, text, font=font)

    # Add padding
    padding = 5
    bbox = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)

    # Draw background rectangle
    draw.rectangle(bbox, fill=bg_color)

    # Draw text
    draw.text(position, text, fill=text_color, font=font)


def visualize_prediction(
    image_path: Path, true_label: int, pred_label: int, confidence: float, output_path: Path
):
    """Create visualization with prediction and ground truth."""
    # Load original image
    image = Image.open(image_path).convert("RGB")

    # Resize for display (keep aspect ratio)
    max_size = 800
    ratio = min(max_size / image.width, max_size / image.height)
    new_size = (int(image.width * ratio), int(image.height * ratio))
    image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Create drawing context
    draw = ImageDraw.Draw(image)

    # Try to load a nice font, fall back to default
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        try:
            font_large = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32
            )
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except:
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()

    # Determine if prediction is correct
    is_correct = true_label == pred_label

    # Choose colors
    if is_correct:
        pred_color = (0, 255, 0)  # Green for correct
        pred_bg = (0, 128, 0)
    else:
        pred_color = (255, 0, 0)  # Red for incorrect
        pred_bg = (128, 0, 0)

    # Draw ground truth at top
    gt_text = f"Ground Truth: Class {true_label}"
    draw_text_with_background(
        draw, gt_text, (10, 10), font_small, text_color=(255, 255, 255), bg_color=(0, 0, 0)
    )

    # Draw prediction below
    pred_text = f"Prediction: Class {pred_label}"
    draw_text_with_background(
        draw, pred_text, (10, 50), font_large, text_color=pred_color, bg_color=pred_bg
    )

    # Draw confidence
    conf_text = f"Confidence: {confidence:.1%}"
    draw_text_with_background(
        draw, conf_text, (10, 95), font_small, text_color=(255, 255, 255), bg_color=(64, 64, 64)
    )

    # Draw status indicator
    status_text = "✓ CORRECT" if is_correct else "✗ WRONG"
    status_y = image.height - 50
    draw_text_with_background(
        draw, status_text, (10, status_y), font_large, text_color=pred_color, bg_color=pred_bg
    )

    # Draw filename at bottom right
    filename = image_path.name
    filename_bbox = draw.textbbox((0, 0), filename, font=font_small)
    filename_width = filename_bbox[2] - filename_bbox[0]
    filename_x = image.width - filename_width - 10
    draw_text_with_background(
        draw,
        filename,
        (filename_x, status_y),
        font_small,
        text_color=(200, 200, 200),
        bg_color=(40, 40, 40),
    )

    # Save
    image.save(output_path, quality=95)


def main():
    parser = argparse.ArgumentParser(description="Visualize random sample predictions")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset directory")
    parser.add_argument(
        "--model-type", type=str, default="baseline", choices=["baseline", "multitask"]
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--output-dir", type=str, default="sample_visualizations", help="Output directory"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("SAMPLE VISUALIZATION")
    print("=" * 80)
    print(f"Model:       {args.model}")
    print(f"Data:        {args.data}")
    print(f"Model type:  {args.model_type}")
    print(f"Samples:     {args.num_samples}")
    print(f"Output:      {args.output_dir}")
    print(f"Device:      {args.device}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model = load_model(args.model, args.model_type, args.device)
    print("✓ Model loaded")

    # Get transform
    transform = get_transform()

    # Scan dataset
    print("\nScanning dataset...")
    samples = scan_dataset(args.data)
    print(f"✓ Found {len(samples)} images")

    # Random sample
    sampled = random.sample(samples, min(args.num_samples, len(samples)))

    print(f"\nProcessing {len(sampled)} samples...")
    print("-" * 80)

    # Process each sample
    for i, (img_path, true_label) in enumerate(sampled, 1):
        # Predict
        pred_label, confidence = predict(model, img_path, transform, args.device)

        # Create visualization
        output_path = output_dir / f"sample_{i:02d}.jpg"
        visualize_prediction(img_path, true_label, pred_label, confidence, output_path)

        # Print result
        status = "✓" if pred_label == true_label else "✗"
        print(
            f"[{i:2d}/{len(sampled)}] {status} {img_path.name:30s} | True: {true_label:2d} | Pred: {pred_label:2d} | Conf: {confidence:.1%}"
        )

    print("-" * 80)
    print(f"\n✓ Visualizations saved to: {args.output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()
