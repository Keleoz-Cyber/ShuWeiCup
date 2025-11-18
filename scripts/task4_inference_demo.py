#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 4: Inference Demo - Visualize Predictions on Sample Images
================================================================

Load trained model and predict on 10 sample images, then draw annotations
showing all task predictions (label_61, crop, disease, severity) with
confidence scores and Grad-CAM heatmaps.

Usage:
    python task4_inference_demo.py \
        --checkpoint checkpoints/task4_multitask/multitask/best.pth \
        --val-meta data/cleaned/metadata/val_metadata.csv \
        --val-dir data/cleaned/val \
        --out-dir outputs/inference_demo \
        --num-samples 10

Output: Annotated images with predictions overlaid
"""

import argparse
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from src.models import MultiTaskModel
from task4train import MultiTaskSeverity3Dataset, SeverityCAMWrapper

# Try to import Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    CAM_AVAILABLE = True
except ImportError:
    CAM_AVAILABLE = False
    print("⚠️  pytorch-grad-cam not available. Install: pip install grad-cam")


# Label mappings (simplified for demo)
SEVERITY_NAMES = {0: "Healthy", 1: "General", 2: "Serious"}

CROP_NAMES = {
    0: "Apple",
    1: "Cherry",
    2: "Corn",
    3: "Grape",
    4: "Citrus",
    5: "Peach",
    6: "Pepper",
    7: "Potato",
    8: "Strawberry",
    9: "Tomato",
}

# Simplified disease names (you can expand this)
DISEASE_NAMES = {
    0: "Healthy",
    1: "Scab",
    2: "Frogeye Spot",
    3: "Cedar Rust",
    4: "Powdery Mildew",
    5: "Cercospora",
    6: "Puccinia",
    7: "Curvularia",
    8: "Dwarf Mosaic",
    9: "Black Rot",
    10: "Black Measles",
    11: "Leaf Blight",
    12: "Greening",
    13: "Bacterial Spot",
    14: "Early Blight",
    15: "Late Blight",
    16: "Leaf Mold",
    17: "Target Spot",
    18: "Septoria",
    19: "Spider Mite",
    20: "Yellow Curl",
    21: "Mosaic Virus",
    22: "Scorch",
    23: "Other",
}


def load_model(checkpoint_path: str, device: torch.device) -> MultiTaskModel:
    """Load trained model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = MultiTaskModel(
        backbone="resnet50",
        pretrained=False,
        dropout=0.3,
        num_classes_61=61,
        num_crops=10,
        num_diseases=28,
        num_severity=3,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    best_acc = checkpoint.get("best_val_acc", 0.0)
    print(f"✅ Model loaded (epoch {epoch}, best val acc: {best_acc:.2f}%)")

    return model


@torch.no_grad()
def predict_image(
    model: MultiTaskModel,
    image: np.ndarray,
    device: torch.device,
    cam_model=None,
) -> Dict:
    """
    Predict all tasks for a single image.

    Args:
        model: Trained MultiTaskModel
        image: RGB image as numpy array (H, W, 3)
        device: torch device
        cam_model: Optional CAM wrapper for heatmap

    Returns:
        Dict with predictions, confidences, and optional CAM
    """
    # Preprocess
    from albumentations import Compose, Normalize, Resize
    from albumentations.pytorch import ToTensorV2

    transform = Compose(
        [
            Resize(224, 224),
            Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    transformed = transform(image=image)
    img_tensor = transformed["image"].unsqueeze(0).to(device)

    # Forward pass
    outputs = model(img_tensor)

    # Get predictions and confidences
    results = {}
    for task in ["label_61", "crop", "disease", "severity"]:
        logits = outputs[task][0]
        probs = F.softmax(logits, dim=0)
        conf, pred = probs.max(0)
        results[task] = {
            "pred": int(pred.item()),
            "conf": float(conf.item()),
        }

    # Generate CAM if available
    cam_heatmap = None
    if cam_model is not None and CAM_AVAILABLE:
        try:
            with torch.enable_grad():
                grayscale_cam = cam_model(input_tensor=img_tensor)[0]
                cam_heatmap = grayscale_cam
        except Exception as e:
            print(f"⚠️  CAM generation failed: {e}")

    results["cam_heatmap"] = cam_heatmap

    return results


def draw_predictions_on_image(
    image: np.ndarray,
    predictions: Dict,
    ground_truth: Dict = None,
    cam_heatmap: np.ndarray = None,
) -> np.ndarray:
    """
    Draw predictions on image with nice formatting.

    Args:
        image: Original RGB image (H, W, 3)
        predictions: Dict with pred/conf for each task
        ground_truth: Optional ground truth labels
        cam_heatmap: Optional CAM heatmap

    Returns:
        Annotated image
    """
    h, w = image.shape[:2]

    # Create canvas: original image + CAM side-by-side + text overlay
    canvas_width = w * 2 if cam_heatmap is not None else w
    canvas_height = h + 200  # Extra space for text
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    # Place original image
    canvas[:h, :w] = image

    # Place CAM overlay if available
    if cam_heatmap is not None:
        # Resize heatmap to match image
        cam_resized = cv2.resize(cam_heatmap, (w, h))
        # Normalize to 0-255
        cam_uint8 = (cam_resized * 255).astype(np.uint8)
        # Apply colormap
        cam_colored = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        # Blend with original
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        overlay = (image * 0.4 + cam_colored * 0.6).astype(np.uint8)
        canvas[:h, w:] = overlay

    # Draw text annotations
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    y_start = h + 30
    line_height = 35

    # Severity (most important)
    sev_pred = predictions["severity"]["pred"]
    sev_conf = predictions["severity"]["conf"]
    sev_name = SEVERITY_NAMES.get(sev_pred, f"Class{sev_pred}")
    sev_color = (0, 200, 0) if sev_pred == 0 else (255, 140, 0) if sev_pred == 1 else (200, 0, 0)
    text = f"Severity: {sev_name} ({sev_conf:.2%})"
    cv2.putText(canvas, text, (10, y_start), font, font_scale, sev_color, thickness)

    # Crop type
    y_start += line_height
    crop_pred = predictions["crop"]["pred"]
    crop_conf = predictions["crop"]["conf"]
    crop_name = CROP_NAMES.get(crop_pred, f"Crop{crop_pred}")
    text = f"Crop: {crop_name} ({crop_conf:.2%})"
    cv2.putText(canvas, text, (10, y_start), font, font_scale, (0, 0, 200), thickness)

    # Disease
    y_start += line_height
    dis_pred = predictions["disease"]["pred"]
    dis_conf = predictions["disease"]["conf"]
    dis_name = DISEASE_NAMES.get(dis_pred, f"Disease{dis_pred}")
    text = f"Disease: {dis_name} ({dis_conf:.2%})"
    cv2.putText(canvas, text, (10, y_start), font, font_scale, (150, 0, 150), thickness)

    # Label 61 class
    y_start += line_height
    l61_pred = predictions["label_61"]["pred"]
    l61_conf = predictions["label_61"]["conf"]
    text = f"61-Class: {l61_pred} ({l61_conf:.2%})"
    cv2.putText(canvas, text, (10, y_start), font, font_scale, (100, 100, 100), thickness)

    # Ground truth (if provided)
    if ground_truth is not None:
        y_start += line_height + 10
        gt_text = (
            f"GT: Sev={ground_truth.get('severity', '?')} Crop={ground_truth.get('crop', '?')}"
        )
        cv2.putText(canvas, gt_text, (10, y_start), font, 0.5, (128, 128, 128), 1)

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Task 4: Inference Demo")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--val-meta", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="outputs/inference_demo")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cam", action="store_true", help="Generate Grad-CAM heatmaps")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)

    # Setup CAM
    cam_model = None
    if args.use_cam and CAM_AVAILABLE:
        try:
            wrapper = SeverityCAMWrapper(model)
            target_layer = wrapper.get_last_conv_layer()
            cam_model = GradCAM(model=wrapper, target_layers=[target_layer])
            print("✅ Grad-CAM initialized\n")
        except Exception as e:
            print(f"⚠️  Grad-CAM initialization failed: {e}\n")

    # Load dataset to get sample images
    print(f"Loading dataset: {args.val_meta}")
    dataset = MultiTaskSeverity3Dataset(
        data_dir=args.val_dir,
        metadata_csv=args.val_meta,
        augment=False,
        image_size=224,
    )
    print(f"Total samples: {len(dataset)}\n")

    # Select random samples
    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    print(f"Processing {len(indices)} random samples...\n")
    print("=" * 60)

    for i, idx in enumerate(indices, 1):
        # Get sample
        img_tensor, labels, meta = dataset[idx]
        image_name = meta["image_name"]

        # Load original image (without normalization)
        img_path = Path(meta["image_path"])
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"⚠️  Failed to load image: {img_path}")
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Predict
        predictions = predict_image(model, image_rgb, device, cam_model)

        # Print predictions
        print(f"\n[{i}/{len(indices)}] {image_name}")
        print(
            f"  Severity: {SEVERITY_NAMES.get(predictions['severity']['pred'], '?')} "
            f"({predictions['severity']['conf']:.2%})"
        )
        print(
            f"  Crop:     {CROP_NAMES.get(predictions['crop']['pred'], '?')} "
            f"({predictions['crop']['conf']:.2%})"
        )
        print(
            f"  Disease:  {DISEASE_NAMES.get(predictions['disease']['pred'], '?')} "
            f"({predictions['disease']['conf']:.2%})"
        )
        print(
            f"  61-Class: {predictions['label_61']['pred']} ({predictions['label_61']['conf']:.2%})"
        )

        # Ground truth
        gt = {
            "severity": labels["severity"],
            "crop": labels["crop"],
            "disease": labels["disease"],
            "label_61": labels["label_61"],
        }
        print(f"  Ground Truth: Sev={gt['severity']} Crop={gt['crop']} Dis={gt['disease']}")

        # Draw annotations
        annotated = draw_predictions_on_image(
            image_rgb,
            predictions,
            ground_truth=gt,
            cam_heatmap=predictions.get("cam_heatmap"),
        )

        # Save
        output_path = out_dir / f"demo_{i:02d}_{image_name}"
        cv2.imwrite(str(output_path), cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        print(f"  ✅ Saved: {output_path.name}")

    print("\n" + "=" * 60)
    print(f"✅ Demo complete! {len(indices)} images saved to: {out_dir}")
    print("\nTo view results:")
    print(f"  ls {out_dir}/demo_*.jpg")


if __name__ == "__main__":
    main()
