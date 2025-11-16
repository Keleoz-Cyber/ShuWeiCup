#!/bin/bash
# Task 4: Evaluation Script for Trained Model
# ============================================
# Run this after training completes to generate comprehensive evaluation

set -e  # Exit on error

# Configuration
CHECKPOINT="checkpoints/task4_multitask/multitask/best.pth"
VAL_META="data/cleaned/metadata/val_metadata.csv"
VAL_DIR="data/cleaned/val"
OUT_DIR="checkpoints/task4_multitask/evaluation"
BACKBONE="resnet50"
BATCH_SIZE=64
CAM_SAMPLES=50
REPORT_SAMPLES=1000  # Set to large number or remove for all samples

echo "========================================"
echo "Task 4: Post-Training Evaluation"
echo "========================================"
echo ""
echo "Configuration:"
echo "  Checkpoint:    $CHECKPOINT"
echo "  Val metadata:  $VAL_META"
echo "  Val dir:       $VAL_DIR"
echo "  Output dir:    $OUT_DIR"
echo "  Backbone:      $BACKBONE"
echo "  Batch size:    $BATCH_SIZE"
echo "  CAM samples:   $CAM_SAMPLES"
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please ensure training has completed successfully."
    exit 1
fi

# Check if validation data exists
if [ ! -f "$VAL_META" ]; then
    echo "ERROR: Validation metadata not found: $VAL_META"
    exit 1
fi

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: Validation directory not found: $VAL_DIR"
    exit 1
fi

# Create output directory
mkdir -p "$OUT_DIR"

# Run evaluation
echo "Starting evaluation..."
echo ""

python task4_evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --val-meta "$VAL_META" \
    --val-dir "$VAL_DIR" \
    --out-dir "$OUT_DIR" \
    --backbone "$BACKBONE" \
    --batch-size "$BATCH_SIZE" \
    --cam-samples "$CAM_SAMPLES" \
    --report-samples "$REPORT_SAMPLES"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "✅ Evaluation completed successfully!"
    echo "========================================"
    echo ""
    echo "Results saved to: $OUT_DIR"
    echo ""
    echo "Generated files:"
    ls -lh "$OUT_DIR" | tail -n +2
    echo ""
    echo "To view the diagnostic report:"
    echo "  cat $OUT_DIR/diagnostic_report.csv | head -20"
    echo ""
    echo "To view CAM images:"
    echo "  ls $OUT_DIR/grad_cam/*.jpg | head -10"
    echo ""
else
    echo ""
    echo "========================================"
    echo "❌ Evaluation failed with exit code: $EXIT_CODE"
    echo "========================================"
    exit $EXIT_CODE
fi
