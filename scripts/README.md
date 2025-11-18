# Scripts Directory

This directory contains utility scripts for data processing, model evaluation, and inference.

## 📁 Directory Structure

```
scripts/
├── README.md                    # This file
├── data_cleaner.py             # Data cleaning and preprocessing
├── evaluate.py                 # General model evaluation
├── task4_evaluate.py           # Task 4 comprehensive evaluation
└── task4_inference_demo.py     # Inference demo with visualizations
```

## 🛠️ Scripts Overview

### 1. Data Cleaner (`data_cleaner.py`)

Cleans and preprocesses the raw agricultural disease dataset.

**Features:**
- Parse train_list.txt and validation list files
- Verify image integrity
- Parse and structure labels using the hierarchy
- Create train/val splits with stratification
- Generate metadata for downstream tasks

**Usage:**
```bash
python scripts/data_cleaner.py --src data/raw --dst data/cleaned
```

**Arguments:**
- `--src`: Source directory containing raw dataset (default: `data/raw`)
- `--dst`: Destination directory for cleaned dataset (default: `data/cleaned`)
- `--val-split`: Validation split ratio if no external validation set (default: 0.15)

---

### 2. Model Evaluation (`evaluate.py`)

General purpose evaluation script for trained models.

**Features:**
- Load trained models (baseline or multitask)
- Evaluate on any dataset with known labels
- Generate classification reports
- Support for both single-task and multi-task models

**Usage:**
```bash
# Evaluate baseline model
python scripts/evaluate.py --model best.pth --data data/cleaned/val

# Evaluate multitask model
python scripts/evaluate.py --model best.pth --data data/cleaned/val --model-type multitask
```

**Arguments:**
- `--model`: Path to model checkpoint (.pth file)
- `--data`: Path to dataset directory
- `--model-type`: Model type (`baseline` or `multitask`, default: `baseline`)
- `--batch-size`: Batch size for evaluation (default: 32)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`)

---

### 3. Task 4 Evaluation (`task4_evaluate.py`)

Comprehensive evaluation script specifically for Task 4 multitask models.

**Features:**
- Load trained multitask model
- Generate comprehensive diagnostic reports
- Create Grad-CAM visualizations
- Evaluate severity metrics with confusion matrix
- Export per-sample predictions to CSV

**Usage:**
```bash
python scripts/task4_evaluate.py \
    --checkpoint checkpoints/task4_multitask/best.pth \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --val-dir data/cleaned/val \
    --out-dir checkpoints/task4_multitask/evaluation \
    --batch-size 64 \
    --cam-samples 50 \
    --report-samples 500
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--val-meta`: Path to validation metadata CSV
- `--val-dir`: Path to validation images directory
- `--out-dir`: Output directory for evaluation results
- `--batch-size`: Batch size (default: 64)
- `--cam-samples`: Number of samples for Grad-CAM visualization (default: 50)
- `--report-samples`: Number of samples for detailed report (default: 500)

**Outputs:**
- `diagnostic_report.csv`: Per-sample predictions and metrics
- `confusion_matrix_severity.png`: Confusion matrix visualization
- `training_curves.png`: Training history plots
- `severity_metrics_multitask.json`: Detailed metrics JSON
- `gradcam_samples/`: Grad-CAM visualizations

---

### 4. Inference Demo (`task4_inference_demo.py`)

Interactive inference demonstration with visual annotations.

**Features:**
- Load trained model and predict on sample images
- Draw annotations with all task predictions
- Display confidence scores
- Generate Grad-CAM heatmaps
- Save annotated images

**Usage:**
```bash
python scripts/task4_inference_demo.py \
    --checkpoint checkpoints/task4_multitask/best.pth \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --val-dir data/cleaned/val \
    --out-dir outputs/inference_demo \
    --num-samples 10
```

**Arguments:**
- `--checkpoint`: Path to trained model checkpoint
- `--val-meta`: Path to validation metadata CSV
- `--val-dir`: Path to validation images directory
- `--out-dir`: Output directory for annotated images
- `--num-samples`: Number of samples to process (default: 10)

**Outputs:**
- Annotated images with predictions overlaid
- Grad-CAM heatmaps showing model attention
- Prediction confidence scores

---

## 🔗 Dependencies

All scripts require the core `src/` modules:
- `src.data_structures`: Label hierarchies and mappings
- `src.dataset`: PyTorch dataset implementations
- `src.models`: Model architectures
- `src.losses`: Loss functions
- `src.trainer`: Training utilities

Make sure to install all dependencies:
```bash
pip install -r requirements.txt
# or with uv
uv sync
```

---

## 📝 Notes

1. **Data Paths**: All scripts expect the standard project directory structure:
   ```
   ShuWeiCamp/
   ├── data/
   │   ├── raw/          # Raw dataset
   │   └── cleaned/      # Cleaned dataset
   ├── checkpoints/      # Model checkpoints
   └── outputs/          # Evaluation outputs
   ```

2. **GPU Support**: Most scripts will automatically use GPU if available. You can force CPU mode with `--device cpu`.

3. **Reproducibility**: Scripts use fixed random seeds where applicable for reproducible results.

4. **Error Handling**: All scripts implement early failure - they will stop immediately if something is wrong rather than trying to recover from corrupted data.

---

## 🚀 Quick Start

**Complete workflow example:**

```bash
# 1. Clean the dataset
python scripts/data_cleaner.py --src data/raw --dst data/cleaned

# 2. Train a model (from project root)
python task4train.py --config config_task4.yaml

# 3. Evaluate the model
python scripts/task4_evaluate.py \
    --checkpoint checkpoints/task4_multitask/best.pth \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --val-dir data/cleaned/val \
    --out-dir checkpoints/task4_multitask/evaluation

# 4. Run inference demo
python scripts/task4_inference_demo.py \
    --checkpoint checkpoints/task4_multitask/best.pth \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --val-dir data/cleaned/val \
    --out-dir outputs/inference_demo
```

---

## 🐛 Troubleshooting

**Import errors:**
```
ModuleNotFoundError: No module named 'src'
```
Solution: Make sure you're running scripts from the project root directory.

**CUDA out of memory:**
Solution: Reduce batch size with `--batch-size` argument.

**Missing dependencies:**
Solution: Install with `pip install pytorch torchvision pandas numpy opencv-python pillow tqdm scikit-learn albumentations timm tensorboard`.

---

For more information, see the main project [README.md](../README.md).