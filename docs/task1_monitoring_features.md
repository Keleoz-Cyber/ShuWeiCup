# Task 1 Training - Real-time Monitoring Features

## Overview

The `task1train.py` script has been enhanced with comprehensive real-time monitoring capabilities, similar to `task2train.py`. These features enable you to track training progress, visualize metrics, and analyze model performance in real-time.

## New Features Added

### 1. TensorBoard Integration

Real-time logging of training metrics to TensorBoard for interactive visualization.

**Logged Metrics:**
- `train/loss` - Training loss per epoch
- `train/acc` - Training accuracy per epoch
- `val/loss` - Validation loss per epoch
- `val/acc` - Validation accuracy per epoch
- `val/tail_acc` - Accuracy on tail (rare) classes
- `val/macro_f1` - Macro-averaged F1 score
- `lr/learning_rate` - Learning rate schedule

**How to Use:**
```bash
# Start TensorBoard after training begins
tensorboard --logdir checkpoints/your_experiment/logs

# Then open http://localhost:6006 in your browser
```

### 2. Training History CSV Export

All training metrics are automatically saved to a CSV file for further analysis.

**File Location:** `{save_dir}/training_history.csv`

**Columns:**
- `epoch` - Epoch number
- `train_loss` - Training loss
- `train_acc` - Training accuracy (%)
- `val_loss` - Validation loss
- `val_acc` - Validation accuracy (%)
- `tail_acc` - Tail class accuracy (%)
- `macro_f1` - Macro F1 score
- `lr` - Learning rate

**Example Usage:**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training history
df = pd.read_csv('checkpoints/task1_exp/training_history.csv')

# Plot custom metrics
plt.plot(df['epoch'], df['val_acc'])
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.show()
```

### 3. Training Curves Visualization

Automatically generated PNG plots showing training progress across all epochs.

**File Location:** `{save_dir}/training_curves.png`

**Four Subplots:**
1. **Loss Curves** - Train vs Validation loss
2. **Accuracy Curves** - Train, Validation, and Tail class accuracy
3. **Macro F1 Score** - Overall F1 performance across all classes
4. **Learning Rate Schedule** - LR changes over epochs (log scale)

**Features:**
- High-resolution (120 DPI) for presentations
- Color-coded for easy interpretation
- Grid lines for precise reading
- Automatic legend positioning

### 4. Confusion Matrix Generation

Confusion matrix is generated when a new best model is saved.

**Files Generated:**
- `{save_dir}/confusion_matrix_best.csv` - Raw confusion matrix data
- `{save_dir}/confusion_matrix_best.png` - Visual heatmap (10x8 inches, 120 DPI)

**Confusion Matrix Features:**
- 61x61 matrix for all disease classes
- Blue color scheme (darker = more predictions)
- Colorbar for value reference
- Saved as both CSV (for analysis) and PNG (for visualization)

**Analysis Tips:**
```python
import pandas as pd
import numpy as np

# Load confusion matrix
cm = pd.read_csv('checkpoints/task1_exp/confusion_matrix_best.csv')

# Find most confused class pairs
cm_array = cm.values
np.fill_diagonal(cm_array, 0)  # Ignore correct predictions
max_confusion_idx = np.unravel_index(cm_array.argmax(), cm_array.shape)
print(f"Most confused: class {max_confusion_idx[0]} -> {max_confusion_idx[1]}")
```

## When Are These Features Available?

The monitoring features are **automatically enabled** when using the custom training loop with any of these options:
- `--balance-sampler` - Weighted sampling for class imbalance
- `--mixup-alpha > 0` - Mixup augmentation
- `--cutmix-alpha > 0` - CutMix augmentation
- `--use-ema` - Exponential moving average

## Example Training Command

```bash
python task1train.py \
    --train-dir data/cleaned/train \
    --val-dir data/cleaned/val \
    --train-meta data/cleaned/metadata/train_metadata.csv \
    --val-meta data/cleaned/metadata/val_metadata.csv \
    --backbone resnet50 \
    --epochs 50 \
    --batch-size 64 \
    --lr 5e-4 \
    --use-ema \
    --mixup-alpha 0.2 \
    --save-dir checkpoints/task1_monitored
```

## Output Directory Structure

After training with monitoring enabled:

```
checkpoints/task1_monitored/
├── logs/                              # TensorBoard logs
│   └── events.out.tfevents.*         # TensorBoard event files
├── training_history.csv               # All metrics in CSV format
├── training_curves.png                # 4-panel training visualization
├── confusion_matrix_best.csv          # Raw confusion matrix
├── confusion_matrix_best.png          # Confusion matrix heatmap
└── best_custom.pth                    # Best model checkpoint
```

## Monitoring During Training

While training is running, you can:

1. **Watch live TensorBoard graphs:**
   ```bash
   tensorboard --logdir checkpoints/task1_monitored/logs
   ```

2. **Check console output** for epoch-by-epoch metrics:
   ```
   Epoch 10/50 | Train Loss 0.4532 | Train Acc 87.34% | LR 0.000450
     [Val] Epoch 10 | Loss 0.5123 | Acc 85.21% | TailAcc 78.45% | MacroF1 0.823
     ✅ New best (custom loop) val acc: 85.21% | TailAcc 78.45% | MacroF1 0.823
     [CM] Saved confusion matrix CSV
     [CM] Saved confusion matrix PNG
   ```

3. **Tail for CSV updates** (requires pandas installed):
   ```bash
   tail -f checkpoints/task1_monitored/training_history.csv
   ```

## Key Differences from task2train.py

While inspired by `task2train.py`, the monitoring in `task1train.py` includes:

✅ **Additional metrics:**
- Tail class accuracy (bottom 25% frequency classes)
- Per-class F1 scores for macro averaging

✅ **Integrated with custom loop:**
- Works seamlessly with EMA, Mixup, CutMix
- Supports multi-phase training schedules

✅ **Enhanced plotting:**
- 2x2 grid layout instead of 1x3
- Separate F1 and accuracy plots
- Tail accuracy tracking

## Tips for Best Results

1. **Monitor TensorBoard in real-time** to catch overfitting early
2. **Compare multiple runs** by using different `--save-dir` paths
3. **Check confusion matrix** to identify problematic class pairs
4. **Use training history CSV** for custom analysis and reporting
5. **Save training curves PNG** for presentations and documentation

## Dependencies

All required packages are standard PyTorch ML libraries:
- `torch` - PyTorch framework
- `tensorboard` - Visualization backend
- `matplotlib` - Plotting library
- `pandas` (optional) - For CSV export (degrades gracefully if missing)

## Troubleshooting

**Q: TensorBoard not showing metrics?**
- Ensure training has started and at least one epoch completed
- Check that the logs directory exists: `ls checkpoints/your_exp/logs/`
- Try refreshing the browser or restarting TensorBoard

**Q: CSV file not created?**
- Pandas might not be installed: `pip install pandas`
- The script will warn but continue without CSV export

**Q: Plots look different from task2train.py?**
- This is expected - task1 has 4 plots vs task2's 3 plots
- Task1 includes tail accuracy and separate F1 plot

**Q: Confusion matrix is too large to read?**
- Open the PNG file in an image viewer with zoom capabilities
- Use the CSV file for programmatic analysis of specific classes

## Summary

The enhanced monitoring features provide comprehensive insights into your Task 1 training runs, enabling:
- ✅ Real-time progress tracking
- ✅ Post-training analysis
- ✅ Easy comparison between experiments
- ✅ Professional-quality visualizations
- ✅ Detailed per-class performance analysis

All features are automatically enabled and require no additional configuration beyond the standard training flags!