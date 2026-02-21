# Data Pipeline Logging Guide

## Overview
The training script now includes comprehensive data logging to help debug and understand the complete data pipeline, from loading to loss calculation.

## Log File Location
When you run `train_local.py`, a detailed debug log file is created:
```
experiments/loss_optimization/results/<run_name>_data_debug.log
```

## What Gets Logged

### 1. **Dataloader Configuration**
At startup, the following is logged:
- Classes being trained
- Batch size
- Iterations per epoch
- Input shape (2D vs 2.5D)
- Datasplit information:
  - Total crops
  - Train/val split counts
  - Classes present
- Transformations applied:
  - Spatial transforms (rotation, flips, etc.)
  - Value transforms (normalization, NaN handling)
- Array info (input/target shapes and scales)
- Dataloader settings:
  - Number of workers
  - Persistent workers
  - Prefetch factor
  - Shuffle status
  - Pin memory

### 2. **First Training Batch (Detailed)**
For the first batch of epoch 1, extensive logging includes:

#### Input Data:
- Shape, dtype, device
- Min, max, mean, std
- NaN/Inf detection
- Per-sample statistics (first 3 samples)

#### Ground Truth/Target:
- Shape, dtype, device
- Per-class analysis:
  - NaN count and location
  - Valid pixel count
  - Value range (min, max, mean)
  - Unique values
  - Binary detection (0/1 values)
  - Positive ratio (pixels > 0.5)

#### After Preprocessing:
- Shape transformations (e.g., squeeze for 2.5D)

#### Model Output (Logits):
- Shape, dtype
- Value range (min, max, mean, std)
- Per-class output statistics

#### Loss Calculation:
- Loss function type
- Loss value
- Criterion configuration

### 3. **Periodic Training Batches**
Every 50 batches during training, less detailed stats are logged:
- Input statistics
- Target statistics per class
- Batch index

### 4. **First Validation Batch (Detailed)**
Similar comprehensive logging as the first training batch, including:
- Input/target properties
- Model output (logits and sigmoid)
- Per-class sigmoid statistics:
  - Min, max, mean values
  - Percentage of pixels > 0.5

### 5. **Validation Results**
After each validation run:
- Validation loss
- Mean Dice score
- Per-class metrics:
  - Dice score
  - True Positives (TP)
  - False Positives (FP)
  - False Negatives (FN)

### 6. **Error Tracking**
- NaN/Inf loss warnings
- Skipped batches

## Key Data Properties Logged

### Data Loading:
- ✅ Batch size
- ✅ Shuffle status (via dataloader config)
- ✅ Number of workers
- ✅ Dataset types

### Data Transformations:
- ✅ Spatial transforms applied
- ✅ Value normalization ([0, 1] range)
- ✅ NaN handling (NaNtoNum)
- ✅ Shape transformations (2D vs 2.5D)

### Ground Truth Properties:
- ✅ Original shape vs processed shape
- ✅ Data type (float32/uint8)
- ✅ Value range and distribution
- ✅ NaN locations and counts
- ✅ Binary vs continuous values
- ✅ Class balance (positive ratios)

### Validation Data:
- ✅ Validation size (batch limit: 50)
- ✅ Random sampling status
- ✅ Same shape as training data
- ✅ Ground truth size comparison

### Model & Loss:
- ✅ Model output range (logits)
- ✅ Sigmoid activation stats
- ✅ Loss function configuration
- ✅ Loss values per batch

## How to Use

### Run Training with Logging:
```bash
# Quick test
python train_local.py --mode quick_test

# Full training
python train_local.py --mode loss_comparison

# Single loss experiment
./run.sh compare-losses unet_2d
```

### View the Log:
```bash
# Find your run's log file
ls -lh experiments/loss_optimization/results/*_data_debug.log

# View the log
less experiments/loss_optimization/results/<run_name>_data_debug.log

# Analyze the log with summary tool
python analyze_logs.py results/<run_name>_data_debug.log

# Analyze all logs
python analyze_logs.py results/*_data_debug.log

# Search for specific info
grep "FIRST TRAINING BATCH" results/*_data_debug.log
grep "Ground truth" results/*_data_debug.log
grep "NaN" results/*_data_debug.log
```

### Log Analyzer Tool

The `analyze_logs.py` script provides a quick summary of your data pipeline:

```bash
python analyze_logs.py results/unet_2d_per_class_weighted_20260210_103015_data_debug.log
```

Output includes:
- Configuration (classes, batch size, input shape)
- Input data range and normalization check
- Ground truth properties (binary check)
- NaN pixel counts and locations
- Class balance (positive ratios)
- Final validation Dice scores

Example output:
```
================================================================================
DATA PIPELINE ANALYSIS: unet_2d_per_class_weighted_20260210_103015
================================================================================

📊 CONFIGURATION:
  Classes: nuc, pm, mito, er_mem, endo_mem
  Batch size: 8
  Input shape: (1, 256, 256)
  Train batches logged: 3
  Val batches logged: 2

📥 INPUT DATA:
  Range: [0.000000, 1.000000]
  ✅ Input properly normalized

🎯 GROUND TRUTH:
  ✅ All classes are binary (0/1)

➕ CLASS BALANCE (positive pixel ratios):
  🟢 mito: 0.1523 (15.23%)
  🟡 nuc: 0.0892 (8.92%)
  🔴 pm: 0.0234 (2.34%)
  🔴 er_mem: 0.0156 (1.56%)
  🔴 endo_mem: 0.0089 (0.89%)

🎲 FINAL VALIDATION DICE:
  🟢 mito: 0.6234
  🟡 nuc: 0.4521
  🔴 pm: 0.2134
  🔴 er_mem: 0.1892
  🔴 endo_mem: 0.1456
  Mean: 0.3247
```

## Common Debugging Scenarios

### 1. Check if data is normalized correctly:
```bash
grep "INPUT:" results/*_data_debug.log | grep "Min:"
# Should see values in [0, 1] range
```

### 2. Check ground truth values:
```bash
grep "Is binary" results/*_data_debug.log
# Should see "True" for all classes
```

### 3. Check for NaN issues:
```bash
grep "Has NaN" results/*_data_debug.log
# Should see which classes have NaN pixels
```

### 4. Check class balance:
```bash
grep "Positive ratio" results/*_data_debug.log
# Shows how many pixels are positive for each class
```

### 5. Check if validation data matches training:
```bash
grep "VALIDATION" results/*_data_debug.log -A 50
# Compare shapes and value ranges with training data
```

### 6. Check model output activation:
```bash
grep "Sigmoid stats" results/*_data_debug.log
# Shows if model is predicting reasonable probabilities
```

## Example Log Snippet

```
2026-02-10 10:30:15 [INFO] ================================================================================
2026-02-10 10:30:15 [INFO] DATALOADER CREATION
2026-02-10 10:30:15 [INFO] ================================================================================
2026-02-10 10:30:15 [INFO] Classes: ['nuc', 'pm', 'mito', 'er_mem', 'endo_mem']
2026-02-10 10:30:15 [INFO] Batch size: 8
2026-02-10 10:30:15 [INFO] Iterations per epoch: 100
2026-02-10 10:30:15 [INFO] Input shape: (1, 256, 256)

2026-02-10 10:30:20 [INFO] ################################################################################
2026-02-10 10:30:20 [INFO] FIRST TRAINING BATCH (EPOCH 1)
2026-02-10 10:30:20 [INFO] ################################################################################
2026-02-10 10:30:20 [INFO] 
2026-02-10 10:30:20 [INFO] INPUT:
2026-02-10 10:30:20 [INFO]   Shape: torch.Size([8, 1, 1, 256, 256])
2026-02-10 10:30:20 [INFO]   Dtype: torch.float32
2026-02-10 10:30:20 [INFO]   Min: 0.000000
2026-02-10 10:30:20 [INFO]   Max: 1.000000
2026-02-10 10:30:20 [INFO]   Mean: 0.523456

2026-02-10 10:30:20 [INFO] TARGET/GROUND TRUTH:
2026-02-10 10:30:20 [INFO]   nuc:
2026-02-10 10:30:20 [INFO]     Has NaN: False
2026-02-10 10:30:20 [INFO]     Is binary (0/1): True
2026-02-10 10:30:20 [INFO]     Positive ratio (>0.5): 0.1234
```

## Tips

1. **First batch is crucial** - Most data pipeline issues show up in the first batch
2. **Check NaN handling** - NaN pixels are expected in targets (unlabeled regions)
3. **Verify normalization** - Input should be [0, 1], not [0, 255]
4. **Class balance** - Very low positive ratios indicate rare/hard classes
5. **Model outputs** - Sigmoid values should span [0, 1] reasonably after a few epochs

## Additional Files Created

Besides the debug log, each run creates:
- `<run_name>_results.json` - Training history and metrics
- `<run_name>_best.pth` - Best model checkpoint
- TensorBoard logs in `tensorboard/<run_name>/`
