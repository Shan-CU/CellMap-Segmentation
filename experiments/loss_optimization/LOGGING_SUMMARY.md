# Data Pipeline Logging - Implementation Summary

## What Was Added

I've added comprehensive data pipeline logging to [train_local.py](train_local.py) to help you understand and debug the complete data flow from loading to loss calculation.

## Key Features

### 1. **Automated Debug Logging**
- Every training run now creates a detailed log file: `results/<run_name>_data_debug.log`
- Logs are created only on the main process (rank 0) to avoid duplication in DDP mode
- Structured logging with timestamps for easy analysis

### 2. **Comprehensive Data Tracking**

The system logs all critical data properties at different pipeline stages:

#### Dataloader Setup:
- Classes, batch size, iterations per epoch
- Input/output shapes (2D vs 2.5D)
- Datasplit information (train/val counts)
- All transformation details
- Dataloader configuration (workers, shuffle, etc.)

#### Training Batches:
- **First batch (detailed)**: Complete analysis of inputs, targets, model outputs, and loss
- **Periodic batches**: Summary statistics every 50 batches
- Per-class ground truth analysis (NaN counts, value ranges, binary checks, positive ratios)
- Model output statistics (logits and sigmoid activations)

#### Validation:
- First validation batch with full details
- Per-class performance metrics (TP, FP, FN, Dice)
- Model prediction distributions

### 3. **Log Analysis Tools**

#### `analyze_logs.py` - Quick Summary Tool
Extracts and displays key information:
- Input normalization check
- Ground truth properties
- Class balance analysis
- NaN pixel statistics
- Final performance metrics

Usage:
```bash
python analyze_logs.py results/*_data_debug.log
```

## Files Modified

### [train_local.py](train_local.py)
Added:
- `setup_data_logger()` - Creates structured file logger
- `log_batch_details()` - Comprehensive batch statistics logging
- `data_logger` parameter to all training functions
- Logging calls at key pipeline stages

### New Files Created

1. **[DATA_LOGGING_GUIDE.md](DATA_LOGGING_GUIDE.md)**
   - Complete guide to what's logged
   - Usage examples
   - Debugging scenarios
   - Tips and best practices

2. **[analyze_logs.py](analyze_logs.py)**
   - Log analysis and summary tool
   - Extracts key metrics
   - Color-coded status indicators
   - Supports multiple log files

3. **[test_logging.py](test_logging.py)**
   - Validates logging system works
   - Creates test log file
   - Verifies log content

## What Gets Logged

### Input Data:
✅ Shape, dtype, device
✅ Value range (min, max, mean, std)
✅ NaN/Inf detection
✅ Per-sample statistics (first batch)
✅ Normalization verification ([0, 1] check)

### Ground Truth/Targets:
✅ Shape and dtype
✅ Per-class analysis:
  - NaN pixel counts and locations
  - Valid pixel counts
  - Value ranges
  - Binary detection (0/1 values)
  - Positive pixel ratios (class balance)
  - Unique values

### Data Transformations:
✅ Spatial transforms (rotation, flip)
✅ Value transforms (normalization, NaN handling)
✅ Shape transformations (2D → 2.5D squeeze)

### Dataloader Configuration:
✅ Batch size
✅ Number of workers
✅ Shuffle status
✅ Persistent workers
✅ Prefetch factor
✅ Pin memory
✅ Dataset types

### Model & Training:
✅ Model output shapes and ranges (logits)
✅ Sigmoid activation statistics
✅ Per-class prediction distributions
✅ Loss function configuration
✅ Loss values
✅ NaN/Inf loss warnings

### Validation:
✅ Validation batch limit
✅ Per-class TP, FP, FN counts
✅ Dice scores per class and mean
✅ Prediction distributions (% > 0.5 threshold)

## How to Use

### Run with Logging (Automatic):
```bash
# Any training mode automatically creates logs
python train_local.py --mode quick_test
python train_local.py --mode loss_comparison
./run.sh compare-losses unet_2d
```

### Analyze Logs:
```bash
# Quick summary
python analyze_logs.py results/<run_name>_data_debug.log

# Compare multiple runs
python analyze_logs.py results/*_data_debug.log

# Manual inspection
less results/<run_name>_data_debug.log
grep "NaN" results/*_data_debug.log
```

## Common Debugging Scenarios

### 1. Check Data Normalization
```bash
grep "INPUT:" results/*_data_debug.log | grep "Min:"
# Should show range [0, 1], not [0, 255]
```

### 2. Identify Class Imbalance
```bash
grep "Positive ratio" results/*_data_debug.log
# Shows percentage of positive pixels per class
```

### 3. Find NaN Issues
```bash
grep "Has NaN" results/*_data_debug.log
# Shows which classes have unlabeled (NaN) regions
```

### 4. Verify Ground Truth is Binary
```bash
grep "Is binary" results/*_data_debug.log
# Should all be "True" for binary segmentation
```

### 5. Check Model Activation
```bash
grep "Sigmoid stats" results/*_data_debug.log
# Shows if model predicts reasonable probabilities
```

## Example Output

When you run training, you'll see:
```
✓ Using config_shenron (hostname: shenron)

==============================================================
Starting experiment: unet_2d_per_class_weighted_20260210_103015
Model: unet_2d (Standard 2D UNet - single slice input)
Input shape: (1, 256, 256) (1 channels)
Batch size: 8
Loss: per_class_weighted
Classes: ['nuc', 'pm', 'mito', 'er_mem', 'endo_mem']
==============================================================
```

And a log file will be created at:
```
experiments/loss_optimization/results/unet_2d_per_class_weighted_20260210_103015_data_debug.log
```

## Benefits

1. **Complete Transparency**: See exactly what data the model receives
2. **Easy Debugging**: Quickly identify normalization, NaN, or class balance issues
3. **Reproducibility**: Full record of data pipeline configuration
4. **Performance Analysis**: Understand which classes are hard to predict
5. **Validation**: Verify ground truth properties match expectations

## Testing

Verify the logging system works:
```bash
python test_logging.py
```

Should output:
```
✅ Log file created
✅ Log file has 7 lines
✅ Test passed! Cleaned up test file.
```

## Next Steps

1. **Run a training experiment** - Logging is automatic
2. **Check the log file** - Look at results/<run_name>_data_debug.log
3. **Use the analyzer** - Run `python analyze_logs.py results/*_data_debug.log`
4. **Debug any issues** - Search logs for warnings or unexpected values

## Notes

- Logging only happens on rank 0 (main process) in DDP mode
- First batch is logged in extensive detail
- Periodic batches (every 50) are logged with summary stats
- Log files are timestamped and never overwritten
- All logs are stored in `experiments/loss_optimization/results/`

For full documentation, see [DATA_LOGGING_GUIDE.md](DATA_LOGGING_GUIDE.md).
