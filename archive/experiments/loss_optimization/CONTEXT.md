# Loss Optimization Experiment Context

> **Purpose**: This document provides complete context for AI assistants to continue work on this experiment without losing any prior conversation history.

## Project Overview

**Project**: CellMap Segmentation Challenge (http://cellmapchallenge.janelia.org/)  
**Task**: Multi-class organelle segmentation from FIB-SEM microscopy images  
**Repository**: CellMap-Segmentation  
**Current Phase**: Loss optimization and 2D vs 2.5D model comparison

## Problem Statement

### Observed Issue
The UNet 2D baseline model (epoch 88) shows **poor performance on nucleus and membrane classes**:

| Class | Dice Score | Precision | Recall | Issue |
|-------|------------|-----------|--------|-------|
| nuc | 0.111 | 0.21 | 0.94 | Critical - high recall, low precision = over-prediction |
| endo_mem | 0.081 | - | - | Worst performing |
| endo_lum | 0.099 | - | - | Very poor |
| pm | 0.113 | - | - | Poor |
| er_mem | 0.116 | - | - | Poor |
| er_lum | 0.143 | - | - | Below average |
| **Overall** | **0.252** | - | - | Baseline |

### Root Cause Analysis

1. **Composite Class Problem**: `nuc` (class 37) is a composite of 10 sub-structures:
   - `"20,21,22,23,24,25,26,27,28,29"` = ne_mem, ne_lum, np_out, np_in, hchrom, nhchrom, echrom, nechrom, nucpl, nucleo
   - These sub-structures have varying characteristics that the model must learn as one

2. **2D Context Limitation**: Nucleus is a large 3D structure that appears inconsistently in 2D slices:
   - Edge slices show only membrane
   - Center slices show chromatin patterns
   - 2D model lacks z-axis context to understand these are the same structure

3. **Class Imbalance**: Using `BCEWithLogitsLoss` with simple pos_weight doesn't account for:
   - Per-class difficulty variation
   - Boundary vs interior pixel importance
   - The Dice-based evaluation metric

## Solutions Implemented

### 1. Improved Loss Functions (`losses.py`)

```python
# Available loss functions:
- DiceLoss          # Direct optimization for Dice metric
- DiceBCELoss       # Dice + BCE hybrid
- FocalLoss         # Down-weights easy examples
- TverskyLoss       # Adjustable precision/recall balance
- ComboLoss         # Dice + Focal combination
- PerClassComboLoss # Per-class weighted Dice+BCE (RECOMMENDED)
```

### 2. Corrected Class Weights

Weights are **inversely proportional to actual Dice scores** (harder classes get higher weights):

```python
CLASS_WEIGHTS = {
    # Hardest classes (Dice < 0.15) - Boost 2.5-3.5x
    'endo_mem': 3.0,   # 0.081 Dice - worst
    'endo_lum': 3.0,   # 0.099 Dice
    'nuc': 3.5,        # 0.111 Dice - critical target
    'pm': 2.5,         # 0.113 Dice
    'er_mem': 2.5,     # 0.116 Dice
    
    # Medium difficulty (0.15 < Dice < 0.25) - Boost 1.5-2x
    'er_lum': 2.0,     # 0.143 Dice
    'ves_mem': 2.0,    # 0.193 Dice
    'mito_mem': 1.8,   # 0.218 Dice
    
    # Easier classes (Dice > 0.25) - Normal weight
    'mito_lum': 1.5,   # 0.257 Dice
    'ves_lum': 1.5,    # 0.270 Dice
    'ecs': 1.5,        # 0.291 Dice
    'golgi_lum': 1.5,  # 0.317 Dice
    
    # Well-performing (Dice > 0.5) - No boost
    'mito_ribo': 1.0,  # 0.643 Dice
    'golgi_mem': 1.0,  # 0.680 Dice - best
}
```

### 3. 2D vs 2.5D Model Comparison

**Hypothesis**: 2.5D model (5 adjacent z-slices as input channels) will improve nucleus segmentation by providing z-axis context.

```python
MODEL_CONFIGS = {
    'unet_2d': {
        'input_channels': 1,
        'input_shape': (1, 256, 256),
        'batch_size': 12,
        'description': 'Standard 2D UNet - single slice'
    },
    'unet_25d': {
        'input_channels': 5,
        'input_shape': (5, 256, 256),
        'batch_size': 8,  # Reduced for memory
        'description': '2.5D UNet - 5 adjacent z-slices'
    }
}
```

## Hardware Configuration

### Target: Shenron Workstation

| Component | Specification |
|-----------|---------------|
| GPUs | 4x NVIDIA RTX 2080 Ti (11GB each) |
| CPU | AMD EPYC 7302 (32 threads) |
| RAM | 62 GB |
| CUDA | 13.1 |
| Driver | 590.48 |

### Data Location
- **Data**: `/volatile/cellmap/data/` (22 datasets downloaded)
- **Workspace**: `/scratch/users/gest9386/CellMap-Segmentation`

### Training Configuration
```python
# Optimized for 4x 2080 Ti
BATCH_SIZE = 12  # Per GPU (2D) or 8 (2.5D)
NUM_WORKERS = 6  # Per dataloader
MIXED_PRECISION = True  # AMP enabled
DDP = True  # 4-GPU distributed training
EPOCHS = 50  # Quick comparison runs
ITERATIONS_PER_EPOCH = 200
```

## File Structure

```
experiments/loss_optimization/
├── CONTEXT.md           # This file - AI context preservation
├── README.md            # User documentation
├── losses.py            # Improved loss function implementations
├── config_shenron.py    # Hardware-optimized configuration
├── train_local.py       # Main training script with DDP
├── test_setup.py        # Environment verification
├── run.sh               # Convenience launcher script
├── setup_shenron.sh     # Environment setup script
└── visualize_results.py # Results plotting utilities
```

## How to Run

### Setup on Shenron
```bash
cd /scratch/users/gest9386
git clone <repo> CellMap-Segmentation  # or git pull
cd CellMap-Segmentation/experiments/loss_optimization
chmod +x *.sh
./setup_shenron.sh
```

### Available Commands
```bash
# Test environment
./run.sh test

# Compare 2D vs 2.5D (PRIORITY EXPERIMENT)
./run.sh compare-models per_class_weighted

# Run specific loss experiments
./run.sh quick per_class_weighted    # Quick 10-epoch test
./run.sh single dice_bce unet_2d     # Single loss experiment
./run.sh compare                      # Compare all losses
./run.sh full                         # Full training run
```

### Expected Output: Model Comparison
```
============================================================
                    MODEL COMPARISON RESULTS
============================================================
Comparing 2D vs 2.5D UNet with per_class_weighted loss

Per-Class Dice Scores:
Class        2D      2.5D    Improvement
-----------------------------------------
nuc          0.111   0.XXX   +XX.X%     <- KEY METRIC
endo_mem     0.081   0.XXX   +XX.X%
pm           0.113   0.XXX   +XX.X%
...
```

## Key Code References

### Loss Functions Location
- **File**: `src/cellmap_segmentation_challenge/losses/loss.py`
- **Current**: `BCEWithLogitsLoss` with `pos_weight`
- **Issue**: Doesn't optimize for Dice metric directly

### Model Architecture
- **File**: `src/cellmap_segmentation_challenge/models/unet2d.py`
- **Class**: `UNet_2D(in_channels, out_channels)`
- **Note**: `in_channels=1` for 2D, `in_channels=5` for 2.5D

### Dataloader
- **File**: `src/cellmap_segmentation_challenge/dataloader/dataloader.py`
- **Class**: `CellMapDataLoader`
- **Key param**: `input_array_info={'shape': (1, 256, 256)}` for 2D or `(5, 256, 256)` for 2.5D

### Classes Definition
- **URL**: https://raw.githubusercontent.com/janelia-cellmap/cellmap-segmentation-challenge/main/src/cellmap_segmentation_challenge/utils/datasplit_csv_info/classes.csv
- **Key**: `nuc` = composite class 37 = `"20,21,22,23,24,25,26,27,28,29"`

## Evaluation Baseline (UNet 2D Epoch 88)

```
Overall Metrics:
  Dice: 0.252
  Precision: 0.472
  Recall: 0.499

Per-Class Performance (sorted by Dice):
  golgi_mem: 0.680 (best)
  mito_ribo: 0.643
  golgi_lum: 0.317
  ecs: 0.291
  ves_lum: 0.270
  mito_lum: 0.257
  mito_mem: 0.218
  ves_mem: 0.193
  er_lum: 0.143
  er_mem: 0.116
  pm: 0.113
  nuc: 0.111 (critical)
  endo_lum: 0.099
  endo_mem: 0.081 (worst)
```

## Next Steps

### Immediate (On Shenron)
1. Clone/pull repository
2. Run `./setup_shenron.sh`
3. Verify with `./run.sh test`
4. Execute `./run.sh compare-models per_class_weighted`

### Analysis Goals
1. **Quantify 2.5D improvement**: Especially for `nuc` class
2. **Identify optimal loss function**: Compare Dice, Focal, per-class weighted
3. **Validate class weights**: Confirm harder classes improve

### If 2.5D Shows Improvement
1. Increase to full training (200+ epochs)
2. Consider 3D model for even better context
3. Submit to cluster for production training

### If No Improvement
1. Investigate nucleus sub-class segmentation
2. Consider attention mechanisms for large structures
3. Explore boundary-aware losses

## Technical Notes

### Mixed Precision (AMP)
- Enabled by default for memory efficiency
- Uses `torch.cuda.amp.autocast()` and `GradScaler`
- Reduces memory ~40%, enabling larger batches

### Distributed Data Parallel (DDP)
- Uses all 4 GPUs automatically
- `torchrun --nproc_per_node=4` for multi-GPU
- Gradient synchronization handled by PyTorch

### Memory Estimates
- **2D (batch=12)**: ~6-7 GB per GPU
- **2.5D (batch=8)**: ~7-8 GB per GPU
- Both fit comfortably in 11GB 2080 Ti

## Contact & References

- **Challenge**: http://cellmapchallenge.janelia.org/
- **Leaderboard**: http://cellmapchallenge.janelia.org/leaderboard
- **GitHub**: https://github.com/janelia-cellmap/cellmap-segmentation-challenge

---

*Last Updated: February 5, 2026*  
*Context Version: 1.0*
