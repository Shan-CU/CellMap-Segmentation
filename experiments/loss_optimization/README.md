# Loss Optimization Experiments

**Goal:** Improve segmentation performance on hard classes (nucleus, ER, endosome) by testing improved loss functions and training strategies.

## Problem Statement

From UNet 2D evaluation (Epoch 88):
- Overall Dice: 0.252 ± 0.352
- High recall (0.94) but low precision (0.21) → Over-predicting

### Per-Class Performance (sorted by Dice)

| Class | Dice | Status | Notes |
|-------|------|--------|-------|
| golgi_mem | 0.680 | 🟢 Good | Best performer |
| mito_ribo | 0.643 | 🟢 Good | |
| golgi_lum | 0.317 | 🟡 Moderate | |
| ecs | 0.291 | 🟡 Moderate | Extracellular space |
| ves_lum | 0.270 | 🟡 Moderate | |
| mito_lum | 0.257 | 🟡 Moderate | |
| mito_mem | 0.218 | 🟡 Moderate | |
| ves_mem | 0.193 | 🔴 Hard | Small structures |
| er_lum | 0.143 | 🔴 Hard | |
| er_mem | 0.116 | 🔴 Critical | Thin, irregular |
| pm | 0.113 | 🔴 Critical | Thin boundary |
| **nuc** | 0.111 | 🔴 Critical | Composite class! |
| endo_lum | 0.099 | 🔴 Critical | |
| endo_mem | 0.081 | 🔴 Critical | Worst performer |

### Important: `nuc` is a Composite Class!

According to the [official classes.csv](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/blob/main/src/cellmap_segmentation_challenge/utils/classes.csv):

```
nuc = ne_mem + ne_lum + np_out + np_in + hchrom + nhchrom + echrom + nechrom + nucpl + nucleo
```

This means the `nuc` label combines 10 different nuclear sub-structures. In a 2D slice, the model sees fragments of these without 3D context, making it extremely challenging.

**Root causes for poor nuc performance:**
1. BCEWithLogitsLoss doesn't handle class imbalance well
2. 2D slices lack 3D context for large composite structures
3. No focus on hard examples during training

## Solutions Implemented

### 1. Improved Loss Functions (`losses.py`)
- **DiceBCELoss**: Combo loss that directly optimizes Dice metric
- **FocalLoss**: Down-weights easy examples, focuses on hard ones
- **TverskyLoss**: Asymmetric loss to balance precision/recall
- **ComboLoss**: Weighted combination of all above

### 2. Per-Class Loss Weighting (based on actual Dice scores)
```python
# Weights inversely proportional to performance
class_loss_weights = {
    # Critical (Dice < 0.15)
    'nuc': 3.5, 'endo_mem': 3.0, 'endo_lum': 3.0,
    'pm': 2.5, 'er_mem': 2.5, 'er_lum': 2.0,
    # Hard (Dice 0.15-0.25)
    'ves_mem': 2.0, 'mito_mem': 1.8,
    # Moderate (Dice 0.25-0.40)
    'mito_lum': 1.5, 'ves_lum': 1.5, 'ecs': 1.5, 'golgi_lum': 1.5,
    # Good (Dice > 0.40)
    'mito_ribo': 1.0, 'golgi_mem': 1.0,
}
```

### 3. 2.5D Input Option
Use 5 adjacent Z-slices as input channels for 3D context without full 3D memory cost.

## Running on Shenron (4× RTX 2080 Ti)

### Initial Setup (one time)
```bash
cd /scratch/users/gest9386
git clone <repo_url> CellMap-Segmentation
cd CellMap-Segmentation/experiments/loss_optimization
chmod +x setup_shenron.sh
./setup_shenron.sh
```

### Run Experiments
```bash
# Activate environment
source ~/miniforge3/bin/activate cellmap

# Quick test (5 epochs, verify everything works)
python train_local.py --mode quick_test

# Compare loss functions (20 epochs each)
python train_local.py --mode loss_comparison

# Full training with best config (100 epochs)
python train_local.py --mode full_train --loss combo

# Monitor with TensorBoard
tensorboard --logdir=runs --port=6006
```

### Hardware Config
| Setting | Value | Rationale |
|---------|-------|-----------|
| GPUs | 4× 2080 Ti (DDP) | 11GB each |
| Batch/GPU | 12 | Safe for 256×256 UNet |
| Effective batch | 48 | 12 × 4 GPUs |
| Workers | 8/GPU | EPYC has 32 threads |
| Mixed precision | FP16 | 2× speedup |

## Files

- `setup_shenron.sh` - Environment setup script
- `losses.py` - Improved loss functions
- `train_local.py` - Main training script
- `config_shenron.py` - Hardware-optimized config
- `evaluate_losses.py` - Compare loss function results

## Expected Improvements

| Class | Baseline Dice | Target Dice |
|-------|---------------|-------------|
| nuc | 0.111 | 0.35+ |
| endo_mem | 0.081 | 0.25+ |
| er_mem | 0.116 | 0.30+ |
| Overall | 0.252 | 0.40+ |
