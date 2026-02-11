# Class-Weighting Experiment

> **Fixed loss:** Per-class Tversky (α=0.6, β=0.4)  
> **Variable:** Class weighting strategy  
> **Model:** UNet 2D, input (1, 256, 256), batch 24  
> **Classes:** `nuc`, `mito_mem`, `er_mem`, `pm`, `golgi_mem`

## Motivation

The Shenron loss-comparison experiments identified **per-class Tversky** with
α=0.6, β=0.4 as the best-performing loss function. This experiment keeps the
loss fixed and systematically varies the class-weighting strategy to find the
optimal way to handle the severe class imbalance in CellMap data.

## Strategies Tested (15 configs)

### Static Weights (6)

| Config | Description |
|--------|-------------|
| `weight_uniform` | All classes weighted equally (baseline) |
| `weight_manual` | Hand-tuned weights from domain knowledge |
| `weight_inv_freq` | Inverse frequency: `1/freq(c)` |
| `weight_sqrt_inv` | Square-root inverse: `1/√freq(c)` |
| `weight_log_inv` | Log inverse: `1/log(1+freq(c))` |
| `weight_effective_num` | Effective number: `(1-β^n)/(1-β)`, β=0.999 |

### Class-Balanced (3)

| Config | β |
|--------|---|
| `cb_beta_0.99` | 0.99 (mild) |
| `cb_beta_0.999` | 0.999 (moderate) |
| `cb_beta_0.9999` | 0.9999 (strong) |

### Balanced Softmax (3)

| Config | τ (temperature) |
|--------|-----------------|
| `balanced_softmax_tau_0.5` | 0.5 (strong adjustment) |
| `balanced_softmax_tau_1.0` | 1.0 (moderate) |
| `balanced_softmax_tau_2.0` | 2.0 (mild) |

### Seesaw (3)

| Config | Mitigation (p) | Compensation (q) |
|--------|-----------------|-------------------|
| `seesaw_default` | 0.8 | 2.0 |
| `seesaw_strong_mitigate` | 1.2 | 2.0 |
| `seesaw_strong_compensate` | 0.8 | 4.0 |

## Quick Start

```bash
# Step 0 — Compute real class frequencies from training data
./run.sh compute-freqs

# Step 1 — Quick smoke test (~5 min)
./run.sh quick

# Step 2 — Full comparison (all 15 configs)
./run.sh compare

# Step 3 — Analyse results
./run.sh analyze

# Optional — TensorBoard
./run.sh tensorboard
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All 15 loss configs, training hyper-params |
| `losses_class_weighting.py` | CB, Balanced Softmax, Seesaw on Tversky base |
| `train.py` | DDP-compatible training loop |
| `compute_class_frequencies.py` | Scan data → `class_frequencies.json` |
| `analyze_results.py` | Plots and summary table |
| `run.sh` | Launcher script |

## Workflow

1. **Compute frequencies** — `compute_class_frequencies.py` scans 100 batches
   and saves per-class voxel counts to `class_frequencies.json`. Copy the
   printed weight dicts into `config.py` to replace the placeholder values.

2. **Train** — Each config trains for **60 epochs × 100 iterations** with
   AMP, AdamW (lr=1e-4), and OneCycleLR.

3. **Analyse** — `analyze_results.py` reads all `*_results.json` files and
   generates:
   - Per-class Dice grouped bar chart
   - Training/validation loss curves
   - Strategy-family comparison plot
   - Console ranking table

## Hardware

- **Rocinante**: 2× RTX 3090 (25 GB), AMD Ryzen 9 5950X, 252 GB RAM
- DDP across both GPUs or single-GPU mode
