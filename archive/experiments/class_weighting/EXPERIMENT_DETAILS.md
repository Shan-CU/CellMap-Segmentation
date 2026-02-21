# Class Weighting Experiment — Full Training Details

> **Status (Feb 11 2026):** 12/15 configs complete. Final 3 (Seesaw family) in progress.

## 1. Experiment Goal

Fix the loss function to **per-class Tversky (α=0.6, β=0.4)** — the best-performing loss from the [loss_optimization](../loss_optimization/) experiment on Shenron — and systematically compare **15 class weighting strategies** to determine which approach best handles the class imbalance in CellMap EM segmentation data.

---

## 2. Hardware

| Component | Spec |
|-----------|------|
| **Machine** | Rocinante |
| **GPUs** | 2× NVIDIA RTX 3090 (24.6 GB VRAM each) |
| **CPU** | AMD Ryzen 9 5950X, 16 cores / 32 threads (96 reported) |
| **RAM** | 252 GB DDR4 |
| **Storage** | Local SSD (zarr data at `/home/spuser/ws/CellMap-Segmentation/data`) |

---

## 3. Model & Data

| Parameter | Value |
|-----------|-------|
| **Model** | UNet 2D (`cellmap_segmentation_challenge.models.UNet_2D`) |
| **Input** | Single Z-slice, shape `(1, 256, 256)`, scale `(8, 8, 8)` nm |
| **Classes** | 5 quick-test: `nuc`, `mito_mem`, `er_mem`, `pm`, `golgi_mem` |
| **Data split** | 85% train / 15% validation (auto-generated `datasplit.csv`) |
| **Augmentations** | Mirror (x/y, p=0.5), transpose (x↔y), rotation (±180°) |
| **Normalization** | Float32, clip to [0, 1], NaN→0 |

---

## 4. Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Epochs** | 60 | Per config |
| **Iterations/epoch** | 100 | Training batches per epoch |
| **Batch size** | 24 | Per process |
| **Optimizer** | AdamW | lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999) |
| **Scheduler** | OneCycleLR | max_lr=1e-4, pct_start=0.05, cosine anneal, **stepped per batch** (6000 total steps) |
| **AMP** | Enabled | `torch.amp.autocast('cuda')` + `GradScaler` |
| **Grad clipping** | Max norm 1.0 | |
| **Validation** | Every epoch | 20 batches, per-class Dice computed |
| **Checkpointing** | Best mean Dice | Saved to `checkpoints/` |
| **torch.compile** | `mode='reduce-overhead'` | Ampere+ optimization (~8 GB VRAM overhead) |
| **cudnn.benchmark** | True | Fixed input size → free speedup |
| **Thread limits** | OMP/MKL/OPENBLAS=4, torch.num_threads=4 | Prevent CPU oversubscription |

### Parallelism Strategy

Instead of DDP (which adds communication overhead for a small model), we run **4 independent single-GPU processes in parallel** — 2 per GPU. This gives near-linear speedup with no gradient sync cost.

```
Round 1:  GPU 0 → config_A, config_B  |  GPU 1 → config_C, config_D
Round 2:  GPU 0 → config_E, config_F  |  GPU 1 → config_G, config_H
...
```

- 15 configs ÷ 4 per round = 4 rounds (last round has 3 configs)
- Each process uses ~8 GB VRAM (model ~356 MB + torch.compile cache)
- 2 DataLoader workers per process (4 procs × 2 = 8 total)
- Resume support: completed configs are auto-skipped on restart

---

## 5. Base Loss Function

All 15 configs use the same underlying loss:

### Per-Class Tversky Loss (α=0.6, β=0.4)

```
Tversky(c) = (TP_c + ε) / (TP_c + α·FP_c + β·FN_c + ε)
Loss(c) = 1 − Tversky(c)
```

- **α=0.6** → higher FP penalty → precision bias
- **β=0.4** → lower FN penalty
- **ε=1e-6** smoothing
- NaN masking: targets with NaN are excluded from TP/FP/FN computation
- Per-class losses are computed independently, then combined via the weighting strategy

This was selected as the best base loss from the Shenron loss comparison experiment (beating Dice+BCE, focal, generalized Dice, etc.).

---

## 6. Class Frequency Analysis

Computed by `compute_class_frequencies.py` scanning 50 training batches (29.6M total voxels):

| Class | Positive Voxels | Frequency | Rarity |
|-------|---------------:|----------:|--------|
| `nuc` | 3,537,231 | 11.93% | Most common |
| `mito_mem` | 1,781,370 | 6.01% | |
| `er_mem` | 1,662,007 | 5.60% | |
| `pm` | 653,198 | 2.20% | |
| `golgi_mem` | 536,370 | 1.81% | Rarest (~6.6× less than nuc) |

---

## 7. The 15 Weighting Strategies

### A. Static Weights (6 configs)

These multiply each class's Tversky loss by a fixed weight, then average:

```
Loss = mean(w_c · TverskyLoss_c)
```

| Config | nuc | mito_mem | er_mem | pm | golgi_mem | How weights are derived |
|--------|-----|----------|--------|----|-----------|------------------------|
| `weight_uniform` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | All equal (baseline) |
| `weight_manual` | 1.800 | 1.300 | 1.600 | 1.600 | 1.100 | Hand-tuned based on domain knowledge |
| `weight_inv_freq` | 0.152 | 0.301 | 0.323 | 0.821 | 1.000 | w = 1/freq, normalized so max=1 |
| `weight_sqrt_inv` | 0.389 | 0.549 | 0.568 | 0.906 | 1.000 | w = 1/√freq, normalized |
| `weight_log_inv` | 0.177 | 0.322 | 0.343 | 0.827 | 1.000 | w = 1/log(1+freq), normalized |
| `weight_effective_num` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | (1−β^n)/(1−β), β=0.999 — collapsed to uniform due to large n |

Weights for `inv_freq`, `sqrt_inv`, `log_inv`, and `effective_num` are auto-loaded from `class_frequencies.json` at import time.

### B. Class-Balanced (CB) Loss (3 configs)

From Cui et al., CVPR 2019. Replaces raw counts with "effective number":

```
E(n) = (1 − β_cb^n) / (1 − β_cb)
w_c = 1 / E(n_c),  normalized so Σw = C
```

Weights are computed **online** — the loss accumulates voxel counts during training and recomputes weights every 50 batches.

| Config | β_cb | Damping | Effect |
|--------|------|---------|--------|
| `cb_beta_0.99` | 0.99 | Mild | Closest to inverse-frequency |
| `cb_beta_0.999` | 0.999 | Medium | Moderate smoothing |
| `cb_beta_0.9999` | 0.9999 | Strong | Closest to uniform |

### C. Balanced Softmax / Logit Adjustment (3 configs)

From Ren et al., NeurIPS 2020 / Menon et al., ICLR 2021. Adjusts logits before sigmoid:

```
adjusted_logit_c = logit_c − τ · (log(n_c) − mean(log(n)))
```

Rare classes get a positive shift → sigmoid output biased upward → lower threshold for detection. Applied **inside** the Tversky loss (not a separate BCE term).

Online estimation: accumulates counts, recomputes adjustments every 50 batches.

| Config | τ | Adjustment strength |
|--------|---|---------------------|
| `balanced_softmax_tau_0.5` | 0.5 | Mild |
| `balanced_softmax_tau_1.0` | 1.0 | Standard (theory-optimal) |
| `balanced_softmax_tau_2.0` | 2.0 | Strong |

### D. Seesaw Loss (3 configs)

From Wang et al., CVPR 2021. Two dynamic factors:

1. **Mitigation** — Reduces loss for frequent classes to prevent their negative gradients from erasing the learning signal of rare classes:
   ```
   mitigation_c = (median_count / count_c)^p
   ```

2. **Compensation** — Increases penalty when a class has too many false positives:
   ```
   compensation_c = 1 + (FP_c / (FP_c + FN_c))^q
   ```

```
Loss = mean(mitigation_c · compensation_c · TverskyLoss_c)
```

Both factors update continuously during training.

| Config | p (mitigation) | q (compensation) | Notes |
|--------|---------------|-------------------|-------|
| `seesaw_default` | 0.8 | 2.0 | Paper defaults |
| `seesaw_strong_mitigate` | 1.0 | 2.0 | Stronger gradient dampening |
| `seesaw_strong_compensate` | 0.8 | 3.0 | Stronger FP penalty |

---

## 8. Validation & Metrics

- **Metric:** Per-class Dice coefficient (from accumulated TP/FP/FN over 20 val batches)
- **Threshold:** 0.5 on sigmoid outputs
- **Ranking:** By mean Dice across all 5 classes at the best epoch
- **Per-class tracking:** Dice for each class logged every epoch to TensorBoard and result JSONs
- **Sigmoid stats:** First validation batch prints min/max/mean/% > 0.5 per class (sanity check)

---

## 9. File Structure

```
experiments/class_weighting/
├── config.py                    # All hyperparameters, 15 LOSS_CONFIGS, auto-loads frequencies
├── losses_class_weighting.py    # 4 loss classes + factory function
│   ├── _PerClassTverskyBase     # Shared NaN-safe per-class Tversky computation
│   ├── PerClassWeightedTversky  # Static weights (6 configs)
│   ├── ClassBalancedTverskyLoss # CB loss with online/precomputed weights
│   ├── BalancedSoftmaxTverskyLoss # Logit adjustment inside Tversky
│   ├── SeesawTverskyLoss        # Mitigation + compensation factors
│   └── get_weighting_loss()     # Factory: type string → loss module
├── train.py                     # Training loop, validation, CLI, summary table
├── compute_class_frequencies.py # Scans data → class_frequencies.json
├── analyze_results.py           # Generates plots (grouped bars, training curves, family comparison)
├── run.sh                       # Launcher (compare, quick, single, summary, analyze, tensorboard)
├── class_frequencies.json       # Computed from 50 batches (auto-loaded by config.py)
├── datasplit.csv                # Train/val split (auto-generated)
├── README.md                    # Quick-start documentation
├── checkpoints/                 # Best model per config (~356 MB each)
├── results/                     # Per-config JSON with full training history
└── runs/                        # TensorBoard event files
```

---

## 10. How to Run

```bash
# 1. Compute class frequencies (one-time, ~5 min)
./run.sh compute-freqs

# 2. Run all 15 configs (4 parallel, ~20 hours total)
./run.sh compare

# 3. View results table
./run.sh summary

# 4. Generate analysis plots
./run.sh analyze

# 5. Run a single config
./run.sh single balanced_softmax_tau_1.0

# 6. TensorBoard
./run.sh tensorboard
```

---

## 11. Results (12/15 complete)

| Rank | Config | Mean Dice | nuc | mito_mem | er_mem | pm | golgi_mem | Time (min) |
|------|--------|-----------|-----|----------|--------|----|-----------|------------|
| 1 | **balanced_softmax_tau_1.0** | **0.5711** | 0.8511 | 0.6982 | 0.4284 | 0.5133 | 0.3645 | 390 |
| 2 | weight_inv_freq | 0.5694 | 0.8530 | 0.7137 | 0.4358 | 0.5045 | 0.3399 | 400 |
| 3 | weight_log_inv | 0.5678 | 0.8015 | 0.6838 | 0.4285 | 0.4902 | 0.4350 | 390 |
| 4 | weight_sqrt_inv | 0.5616 | 0.8653 | 0.7054 | 0.3904 | 0.4702 | 0.3768 | 409 |
| 5 | cb_beta_0.99 | 0.5475 | 0.7746 | 0.6982 | 0.4406 | 0.5033 | 0.3208 | 397 |
| 6 | weight_effective_num | 0.5459 | 0.7281 | 0.7034 | 0.4320 | 0.5144 | 0.3515 | 387 |
| 7 | weight_uniform | 0.5416 | 0.7047 | 0.7121 | 0.4214 | 0.5285 | 0.3414 | 409 |
| 8 | cb_beta_0.999 | 0.5269 | 0.7098 | 0.6998 | 0.4300 | 0.4550 | 0.3401 | 403 |
| 9 | balanced_softmax_tau_0.5 | 0.5225 | 0.7084 | 0.7128 | 0.4129 | 0.4564 | 0.3222 | 395 |
| 10 | cb_beta_0.9999 | 0.5188 | 0.7559 | 0.6354 | 0.3860 | 0.4748 | 0.3421 | 397 |
| 11 | weight_manual | 0.5143 | 0.7753 | 0.6623 | 0.3899 | 0.4662 | 0.2775 | 411 |
| 12 | balanced_softmax_tau_2.0 | 0.5036 | 0.7766 | 0.6401 | 0.4395 | 0.4566 | 0.2052 | 400 |

**Missing:** `seesaw_default`, `seesaw_strong_mitigate`, `seesaw_strong_compensate` (currently training)

### Best Per-Class

| Class | Best Dice | Config |
|-------|-----------|--------|
| nuc | 0.8653 | weight_sqrt_inv |
| mito_mem | 0.7137 | weight_inv_freq |
| er_mem | 0.4406 | cb_beta_0.99 |
| pm | 0.5285 | weight_uniform |
| golgi_mem | 0.4350 | weight_log_inv |

### Key Observations

1. **Balanced Softmax τ=1.0 is the overall winner** (0.5711) — the theory-optimal temperature works best
2. **Simple inverse-frequency weighting is a close second** (0.5694) — sometimes simpler is better
3. **Log-inverse gets the best golgi_mem** (0.4350) — the rarest class benefits from moderate reweighting
4. **Uniform baseline ranks 7th** (0.5416) — class weighting provides ~5% improvement
5. **Manual weights underperform** (rank 11) — data-driven beats hand-tuned
6. **Strong adjustments hurt**: τ=2.0 is worst in its family, β_cb=0.9999 is worst in CB family
7. **Effective number weights collapsed to uniform** due to β^n underflow with millions of voxels — the online CB loss handles this correctly

---

## 12. Reproducing This Experiment

```bash
cd /home/spuser/ws/CellMap-Segmentation
micromamba activate csc
cd experiments/class_weighting

# Ensure frequencies are computed
./run.sh compute-freqs

# Full comparison (~20 hours on 2× RTX 3090)
./run.sh compare

# Or run a single config (~6.5 hours)
./run.sh single balanced_softmax_tau_1.0
```

Environment: `micromamba` env `csc` with PyTorch 2.x, cellmap-segmentation-challenge package installed from this repo.
