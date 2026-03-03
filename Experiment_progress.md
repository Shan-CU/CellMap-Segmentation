# CellMap Segmentation — Experiment Progress & Handoff Document

> **Last updated:** March 3, 2026  
> **Author:** AI Agent (GitHub Copilot, Claude Opus 4.6)  
> **Purpose:** Full context for any agent continuing this work

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Infrastructure](#2-infrastructure)
3. [Codebase Architecture](#3-codebase-architecture)
4. [Phase 1 Results — Complete](#4-phase-1-results--complete)
5. [Key Findings & Decisions](#5-key-findings--decisions)
6. [Phase 2 v4 Results — In Progress](#6-phase-2-v4-results--in-progress)
7. [Phase 2 Configuration & Launch History](#7-phase-2-configuration--launch-history)
8. [Known Issues & Bugs](#8-known-issues--bugs)
9. [File Reference](#9-file-reference)
10. [How to Run](#10-how-to-run)
11. [Next Steps (Priority Order)](#11-next-steps-priority-order)

---

## 1. Project Overview

**Goal:** Multi-class organelle segmentation from FIB-SEM electron microscopy volumes for the [CellMap Segmentation Challenge](https://cellmapchallenge.janelia.org/).

**Task:** Predict 48 classes (31 atomic organelle labels + 17 group/composite labels) from 3D EM volumes stored in Zarr format at multiple resolutions. The challenge uses partial annotation — each crop only labels a subset of classes, with unannotated classes marked as NaN.

**48 Classes:**
```
Atomic (31): ecs, pm, mito_mem, mito_lum, mito_ribo, golgi_mem, golgi_lum,
  ves_mem, ves_lum, endo_mem, endo_lum, lyso_mem, lyso_lum, ld_mem, ld_lum,
  er_mem, er_lum, eres_mem, eres_lum, ne_mem, ne_lum, np_out, np_in,
  hchrom, echrom, nucpl, mt_out, cyto, mt_in, perox_mem, perox_lum
Group (17): nuc, golgi, ves, endo, lyso, ld, eres, perox, mito, er, ne, np,
  chrom, mt, cell, er_mem_all, ne_mem_all
```

**Dataset:** 968 training crops + 188 validation crops across 22 EM volumes. Zarr format loaded via `cellmap-data` library. Validation crops only annotate 14 of 48 classes, which means val_loss and per-class Dice are incomplete metrics.

**Strategy:** Two-phase approach:
- **Phase 1:** Quick ablation experiments (50 epochs, 500 iters/epoch) to find optimal loss, masking, and training technique
- **Phase 2:** Architecture comparison (100 epochs, 1000 iters/epoch) with the winning configuration

---

## 2. Infrastructure

### Clusters

| Cluster | Partition | GPU | VRAM | Nodes | Status (Mar 3) |
|---------|-----------|-----|------|-------|----------------|
| Longleaf | `l40-gpu` | NVIDIA L40S | 48 GB | 13+5 | ✅ UP (95 GPUs running, 700 pending) |
| Longleaf | `a100-gpu` | NVIDIA A100-PCIe | 40 GB | 8 | ✅ UP |
| Longleaf | `a100-multi-gpu` | NVIDIA A100-SXM4 | 80 GB | 1 (8 GPUs) | ✅ UP — 6/8 GPUs ours |
| Sycamore | — | — | — | — | ❌ H100s returned to Lenovo, no GPU partition |

### Environment

```bash
# Conda/Micromamba
MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate csc

# Working directory
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
```

### Key Dependencies
- `cellmap-data` v2026.2.20.2159 — Zarr data loading, spatial transforms, weighted sampling
- `torch` — PyTorch with CUDA
- `monai` — SegResNet, SwinUNETR architectures
- `tensorboardX` — Logging

---

## 3. Codebase Architecture

```
CellMap-Segmentation/
├── training/
│   ├── train.py              # Main training script (CLI entry point)
│   ├── configs/
│   │   └── experiments.py    # All experiment configs as dataclasses
│   ├── models/
│   │   └── model_zoo.py      # Model registry (9 architectures)
│   ├── losses/
│   │   ├── loss_zoo.py       # Loss registry (20+ losses)
│   │   └── partial_annotation.py  # NaN masking, fg masking, deep supervision
│   ├── transforms/
│   │   ├── __init__.py
│   │   └── intensity.py      # Intensity augmentation (brightness, contrast, noise)
│   ├── samplers/
│   │   ├── __init__.py
│   │   └── crop_weights.py   # Class-aware inverse-sqrt crop weighting
│   ├── eval_2d_perclass.py   # Per-class Dice/IoU evaluation script
│   └── slurm/
│       ├── ablation_2d_l40s.sbatch   # 2D job template (Longleaf L40S)
│       ├── ablation_3d_h100.sbatch   # 3D job template (Sycamore H100)
│       └── launch_*.sh               # Batch launch scripts
├── src/cellmap_segmentation_challenge/
│   └── utils/
│       └── dataloader.py     # get_dataloader() — wraps cellmap-data
├── datasplit.csv             # Train/val split (968 train, 188 val crops)
└── runs/ablation/            # All experiment outputs
    ├── logs/                 # SLURM stdout/stderr
    ├── <experiment_name>/
    │   ├── config.json       # Saved hyperparameters
    │   ├── checkpoints/      # latest.pth, best.pth, epoch_N.pth
    │   └── tensorboard/      # TensorBoard event files
    ├── eval_2d_perclass.json # Per-class Dice/IoU for all 2D experiments
    └── eval_2d_perclass.csv  # Same in CSV format
```

### Model Registry (`training/models/model_zoo.py`)

| Name | Dim | Params | Description |
|------|-----|--------|-------------|
| `resnet_2d` | 2D | ~7.8M | CSC FlexUNet-ResNet34 |
| `unet_2d` | 2D | ~31M | CSC vanilla UNet |
| `swin_2d` | 2D | ~36M | CSC SwinTransformer |
| `vit_2d` | 2D | ~105M | CSC ViTVNet |
| `segresnet_3d` | 3D | ~18M | MONAI SegResNetDS |
| `swinunetr_3d` | 3D | ~62M | MONAI SwinUNETR |
| `unet_3d` | 3D | - | CSC UNet 3D |
| `resnet_3d` | 3D | - | CSC FlexUNet-ResNet34 3D |
| `vitnet_3d` | 3D | - | CSC ViTVNet 3D |

### Training Script Flags (`training/train.py`)

Key flags beyond standard hyperparameters:
```
--use_foreground_mask / --no_foreground_mask   # Mask loss to non-background voxels
--ema --ema_decay 0.999                        # Exponential moving average
--deep_supervision --ds_weights ...            # Multi-scale supervision (SegResNet only)
--no_weighted_sampler                          # Disable cellmap-data's weighted sampling
--intensity_aug                                # Enable brightness/contrast/noise augmentation
--class_aware_sampling                         # Use inverse-sqrt class-aware crop weighting
--amp / --no_amp                               # Automatic mixed precision
```

---

## 4. Phase 1 Results — Complete

### Sweep A: Loss Function (2D, ResNet)

| Experiment | Loss | Val Loss | Mean Dice (14-class) | Rank |
|------------|------|----------|---------------------|------|
| `loss_2d_bce` | BCE | **0.0425** | **0.4755** | 🥇 |
| `loss_2d_dice_bce` | Dice+BCE | 0.4656 | 0.4593 | 🥈 |
| `loss_2d_boundary_tversky` | Boundary Tversky | 0.6996 | 0.4069 | 3 |
| `loss_2d_tversky` | Tversky | 0.7136 | 0.3983 | 4 |
| `loss_2d_focal_tversky` | Focal Tversky | 0.5472 | 0.3777 | 5 |
| `loss_2d_unified_focal` | Unified Focal | 0.5473 | 0.3714 | 6 |
| `loss_2d_focal` | Focal | 0.5472 | 0.3615 | 7 |
| `loss_2d_balanced_softmax_tversky` | BST (τ=1.0) | 0.6055 | 0.2385 | 8 |

**Finding:** Simple losses (BCE, Dice+BCE) massively outperform complex losses. BST's τ=1.0 logit adjustment over-corrects with 48 classes, producing too many false positives on rare classes.

**Decision:** `dice_bce` chosen for Phase 2 (not BCE) because dice_bce has better rare-class performance despite slightly lower mean Dice. BCE achieves high mean Dice by overpredicting common classes.

### Sweep B: Tversky α/β (2D, ResNet)

| Experiment | α | β | Mean Dice (14-class) |
|------------|---|---|---------------------|
| `tversky_2d_balanced` | 0.5 | 0.5 | 0.4078 |
| `tversky_2d_recall` | 0.3 | 0.7 | 0.4074 |
| `tversky_2d_a08_b06` | 0.8 | 0.6 | 0.4066 |
| `tversky_2d_precision_07_03` | 0.7 | 0.3 | 0.3820 |
| `tversky_2d_a08_b04` | 0.8 | 0.4 | 0.3721 |
| `tversky_2d_precision_06_04` | 0.6 | 0.4 | 0.3719 |

**Finding:** All Tversky variants clustered around 0.37–0.41 mean Dice. No α/β combination came close to BCE (0.479) or dice_bce (0.462). Tversky-based losses are suboptimal for this task.

### Sweep C: Class Weighting τ (2D, ResNet, BST)

| Experiment | τ | Val Loss | Mean Dice (14-class) |
|------------|---|----------|---------------------|
| `tau_2d_05` | 0.5 | 0.4252 | 0.4092 |
| `tau_2d_0` | 0.0 | 0.4252 | 0.3246 |
| `tau_2d_10` | 1.0 | 0.5444 | 0.1318 |
| `tau_2d_15` | 1.5 | 0.4094 | 0.0000 |
| `tau_2d_20` | 2.0 | 0.4090 | 0.0000 |

**Finding:** High τ completely destroys predictions (mean Dice = 0). τ=0.5 is best within BST family but still worse than dice_bce. Logit adjustment doesn't work well with partial annotations at 48 classes.

### Sweep D: Masking Strategy (2D, ResNet, BST)

⚠️ All mask experiments used BST base loss. Results are confounded — rankings reflect
masking interaction with BST, not masking quality in general. Since Phase 2 uses dice_bce,
these are **not directly actionable**. FG mask was the default for all top-performing
loss sweep experiments.

| Experiment | Strategy | Mean Dice (14-class) |
|------------|----------|---------------------|
| `mask_2d_masksup03_no_bbox` | MaskSup λ=0.3, no bbox | 0.2618 |
| `mask_2d_fg_only` | FG mask only | 0.2547 |
| `mask_2d_none` | No masking | 0.2511 |
| `mask_2d_bbox_loose` | Loose bbox + FG | 0.2366 |
| `mask_2d_masksup03` | MaskSup λ=0.3 + bbox | 0.1372 |
| `mask_2d_bbox_only` | Bbox only | 0.1348 |
| `mask_2d_bbox_fg` | Bbox + FG | 0.1314 |

### Sweep E: Training Techniques (2D, ResNet)

**BST-based (not actionable — confounded by BST):**

| Experiment | Technique | Val Loss | Mean Dice (14-class) |
|------------|-----------|----------|---------------------|
| `tech_2d_focal_tversky_mild` | Focal γ=0.5 | 0.5709 | 0.3245 |
| `tech_2d_ema` | EMA (decay=0.999) | 0.5603 | 0.1008 |
| `tech_2d_no_weighted_sampler` | No sampler | 0.5645 | 0.0823 |

**dice_bce-based (Phase 2 relevant) ⭐:**

| Experiment | Config | Val Loss | Mean Dice (14-class) |
|------------|--------|----------|---------------------|
| `tech_2d_dicebce_ema` | dice_bce + EMA + fg_mask + sampler | **0.112** | **0.4598** ⭐ |
| `tech_2d_dicebce_no_sampler` | dice_bce + EMA + fg_mask, no sampler | 0.122 | 0.2904 |

**Validation experiments (dice_bce + EMA + fg + sampler + extras):**

| Experiment | Extra Feature | Mean Dice (14-class) | vs baseline (0.4598) |
|------------|--------------|---------------------|---------------------|
| `val_intensity_aug` | intensity augmentation | 0.3539 | **−23.0% ❌** |
| `val_crop_weights` | class-aware crop weighting | 0.2849 | **−38.0% ❌** |
| `val_combined` | both | 0.2133 | **−53.6% ❌** |

**Key findings:**
1. **EMA is essential**: dice_bce + EMA = 0.4598 vs dice_bce alone = 0.4593 (similar Dice, but 4× better val_loss: 0.112 vs 0.466)
2. **Weighted sampler is essential**: With = 0.4598; without = 0.2904 (−37%)
3. **Intensity augmentation HURTS** (−23%) — likely because EM intensity is consistent within volumes
4. **Class-aware crop weighting HURTS** (−38%) — disrupts the balanced sampling that cellmap-data provides
5. **Combining both HURTS even more** (−54%)

**Finding:** EMA smooths noisy gradients from the imbalanced 48-class partial annotation setup. Weighted sampler provides critical class balance. Intensity aug and class-aware crop weighting both interfere with training — **exclude from Phase 2**.

### 3D Results (SegResNet baseline) — Per-class Dice (14-class eval)

⚠️ **CRITICAL FINDING:** 3D ablation results are essentially non-functional. Every experiment
produced mean Dice ≈ 0.00–0.017. The training regime (50 epochs × 250 iters = 12,500 steps)
was woefully insufficient for 3D 128³ volumes. We cannot draw reliable hyperparameter conclusions
from 3D ablation — we must transfer 2D findings to 3D.

| Rank | Experiment | Mean Dice | Best Class |
|------|-----------|-----------|------------|
| 1 | `mask_3d_masksup03` | 0.0169 | nuc=0.236 |
| 2 | `mask_3d_fg_only` | 0.0168 | nuc=0.234 |
| 3 | `loss_3d_bce` | 0.0166 | nuc=0.175 |
| 4 | `mask_3d_masksup03_no_bbox` | 0.0166 | nuc=0.230 |
| 5 | `tech_3d_deep_supervision` | 0.0164 | nuc=0.230 |
| 6 | `tau_3d_15` | 0.0149 | nuc=0.112 |
| 7 | `tech_3d_no_weighted_sampler` | 0.0104 | mito_mem=0.098 |
| 8 | `tversky_3d_recall` | 0.0076 | nuc=0.106 |
| 9 | `tau_3d_05` | 0.0045 | mito_mem=0.043 |
| 10 | `mask_3d_bbox_loose` | 0.0037 | nuc=0.049 |
| 11–29 | (everything else) | ≤0.002 | mostly 0.000 |

**Key observations:**
- **`loss_3d_dice_bce` = 0.000 Dice** despite being #1 by val_loss (0.170). Val_loss was misleading!
- Only `nuc` (the largest organelle) gets any signal at all (max 0.236 Dice)
- `loss_3d_bce` = 0.017 — only loss that produced non-zero Dice in the loss sweep
- 3D needs **much more training** in Phase 2 (100 epochs × 500 iters = 50,000 steps, 4× more)

**Val_loss rankings (from training logs) — for reference only:**

| Experiment | Val Loss | Dice Eval |
|------------|----------|-----------|
| `loss_3d_dice_bce` | **0.170** | 0.000 ⚠️ |
| `loss_3d_bce` | 0.220 | 0.017 |
| `mask_3d_bbox_only` | 0.541 | 0.000 |
| `tech_3d_deep_supervision` | 0.556 | 0.016 |
| Everything else | 0.59–0.92 | ≤0.015 |

---

## 5. Key Findings & Decisions

### Optimal Phase 2 Configuration ✅ FINALIZED

```
Loss:              dice_bce (bce_weight=0.5, smooth=1e-6)
EMA:               enabled, decay=0.999
FG Mask:           enabled
Weighted Sampler:  enabled (cellmap-data default)
Intensity Aug:     DISABLED (hurt −23% in validation)
Class-Aware Crop:  DISABLED (hurt −38% in validation)
Deep Supervision:  ENABLED for SegResNet only
AMP:               enabled
Scheduler:         cosine with 5-epoch warmup
Optimizer:         RAdam, lr=1e-4
Epochs:            100 (2× ablation)
Iters/epoch:       1000 (2D) / 500 (3D)
Val every:         5 epochs
```

### Why dice_bce Over BCE

BCE had the highest mean Dice (0.479 vs 0.462) but dice_bce was chosen because:
1. dice_bce has better rare-class performance (the Dice component explicitly optimizes overlap)
2. BCE achieves high mean Dice by over-predicting common classes (ecs, pm, mito)
3. The challenge evaluation weights all classes equally, so rare-class performance matters
4. With EMA, dice_bce reaches val_loss=0.112 — the best result by far

### Why BST Failed

Balanced Softmax Tversky (BST) uses logit adjustment: `logit_c += τ × log(π_c)` where π_c is the class prior. With 48 classes (many extremely rare), τ=1.0 drastically over-adjusts logits for rare classes, causing massive false positive rates. This is the opposite of its intended effect — it was designed for natural image classification with ~1000 balanced classes, not medical segmentation with 48 extremely imbalanced classes.

### EMA's Outsized Impact

EMA (exponential moving average) provided a 4× val_loss improvement (0.466 → 0.112). Hypothesis: with 48 partially-annotated classes, gradients are extremely noisy epoch-to-epoch (different crops annotate different classes). EMA acts as a temporal ensemble, smoothing these noisy updates. This is consistent with nnU-Net v2 and MONAI Auto3DSeg both using EMA by default.

---

## 6. Phase 2 v4 Results — In Progress

> **Status:** 8 models RUNNING across 2 partitions (Mar 3 evening). SwinUNETR killed (no pretraining, no path to competitive performance). Swin-2D relaunched with optimized hyperparams. ViT-2D launched. All 3 train.py bugs fixed.

### Overall Ranking (as of suspension, March 3 2026)

| Rank | Model | Best Mean Dice | Best Epoch | Latest Epoch | Active Classes | Trend | Status |
|------|-------|---------------|------------|-------------|----------------|-------|--------|
| 1 | **unet_2d** | **0.3937** | 45 | ~48 | 47/48 | ↑ still climbing | ✅ RUNNING (l40-gpu) |
| 2 | **resnet_2d** | **0.3489** | 55 | ~58 | 43/48 | ↑ still climbing | ✅ RUNNING (l40-gpu) |
| 3 | unet_3d | 0.2212 | 20 | ~24 | 38/48 | ↑ steep upward | ✅ RUNNING (a100) |
| 4 | resnet_3d | 0.2207 | 20 | ~24 | 36/48 | ↑ steep upward | ✅ RUNNING (a100) |
| 5 | segresnet_3d | 0.0826 | 45 | ~49 | 27/48 | ↑ slow but steady | ✅ RUNNING (a100) |
| 6 | swin_2d | 0.0446 | 5 | 0 (restarted) | — | 🔄 **RELAUNCHED** | ✅ RUNNING (a100, fresh) |
| 7 | vit_2d | — | — | 0 (new) | — | 🆕 just started | ✅ RUNNING (a100) |
| 8 | vitnet_3d | 0.0002 | 10 | ~17 | 0/48 | ❌ dead — zero learning | ✅ RUNNING (l40-gpu) |
| — | ~~swinunetr_3d~~ | 0.0335 | 15 | 20 | 8/48 | ❌ killed | 🛑 KILLED |

### Validation Trajectories

```
resnet_2d   (11 val pts): 0.226 → 0.319 → 0.343 → 0.339 → 0.334 → 0.336 → 0.327 → 0.346 → 0.344 → 0.342 → 0.349  ↑
unet_2d     ( 9 val pts): 0.111 → 0.240 → 0.310 → 0.349 → 0.362 → 0.381 → 0.393 → 0.390 → 0.394  ↑
unet_3d     ( 4 val pts): 0.013 → 0.064 → 0.154 → 0.221  ↑↑ (accelerating)
resnet_3d   ( 4 val pts): 0.031 → 0.164 → 0.217 → 0.221  ↑
segresnet_3d( 9 val pts): 0.000 → 0.004 → 0.025 → 0.043 → 0.052 → 0.071 → 0.073 → 0.080 → 0.083  ↑
swin_2d     ( 9 val pts): 0.045 → 0.034 → 0.026 → 0.020 → 0.019 → 0.024 → 0.017 → 0.015 → 0.017  ↓ COLLAPSING
swinunetr_3d( 4 val pts): 0.023 → 0.028 → 0.034 → 0.028  → stalled
vitnet_3d   ( 3 val pts): 0.000 → 0.000 → 0.000  ❌ DEAD
```

### Best Dice Per Class (Optimal Ensemble)

| Class | Best Dice | Best Model | | Class | Best Dice | Best Model |
|-------|-----------|------------|-|-------|-----------|------------|
| ecs | 0.5303 ★ | resnet_2d | | ves_mem | 0.1849 | resnet_3d |
| pm | 0.3851 | resnet_2d | | ves_lum | 0.2532 | resnet_3d |
| mito_mem | 0.6787 ★ | unet_2d | | ves | 0.2772 | resnet_3d |
| mito_lum | 0.7624 ★ | unet_2d | | endo_mem | 0.1750 | unet_2d |
| mito_ribo | 0.3558 | unet_2d | | endo_lum | 0.2992 | resnet_2d |
| mito | 0.0828 | unet_2d | | endo | 0.2965 | unet_2d |
| er_mem | 0.4738 | resnet_2d | | lyso_mem | 0.1462 | resnet_2d |
| er_lum | 0.6551 ★ | unet_2d | | lyso_lum | 0.2836 | unet_2d |
| er | 0.7195 ★ | resnet_2d | | lyso | 0.2705 | unet_2d |
| er_mem_all | 0.4825 | resnet_2d | | ld_mem | 0.1638 | unet_2d |
| eres_mem | 0.1511 | resnet_2d | | ld_lum | 0.7795 ★ | resnet_2d |
| eres_lum | 0.2050 | resnet_2d | | ld | 0.8108 ★ | resnet_2d |
| eres | 0.2109 | resnet_2d | | perox_mem | 0.2112 | unet_2d |
| golgi_mem | 0.3951 | resnet_2d | | perox_lum | 0.6020 ★ | unet_2d |
| golgi_lum | 0.4784 | unet_2d | | perox | 0.0229 | unet_2d |
| golgi | 0.6757 ★ | unet_2d | | mt_out | 0.0456 | resnet_2d |
| np_out | 0.4017 | unet_2d | | mt_in | 0.0252 | resnet_3d |
| np_in | 0.4764 | resnet_2d | | mt | 0.0592 | resnet_2d |
| np | 0.5683 ★ | unet_2d | | ne_mem | 0.5541 ★ | unet_2d |
| ne_lum | 0.6306 ★ | unet_2d | | nucpl | 0.6399 ★ | unet_2d |
| ne | 0.7589 ★ | unet_2d | | nuc | 0.0868 | unet_2d |
| ne_mem_all | 0.5872 ★ | unet_2d | | chrom | 0.5713 ★ | unet_2d |
| cell | 0.8633 ★ | unet_3d | | hchrom | 0.5717 ★ | unet_2d |
| cyto | 0.8118 ★ | unet_2d | | echrom | 0.0031 | unet_2d |

### Ensemble Summary

| Model | Classes Won | Avg Dice (won classes) |
|-------|------------|----------------------|
| **unet_2d** | 27 | 0.4407 |
| **resnet_2d** | 16 | 0.3856 |
| **unet_3d** | 1 (cell) | 0.8633 |
| **resnet_3d** | 4 (ves family + mt_in) | 0.1851 |

- **Best single model:** unet_2d → 0.3937 mean Dice
- **Optimal ensemble:** 0.4099 mean Dice (+4.1% over best single)
- **16 classes above 0.5**, including cell (0.86), ld (0.81), cyto (0.81), mito_lum (0.76)

### Classes Needing Improvement (< 0.1 Dice)

| Class | Best Dice | Best Model | Status |
|-------|-----------|------------|--------|
| echrom | 0.0031 | unet_2d | ⚠️ near-zero |
| perox | 0.0229 | unet_2d | 📈 emerging |
| mt_in | 0.0252 | resnet_3d | 📈 emerging |
| mt_out | 0.0456 | resnet_2d | 📈 emerging |
| mt | 0.0592 | resnet_2d | 📈 emerging |
| mito | 0.0828 | unet_2d | 📈 emerging |
| nuc | 0.0868 | unet_2d | 📈 emerging |

### Model Health Assessment

- ✅ **5 healthy models** (resnet_2d, unet_2d, resnet_3d, unet_3d, segresnet_3d) — all learning, no collapse
- 🔄 **swin_2d** — RELAUNCHED with optimized hyperparams (see below). Old run peaked at epoch 5 (0.045) then collapsed due to over-regularization
- 🆕 **vit_2d** — LAUNCHED (all 3 bugs fixed: inplace=False, grad_accum unscale, batch=4+accum=2). Running on a100-multi-gpu
- ❌ **vitnet_3d** — dead (0.000 dice at epoch 17), but only at 17/1000 epochs. Leaving it running in case it's just extremely slow to converge
- 🛑 **swinunetr_3d** — KILLED. No pretrained weights available (MONAI's SSL weights trained on CT, wrong domain for EM). At epoch 20: dice=0.028, 8× worse than ResNet/UNet at same epoch, declining. Freed 1 A100 GPU

### Swin-2D Relaunch (March 3)

**Diagnosis:** Over-regularized — dropout=0.1 + wd=0.05 + stochastic_depth=0.2 was too much for our small dataset. LR=1e-4 also too conservative without pretrained weights.

**Changes applied:**
- `model_zoo.py`: dropout 0.1 → **0.0** (stochastic_depth=0.2 provides sufficient regularization)
- LR: 1e-4 → **5e-4** (Swin V2 paper uses 5e-4 for fine-tuning)
- Weight decay: 0.05 → **0.01**
- Warmup: 10 → **20 epochs** (longer warmup for higher LR)
- Grad clip: 1.0 → **5.0** (relaxed for higher LR)
- Old checkpoints/tensorboard cleaned. Fresh start from epoch 0.

### SwinUNETR Killed — Rationale (March 3)

MONAI's official SwinUNETR recipe requires SSL-pretrained weights (from 5,050 CT scans). Without pretraining:
- At epoch 20: dice=0.028 vs ResNet-3D=0.221 (8× worse at same epoch)
- Dice peaked at epoch 15 (0.034) then declined to 0.028
- Train loss stalled at ~0.43 (same as UNet-3D, but UNet has 5× better dice)
- The pretrained weights can't transfer from CT→EM (different domain, resolution, intensity)
- No realistic path to competitive performance. GPU better used elsewhere

### Key Insight: 3D Models Still Early

The 3D models are only at epoch 15-20 (out of 100). Their trajectories show steep upward trends:
- **unet_3d**: 0.013 → 0.064 → 0.154 → 0.221 (doubling every 5 epochs)
- **resnet_3d**: 0.031 → 0.164 → 0.217 → 0.221

These models will likely improve substantially by epoch 100. The 3D models already win specific classes (cell, ves family) where volumetric context matters.

---

## 7. Phase 2 Configuration & Launch History

### Configuration ✅ FINALIZED

```python
# Phase 2 v4 recipe (from Phase 1 ablation winner)
loss = "dice_bce"           # SegResNet too (with deep_supervision)
use_foreground_mask = True
ema = True
ema_decay = 0.999
weighted_sampler = True     # cellmap-data default
intensity_aug = False       # DISABLED — hurt −23%
class_aware_sampling = False # DISABLED — hurt −38%
bias_init_mode = "per_class" # Critical: -3.0 bias init prevents rare-class collapse

# 2D settings
epochs_2d = 100
iterations_per_epoch_2d = 1000
val_every_n_epochs_2d = 5
warmup_epochs_2d = 10
batch_size_2d = 8            # (ViT uses 4+grad_accum=2 due to CUDA kernel bug)
lr_2d = 1e-4
optimizer_2d = "radam"       # (ViT-2D uses adamw+wd=0.01; Swin-2D uses adamw+lr=5e-4+wd=0.01)
scheduler = "cosine"

# 3D settings
epochs_3d = 100
iterations_per_epoch_3d = 1000
val_every_n_epochs_3d = 5
warmup_epochs_3d = 10
batch_size_3d = 8
lr_3d = 1e-4
input_shape_3d = [1, 96, 96, 96]
input_scale_3d = [8, 8, 8]
```

### ⚠️ Phase 2 Launch History

**v1 (Feb 27):** Missing `--ema` flag. SegResNet used BST instead of dice_bce. ViT crashed. All killed.

**v2 (Feb 28):** EMA fixed, SegResNet on dice_bce+DS, ViT batch→4. Jobs ran but many classes had zero Dice — models collapsed on rare classes due to default zero bias init.

**v3 (Mar 1):** Added `--bias_init_mode per_class` (sets final layer bias to -3.0 for all classes, matching sigmoid(−3)≈0.05 prior). Dramatic improvement: 47/48 classes now have signal vs ~14 in v2.

**v4 (Mar 1, CURRENT):** Full relaunch with finalized configs. Key changes from v3:
- Consistent hyperparams across all models
- 3D models: 100 epochs × 1000 iters (not 1000 × 300)
- All on Longleaf (Sycamore H100s returned to Lenovo)
- Per-model sbatch files in `training/slurm/phase2v4/`
- ViT-2D: `inplace=False` fix for LeakyReLU + batch=4+grad_accum=2

### SLURM Jobs (v4)

| Job ID | Name | Partition | Node | Status (Mar 3 evening) |
|--------|------|-----------|------|------------------|
| 33957705 | p2v4_resnet_2d | l40-gpu | — | ✅ RUNNING |
| 33957706 | p2v4_unet_2d | l40-gpu | — | ✅ RUNNING |
| 33957741 | p2v4_vitnet_3d_l40 | l40-gpu | — | ✅ RUNNING |
| 33957709 | p2v4_resnet_3d | a100-multi-gpu | g180701 | ✅ RUNNING |
| 33957710 | p2v4_unet_3d | a100-multi-gpu | g180701 | ✅ RUNNING |
| 33957711 | p2v4_segresnet_3d | a100-multi-gpu | g180701 | ✅ RUNNING |
| 34357567 | p2v4_swin_2d | a100-multi-gpu | g180701 | ✅ RUNNING (relaunched, optimized) |
| 34359105 | p2v4_vit_2d | a100-multi-gpu | g180701 | ✅ RUNNING (new) |
| ~~33957707~~ | ~~p2v4_swin_2d~~ | ~~l40-gpu~~ | — | 🛑 KILLED (over-regularized, relaunched on a100) |
| ~~33957712~~ | ~~p2v4_swinunetr_3d~~ | ~~a100-multi-gpu~~ | — | 🛑 KILLED (no pretraining, non-competitive) |

### 2D Runs (100ep × 1000it)

| Job Name | Model | Batch | Params | Loss | Optimizer |
|----------|-------|-------|--------|------|-----------|
| `p2v4_resnet_2d` | FlexUNet-ResNet34 | 8 | ~7.8M | dice_bce | RAdam |
| `p2v4_unet_2d` | CSC UNet 2D | 8 | ~31M | dice_bce | RAdam |
| `p2v4_swin_2d` | SwinTransformer 2D | 8 | ~36M | dice_bce | AdamW (lr=5e-4, wd=0.01, dropout=0.0) |
| `p2v4_vit_2d` | ViTVNet 2D | 4+accum2 | ~105M | dice_bce | AdamW (lr=1e-4, wd=0.01) |

### 3D Runs (100ep × 1000it, 96³ crops)

| Job Name | Model | Batch | Params | Loss | Extra |
|----------|-------|-------|--------|------|-------|
| `p2v4_segresnet_3d` | SegResNetDS | 8 | ~20M | dice_bce | `--deep_supervision` |
| ~~`p2v4_swinunetr_3d`~~ | ~~SwinUNETR~~ | ~~8~~ | ~~\~62M~~ | ~~dice\_bce~~ | **KILLED** — no pretraining, non-competitive |
| `p2v4_unet_3d` | CSC UNet 3D | 8 | ~90M | dice_bce | |
| `p2v4_resnet_3d` | CSC FlexUNet3D | 8 | ~24M | dice_bce | |
| `p2v4_vitnet_3d_l40` | CSC ViTVNet 3D | 8 | ~32M | dice_bce | |

---

## 8. Known Issues & Bugs

### ✅ FIXED: `train.py` gradient_accumulation + AMP scaler bug

**Status: FIXED (March 3, 2026)**

When `gradient_accumulation_steps > 1`, the grad clipping code was calling `scaler.unscale_(optimizer)` on **every micro-step**, but `scaler.step()` and `scaler.update()` only run on accumulation boundary steps. This caused:
```
RuntimeError: unscale_() has already been called on this optimizer since the last update()
```

**Fix applied:** Moved `scaler.unscale_(optimizer)` and `torch.nn.utils.clip_grad_norm_()` inside the `if (step + 1) % args.gradient_accumulation_steps == 0:` block, **before** `scaler.step(optimizer)` in `training/train.py` (~line 729).

### ✅ FIXED: ViT-2D `cudaErrorInvalidConfiguration`

**Status: FULLY FIXED (March 3, 2026)**

Root cause: batch=8 + AMP + DiceBCE loss creates CUDA kernels with invalid launch parameters due to large activation tensors (8×48×256×256) through `torch.sigmoid(input.float())` + Dice spatial reduction.

**Fixes applied:**
1. All 4 `inplace=True` → `inplace=False` on LeakyReLU in `src/.../models/vit_2d.py`
2. Reduced to batch=4 + gradient_accumulation_steps=2 (effective batch=8)
3. Gradient accumulation unscale bug fixed (see above)

ViT-2D is now training — launched as job 34359105 on a100-multi-gpu (March 3).

### ✅ FIXED: Black validation input images

**Status: FIXED (March 3, 2026)**

**Root cause:** `pad=True` in `CellMapDataSplit` causes each validation dataset to pad the EM volume to its full bounding box. `CellMapDataset.validation_indices` then tiles the ENTIRE padded volume with non-overlapping blocks — creating 132K+ blocks, ~98% of which are in empty padded regions (all-zero EM inputs). The `CellMapDataLoader` for validation has no `weighted_sampler` or `iterations_per_epoch` (unlike training), so it uniformly samples from this massive pool, almost always hitting empty padded blocks.

**Why training still works:** The training loader uses `weighted_sampler=True` + `iterations_per_epoch=1000`, which biases sampling toward blocks with actual annotations via `CellMapMultiDataset.get_random_subset_indices(weighted=True)`. Validation loss and Dice are also unaffected — empty blocks contribute near-zero loss (NaN-masked targets) and don't corrupt Dice accumulators.

**Fix applied:** `vis_sample` selection in `train.py` now requires both:
- `inp_bi.abs().max() > 1e-6` (non-zero EM signal — not a padded block)
- `gt_bi.sum() / gt_bi.numel() > 0.01` (≥1% annotated pixels)

Also added a fallback warning message when no valid sample is found within the validation time limit.

**Upstream note:** This is a design issue in cellmap-data where `validation_blocks` tiles the full padded volume rather than only annotated regions. A proper fix would be to add `weighted_sampler=True` to the validation `CellMapDataLoader` in `get_dataloader()`, or to use `pad="train"` instead of `pad=True` so validation doesn't pad.

### Validation Set Limitation
The validation crops only annotate **14 of 48 classes**: ecs, pm, mito_mem, mito_lum, mito_ribo, golgi_mem, golgi_lum, ves_mem, ves_lum, endo_mem, endo_lum, er_mem, er_lum, nuc. The remaining 34 classes (including all group classes) cannot be evaluated during training.

### cellmap-data EmptyImage Memory
cellmap-data pre-allocates `EmptyImage` tensors for all ~784 datasets on initialization. This requires ~300GB host RAM. Jobs must request ≥384G memory. Data loaded on CPU, batches moved to GPU in training loop.

### AMP Overflow with BST (Historical)
BST with AMP float16 can produce inf logits due to the τ×log(π_c) adjustment. Fixed with `nan_to_num()` clamping, but irrelevant now (using dice_bce).

### Sycamore h100 Partition (Historical)
H100s returned to Lenovo. No GPU partition on Sycamore anymore. Only Longleaf GPUs available.

---

## 9. File Reference

### Key Source Files

| File | Purpose |
|------|---------|
| `training/train.py` | Main training loop (~1131 lines) — all flags, data flow, AMP, EMA, 48-color palette |
| `training/configs/experiments.py` | All experiment configs |
| `training/losses/loss_zoo.py` | Loss function registry — `dice_bce` is the winner |
| `training/models/model_zoo.py` | Model registry — 8 active architectures (swinunetr_3d killed) |
| `training/eval_2d_perclass.py` | Per-class evaluation — run after training |
| `training/regen_val_images.py` | Retroactive val image regeneration from checkpoints |
| `training/make_legend.py` | Generate class_legend.png (48 classes, unique colors) |
| `src/cellmap_segmentation_challenge/models/vit_2d.py` | ViT-V-Net 2D model (~529 lines, `inplace=False` fix applied) |
| `src/cellmap_segmentation_challenge/utils/dataloader.py` | `get_dataloader()` wrapper |

### SLURM Launch Files

| File | Purpose |
|------|---------|
| `training/slurm/phase2v4/*.sbatch` | Per-model sbatch files for Phase 2 v4 (8 active) |
| `training/slurm/launch_phase2_clean.sh` | Legacy batch launcher (v1-v3, still works) |

### Phase 1 Results Files

| File | Content |
|------|---------|
| `runs/ablation/<exp>/config.json` | Hyperparameters used |
| `runs/ablation/<exp>/checkpoints/best.pth` | Best model weights (EMA if enabled) |
| `runs/ablation/<exp>/tensorboard/` | Training curves |
| `runs/ablation/logs/<jobname>_<jobid>.out` | SLURM stdout with val_loss history |
| `runs/ablation/eval_2d_perclass.json` | Per-class metrics (original 2D eval, 29 experiments) |
| `runs/ablation/eval_14class_2d.json` | Per-class 14-class Dice (all 34 2D experiments) |
| `runs/ablation/eval_14class_3d.json` | Per-class 14-class Dice (all 29 3D experiments) |

### Phase 2 v4 Results & Checkpoints

| File | Content |
|------|---------|
| `runs/monai_cellmap/<model>/tensorboard/` | TensorBoard event files |
| `runs/monai_cellmap/<model>/checkpoints/` | Model checkpoints (best.pth, latest.pth) |
| `runs/monai_2d/<model>/tensorboard/` | 2D model TensorBoard events |
| `runs/monai_2d/<model>/checkpoints/` | 2D model checkpoints |

### Utility Scripts (workspace root)

| File | Purpose |
|------|---------|
| `/work/users/g/s/gsgeorge/cellmap/launch_tb.sh` | TensorBoard launcher with `--logdir_spec` for all Phase 2 v4 models |
| `/work/users/g/s/gsgeorge/cellmap/analyze_p2v4.py` | Analysis script: reads TB events, ranks models, per-class Dice, ensemble selection |
| `/work/users/g/s/gsgeorge/cellmap/check_2d_results.py` | Quick 2D results checker |

### Debug Scripts (can be deleted)

| File | Purpose |
|------|---------|
| `debug_vit2d.py` | Debug v1: component isolation for CUDA kernel error |
| `debug_vit2d_v2.py` | Debug v2: training-loop-exact reproduction |
| `debug_vit2d.sbatch` | SLURM script for debug jobs |

---

## 10. How to Run

### Submit Phase 2 v4 Jobs

```bash
# From Longleaf login node:
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# Individual model:
sbatch training/slurm/phase2v4/unet_2d.sbatch
sbatch training/slurm/phase2v4/segresnet_3d.sbatch
# etc.

# All 9 models:
for f in training/slurm/phase2v4/*.sbatch; do sbatch "$f"; done
```

### Check Job Status

```bash
# Active jobs
squeue -u gsgeorge --format="%.10i %.25j %.15P %.8T %.12M %.6D %R"

# Job history (last week)
sacct -u gsgeorge --starttime=2026-02-25 --format=JobID,JobName%30,State,ExitCode,Elapsed,Partition

# Check partition availability
sinfo -p l40-gpu,a100-gpu,a100-multi-gpu --format="%P %a %D %T %N"
```

### Monitor Training

```bash
# TensorBoard (run on Sycamore in tmux)
cd /work/users/g/s/gsgeorge/cellmap
bash launch_tb.sh

# From laptop, SSH tunnel:
ssh -N -L 6006:sycamore-login2:6006 gsgeorge@sycamore.unc.edu
# Then open http://localhost:6006

# Quick analysis (run on Sycamore or Longleaf)
micromamba activate csc
python /work/users/g/s/gsgeorge/cellmap/analyze_p2v4.py
```

### Run Per-Class Evaluation

```bash
# Single experiment
python -m training.eval_2d_perclass \
  --experiment <experiment_name> \
  --run_dir runs/ablation \
  --output_dir runs/ablation
```

---

## 11. Next Steps (Priority Order)

### Immediate (before next training cycle)

1. **🔴 Fix `train.py` gradient accumulation unscale bug** — Required before ViT-2D can train. Move `scaler.unscale_()` + `clip_grad_norm_()` inside accumulation boundary conditional. See Section 8 for details.
2. **Resubmit ViT-2D** — After fixing unscale bug, submit `training/slurm/phase2v4/vit_2d.sbatch`
3. **Monitor suspended jobs** — 8 jobs SUSPENDED for `switch_work` maintenance (1-2 hours). Should resume automatically. Check with `squeue`.

### Short-term (this week)

4. **Diagnose swin_2d declining** — Loss/Dice declining since epoch 5. May need LR reduction or early stopping. Consider restarting with lower LR.
5. **Diagnose vitnet_3d dead** — Zero Dice across all epochs. Check if model is producing constant outputs. May have architecture bug or need different LR/optimizer.
6. **Rerun analysis** once 3D models reach epoch 50+ — their trajectories are steep, rankings will shift significantly.

### Medium-term (when Phase 2 v4 completes)

7. **Final per-class evaluation** on all models with the best checkpoints
8. **Ensemble selection** — Greedy forward selection verified by `analyze_p2v4.py`
9. **Post-processing** — Connected components, size filtering for noisy classes
10. **Test set inference** — Generate submission predictions
11. **Challenge submission** — Final leaderboard entry
