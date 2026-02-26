# CellMap Segmentation — Experiment Progress & Handoff Document

> **Last updated:** February 25, 2026  
> **Author:** AI Agent (GitHub Copilot, Claude Opus 4.6)  
> **Purpose:** Full context for any agent continuing this work

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Infrastructure](#2-infrastructure)
3. [Codebase Architecture](#3-codebase-architecture)
4. [Phase 1 Results — Complete](#4-phase-1-results--complete)
5. [Key Findings & Decisions](#5-key-findings--decisions)
6. [In-Flight Experiments](#6-in-flight-experiments)
7. [Phase 2 Plan — Architecture Comparison](#7-phase-2-plan--architecture-comparison)
8. [Known Issues & Caveats](#8-known-issues--caveats)
9. [File Reference](#9-file-reference)
10. [How to Run Experiments](#10-how-to-run-experiments)

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

| Cluster | Partition | GPU | VRAM | Host RAM | Account | Use |
|---------|-----------|-----|------|----------|---------|-----|
| Longleaf | `l40-gpu` | NVIDIA L40S | 48 GB | ~1 TB | `rc_cburch_pi` | 2D experiments |
| Sycamore | `h100_sn` | NVIDIA H100 | 80 GB | ~1 TB | `rc_alain_pi` | 3D experiments |
| Sycamore | `h100_mn` | NVIDIA H100 (multi) | 80 GB | ~1 TB | `rc_alain_pi` | ⚠️ Avoid — jobs get stuck |

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

| Experiment | Loss | Val Loss | Mean Dice | Rank |
|------------|------|----------|-----------|------|
| `loss_2d_bce` | BCE | **0.0425** | **0.479** | 🥇 |
| `loss_2d_dice_bce` | Dice+BCE | 0.4656 | 0.462 | 🥈 |
| `loss_2d_boundary_tversky` | Boundary Tversky | 0.6996 | 0.410 | 3 |
| `loss_2d_tversky` | Tversky | 0.7136 | 0.393 | 4 |
| `loss_2d_focal_tversky` | Focal Tversky | 0.5472 | 0.378 | 5 |
| `loss_2d_unified_focal` | Unified Focal | 0.5473 | 0.372 | 6 |
| `loss_2d_focal` | Focal | 0.5472 | 0.362 | 7 |
| `loss_2d_balanced_softmax_tversky` | BST (τ=1.0) | 0.6055 | 0.240 | 8 |

**Finding:** Simple losses (BCE, Dice+BCE) massively outperform complex losses. BST's τ=1.0 logit adjustment over-corrects with 48 classes, producing too many false positives on rare classes.

**Decision:** `dice_bce` chosen for Phase 2 (not BCE) because dice_bce has better rare-class performance despite slightly lower mean Dice. BCE achieves high mean Dice by overpredicting common classes.

### Sweep B: Tversky α/β (2D, ResNet)

| Experiment | α | β | Mean Dice |
|------------|---|---|-----------|
| `tversky_2d_balanced` | 0.5 | 0.5 | 0.408 |
| `tversky_2d_a08_b06` | 0.8 | 0.6 | 0.409 |
| `tversky_2d_recall` | 0.3 | 0.7 | 0.409 |
| `tversky_2d_precision_07_03` | 0.7 | 0.3 | 0.386 |
| `tversky_2d_a08_b04` | 0.8 | 0.4 | 0.371 |
| `tversky_2d_precision_06_04` | 0.6 | 0.4 | 0.374 |

**Finding:** All Tversky variants clustered around 0.37–0.41 mean Dice. No α/β combination came close to BCE (0.479) or dice_bce (0.462). Tversky-based losses are suboptimal for this task.

### Sweep C: Class Weighting τ (2D, ResNet, BST)

| Experiment | τ | Val Loss | Mean Dice |
|------------|---|----------|-----------|
| `tau_2d_20` | 2.0 | 0.4090 | 0.000 |
| `tau_2d_15` | 1.5 | 0.4094 | 0.000 |
| `tau_2d_10` | 1.0 | 0.5444 | 0.133 |
| `tau_2d_05` | 0.5 | 0.4252 | 0.408 |
| `tau_2d_0` | 0.0 | 0.4252 | 0.323 |

**Finding:** High τ completely destroys predictions (mean Dice = 0). τ=0.5 is best within BST family but still worse than dice_bce. Logit adjustment doesn't work well with partial annotations at 48 classes.

### Sweep D: Masking Strategy (2D, ResNet, BST)

| Experiment | Strategy | Mean Dice |
|------------|----------|-----------|
| `mask_2d_masksup03_no_bbox` | MaskSup λ=0.3, no bbox | 0.263 |
| `mask_2d_fg_only` | FG mask only | 0.254 |
| `mask_2d_none` | No masking | 0.250 |
| `mask_2d_bbox_loose` | Loose bbox + FG | 0.240 |
| `mask_2d_masksup03` | MaskSup λ=0.3 + bbox | 0.136 |
| `mask_2d_bbox_only` | Bbox only | 0.134 |
| `mask_2d_bbox_fg` | Bbox + FG | 0.133 |

**Finding:** FG masking provides modest improvement. Bbox masking hurts when combined with BST (bbox over-constrains already over-adjusted logits). Best strategy is simple FG mask.

### Sweep E: Training Techniques (2D, ResNet)

**Original (with BST base loss):**

| Experiment | Technique | Val Loss | Mean Dice |
|------------|-----------|----------|-----------|
| `tech_2d_ema` | EMA (decay=0.999) | 0.5603 | 0.102 |
| `tech_2d_no_weighted_sampler` | No sampler | 0.5645 | 0.082 |
| `tech_2d_focal_tversky_mild` | Focal γ=0.5 | 0.5709 | 0.325 |

**Re-validated with dice_bce (winning loss):**

| Experiment | Config | Val Loss | Notes |
|------------|--------|----------|-------|
| `tech_2d_dicebce_ema` | dice_bce + EMA + fg_mask + sampler | **0.112** | ⭐ **4× improvement over no-EMA (0.466)** |
| `tech_2d_dicebce_no_sampler` | dice_bce + EMA + fg_mask, no sampler | **0.122** | Sampler helps ~9% |

**Finding:** EMA is the single biggest improvement discovered. It smooths noisy gradients from the imbalanced 48-class partial annotation setup. Val loss dropped from 0.466 → 0.112 (4× better). Weighted sampler provides modest additional benefit (0.112 vs 0.122).

### 3D Results (SegResNet baseline)

| Experiment | Val Loss | Notes |
|------------|----------|-------|
| `loss_3d_dice_bce` | **0.170** | 🥇 Best 3D loss |
| `loss_3d_bce` | 0.220 | Runner-up |
| `mask_3d_bbox_only` | 0.541 | |
| `tech_3d_deep_supervision` | 0.556 | Deep supervision helps SegResNet |
| `tau_3d_20` | 0.597 | Partial result (was still running) |
| `loss_3d_unified_focal` | 0.667 | |
| `tech_3d_ema` | 0.691 | EMA with BST (not re-validated with dice_bce) |
| `loss_3d_balanced_softmax_tversky` | 0.695 | BST fails in 3D too |
| `mask_3d_bbox_fg` | 0.695 | |
| `mask_3d_bbox_loose` | 0.695 | |
| `loss_3d_boundary_tversky` | 0.720 | |
| `loss_3d_focal` | 0.723 | |
| `tech_3d_focal_tversky_mild` | 0.722 | |
| `tversky_3d_a08_b06` | 0.730 | |
| `tech_3d_no_weighted_sampler` | 0.736 | |
| `loss_3d_tversky` | 0.740 | |
| `tversky_3d_precision_06_04` | 0.740 | |
| `tversky_3d_precision_07_03` | 0.747 | |
| `mask_3d_masksup03` | 0.749 | |
| `mask_3d_masksup03_no_bbox` | 0.749 | |
| `tau_3d_0` | 0.777 | |
| `tau_3d_05` | 0.777 | |
| `tversky_3d_balanced` | 0.804 | |
| `tversky_3d_recall` | 0.814 | |
| `tversky_3d_a08_b04` | 0.919 | |
| `mask_3d_fg_only` | PENDING | Job 1820948 still running |
| `mask_3d_none` | PENDING | Job 1820947 still running |

**Finding:** dice_bce wins in 3D too (0.170 vs next best 0.220). Same pattern as 2D — simple losses outperform complex ones. Deep supervision provides significant benefit for SegResNet (0.556 vs 0.695 for BST).

---

## 5. Key Findings & Decisions

### Optimal Phase 2 Configuration

```
Loss:             dice_bce (bce_weight=0.5, smooth=1e-6)
EMA:              enabled, decay=0.999
FG Mask:          enabled
Weighted Sampler: enabled (cellmap-data default)
Intensity Aug:    TBD (waiting on val_intensity_aug results)
Class-Aware:      TBD (waiting on val_crop_weights results)
AMP:              enabled
Scheduler:        cosine with 5-epoch warmup
Optimizer:        RAdam, lr=1e-4
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

## 6. In-Flight Experiments

### Validation: Data Loading Improvements (Longleaf L40S)

These validate features cherry-picked from the OrganelleSeg repo (coworker Greg's pipeline):

| Job ID | Name | Config | Status | Expected Finish |
|--------|------|--------|--------|-----------------|
| 33019765 | `val_intensity_aug` | dice_bce + EMA + fg_mask + **intensity aug** | RUNNING (~1h in) | ~4-5h from submission |
| 33019782 | `val_crop_weights` | dice_bce + EMA + fg_mask + **class-aware sampling** | RUNNING (~1h in) | ~4-5h from submission |
| 33019784 | `val_combined` | dice_bce + EMA + fg_mask + **both** | RUNNING (~1h in) | ~4-5h from submission |

**Baseline:** `tech_2d_dicebce_ema` → val_loss = 0.112

**Intensity augmentation** (`--intensity_aug`):
- RandomBrightness ±0.1, RandomContrast 0.8–1.2, RandomGaussianNoise σ=0.01–0.05
- Applied to raw EM inputs via `train_raw_value_transforms` (train only, not val)
- Rationale: EM data has natural intensity variation across sections; augmentation should improve generalization

**Class-aware crop weighting** (`--class_aware_sampling`):
- Replaces default weighted sampler with inverse-sqrt class-aware weighting
- `weight(crop) = 0.7 × mean(1/√(global_count(c))) + 0.3 × uniform`
- Rationale: upweight crops containing rare organelles (NE, peroxisomes, MT-inner)

### Remaining 3D Ablations (Sycamore H100)

| Job ID | Name | Status | Expected Finish |
|--------|------|--------|-----------------|
| 1820946 | `tau_3d_20` | RUNNING (~9h in) | ~3-6h from now |
| 1820947 | `mask_3d_none` | RUNNING (~9h in) | ~3-6h from now |
| 1820948 | `mask_3d_fg_only` | RUNNING (~9h in) | ~3-6h from now |

---

## 7. Phase 2 Plan — Architecture Comparison

### Configuration

Once validation experiments complete, finalize the Phase 2 base config:

```python
# In training/configs/experiments.py
loss = "dice_bce"
use_foreground_mask = True
ema = True
ema_decay = 0.999
epochs = 100              # 2× ablation
iterations_per_epoch = 1000  # 2× ablation
val_every_n_epochs = 5
intensity_aug = TBD       # Include if val_intensity_aug ≤ 0.112
class_aware_sampling = TBD  # Include if val_crop_weights ≤ 0.112
```

### ⚠️ IMPORTANT: Update Phase 2 Defaults Before Launching

The `make_arch_comparison_2d()` and `make_arch_comparison_3d()` functions in `training/configs/experiments.py` still have `loss="balanced_softmax_tversky"` as default. **These MUST be updated to `loss="dice_bce"` before launching Phase 2.**

Similarly, `training/slurm/launch_arch_comparison.sh` has `BEST_LOSS="balanced_softmax_tversky"` — also needs updating.

### 2D Runs (Longleaf L40S)

| Model | Est. Time | Input Shape | Params |
|-------|-----------|-------------|--------|
| `resnet_2d` (FlexUNet-ResNet34) | ~6h | [1, 256, 256] | ~7.8M |
| `unet_2d` | ~5h | [1, 256, 256] | ~31M |
| `swin_2d` (SwinTransformer) | ~10h | [1, 256, 256] | ~36M |
| `vit_2d` (ViTVNet) | ~10h | [1, 256, 256] | ~105M |

### 3D Runs (Sycamore H100 — `h100_sn` ONLY)

| Model | Est. Time | Input Shape | Batch | Params |
|-------|-----------|-------------|-------|--------|
| `segresnet_3d` (SegResNetDS) | ~24h | [128, 128, 128] | 2 | ~18M |
| `swinunetr_3d` (SwinUNETR) | ~36h | [128, 128, 128] | 2 | ~62M |
| `unet_3d` | ~18h | [128, 128, 128] | 2 | - |
| `resnet_3d` | ~20h | [128, 128, 128] | 2 | - |

### Launch Commands

```bash
# 2D on Longleaf (repeat for each model: resnet_2d, unet_2d, swin_2d, vit_2d)
ssh longleaf.unc.edu 'cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation && \
  EXPERIMENT_NAME=arch_2d_resnet MODEL_NAME=resnet_2d LOSS_NAME=dice_bce \
  USE_FG_MASK=true EPOCHS=100 ITERS=1000 \
  EXTRA_ARGS="--ema --ema_decay 0.999 --val_every_n_epochs 5" \
  sbatch --export=ALL --job-name=arch_2d_resnet training/slurm/ablation_2d_l40s.sbatch'

# 3D on Sycamore (repeat for each model: segresnet_3d, swinunetr_3d, unet_3d, resnet_3d)
ssh sycamore 'cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation && \
  EXPERIMENT_NAME=arch_3d_segresnet MODEL_NAME=segresnet_3d LOSS_NAME=dice_bce \
  USE_FG_MASK=true EPOCHS=100 ITERS=500 BATCH_SIZE=2 \
  EXTRA_ARGS="--ema --ema_decay 0.999 --val_every_n_epochs 5 --input_shape 128 128 128" \
  sbatch --export=ALL --job-name=arch_3d_segresnet \
  --partition=h100_sn --account=rc_alain_pi \
  training/slurm/ablation_3d_h100.sbatch'
```

Add `--intensity_aug` and/or `--class_aware_sampling` to `EXTRA_ARGS` if validation results are positive.

---

## 8. Known Issues & Caveats

### Validation Set Limitation
The validation crops only annotate **14 of 48 classes**: ecs, pm, mito_mem, mito_lum, mito_ribo, golgi_mem, golgi_lum, ves_mem, ves_lum, endo_mem, endo_lum, er_mem, er_lum, nuc. The remaining 34 classes (including all group classes) cannot be evaluated during training. This means:
- `val_loss` only measures performance on 14 classes
- Per-class Dice is missing for 34 classes
- The true challenge leaderboard score could differ significantly from val metrics

### Sycamore h100_mn Partition
Jobs on `h100_mn` (multi-node) partition tend to get stuck in PENDING indefinitely. Always use `h100_sn` (single-node) for Sycamore jobs.

### AMP Overflow with BST
BST with AMP float16 can produce inf logits due to the τ×log(π_c) adjustment for very rare classes. This was fixed by adding `nan_to_num()` clamping, but is irrelevant now that we've moved to dice_bce.

### cellmap-data EmptyImage Memory
cellmap-data pre-allocates `EmptyImage` tensors for all ~784 datasets on initialization. This requires ~300GB host RAM. Jobs must request ≥384G memory. Data should be loaded on CPU (`device="cpu"` in get_dataloader) and batches moved to GPU in the training loop.

### Git State
As of commit `e5d9f92` (Feb 25, 2026):
- All Phase 1 infrastructure is committed
- Intensity augmentation and class-aware sampling are committed
- 3 validation experiment configs are committed
- Phase 2 defaults still reference BST (need updating before Phase 2 launch)

---

## 9. File Reference

### Key Files to Read

| File | Purpose |
|------|---------|
| `training/train.py` | Main training loop — understand all flags and data flow |
| `training/configs/experiments.py` | All experiment configs — modify for Phase 2 |
| `training/losses/loss_zoo.py` | Loss function registry — `dice_bce` is the winner |
| `training/models/model_zoo.py` | Model registry — 9 architectures |
| `training/eval_2d_perclass.py` | Per-class evaluation — run after training |
| `training/transforms/intensity.py` | Intensity augmentation implementation |
| `training/samplers/crop_weights.py` | Class-aware sampling implementation |
| `src/cellmap_segmentation_challenge/utils/dataloader.py` | `get_dataloader()` wrapper |
| `runs/ablation/eval_2d_perclass.json` | Full per-class Dice/IoU results for all 2D experiments |

### Key Results Files

| File | Content |
|------|---------|
| `runs/ablation/<exp>/config.json` | Hyperparameters used |
| `runs/ablation/<exp>/checkpoints/best.pth` | Best model weights (EMA if enabled) |
| `runs/ablation/<exp>/tensorboard/` | Training curves |
| `runs/ablation/logs/<jobname>_<jobid>.out` | SLURM stdout with val_loss history |
| `runs/ablation/eval_2d_perclass.json` | Per-class metrics (2D Phase 1 only) |

---

## 10. How to Run Experiments

### Submit a Single Experiment

```bash
# Via SSH to Longleaf (2D)
ssh longleaf.unc.edu 'cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation && \
  EXPERIMENT_NAME=<name> MODEL_NAME=<model> LOSS_NAME=<loss> \
  USE_FG_MASK=true EXTRA_ARGS="--ema --ema_decay 0.999" \
  sbatch --export=ALL --job-name=<name> training/slurm/ablation_2d_l40s.sbatch'

# Via SSH to Sycamore (3D) — ALWAYS use h100_sn
ssh sycamore 'cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation && \
  EXPERIMENT_NAME=<name> MODEL_NAME=<model> LOSS_NAME=<loss> \
  USE_FG_MASK=true BATCH_SIZE=2 \
  EXTRA_ARGS="--ema --ema_decay 0.999 --input_shape 128 128 128" \
  sbatch --export=ALL --job-name=<name> \
  --partition=h100_sn --account=rc_alain_pi \
  training/slurm/ablation_3d_h100.sbatch'
```

### Check Job Status

```bash
# Longleaf
ssh longleaf.unc.edu 'squeue -u gsgeorge --format="%.10i %.20j %.10P %.8T %.12M %.6D %R"'
# Sycamore
ssh sycamore 'squeue -u gsgeorge --format="%.10i %.20j %.10P %.8T %.12M %.6D %R"'
# Completed job history
ssh longleaf.unc.edu 'sacct -u gsgeorge --starttime=2026-02-23 --format=JobID,JobName%30,State,ExitCode,Elapsed'
```

### Read Results

```bash
# Last lines of job output (includes "Best val loss: X.XXXX")
ssh longleaf.unc.edu 'tail -5 /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/runs/ablation/logs/<jobname>_<jobid>.out'

# Grep best val loss from all log files
for f in runs/ablation/logs/*.out; do
  name=$(basename "$f" | sed 's/_[0-9]*.out//');
  best=$(grep -o "Best val loss: [0-9.]*" "$f" | tail -1 | awk '{print $NF}');
  echo "$name: $best";
done | sort -t: -k2 -n

# TensorBoard
tensorboard --logdir runs/ablation/<experiment_name>/tensorboard
```

### Run Per-Class Evaluation

```bash
# Single experiment
python -m training.eval_2d_perclass \
  --experiment <experiment_name> \
  --run_dir runs/ablation \
  --output_dir runs/ablation

# Results written to runs/ablation/eval_2d_perclass.json and .csv
```

---

## Next Steps (for continuing agent)

1. **Check validation results** — when jobs 33019765, 33019782, 33019784 complete:
   - Read results: `tail -20 runs/ablation/logs/val_*_<jobid>.out`
   - Compare val_loss to baseline 0.112 (`tech_2d_dicebce_ema`)
   - If improvement: include in Phase 2 config's `EXTRA_ARGS`
   - If worse: exclude from Phase 2 config

2. **Check remaining 3D results** — when jobs 1820946, 1820947, 1820948 complete:
   - These complete the 3D masking/weighting ablation picture
   - Key question: does fg_mask help in 3D? (It helped modestly in 2D)

3. **Update Phase 2 defaults** — in `training/configs/experiments.py`:
   - Change `make_arch_comparison_2d()` default loss from `"balanced_softmax_tversky"` to `"dice_bce"`
   - Change `make_arch_comparison_3d()` default loss similarly
   - Add `ema=True, ema_decay=0.999` to both functions
   - Add `use_foreground_mask=True` to both functions
   - Optionally add `intensity_aug` and `class_aware_sampling` based on validation results
   - Update `training/slurm/launch_arch_comparison.sh` `BEST_LOSS` variable

4. **Launch Phase 2** — 8 architecture comparison experiments:
   - 4 × 2D on Longleaf L40S (resnet_2d, unet_2d, swin_2d, vit_2d)
   - 4 × 3D on Sycamore H100 `h100_sn` (segresnet_3d, swinunetr_3d, unet_3d, resnet_3d)
   - Use 100 epochs, 1000 iters/epoch (2D) or 500 iters/epoch (3D)

5. **Run per-class evaluation** on Phase 2 results to determine final model selection

6. **Consider 3D EMA re-validation** — EMA was only validated with dice_bce in 2D. The 3D EMA result (0.691) used BST. Consider running `dice_bce + EMA` in 3D before Phase 2 3D launches, or just include EMA by default given the massive 2D improvement.

7. **Consider deep supervision for 3D** — `tech_3d_deep_supervision` (val_loss=0.556) was much better than the BST baseline (0.695). With dice_bce + EMA, deep supervision may stack further improvements for SegResNet. Add `--deep_supervision` to SegResNet 3D Phase 2 run.
