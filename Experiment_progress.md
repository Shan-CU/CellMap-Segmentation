WH# CellMap Segmentation — Experiment Progress & Handoff Document

> **Last updated:** February 26, 2026  
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

## 6. In-Flight Experiments

### Validation: Data Loading Improvements (Longleaf L40S)

These validate features cherry-picked from the OrganelleSeg repo (coworker Greg's pipeline):

| Job ID | Name | Config | Status | Expected Finish |
|--------|------|--------|--------|-----------------|
| 33019765 | `val_intensity_aug` | dice_bce + EMA + fg_mask + **intensity aug** | RUNNING (~1h in) | ~4-5h from submission |
| 33019782 | `val_crop_weights` | dice_bce + EMA + fg_mask + **class-aware sampling** | RUNNING (~1h in) | ~4-5h from submission |
| 33019784 | `val_combined` | dice_bce + EMA + fg_mask + **both** | RUNNING (~1h in) | ~4-5h from submission |

**Baseline:** `tech_2d_dicebce_ema` → val_loss = 0.112

### Validation Results — ❌ All HURT Performance

| Experiment | Extra Feature | Val Loss | Mean Dice (14-class) | Verdict |
|------------|--------------|----------|---------------------|---------|
| `val_intensity_aug` | intensity augmentation | 0.1349 | 0.3539 | ❌ −23% |
| `val_crop_weights` | class-aware crop weighting | 0.1228 | 0.2849 | ❌ −38% |
| `val_combined` | both | 0.1350 | 0.2133 | ❌ −54% |

**Conclusion:** Neither feature helps. Both are excluded from Phase 2.

### All 3D Ablations — ✅ Complete

All 29 3D experiments are complete. See Section 4 for full Dice eval results.
3D results were essentially non-functional (max Dice = 0.017) due to insufficient training
(50 epochs × 250 iters = 12,500 steps for 128³ volumes).

---

## 7. Phase 2 Plan — Architecture Comparison

### Configuration ✅ FINALIZED

```python
# Phase 2 recipe (from Phase 1 ablation winner)
loss = "dice_bce"           # SegResNet too (with deep_supervision)
use_foreground_mask = True
ema = True
ema_decay = 0.999
weighted_sampler = True     # cellmap-data default
intensity_aug = False        # DISABLED — hurt −23%
class_aware_sampling = False # DISABLED — hurt −38%

# 2D settings
epochs_2d = 100
iterations_per_epoch_2d = 1000
val_every_n_epochs_2d = 5
warmup_epochs_2d = 10
batch_size_2d = 8            # (ViT uses 4 due to CUDA kernel bug)
lr_2d = 1e-4

# 3D settings
epochs_3d = 1000
iterations_per_epoch_3d = 300
val_every_n_epochs_3d = 5
warmup_epochs_3d = 34
batch_size_3d = 8
lr_3d = 4e-4                 # linearly scaled from 1e-4 @ batch=2
input_shape_3d = [96, 96, 96]  # 768nm box, 152 datasets (vs 78 at 128^3)
```

### ⚠️ Phase 2 Launch History

**v1 launch (Feb 27):** Missing `--ema` flag (sbatch had `--ema_decay 0.999` but not `--ema`).
SegResNet used BST instead of dice_bce. ViT crashed (CUDA kernel error at batch=8).
All 8 jobs killed and relaunched as v2.

**v2 launch (Feb 28):** All fixes applied:
- EMA enabled (`--ema --ema_decay 0.999` baked into both sbatch files)
- SegResNet switched to `dice_bce + --deep_supervision`
- ViT batch reduced to 4 (CUDA kernel launch bug with BatchNorm2d + AMP at batch=8)
- All 9 models on Longleaf L40S single GPU

### 2D Runs (Longleaf L40S, 100ep × 1000it)

| Job Name | Model | Batch | Params | Loss |
|----------|-------|-------|--------|------|
| `p2_resnet_2d` | FlexUNet-ResNet34 | 8 | ~7.8M | dice_bce |
| `p2_unet_2d` | CSC UNet 2D | 8 | ~31M | dice_bce |
| `p2_swin_2d` | SwinTransformer 2D | 8 | ~36M | dice_bce |
| `p2_vit_2d` | ViTVNet 2D | **4** | ~105M | dice_bce |

### 3D Runs (Longleaf L40S, 1000ep × 300it, 96³ crops)

| Job Name | Model | Batch | Params | Loss | Extra |
|----------|-------|-------|--------|------|-------|
| `p2_segresnet_3d` | SegResNetDS | 8 | ~20M | **dice_bce** | `--deep_supervision` |
| `p2_swinunetr_3d` | SwinUNETR | 8 | ~62M | dice_bce | |
| `p2_unet_3d` | MONAI UNet 3D | 8 | ~90M | dice_bce | |
| `p2_resnet_3d` | MONAI ResNet 3D | 8 | ~24M | dice_bce | |
| `p2_vitnet_3d` | MONAI ViTAutoEnc | 8 | ~32M | dice_bce | |

### Launch Command

```bash
# From longleaf login node:
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
bash training/slurm/launch_phase2_clean.sh        # all 9 jobs
bash training/slurm/launch_phase2_clean.sh --2d-only  # 4 × 2D only
bash training/slurm/launch_phase2_clean.sh --3d-only  # 5 × 3D only
```

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
As of Phase 2 v2 relaunch (Feb 28, 2026):
- All Phase 1 infrastructure committed
- Memory leak fix committed (2c4aa84): cellmap-data PR#64 + monkey-patch + MALLOC_ARENA_MAX=2
- Phase 2 sbatch files have `--ema --ema_decay 0.999` baked in
- SegResNet uses `dice_bce + --deep_supervision` (not BST)
- ViT batch=4 fix in launch script
- Launch script: `training/slurm/launch_phase2_clean.sh`

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
| `runs/ablation/eval_2d_perclass.json` | Per-class metrics (original 2D eval, 29 experiments) |
| `runs/ablation/eval_14class_2d.json` | Per-class 14-class Dice (all 34 2D experiments) |
| `runs/ablation/eval_14class_3d.json` | Per-class 14-class Dice (all 29 3D experiments) |
| `runs/ablation/eval_all_perclass_2d.csv` | 2D leaderboard CSV |
| `runs/ablation/eval_all_perclass_3d.csv` | 3D leaderboard CSV |

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

1. ✅ ~~Check validation results~~ Done — intensity_aug (−23%), crop_weights (−38%), combined (−54%). All hurt. Excluded.
2. ✅ ~~Check remaining 3D results~~ Done — All 29 3D experiments complete. 3D ablation was non-functional (max Dice 0.017).
3. ✅ ~~Fix Phase 2 launch~~ Done — EMA enabled, SegResNet on dice_bce+DS, ViT batch=4.
4. **Monitor Phase 2 training** — 9 jobs running on Longleaf L40S:
   - 2D first val: ~epoch 10 (~10h), 2D done: ~100h
   - 3D first val: ~epoch 30 (~48h), 3D done: ~700h (~29 days)
5. **Run per-class evaluation** on Phase 2 results to determine final model selection
6. **Final submission preparation** — ensemble, post-processing, test set inference
