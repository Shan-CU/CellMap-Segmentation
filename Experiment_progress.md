# CellMap Segmentation — Experiment Progress

> **Last updated:** 2026-02-23
> **Status:** Phase 1 ablation running — 2D complete, 3D in progress

---

## 1. Project Overview

We are competing in the **CellMap Segmentation Challenge (CSC)**, a multi-label volumetric segmentation benchmark for FIB-SEM electron microscopy. The goal is to segment **48 organelle classes** (derived from 35 atomic label classes) across 22 diverse biological samples.

**Key challenge properties:**
- **Partial annotations**: Each crop only labels a subset of classes. Unlabeled classes appear as NaN. A loss function must not penalize unannotated classes.
- **Extreme class imbalance**: `ecs` (extracellular space) and `pm` (plasma membrane) dominate. Rare classes like `mt_out` (microtubule outer) or `ld_mem` (lipid droplet membrane) may appear in only a few crops.
- **Multi-resolution zarr**: Raw EM data is stored as multi-scale zarr at resolutions from 4nm to 128nm. We train at 8nm isotropic.
- **289 annotated crops** across 22 biological volumes, totaling ~42.3 billion voxels.
- **Evaluation**: Per-class IoU on a held-out test set (48 classes tested on the leaderboard).

---

## 2. Architecture: NIfTI→Zarr Pipeline Overhaul

### 2.1 What Changed

We replaced the original NIfTI-based MONAI data pipeline with a **native zarr pipeline** using the official `cellmap-data` library (v2026.2.20). This was a complete rewrite of the training infrastructure.

**Old pipeline (broken):**
- Converted zarr crops → NIfTI files (lossy, slow, ~200GB disk)
- Used MONAI's `CacheDataset` + `RandCropByPosNegLabel`
- Only supported a subset of classes
- Scale/resolution handling was incorrect for some volumes

**New pipeline (current):**
- Reads zarr directly via `cellmap_data.CellMapDataLoader`
- Correct multi-resolution handling (8nm isotropic target)
- All 35 atomic classes + 48 tested classes
- `CellMapDataSplit` + `CellMapMultiDataset` with `weighted_sampler` for rare-class upsampling
- `EmptyImage` placeholder tensors for classes not present in a given crop (value=-100, treated as NaN → annotation mask)

### 2.2 Key Files

| File | Purpose |
|------|---------|
| `training/train.py` | Main training script (~600 lines). Handles data loading, training loop, validation, checkpointing. |
| `training/losses/partial_annotation.py` | `PartialTverskyLoss` and `BalancedSoftmaxTverskyLoss` — our core losses |
| `training/losses/focal_tversky.py` | `FocalTverskyLoss` and `AsymmetricUnifiedFocalLoss` |
| `training/losses/boundary_loss.py` | `BoundaryWeightedTverskyLoss` — upweights near membranes |
| `training/losses/loss_zoo.py` | Registry of 22 loss configurations (builders + names) |
| `training/models/model_zoo.py` | Registry of 8 models (4 × 2D, 4 × 3D) |
| `training/configs/experiments.py` | All 59 experiment definitions as `ExperimentConfig` dataclasses |
| `training/slurm/ablation_2d_h100.sbatch` | SLURM job template for 2D experiments |
| `training/slurm/ablation_3d_h100.sbatch` | SLURM job template for 3D experiments |
| `training/slurm/launch_ablation_2d_h100.sh` | Launches all 29 2D experiments |
| `training/slurm/launch_ablation_3d_h100.sh` | Launches all 30 3D experiments |
| `datasplit.csv` | Train/val split (1156 entries — each row is a crop×class group) |

### 2.3 Data Flow

```
datasplit.csv
    ↓
cellmap_data.CellMapDataLoader(
    datasplit_path, classes, input_array_info, target_array_info,
    spatial_transforms, weighted_sampler, device="cpu"
)
    ↓
CellMapMultiDataset → per-crop CellMapDataset
    → Each dataset has input_arrays (EM) + target_arrays (per-class labels)
    → Missing classes → EmptyImage(value=-100, shape=patch_shape)
    ↓
DataLoader(num_workers=8, batch_size=B)
    ↓
Training loop: batch[input_key].to("cuda"), batch[target_key].to("cuda")
    → NaN in targets → annotation_mask (B, C) — 1=annotated, 0=skip
    → targets.nan_to_num(0.0) → clean targets for loss
    → loss_fn.set_annotation_mask(mask); loss_fn(logits, targets_clean)
```

### 2.4 The `device="cpu"` Fix

`cellmap_data` creates `EmptyImage` objects for every (dataset × missing_class) combination. Each pre-allocates a full-size tensor (`torch.ones(patch_shape) * -100`). With ~784 training datasets and 48 classes, most slots are empty, creating ~41,000 `EmptyImage` tensors.

- **2D** (1×256×256): 0.25 MB each → ~10 GB total → fits in 384 GB
- **3D** (128×128×128): 8 MB each → ~321 GB total → needs 512 GB

We pass `device="cpu"` to `get_dataloader()` so these tensors stay on host RAM (not GPU VRAM). The training loop moves each batch to GPU individually.

---

## 3. Loss Functions

### 3.1 Partial Annotation Handling

All our custom losses support **partial annotation masking**:

1. `cellmap_data` returns NaN for unannotated classes
2. `get_annotation_mask_from_targets(targets)` → `(B, C)` binary mask (1 if channel has any non-NaN)
3. `loss_fn.set_annotation_mask(mask)` → loss is computed only on annotated channels
4. For losses that don't support explicit masks (BCE, Focal, Dice+BCE), we use CSC-style NaN masking: `(loss * ~isnan).sum() / ~isnan.sum()`

### 3.2 Loss Registry

| Name | Type | Description |
|------|------|-------------|
| `bce` | Baseline | `BCEWithLogitsLoss` (CSC default) |
| `focal` | Baseline | Focal loss (γ=2.0) — down-weights easy examples |
| `dice_bce` | Baseline | 50/50 Dice + BCE combination |
| `tversky` | Core | Per-channel Tversky (α=0.6, β=0.4) with annotation masking |
| `balanced_softmax_tversky` | **Our Best** | Logit-adjusted Tversky + bbox masking + foreground masking |
| `focal_tversky` | Advanced | `(1 - Tversky)^γ` (γ=0.75) — focal on hard classes |
| `focal_tversky_g05` | Advanced | Same with mild γ=0.5 |
| `unified_focal` | Advanced | Asymmetric Unified Focal (Yeung 2022) — Focal Tversky + Dice Focal |
| `boundary_tversky` | Advanced | Tversky + Gaussian boundary upweighting (σ=3, weight=5×) |
| `tversky_balanced` | α/β variant | α=0.5, β=0.5 (= Dice) |
| `tversky_precision` | α/β variant | α=0.7, β=0.3 (precision bias) |
| `tversky_recall` | α/β variant | α=0.3, β=0.7 (recall bias) |
| `tversky_a08_b04` | α/β variant | α=0.8, β=0.4 (strong precision) |
| `tversky_a08_b06` | α/β variant | α=0.8, β=0.6 (precision + high FN penalty) |
| `bst_tau0` | τ variant | BalancedSoftmaxTversky with τ=0 (no adjustment) |
| `bst_tau05` | τ variant | τ=0.5 |
| `bst_tau15` | τ variant | τ=1.5 |
| `bst_tau20` | τ variant | τ=2.0 (strong) |
| `bst_no_bbox` | Mask variant | No bbox masking (bbox_bg_weight=1.0) |
| `bst_bbox_loose` | Mask variant | Loose bbox (pad=0.2, bg_weight=0.1) |
| `bst_masksup03` | Mask variant | Mask-supervised reconstruction (ratio=0.3) |
| `bst_masksup03_no_bbox` | Mask variant | masksup=0.3 but no bbox |

### 3.3 BalancedSoftmaxTverskyLoss (our flagship)

This is our most sophisticated loss. It combines:

1. **Logit adjustment** (Balanced Softmax): `adjusted_logit_c = logit_c − τ·(log(n_c) − mean(log(n)))` where `n_c` is accumulated foreground count per class. This shifts the decision boundary to compensate for class imbalance. `τ` controls strength (0 = off, 2.0 = very strong).

2. **Per-channel Tversky** (α=0.6, β=0.4): Penalizes FP more than FN to boost precision.

3. **Spatial bounding-box masking**: For each annotated class, computes the tight bounding box of foreground voxels, pads by `bbox_pad_fraction`, assigns weight=1.0 inside and `bbox_bg_weight` (0.05) outside. This prevents false-positive penalty on background regions far from any annotated object.

4. **Foreground masking**: Zeros loss on black padding regions (EM intensity < 0.01 threshold).

5. **Mask-supervised reconstruction** (optional): Randomly masks `masksup_ratio` of annotated voxels and adds a reconstruction Tversky loss — forces the model to predict masked-out annotations from context.

### 3.4 AMP Float16 NaN Fix (Resolved)

`FocalTverskyLoss` and `AsymmetricUnifiedFocalLoss` compute `(1 - tversky)^γ` where γ is fractional (0.75). Two interacting bugs caused complete model corruption:

1. **Forward pass NaN**: Under AMP float16, the Tversky index can slightly exceed 1.0, making the base of `pow()` negative → NaN.
2. **Backward pass gradient explosion**: The derivative of $x^\gamma$ is $\gamma \cdot x^{\gamma-1}$, which → ∞ as $x → 0$ when $\gamma < 1$. In float16 this becomes inf, poisoning all model weights with NaN within ~7 epochs.
3. **Silent 0.0 "best" overwrite**: When all val batches produce NaN, `val_steps=0` → `avg_val_loss = 0/1 = 0.0`, which beat any real val loss, so a 100%-NaN model was saved as `best.pth`.

**Fix (3 parts):**
- `focal_tversky.py`: Cast `input` to **float32** before `sigmoid()` (exits AMP autocast), add **`eps=1e-6`** lower clamp: `.clamp(min=eps, max=1.0).pow(gamma)` — bounds the backward gradient at $\gamma \cdot \varepsilon^{\gamma-1}$.
- `focal_tversky.py`: Same fix for `AsymmetricUnifiedFocalLoss`.
- `train.py`: Guard best-model save with `val_steps > 0` — 0-batch validation cannot overwrite a real checkpoint.

**NaN-skip guards** (safety net, should no longer trigger):
- Training loop: `if not torch.isfinite(loss): optimizer.zero_grad(); continue`
- Validation loop: `if torch.isfinite(vloss): val_loss += vloss.item(); val_steps += 1`

All 6 affected experiments (3 2D + 3 3D) were relaunched with the fix as jobs 1793665–1793670.

---

## 4. Models

### 4.1 Model Registry

| Name | Dim | Params | Source | Description |
|------|-----|--------|--------|-------------|
| `resnet_2d` | 2D | 7.8M | CSC | ResNet with 6 blocks, 2 downsampling layers, ngf=64 |
| `unet_2d` | 2D | 31M | CSC | Standard U-Net |
| `swin_2d` | 2D | 36M | CSC | Swin Transformer |
| `vit_2d` | 2D | 105M | CSC | ViT-VNet |
| `segresnet_3d` | 3D | 20M | MONAI | SegResNetDS (blocks_down=[1,2,2,4]) |
| `swinunetr_3d` | 3D | 62M | MONAI | SwinUNETR |
| `unet_3d` | 3D | — | CSC | UNet 3D |
| `resnet_3d` | 3D | — | CSC | ResNet 3D |

### 4.2 Phase 1 Baselines

- **2D baseline**: `resnet_2d` (7.8M params) — fast, input_shape=[1, 256, 256], batch_size=8
- **3D baseline**: `segresnet_3d` (20M params) — MONAI SegResNetDS, input_shape=[128, 128, 128], batch_size=1

---

## 5. Experiment Design

### 5.1 Ablation Structure

**59 experiments** across **5 sweeps**, each run in **2D and 3D**:

| Sweep | What it tests | # Experiments | 2D model | 3D model |
|-------|---------------|---------------|----------|----------|
| **A: Loss Function** | Which base loss performs best | 8 | resnet_2d | segresnet_3d |
| **B: Tversky α/β** | Precision-recall tradeoff | 6 | resnet_2d | segresnet_3d |
| **C: Class Weighting (τ)** | How much logit adjustment helps | 5 | resnet_2d | segresnet_3d |
| **D: Masking Strategy** | Spatial masking configurations | 7 | resnet_2d | segresnet_3d |
| **E: Training Techniques** | EMA, sampler, deep supervision | 3 (2D) + 4 (3D) | resnet_2d | segresnet_3d |

All experiments hold constant: 50 epochs, 500 iters/epoch (2D) / 500 iters/epoch (3D sbatch override), lr=1e-4, RAdam optimizer, cosine LR with 5-epoch warmup, AMP float16, grad clipping 1.0, seed 42.

### 5.2 How Sweeps Compose

The sweeps are designed to be **composable** — each tests one axis independently:

1. **Sweep A** → picks the best **base loss** (e.g., `balanced_softmax_tversky` vs `focal_tversky`)
2. **Sweep B** → picks the best **α/β** for Tversky-family losses
3. **Sweep C** → picks the best **τ** for logit adjustment
4. **Sweep D** → picks the best **masking** strategy (bbox, foreground, masksup)
5. **Sweep E** → picks which **training techniques** help (EMA, sampler, deep supervision)

The winning combination from each sweep becomes the **Phase 2 baseline** for architecture comparison.

Example composition: If A→`balanced_softmax_tversky`, B→`α=0.8,β=0.4`, C→`τ=2.0`, D→`bbox_loose + fg`, E→`EMA`, then the Phase 2 config is:
```
BalancedSoftmaxTverskyLoss(tau=2.0, alpha=0.8, beta=0.4, bbox_pad_fraction=0.2, bbox_bg_weight=0.1)
+ foreground_mask=True + EMA(decay=0.999)
```

### 5.3 Sweep Details

#### Sweep A: Loss Function (8 experiments)

| Experiment | Loss | Key Difference |
|------------|------|----------------|
| `loss_2d_bce` | BCEWithLogitsLoss | CSC default, per-voxel, no class awareness |
| `loss_2d_focal` | Focal (γ=2.0) | Down-weights easy voxels |
| `loss_2d_dice_bce` | Dice + BCE | Region-based + voxel-based combo |
| `loss_2d_tversky` | Tversky (α=0.6, β=0.4) | Precision-biased region loss |
| `loss_2d_balanced_softmax_tversky` | BST (τ=1.0) | Full pipeline: logit adj + bbox + fg mask |
| `loss_2d_focal_tversky` | FocalTversky (γ=0.75) | Focal modulation on hard classes |
| `loss_2d_unified_focal` | AsymUnifiedFocal | SOTA compound: FocalTversky + DiceFocal |
| `loss_2d_boundary_tversky` | BoundaryTversky | Distance-transform upweighting near membranes |

#### Sweep B: Tversky α/β (6 experiments)

| Experiment | α | β | Bias |
|------------|---|---|------|
| `tversky_2d_balanced` | 0.5 | 0.5 | Neutral (= Dice) |
| `tversky_2d_precision_06_04` | 0.6 | 0.4 | Mild precision |
| `tversky_2d_precision_07_03` | 0.7 | 0.3 | Moderate precision |
| `tversky_2d_recall` | 0.3 | 0.7 | Recall bias |
| `tversky_2d_a08_b04` | 0.8 | 0.4 | Strong precision |
| `tversky_2d_a08_b06` | 0.8 | 0.6 | Precision + high FN penalty |

#### Sweep C: Class Weighting τ (5 experiments)

All use `BalancedSoftmaxTverskyLoss` with different τ:

| Experiment | τ | Effect |
|------------|---|--------|
| `tau_2d_0` | 0.0 | No logit adjustment (plain Tversky + bbox) |
| `tau_2d_05` | 0.5 | Mild adjustment |
| `tau_2d_10` | 1.0 | Default (= `balanced_softmax_tversky`) |
| `tau_2d_15` | 1.5 | Strong adjustment |
| `tau_2d_20` | 2.0 | Very strong adjustment |

#### Sweep D: Masking Strategy (7 experiments)

All use `BalancedSoftmaxTverskyLoss` variants:

| Experiment | Foreground Mask | BBox Mask | Masksup | Description |
|------------|----------------|-----------|---------|-------------|
| `mask_2d_none` | ❌ | ❌ | ❌ | No masking at all |
| `mask_2d_fg_only` | ✅ | ❌ | ❌ | Only foreground (black padding) mask |
| `mask_2d_bbox_only` | ❌ | ✅ | ❌ | Only bbox spatial mask |
| `mask_2d_bbox_fg` | ✅ | ✅ | ❌ | Bbox + foreground (the default BST config) |
| `mask_2d_bbox_loose` | ✅ | ✅ (loose) | ❌ | Wider bbox (pad=0.2, bg=0.1) |
| `mask_2d_masksup03` | ✅ | ✅ | ✅ (0.3) | Mask-supervised reconstruction |
| `mask_2d_masksup03_no_bbox` | ✅ | ❌ | ✅ (0.3) | Masksup without bbox |

#### Sweep E: Training Techniques (3 2D + 4 3D)

| Experiment | Technique | Description |
|------------|-----------|-------------|
| `tech_2d_ema` | EMA (decay=0.999) | Exponential moving average of weights |
| `tech_2d_no_weighted_sampler` | Uniform sampling | Disables rare-class upsampling |
| `tech_2d_focal_tversky_mild` | FocalTversky γ=0.5 | Milder focal than sweep A's γ=0.75 |
| `tech_3d_deep_supervision` | Deep supervision | SegResNetDS with dsdepth=4 (3D only) |

---

## 6. Infrastructure

### 6.1 Cluster

**UNC Sycamore HPC** — `h100_sn` partition:
- **Per node:** 1.54 TB RAM, 256 CPU cores, 4× NVIDIA H100 80GB HBM3
- **QOS limit:** 12 GPUs per user concurrent
- **Max time:** 5 days per job

### 6.2 SLURM Configuration

| Parameter | 2D Jobs | 3D Jobs |
|-----------|---------|---------|
| Partition | h100_sn | h100_sn |
| GPUs | 1× H100 | 1× H100 |
| CPUs | 32 | 32 |
| RAM | 384 GB | **512 GB** |
| Time limit | 3 days | 3 days |
| Batch size | 8 | 1 |
| Input shape | 1×256×256 | 128×128×128 |
| num_workers | 8 | 8 |
| Wall time | ~3 hours | ~8-12 hours (est.) |

The 3D jobs need 512 GB because EmptyImage tensors at 128³ consume ~321 GB of host RAM.

### 6.3 Output Structure

```
runs/ablation/
├── logs/
│   ├── abl_{experiment_name}_{jobid}.out   # stdout (training progress)
│   └── abl_{experiment_name}_{jobid}.err   # stderr (warnings, errors)
├── {experiment_name}/
│   ├── config.json                         # Full argparse config
│   ├── checkpoints/
│   │   ├── best.pth                        # Best val_loss model state
│   │   ├── latest.pth                      # Resume checkpoint
│   │   └── epoch_{N}.pth                   # Periodic (every 10 epochs)
│   └── tensorboard/
│       └── events.out.tfevents.*           # TensorBoard logs
```

---

## 7. Results (Phase 1 — 2D)

### 7.1 Status

| Category | Count | Status |
|----------|-------|--------|
| 2D completed | 25/29 | ✅ All converged at 50 epochs |
| 2D pending | 4 | `no_weighted_sampler` (1793621) + 3 focal tversky relaunches (1793665–1793667) |
| 3D running | 8/33 | Loss sweep A (6 non-focal) + 2 from sweep B |
| 3D pending | 25/33 | Sweeps B–E + 3 focal tversky relaunches (1793668–1793670) |

*Note: Focal tversky experiments (focal_tversky, unified_focal, focal_tversky_mild) were relaunched fresh after the float32+epsilon fix — see §8.7.*

### 7.2 2D Validation Loss Results

**⚠️ IMPORTANT: Validation losses are NOT directly comparable across different loss function families.** BCE/Focal operate on different scales than Tversky-family losses. Within each sweep, losses ARE comparable since they use the same loss function with different hyperparameters. True comparison requires downstream metrics (per-class IoU on held-out data).

#### Sweep A: Loss Function

| Experiment | Best Val Loss | Notes |
|------------|:------------:|-------|
| `loss_2d_bce` | 0.0425 | BCE scale (very different from Tversky) |
| `loss_2d_focal` | 0.0023 | Focal scale (even smaller) |
| `loss_2d_dice_bce` | 0.4656 | Dice+BCE scale |
| `loss_2d_tversky` | 0.7136 | Tversky baseline |
| `loss_2d_balanced_softmax_tversky` | 0.6055 | BST with all features |
| `loss_2d_focal_tversky` | ⏳ | Relaunched (1793665) — previous run had NaN-corrupted weights |
| `loss_2d_unified_focal` | ⏳ | Relaunched (1793666) — previous run had NaN-corrupted weights |
| `loss_2d_boundary_tversky` | 0.6996 | Slight improvement over plain Tversky |

*Focal Tversky and Unified Focal originally produced 100% NaN weights due to float16 backward gradient explosion (see §8.7). Relaunched with float32+epsilon fix.*

#### Sweep B: Tversky α/β

| Experiment | α | β | Best Val Loss |
|------------|---|---|:------------:|
| `tversky_2d_balanced` | 0.5 | 0.5 | 0.6950 |
| `tversky_2d_precision_06_04` | 0.6 | 0.4 | 0.5614 |
| `tversky_2d_precision_07_03` | 0.7 | 0.3 | 0.6442 |
| `tversky_2d_recall` | 0.3 | 0.7 | 0.6773 |
| **`tversky_2d_a08_b04`** | **0.8** | **0.4** | **0.5712** |
| `tversky_2d_a08_b06` | 0.8 | 0.6 | 0.7274 |

**Winner:** `α=0.6, β=0.4` (the default) has the lowest val loss at 0.5614. Strong precision (α=0.8) with moderate β=0.4 is competitive at 0.5712.

#### Sweep C: Class Weighting (τ)

| Experiment | τ | Best Val Loss |
|------------|---|:------------:|
| `tau_2d_0` | 0.0 | 0.5225 |
| `tau_2d_05` | 0.5 | 0.4252 |
| `tau_2d_10` | 1.0 | 0.5444 |
| **`tau_2d_15`** | **1.5** | **0.4094** |
| **`tau_2d_20`** | **2.0** | **0.4090** |

**Winner:** τ=1.5 and τ=2.0 are effectively tied at ~0.409. Both beat the default τ=1.0 significantly. Stronger logit adjustment helps.

#### Sweep D: Masking Strategy

| Experiment | FG | BBox | Masksup | Best Val Loss |
|------------|:--:|:----:|:-------:|:------------:|
| `mask_2d_none` | ❌ | ❌ | ❌ | 0.6782 |
| `mask_2d_fg_only` | ✅ | ❌ | ❌ | 0.4871 |
| `mask_2d_bbox_only` | ❌ | ✅ | ❌ | 0.7192 |
| `mask_2d_bbox_fg` | ✅ | ✅ | ❌ | 0.5467 |
| `mask_2d_bbox_loose` | ✅ | ✅ | ❌ | 0.3921 |
| `mask_2d_masksup03` | ✅ | ✅ | ✅ | 0.5515 |
| **`mask_2d_masksup03_no_bbox`** | ✅ | ❌ | ✅ | **0.3836** |

**Key findings:**
- Foreground masking alone (0.4871) is a huge win over nothing (0.6782)
- Tight bbox masking alone (0.7192) actually HURTS — worse than no masking
- **masksup=0.3 without bbox (0.3836) is the overall best**
- Loose bbox (0.3921) is competitive
- The default tight bbox+fg config (0.5467) is mediocre — the bbox is too restrictive

#### Sweep E: Training Techniques

| Experiment | Technique | Best Val Loss |
|------------|-----------|:------------:|
| `tech_2d_ema` | EMA (0.999) | 0.5603 |
| `tech_2d_no_weighted_sampler` | Uniform sampling | (resubmitted — pending) |
| `tech_2d_focal_tversky_mild` | FocalTversky γ=0.5 | ⏳ Relaunched (1793667) — previous run had NaN-corrupted weights |

EMA (0.5603) is better than the BST baseline (0.6055). Compare against `balanced_softmax_tversky` default (0.6055): EMA helps.

---

## 8. Bugs Fixed

### 8.1 EmptyImage CUDA OOM (early)
**Problem:** `get_dataloader()` defaulted to moving all data to GPU, causing ~784 EmptyImage tensors to consume all VRAM.
**Fix:** Pass `device="cpu"` to `get_dataloader()`. Training loop does per-batch `.to("cuda")`.

### 8.2 Host RAM OOM (2D)
**Problem:** 2D jobs crashed with 128 GB SLURM allocation.
**Fix:** Increased to 384 GB. 2D jobs plateau at ~265 GB.

### 8.3 Host RAM OOM (3D)
**Problem:** 3D jobs with 128³ patches create ~41,000 EmptyImage tensors × 8 MB each = 321 GB.
**Fix:** Increased SLURM `--mem` from 384G to 512G for 3D jobs. (Long-term fix: make EmptyImage lazy.)

### 8.4 AMP Float16 NaN in Focal Tversky (initial, incomplete)
**Problem:** `(1 - tversky)^0.75` produces NaN when tversky > 1.0 due to float16 rounding.
**Initial fix:** `.clamp(min=0.0, max=1.0)` before `.pow(gamma)`. This prevented forward NaN but not backward gradient explosion — see §8.7 for the complete fix.

### 8.5 NaN-Skip Guards
**Problem:** Even with the clamp fix, if a loss produces NaN/Inf (e.g., division by zero edge cases), it poisons model weights.
**Fix:** Added `if not torch.isfinite(loss): optimizer.zero_grad(); continue` in training loop, and corresponding check in validation.

### 8.6 torch.load weights_only
**Problem:** `tech_2d_no_weighted_sampler` failed on resume because `torch.load(..., weights_only=True)` rejected numpy scalars in the checkpoint (PyTorch 2.6 changed the default).
**Fix:** Deleted stale checkpoint so the job starts fresh. `weights_only=True` is kept for security.

### 8.7 Focal Tversky Float16 Backward Gradient Explosion (complete fix)
**Problem:** Even with the forward clamp (§8.4), all 3 focal Tversky variants produced 100% NaN model weights after ~7 epochs. Root cause: the gradient of `x^γ` is `γ·x^(γ-1)` which → ∞ as x→0 when γ<1. Under AMP float16, this gradient overflow poisons all weights. The NaN-skip guard then silently skipped every batch for epochs 8-50 (no training occurred). Additionally, `val_steps=0` → `avg_val_loss=0.0` → the NaN-corrupted model was saved as `best.pth`.
**Fix (3 parts):**
1. `focal_tversky.py`: Cast `input.float()` before `sigmoid()` so the entire loss computation runs in float32, keeping backward gradients finite.
2. `focal_tversky.py`: Clamp lower bound to `eps=1e-6` (not 0.0) before `.pow(gamma)` — bounds the backward gradient at `γ·ε^(γ-1)` ≈ 750 (finite).
3. `train.py`: Guard `if val_steps > 0 and avg_val_loss < best_val_loss:` — prevents 0-batch validation from overwriting real checkpoints.

**Affected experiments:** `loss_{2d,3d}_focal_tversky`, `loss_{2d,3d}_unified_focal`, `tech_{2d,3d}_focal_tversky_mild` — all 6 relaunched as jobs 1793665–1793670.

---

## 9. Current Job Status (2026-02-23)

### Running Jobs (as of ~47 min into launch)

| Job ID | Experiment | Status | Notes |
|--------|-----------|--------|-------|
| 1793588 | `loss_3d_bce` | RUNNING | ~47 min in |
| 1793589 | `loss_3d_focal` | RUNNING | ~47 min in |
| 1793590 | `loss_3d_dice_bce` | RUNNING | ~47 min in |
| 1793591 | `loss_3d_tversky` | RUNNING | ~47 min in |
| 1793592 | `loss_3d_balanced_softmax_tversky` | RUNNING | ~47 min in |
| 1793595 | `loss_3d_boundary_tversky` | RUNNING | ~47 min in |
| 1793596 | `tversky_3d_balanced` | RUNNING | ~9 min in |
| 1793597 | `tversky_3d_precision_06_04` | RUNNING | ~9 min in |
| 1793598–1793617 | Sweeps B–E (3D) remaining | PENDING | — |
| 1793621 | `tech_2d_no_weighted_sampler` | PENDING | 2D relaunch |
| 1793665 | `loss_2d_focal_tversky` | PENDING | 🔧 Fixed relaunch |
| 1793666 | `loss_2d_unified_focal` | PENDING | 🔧 Fixed relaunch |
| 1793667 | `tech_2d_focal_tversky_mild` | PENDING | 🔧 Fixed relaunch |
| 1793668 | `loss_3d_focal_tversky` | PENDING | 🔧 Fixed relaunch |
| 1793669 | `loss_3d_unified_focal` | PENDING | 🔧 Fixed relaunch |
| 1793670 | `tech_3d_focal_tversky_mild` | PENDING | 🔧 Fixed relaunch |

**Total:** 8 running + 27 pending = 35 jobs (original 30 3D + 1 2D relaunch + 3 2D focal fix + 3 3D focal fix — minus 2 cancelled).

**Timing estimate:** Epoch 1 takes ~17min (500 iters × 2.1s/iter). 50 epochs ≈ ~15 hours per job. With 35 jobs and 8 GPU slots active, expect ~5 waves × 15h ≈ 3 days. **All complete by ~Wednesday Feb 26.**

---

## 10. Known Issues & TODOs

### 10.1 ~~Focal Tversky Validation NaN~~ ✅ RESOLVED

**Root cause identified and fixed** (see §8.7). The backward gradient of `x^γ` (γ<1) explodes to ∞ as x→0 in float16, poisoning all model weights. Fix: float32 cast + epsilon lower clamp. All 6 experiments relaunched (1793665–1793670).

### 10.2 EmptyImage Memory Optimization

The current approach pre-allocates full-size tensors for ~41K EmptyImage objects. A proper fix would monkey-patch `EmptyImage.__init__` to store only the scalar value and shape, expanding to full tensor only in `__getitem__`. This would reduce 3D memory from 321 GB to near zero.

### 10.3 Checkpoint Resume Compatibility

`torch.load(..., weights_only=True)` fails if checkpoint contains numpy scalars (from scheduler/optimizer state). Options:
- Use `torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])` before loading
- Ensure checkpoint saving avoids numpy (cast to Python scalars)
- Or just delete stale checkpoints and restart fresh

---

## 11. Next Steps

### 11.1 Immediate

1. ~~Investigate focal Tversky validation NaN~~ ✅ Fixed and relaunched (§8.7)
2. **Wait for all jobs** to complete (~Wednesday Feb 26)
3. **Verify focal Tversky relaunches** produce real val_loss values (not 0.0000/0 batches)
4. **Analyze 2D focal tversky results** once relaunches complete (~3 hours)
5. **Analyze 3D results** — do they agree with 2D findings?

### 11.2 Phase 2: Architecture Comparison

Once the winning loss/masking/technique combination is determined from Phase 1:

1. Run **4 architectures × 2D**: resnet_2d, unet_2d, swin_2d, vit_2d
2. Run **4 architectures × 3D**: segresnet_3d, swinunetr_3d, unet_3d, resnet_3d
3. 100 epochs, 1000 iters/epoch (longer training)
4. All use the Phase 1 winning configuration

Config generators already exist: `make_arch_comparison_2d()` and `make_arch_comparison_3d()` in `experiments.py`.

### 11.3 Phase 3: Final Submission

1. Train winning architecture + loss with full resources (longer, more data)
2. Generate predictions on test set
3. Submit to CellMap leaderboard

### 11.4 Future Improvements (Deferred)

- **Rare-class oversampling**: Beyond weighted_sampler, explicit mining of rare classes
- **Test-time augmentation (TTA)**: Flip/rotate ensemble for final predictions
- **Model ensembling**: Combine 2D + 3D predictions
- **Sliding window inference**: For full-volume prediction at test time

---

## 12. Quick Reference: How to Run

### Launch all 2D experiments
```bash
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
bash training/slurm/launch_ablation_2d_h100.sh
```

### Launch all 3D experiments
```bash
bash training/slurm/launch_ablation_3d_h100.sh
```

### Launch a single experiment
```bash
EXPERIMENT_NAME="loss_2d_bce" MODEL_NAME="resnet_2d" LOSS_NAME="bce" \
sbatch --export=ALL --job-name="abl_loss_2d_bce" training/slurm/ablation_2d_h100.sbatch
```

### Monitor jobs
```bash
squeue -u $USER                        # all jobs
squeue -u $USER -t RUNNING             # running only
tail -f runs/ablation/logs/abl_*_*.out # live training output
grep "Val loss:" runs/ablation/logs/abl_*_*.out | sort  # all val losses
```

### Check results
```bash
# Best val loss per experiment
for dir in runs/ablation/*/; do
    name=$(basename "$dir")
    [[ "$name" == "logs" ]] && continue
    logfile=$(ls runs/ablation/logs/abl_${name}_*.out 2>/dev/null | tail -1)
    if [ -n "$logfile" ]; then
        best=$(grep "New best model" "$logfile" | tail -1 | grep -oP 'val_loss=\K[0-9.]+')
        echo "$name: $best"
    fi
done | sort -t: -k2 -n
```

### TensorBoard
```bash
tensorboard --logdir runs/ablation/ --port 6006
```
