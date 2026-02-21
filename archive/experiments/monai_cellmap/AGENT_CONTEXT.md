# Agent Context: CellMap Segmentation — MONAI 3D Pipeline

> **Purpose**: Handoff document for AI coding agents. Contains complete project state,
> architecture, training results, file inventory, and next-step instructions so a new
> agent can continue without any context loss.
>
> **Last Updated**: 2026-02-16 (Monday), ~3 days into training.

---

## 1. Project Overview

**Goal**: Build a competitive entry for the [CellMap Segmentation Challenge](https://cellmapsegmentationchallenge.janelia.org/) — 3D multi-label organelle segmentation from FIB-SEM electron microscopy volumes.

**Task**: Predict 14 binary organelle masks per voxel:
```
ecs, pm, mito_mem, mito_lum, mito_ribo, golgi_mem, golgi_lum,
ves_mem, ves_lum, endo_mem, endo_lum, er_mem, er_lum, nuc
```

**Challenge**: Partial annotations — not every crop has every class labeled. Loss and metrics must mask unannotated channels.

**User**: gsgeorge (UNC Chapel Hill, PI: cburch)

---

## 2. Infrastructure

| Resource | Details |
|----------|---------|
| **Cluster** | UNC Longleaf HPC (`longleaf.unc.edu`) |
| **Node** | g181003 (reserved via Slurm reservation `gsgeorge_9034`) |
| **GPUs** | 8× NVIDIA L40S 48 GB each (6 used for 3D training) |
| **CPUs** | 64 total, 48 allocated |
| **RAM** | 1007 GB total, ~200–290 GB used during training |
| **Partition** | `l40-gpu` |
| **Account** | `rc_cburch_pi`, QOS `gpu_access` |
| **Conda env** | `csc` (via micromamba) |
| **Python** | 3.x with PyTorch 2.10.0+cu128, MONAI 1.5.2 |
| **OnDemand** | https://ondemand.rc.unc.edu (TensorBoard accessible here) |

### Key Paths
```
REPO        = /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
EXPERIMENT  = $REPO/experiments/monai_cellmap
CONFIGS     = $EXPERIMENT/configs/
CONFIGS_2D  = $EXPERIMENT/configs_2d/
DATA        = $EXPERIMENT/data/
LOSSES      = $EXPERIMENT/losses/
MODELS      = $EXPERIMENT/models/
SLURM       = $EXPERIMENT/slurm/
RUNS        = /work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap
LOGS        = $REPO/logs
DATALIST    = $REPO/auto3dseg/nifti_data/datalist.json
DATASPLIT   = $REPO/datasplit.csv
RAW_DATA    = $REPO/data/   (22 datasets: jrc_cos7-1a, jrc_hela-2, etc.)
```

### Checkpoint Locations
```
$RUNS/segresnet_ds/checkpoint_best.pth       (998 MB, ~559 epoch checkpoints)
$RUNS/flexunet_resnet34/checkpoint_best.pth  (808 MB, 602 epoch checkpoints)
$RUNS/swinunetr/checkpoint_best.pth          (841 MB, ~573 epoch checkpoints)
```

### TensorBoard Event Files
```
$RUNS/segresnet_ds/tb/events.out.tfevents.*.g181003.*
$RUNS/flexunet_resnet34/tb/events.out.tfevents.*.g181003.*
$RUNS/swinunetr/tb/events.out.tfevents.*.g181003.*
```

---

## 3. Architecture

### Pipeline Design: "Crop-First"
Standard approach (load full volume → expand to 14 channels → crop) is infeasible because some volumes expand to 84 GB in float32 multi-channel. Instead:

1. Load NIfTI volume (uint8 image + uint8/16 integer label)
2. Random-crop patches at **integer-label resolution** (small footprint)
3. Expand integer labels → 14-channel binary **on the small crop** (~50 MB)
4. Normalize, augment, return

This keeps peak worker RAM at one decompressed NIfTI (~hundreds of MB) instead of 84 GB.

### Three 3D Models (trained in parallel)

| Model | Backbone | Patch Size | Batch | LR | Epochs | GPUs | Port |
|-------|----------|-----------|-------|-----|--------|------|------|
| **SegResNet** | SegResNetDS (deep supervision) | 128³ | 2 | 2e-4 | 600 | 0,1 | 29500 |
| **FlexUNet** | FlexibleUNet + ResNet34 | 96³ | 4 | 1e-3 | 600 | 2,3 | 29501 |
| **SwinUNETR** | SwinUNETR v2 (feature_size=48) | 96³ | 2 | 1e-4 | 600 | 4,5 | 29502 |

### Loss Function: Balanced Softmax Tversky
- **Base**: Per-channel Tversky loss (α=0.6 FP penalty, β=0.4 FN penalty)
- **Weighting**: Balanced Softmax (τ=1.0) — shifts logits by online class-frequency estimates so rare classes get a positive offset
- **Masking**: Annotation mask (B, C) zeroes out loss for unannotated channels
- **Deep supervision** (SegResNet only): Multi-scale loss with weights [1.0, 0.5, 0.25, 0.125]

### Augmentation
- Random flips (all 3 axes, p=0.5 each)
- Random 90° rotation (axial plane, p=0.75)
- **Mixup** (FlexUNet only): Beta(1,1) distribution, p=1.0
- Intensity normalization (zero-mean, unit-var)

### Training Details
- Optimizer: AdamW (SegResNet, SwinUNETR), Adam (FlexUNet)
- Schedule: Cosine with 5% linear warmup
- Precision: bfloat16 autocast (L40S native)
- Gradient clipping: max_norm=1.0
- DDP via torchrun (2 GPUs per model, NCCL backend)
- Validation: every 5 epochs
- Checkpointing: every epoch (full state: model + optimizer + scheduler)

---

## 4. Training Results (as of 2026-02-16)

### Job: 30530860 (started 2026-02-13 ~4:06 PM EST)

| Model | Final Epoch | Status | Best Mean Dice | Best Epoch | Final Loss |
|-------|------------|--------|---------------|------------|------------|
| **FlexUNet** | 599/599 | ✅ COMPLETE | **0.2329** | 204 | 0.7909 |
| **SegResNet** | 599/599 | ✅ COMPLETE ~6PM | 0.1585 | 519 | 0.7333 |
| **SwinUNETR** | 599/599 | ✅ COMPLETE ~4:30PM | 0.1787 | 139 | 0.7319 |

> **Detailed per-class Dice, GPU utilization, training dynamics, and Round 2 recommendations: see Section 16.**

### Key Observations from TensorBoard
- **Best metrics hit very early**: FlexUNet best ep204, SwinUNETR ep139 — long tail was wasted.
- **SegResNet**: Slow but steady. Best Dice at ep519 — latest improvement among 3 models.
- **FlexUNet**: Best model overall. Best Dice 0.2329. Only model with non-zero ves and decent endo.
- **SwinUNETR**: Best at mito, er. Best at ep139 — peaked earliest, most volatile val.
- **Validation is very noisy** — small val set + multi-label + random crops = ±0.10 swings.
- **Models are highly complementary** — per-class ensemble is the right strategy.

---

## 5. File Inventory

### Source Files (experiments/monai_cellmap/)
```
configs/
  __init__.py
  common_config.py          # Base config: 14 classes, loss params, DDP, resources
  cfg_segresnet.py          # SegResNet: 128³, batch 2, lr 2e-4, deep supervision
  cfg_flexunet_resnet.py    # FlexUNet: 96³, batch 4, lr 1e-3, Mixup p=1.0
  cfg_swinunetr.py          # SwinUNETR v2: 96³, batch 2, lr 1e-4

configs_2d/
  train_2d_unet.py          # 2D UNet via CSC framework (NOT YET RUN)
  train_2d_swin.py          # 2D Swin via CSC framework (NOT YET RUN)

data/
  __init__.py
  ds_cellmap.py             # Crop-first dataset, ParseAnnotationMaskd, flat_collate_fn

losses/
  __init__.py
  partial_annotation.py     # PartialTverskyLoss, BalancedSoftmaxTverskyLoss, DS wrapper

models/
  __init__.py
  mdl_cellmap.py            # Net class: backbone + loss + Mixup, forward returns dict

slurm/
  train_reserved.sbatch     # 3D training: 3 models × 2 GPUs, reservation gsgeorge_9034
  train_2d_reserved.sbatch  # 2D training: sequential, 1 GPU (NOT YET RUN)
  probe_resources.sbatch    # Resource probing utility

train.py                    # Main training loop: DDP, mixed precision, checkpoint, TB
utils.py                    # DDP setup, CosineWarmupScheduler, optimizer/checkpoint utils
README.md                   # Experiment documentation
AGENT_CONTEXT.md            # THIS FILE — agent handoff context
```

### Data Format
- **Raw data**: Zarr volumes in `$REPO/data/jrc_*/` (22 datasets)
- **NIfTI conversion**: Pre-converted to NIfTI in `$REPO/auto3dseg/nifti_data/`
- **Datalist**: `$REPO/auto3dseg/nifti_data/datalist.json` — JSON with `training` and `validation` arrays, each entry has `image`, `label`, `annotated_classes` keys
- **Datasplit**: `$REPO/datasplit.csv` — CSV with split, zarr path, EM key, label key

---

## 6. Git State

- **Branch**: `main` (commit `53c4403`)
- **Remote**: GitHub (pushed and up to date)
- **Old branch**: `experiment/monai-3d-l40s` exists on remote but `main` is source of truth
- **Staged but uncommitted**: auto3dseg files (partial_annotation.py, patch_templates.py, run_auto3dseg.py) from a stash pop
- **.gitignore**: blocks `data/` directory — Python source files in `data/` were force-added with `-f`

---

## 7. HPC Coordination

- **Reservation**: `gsgeorge_9034` extends into next week
- **Admin contact**: Rob (HPC admin) — coordinates via email
- **2D models**: NOT running yet. Need 256g+ RAM or different node. Rob will coordinate mid-week.
- **GPU accounting bug**: 2 phantom GPUs in Slurm accounting. Decision: leave alone, Rob will help.

---

## 8. Immediate Next Steps: Round 1.5 (No Retraining)

These apply to the **existing checkpoints** at inference time:

### 8.1 Inference Pipeline
Build `experiments/monai_cellmap/inference.py`:
- Load `checkpoint_best.pth` for each model
- Sliding window inference with MONAI's `sliding_window_inference()`
  - Use larger window than training (e.g., 128³ or 192³ with overlap 0.5)
  - Gaussian importance weighting
- Sigmoid → threshold per class
- Save predictions as Zarr or NIfTI

### 8.2 Test-Time Augmentation (TTA)
- 8 flip combinations (all combinations of x/y/z flips)
- For each: flip input → predict → flip prediction back → average
- Can be integrated into inference.py

### 8.3 Per-Class Ensemble
Based on validation Dice (Section 4 above):
- For each of the 14 classes, select whichever model has the highest validation Dice
- At inference: run all 3 models, then select per-class predictions from the best model
- This is the **simplest ensemble** and leverages the different strengths shown in training

### 8.4 Post-Processing (per class)
- **Connected components**: remove small disconnected blobs (especially for nuc, mito, golgi)
- **Size filtering**: minimum volume thresholds per class
- **Morphological operations**: closing for membranes (pm, mito_mem, er_mem)
- **Class-specific thresholds**: tune sigmoid cutoff per class on validation data (not all classes work best at 0.5)

### 8.5 Inference Slurm Script
`experiments/monai_cellmap/slurm/inference.sbatch`:
- 1 GPU, adequate RAM
- Runs inference.py on test/validation volumes
- Saves predictions to `$RUNS/predictions/`

---

## 9. Future Rounds (Require Retraining)

### Round 2: Fine-tuning from current checkpoints
- **Pseudo-labeling**: Use current models to predict on unlabeled volumes → treat as GT → fine-tune ~200 more epochs
- **SwinUNETR with pre-trained weights**: MONAI provides self-supervised pre-trained weights → new run from epoch 0 but converges faster
- **Upweight rare classes**: Increase loss weight for ves, pm, endo, mito_ribo
- **Target**: 0.38–0.48 mean Dice

### Round 3: Full retrain
- **nnU-Net**: Auto-configured U-Net pipeline, consistently wins challenges
- **Cascade approach**: Coarse → fine model
- **Target**: 0.50+

---

## 10. Useful Commands

### Check job status
```bash
ssh longleaf.unc.edu 'squeue -j 30530860 -o "%.10i %.8T %.10M %.10l"'
```

### Check training progress
```bash
ssh longleaf.unc.edu 'for m in segresnet flexunet swinunetr; do echo "=== $m ==="; grep -E "Epoch [0-9]+/599 \| Loss" /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/${m}_30530860.out | tail -3; done'
```

### Check per-class Dice
```bash
ssh longleaf.unc.edu 'for m in segresnet flexunet swinunetr; do echo "=== $m ==="; grep -E "(ecs:|pm:|mito|golgi|ves_|endo|er_|nuc:)" /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/${m}_30530860.out | tail -14; done'
```

### Check RAM usage
```bash
ssh longleaf.unc.edu 'grep MONITOR /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/monai_cellmap_30530860.out | tail -3'
```

### TensorBoard (via OnDemand)
```bash
# In an OnDemand interactive session:
micromamba activate csc
tensorboard --logdir /work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap --port 6006 --bind_all
```

### Activate environment
```bash
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc
```

---

## 11. Known Issues & Workarounds

1. **`pin_memory=True` causes slowdowns**: Set to False (MONAI #3116)
2. **`LD_LIBRARY_PATH` for sqlite3/libstdc++**: Needed for CSC framework, set in 2D sbatch
3. **SpatialPadd required**: Some CellMap volumes have dimensions < patch size — pad before crop
4. **`.gitignore` blocks `data/`**: Use `git add -f` for Python files inside data/
5. **Slurm GPU phantom accounting**: 2 ghost GPUs in accounting — ignore, Rob will fix
6. **2D models OOM at 128g**: Need 256g+ RAM for validation. Coordinate with Rob.
7. **Checkpoint saves every epoch**: ~800–1000 MB each × 600 epochs = significant disk. May want to clean up old epoch checkpoints after training completes, keeping only `checkpoint_best.pth` and `checkpoint_last.pth`.

---

## 12. Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| Crop-first (not CacheDataset/PersistentDataset) | Largest volume expands to 84 GB multi-channel. Crop at integer-label resolution keeps peak RAM per worker to hundreds of MB. |
| Tversky α=0.6, β=0.4 | From loss_optimization experiment — best base loss (precision-biased Tversky). |
| Balanced Softmax τ=1.0 | From class_weighting experiment — best weighting strategy (mean Dice 0.5711). |
| 3 architectures in parallel | Different inductive biases → per-class ensemble. SwinUNETR for attention, SegResNet for residual features, FlexUNet for speed. |
| bfloat16 not fp16 | L40S has native bf16 support. No loss scaling needed. |
| Eval every 5 epochs | Balance between monitoring granularity and training speed. |
| `save_weights_only=False` | Full state (optimizer, scheduler) saved for proper resume. |
| `num_workers=4` | 4 workers/rank × 6 ranks = 24 total. Keeps RAM under control (~200–290 GB / 1007 GB). |

---

## 13. Challenge Submission Details

### Evaluation
- **Both semantic AND instance segmentation** are scored
- Instance-segmented classes: `nuc`, `mito`, `ves`, `endo`, `cell`, `vim`, `lyso`, `ld`, `perox`, `np`, `mt`
- Connected components are run automatically on submissions for instance classes
- Advanced post-processing (watershed) can be done before submission for better instance seg
- Different organelles have different native resolutions (8nm, 16nm, 32nm)
- Evaluations take up to 3 hours after submission
- No daily submission limit (currently)

### Submission Format
- **Zarr-2** file, zipped, uploaded via https://cellmapchallenge.janelia.org/submissions/
- Structure: `submission.zarr/<test_crop_name>/<label_name>/` with scale/offset metadata
- Convenience: `csc pack-results` auto-packages predictions into correct format
- Predictions must match test crop geometry (scale, offset, shape in nanometers)
- Submitting higher-resolution data leads to best results after resampling

### Inference Files (Round 1.5)
```
experiments/monai_cellmap/
  inference.py              # Sliding window + TTA + per-class ensemble + post-processing
  tune_thresholds.py        # Sweep per-class sigmoid thresholds on val probability maps
  evaluate_ensemble.py      # Quick crop-based eval of all 3 models → ensemble map
  slurm/inference.sbatch    # 3-step: ensemble inference → threshold tuning → re-evaluate
```

---

## 14. Reservation & Node Status (as of 2026-02-16)

- **Reservation**: `gsgeorge_9034` — ACTIVE
- **Start**: 2026-02-12 17:29:34
- **End**: 2026-02-23 17:29:34 (~7 days remaining)
- **Node**: g181003 (8× L40S 48GB, 64 CPUs, 1007 GB RAM)
- **Flags**: OVERLAP, IGNORE_JOBS, SPEC_NODES, MAGNETIC
- **Users**: gsgeorge, robz, jennyw

---

## 15. Roadmap to Leaderboard #1

### Phase 1: Baseline Submission (This Week) — Target: 0.25–0.35
**Hardware**: 1 GPU, 128g RAM, ~12 hours
1. ✅ Training complete (FlexUNet done, SegResNet/SwinUNETR finishing tonight)
2. Run `evaluate_ensemble.py` — get true per-class ensemble map
3. Run `inference.sbatch` — ensemble + TTA + post-processing on validation
4. Tune thresholds via `tune_thresholds.py`
5. Adapt output to Zarr format (bridge NIfTI predictions → Zarr with correct metadata)
6. Run `csc predict` on test crops → `csc pack-results` → SUBMIT
7. Get leaderboard baseline score

### Phase 2: Close the Gap (Next Week) — Target: 0.35–0.45
**Hardware**: 4–6 GPUs, 256g RAM, ~3–4 days
8. Instance segmentation post-processing (watershed for nuc, mito, ves, endo — huge differentiator)
9. Pseudo-labeling (run models on unlabeled volumes → fine-tune from checkpoints ~200 epochs)
10. SwinUNETR with MONAI self-supervised pretrained weights
11. 2D model ensemble (coordinate with Rob for 2D jobs)
12. Multi-resolution awareness (predict at native resolution per class)

### Phase 3: Competitive Edge (Week 3) — Target: 0.45–0.55
**Hardware**: 6–8 GPUs, 480g RAM, ~5 days
13. nnU-Net auto-configured baseline (ensemble with MONAI models)
14. Larger training patches (192³ or 256³)
15. Better augmentation (elastic deformation, per-dataset intensity, copy-paste for rare classes)
16. Class-specific models for hard classes (ves, endo, pm)

### Phase 4: Top of Leaderboard (Week 4+) — Target: 0.55–0.65+
**Hardware**: 8 GPUs, 480g RAM, ongoing
17. Multi-scale cascade (coarse → fine)
18. Model soup / stochastic weight averaging
19. Test-time optimization (adapt to each volume's intensity distribution)
20. 2.5D consensus (predict from x, y, z planes with 2D models, average)
21. Iterative pseudo-labeling (multiple rounds)

### Biggest Bang-for-Buck Items (ranked)
1. 🏆 Instance segmentation post-processing (watershed) — +0.05–0.10
2. 🏆 nnU-Net — auto-configures everything, consistently wins challenges
3. 🏆 Multi-resolution predictions — predict at native crop resolution
4. Pseudo-labeling — expands training data 5–10×
5. Per-class threshold tuning — already built
6. SwinUNETR pretrained weights — faster convergence, better features

### Hardware Requirements per Step
| Step | GPUs | RAM | Time |
|------|------|-----|------|
| Round 1.5: Inference + ensemble + TTA | 1× L40S | 128 GB | ~6–12 hrs |
| Round 1.5: Threshold tuning | CPU only | 32 GB | ~10 min |
| Round 2: Pseudo-label generation | 1× L40S | 128 GB | ~12 hrs |
| Round 2: Fine-tune from checkpoints | 4× L40S | 256 GB | ~1–2 days |
| Round 2: SwinUNETR pretrained | 2× L40S | 256 GB | ~2 days |
| Round 3: nnU-Net | 2× L40S | 256 GB | ~3–5 days |
| 2D models (CSC framework) | 1× L40S | 256 GB | ~1–2 days |
| Final submission packaging | CPU only | 64 GB | ~1 hr |

---

## 16. Round 1 Training Analysis (Final, 2026-02-16)

### Final Training Results

| Model | Epochs | Best Mean Dice | Best Epoch | Final Loss | Status |
|-------|--------|---------------|------------|------------|--------|
| **FlexUNet** | 599/599 | **0.2329** | 204 | 0.7909 | ✅ Complete |
| **SegResNet** | 599/599 | 0.1585 | 519 | 0.7333 | ✅ Complete ~6 PM |
| **SwinUNETR** | 599/599 | 0.1787 | 139 | 0.7319 | ✅ Complete ~4:30 PM |

### Per-Class Dice at Best Checkpoint

| Class | SegResNet | FlexUNet | SwinUNETR | Best |
|-------|-----------|----------|-----------|------|
| nuc | **0.5145** | 0.4042 | 0.4959 | SegResNet |
| ves_lum | 0.0012 | **0.5008** | 0.0003 | FlexUNet |
| ves_mem | 0.0024 | **0.4685** | 0.0007 | FlexUNet |
| mito_ribo | 0.0011 | **0.3846** | 0.4615 | SwinUNETR |
| golgi_lum | **0.3761** | 0.3193 | 0.3394 | SegResNet |
| golgi_mem | **0.3380** | 0.3165 | 0.2848 | SegResNet |
| ecs | **0.3131** | 0.2869 | 0.2739 | SegResNet |
| mito_lum | **0.2185** | 0.1669 | 0.1664 | SegResNet |
| mito_mem | 0.1756 | 0.1276 | **0.1865** | SwinUNETR |
| er_lum | 0.1069 | 0.0857 | **0.1191** | SwinUNETR |
| er_mem | 0.0877 | 0.0704 | **0.0999** | SwinUNETR |
| pm | 0.0535 | **0.0714** | 0.0591 | FlexUNet |
| endo_lum | 0.0162 | **0.0342** | 0.0062 | FlexUNet |
| endo_mem | 0.0140 | **0.0229** | 0.0083 | FlexUNet |

### GPU Memory Utilization (stable across entire 3-day run)

| Model | GPUs | VRAM/GPU | VRAM Total | Utilization | Headroom |
|-------|------|----------|------------|-------------|----------|
| **SegResNet** | 0, 1 | 25.1 GB | 46 GB | **55%** | ~21 GB — can increase batch or crop size |
| **FlexUNet** | 2, 3 | 13.6 GB | 46 GB | **30%** | ~32 GB — massive headroom |
| **SwinUNETR** | 4 | 38.6 GB | 46 GB | **84%** | ~7 GB — near limit |
| **SwinUNETR** | 5 | 41.4 GB | 46 GB | **90%** | ~5 GB — DO NOT increase |

- **RAM**: Peak 363 GB / 1007 GB (36%). Typical 150-250 GB.
- Memory was rock-stable — no spikes or OOM across 870+ monitor snapshots.

### Training Batch Sizes (Round 1)

| Model | batch_size | num_samples | GPUs | Effective patches/step | Crop size |
|-------|-----------|-------------|------|----------------------|-----------|
| SegResNet | 2 | 4 | 2 | 16 | 128³ |
| FlexUNet | 4 | 4 | 2 | 32 | 128³ |
| SwinUNETR | 2 | 4 | 2 | 16 | 128³ |

### Loss Progression

| Model | Epoch 0 | Epoch 50 | Final | Δ |
|-------|---------|----------|-------|---|
| SegResNet | 0.8061 | 0.7470 | 0.7333 | -0.073 |
| FlexUNet | 0.8749 | 0.8423 | 0.7909 | -0.084 |
| SwinUNETR | 0.8091 | 0.7584 | 0.7319 | -0.077 |

### Key Findings for Round 2

1. **Best metrics hit early, then plateau**: FlexUNet best at epoch 204, SwinUNETR at 139 — 
   late training was wasted. Need warm restarts or cyclic LR.
2. **Extreme validation noise**: Val Dice bounces ±0.10 between epochs. Small val set + 
   multi-label + random crops = high variance.
3. **Class imbalance is the #1 issue**: endo, pm, ves near zero. Tversky α=0.6/β=0.4 
   biases toward precision (penalizes FP more). Round 2: flip to α=0.3/β=0.7 for rare classes.
4. **Mixup + partial annotations interaction**: Mixed targets dilute toward zero for 
   unannotated classes, subtly suppressing recall. Consider disabling Mixup for Round 2 
   or using CutMix instead.
5. **SegResNet & FlexUNet have massive GPU headroom**: Can increase to 192³ crops or 
   larger batch sizes. SwinUNETR is maxed at 90% VRAM — keep at 128³.
6. **Models are highly complementary**: SegResNet best at large structures (nuc, golgi, ecs),
   FlexUNet best at rare/tiny (ves, endo, pm), SwinUNETR best at mid-range (mito, er).
   Per-class ensemble is the right strategy.

### Round 2 Config Recommendations

| Model | Crop Size | batch_size | GPUs | LR Schedule | α/β Strategy |
|-------|-----------|-----------|------|-------------|--------------|
| SegResNet | 192³ | 2 | 2 | Cosine warm restarts | Per-class |
| FlexUNet | 192³ | 3 | 2 | Cosine warm restarts | Per-class |
| SwinUNETR | 128³ | 2 | 2 | Cosine warm restarts + pretrained | Per-class |

---

## 17. Auto3DSeg v2 — Optimized Launch (Feb 17)

### Diagnosis of v1 Failures
Previous Auto3DSeg attempts (jobs 30266729–30463620) all crashed due to:
1. **Outlier crop**: `jrc_cos7-1a_crop247` — 8.2GB compressed, shape 1796×1500×2400 = 25.9GB as float32.
   Not properly cropped during Zarr→NIfTI conversion. Has `annotated_classes: ""` (no annotations).
2. **DiNTS search phase**: Uses `cache_rate: 1` (100% dataset cached in RAM). With the 25.9GB outlier
   and 7 DDP ranks, RAM usage would exceed ~1TB.
3. **Stale datastats.yaml**: Generated with outlier included, so auto-computed roi_size/batch_size
   were skewed by the 1796×1500×2400 shape.

### v2 Fixes Applied
1. **Removed outlier**: `jrc_cos7-1a_crop247` removed from `datalist.json` (230 train, 46 val)
2. **Fresh data analysis**: Step 0 deletes stale `datastats.yaml` and re-runs DataAnalyzer
3. **GPU optimization**: Added `get_gpu_customization_specs()` to `run_auto3dseg.py`:
   - MONAI BundleGen now runs random-search trials on actual GPU to find optimal batch sizes
   - L40S 48GB specs: SegResNet batch [1,6] / sw_batch [1,12], SwinUNETR [1,4]/[1,8], DiNTS [1,2]/[1,4]
   - Replaces hardcoded conservative values with empirically determined optimal ones
4. **Clean state**: Stale `progress.yaml`, `cache.yaml` removed before each step
5. **Config validation**: Post-generation summary prints roi_size, batch_size, epochs per algorithm
6. **8 GPUs**: Uses all L40S GPUs for DDP training (vs 7 in v1)

### Job Details
- **Job ID**: 31252136 (`auto3dseg_v2`)
- **Dependency**: `afterany:31119803` (waits for inference job to finish)
- **Node**: g181003 (8× L40S 48GB)
- **Wall time**: 6 days (reservation expires Feb 23)
- **Pipeline**: analyze → generate (GPU opt) → patch → train (3 algos × 1 fold)
- **Expected algorithms**: SegResNet, SwinUNETR, DiNTS (each auto-computed epochs)

### Key File Changes
- `auto3dseg/nifti_data/datalist.json`: Removed crop247 outlier (231→230 training)
- `auto3dseg/run_auto3dseg.py`: Added `get_gpu_customization_specs()` shared helper,
  wired `gpu_customization_specs` through `run_bundle_generation()` and `run_full_pipeline()`
- `auto3dseg/auto3dseg_train_reserved.sbatch`: Complete rewrite with 4-step pipeline,
  GPU optimization, outlier sanity check, config summary debugging
