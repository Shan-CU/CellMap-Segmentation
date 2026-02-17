# Agent Context: CellMap Segmentation — Round 2 Handoff

> **Purpose**: Handoff document for AI coding agents continuing this project.
> Complete project state, what changed since Round 1, what's running, what to do next.
>
> **Last Updated**: 2026-02-17 (Tuesday), ~6 PM EST
> **Previous Context**: `experiments/monai_cellmap/AGENT_CONTEXT.md` (Round 1 — now stale)

---

## 0. TL;DR — What's Happening Right Now

**All 4 models are RUNNING across 2 clusters:**

| Cluster | Node | Job ID | Model | Patch | Batch | GPU | Status |
|---------|------|--------|-------|-------|-------|-----|--------|
| **Sycamore H100** | g15070304 | 1690610 | SegResNet-Wide 48f | **192³** | 2 | 2× H100 80GB | ✅ RUNNING |
| **Sycamore H100** | g15070306 | 1690611 | FlexUNet ResNet34 | **192³** | 4 | 2× H100 80GB | ✅ RUNNING |
| **Longleaf L40S** | g181003 | 31457329 | SwinUNETR v2 | 96³ | 2 | 2× L40S 48GB | ✅ RUNNING (epoch ~5) |
| **Longleaf L40S** | g181003 | 31462469 | SegResNet 32f | 128³ | 2 | 2× L40S 48GB | ✅ RUNNING |

### Key Architecture Decision
The two most VRAM-hungry models (SegResNet-Wide, FlexUNet) were moved to Sycamore's H100 80GB GPUs
to exploit **192³ patches** (up from 128³ on L40S). This gives 3.375× more voxels per patch = much
richer spatial context. The lighter models (SwinUNETR, SegResNet-32f) remain on the Longleaf L40S reservation.

### Two Clusters In Use
- **Sycamore** (`sycamore-login1`): H100 nodes, partition `h100_mn`, account `rc_alain_pi`, QOS `h100_mn`
- **Longleaf** (`longleaf.unc.edu`): L40S nodes, partition `l40-gpu`, account `rc_cburch_pi`, reservation `gsgeorge_9034`
- Both share the same `/work/` and `/nas/` filesystems — data and code paths are identical

---

## 1. What Changed: Round 1 → Round 2

### The Critical Discovery: 48 Tested Classes, Not 14

The CellMap challenge evaluates **48 tested classes** — 35 atomic + 16 group compositions (some overlap, net 48). Round 1 only trained 14 atomic classes, meaning **34/48 classes scored 0** on the leaderboard. V1's effective leaderboard score was ~0.063 (not the 0.216 validation Dice we measured).

### All Changes Made

| # | Change | Why | Impact |
|---|--------|-----|--------|
| 1 | **14 → 35 atomic classes** | Challenge evaluates 48 classes (35 atomic + 16 groups via OR) | 34 previously-zero classes now trained |
| 2 | **New NIfTI converter v2** | V1 converter had a critical bug (loaded entire EM volumes, 163 GB each) | 262 crops converted in 167s (was stuck for 30+ min) |
| 3 | **SegResNet-Wide+FlexUNet on H100 192³** | H100 80GB allows much bigger patches than L40S 48GB | 3.375× more voxels per patch |
| 4 | **SegResNet 32f: 128³** | 144³ caused CUDA OOM at 45 GB (deep supervision + 35 classes) | Safe at ~32 GB on L40S |
| 5 | **SwinUNETR: dropout 0.1** | Peaked at ep139 then overfit (0% dropout in V1) | Regularization to extend useful training |
| 6 | **Mixup disabled** | Mixup + partial annotations = mixed targets dilute unannotated channels | Better rare-class recall |
| 7 | **600 → 300 epochs** | Best metrics hit at ep139-519; long tail wasted | Faster turnaround |
| 8 | **4th model: SegResNet-Wide (48f)** | Wider feature maps for ensemble diversity | Different error profile for ensemble |
| 9 | **Checkpoint every 50 epochs** | V1 saved every epoch = 600 × 800 MB = 480 GB waste | Disk savings |
| 10 | **α=0.6, β=0.4 KEPT** | Experimentally validated: α=0.6/β=0.4 → Dice 0.370 vs α=0.3/β=0.7 → 0.028 (collapsed) | No change needed |
| 11 | **Split across 2 clusters** | Longleaf L40S reservation + Sycamore H100s = more GPUs, bigger patches | All 4 models running simultaneously |

### V2 Converter Bug Fix (Critical)

The v2 converter (`auto3dseg/convert_zarr_to_nifti_v2.py`) had a regression where `get_em_array()` called `np.asarray(arr)` on the **entire dataset EM volume** (up to 163 GB for cos7-1a) for every crop. V1's converter correctly used zarr coordinate transforms to slice only the crop's EM region.

**Fix**: Added `_get_crop_offset_and_shape()` and `get_em_region()` that parse zarr `multiscales[0].datasets[].coordinateTransformations` to compute voxel offsets and slice only the needed region. Memory dropped from 1283 GB (48 workers) to 81 GB (16 workers), conversion from stuck to 167 seconds.

---

## 2. V2 Training Configuration

### Four Models on 2 Clusters, 8 GPUs Total

| Cluster | Node | Model | Config File | Crop | Batch | VRAM (est.) | Port |
|---------|------|-------|-------------|------|-------|-------------|------|
| Sycamore H100 | g15070304 | SegResNet-Wide 48f | `cfg_segresnet_wide.py` | **192³** | 2 | ~65 GB / 80 GB | 29600 |
| Sycamore H100 | g15070306 | FlexUNet ResNet34 | `cfg_flexunet_resnet.py` | **192³** | 4 | ~55 GB / 80 GB | 29601 |
| Longleaf L40S | g181003 | SwinUNETR v2 | `cfg_swinunetr.py` | 96³ | 2 | ~42 GB / 48 GB | 29502 |
| Longleaf L40S | g181003 | SegResNet 32f | `cfg_segresnet.py` | 128³ | 2 | ~32 GB / 48 GB | 29500 |

### VRAM Lessons Learned
- **SegResNet 144³ caused CUDA OOM** (job 31456536): Deep supervision + 35 classes + Tversky loss
  on 144³ patches consumed ~45 GB (the `((1.0 - pred) * target).sum()` in loss computation). Reduced to 128³.
- **FlexUNet 120g system RAM OOM** (job 31456537): `cache_rate=1.0` on 262 NIfTI files + 2 DDP ranks ×
  4 workers exceeded 120 GB. H100 nodes have 1.5 TB RAM so this is no longer an issue.

### Loss Function
- **BalancedSoftmaxTverskyLoss**: α=0.6 (FP penalty), β=0.4 (FN penalty), τ=1.0
- Balanced Softmax shifts logits by online class-frequency estimates for rare-class boosting
- Annotation mask (B, C) zeroes out loss for unannotated channels per crop
- Deep supervision for both SegResNet variants: weights [1.0, 0.5, 0.25, 0.125]

### Data
- **35 atomic classes**, integer labels 1-35 → 35-channel binary at training time
- **262 NIfTI pairs** in `auto3dseg/nifti_data_v2/` (217 train, 45 val)
- **Datalist**: `auto3dseg/nifti_data_v2/datalist.json`
- **27 crops skipped** during conversion (no labels or all empty — legitimate)
- All 35 class indices (0-34) present in the dataset
- Per-class coverage: cyto 72.5% of crops → nucleo 0.4% (extreme imbalance)

### 35 Atomic Classes (in order, 0-indexed)
```
 0: ecs         7: ves_mem    14: lyso_mem   21: ne_lum    28: cyto
 1: pm          8: ves_lum    15: lyso_lum   22: np_out    29: mt_in
 2: mito_mem    9: endo_mem   16: ld_mem     23: np_in     30: perox_mem
 3: mito_lum   10: endo_lum   17: ld_lum     24: hchrom    31: perox_lum
 4: mito_ribo  11: er_mem     18: eres_mem   25: echrom    32: nhchrom
 5: golgi_mem  12: er_lum     19: eres_lum   26: nucpl     33: nechrom
 6: golgi_lum  13: nuc        20: ne_mem     27: mt_out    34: nucleo
```

### 16 Group Classes (composed at inference via OR)
```
mito       = mito_mem | mito_lum | mito_ribo
golgi      = golgi_mem | golgi_lum
ves        = ves_mem | ves_lum
endo       = endo_mem | endo_lum
lyso       = lyso_mem | lyso_lum
ld         = ld_mem | ld_lum
eres       = eres_mem | eres_lum
perox      = perox_mem | perox_lum
ne         = ne_mem | ne_lum | np_out | np_in
np         = np_out | np_in
chrom      = hchrom | nhchrom | echrom | nechrom
mt         = mt_out | mt_in
er         = er_mem | er_lum | eres_mem | eres_lum | ne_mem | ne_lum | np_out | np_in
er_mem_all = er_mem | eres_mem | ne_mem
ne_mem_all = ne_mem | np_out | np_in
cell       = (everything intracellular — 33 atomic classes)
```

35 atomic + 16 groups = 48 tested classes on the leaderboard (some groups overlap but each is scored independently).

---

## 3. Key File Inventory

### Paths
```
REPO        = /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
EXPERIMENT  = $REPO/experiments/monai_cellmap
CONFIGS     = $EXPERIMENT/configs/
LOSSES      = $EXPERIMENT/losses/
MODELS      = $EXPERIMENT/models/
SLURM       = $EXPERIMENT/slurm/
RUNS        = /work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap
LOGS        = $REPO/logs
V2_DATA     = $REPO/auto3dseg/nifti_data_v2/
V1_DATA     = $REPO/auto3dseg/nifti_data/     (old 14-class data, still exists)
RAW_ZARR    = $REPO/data/                      (22 datasets: jrc_cos7-1a, jrc_hela-2, etc.)
```

### Source Files
```
experiments/monai_cellmap/
├── configs/
│   ├── common_config.py          # 35 classes, loss params, DDP settings
│   ├── cfg_segresnet.py          # SegResNet 32f: 128³, batch 2, deep supervision (L40S)
│   ├── cfg_segresnet_wide.py     # SegResNet 48f: 192³, batch 2, deep supervision (H100)
│   ├── cfg_flexunet_resnet.py    # FlexUNet: 192³, batch 4, Mixup disabled (H100)
│   └── cfg_swinunetr.py          # SwinUNETR: 96³, batch 2, dropout 0.1 (L40S)
├── data/
│   └── ds_cellmap.py             # Crop-first dataset, multi-channel label expansion
├── losses/
│   └── partial_annotation.py     # BalancedSoftmaxTverskyLoss, annotation masking, DS wrapper
├── models/
│   └── mdl_cellmap.py            # Net wrapper: backbone + loss + Mixup, forward returns dict
├── slurm/
│   ├── train_r2_segresnet_wide_h100.sbatch  # SegResNet-Wide → Sycamore H100 (ACTIVE)
│   ├── train_r2_flexunet_h100.sbatch        # FlexUNet → Sycamore H100 (ACTIVE)
│   ├── train_r2_segresnet.sbatch            # SegResNet 32f → Longleaf L40S reservation (ACTIVE)
│   ├── train_r2_swinunetr.sbatch            # SwinUNETR → Longleaf L40S reservation (ACTIVE)
│   ├── train_r2_segresnet_nores.sbatch      # SegResNet 32f → Longleaf L40S no-reservation (backup)
│   ├── train_r2_flexunet_nores.sbatch       # FlexUNet → Longleaf L40S no-reservation (backup)
│   ├── train_r2_swinunetr_nores.sbatch      # SwinUNETR → Longleaf L40S no-reservation (backup)
│   ├── train_r2_segresnet_wide_nores.sbatch # SegResNet-Wide → Longleaf no-reservation (backup)
│   ├── train_r2_segresnet_wide.sbatch       # SegResNet-Wide → Longleaf reservation (superseded)
│   ├── train_r2_flexunet.sbatch             # FlexUNet → Longleaf reservation (superseded)
│   ├── train_r2.sbatch                      # Original combined 8-GPU script (superseded)
│   ├── train_reserved.sbatch                # V1 training (historical)
│   └── inference.sbatch                     # V1 inference (historical)
├── train.py                      # Main training loop: DDP, bf16, TB logging
├── utils.py                      # DDP setup, scheduler, checkpoint utilities
├── inference.py                  # Inference pipeline (V1 — needs update for V2 groups)
├── tune_thresholds.py            # Per-class threshold sweep on val probability maps
├── evaluate_ensemble.py          # Quick ensemble evaluation → ensemble map
├── AGENT_CONTEXT.md              # Round 1 context (STALE — use THIS file instead)
├── AGENT_CONTEXT_R2.md           # THIS FILE
└── IMPLEMENTATION_SPEC.md        # Architecture specification
```

### V2 Data Files
```
auto3dseg/nifti_data_v2/
├── datalist.json                 # 217 training + 45 validation entries
├── images/                       # 262 NIfTI EM images (*.nii.gz)
└── labels/                       # 262 NIfTI integer labels (*.nii.gz, values 0-35)
```

### V2 Output Directories (will be created by training)
```
$RUNS/segresnet_ds_r2/            # SegResNet 32f checkpoints + TB
$RUNS/flexunet_resnet34_r2/       # FlexUNet checkpoints + TB
$RUNS/swinunetr_r2/               # SwinUNETR checkpoints + TB
$RUNS/segresnet_wide_r2/          # SegResNet-Wide 48f checkpoints + TB (NEW)
```

### V1 Output Directories (preserved, do not delete)
```
$RUNS/segresnet_ds/               # V1 SegResNet (best epoch 519, Dice 0.158)
$RUNS/flexunet_resnet34/          # V1 FlexUNet (best epoch 204, Dice 0.233)
$RUNS/swinunetr/                  # V1 SwinUNETR (best epoch 139, Dice 0.179)
$RUNS/predictions/                # V1 inference outputs + thresholds.json
$RUNS/predictions_tuned/          # V1 Step 3 partial outputs (9/46 done, cancelled)
$RUNS/ensemble_map.json           # V1 per-class best model map
```

---

## 4. Infrastructure

### Dual-Cluster Setup

| Resource | Sycamore (H100) | Longleaf (L40S) |
|----------|-----------------|-----------------|
| **Login** | `sycamore-login1` (VS Code connects here) | `ssh longleaf.unc.edu` from Sycamore |
| **Nodes** | g15070304, g15070306 | g181003 (reserved) |
| **GPUs** | 4× H100 80GB per node (2 used each) | 8× L40S 48GB (4 used by us) |
| **CPUs** | 256 per node, 32 allocated per job | 64 total, 12 per job |
| **RAM** | 1.5 TB per node, 400 GB allocated | 1 TB total, 200 GB per job |
| **Partition** | `h100_mn` | `l40-gpu` |
| **Account** | `rc_alain_pi` | `rc_cburch_pi` |
| **QOS** | `h100_mn` | `gpu_access` |
| **Reservation** | None (standard queue, but mostly idle) | `gsgeorge_9034` (until **2026-03-17**) |
| **Max time** | 5 days | 5 days |

**Shared filesystem**: Both clusters mount `/work/` and `/nas/` — all code, data, and output paths are identical.

### Environment Activation (same on both clusters)
```bash
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc
```

### TensorBoard (V1 + V2 comparison)
```bash
# On Sycamore or via Longleaf OnDemand desktop:
micromamba activate csc
tensorboard --logdir /work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap \
  --host 0.0.0.0 --port 6006 --reload_interval 30
# Shows 7 runs: 3 V1 + 4 V2, auto-discovered from tb/ subdirectories
# V1: flexunet_resnet34/tb, segresnet_ds/tb, swinunetr/tb
# V2: flexunet_resnet34_r2/tb, segresnet_ds_r2/tb, swinunetr_r2/tb, segresnet_wide_r2/tb
```

---

## 5. Round 1 Results (For Reference)

| Model | Best Dice (14 classes) | Best Epoch | Effective Leaderboard Score |
|-------|----------------------|------------|---------------------------|
| FlexUNet | **0.233** | 204 | ~0.068 (34/48 classes = 0) |
| SwinUNETR | 0.179 | 139 | ~0.052 |
| SegResNet | 0.158 | 519 | ~0.046 |

V1 scored poorly because it only trained 14/48 tested classes. Group compositions (mito, golgi, ves, etc.) and 21 additional atomic classes all scored zero.

---

## 6. Next Steps (In Priority Order)

### Step 1: Monitor Training (ALL 4 MODELS ARE RUNNING)
Training is already launched. Monitor for OOM crashes or loss divergence:
```bash
# H100 jobs (Sycamore — run directly):
tail -5 /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/segresnet_wide_r2_1690610.err
tail -5 /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/flexunet_r2_1690611.err

# L40S jobs (Longleaf — via ssh):
ssh longleaf.unc.edu 'tail -5 /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/swinunetr_r2_31457329.err'
ssh longleaf.unc.edu 'tail -5 /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/segresnet_r2_31462469.err'

# Check all jobs are still running:
squeue -u gsgeorge  # Sycamore
ssh longleaf.unc.edu 'squeue -u gsgeorge'  # Longleaf
```
- Expected runtime: ~2-3 days for 300 epochs
- **Critical**: If H100 jobs OOM at 192³, fall back to 160³ (update configs and resubmit)

### Step 2: Update inference.py for V2 (While Training Runs)
The existing `inference.py` already has `GROUP_CLASSES` and 35-class support, but needs updates:
- Update `DEFAULT_ENSEMBLE_MAP` to include all 35 atomic classes (currently only 14)
- Add SegResNet-Wide as a 4th model option in the ensemble
- Update checkpoint paths to use `_r2` directories
- The group composition logic (OR-ing atomic predictions) is already implemented

### Step 3: Build Submission Pipeline (While Training Runs)
- Convert NIfTI predictions → Zarr-2 format with correct metadata (scale, offset)
- Use `csc pack-results` or build custom zarr packing
- Test on validation set first

### Step 4: Post-Training — Inference + Ensemble
1. Run `evaluate_ensemble.py` on V2 models → get new per-class best-model map
2. Run `tune_thresholds.py` → get optimal per-class sigmoid thresholds
3. Run `inference.py` with `--ensemble --tta --postprocess` on validation
4. Compute validation Dice on all 48 tested classes
5. Run on test crops → package → submit

### Step 5: Iterate
- If specific classes are weak, consider class-specific models or augmentation
- Pseudo-labeling: use V2 predictions on unlabeled volumes → fine-tune
- Instance segmentation post-processing (watershed) for mito, ves, nuc, etc.

---

## 7. Are We On Track to Top the Leaderboard?

### Honest Assessment

**Short answer**: We're positioned for a massive improvement over V1, but topping the leaderboard depends on competition quality.

### What We've Fixed (V1 → V2)

| Problem | V1 Score Impact | V2 Fix | Expected Improvement |
|---------|----------------|--------|---------------------|
| Only 14/48 classes trained | 34 classes = 0, floor score | All 48 classes covered | **+200-400%** on leaderboard |
| Small crops (96-128³) | Limited context for large structures | 128-144³ crops | +5-15% Dice |
| Mixup + partial annotations | Suppressed rare class recall | Mixup disabled | +5-10% rare class Dice |
| SwinUNETR overfitting | Peaked at ep139, wasted 461 epochs | Dropout 0.1 | +5-10% SwinUNETR Dice |
| 3 models on 6 GPUs | 2 GPUs idle | 4 models on 8 GPUs | Better ensemble diversity |
| 600 wasted epochs | Diminishing returns after ep200-519 | 300 epochs | Faster iteration |

### V2 Score Projection

| Scenario | Reasoning | Est. Mean Dice (48 classes) |
|----------|-----------|---------------------------|
| **Conservative** | 35 classes trained but many rare; some classes near-zero | 0.15 - 0.20 |
| **Expected** | Better crops + disabled Mixup + all classes + ensemble of 4 | **0.25 - 0.35** |
| **Optimistic** | Models learn rare classes well + group compositions boost score | 0.35 - 0.45 |

### V1 → V2 Multiplier

V1 effective leaderboard score was **~0.063** (0.216 Dice on 14 classes × 14/48 coverage). Even the conservative V2 estimate of 0.15 would be a **2.4× improvement**. The expected case of 0.30 would be **4.8× improvement**.

### Key Risks

1. **H100 192³ patches untested**: VRAM estimates are theoretical. SegResNet-Wide 48f at 192³ with deep supervision + 35 classes is the most aggressive config. If it OOMs, fall back to 160³.
2. **Rare classes may still score near-zero**: nucleo, nhchrom, nechrom have <1% crop coverage. Even with Balanced Softmax, the model may not learn them well.
3. **Group compositions untested**: The OR-ing logic in inference.py needs testing. If it has bugs, 16/48 classes score zero.
4. **Competition unknown**: We don't know what other teams have submitted. If top scores are >0.50, we need more rounds.
5. **H100 node stability**: Sycamore H100s just came back from maintenance. If they go down again, jobs are lost (no checkpoint resume from epoch 0). Monitor frequently in the first 24 hours.

### What Would Actually Get Us to #1

The biggest remaining levers (post-V2 training):

1. **Instance segmentation post-processing** (watershed for nuc, mito, ves, etc.) — the challenge scores instance seg separately, this is a huge differentiator
2. **nnU-Net** — auto-configured pipeline that consistently wins medical imaging challenges
3. **Multi-resolution predictions** — different organelles have different native resolutions (8nm, 16nm, 32nm)
4. **Pseudo-labeling** — use V2 predictions on unlabeled volumes → fine-tune → 5-10× more training data
5. **Test-time augmentation** — 8 flip combinations, already implemented in inference.py

### Bottom Line

V2 is a **necessary foundation** — we couldn't compete at all with V1 (34/48 classes missing). V2 makes us competitive. Whether it tops the leaderboard depends on:
- How well the models learn the 21 new (often rare) classes
- Quality of the ensemble and post-processing
- What the competition has done

**We have 28 days of reservation left** and a solid pipeline. Even if V2 isn't #1, it gives us the infrastructure for rapid iteration (V3, V4, etc.).

---

## 8. Useful Commands

### Check all jobs (both clusters)
```bash
# Sycamore H100 jobs:
squeue -u gsgeorge
# Longleaf L40S jobs:
ssh longleaf.unc.edu 'squeue -u gsgeorge'
```

### Check training progress
```bash
# H100: SegResNet-Wide
tail -5 /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/segresnet_wide_r2_1690610.err
# H100: FlexUNet
tail -5 /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/flexunet_r2_1690611.err
# L40S: SwinUNETR
tail -5 /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/swinunetr_r2_31457329.err
# L40S: SegResNet
tail -5 /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/segresnet_r2_31462469.err
```

### Cancel jobs
```bash
# Sycamore H100:
scancel 1690610 1690611
# Longleaf L40S:
ssh longleaf.unc.edu 'scancel 31457329 31462469'
```

### Resubmit if crashed
```bash
# H100 (from Sycamore):
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/monai_cellmap
sbatch slurm/train_r2_segresnet_wide_h100.sbatch
sbatch slurm/train_r2_flexunet_h100.sbatch

# L40S reservation (from Longleaf):
ssh longleaf.unc.edu 'cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/monai_cellmap && sbatch slurm/train_r2_segresnet.sbatch'
ssh longleaf.unc.edu 'cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/monai_cellmap && sbatch slurm/train_r2_swinunetr.sbatch'
```

### TensorBoard (V1 + V2 comparison)
```bash
# On Sycamore or via Longleaf OnDemand desktop:
micromamba activate csc
tensorboard --logdir /work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap \
  --host 0.0.0.0 --port 6006 --reload_interval 30
```

### Check H100 partition availability
```bash
sinfo -p h100_mn -N --format="%.20N %.12T %.12e %.12m %.6c %.12G"
```

---

## 9. Known Issues & Gotchas

1. **Two clusters**: H100 jobs on Sycamore (sbatch directly), L40S jobs on Longleaf (via `ssh longleaf.unc.edu`). Shared filesystem means data/code paths are identical.
2. **192³ on H100 is untested**: VRAM estimates are theoretical. If SegResNet-Wide OOMs at 192³ (~65 GB est.), reduce to 160³ in `cfg_segresnet_wide.py` and resubmit. FlexUNet 192³ b4 has more headroom (~55 GB) and is less likely to OOM.
3. **SegResNet 144³ proved unsafe**: Deep supervision + 35 classes + Tversky loss consumed ~45 GB on L40S (not the estimated 37 GB). Config was reduced to 128³. Do NOT increase back to 144³.
4. **`pin_memory=True` causes slowdowns**: Set to False (MONAI #3116)
5. **SpatialPadd required**: Some crops have dimensions < patch size — padding is handled in the dataset
6. **Reservation users**: gsgeorge, robz, jennyw can all submit to reservation `gsgeorge_9034`. jennyw had 3 GPU jobs running earlier.
7. **`.gitignore` blocks `data/`**: Use `git add -f` for Python files inside data/
8. **common_config num_workers=4**: Each model has 2 DDP ranks × 4 workers = 8 workers. On H100 nodes (32 CPUs allocated) this is fine. On L40S (12 CPUs per job) this is tight but workable.
9. **cache_rate=1.0 needs RAM**: 262 NIfTI files cached in RAM. H100 sbatch requests 400g (1.5 TB available). L40S sbatch requests 200g. The original 120g on L40S caused SLURM OOM kills.
10. **H100 node contention**: g15070304 has 1 GPU used by user `shuhang` (job 1674661, 5-day limit). Our job uses 2 GPUs on that node = 3/4 used. g15070306 was fully idle.

---

## 10. Git State

- **Branch**: `main`
- **Key commits**: V1 training code at `53c4403`, V2 changes are uncommitted
- **Uncommitted changes**: Updated configs (35 classes, 192³ H100 patches), `cfg_segresnet_wide.py` (new), `convert_zarr_to_nifti_v2.py` (fixed), H100+L40S sbatch scripts, `inference.py` (group classes), `common_config.py` (35 classes), this handoff doc
- **Recommendation**: Commit V2 changes for a clean restore point
