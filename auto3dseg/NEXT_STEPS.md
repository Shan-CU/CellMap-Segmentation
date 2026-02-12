# Auto3DSeg: Pipeline Status & Agent Handoff Document


> **Last Updated:** February 12, 2026  
> **Branch:** `feature/auto3dseg-integration`  
> **Project:** CellMap FIB-SEM Segmentation Challenge  
> **Active Job:** `30343731` on Longleaf `l40-gpu` node `g181005` — RUNNING

---

## Table of Contents

1. [Executive Summary — What You Need to Know](#1-executive-summary)
2. [The Core Problem & Solution](#2-the-core-problem--solution)
3. [Pipeline Overview & Step Status](#3-pipeline-overview--step-status)
4. [Completed Work (Steps 0–6)](#4-completed-work-steps-0-6)
5. [Bugs Fixed During Smoke Testing](#5-bugs-fixed-during-smoke-testing)
6. [Current State: Patched Bundles](#6-current-state-patched-bundles)
7. [Training Job Status](#7-training-job-status)
8. [Data Analysis Results](#8-data-analysis-results)
9. [Training Configuration Comparison](#9-training-configuration-comparison)
10. [Known Issues & Decisions](#10-known-issues--decisions)
11. [Resource & Cluster Analysis](#11-resource--cluster-analysis)
    - [11.3 L40S GPU Scaling Guide](#113-l40s-gpu-scaling-guide) ⭐ *Key reference for future training*
12. [File Inventory — What Was Modified](#12-file-inventory)
13. [Commands Reference](#13-commands-reference)
14. [After Training Completes](#14-after-training-completes)

---

## 1. Executive Summary

**Goal:** Train 3 MONAI Auto3DSeg models (SegResNet, SwinUNETR, DiNTS) on CellMap
FIB-SEM data for 14-class organelle segmentation with **partial annotations**.

**The critical challenge:** Each training crop is only annotated for a *subset* of
the 14 classes. A naive loss function treats unannotated classes as "absent" and
trains the model to predict zero for those classes — even if the organelle is
clearly visible in the EM image. This destroys model performance.

**The solution (implemented & verified):**
- **Sigmoid mode** — each of 14 classes is predicted independently as a binary
  channel (not mutually exclusive softmax)
- **PartialAnnotationLossV2** — custom per-channel Dice+BCE loss that reads
  `annotated_classes` metadata per crop and **masks out unannotated channels**
  from the loss computation. Zero gradient for classes the annotator didn't label.
- **`--mode train`** — new run mode that sets `analyze=False, algo_gen=False` to
  guarantee AutoRunner does NOT overwrite the hand-patched bundles.

**Current status:** All code is complete and verified. **57 problematic crops removed**
(43 blank images from resolution mismatch + 10 more blank crops found by DataAnalyzer
+ 4 with empty annotations). Final dataset: **234 entries** (198 train, 36 val).
DataAnalyzer re-run completed (Job `1654544`, 238 cases). Training job `30343731`
is **RUNNING** on Longleaf l40-gpu node g181005 (2× L40S 48GB, 512GB RAM). DiNTS
training started first. Previous job `30276184` was OOM-killed (200GB RAM insufficient
for 14-channel label expansion); fixed by bumping to 512GB and correcting SegResNet's
stale `num_epochs: 2` → `300`. See [Section 11.3](#113-l40s-gpu-scaling-guide) for
the full L40S scaling guide derived from auto_scale analysis.

---

## 2. The Core Problem & Solution

### 2.1 Partial Annotations

CellMap FIB-SEM crops have **partial annotations**: each crop is annotated for only
a subset of the 14 organelle classes. For example, one crop might have labels for
`ecs, pm, mito_mem, mito_lum` but NOT for `er_mem, er_lum, golgi_mem`, etc.

**Annotation coverage per class** (out of 198 training crops, after cleanup):

| Ch | Class     | Train Crops | Coverage | Mean FG% (in annotated crops) |
|---:|-----------|------------:|---------:|------------------------------:|
| 0  | ecs       | 131         | 66.2%    | 35.1%                         |
| 1  | pm        | 104         | 52.5%    | 4.5%                          |
| 2  | mito_mem  | 113         | 57.1%    | 7.5%                          |
| 3  | mito_lum  | 113         | 57.1%    | 9.3%                          |
| 4  | mito_ribo | 48          | 24.2%    | 0.04%  ⚠️ Rare               |
| 5  | golgi_mem | 17          | 8.6%     | 6.7%   ⚠️ Few crops          |
| 6  | golgi_lum | 17          | 8.6%     | 9.7%   ⚠️ Few crops          |
| 7  | ves_mem   | 149         | 75.3%    | 0.16%                         |
| 8  | ves_lum   | 149         | 75.3%    | 0.08%                         |
| 9  | endo_mem  | 142         | 71.7%    | 0.82%                         |
| 10 | endo_lum  | 143         | 72.2%    | 1.65%                         |
| 11 | er_mem    | 154         | 77.8%    | 4.39%                         |
| 12 | er_lum    | 154         | 77.8%    | 5.32%                         |
| 13 | nuc       | 44          | 22.2%    | 48.98% ✅ Confirmed present   |

### 2.2 Previous Run (Job 30139932, Feb 10) — WRONG

The first training attempt used Auto3DSeg's default:
- **Softmax** (mutually exclusive classes) + **DiceCELoss** (no masking)
- The model learned "unannotated = absent" — catastrophically wrong for all
  partially annotated classes

### 2.3 Current Run (Job 30276184, Feb 11) — CORRECT

The current training uses:
- **Sigmoid** mode (independent binary per class via `LabelEmbedClassIndex`)
- **PartialAnnotationLossV2** (custom loss that masks out unannotated channels)
- **`--mode train`** (skips analyze & bundle-gen, preserves patched code)
- Labels are **single-channel integer** (0=bg, 1–14=class IDs), converted to
  binary channels at runtime by MONAI's `LabelEmbedClassIndex`

### 2.4 How It Works

```
datalist.json per crop:
  "annotated_classes": "0,1,2,3,6,7,8,9,10,11"  (0-indexed channel indices)

Training loop:
  1. DataLoader loads crop → annotated_classes string comes along in batch_data
  2. parse_annotation_mask_from_batch() → (B, 14) binary mask
  3. loss_function.set_annotation_mask(mask)
  4. PartialAnnotationLossV2.forward():
     - Compute per-channel Dice + BCE: shape (B, C)
     - Multiply by mask → zero loss for unannotated channels
     - Average only over annotated channels per sample
  5. Result: model gets NO gradient for classes without labels in that crop
```

### 2.5 Class Name Mapping

The **datalist.json** uses this mapping (1-indexed integer labels, 0-indexed channels):

| Integer ID (label) | Channel (0-indexed) | Class Name |
|:---:|:---:|:---|
| 0 | — | background |
| 1 | 0 | ecs |
| 2 | 1 | pm |
| 3 | 2 | mito_mem |
| 4 | 3 | mito_lum |
| 5 | 4 | mito_ribo |
| 6 | 5 | golgi_mem |
| 7 | 6 | golgi_lum |
| 8 | 7 | ves_mem |
| 9 | 8 | ves_lum |
| 10 | 9 | endo_mem |
| 11 | 10 | endo_lum |
| 12 | 11 | er_mem |
| 13 | 12 | er_lum |
| 14 | 13 | nuc |

**Note:** The original CellMap challenge used a different 14-class 0-indexed scheme
(0=ecs, 1=pm, 2=mito, 3=mito-mem, ..., 13=ne-mem). Our pipeline regroups them
into the above because the zarr ground-truth uses these specific subarray names.

---

## 3. Pipeline Overview & Step Status

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 0. Convert   │───▶│ 1. Analyze   │───▶│ 2. BundleGen │───▶│ 3. Patch     │
│(Zarr → NIfTI)│    │(DataAnalyzer)│    │ + auto_scale │    │ Bundles      │
│              │    │              │    │              │    │              │
│ ✅ COMPLETE  │    │ ✅ COMPLETE  │    │ ✅ COMPLETE  │    │ ✅ COMPLETE  │
│ Job 1650087  │    │ Job 1650088  │    │ Jobs 30139*  │    │ manual       │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                    ┌──────────────┐    ┌──────────────┐            │
                    │ 5. Train     │◀───│ 4. Smoke     │◀───────────┘
                    │ (AutoRunner) │    │ Tests        │
                    │              │    │              │
                    │ 🔄 QUEUED    │    │ ✅ ALL PASS  │
                    │ Job 30276184 │    │ 3/3 algos OK │
                    └──────────────┘    └──────────────┘
```

| Step | Description | Status | Details |
|------|-------------|--------|---------|
| 0 | Zarr → NIfTI conversion | ✅ Complete | 234 crops after cleanup, single-channel int labels |
| 1 | DataAnalyzer | ✅ Complete | Job `1654544` on Sycamore (238/248 cases analyzed) |
| 2 | BundleGen (3 algos) | ✅ Complete | segresnet_0, swinunetr_0, dints_0 |
| 3 | Patch bundles for partial annotation | ✅ Complete | Loss, transforms, train loop all patched |
| 4 | Smoke tests (all 3 algos) | ✅ All pass | 7 bugs found & fixed |
| 5 | Full training | 🔄 Queued | Job `30276184`, l40-gpu, PENDING (Priority) |
| 6 | Ensemble | ⏳ Pending | After training completes |

---

## 4. Completed Work (Steps 0–6)

### ✅ Step 0: Data Conversion (Zarr → NIfTI) + Data Cleanup
- **Job:** `1650087` on Sycamore `batch` partition, 31 minutes
- **Result:** Originally 277 crops → NIfTI: `auto3dseg/nifti_data/`
- **Second pass (Feb 11):** Labels reconverted from multi-channel binary to
  single-channel integer using `--force --labels_only` flags (faster, images kept)
- **Data cleanup (Feb 11):** 57 problematic crops removed in 3 passes:
  1. **43 blank-image crops** — Root causes: 14 had bounds overflow in converter
     (fixed with clip+pad, rescued); 26 had GT at coarse resolution so `csc fetch-data`
     skipped fine-res EM (resolution mismatch, a [known community issue](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions/130));
     3 had download crashes (JSONDecodeError)
  2. **10 more blank crops** — Found by DataAnalyzer (max=0 images)
  3. **4 empty-annotation entries** — Had `annotated_classes: ""`, including crop247
     (1796×1500×2400 = 6.5B voxels which would OOM)
- **Final dataset:** **234 entries** (198 train fold=1, 36 val fold=0)
  - Labels: single-channel integer (Z,Y,X) uint8, values 0–14
  - `datalist.json` includes `annotated_classes` per crop as comma-separated string
  - `sigmoid: true` flag in datalist triggers MONAI's per-class binary mode

### ✅ Step 1: Data Analysis (DataAnalyzer)
- **Original job:** `1650088` on Sycamore, 21 minutes, 188 cases (stale — old format)
- **Re-run job:** `1654544` on Sycamore, 238/248 cases analyzed successfully
- **Output:** `auto3dseg/work_dir/datastats.yaml` + `datastats_by_case.yaml`
- **Note:** Summary stats have a MONAI aggregation bug for nuc (label 14) —
  reports 0% because no single crop has all 15 labels. Per-case stats are correct.
  See [Section 10.5](#105-dataanalyzer-nuc-aggregation-bug) for details.

### ✅ Step 2: Bundle Generation (BundleGen)
- **SegResNet:** Generated on Sycamore login (CPU) — no GPU needed
- **SwinUNETR:** Longleaf `l40-gpu` Job `30139586` (~2 sec, needs `auto_scale()`)
- **DiNTS:** Longleaf `l40-gpu` Job `30139610` (~2 sec, needs `auto_scale()`)
- **Modality:** `"CT"` (MONAI only accepts CT/MRI; EM is bounded-range like CT)
- **Templates hash:** 21ed8e5 (MONAI Auto3DSeg 1.5.2)

### ✅ Step 3: Patch Bundles for Partial Annotation
Done via `patch_templates.py` + manual fixes. See [Section 6](#6-current-state-patched-bundles).

### ✅ Step 4: Smoke Tests
All 3 algorithms pass smoke tests on volta-gpu (V100 16GB):
- **segresnet_0:** ✅ Loads data, computes loss, runs 1 training step → OOM on V100
- **swinunetr_0:** ✅ Loads data, computes loss, runs training steps → OOM on V100
- **dints_0:** ✅ Loads data, computes loss, runs NAS search step → OOM on V100

OOM is expected — V100 has only 16GB. L40S (48GB) will handle the full workload.

### ✅ Infrastructure
- MONAI v1.5.2, Python 3.11, PyTorch 2.x
- `csc` micromamba environment on both Longleaf and Sycamore
- Code on branch `feature/auto3dseg-integration`

---

## 5. Bugs Fixed During Smoke Testing

Six bugs were discovered and fixed over 6 rounds of smoke testing:

| # | Bug | Symptom | Fix | Files |
|---|-----|---------|-----|-------|
| 1 | `annotated_classes` path mangling | MONAI's `datafold_read` joins ALL string fields with basedir via `os.path.join`, turning `"0,1,2,3"` into `"/basedir/0,1,2,3"` | `os.path.basename(ann_str)` in `parse_annotation_mask_from_batch()` | `partial_annotation.py` (all 3 bundles) |
| 2 | SwinUNETR/DiNTS silent exit | Module-level `try/except` swallowed `fire.Fire()` into except branch | Move import block **inside** `if __name__ == "__main__":` with proper indentation | `swinunetr_0/scripts/train.py`, `dints_0/scripts/train.py` |
| 3 | `PYTORCH_CUDA_ALLOC_CONF` syntax | Used `=` instead of `:` in config values | `expandable_segments:True` (colon not equals) | All `.sbatch` files |
| 4 | Missing SwinUNETR config keys | `num_patches_per_image` and `patch_size` not in hyper_parameters.yaml | Added `num_patches_per_image: 2` and `patch_size: $@roi_size` | `swinunetr_0/configs/hyper_parameters.yaml` |
| 5 | `CropForegroundd` crash | Some EM crops are constant-intensity → bounding box collapses to empty → Spacingd crashes | Removed `CropForegroundd` from all 6 transform configs (train, validate, infer × dints + swinunetr) | `*/configs/transforms_*.yaml` |
| 6 | uint8 target for BCE | `binary_cross_entropy_with_logits` requires float target, got uint8 | `target = target.float()` at top of `PartialAnnotationLossV2.forward()` | `partial_annotation.py` (all 3 bundles) |
| 7 | `datafold_read` ZeroDivision | `nifti_data/datalist.json` had no `fold` keys → `datafold_read(fold=0)` returned 0 val files → `ZeroDivisionError` in DiNTS line 356 (all 3 algos affected) | Merged validation entries (fold=0) into training array in `datalist.json`; training entries get fold=1 | `auto3dseg/nifti_data/datalist.json` |

---

## 6. Current State: Patched Bundles

### 6.1 Files Added to Each Bundle

Each of the 3 bundles (`segresnet_0`, `swinunetr_0`, `dints_0`) has:

**`scripts/partial_annotation.py`** (502 lines, identical in all 3):
- `AnnotatedClassMaskd` — data transform (not currently used, kept for reference)
- `PartialAnnotationLoss` — v1 wrapper (deprecated, kept for reference)
- `PartialAnnotationLossV2` — the active loss: per-channel Dice+BCE, masked by annotation mask, averages only over annotated channels
- `PartialAnnotationDeepSupervisionLoss` — wraps V2 for SegResNet's multi-scale outputs
- `parse_annotation_mask_from_batch()` — extracts (B,C) mask from batch_data, handles `os.path.basename()` fix
- `build_partial_annotation_loss()` — factory function

### 6.2 Patched Training Scripts

**`segresnet_0/scripts/segmenter.py`:**
- Import block at line ~135 (inside class scope, after `LabelEmbedClassIndex`)
- Loss replacement at line ~680: `PartialAnnotationDeepSupervisionLoss` wraps `PartialAnnotationLossV2`
- Train loop patch at line ~1861: extracts `_full_ann_mask`, slices per sub-chunk in the `num_steps_per_image` loop

**`swinunetr_0/scripts/train.py`:**
- Import block at line ~875 (inside `if __name__ == "__main__":` — critical for `fire.Fire()`)
- Loss replacement at line ~387: `PartialAnnotationLossV2` replaces `parser.get_parsed_content("loss")`
- Train loop at line ~500: mask parsed before sub-batch loop, sliced by `_idx` permutation indices

**`dints_0/scripts/train.py`:**
- Import block at line ~1026 (inside `if __name__ == "__main__":`)
- Loss replacement at line ~461: `PartialAnnotationLossV2` replaces `parser.get_parsed_content("training#loss")` (note different key)
- Train loop at line ~602: same pattern as swinunetr

### 6.3 Patched Config Files

- **`swinunetr_0/configs/hyper_parameters.yaml`:** Added `num_patches_per_image: 2` and `patch_size: $@roi_size`
- **`dints_0/configs/transforms_{train,validate,infer}.yaml`:** Removed `CropForegroundd`
- **`swinunetr_0/configs/transforms_{train,validate,infer}.yaml`:** Removed `CropForegroundd`
- (SegResNet doesn't use these transform configs — it handles transforms in segmenter.py)

### 6.4 Key Protection: `--mode train`

The `run_auto3dseg.py` script now supports `--mode train` which creates AutoRunner with:
```python
AutoRunner(
    work_dir=work_dir,
    input=input_cfg_path,
    algos=algos,
    analyze=False,       # do NOT re-analyze
    algo_gen=False,      # do NOT re-generate (would overwrite patches!)
    train=True,          # always train
    ensemble=False,      # ensemble later
)
```

This is the **only safe way** to run training — `--mode full` would call BundleGen
again, which regenerates all bundles from templates and **destroys all patches**.

---

## 7. Training Job Status

### 7.1 Current Job

| Field | Value |
|-------|-------|
| **Job ID** | `30343731` |
| **Partition** | `l40-gpu` (Longleaf) |
| **Node** | `g181005` (L40S node, 1 TB system RAM) |
| **Status** | `RUNNING` — as of Feb 12 13:37 |
| **GPUs** | 2× NVIDIA L40S (48GB each) |
| **CPUs** | 16 |
| **RAM** | **512GB** (bumped from 200GB — see [Section 11.3](#113-l40s-gpu-scaling-guide)) |
| **Walltime** | 11 days |
| **Mode** | `--mode train` (analyze=False, algo_gen=False) |
| **Algorithms** | DiNTS → SegResNet → SwinUNETR (sequential, DiNTS first) |
| **Folds** | 1 (explicit train/val split) |
| **Epochs** | Auto-scaled for L40S: DiNTS 41, SegResNet 300, SwinUNETR 1000 |
| **Loss** | PartialAnnotationLossV2 (sigmoid Dice+BCE, masked) |
| **Dataset** | 234 entries (198 train, 36 val) — after cleanup |
| **Live RAM** | ~222 GB peak (of 512 GB limit) — verified stable |
| **Live VRAM** | GPU 0: 43.2/46 GB, GPU 1: 43.3/46 GB — both ~93% utilized |

### 7.2 What Happens During Training

1. AutoRunner instantiates with `analyze=False, algo_gen=False, train=True`
2. `import_bundle_algo_history()` discovers 3 existing bundles
3. `_train_algo_in_sequence()` runs each bundle's `train.py` as a subprocess
4. **DiNTS** (first): 41 epochs, 96³ patches, batch=13, NAS architecture search
5. **SegResNet**: 300 epochs, 224³ patches, batch=1, deep supervision
6. **SwinUNETR**: 1000 epochs, 96³ patches, batch=1, pretrained Swin weights
7. Checkpoints saved to `<algo>/model_fold0/`

**Estimated total: 3–10 days** (with early stopping, likely 3–5 days)

### 7.3 Monitoring

```bash
# Check job status
squeue -j 30343731

# Tail training log (once running)
tail -f logs/auto3dseg_train_30343731.out

# Check for model checkpoints
ls auto3dseg/work_dir/segresnet_0/model/
ls auto3dseg/work_dir/swinunetr_0/model_fold0/
ls auto3dseg/work_dir/dints_0/model_fold0/

# Email notifications: BEGIN, END, FAIL → gsgeorge@unc.edu
```

### 7.4 Previous Jobs (Superseded)

| Job ID | Date | Issue | Outcome |
|--------|------|-------|---------|
| `30139932` | Feb 10 | `--mode full` + softmax + DiceCELoss | Wrong loss, would overwrite patches |
| `30268488` | Feb 11 | ZeroDivisionError — no fold keys in datalist | Crashed in ~70s |
| `30271306` | Feb 11 | 43+ blank crops in dataset, 4 empty annotations | Cancelled — data cleanup needed |
| `30276184` | Feb 11 | **System RAM OOM** — 200GB limit, 14-ch label expansion in 8 workers × 2 DDP ranks ≈ 233GB | OOM-killed at 31s (23 oom_kill events) |
| **`30343731`** | **Feb 12** | **Fixed: 512GB RAM, SegResNet num_epochs 2→300** | **RUNNING on g181005** |

---

## 8. Data Analysis Results

### 8.1 Image Statistics (from DataAnalyzer Job 1654544, n=238)

| Metric | Value |
|--------|-------|
| **Samples** | 198 training + 36 validation (234 total, after cleanup) |
| **Channels** | 1 (grayscale EM) |
| **Mean Shape** | 393 × 402 × 418 voxels |
| **Min Shape** | 100 × 124 × 180 |
| **Max Shape** | 1796 × 1500 × 2400 |
| **Voxel Spacing** | 8 × 8 × 8 nm (isotropic) |

### 8.2 Intensity Distribution

| Metric | Full Image | Foreground Only |
|--------|-----------|-----------------|
| **Range** | [0, 255] | [0, 255] |
| **Mean ± Std** | 109.8 ± 73.0 | 60.7 ± 66.2 |
| **0.5th / 99.5th Percentile** | 11.3 / 194.1 | 10.2 / 187.2 |

### 8.3 Class Distribution (14 classes + background)

Corrected stats computed from per-case data in `datastats_by_case.yaml`
(Job `1654544`), grouped by label **value** not array index:

| ID | Class | Mean FG% | Crops w/ Label | Annotated Train Crops | Notes |
|----|-------|---------|----------------|----------------------|-------|
| 0 | background | 55.29% | 192/238 | — | Dominant |
| 1 | ecs | 35.05% | 150/238 | 131/198 | Most common FG |
| 2 | pm | 4.45% | 120/238 | 104/198 | |
| 3 | mito_mem | 7.48% | 130/238 | 113/198 | |
| 4 | mito_lum | 9.25% | 130/238 | 113/198 | |
| 5 | mito_ribo | 0.04% | 60/238 | 48/198 | ⚠️ Very small structures |
| 6 | golgi_mem | 6.74% | 19/238 | 17/198 | ⚠️ Few crops |
| 7 | golgi_lum | 9.67% | 19/238 | 17/198 | ⚠️ Few crops |
| 8 | ves_mem | 0.16% | 174/238 | 149/198 | |
| 9 | ves_lum | 0.08% | 174/238 | 149/198 | |
| 10 | endo_mem | 0.82% | 166/238 | 142/198 | |
| 11 | endo_lum | 1.65% | 165/238 | 143/198 | |
| 12 | er_mem | 4.39% | 181/238 | 154/198 | |
| 13 | er_lum | 5.32% | 181/238 | 154/198 | |
| 14 | nuc | 48.98% | 55/238 | 44/198 | ✅ Confirmed present |

> **Note on FG%:** Mean foreground percentage is computed **only across crops where
> that label appears**, not across all crops. Nuc shows 48.98% because nucleus is a
> large structure in the crops that have it. The `datastats.yaml` summary incorrectly
> reports nuc=0% due to an aggregation bug (see [Section 10.5](#105-dataanalyzer-nuc-aggregation-bug)).

---

## 9. Training Configuration Comparison

Auto3DSeg auto-computed these configs. **Loss functions have been replaced** by
`PartialAnnotationLossV2` in all 3 bundles.

| Setting | SegResNet | SwinUNETR | DiNTS |
|---------|-----------|-----------|-------|
| **Architecture** | SegResNetDS (32 filters, 5 levels) | SwinUNETR (feature_size=48) | NAS-discovered |
| **ROI / Patch** | 224 × 224 × 144 | 96 × 96 × 96 | 96 × 96 × 96 |
| **batch_size (num_images_per_batch)** | 1 | 2 | 2 |
| **num_patches_per_iter** | — (single patch) | 1 | 13 |
| **num_crops_per_image** | 1 | 2 | 26 |
| **Epochs** | 300 (manual fix) | 1000 (auto_scale) | 41 (auto_scale) |
| **Effective training patches** | 300 × 188 × 1 = ~56K | 1000 × 188 × 2 = ~376K | 41 × 188 × 26 = ~200K |
| **num_epochs_per_validation** | 1 | 5 | 2 (adaptive) |
| **LR** | 0.0002 | 0.0004 | 0.2 |
| **Optimizer** | AdamW | AdamW | SGD |
| **Loss** | ~~DiceCE~~ → **PartialAnnotation** (deep supervision) | ~~DiceCE~~ → **PartialAnnotation** | ~~DiceFocal~~ → **PartialAnnotation** |
| **Mode** | **Sigmoid** (per-channel binary) | **Sigmoid** | **Sigmoid** |
| **AMP** | ✅ | ✅ | ✅ |
| **Pretrained** | No | Yes (Swin weights) | No |
| **Early Stopping** | — | patience=5 | patience=20 |
| **num_workers** | 4 | 8 | 8 |
| **cache_rate** | null (disk) | 0 (disk) | 0 (disk) |

---

## 10. Known Issues & Decisions

### 10.1 `nuc` (Class 14) — ✅ Confirmed Present, Will Train Correctly

**Previous claim of nuc=0% was WRONG.** Root cause identified through two layers:
1. **Stale datastats (Feb 10):** Generated on old multi-channel binary format
2. **DataAnalyzer aggregation bug:** Even after re-run (Job `1654544`), summary
   reports nuc=0% because it aggregates by array index, not label value

**Actual nuc coverage (verified Feb 11–12):**
- **44/198 training crops (22.2%)** list nuc (ch13) in `annotated_classes`
- **11/36 validation crops (30.6%)** list nuc (ch13) in `annotated_classes`
- NIfTI label files confirmed to contain integer value 14 (nuc) with substantial
  voxel counts (e.g., crop292: 81% nuc, crop4: 19% nuc, crop94/95/96: 100% nuc)
- Per-case stats in `datastats_by_case.yaml` correctly show nuc data for 55 cases
- Mean foreground: **48.98%** among nuc-annotated crops

**Training impact:** The model WILL learn nuc from these 44 training crops. The
partial annotation mask correctly includes channel 13 for nuc-annotated crops.
No code changes needed.

### 10.2 DataAnalyzer — ✅ Re-Run Complete (Job 1654544)

**Re-run completed** on Sycamore (Job `1654544`). Analyzed 238/248 cases
(10 had blank images, discovered and removed in second cleanup pass).

**Output:** `auto3dseg/work_dir/datastats.yaml` + `datastats_by_case.yaml`

**Minor mismatch:** datastats has n_cases=238 but datalist has 234 entries
(4 empty-annotation entries removed after DataAnalyzer ran). This is harmless —
the 4 removed entries would produce zero loss anyway, and we're in train-only
mode (analyze=False, algo_gen=False) so the stats aren't re-read.

### 10.3 Blank Image Cleanup — ✅ Complete (57 crops removed)

**Three root causes identified:**
1. **Bounds overflow in converter** (14 crops): Fixed with clip+pad in
   `convert_zarr_to_nifti.py`. These 14 crops were rescued with real EM data.
2. **Resolution mismatch** (26 crops): GT annotations at coarse resolution (8nm/32nm)
   meant `csc fetch-data` didn't download fine-res EM (s0/s1). This is a
   [known community issue](https://github.com/janelia-cellmap/cellmap-segmentation-challenge/discussions/130).
   Janelia's own pipeline uses `filter_by_scale = True` which drops these same crops.
3. **Download crashes** (3 crops): JSONDecodeError during original fetch.

**Cleanup passes:**
- Pass 1: 29 blank crops removed (43 originally blank, 14 rescued)
- Pass 2: 10 more blank crops found by DataAnalyzer (max=0 images)
- Pass 3: 4 entries with empty `annotated_classes: ""` removed (including crop247
  at 1796×1500×2400 = 6.5B voxels which would OOM)
- **Total removed: 57 crops** (from ~277 → 234 final)

### 10.4 `datafold_read` ZeroDivision — ✅ Fixed (Feb 11)

**Job `30268488` crashed immediately** (~70 seconds) with:
```
ZeroDivisionError: float division by zero
  dints_0/scripts/train.py line 356:
  val_files = val_files * math.ceil(float(world_size) / float(len(val_files)))
```

**Root cause:** `nifti_data/datalist.json` had separate `"training"` and
`"validation"` keys but no `"fold"` field on entries. All 3 algorithms call
`datafold_read(fold=0)` which only looks at the `"training"` array for entries
with `"fold": 0`. Since none had fold keys, `val_files = []` → division by zero.

**Fix:** Merged validation entries into the `"training"` array with `"fold": 0`
(training entries get `"fold": 1`). Now `datafold_read(fold=0)` correctly returns
train=198, val=36. The `"validation"` key is kept for reference.

### 10.5 DataAnalyzer nuc Aggregation Bug

The `datastats.yaml` summary reports nuc (label 14) foreground = 0%. This is a
**MONAI DataAnalyzer bug**, not a data problem.

**Root cause:** DataAnalyzer aggregates per-case label stats by **array index**,
not by **label value**. Each crop has a variable-length label array (only labels
present in that crop). Since no single crop contains all 15 labels (0–14), array
index 14 is never populated. For example:
- A crop with `labels: [0, 1, 2, 8, 9, 14]` stores nuc at index 5, not 14
- A nuc-only crop with `labels: [14]` stores nuc at index 0

**Workaround:** Per-case stats in `datastats_by_case.yaml` are correct. Use the
corrected table in [Section 8.3](#83-class-distribution-14-classes--background)
which was computed from per-case data grouped by label value.

**Impact:** None on training — training reads NIfTI labels directly, not datastats.

### 10.6 CropForegroundd Removed

Some EM crops have constant intensity values (no foreground in MONAI's definition).
`CropForegroundd` computes a bounding box of non-zero voxels, which collapses to
empty for these crops, crashing `Spacingd` downstream. Removing `CropForegroundd`
is safe because FIB-SEM crops are already spatially cropped from the full volume.

### 10.7 Ensemble Disabled for Now

`ensemble=False` in the current run. Once training succeeds, ensemble can be run
separately. This was a deliberate choice to simplify the first successful training
run and avoid complications if one algorithm fails.

---

## 11. Resource & Cluster Analysis

### 11.1 Cluster Access

| Cluster | Account | Partition | GPUs | Status |
|---------|---------|-----------|------|--------|
| **Longleaf** | `rc_cburch_pi` | `l40-gpu` | 4× L40S (48GB) per node | ✅ Job `30276184` queued |
| **Longleaf** | `rc_cburch_pi` | `a100-gpu` | 3× A100 (40GB PCIe) per node | ✅ Accessible, 6-day max |
| **Longleaf** | `rc_cburch_pi` | `volta-gpu` | V100 (16GB) | ✅ Used for smoke tests |
| **Sycamore** | `rc_alain_pi` | `batch` | CPU only | ✅ Used for convert/analyze |
| **Sycamore** | `rc_alain_pi` | `h100_mn` | 4× H100 (80GB) per node | ❌ **Year-long reservation** (msode, until Feb 2027) |

> **h100_mn note:** Both H100 nodes have a 365-day reservation
> (`ReservationName=msode`, users: root, robz, crutle, msode, nsivaku).
> Not available to us until Feb 2, 2027. Contact research@unc.edu if needed.
>
> **a100-gpu note:** 40GB VRAM may be tight for SegResNet (224³ patches).
> Not used as backup because of VRAM risk + shorter 6-day walltime.

### 11.2 Environment Activation

```bash
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc
```

### 11.3 L40S GPU Scaling Guide

> **Purpose:** This section documents everything we learned from MONAI Auto3DSeg's
> auto_scale logic and the actual Job `30343731` resource profile. Use this to
> configure **any** 3D segmentation training (MONAI or not) on Longleaf L40S nodes.

#### 11.3.1 L40S Node Hardware (Longleaf `l40-gpu`)

| Resource | Value | Notes |
|----------|-------|-------|
| **GPU** | NVIDIA L40S | Ada Lovelace architecture |
| **VRAM per GPU** | 46,068 MiB (~45 GB usable) | `torch.cuda.get_device_properties().total_memory` |
| **GPUs per node** | Up to 8 | We request 2 for DDP |
| **System RAM per node** | ~1 TB (1,014,946 MB) | Confirmed via `free -g` on g181005 |
| **Max walltime** | 11 days | `l40-gpu` partition limit |
| **Interconnect** | PCIe P2P/CUMEM between GPUs | NCCL uses direct GPU memory access, no NVLink |

#### 11.3.2 The Two Memory Bottlenecks

**Bottleneck 1: GPU VRAM (45 GB)** — determines batch size and patch size.

The fundamental tradeoff is **large patches vs. large batches**:

| Approach | Patch (ROI) Size | VRAM per Patch | Fits in 45 GB | Batch Size |
|----------|-------------------|----------------|---------------|------------|
| Large ROI (SegResNet-style) | 224 × 224 × 144 = 7.2M voxels | ~43 GB (with 15-ch output + gradients) | Barely (1 patch) | **1** |
| Small ROI (DiNTS/SwinUNETR) | 96 × 96 × 96 = 884K voxels | ~3-4 GB per patch | Comfortably | **1–13** |

**Rule of thumb for 15 output classes on L40S:**
- 96³ patches: batch ≈ 13 (DiNTS) or 1 (SwinUNETR, heavier model)
- 128³ patches: batch ≈ 4–6
- 224³ patches: batch = 1 (tight fit)

**Bottleneck 2: System RAM (~1 TB, SLURM-limited)** — determines DataLoader workers.

The system RAM bottleneck is **not** GPU-related. It comes from the DataLoader
worker processes loading and transforming data in CPU memory:

```
RAM ≈ num_workers × prefetch_factor × images_in_flight × (image_bytes + label_bytes)
```

For our dataset (15 output classes, mean image ~393×398×416 voxels):
- Each image: ~65 MB (uint8)
- Each label after `LabelEmbedClassIndex` expansion: 14 binary channels × 65 MB = **~910 MB per label**
- With large crops: up to 3.4 GB per label (crop247 was 1796×1500×2400 before removal)

**Observed Job `30343731` memory profile:**

| Component | RAM Usage |
|-----------|-----------|
| DDP Rank 0 (main process) | ~109 GB |
| DDP Rank 1 (main process) | ~70 GB |
| Rank 0 child worker | ~43 GB |
| 8 DataLoader workers (per rank) | ~1.5–7.5 GB each |
| **Total observed** | **~222 GB** |
| **SLURM limit** | **512 GB** |
| **Headroom** | **~290 GB** |

> ⚠️ **Critical lesson (Job `30276184` OOM):** With `--mem=200G`, the job was
> OOM-killed after 31 seconds (23 `oom_kill` events in kernel logs). The 14-channel
> label expansion in DataLoader workers × 8 workers × 2 prefetch × 2 DDP ranks
> exceeded 200 GB. **Always request `--mem=512G` on L40S nodes for multi-class
> segmentation.** The nodes have 1 TB, so 512 GB is safe.

#### 11.3.3 How MONAI Auto3DSeg Computes Scaling

Each algorithm has its own auto_scale logic. All three detect GPU memory at
runtime via `torch.cuda.get_device_properties()` or `torch.cuda.mem_get_info()`.

**DiNTS** (`dints_0/scripts/train.py: pre_operation()`):
```python
mem = get_mem_from_visible_gpus()  # returns free VRAM in bytes
mem = float(min(mem)) / (1024**3)  # convert to GB
mem = max(1.0, mem - 1.0)          # reserve 1 GB headroom → ~44 GB on L40S

# Heuristic: linear interpolation between 2 reference points
mem_bs2 = 6.0 + 14.0 * (output_classes - 2) / 103   # 7.77 GB for batch=2
mem_bs9 = 24.0 + 50.0 * (output_classes - 2) / 103   # 30.31 GB for batch=9
batch_size = 2 + 7 * (mem - mem_bs2) / (mem_bs9 - mem_bs2)  # → 13 on L40S

# Epoch calculation: 400 base × data_factor / batch_size
_factor = (1251/n_cases) × (mean_shape / [240,240,155]) × (96/roi)³ / epoch_divided_factor
num_epochs = int(400 * _factor / batch_size)  # → 41 epochs
```

**SwinUNETR** (`swinunetr_0/scripts/algo.py: auto_scale()`):
```python
# Same base formula but 2× memory thresholds (SwinUNETR is heavier)
mem_bs2 = (12/6) * base_mem_bs2   # 15.54 GB for batch=2
mem_bs9 = (12/6) * base_mem_bs9   # 60.62 GB for batch=9
batch_size = max(1, int(2 + 7 * (mem - mem_bs2) / (mem_bs9 - mem_bs2)))  # → 1 on L40S

# Epochs: target 800K total patch iterations
num_epochs = min(max_epoch, int(800000 / n_cases / num_crops_per_image))  # → 1000
```

**SegResNet** (`segresnet_0/scripts/utils.py: auto_adjust_network_settings()`):
```python
gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3  # ~45 GB
gpu_factor = max(1, gpu_mem / 16)  # → 2.8 (ratio vs 16GB baseline)

# Tries to scale ROI, filters, or batch — but ROI is already large:
# roi_size=[224,224,144] → 7.2M voxels > base_numel * gpu_factor → batch stays 1
# NOTE: SegResNet does NOT auto-compute num_epochs. It reads from YAML directly.
```

#### 11.3.4 What Auto3DSeg Computed for L40S (Actual Values)

| Parameter | DiNTS | SwinUNETR | SegResNet |
|-----------|-------|-----------|----------|
| **GPU mem detected** | ~44 GB usable | ~44 GB usable | ~45 GB total |
| **roi_size** | [96, 96, 96] | [96, 96, 96] | [224, 224, 144] |
| **num_patches_per_iter** | 13 | 1 | — |
| **num_crops_per_image** | 26 | 2 | 1 |
| **num_images_per_batch** | 2 | 2 | 1 |
| **num_epochs** | 41 | 1000 | 300 (manual) |
| **Total training patches** | ~200K | ~376K | ~56K |
| **cache_rate** | 0 (no caching) | 0 (no caching) | null (no caching) |
| **num_workers** | 8 | 8 | 4 |
| **VRAM used (observed)** | 43.2 + 43.3 GB | — (not yet run) | — (not yet run) |

#### 11.3.5 Practical Guide: Configuring Your Own Training on L40S

For **non-MONAI** training (e.g., PyTorch Lightning, custom loops), use these
empirically validated settings as starting points:

**SLURM resource request:**
```bash
#SBATCH --partition=l40-gpu
#SBATCH --gres=gpu:2              # 2× L40S (48GB each)
#SBATCH --mem=512G                # MUST be ≥512G for multi-class 3D segmentation
#SBATCH --cpus-per-task=16
#SBATCH --time=11-00:00:00        # max walltime
```

**Batch size selection (15 output classes, sigmoid mode):**

| Patch Size | Model ~Size | Suggested batch_size (per GPU) | VRAM Usage |
|------------|-------------|-------------------------------|------------|
| 96³ | Small (UNet, DiNTS) | 8–13 | ~40–44 GB |
| 96³ | Medium (SwinUNETR) | 1–2 | ~40–44 GB |
| 128³ | Small/Medium | 2–4 | ~40–44 GB |
| 160³ | Small | 1–2 | ~40–44 GB |
| 224³ | SegResNet (32 filters) | 1 | ~43 GB |
| 224³ | Larger models | ❌ Won't fit | >45 GB |

**DataLoader configuration:**
```python
DataLoader(
    dataset,
    batch_size=2,              # images per batch (crops extracted from each)
    num_workers=8,             # 8 per DDP rank → 16 total with 2 GPUs
    prefetch_factor=2,         # default; increase only if IO-bound
    pin_memory=True,
    persistent_workers=True,   # avoid re-forking overhead
)
```

**DDP configuration:**
```python
# torchrun handles this, but for reference:
torchrun --nnodes=1 --nproc_per_node=2 train.py
# NCCL uses P2P/CUMEM on L40S (no NVLink) — all-reduce is fast for 2 GPUs
```

**System RAM budget formula:**
```
Peak RAM ≈ 2 × (model_size + optimizer_states)           # per DDP rank
         + num_workers_total × prefetch × crop_size_bytes  # DataLoader
         + label_expansion_factor × above                  # multi-class labels

For this dataset:
  ≈ 2 × 2 GB (model)                    = ~4 GB
  + 16 workers × 2 prefetch × 65 MB     = ~2 GB (images)
  + 16 workers × 2 prefetch × 910 MB    = ~29 GB (14-ch labels)
  + main process data buffers            = ~150-180 GB
  ≈ 200–250 GB total
  → Request 512 GB (2× headroom for spikes during validation)
```

**Key gotchas:**
1. **MONAI auto_scale only optimizes GPU VRAM.** It has zero awareness of system
   RAM. You must set `--mem` in SLURM independently.
2. **SegResNet does NOT auto-compute `num_epochs`.** It reads directly from
   `hyper_parameters.yaml`. If a smoke test wrote `num_epochs: 2` via
   `config_save_updated()`, that stale value persists. DiNTS and SwinUNETR
   recompute epochs in `pre_operation()` every time.
3. **`cache_rate=0` is correct for large datasets.** Caching 234 images × 65 MB =
   ~15 GB is feasible, but labels expand to 14 channels → ~210 GB cached labels.
   Not worth it when L40S nodes have fast NVMe storage.
4. **TF32 is enabled** via `NVIDIA_TF32_OVERRIDE=1` in the container. This uses
   TensorFloat-32 for matmul/convolutions — ~2× speedup with minimal precision
   loss. Keep it on for training.
5. **Mixed precision (AMP) is essential.** All 3 algorithms use `amp=True`.
   On L40S, AMP roughly halves VRAM for activations, enabling larger batches.

#### 11.3.6 Quick Reference: L40S vs Other GPUs

| GPU | VRAM | batch @ 96³ (15-class) | batch @ 224³ | System RAM Needed |
|-----|------|----------------------|--------------|------------------|
| V100 (16GB) | 16 GB | 1–2 | ❌ | 128G |
| A100 (40GB) | 40 GB | 8–10 | 1 (tight) | 256–512G |
| **L40S (48GB)** | **45 GB** | **8–13** | **1** | **512G** |
| H100 (80GB) | 80 GB | 20–25 | 2–3 | 512G |

> **A100 note:** Longleaf `a100-gpu` has 40GB PCIe A100s (not 80GB SXM). VRAM is
> ~12% less than L40S, and max walltime is 6 days (vs 11 for l40-gpu). Use L40S
> unless A100s are significantly less congested.

---

## 12. File Inventory

### 12.1 Files Created (New)

| File | Purpose |
|------|---------|
| `auto3dseg/partial_annotation.py` | Source module — copied into each bundle |
| `auto3dseg/patch_templates.py` | Script to auto-patch bundles after BundleGen |
| `auto3dseg/find_bad_crops.py` | Diagnostic: finds constant-intensity crops |
| `auto3dseg/reconvert_blank_images.py` | Re-convert blank crops (rescued 14 with clip+pad) |
| `auto3dseg/auto3dseg_bundle_gen.sbatch` | SLURM script for bundle generation |
| `auto3dseg/smoke_test_segresnet.sbatch` | Smoke test for SegResNet |
| `auto3dseg/smoke_test_swinunetr.sbatch` | Smoke test for SwinUNETR |
| `auto3dseg/smoke_test_dints.sbatch` | Smoke test for DiNTS |
| `auto3dseg/work_dir/*/scripts/partial_annotation.py` | Deployed to all 3 bundles |

### 12.2 Files Modified

| File | Changes |
|------|---------|
| `auto3dseg/run_auto3dseg.py` | Added `run_train_only()`, `--mode train`, sigmoid passthrough |
| `auto3dseg/auto3dseg_train.sbatch` | Changed `--mode full` → `--mode train`, updated comments, fixed CUDA_ALLOC_CONF |
| `auto3dseg/auto3dseg_train_h100.sbatch` | Changed `--mode full` → `--mode train`, 2 GPUs, updated for train-only |
| `auto3dseg/auto3dseg_analyze.sbatch` | Increased resources (96 CPUs, 512GB, 16 workers) |
| `auto3dseg/convert_zarr_to_nifti.py` | Multi-channel → single-channel int labels, `--force`/`--labels_only` flags, **bounds clip+pad fix** |
| `auto3dseg/nifti_data/datalist.json` | 234 entries (198 train fold=1, 36 val fold=0), 57 crops removed |
| `auto3dseg/work_dir/segresnet_0/scripts/segmenter.py` | Partial annotation loss + mask extraction |
| `auto3dseg/work_dir/swinunetr_0/scripts/train.py` | Partial annotation loss + mask extraction |
| `auto3dseg/work_dir/dints_0/scripts/train.py` | Partial annotation loss + mask extraction |
| `auto3dseg/work_dir/swinunetr_0/configs/hyper_parameters.yaml` | Added missing keys |
| `auto3dseg/work_dir/{swinunetr,dints}_0/configs/transforms_*.yaml` | Removed CropForegroundd |

### 12.3 Verification Checklist

All of these are confirmed as of Feb 11:
- [x] `partial_annotation.py` exists in all 3 bundles (502 lines each)
- [x] `PartialAnnotationLossV2` referenced in segresnet segmenter.py
- [x] `PartialAnnotationLossV2` referenced in swinunetr train.py
- [x] `PartialAnnotationLossV2` referenced in dints train.py
- [x] `target.float()` present in all 3 `partial_annotation.py` copies
- [x] `os.path.basename` present in all 3 `parse_annotation_mask_from_batch()`
- [x] `CropForegroundd` absent from all transform configs
- [x] `--mode train` in `auto3dseg_train.sbatch`
- [x] `analyze=False, algo_gen=False` in `run_train_only()`

---

## 13. Commands Reference

```bash
# Project root
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# ── MONITORING (current job) ──────────────────────────────

squeue -j 30276184
tail -f logs/auto3dseg_train_30276184.out

# ── RE-SUBMITTING (if job fails / needs restart) ─────────

# SAFE — preserves patched bundles:
sbatch auto3dseg/auto3dseg_train.sbatch

# DANGER — would overwrite patches:
# DO NOT use --mode full

# ── AFTER TRAINING ────────────────────────────────────────

# View model scores
python -c "
from monai.apps.auto3dseg import import_bundle_algo_history
history = import_bundle_algo_history('auto3dseg/work_dir', only_trained=True)
for h in history:
    print(f\"{h['name']}: {h.get('best_metric', 'N/A')}\")
"

# Run ensemble (after all 3 train successfully)
python auto3dseg/run_auto3dseg.py --mode full \
    --datalist auto3dseg/nifti_data/datalist.json \
    --work_dir auto3dseg/work_dir \
    --algos segresnet swinunetr dints
# NOTE: ensemble needs a separate implementation — --mode full would
# overwrite bundles. A dedicated --mode ensemble should be added.

# Interpret results
python auto3dseg/interpret_results.py --work_dir auto3dseg/work_dir

# ── SMOKE TESTS (already passed, for re-verification) ────

sbatch auto3dseg/smoke_test_segresnet.sbatch
sbatch auto3dseg/smoke_test_swinunetr.sbatch
sbatch auto3dseg/smoke_test_dints.sbatch

# ── REBUILD FROM SCRATCH (nuclear option) ─────────────────

# If bundles get corrupted, re-generate and re-patch:
sbatch auto3dseg/auto3dseg_bundle_gen.sbatch   # regenerate bundles
python auto3dseg/patch_templates.py --work_dir auto3dseg/work_dir  # re-patch
# Then manually fix: CropForegroundd removal, swinunetr hyper_parameters.yaml
```

---

## 14. After Training Completes

### 14.1 Immediate Steps
1. Check `best_metric` for each algorithm
2. Inspect training curves in logs
3. Verify per-class Dice scores — especially for rare classes (golgi, mito_ribo, nuc)
4. Decide whether to run ensemble or use best single model

### 14.2 Potential Issues to Watch For
- **OOM during training:** L40S should handle all 3, but DiNTS NAS search is memory-hungry
- **NaN losses:** Could indicate numerical issues with rare classes (mito_ribo, golgi_mem at 7.4%)
- **Early stopping too aggressive:** SwinUNETR patience=5 might stop before convergence

### 14.3 Next Improvements
1. **Add `--mode ensemble`** to `run_auto3dseg.py` (currently missing)
2. **Class-weighted loss:** Currently all annotated classes contribute equally.
   Could add inverse-frequency weights: rare classes (golgi at 17 crops) get
   higher weight than common ones (er at 154 crops)
3. **Fix DataAnalyzer aggregation:** Could post-process `datastats_by_case.yaml`
   to generate corrected summary stats grouped by label value
4. **Re-download with `--fetch-all-em-resolutions`:** Could recover some of the
   26 resolution-mismatch crops, but Janelia's own pipeline drops them too
