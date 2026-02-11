# Auto3DSeg: Pipeline Status & Experimental Results

> **Last Updated:** February 10, 2026  
> **Branch:** `feature/auto3dseg-integration`  
> **Project:** CellMap FIB-SEM Segmentation Challenge

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Completed Work](#2-completed-work)
3. [Data Analysis Results](#3-data-analysis-results)
4. [Bundle Generation Results](#4-bundle-generation-results)
5. [Training Configuration Comparison](#5-training-configuration-comparison)
6. [Training Job Status](#6-training-job-status)
7. [Resource & Cluster Analysis](#7-resource--cluster-analysis)
8. [Apply Insights to Existing Pipeline](#8-apply-insights-to-existing-pipeline)
9. [Advanced Customization](#9-advanced-customization)
10. [Cluster Resources](#10-cluster-resources)
11. [Commands Reference](#11-commands-reference)

---

## 1. Pipeline Overview

MONAI Auto3DSeg is a 4-stage automated pipeline for 3D medical/biological image segmentation:

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. Analyze   │───▶│ 2. BundleGen │───▶│ 3. Train     │───▶│ 4. Ensemble  │
│(DataAnalyzer)│    │ (BundleGen)  │    │ (AutoRunner)  │    │(AlgoEnsemble)│
│              │    │              │    │              │    │              │
│ Compute data │    │ Generate     │    │ Sequential   │    │ Rank models, │
│ statistics   │    │ algorithm    │    │ training of  │    │ ensemble top │
│ & report     │    │ bundles with │    │ all algos    │    │ predictions  │
│              │    │ optimized    │    │              │    │              │
│ ✅ COMPLETE  │    │ configs      │    │ 🔄 QUEUED    │    │              │
│              │    │ ✅ COMPLETE  │    │ Job 30139932 │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 2. Completed Work

### ✅ Step 0: Data Conversion (Zarr → NIfTI)
- **Job:** `1650087` on Sycamore `batch` partition
- **Duration:** 31 minutes
- **Result:** 244/245 crops converted to NIfTI format
  - **204 training** samples
  - **40 validation** samples
- **Output:** `auto3dseg/nifti_data/` (2.9 GB) + `auto3dseg/nifti_data/datalist.json`

### ✅ Step 1: Data Analysis (DataAnalyzer)
- **Job:** `1650088` on Sycamore `batch` partition
- **Duration:** 21 minutes, 188 cases analyzed
- **Output:** `auto3dseg/work_dir/datastats.yaml` (15 KB)
- **Output:** `auto3dseg/work_dir/datastats_by_case.yaml` (802 KB)

### ✅ Step 2: Bundle Generation (BundleGen)
- **SegResNet:** Generated on Sycamore login node (CPU) — no GPU needed
- **SwinUNETR:** Generated on Longleaf `l40-gpu` (Job `30139586`, ~2 sec on L40S)
  - Required GPU for `auto_scale()` memory detection
- **DiNTS:** Generated on Longleaf `l40-gpu` (Job `30139610`, ~2 sec on L40S)
  - Required GPU for `auto_scale()` memory detection
- **Modality fix:** Changed from `"EM"` to `"CT"` — MONAI only accepts CT/MRI.
  CT is correct because EM data has a fixed bounded intensity range like CT,
  which maps to `"range"` normalization (percentile clipping → [0,1]).
  MRI would use z-score normalization which is wrong for EM.

### ✅ Infrastructure
- MONAI v1.5.2 + nibabel v5.3.3 installed in `csc` micromamba environment
- SLURM scripts configured for both Sycamore (CPU) and Longleaf (GPU)
- Code pushed to GitHub branch `feature/auto3dseg-integration`

---

## 3. Data Analysis Results

### 3.1 Image Statistics

| Metric | Value |
|--------|-------|
| **Samples** | 204 training + 40 validation |
| **Channels** | 1 (grayscale EM) |
| **Mean Shape** | 393 × 398 × 416 voxels |
| **Min Shape** | 100 × 124 × 180 |
| **Max Shape** | 1796 × 1500 × 2400 |
| **Voxel Spacing** | 8 × 8 × 8 nm (isotropic) |

### 3.2 Intensity Distribution

| Metric | Full Image | Foreground Only |
|--------|-----------|-----------------|
| **Range** | [0, 255] | [0, 255] |
| **Mean ± Std** | 109.8 ± 73.0 | 60.7 ± 66.2 |
| **Median** | 144.8 | 25.7 |
| **0.5th Percentile** | 11.3 | 10.2 |
| **99.5th Percentile** | 194.1 | 187.2 |

**Recommendation:** Clip intensities to **[11, 194]** (0.5th–99.5th percentile) → scale to [0, 1].

### 3.3 Class Distribution (14 classes + background)

| ID | Class | Mean Foreground % | Notes |
|----|-------|-------------------|-------|
| 0 | background | 63.52% | Dominant |
| 1 | ecs | 14.53% | Most common foreground class |
| 2 | pm | 7.00% | |
| 3 | mito_mem | 3.38% | |
| 4 | mito_lum | 3.86% | |
| 5 | mito_ribo | 1.20% | ⚠️ Rare |
| 6 | golgi_mem | 1.30% | ⚠️ Rare |
| 7 | golgi_lum | 3.05% | |
| 8 | ves_mem | 3.87% | |
| 9 | ves_lum | 7.00% | |
| 10 | endo_mem | 4.75% | |
| 11 | endo_lum | 6.35% | |
| 12 | er_mem | 12.58% | Second most common |
| 13 | er_lum | 10.46% | |
| 14 | nuc | 0.00% | ❌ No data in training set |

### 3.4 Key Findings

- **Class imbalance:** 12.2× between most common (`ecs` 14.53%) and rarest (`mito_ribo` 1.20%)
- **`nuc` class is empty:** 0% representation across all training crops — no nuclear annotations exist
- **Background dominates:** 63.5% of all voxels are background
- **Highly variable crop sizes:** 100³ to 2400³ voxels (24× range)
- **Isotropic spacing:** No need for anisotropic patch sizes or resampling
- **Foreground is darker:** Mean FG intensity (60.7) << overall mean (109.8) — structures are dark on bright background (typical of EM)

---

## 4. Bundle Generation Results

### 4.1 Available Templates

MONAI Auto3DSeg ships 4 algorithm templates. We generated 3 (skipping 2D):

| Template | Generated? | Type | Description |
|----------|:---:|------|-------------|
| **segresnet** | ✅ | 3D CNN | Fast encoder-decoder with residual blocks. Strong baseline. |
| **swinunetr** | ✅ | 3D ViT | Shifted-window vision transformer + UNet decoder. SOTA for 3D seg. |
| **dints** | ✅ | NAS | Differentiable Neural Architecture Search. Auto-discovers optimal topology. |
| **segresnet2d** | ❌ | 2D CNN | Slice-wise — irrelevant for isotropic 3D EM data. |

### 4.2 Bundle Output Structure

```
auto3dseg/work_dir/
├── datastats.yaml                    # ✅ Data analysis
├── algorithm_templates/              # Downloaded MONAI templates
│   ├── segresnet/
│   ├── swinunetr/
│   ├── dints/
│   └── segresnet2d/
├── segresnet_0/                      # ✅ Generated bundle
│   ├── configs/hyper_parameters.yaml
│   └── scripts/
├── swinunetr_0/                      # ✅ Generated bundle
│   ├── configs/hyper_parameters.yaml
│   ├── configs/network.yaml
│   ├── configs/transforms_*.yaml
│   └── scripts/
└── dints_0/                          # ✅ Generated bundle
    ├── configs/hyper_parameters.yaml
    ├── configs/hyper_parameters_search.yaml
    ├── configs/network.yaml
    ├── configs/network_search.yaml
    ├── configs/transforms_*.yaml
    └── scripts/
```

---

## 5. Training Configuration Comparison

Auto3DSeg automatically computed optimal hyperparameters for each algorithm based on our dataset statistics:

### 5.1 Architecture & Training

| Setting | SegResNet | SwinUNETR | DiNTS |
|---------|-----------|-----------|-------|
| **Architecture** | SegResNetDS (32 init filters, 5 levels) | SwinUNETR (feature_size=48) | NAS-discovered |
| **ROI / Patch Size** | 224 × 224 × 144 | 96 × 96 × 96 | 96 × 96 × 96 |
| **Batch Size** | 1 | 2 | 2 |
| **Epochs** | 426 | 354 | 200 (+ NAS search phase) |
| **Learning Rate** | 0.0002 | 0.0004 | 0.2 |
| **Optimizer** | AdamW (wd=1e-5) | AdamW (wd=1e-5) | SGD (momentum=0.9, wd=4e-5) |
| **Loss Function** | DiceCE | DiceCE | DiceFocal |
| **LR Schedule** | Default | WarmupCosine | Polynomial (power=0.5) |
| **AMP** | ✅ | ✅ | ✅ |
| **Pretrained** | No | Yes (Swin pretrained weights) | No (architecture is discovered) |
| **Early Stopping** | — | patience=5 | patience=20 |

### 5.2 Data Processing

| Setting | SegResNet | SwinUNETR | DiNTS |
|---------|-----------|-----------|-------|
| **Normalization** | Range [10.2, 187.2] → [0,1] | Range (resample) | Range (resample) |
| **Spacing** | 8 × 8 × 8 nm | 8 × 8 × 8 nm | 8 × 8 × 8 nm |
| **Output Classes** | 15 (14 + background) | 15 | 15 |
| **Softmax/Sigmoid** | Softmax (multi-class) | Softmax | Softmax |
| **Cache Rate** | Auto | 0 (on-the-fly) | 0 (on-the-fly) |

### 5.3 Key Observations

1. **SegResNet uses much larger patches** (224³ vs 96³) — sees more context per sample but can only fit batch=1. Better for large structures (ecs, er).
2. **SwinUNETR leverages pretrained weights** — significant advantage for convergence. Uses sliding window with 6 patches per iteration.
3. **DiNTS uses DiceFocal loss** — better for class imbalance (our rare classes like mito_ribo, golgi_mem). Also uses SGD with high LR + polynomial decay, which can find sharper minima.
4. **All use isotropic spacing** (8nm) — correctly identified from our data analysis.
5. **GPU auto-tuning enabled** — AutoRunner will probe GPU memory at startup and increase batch sizes to fill available VRAM (L40S 48GB).

---

## 6. Training Job Status

### 6.1 Current Job

| Field | Value |
|-------|-------|
| **Job ID** | `30139932` |
| **Partition** | `l40-gpu` (Longleaf) |
| **Status** | `PENDING` (Resources) |
| **Queue Position** | #54 of 154 pending |
| **GPUs** | 2× NVIDIA L40S (48GB each) |
| **CPUs** | 16 |
| **RAM** | 200GB |
| **Walltime** | 11 days |
| **Algorithms** | SegResNet, SwinUNETR, DiNTS (sequential) |
| **Folds** | 1 (explicit train/val split) |
| **Epochs** | Auto-computed per algorithm (426, 354, 200) |

### 6.2 What Will Happen When It Starts

1. **Analysis** — skipped (cached `datastats.yaml` exists)
2. **Bundle Generation** — re-generated on GPU (auto_scale works on L40S)
3. **SegResNet Training** — ~1-3 days, 426 epochs
4. **SwinUNETR Training** — ~2-4 days, 354 epochs, pretrained weights
5. **DiNTS NAS Search + Training** — ~2-5 days, architecture search then 200 epochs
6. **Ensemble** — combines best predictions from all 3 models

**Estimated total: 5-10 days** (with auto-tuning and early stopping, likely closer to 3-5 days)

### 6.3 Monitoring

```bash
# Check job status
ssh longleaf.unc.edu "squeue -j 30139932"

# Tail training log (once running)
ssh longleaf.unc.edu "tail -f /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/auto3dseg_train_30139932.out"

# Check for trained model checkpoints
ls auto3dseg/work_dir/segresnet_0/model/
ls auto3dseg/work_dir/swinunetr_0/model_fold0/
ls auto3dseg/work_dir/dints_0/model_fold0/

# Email notifications configured for BEGIN, END, FAIL → gsgeorge@unc.edu
```

---

## 7. Resource & Cluster Analysis

### 7.1 Cluster Access

| Cluster | Account | GPU Partition | GPUs | Status |
|---------|---------|---------------|------|--------|
| **Longleaf** | `rc_cburch_pi` | `l40-gpu` | L40/L40S (48GB) | ✅ Active — job queued |
| **Longleaf** | `rc_cburch_pi` | `a100-gpu` | A100 PCIe (40GB) | ❌ No partition access |
| **Longleaf** | `rc_cburch_pi` | `volta-gpu` | V100 (16GB) | ❌ Too small for our workload |
| **Sycamore** | `rc_alain_pi` | `h100_mn` | H100 (80GB) | ⚠️ Under maintenance |
| **Sycamore** | `rc_alain_pi` | `batch` | CPU only | ✅ Used for convert/analyze |

### 7.2 GPU Memory Analysis

| Algorithm | Min VRAM Needed | L40S (48GB) | A100 (40GB) | V100 (16GB) |
|-----------|----------------|:-----------:|:-----------:|:-----------:|
| SegResNet (224³, batch=1) | ~12-16 GB | ✅ batch=2-4 | ✅ batch=2-3 | ⚠️ batch=1 |
| SwinUNETR (96³, batch=2) | ~8-12 GB | ✅ batch=4-6 | ✅ batch=3-4 | ⚠️ batch=1 |
| DiNTS (96³, batch=2) | ~10-14 GB | ✅ batch=4 | ✅ batch=3 | ❌ Too tight |

### 7.3 Why Sequential Not Parallel?

We chose sequential training (one job, all 3 algos) over 3 parallel jobs because:
- **Queue is the bottleneck:** L40-gpu has 154 pending jobs. 3 jobs = 3× the wait.
- **Ensemble is automatic:** AutoRunner handles model ranking and ensemble in one job.
- **No competing with yourself:** 3 jobs from the same account don't help scheduler priority.
- **Shared work_dir:** Parallel jobs writing to the same directory could cause conflicts.

---

## 8. Apply Insights to Existing Pipeline

Use the data analysis results to improve your **existing CellMap training code** (in `src/cellmap_segmentation_challenge/`).

### 8.1 Loss Function → `DiceFocalLoss`
```python
from monai.losses import DiceFocalLoss

loss = DiceFocalLoss(
    include_background=False,
    to_onehot_y=True,
    softmax=True,
    focal_weight=[
        1.0,   # ecs (14.53%)
        2.1,   # pm (7.00%)
        4.3,   # mito_mem (3.38%)
        3.8,   # mito_lum (3.86%)
        12.1,  # mito_ribo (1.20%) ← heavy upweight
        11.2,  # golgi_mem (1.30%) ← heavy upweight
        4.8,   # golgi_lum (3.05%)
        3.8,   # ves_mem (3.87%)
        2.1,   # ves_lum (7.00%)
        3.1,   # endo_mem (4.75%)
        2.3,   # endo_lum (6.35%)
        1.2,   # er_mem (12.58%)
        1.4,   # er_lum (10.46%)
        # nuc excluded — 0% data
    ],
    gamma=2.0,
)
```

### 8.2 Normalization → Percentile Clipping
```python
from monai.transforms import ScaleIntensityRanged

normalize = ScaleIntensityRanged(
    keys=["image"],
    a_min=11.0,   # 0.5th percentile
    a_max=194.0,  # 99.5th percentile
    b_min=0.0,
    b_max=1.0,
    clip=True,
)
```

### 8.3 Drop `nuc` Class
Remove `nuc` (class 14) from training — it has 0% representation. Predictions will always be wrong and waste model capacity.

### 8.4 Patch Size
```python
# For L40S (48GB VRAM):
patch_size = (192, 192, 192)  # Aggressive, may need smaller batch
# OR safer:
patch_size = (128, 128, 128)  # Allows larger batch size
```

---

## 9. Advanced Customization

### 9.1 GPU-Aware Optimization
```python
runner.set_gpu_customization(
    gpu_customization=True,
    gpu_customization_specs={
        "universal": {
            "num_trials": 3,
            "range_num_images_per_batch": [1, 4],
            "range_num_sw_batch_size": [1, 8],
        }
    }
)
```

### 9.2 Hyperparameter Optimization (HPO)
```python
runner = AutoRunner(work_dir="auto3dseg/work_dir", input="input.yaml", hpo=True)
runner.set_nni_search_space({
    "learning_rate": {"_type": "choice", "_value": [0.0001, 0.001, 0.01]},
})
runner.run()
```

### 9.3 Ensemble Methods
```python
runner.set_ensemble_method("AlgoEnsembleBestByFold")  # Default
runner.set_ensemble_method("AlgoEnsembleBestN", n_best=3)  # Top N
```

### 9.4 MLflow Tracking
```python
runner = AutoRunner(
    work_dir="auto3dseg/work_dir",
    input="input.yaml",
    mlflow_tracking_uri="./auto3dseg/mlruns",
    mlflow_experiment_name="cellmap-auto3dseg"
)
```

---

## 10. Cluster Resources

### Sycamore (CPU Work)
| Setting | Value |
|---------|-------|
| Partition | `batch` |
| Account | `rc_alain_pi` |
| QOS | `normal` |
| Hardware | 192 CPUs (AMD EPYC 9684X), 1.5 TB RAM |
| Use for | Data conversion, analysis |

### Longleaf (GPU Training)
| Setting | Value |
|---------|-------|
| Partition | `l40-gpu` |
| Account | `rc_cburch_pi` |
| QOS | `gpu_access` |
| Hardware | L40S (48 GB VRAM), up to 8 per node |
| Max walltime | 11 days |
| Use for | Training, ensemble, inference |

### Environment Activation
```bash
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc
```

---

## 11. Commands Reference

```bash
# Project root
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# ── COMPLETED ─────────────────────────────────────────────

# Convert Zarr → NIfTI (Sycamore, ~30 min)
sbatch auto3dseg/auto3dseg_convert.sbatch

# Analyze data (Sycamore, ~20 min)
sbatch auto3dseg/auto3dseg_analyze.sbatch

# Generate bundles (Longleaf GPU, ~2 min each)
sbatch auto3dseg/generate_swinunetr.sbatch
sbatch auto3dseg/generate_dints.sbatch

# ── CURRENT ───────────────────────────────────────────────

# Full pipeline: Train → Ensemble (Longleaf, 3-10 days)
sbatch auto3dseg/auto3dseg_train.sbatch

# ── MONITORING ────────────────────────────────────────────

# Check job status
squeue -j 30139932

# Watch training logs
tail -f logs/auto3dseg_train_30139932.out

# ── AFTER TRAINING ────────────────────────────────────────

# View model scores
python -c "
from monai.apps.auto3dseg import import_bundle_algo_history
history = import_bundle_algo_history('auto3dseg/work_dir', only_trained=True)
for h in history:
    print(f\"{h['name']}: {h.get('best_metric', 'N/A')}\")
"

# Interpret results
python auto3dseg/interpret_results.py --work_dir auto3dseg/work_dir
```
