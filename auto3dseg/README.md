# Auto3DSeg Integration for CellMap Segmentation Challenge

This directory contains scripts to use [MONAI Auto3DSeg](https://monai.readthedocs.io/en/stable/auto3dseg.html)
to automatically analyze your CellMap FIB-SEM data and determine the best model
architectures, hyperparameters, and training strategies for your specific organelle
segmentation classes.

## Overview

Auto3DSeg is a comprehensive MONAI pipeline that:
1. **Analyzes** your dataset (intensity distributions, shapes, spacing, label statistics)
2. **Generates** optimized algorithm bundles (SegResNet, DiNTS, SwinUNETR) tailored to your data
3. **Trains** multiple models with cross-validation
4. **Ensembles** the best models for maximum accuracy

## Pipeline Steps

### Step 1: Convert CellMap Zarr Data → NIfTI

Auto3DSeg expects NIfTI (`.nii.gz`) files. Our converter extracts 3D crops from
the zarr volumes and saves paired image/label NIfTI files.

```bash
python auto3dseg/convert_zarr_to_nifti.py \
    --datasplit datasplit.csv \
    --output_dir auto3dseg/nifti_data \
    --scale s0 \
    --target_spacing 8 8 8 \
    --max_crops 0
```

Options:
- `--scale`: Which zarr scale level to read (default: `s0` for highest resolution).
  Use `s1` or `s2` for faster initial experiments.
- `--target_spacing`: Voxel spacing in nm (default: 8 8 8 to match existing training)
- `--max_crops`: Limit number of crops to convert (0 = all, useful for testing)

### Step 2: Run Data Analysis Only

To just analyze your data without training (fast, ~10 min):

```bash
python auto3dseg/run_auto3dseg.py \
    --mode analyze \
    --work_dir auto3dseg/work_dir \
    --datalist auto3dseg/nifti_data/datalist.json
```

This produces `auto3dseg/work_dir/datastats.yaml` with detailed statistics about
your dataset including intensity distributions, shape/spacing stats, and label
class distributions. Review this to understand your data characteristics.

### Step 3: Run Full Auto3DSeg Pipeline

```bash
python auto3dseg/run_auto3dseg.py \
    --mode full \
    --work_dir auto3dseg/work_dir \
    --datalist auto3dseg/nifti_data/datalist.json \
    --num_fold 5 \
    --num_epochs 100 \
    --algos segresnet swinunetr
```

### Step 4: Run on SLURM Cluster

```bash
# Analyze data only (quick job, ~30 min)
sbatch auto3dseg/auto3dseg_analyze.sbatch

# Full pipeline (long job, multi-day)
sbatch auto3dseg/auto3dseg_train.sbatch
```

## Output Structure

```
auto3dseg/
├── nifti_data/               # Converted NIfTI files
│   ├── images/               # EM image volumes
│   ├── labels/               # Multi-class label volumes
│   └── datalist.json         # Auto3DSeg-compatible data list
├── work_dir/                 # Auto3DSeg output
│   ├── datastats.yaml        # Dataset analysis report ← REVIEW THIS
│   ├── algorithm_templates/  # Generated algo templates
│   ├── segresnet_0/          # SegResNet bundle (fold 0)
│   ├── swinunetr_0/          # SwinUNETR bundle (fold 0)
│   └── ensemble_output/      # Final ensemble predictions
└── logs/                     # SLURM logs
```

## Key Considerations for CellMap Data

1. **Multi-class labels**: The converter creates a single integer-labeled volume
   where each class gets a unique integer ID (background=0). Auto3DSeg handles
   multi-class segmentation natively.

2. **Spacing**: CellMap data is in nanometers. We set the NIfTI spacing to match
   the voxel size used in training (default 8nm isotropic).

3. **Scale selection**: For quick experiments, use `--scale s1` (4nm) or `s2` (8nm)
   during conversion to reduce data size. For final runs, use `s0` (2nm).

4. **Memory**: Some crops are large. The converter handles chunked reading.
   On the cluster, request ≥200GB RAM for the full pipeline.

5. **Instance vs Semantic**: Auto3DSeg performs semantic segmentation. The converter
   binarizes instance labels (any nonzero voxel → class present). For instance
   segmentation, you'll need post-processing.

## Interpreting Results

After running the analysis step, check `work_dir/datastats.yaml` for:

- **`image_stats.shape`**: Distribution of crop sizes → influences patch size selection
- **`image_stats.spacing`**: Voxel spacing consistency
- **`image_foreground_stats.intensity`**: Intensity range → influences normalization
- **`label_stats`**: Per-class voxel percentages → reveals class imbalance

After training, compare algorithm performance across folds to identify:
- Which architecture works best for your data
- Optimal patch size and batch size
- Best learning rate and augmentation strategy

These insights can then be transferred back to your existing CellMap training
pipeline for further optimization.
