#!/bin/bash
#SBATCH --job-name=auto3dseg_convert
#SBATCH --partition=batch
#SBATCH --account=rc_alain_pi
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/auto3dseg_convert_%j.out
#SBATCH --error=logs/auto3dseg_convert_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@unc.edu

# ============================================================
# Step 1: Convert CellMap Zarr Data → NIfTI for Auto3DSeg
#
# This job reads zarr crops from datasplit.csv and saves them
# as NIfTI files that MONAI Auto3DSeg can process.
#
# Time: ~1-4 hours depending on number of crops
# Resources: CPU only, high memory for large zarr reads
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

PROJECT_DIR=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
cd $PROJECT_DIR

# --- Environment setup ---
module purge

# Activate micromamba environment
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

echo "Python: $(which python)"
echo "MONAI version: $(python -c 'import monai; print(monai.__version__)' 2>/dev/null || echo 'not installed')"

mkdir -p logs auto3dseg/nifti_data

# --- Run conversion ---
# Convert all crops at the default 8nm spacing
# Use --max_crops N for a quick test run
python auto3dseg/convert_zarr_to_nifti.py \
    --datasplit datasplit.csv \
    --output_dir auto3dseg/nifti_data \
    --target_spacing 8 8 8

echo "Conversion complete at $(date)"
echo "Next step: sbatch auto3dseg/auto3dseg_analyze.sbatch"
