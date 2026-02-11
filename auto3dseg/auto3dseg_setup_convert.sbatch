#!/bin/bash
#SBATCH --job-name=a3ds_convert
#SBATCH --partition=acompile
#SBATCH --qos=compile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/auto3dseg_convert_%j.out
#SBATCH --error=logs/auto3dseg_convert_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gest9386@colorado.edu

# ============================================================
# Step 1: Convert zarr → NIfTI on acompile (instant start)
# Then submit Step 2 automatically when done.
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

PROJECT_DIR=/projects/gest9386/CellMap-Segmentation
cd $PROJECT_DIR
mkdir -p logs auto3dseg/nifti_data

# --- Environment setup ---
module purge
module load anaconda

# Create env if it doesn't exist
if ! conda env list 2>/dev/null | grep -q "csc"; then
    echo "Creating conda env 'csc'..."
    conda create -n csc python=3.11 -y
fi

conda activate csc
pip install --quiet 'monai[all]' nibabel pyyaml optuna zarr

echo "Python: $(which python)"
python -c "import monai; print(f'MONAI {monai.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# --- Convert ---
DATALIST="auto3dseg/nifti_data/datalist.json"

if [ -f "$DATALIST" ]; then
    echo "SKIPPING: $DATALIST already exists"
else
    echo "Converting zarr to NIfTI..."
    python auto3dseg/convert_zarr_to_nifti.py \
        --datasplit datasplit.csv \
        --output_dir auto3dseg/nifti_data
fi

if [ ! -f "$DATALIST" ]; then
    echo "ERROR: Conversion failed — $DATALIST not found"
    exit 1
fi

echo ""
echo "Conversion complete at $(date)"
echo "Datalist: $DATALIST"
echo ""
echo "Now submit step 2 (analysis + bundle gen):"
echo "  module load slurm/alpine"
echo "  sbatch auto3dseg/auto3dseg_setup_gpu.sbatch"
