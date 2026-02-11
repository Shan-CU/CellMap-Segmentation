#!/bin/bash
#SBATCH --job-name=auto3dseg_setup
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/auto3dseg_setup_%j.out
#SBATCH --error=logs/auto3dseg_setup_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gest9386@colorado.edu

# ============================================================
# Auto3DSeg Setup — All pre-training steps in one fast job
#
# Runs on a single A100 node:
#   1. Install conda env (if needed)
#   2. Convert zarr → NIfTI (~30 min with 32 CPUs)
#   3. Data analysis (~20 min)
#   4. Bundle generation (SegResNet, SwinUNETR, DiNTS) (~5 min with GPU)
#
# After this completes, submit the training job:
#   module load slurm/blanca
#   sbatch auto3dseg/auto3dseg_train_blanca.sbatch
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

PROJECT_DIR=/projects/gest9386/CellMap-Segmentation
cd $PROJECT_DIR

mkdir -p logs auto3dseg/work_dir auto3dseg/nifti_data

# --- Environment setup ---
module purge
module load anaconda

# Create env if it doesn't exist
if ! conda env list | grep -q "csc"; then
    echo "Creating conda env 'csc'..."
    conda create -n csc python=3.11 -y
fi

conda activate cellmap

# Install packages (pip will skip already-installed)
pip install --quiet 'monai[all]' nibabel pyyaml optuna

# Fix libstdc++ for DiNTS/optuna/sqlite3
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
python -c "import monai; print(f'MONAI {monai.__version__}')"

# ============================================================
# Step 1: Convert zarr → NIfTI
# ============================================================
DATALIST="auto3dseg/nifti_data/datalist.json"

if [ -f "$DATALIST" ]; then
    echo ""
    echo "=== SKIPPING CONVERSION: $DATALIST already exists ==="
    echo ""
else
    echo ""
    echo "=== Step 1: Converting zarr to NIfTI ==="
    echo "Start: $(date)"
    python auto3dseg/convert_zarr_to_nifti.py \
        --datasplit datasplit.csv \
        --output_dir auto3dseg/nifti_data
    echo "Conversion done: $(date)"
fi

# Verify datalist exists
if [ ! -f "$DATALIST" ]; then
    echo "ERROR: $DATALIST not found after conversion!"
    exit 1
fi

# ============================================================
# Step 2: Data Analysis
# ============================================================
if [ -f "auto3dseg/work_dir/datastats.yaml" ]; then
    echo ""
    echo "=== SKIPPING ANALYSIS: datastats.yaml already exists ==="
    echo ""
else
    echo ""
    echo "=== Step 2: Running Data Analysis ==="
    echo "Start: $(date)"
    python auto3dseg/run_auto3dseg.py \
        --mode analyze \
        --datalist $DATALIST \
        --work_dir auto3dseg/work_dir
    echo "Analysis done: $(date)"
fi

# ============================================================
# Step 3: Bundle Generation (needs GPU)
# ============================================================
echo ""
echo "=== Step 3: Generating Algorithm Bundles ==="
echo "Start: $(date)"
python auto3dseg/run_auto3dseg.py \
    --mode generate \
    --datalist $DATALIST \
    --work_dir auto3dseg/work_dir \
    --algos segresnet swinunetr dints \
    --num_fold 1 \
    --device cuda
echo "Bundle generation done: $(date)"

# ============================================================
# Summary
# ============================================================
echo ""
echo "========================================"
echo "Auto3DSeg setup complete at $(date)"
echo "========================================"
echo ""
echo "Generated files:"
ls -la auto3dseg/work_dir/
echo ""
echo "Next step — submit training:"
echo "  module load slurm/blanca"
echo "  sbatch auto3dseg/auto3dseg_train_blanca.sbatch"
echo ""
echo "Or on Alpine:"
echo "  module load slurm/alpine"
echo "  sbatch auto3dseg/auto3dseg_train_alpine.sbatch"
