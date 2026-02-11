#!/bin/bash
#SBATCH --job-name=a3ds_bundlegen
#SBATCH --partition=atesting_a100
#SBATCH --qos=testing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/auto3dseg_bundlegen_%j.out
#SBATCH --error=logs/auto3dseg_bundlegen_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gest9386@colorado.edu

# ============================================================
# Step 2: Data Analysis + Bundle Generation on atesting_a100
# (instant start, 1hr limit — more than enough)
#
# Run AFTER conversion completes:
#   module load slurm/alpine
#   sbatch auto3dseg/auto3dseg_setup_gpu.sbatch
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

PROJECT_DIR=/projects/gest9386/CellMap-Segmentation
cd $PROJECT_DIR
mkdir -p logs auto3dseg/work_dir

# --- Environment setup ---
module purge
module load anaconda
conda activate csc

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
python -c "import monai; print(f'MONAI {monai.__version__}')"

DATALIST="auto3dseg/nifti_data/datalist.json"
if [ ! -f "$DATALIST" ]; then
    echo "ERROR: $DATALIST not found! Run conversion first."
    exit 1
fi

# --- Step 2a: Data Analysis ---
if [ -f "auto3dseg/work_dir/datastats.yaml" ]; then
    echo "SKIPPING analysis: datastats.yaml already exists"
else
    echo ""
    echo "=== Running Data Analysis ==="
    echo "Start: $(date)"
    python auto3dseg/run_auto3dseg.py \
        --mode analyze \
        --datalist $DATALIST \
        --work_dir auto3dseg/work_dir
    echo "Analysis done: $(date)"
fi

# --- Step 2b: Bundle Generation ---
echo ""
echo "=== Generating Algorithm Bundles ==="
echo "Start: $(date)"
python auto3dseg/run_auto3dseg.py \
    --mode generate \
    --datalist $DATALIST \
    --work_dir auto3dseg/work_dir \
    --algos segresnet swinunetr dints \
    --num_fold 1 \
    --device cuda
echo "Bundle generation done: $(date)"

# --- Summary ---
echo ""
echo "========================================"
echo "Setup complete at $(date)"
echo "========================================"
echo ""
ls -la auto3dseg/work_dir/
echo ""
echo "Next: submit training job"
echo "  module load slurm/blanca"
echo "  sbatch auto3dseg/auto3dseg_train_blanca.sbatch"
echo ""
echo "Or on Alpine:"
echo "  module load slurm/alpine"
echo "  sbatch auto3dseg/auto3dseg_train_alpine.sbatch"
