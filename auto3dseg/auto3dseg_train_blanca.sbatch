#!/bin/bash
#SBATCH --job-name=auto3dseg_train
#SBATCH --partition=blanca-biokem
#SBATCH --qos=blanca-biokem
#SBATCH --account=blanca-biokem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/auto3dseg_train_%j.out
#SBATCH --error=logs/auto3dseg_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gest9386@colorado.edu

# ============================================================
# Auto3DSeg Training — Blanca biokem (2x NVIDIA A100 on bgpu-biokem2)
#
# Runs the complete MONAI Auto3DSeg pipeline:
#   1. Data analysis (if not already cached)
#   2. Algorithm bundle generation (SegResNet, SwinUNETR, DiNTS)
#   3. Model training
#   4. Model ensemble for best predictions
#
# Hardware: 2x A100, 16 CPUs, 200GB RAM
# Walltime: 7 days (max for high-priority Blanca QOS)
#
# The pipeline caches intermediate results, so it can be safely
# restarted if it times out — it will resume from where it left off.
#
# Prerequisites:
#   module load slurm/blanca   (before submitting)
#   conda env with: monai[all], nibabel, pyyaml, optuna, torch
#   NIfTI data converted: python auto3dseg/convert_to_nifti.py ...
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# --- CHANGE THIS to your repo path on Alpine/Blanca ---
PROJECT_DIR="$HOME/CellMap-Segmentation"
# e.g. /projects/gest9386/CellMap-Segmentation or wherever you cloned it
cd $PROJECT_DIR

# --- Environment setup ---
module purge
module load anaconda
# If using conda:
conda activate csc
# If using micromamba instead, uncomment and adjust:
# export MAMBA_EXE="$HOME/.local/bin/micromamba"
# export MAMBA_ROOT_PREFIX="$HOME/micromamba"
# eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
# micromamba activate csc

# Fix: Ensure conda's libstdc++ (which has CXXABI_1.3.15) takes priority
# over the system /lib64/libstdc++ (which doesn't). Required by DiNTS → optuna → sqlite3.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
python -c "import monai; print(f'MONAI {monai.__version__}')"

mkdir -p logs auto3dseg/work_dir

# --- Performance tuning ---
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,max_split_size_mb:512,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY
export NCCL_DEBUG=INFO
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

# --- Verify datalist ---
DATALIST="auto3dseg/nifti_data/datalist.json"
if [ ! -f "$DATALIST" ]; then
    echo "ERROR: $DATALIST not found!"
    echo "Run the conversion step first:"
    echo "  python auto3dseg/convert_to_nifti.py --data_dir data --output_dir auto3dseg/nifti_data --manifest datasplit.csv"
    exit 1
fi

# --- Determine GPU count ---
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using $N_GPUS GPUs"

# ============================================================
# Configuration
# ============================================================
ALGOS="segresnet swinunetr dints"
NUM_FOLD=1
NUM_EPOCHS=0   # 0 = use auto-computed per algorithm

echo ""
echo "Configuration:"
echo "  Algorithms: $ALGOS"
echo "  Folds:      $NUM_FOLD"
echo "  Epochs:     $NUM_EPOCHS (0 = use auto-computed per algorithm)"
echo "  GPUs:       $N_GPUS"
echo ""

# --- Run pipeline ---
CMD="python auto3dseg/run_auto3dseg.py \
    --mode full \
    --datalist $DATALIST \
    --work_dir auto3dseg/work_dir \
    --algos $ALGOS \
    --num_fold $NUM_FOLD \
    --num_epochs $NUM_EPOCHS \
    --num_gpus $N_GPUS \
    --device cuda \
    --workers 4"

echo "Running: $CMD"
eval $CMD

echo ""
echo "Auto3DSeg pipeline finished at $(date)"
echo "Results in: auto3dseg/work_dir/"
echo ""
echo "Key outputs to review:"
echo "  auto3dseg/work_dir/datastats.yaml    - Dataset statistics"
echo "  auto3dseg/work_dir/<algo>_<fold>/     - Trained model bundles"
echo "  auto3dseg/work_dir/ensemble_output/   - Ensemble predictions"
