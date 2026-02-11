#!/bin/bash
#SBATCH --job-name=auto3dseg_a100
#SBATCH --partition=a100-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/auto3dseg_train_%j.out
#SBATCH --error=logs/auto3dseg_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@unc.edu

# ============================================================
# Auto3DSeg Full Pipeline on A100 GPUs
# Same as auto3dseg_train.sbatch but targeting a100-gpu partition
# 2x A100 (40GB each), 16 CPUs, 200GB RAM, 6-day walltime
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

PROJECT_DIR=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
cd $PROJECT_DIR

# --- Environment setup ---
module purge

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

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
NUM_EPOCHS=0  # 0 = use auto-computed per algorithm

echo ""
echo "Configuration:"
echo "  Algorithms: $ALGOS"
echo "  Folds:      $NUM_FOLD"
echo "  Epochs:     $NUM_EPOCHS (0 = use auto-computed per algorithm)"
echo "  GPUs:       $N_GPUS"
echo ""

# --- Build command ---
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
