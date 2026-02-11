#!/bin/bash
#SBATCH --job-name=unet_2d_ll
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=128g
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/unet_2d_longleaf_%j.out
#SBATCH --error=logs/unet_2d_longleaf_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# UNet 2D Training - Optimized for UNC Longleaf (A100 40GB)
# ============================================================
# Lessons from Sycamore H100 runs applied:
# - InstanceNorm prevents mode collapse
# - Focal Loss (γ=2, α=1) + 0.5×Dice works well
# - num_workers=8 is stable for 2D
# - 2 GPUs sufficient for 2D models (I/O bound)
# ============================================================

MODEL="unet"
DIM="2d"

# Optimized for A100 40GB (vs H100 80GB - reduce batch slightly)
EPOCHS=200
BATCH_SIZE=24                    # Reduced from 32 for 40GB VRAM
ITERATIONS_PER_EPOCH=1000
LEARNING_RATE=5e-5               # Same as Sycamore (prevents mode collapse)
GRADIENT_ACCUMULATION=4          # Effective batch = 24*4*2GPUs = 192
NUM_WORKERS=8

echo "=============================================="
echo "CellMap Model Comparison - UNet 2D"
echo "=============================================="
echo "Cluster: UNC Longleaf (A100 40GB)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2x A100 40GB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="
echo "Model: $MODEL"
echo "Dimension: $DIM"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE (per GPU)"
echo "Iterations/Epoch: $ITERATIONS_PER_EPOCH"
echo "Learning Rate: $LEARNING_RATE"
echo "Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * 2))"
echo "=============================================="

# Navigate to project root
cd $SLURM_SUBMIT_DIR/../../../..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Create directories
mkdir -p logs
mkdir -p experiments/model_comparison/checkpoints
mkdir -p experiments/model_comparison/tensorboard
mkdir -p experiments/model_comparison/visualizations
mkdir -p experiments/model_comparison/metrics
mkdir -p experiments/model_comparison/slurm/longleaf/logs

# ============================================================
# Longleaf Environment Setup
# ============================================================
module purge
module load cuda/12.2

# Use micromamba (user-installed) - Fixed for batch mode
# Use absolute paths for non-interactive batch environment
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
# Environment activation via micromamba run (works in non-interactive batch)
# Using micromamba run instead of activate for batch compatibility

echo "Python: $($MAMBA_EXE run -n csc which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ============================================================
# A100 Optimizations
# ============================================================
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# A100-specific TF32
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1

# torch.compile cache
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

echo "Data directory: $PROJECT_ROOT/data"

N_GPUS=2
echo "Using $N_GPUS GPUs with DDP"

# ============================================================
# Training
# ============================================================
echo ""
echo "Starting training: $MODEL ($DIM)..."
echo "Estimated completion: ~24-36 hours"
echo ""

$MAMBA_EXE run -n csc torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --iterations_per_epoch $ITERATIONS_PER_EPOCH \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --pin_memory \
    --amp \
    --compile \
    --save_features \
    --vis_every 10

EXIT_CODE=$?

echo ""
echo "=============================================="
echo "Training Complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
