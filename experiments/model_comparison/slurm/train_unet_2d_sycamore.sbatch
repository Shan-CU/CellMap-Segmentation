#!/bin/bash
#SBATCH --job-name=unet_2d_syc
#SBATCH --partition=h100_sn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --mem=384G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/unet_2d_sycamore_%j.out
#SBATCH --error=logs/unet_2d_sycamore_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@unc.edu

# ============================================================
# UNet 2D Training - Optimized for UNC Sycamore (H100 80GB)
# ============================================================
# Architecture: UNet 2D (CNN - Memory Efficient)
# Cluster: UNC Sycamore h100_sn partition
# GPUs: 2x H100 80GB (2D models are I/O-bound, 2 GPUs sufficient)
# Expected training time: ~18-24 hours for 200 epochs
# Memory usage: ~8-12 GB per GPU
# ============================================================

MODEL="unet"
DIM="2d"

# Optimized hyperparameters for 2D UNet on 2x H100 80GB with DDP
# H100 is ~1.5x faster than A100, so we can increase batch size
# Using 2 GPUs (2D is I/O-bound, more GPUs = diminishing returns)
EPOCHS=200
BATCH_SIZE=48                    # Increased from 32 (H100 faster + more memory bandwidth)
ITERATIONS_PER_EPOCH=1000        # Shared across all 2D models for fair comparison
LEARNING_RATE=1e-4
GRADIENT_ACCUMULATION=2          # Effective batch = 48*2*2GPUs = 192
NUM_WORKERS=12                   # 48 CPUs / 4 = 12 workers per GPU (with headroom)

echo "=============================================="
echo "CellMap Model Comparison - UNet 2D"
echo "=============================================="
echo "Cluster: UNC Sycamore (H100)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2x H100 80GB"
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
cd $SLURM_SUBMIT_DIR/../../..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Create directories
mkdir -p logs
mkdir -p experiments/model_comparison/checkpoints
mkdir -p experiments/model_comparison/tensorboard
mkdir -p experiments/model_comparison/visualizations
mkdir -p experiments/model_comparison/metrics

# ============================================================
# Sycamore Environment Setup (micromamba)
# ============================================================
# Sycamore uses user-managed micromamba, not module system for Python
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ============================================================
# H100 Optimizations
# ============================================================
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))  # Half CPUs for OMP, rest for dataloader
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# H100-specific: Enable TF32 for matmul (no accuracy loss, faster)
export NVIDIA_TF32_OVERRIDE=1

# Sycamore uses /work for high-throughput I/O (already optimal)
echo "Data directory: $PROJECT_ROOT/data"

N_GPUS=2
echo "Using $N_GPUS GPUs with DDP"

# Start TensorBoard in background
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
tensorboard --logdir=experiments/model_comparison/tensorboard --port=$TB_PORT --bind_all &
TB_PID=$!
echo "TensorBoard started on port $TB_PORT"

# ============================================================
# Training
# ============================================================
echo ""
echo "Starting training: $MODEL ($DIM)..."
echo "Estimated completion: ~18-24 hours"
echo ""

torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --iterations_per_epoch $ITERATIONS_PER_EPOCH \
    --lr $LEARNING_RATE \
    --num_workers 4 \
    --save_features \
    --vis_every 10

EXIT_CODE=$?

# Cleanup
kill $TB_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "Training Complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
