#!/bin/bash
#SBATCH --job-name=unet_3d_syc
#SBATCH --partition=h100_sn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --mem=512G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/unet_3d_sycamore_%j.out
#SBATCH --error=logs/unet_3d_sycamore_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@unc.edu

# ============================================================
# UNet 3D Training - Optimized for UNC Sycamore (H100 80GB)
# ============================================================
# Architecture: UNet 3D (CNN - Moderate Memory for 3D)
# Cluster: UNC Sycamore h100_sn partition
# GPUs: 4x H100 80GB (3D benefits from more parallelism)
# Expected training time: ~20-28 hours for 150 epochs
# Memory usage: ~20-30 GB per GPU
# Input shape: (32, 256, 256) - 32 slices of 256x256
# ============================================================

MODEL="unet"
DIM="3d"

# Optimized hyperparameters for 3D UNet on 4x H100 80GB with DDP
# With 4 GPUs we can increase batch size significantly
EPOCHS=150
BATCH_SIZE=8                     # Increased from 6 (4 GPUs + H100 speed)
ITERATIONS_PER_EPOCH=500         # Increased from 250 (more compute available)
LEARNING_RATE=1e-4
GRADIENT_ACCUMULATION=4          # Effective batch = 8*4*4GPUs = 128
NUM_WORKERS=8                    # 64 CPUs / 8 = 8 workers per GPU

echo "=============================================="
echo "CellMap Model Comparison - UNet 3D"
echo "=============================================="
echo "Cluster: UNC Sycamore (H100)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x H100 80GB"
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
echo "Effective Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * 4))"
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

# Sycamore Environment Setup (micromamba)
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# H100 Optimizations for 3D models
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))  # Split across 4 GPUs
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1

# NVLink optimizations for 4-GPU communication
export NCCL_P2P_LEVEL=NVL
export NCCL_SHM_DISABLE=0

N_GPUS=4
echo "Using $N_GPUS GPUs with DDP"

# Start TensorBoard in background
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
tensorboard --logdir=experiments/model_comparison/tensorboard --port=$TB_PORT --bind_all &
TB_PID=$!
echo "TensorBoard started on port $TB_PORT"

# Training
echo ""
echo "Starting training: $MODEL ($DIM)..."
echo "Estimated completion: ~20-28 hours"
echo ""

torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --iterations_per_epoch $ITERATIONS_PER_EPOCH \
    --lr $LEARNING_RATE \
    --save_features \
    --vis_every 5

EXIT_CODE=$?

kill $TB_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "Training Complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
