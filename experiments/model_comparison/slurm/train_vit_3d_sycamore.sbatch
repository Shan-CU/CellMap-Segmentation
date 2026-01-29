#!/bin/bash
#SBATCH --job-name=vit_3d_syc
#SBATCH --partition=h100_sn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --mem=1500G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/vit_3d_sycamore_%j.out
#SBATCH --error=logs/vit_3d_sycamore_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# ViT 3D Training - Optimized for UNC Sycamore (H100 80GB)
# ============================================================
# Architecture: ViT-V-Net 3D (Transformer - Extreme Memory)
# Cluster: UNC Sycamore h100_sn partition
# GPUs: 4x H100 80GB (essential for 3D global attention)
# Expected training time: ~40-48 hours for 150 epochs
# Memory usage: ~60-75 GB per GPU (most memory intensive model)
# Input shape: (32, 256, 256) - 32 slices of 256x256
# ============================================================

MODEL="vit"
DIM="3d"

# Optimized hyperparameters for 3D ViT on 4x H100 80GB with DDP
# ViT 3D has global attention - most memory intensive model
# With 4 GPUs we can increase batch size from 1 to 2
EPOCHS=150
BATCH_SIZE=2                     # H100 + AMP optimized (matches config_base.py)
ITERATIONS_PER_EPOCH=500         # Increased from 250
LEARNING_RATE=3e-5               # Lower LR for stable 3D transformer training
GRADIENT_ACCUMULATION=16         # Effective batch = 2*16*4GPUs = 128
NUM_WORKERS=8

echo "=============================================="
echo "CellMap Model Comparison - ViT 3D"
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

# H100 Optimizations for 3D ViT (Updated for maximum performance)
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# H100 Transformer optimizations
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

# Gradient checkpointing essential for 3D ViT
export GRADIENT_CHECKPOINTING=1

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
echo "Estimated completion: ~40-48 hours"
echo "NOTE: Most memory intensive model - using heavy gradient accumulation"
echo ""

torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --iterations_per_epoch $ITERATIONS_PER_EPOCH \
    --lr $LEARNING_RATE \
    --num_workers 2 \
    --pin_memory \
    --amp \
    --compile \
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
