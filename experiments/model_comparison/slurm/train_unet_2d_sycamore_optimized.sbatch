#!/bin/bash
#SBATCH --job-name=unet_2d_opt
#SBATCH --partition=h100_sn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:2
#SBATCH --mem=768G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/unet_2d_optimized_%j.out
#SBATCH --error=logs/unet_2d_optimized_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# UNet 2D Training - OPTIMIZED VERSION
# ============================================================
# Key fixes applied to prevent mode collapse:
# 1. InstanceNorm instead of BatchNorm (in config)
# 2. Dropout 0.2 for regularization (in config) 
# 3. Lower learning rate: 5e-5 (from 1e-4)
# 4. Combined BCE + 0.5×Dice loss (in train_comparison.py)
# 5. Lower batch size: 32 (from 64) for InstanceNorm stability
# 6. Lower class weight cap: 25 (from 50)
# ============================================================
# Previous UNet run collapsed to predicting zeros (Dice ~0.01)
# ResNet-2D with InstanceNorm achieved Dice 0.40+ with same data
# These optimizations align UNet with ResNet's successful settings
# ============================================================

MODEL="unet"
DIM="2d"

# Optimized hyperparameters - FIXES MODE COLLAPSE
EPOCHS=200
BATCH_SIZE=32                    # Lower for InstanceNorm stability (was 64)
ITERATIONS_PER_EPOCH=1000
LEARNING_RATE=5e-5               # Lower LR (was 1e-4) - critical for stability
GRADIENT_ACCUMULATION=4          # Effective batch = 32*4*2GPUs = 256
NUM_WORKERS=4                    # Reduced workers to avoid memory pressure

echo "=============================================="
echo "CellMap Model Comparison - UNet 2D (OPTIMIZED)"
echo "=============================================="
echo "Cluster: UNC Sycamore (H100)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2x H100 80GB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="
echo "OPTIMIZATIONS APPLIED:"
echo "  - InstanceNorm (not BatchNorm)"
echo "  - Dropout: 0.2"
echo "  - Learning Rate: $LEARNING_RATE (lower)"
echo "  - Batch Size: $BATCH_SIZE (smaller)"
echo "  - Combined BCE + Dice loss"
echo "  - Class weight cap: 25"
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
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# H100-specific optimizations
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1

# torch.compile cache directory
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

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
echo "Starting optimized training: $MODEL ($DIM)..."
echo "Expected: Dice should reach 0.30+ (vs 0.01 with original settings)"
echo ""

torchrun --standalone --nproc_per_node=$N_GPUS \
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

# Cleanup
kill $TB_PID 2>/dev/null || true

echo ""
echo "=============================================="
echo "Training Complete"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "=============================================="

exit $EXIT_CODE
