#!/bin/bash
#SBATCH --job-name=3d_monster
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/logs/%x_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/logs/%x_%j.err
#SBATCH --partition=h100_mn
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=48
#SBATCH --mem=1400G
#SBATCH --time=5-00:00:00

# ============================================================================
# MONSTER 3D TRAINING JOB - MAXIMIZING H100 + 1.5TB RAM
# ============================================================================
# This job template is designed to maximize throughput on Sycamore H100 nodes:
#
# Node specs:
#   - 4x NVIDIA H100 80GB (320GB GPU memory total, NVLink connected)
#   - 256 CPU cores (2x AMD Bergamo 128-core)
#   - 1.5TB System RAM
#   - NDR InfiniBand (2x 800Gbps)
#
# This template uses:
#   - All 4 H100 GPUs
#   - 192 CPU cores (48 per GPU) for data loading
#   - 1.4TB RAM (leaves 100GB for OS)
#   - 16-20 DataLoader workers with NO persistent workers (prevents memory leak)
#
# Memory budget:
#   - ~80GB for model/gradients across 4 GPUs
#   - ~50GB per DataLoader worker
#   - 20 workers × 50GB = ~1000GB
#   - Total: ~1080GB, well under 1400GB limit
# ============================================================================

# Exit on error
set -e

# ============================================================================
# CONFIGURATION - EDIT THESE FOR YOUR MODEL
# ============================================================================

# Model to train: resnet_3d, unet_3d, swin_3d, or vit_3d
MODEL="REPLACE_WITH_BEST_3D_MODEL"

# Batch size per GPU (can be higher with fast data loading)
BATCH_SIZE=4

# Number of DataLoader workers
# Sweet spot: 16-20 for 3D data with 1.5TB RAM
NUM_WORKERS=16

# Prefetch factor (batches per worker to preload)
# Higher = more memory, faster training
PREFETCH=4

# Learning rate (should match model defaults)
LEARNING_RATE=3e-5

# Number of epochs (3D trains slower, plan accordingly)
EPOCHS=150

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "============================================"
echo "MONSTER 3D TRAINING JOB"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Model: $MODEL"
echo "GPUs: $SLURM_GPUS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "Workers: $NUM_WORKERS"
echo "Prefetch: $PREFETCH"
echo "============================================"

# Check GPU availability
nvidia-smi

# Check memory
free -h

# Initialize micromamba
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

# Navigate to repo
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# ============================================================================
# TRAINING COMMAND
# ============================================================================

# Use torchrun for multi-GPU DDP training
# Key optimizations:
#   --num_workers: Many workers for parallel data loading
#   --prefetch_factor: Preload batches to hide I/O latency
#   NO --persistent_workers: Prevents memory leak (workers respawn each epoch)
#   --pin_memory: Faster CPU→GPU transfer
#   --amp: BFloat16 mixed precision for H100

torchrun --standalone --nproc_per_node=4 \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim 3D \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --prefetch_factor $PREFETCH \
    --pin_memory \
    --amp

echo "============================================"
echo "Training complete!"
echo "============================================"
