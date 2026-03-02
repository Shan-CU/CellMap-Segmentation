#!/bin/bash
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --job-name=canary_v7_batch8
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Canary v7: batch=8 VRAM test at 96^3
# ============================================================================
# Goal: Confirm batch=8 fits in L40S 48 GB at 96^3.
# Corrected estimate: ~12.4 GB (26%). Should be very safe.
# Runs 8 training steps + reports peak GPU memory.
# If this works, we'll also try batch=12 to find the real ceiling.
# ============================================================================

set -euo pipefail

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# Memory leak mitigations
export CELLMAP_TENSORSTORE_CACHE_BYTES=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536

# Log memory every step for the canary
export MEMORY_LOG_STEPS=1

echo "============================================"
echo "Canary v7: batch=8, 96^3, segresnet_3d"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

# LR scaled linearly: 1e-4 @ batch=2 → 4e-4 @ batch=8
python -m training.train \
    --experiment_name canary_v7_batch8 \
    --run_dir runs/ablation \
    --model segresnet_3d \
    --loss DiceCELoss \
    --batch_size 8 \
    --input_shape 96 96 96 \
    --input_scale 8 8 8 \
    --epochs 1 \
    --iterations_per_epoch 8 \
    --learning_rate 4e-4 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --warmup_epochs 1 \
    --val_every_n_epochs 999 \
    --num_workers 4 \
    --amp

echo ""
echo "=== Peak GPU memory ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "Canary v7 complete — batch=8 fits!"
