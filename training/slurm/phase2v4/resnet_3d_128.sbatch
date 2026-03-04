#!/bin/bash
#SBATCH --job-name=p2v4_resnet_3d_128
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v4_resnet_3d_128_%j.out
#SBATCH --error=runs/ablation/logs/p2v4_resnet_3d_128_%j.err

# ============================================================================
# Phase 2 v4: ResNet 3D — 128³ ROI (CSC Default)
# ============================================================================
# Architecture: CSC ResNet 3D (~7.8M params)
#   - ngf=64, n_blocks=6, n_downsampling=2
#
# FIX: ROI 96³ → 128³ to match CSC example default (train_3D.py).
#   - 96³ = 884K voxels → 128³ = 2.1M voxels (2.37× more context)
#   - Physical context: 0.77µm → 1.02µm per crop at 8nm spacing
#   - Memory: ~28GB at 96³ b=8 → ~66GB at 128³ b=8 (fits A100 80GB)
#
# Everything else identical to the original p2v4_resnet_3d run.
# This is a fresh start (no checkpoint resume) with correct ROI.
#
# Recipe: dice_bce + EMA(0.999) + fg_mask + weighted_sampler
# Training: 1000 epochs × 300 iters, batch=8, 128³ crops @ 8nm
# ============================================================================

set -euo pipefail

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# --- Memory leak mitigations ---
export CELLMAP_TENSORSTORE_CACHE_BYTES=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================"
echo "Phase 2 v4: ResNet 3D — 128³ ROI (CSC Default)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "============================================"

python -m training.train \
    --experiment_name p2v4_resnet_3d_128 \
    --run_dir runs/ablation \
    --model resnet_3d \
    --loss dice_bce \
    --batch_size 8 \
    --input_shape 128 128 128 \
    --input_scale 8 8 8 \
    --epochs 1000 \
    --iterations_per_epoch 300 \
    --learning_rate 4e-4 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --warmup_epochs 34 \
    --validation_time_limit 600 \
    --val_every_n_epochs 5 \
    --best_metric val_dice \
    --num_workers 4 \
    --amp \
    --ema --ema_decay 0.999 \
    --bias_init_mode per_class \
    --optimizer radam \
    --weight_decay 0.0 \
    --random_seed 42 \
    --use_foreground_mask \
    --datasplit_path datasplit.csv \
    --filter_by_scale
