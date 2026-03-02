#!/bin/bash
#SBATCH --job-name=p2v4_unet_3d
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v4_unet_3d_%j.out
#SBATCH --error=runs/ablation/logs/p2v4_unet_3d_%j.err

# ============================================================================
# Phase 2 v4: UNet 3D on A100 80GB
# ============================================================================
# Architecture: Standard UNet (unet_model_3D.py) — ~31M params
#   - 4 down/up stages (64→128→256→512→1024)
#   - v2 used BatchNorm → NOW using InstanceNorm (standard for medical seg)
#   - dropout=0.1 (light regularization)
#
# Optimizer: RAdam (default), lr=4e-4, wd=0
#   - LR scaled linearly: base 1e-4 @ batch=2 → 4e-4 @ batch=8
#
# Key fixes:
#   - bias_init=-3.0 (RetinaNet focal prior)
#   - InstanceNorm via model_kwargs (was BatchNorm in v2)
#   - dropout=0.1 via model_kwargs
#
# Recipe: dice_bce + EMA(0.999) + fg_mask + weighted_sampler
# Training: 1000 epochs × 300 iters, batch=8, 96³ crops @ 8nm
# Validation: every 5 epochs, per-class Dice (48 classes), 600s time limit
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

echo "============================================"
echo "Phase 2 v4: UNet 3D (InstanceNorm + dropout)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

python -m training.train \
    --experiment_name p2v4_unet_3d \
    --run_dir runs/ablation \
    --model unet_3d \
    --loss dice_bce \
    --batch_size 8 \
    --input_shape 96 96 96 \
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
    --filter_by_scale \
    --model_kwargs '{"use_instancenorm": true, "dropout": 0.1}'
