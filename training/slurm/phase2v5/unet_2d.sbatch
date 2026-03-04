#!/bin/bash
#SBATCH --job-name=p2v5_unet_2d
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v5_unet_2d_%j.out
#SBATCH --error=runs/ablation/logs/p2v5_unet_2d_%j.err

# ============================================================================
# Phase 2 v5: UNet 2D — Optimized per CSC + Competition Research
# ============================================================================
# v4→v5 changes based on CSC official docs, Lauenburg (competition), UNet baseline:
#
#   1. Optimizer:  RAdam wd=0     → AdamW wd=1e-4   (Lauenburg + UNet baseline)
#   2. Epochs:     100 × 1000    → 300 × 1000 = 300K steps  (CSC default=1M)
#   3. Grad accum: 1 (eff=8)     → 2 (eff=16)       (Lauenburg=2, baseline=2)
#   4. eta_min:    0 (LR dies)   → 1e-6             (Lauenburg CosineAnnealing)
#   5. Warmup:     10 epochs     → 15 epochs (5%)   (UNet baseline=5%)
#   6. Intensity:  none          → --intensity_aug   (Lauenburg: GaussianBlur p=0.2)
#   7. Loss:       dice_bce 0.5  → bce_weight=0.4   (Lauenburg: 60% dice / 40% BCE)
#   8. Smooth:     1e-6          → 1e-3             (Lauenburg: smooth=1e-3)
#   9. Grad clip:  1.0           → 0.5              (Lauenburg=0.5)
#  10. Val freq:   every 5       → every 10          (3× more epochs, save val time)
#
# Architecture: Standard UNet (unet_model_2D.py) — ~90M params w/ 48 classes
#   - 4 down/up stages (64→128→256→512→1024), InstanceNorm, dropout=0.1
#
# Recipe: dice_bce(0.6/0.4) + AdamW + EMA(0.999) + fg_mask + weighted_sampler
# Training: 300 epochs × 1000 iters, batch=8 × grad_accum=2 (eff 16), 256×256
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
echo "Phase 2 v5: UNet 2D (Optimized)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

python -m training.train \
    --experiment_name p2v5_unet_2d \
    --run_dir runs/ablation \
    --model unet_2d \
    --loss dice_bce \
    --loss_kwargs '{"bce_weight": 0.4, "smooth": 1e-3}' \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --input_shape 1 256 256 \
    --input_scale 8 8 8 \
    --epochs 300 \
    --iterations_per_epoch 1000 \
    --learning_rate 1e-4 \
    --optimizer adamw \
    --weight_decay 1e-4 \
    --max_grad_norm 0.5 \
    --scheduler cosine \
    --warmup_epochs 15 \
    --eta_min 1e-6 \
    --validation_time_limit 600 \
    --val_every_n_epochs 10 \
    --best_metric val_dice \
    --num_workers 8 \
    --amp \
    --ema --ema_decay 0.999 \
    --bias_init_mode per_class \
    --random_seed 42 \
    --use_foreground_mask \
    --intensity_aug \
    --datasplit_path datasplit.csv \
    --filter_by_scale \
    --model_kwargs '{"use_instancenorm": true, "dropout": 0.1}'
