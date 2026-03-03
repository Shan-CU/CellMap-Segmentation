#!/bin/bash
#SBATCH --job-name=p2v4_swin_2d
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v4_swin_2d_%j.out
#SBATCH --error=runs/ablation/logs/p2v4_swin_2d_%j.err

# ============================================================================
# Phase 2 v4: Swin Transformer V2 2D on A100 80GB (moved from L40S to free slot for ViT-2D)
# ============================================================================
# Architecture: Swin V2 (SwinTransformerBlock.py) — ~36M params
#   - embed_dim=96, depths=[2,2,6,2], window_size=8
#   - Cosine attention (Swin V2), PatchMergingV2
#   - UNet-style decoder with Unet_Block_Up, SegmentationHead (2× upsample)
#   - Decoder uses InstanceNorm2d (fixed from BatchNorm2d for stability)
#
# Optimizer: AdamW, lr=5e-4, wd=0.01, warmup=20, grad_clip=5.0
#   - Previous config (lr=1e-4, wd=0.05, dropout=0.1) was heavily over-regularized
#   - Reduced wd 0.05→0.01, removed dropout (0.1→0.0 in model_zoo.py)
#   - Raised LR 1e-4→5e-4 (Swin V2 paper uses 5e-4 for fine-tuning)
#   - Extended warmup 10→20, relaxed grad_clip 1.0→5.0 for higher LR
#
# Key fix: bias_init=-3.0 (v2/v3 collapsed to 0.000 dice without this)
#
# Recipe: dice_bce + EMA(0.999) + fg_mask + weighted_sampler
# Training: 100 epochs × 1000 iters, batch=8, 256×256 patches
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
echo "Phase 2 v4: Swin Transformer V2 2D (AdamW, lr=5e-4, wd=0.01, dropout=0.0)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

python -m training.train \
    --experiment_name p2v4_swin_2d \
    --run_dir runs/ablation \
    --model swin_2d \
    --loss dice_bce \
    --batch_size 8 \
    --input_shape 1 256 256 \
    --input_scale 8 8 8 \
    --epochs 100 \
    --iterations_per_epoch 1000 \
    --learning_rate 5e-4 \
    --max_grad_norm 5.0 \
    --scheduler cosine \
    --warmup_epochs 20 \
    --validation_time_limit 600 \
    --val_every_n_epochs 5 \
    --best_metric val_dice \
    --num_workers 8 \
    --amp \
    --ema --ema_decay 0.999 \
    --bias_init_mode per_class \
    --optimizer adamw \
    --weight_decay 0.01 \
    --random_seed 42 \
    --use_foreground_mask \
    --datasplit_path datasplit.csv \
    --filter_by_scale
