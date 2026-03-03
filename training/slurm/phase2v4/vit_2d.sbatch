#!/bin/bash
#SBATCH --job-name=p2v4_vit_2d
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v4_vit_2d_%j.out
#SBATCH --error=runs/ablation/logs/p2v4_vit_2d_%j.err

# ============================================================================
# Phase 2 v4: ViT-V-Net 2D on A100 80GB (moved from L40S — l40 queue 2-day wait)
# ============================================================================
# Architecture: 2D adaptation of ViT-V-Net (vit_2d.py) — ~105M params
#   - hidden_size=768, 12 layers, 12 heads, patch_size=16, img_size=256
#   - CNN encoder with dynamic channels, ViT transformer backbone
#   - Largest model in the 2D lineup
#
# Optimizer: AdamW, lr=1e-4, wd=0.01
#   - ViT-V-Net paper (Chen et al. MIDL 2021): Adam, lr=1e-4, wd=0
#   - We use AdamW + wd=0.01 (DeiT-style) for this 105M param model —
#     transformers benefit from weight decay regularization
#
# Key fix: bias_init=-3.0 (v2/v3 collapsed to 0.000 dice without this)
#
# Note: batch=4 + grad_accum=2 (effective batch=8). batch=8+AMP triggers
#   cudaErrorInvalidConfiguration in DiceBCE backward (PyTorch 2.10 + CUDA 12.8
#   kernel launch bug with large activation tensors). batch=4+AMP is stable.
#
# Recipe: dice_bce + EMA(0.999) + fg_mask + weighted_sampler
# Training: 100 epochs × 1000 iters, batch=4 (×2 accum), 256×256 patches
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
echo "Phase 2 v4: ViT-V-Net 2D (AdamW + wd=0.01)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

python -m training.train \
    --experiment_name p2v4_vit_2d \
    --run_dir runs/ablation \
    --model vit_2d \
    --loss dice_bce \
    --batch_size 4 \
    --input_shape 1 256 256 \
    --input_scale 8 8 8 \
    --epochs 100 \
    --iterations_per_epoch 1000 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --gradient_accumulation_steps 2 \
    --scheduler cosine \
    --warmup_epochs 10 \
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
