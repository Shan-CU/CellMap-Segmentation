#!/bin/bash
#SBATCH --job-name=p2v5_resnet_3d
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v5_resnet_3d_%j.out
#SBATCH --error=runs/ablation/logs/p2v5_resnet_3d_%j.err

# ============================================================================
# Phase 2 v5: ResNet 3D — 128³ ROI, Optimized (1× A100 80GB)
# ============================================================================
# v4→v5 changes based on CSC source, Lauenburg (competition), Auto3DSeg:
#
#   1. Optimizer:  RAdam wd=0     → AdamW wd=1e-4   (Lauenburg + all baselines)
#   2. LR:         4e-4          → 1e-4             (CSC default, Auto3DSeg=2e-4)
#   3. Iters:      300/epoch     → 500/epoch        (300K→500K total steps)
#   4. Grad accum: 1 (eff=8)     → 2 (eff=16)       (Lauenburg=2)
#   5. eta_min:    0 (LR dies)   → 1e-6             (Lauenburg CosineAnnealing)
#   6. Warmup:     34 epochs     → 50 epochs (5%)   (Auto3DSeg: 5% of total)
#   7. Intensity:  none          → --intensity_aug   (Lauenburg: GaussianBlur p=0.2)
#   8. Loss:       dice_bce 0.5  → bce_weight=0.4   (Lauenburg: 60% dice / 40% BCE)
#   9. Smooth:     1e-6          → 1e-3             (Lauenburg: smooth=1e-3)
#  10. Grad clip:  1.0           → 0.5              (Lauenburg=0.5)
#  11. Val freq:   every 5       → every 10          (longer runs, save val time)
#
# Architecture: CSC ResNet 3D (~7.8M params)
#   - ngf=64, n_blocks=6, n_downsampling=2
#   - Single GPU: batch=8 × grad_accum=2 = effective batch=16
#   - 128³ crops at 8nm spacing (CSC default ROI)
#   - Memory at 128³ b=8: ~66GB (fits A100 80GB with headroom)
#
# Recipe: dice_bce(0.6/0.4) + AdamW + EMA(0.999) + fg_mask + weighted_sampler
# Training: 1000 epochs × 500 iters = 500K steps, eff_batch=16, 128³ @ 8nm
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
echo "Phase 2 v5: ResNet 3D — 128³ ROI (Optimized)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "============================================"

python -m training.train \
    --experiment_name p2v5_resnet_3d \
    --run_dir runs/ablation \
    --model resnet_3d \
    --loss dice_bce \
    --loss_kwargs '{"bce_weight": 0.4, "smooth": 1e-3}' \
    --batch_size 8 \
    --gradient_accumulation_steps 2 \
    --input_shape 128 128 128 \
    --input_scale 8 8 8 \
    --epochs 1000 \
    --iterations_per_epoch 500 \
    --learning_rate 1e-4 \
    --optimizer adamw \
    --weight_decay 1e-4 \
    --max_grad_norm 0.5 \
    --scheduler cosine \
    --warmup_epochs 50 \
    --eta_min 1e-6 \
    --validation_time_limit 600 \
    --val_every_n_epochs 10 \
    --best_metric val_dice \
    --num_workers 4 \
    --amp \
    --ema --ema_decay 0.999 \
    --bias_init_mode per_class \
    --random_seed 42 \
    --use_foreground_mask \
    --intensity_aug \
    --datasplit_path datasplit.csv \
    --filter_by_scale
