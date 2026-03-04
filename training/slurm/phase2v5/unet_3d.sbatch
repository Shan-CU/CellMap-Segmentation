#!/bin/bash
#SBATCH --job-name=p2v5_unet_3d
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v5_unet_3d_%j.out
#SBATCH --error=runs/ablation/logs/p2v5_unet_3d_%j.err

# ============================================================================
# Phase 2 v5: UNet 3D — 128³ ROI, Optimized (2× A100 80GB DDP)
# ============================================================================
# v4→v5 changes based on CSC source, Lauenburg (competition), Auto3DSeg:
#
#   1. Optimizer:  RAdam wd=0     → AdamW wd=1e-4   (Lauenburg + all baselines)
#   2. LR:         4e-4          → 1e-4             (CSC default, Auto3DSeg=2e-4)
#   3. Iters:      300/epoch     → 500/epoch        (300K→500K total steps)
#   4. eta_min:    0 (LR dies)   → 1e-6             (Lauenburg CosineAnnealing)
#   5. Warmup:     34 epochs     → 50 epochs (5%)   (Auto3DSeg: 5% of total)
#   6. Intensity:  none          → --intensity_aug   (Lauenburg: GaussianBlur p=0.2)
#   7. Loss:       dice_bce 0.5  → bce_weight=0.4   (Lauenburg: 60% dice / 40% BCE)
#   8. Smooth:     1e-6          → 1e-3             (Lauenburg: smooth=1e-3)
#   9. Grad clip:  1.0           → 0.5              (Lauenburg=0.5)
#  10. Val freq:   every 5       → every 10          (longer runs, save val time)
#
# Architecture: CSC UNet 3D (~31M params)
#   - InstanceNorm + dropout=0.1
#   - DDP across 2 GPUs: batch=4/GPU → effective batch=8
#   - 128³ crops at 8nm spacing (CSC default ROI)
#
# Recipe: dice_bce(0.6/0.4) + AdamW + EMA(0.999) + fg_mask + weighted_sampler
# Training: 1000 epochs × 500 iters = 500K steps, eff_batch=8, 128³ @ 8nm
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

# --- NCCL tuning for intra-node NVLink ---
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

echo "============================================"
echo "Phase 2 v5: UNet 3D — 128³ ROI (Optimized)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

torchrun --nproc_per_node=2 --master_port=29501 \
    -m training.train \
    --experiment_name p2v5_unet_3d \
    --run_dir runs/ablation \
    --model unet_3d \
    --loss dice_bce \
    --loss_kwargs '{"bce_weight": 0.4, "smooth": 1e-3}' \
    --batch_size 4 \
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
    --filter_by_scale \
    --model_kwargs '{"use_instancenorm": true, "dropout": 0.1}'
