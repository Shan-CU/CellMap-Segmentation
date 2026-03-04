#!/bin/bash
#SBATCH --job-name=p2v4_unet_3d_128
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v4_unet_3d_128_%j.out
#SBATCH --error=runs/ablation/logs/p2v4_unet_3d_128_%j.err

# ============================================================================
# Phase 2 v4: UNet 3D — 128³ ROI, Multi-GPU DDP (2× A100 80GB)
# ============================================================================
# Architecture: CSC UNet 3D (~31M params)
#   - InstanceNorm + dropout=0.1
#
# FIX: ROI 96³ → 128³ to match CSC example default (train_3D.py).
#   - 96³ = 884K voxels → 128³ = 2.1M voxels (2.37× more context)
#   - Memory at 128³ b=8 single GPU: ~102GB (exceeds 80GB)
#   - Solution: DDP across 2 GPUs, batch=4/GPU = effective batch=8
#   - Per-GPU memory: ~51GB (safe margin on 80GB)
#
# WHY DDP:
#   UNet 3D has skip connections that store full-resolution feature maps.
#   At 96³ b=8 it used 43GB. At 128³ that scales to ~102GB (2.37×).
#   Splitting batch=4 per GPU brings it to ~51GB each.
#
# Recipe: dice_bce + EMA(0.999) + fg_mask + weighted_sampler
# Training: 1000 epochs × 300 iters, effective_batch=8, 128³ crops @ 8nm
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
echo "Phase 2 v4: UNet 3D — 128³ ROI, Multi-GPU DDP (2× A100 80GB)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

torchrun --nproc_per_node=2 --master_port=29501 \
    -m training.train \
    --experiment_name p2v4_unet_3d_128 \
    --run_dir runs/ablation \
    --model unet_3d \
    --loss dice_bce \
    --batch_size 4 \
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
    --model_kwargs '{"use_instancenorm": true, "dropout": 0.1}' \
    --random_seed 42 \
    --use_foreground_mask \
    --datasplit_path datasplit.csv \
    --filter_by_scale
