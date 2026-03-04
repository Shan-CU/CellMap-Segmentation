#!/bin/bash
#SBATCH --job-name=p2v5_segresnet_3d
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v5_segresnet_3d_%j.out
#SBATCH --error=runs/ablation/logs/p2v5_segresnet_3d_%j.err

# ============================================================================
# Phase 2 v5: SegResNetDS 3D — Multi-GPU DDP (2× A100 80GB)
# ============================================================================
# v4 was already well-configured (matched Auto3DSeg closely). v5 additions:
#
#   1. eta_min:    0 (LR dies)   → 1e-6             (prevent LR collapse at end)
#   2. Loss:       dice_bce 0.5  → bce_weight=0.4   (Lauenburg: 60% dice / 40% BCE)
#   3. Smooth:     1e-6          → 1e-3             (Lauenburg: smooth=1e-3)
#   4. Grad clip:  1.0           → 0.5              (Lauenburg=0.5, more stable)
#   5. Iters:      300/epoch     → 500/epoch        (300K→500K total steps)
#   6. Val freq:   every 5       → every 10          (save val time, still enough)
#
# Kept from v4 (already optimal per Auto3DSeg research):
#   ✅ AdamW lr=2e-4 wd=1e-5 (exact Auto3DSeg match)
#   ✅ Cosine scheduler with warmup (functionally = WarmupCosineSchedule)
#   ✅ Deep supervision enabled
#   ✅ Intensity augmentation enabled
#   ✅ ROI 224³ (Auto3DSeg computed optimal)
#   ✅ Batch=1/GPU × 2GPU × grad_accum=2 = eff_batch=4
#
# Architecture: MONAI SegResNetDS (~87M params)
#   - blocks_down=(1,2,2,4,4), dsdepth=4, 5-level encoder
#   - Deep supervision with 4 auxiliary outputs
#   - DDP across 2 GPUs, ~35GB peak per GPU
#
# Recipe: dice_bce(0.6/0.4) + AdamW + EMA(0.999) + fg_mask + deep_supervision
# Training: 1000 epochs × 500 iters = 500K steps, eff_batch=4, 224³ @ 8nm
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
echo "Phase 2 v5: SegResNetDS 3D — Multi-GPU DDP (2× A100 80GB)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

torchrun --nproc_per_node=2 --master_port=29500 \
    -m training.train \
    --experiment_name p2v5_segresnet_3d \
    --run_dir runs/ablation \
    --model segresnet_3d \
    --loss dice_bce \
    --loss_kwargs '{"bce_weight": 0.4, "smooth": 1e-3}' \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --input_shape 224 224 224 \
    --input_scale 8 8 8 \
    --epochs 1000 \
    --iterations_per_epoch 500 \
    --learning_rate 2e-4 \
    --optimizer adamw \
    --weight_decay 1e-5 \
    --max_grad_norm 0.5 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --eta_min 1e-6 \
    --validation_time_limit 600 \
    --val_every_n_epochs 10 \
    --best_metric val_dice \
    --num_workers 4 \
    --amp \
    --ema --ema_decay 0.999 \
    --bias_init_mode per_class \
    --deep_supervision \
    --intensity_aug \
    --random_seed 42 \
    --use_foreground_mask \
    --datasplit_path datasplit.csv \
    --filter_by_scale
