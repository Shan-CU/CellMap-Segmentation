#!/bin/bash
#SBATCH --job-name=p2v4_segresnet_3d
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v4_segresnet_3d_%j.out
#SBATCH --error=runs/ablation/logs/p2v4_segresnet_3d_%j.err

# ============================================================================
# Phase 2 v4: SegResNetDS 3D — Multi-GPU DDP (2× A100 80GB)
# ============================================================================
# Architecture: MONAI SegResNetDS (model_zoo.py) — ~87M params
#   - blocks_down=(1,2,2,4,4), dsdepth=4, InstanceNorm
#   - 5-level encoder, deep supervision with 4 auxiliary outputs
#
# WHY MULTI-GPU:
#   Single-GPU OOM'd at ROI=224³ batch=4 (forward: ~62GB + loss: ~26GB > 80GB).
#   Auto3DSeg defaults to batch=1 with large ROI. Their philosophy:
#     auto_scale_roi: false  — NEVER shrink ROI
#     auto_scale_batch: true — shrink batch to fit
#   We use 2 GPUs with DDP: batch=1/GPU → effective batch=2
#   + gradient_accumulation_steps=2 → effective batch=4
#   Each GPU: ~35GB peak (well within 80GB headroom)
#
# Memory math per GPU:
#   Forward+backward ROI=224³ batch=1: ~17GB (from profiler)
#   DS loss overhead (224³×48 fp32): ~6.5GB
#   Total: ~24GB — 56GB headroom per GPU
#
# DDP details:
#   - torchrun --nproc_per_node=2 spawns 2 processes
#   - train.py has built-in DDP (setup_ddp, DistributedDataParallel)
#   - Per-rank seeding ensures different random crops per GPU
#   - Only rank 0 logs, saves checkpoints, writes TensorBoard
#   - NCCL backend with SXM4 NVLink for fast gradient sync
#
# Recipe: dice_bce + EMA(0.999) + fg_mask + weighted_sampler + intensity_aug
# Training: 1000 epochs × 300 iters, effective_batch=4, 224³ crops @ 8nm
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- NCCL tuning for intra-node NVLink ---
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL

echo "============================================"
echo "Phase 2 v4: SegResNetDS 3D — Multi-GPU DDP (2× A100 80GB)"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "============================================"

torchrun --nproc_per_node=2 --master_port=29500 \
    -m training.train \
    --experiment_name p2v4_segresnet_3d \
    --run_dir runs/ablation \
    --model segresnet_3d \
    --loss dice_bce \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --input_shape 224 224 224 \
    --input_scale 8 8 8 \
    --epochs 1000 \
    --iterations_per_epoch 300 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_time_limit 600 \
    --val_every_n_epochs 5 \
    --best_metric val_dice \
    --num_workers 4 \
    --amp \
    --ema --ema_decay 0.999 \
    --bias_init_mode per_class \
    --optimizer adamw \
    --weight_decay 1e-5 \
    --deep_supervision \
    --intensity_aug \
    --random_seed 42 \
    --use_foreground_mask \
    --datasplit_path datasplit.csv \
    --filter_by_scale
