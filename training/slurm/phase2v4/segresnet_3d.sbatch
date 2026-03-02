#!/bin/bash
#SBATCH --job-name=p2v4_segresnet_3d
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/p2v4_segresnet_3d_%j.out
#SBATCH --error=runs/ablation/logs/p2v4_segresnet_3d_%j.err

# ============================================================================
# Phase 2 v4: SegResNetDS 3D on A100 80GB
# ============================================================================
# Architecture: MONAI SegResNetDS (model_zoo.py) — ~87M params
#   - FULLY RECONFIGURED from v2 to match MONAI Auto3DSeg template:
#     v2 (WRONG):   blocks_down=(1,2,2,4), dsdepth=1, BatchNorm, ~20M params
#     v4 (CORRECT): blocks_down=(1,2,2,4,4), dsdepth=4, InstanceNorm, ~87M
#   - 5-level encoder, deep supervision with 4 auxiliary outputs
#
# Optimizer: AdamW, lr=2e-4, wd=1e-5
#   - From MONAI Auto3DSeg SegResNet template (hyper_parameters.yaml):
#     AdamW, lr=2e-4, wd=1e-5
#   - LR: base 1e-4 @ batch=2, scaled to 2e-4 @ batch=4
#   - batch=4 (not 8) because 87M model + deep supervision is memory heavy
#
# Key fixes:
#   - bias_init=-3.0 (RetinaNet focal prior)
#   - --deep_supervision flag (wraps loss in PartialAnnotationDeepSupervisionLoss)
#   - InstanceNorm (was BatchNorm in v2)
#   - 5-level encoder (was 4-level in v2)
#
# Recipe: dice_bce + EMA(0.999) + fg_mask + weighted_sampler
# Training: 1000 epochs × 300 iters, batch=4, 96³ crops @ 8nm
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

echo "============================================"
echo "Phase 2 v4: SegResNetDS 3D (AdamW + DeepSup + 87M params)"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "============================================"

python -m training.train \
    --experiment_name p2v4_segresnet_3d \
    --run_dir runs/ablation \
    --model segresnet_3d \
    --loss dice_bce \
    --batch_size 4 \
    --input_shape 96 96 96 \
    --input_scale 8 8 8 \
    --epochs 1000 \
    --iterations_per_epoch 300 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --warmup_epochs 34 \
    --validation_time_limit 600 \
    --val_every_n_epochs 5 \
    --best_metric val_dice \
    --num_workers 4 \
    --amp \
    --ema --ema_decay 0.999 \
    --bias_init -3.0 \
    --optimizer adamw \
    --weight_decay 1e-5 \
    --deep_supervision \
    --random_seed 42 \
    --use_foreground_mask \
    --datasplit_path datasplit.csv \
    --filter_by_scale
