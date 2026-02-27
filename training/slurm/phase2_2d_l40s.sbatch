#!/bin/bash
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Phase 2: 2D Architecture Comparison on Longleaf L40S (single GPU)
# ============================================================================
# Recipe: dice_bce + EMA(0.999) + fg_mask ON + weighted_sampler ON
# Models: resnet_2d, unet_2d, swin_2d, vit_2d
# Training: 100 epochs × 1000 iters, batch=8, 256×256 patches
# Validation: every 10 epochs with per-class Dice (48 classes)
# Best model saved by val_dice; periodic checkpoints every 10 epochs
# ============================================================================

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?Must set EXPERIMENT_NAME}"
MODEL_NAME="${MODEL_NAME:-resnet_2d}"
LOSS_NAME="${LOSS_NAME:?Must set LOSS_NAME}"
LOSS_KWARGS="${LOSS_KWARGS:-}"
USE_FG_MASK="${USE_FG_MASK:-true}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-100}"
ITERS="${ITERS:-1000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================"
echo "Phase 2 (L40S 2D): ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_NAME} | Loss: ${LOSS_NAME}"
echo "Cluster: Longleaf | Partition: l40-gpu"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Batch: ${BATCH_SIZE} (single GPU)"
echo "============================================"

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

CMD="python -m training.train \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_dir runs/ablation \
    --model ${MODEL_NAME} \
    --loss ${LOSS_NAME} \
    --batch_size ${BATCH_SIZE} \
    --input_shape 1 256 256 \
    --input_scale 8 8 8 \
    --epochs ${EPOCHS} \
    --iterations_per_epoch ${ITERS} \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_time_limit 600 \
    --val_every_n_epochs 10 \
    --best_metric val_dice \
    --num_workers 8 \
    --amp"

if [ "${USE_FG_MASK}" = "false" ]; then
    CMD="${CMD} --no_foreground_mask"
fi

if [ -n "${LOSS_KWARGS}" ]; then
    CMD="${CMD} --loss_kwargs ${LOSS_KWARGS}"
fi

if [ -n "${EXTRA_ARGS}" ]; then
    CMD="${CMD} ${EXTRA_ARGS}"
fi

echo "Running: ${CMD}"
eval ${CMD}
echo "Done: ${EXPERIMENT_NAME}"
