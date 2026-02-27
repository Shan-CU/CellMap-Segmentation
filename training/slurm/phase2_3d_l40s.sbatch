#!/bin/bash
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=768G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Phase 2: 3D training on Longleaf L40S (single GPU, batch=2)
# ============================================================================
# Fallback for when A100 is unavailable. Most 3D models (20-60M params)
# fit in L40S 48GB VRAM at batch=2 with AMP.
# batch=2 is the nnU-Net default — works fine with BatchNorm.
# UNet_3D (90M) may be tight; monitor first epoch for OOM.
# ============================================================================

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?Must set EXPERIMENT_NAME}"
MODEL_NAME="${MODEL_NAME:-segresnet_3d}"
LOSS_NAME="${LOSS_NAME:?Must set LOSS_NAME}"
LOSS_KWARGS="${LOSS_KWARGS:-}"
USE_FG_MASK="${USE_FG_MASK:-true}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-1000}"
ITERS="${ITERS:-300}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================"
echo "Phase 2 (L40S 3D): ${EXPERIMENT_NAME}"
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

# Prevent host-RAM OOM from CellMapDataLoader.refresh() memory leak:
# refresh() recreates the PyTorch DataLoader each epoch. With
# persistent_workers=True (default), old worker processes stay alive and
# accumulate TensorStore chunk cache, causing OOM at ~536G within epoch 1.
# Fix: --persistent_workers false lets workers die on refresh, releasing cache.
# Safety: also cap TensorStore cache to 512 MiB (default 2 GiB).
export CELLMAP_TENSORSTORE_CACHE_BYTES=$((512 * 1024 * 1024))

CMD="python -m training.train \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_dir runs/ablation \
    --model ${MODEL_NAME} \
    --loss ${LOSS_NAME} \
    --batch_size ${BATCH_SIZE} \
    --input_shape 128 128 128 \
    --input_scale 8 8 8 \
    --epochs ${EPOCHS} \
    --iterations_per_epoch ${ITERS} \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --warmup_epochs 34 \
    --validation_time_limit 600 \
    --val_every_n_epochs 30 \
    --num_workers 4 \
    --persistent_workers false \
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
