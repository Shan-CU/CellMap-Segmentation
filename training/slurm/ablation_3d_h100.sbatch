#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --account=rc_alain_pi
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Phase 1 Ablation: 3D experiment on Sycamore H100
# ============================================================================

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?Must set EXPERIMENT_NAME}"
MODEL_NAME="${MODEL_NAME:-segresnet_3d}"
LOSS_NAME="${LOSS_NAME:?Must set LOSS_NAME}"
LOSS_KWARGS="${LOSS_KWARGS:-}"
USE_FG_MASK="${USE_FG_MASK:-true}"
BATCH_SIZE="${BATCH_SIZE:-1}"
EPOCHS="${EPOCHS:-50}"
ITERS="${ITERS:-500}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================"
echo "Ablation: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_NAME} | Loss: ${LOSS_NAME}"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

eval "$(micromamba shell hook -s bash)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

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
    --warmup_epochs 5 \
    --validation_time_limit 300 \
    --val_every_n_epochs 5 \
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
