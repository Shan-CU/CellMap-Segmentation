#!/bin/bash
#SBATCH --job-name=ablation_3d
#SBATCH --partition=h100_sn
#SBATCH --account=rc_alain_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Phase 1 Ablation: 3D experiments on L40S
#
# Runs a SINGLE experiment specified by environment variables.
# Submit via the launcher script: training/slurm/launch_ablation_3d.sh
# ============================================================================

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?Must set EXPERIMENT_NAME}"
MODEL_NAME="${MODEL_NAME:-segresnet_3d}"
LOSS_NAME="${LOSS_NAME:?Must set LOSS_NAME}"
LOSS_KWARGS="${LOSS_KWARGS:-{}}"
USE_FG_MASK="${USE_FG_MASK:-true}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-50}"
ITERS="${ITERS:-250}"

echo "============================================"
echo "Ablation Experiment (3D): ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_NAME} | Loss: ${LOSS_NAME}"
echo "Epochs: ${EPOCHS} | Iters/epoch: ${ITERS}"
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
    --validation_time_limit 180 \
    --val_every_n_epochs 5 \
    --amp"

if [ "${USE_FG_MASK}" = "false" ]; then
    CMD="${CMD} --no_foreground_mask"
fi

if [ "${LOSS_KWARGS}" != "{}" ]; then
    CMD="${CMD} --loss_kwargs '${LOSS_KWARGS}'"
fi

echo "Running: ${CMD}"
eval ${CMD}
echo "Done: ${EXPERIMENT_NAME}"
