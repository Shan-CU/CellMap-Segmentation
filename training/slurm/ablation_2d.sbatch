#!/bin/bash
#SBATCH --job-name=ablation_2d
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Phase 1 Ablation: 2D experiments on L40S
#
# Runs a SINGLE experiment specified by EXPERIMENT_NAME and LOSS_NAME.
# Submit via the launcher script: training/slurm/launch_ablation_2d.sh
# ============================================================================

set -euo pipefail

# These are set by the launcher script via --export
EXPERIMENT_NAME="${EXPERIMENT_NAME:?Must set EXPERIMENT_NAME}"
MODEL_NAME="${MODEL_NAME:-resnet_2d}"
LOSS_NAME="${LOSS_NAME:?Must set LOSS_NAME}"
LOSS_KWARGS="${LOSS_KWARGS:-{}}"
USE_FG_MASK="${USE_FG_MASK:-true}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-50}"
ITERS="${ITERS:-500}"

echo "============================================"
echo "Ablation Experiment: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_NAME} | Loss: ${LOSS_NAME}"
echo "Epochs: ${EPOCHS} | Iters/epoch: ${ITERS}"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================"

# Activate environment
eval "$(micromamba shell hook -s bash)"
micromamba activate csc

# Move to repo root
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# Create log directory
mkdir -p runs/ablation/logs

# Build command
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
    --warmup_epochs 5 \
    --validation_time_limit 120 \
    --val_every_n_epochs 5 \
    --amp"

# Add foreground mask flag
if [ "${USE_FG_MASK}" = "false" ]; then
    CMD="${CMD} --no_foreground_mask"
fi

# Add loss kwargs if non-empty
if [ "${LOSS_KWARGS}" != "{}" ]; then
    CMD="${CMD} --loss_kwargs '${LOSS_KWARGS}'"
fi

echo "Running: ${CMD}"
eval ${CMD}
echo "Done: ${EXPERIMENT_NAME}"
