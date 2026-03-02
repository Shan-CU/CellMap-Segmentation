#!/bin/bash
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Phase 2 v3: Enhanced 2D transformer training on A100 80GB (single GPU)
# ============================================================================
# Used for ViT/Swin enhanced runs that benefit from:
#   - Larger batch sizes (batch=16, 2-4× baseline)
#   - A100 Ampere GPU avoids L40S Ada cudaErrorInvalidConfiguration bug
#   - More pixels/step → more diverse gradients → helps data-hungry ViTs
#   - Weight decay 0.05 (DeiT-style regularizer for transformers)
#
# Same recipe as L40S 2D sbatch (dice_bce + EMA + fg_mask) with:
#   - LR scaled linearly with batch size (base 1e-4 @ batch=8)
# ============================================================================

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?Must set EXPERIMENT_NAME}"
MODEL_NAME="${MODEL_NAME:-resnet_2d}"
LOSS_NAME="${LOSS_NAME:?Must set LOSS_NAME}"
LOSS_KWARGS="${LOSS_KWARGS:-}"
USE_FG_MASK="${USE_FG_MASK:-true}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-100}"
ITERS="${ITERS:-1000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================"
echo "Phase 2 v3 (A100 80GB 2D): ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_NAME} | Loss: ${LOSS_NAME}"
echo "Cluster: Longleaf | Partition: a100-multi-gpu"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
echo "Batch: ${BATCH_SIZE} (single GPU)"
echo "============================================"

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

# Linear LR scaling: base_lr=1e-4 at batch=8, scale proportionally.
BASE_LR="1e-4"
BASE_BATCH=8
SCALED_LR=$(python3 -c "print(f'{${BASE_LR} * ${BATCH_SIZE} / ${BASE_BATCH}:.1e}')")
echo "LR scaling: base=${BASE_LR} @ batch=${BASE_BATCH} → ${SCALED_LR} @ batch=${BATCH_SIZE}"

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
    --learning_rate ${SCALED_LR} \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --warmup_epochs 10 \
    --validation_time_limit 600 \
    --val_every_n_epochs 5 \
    --best_metric val_dice \
    --num_workers 8 \
    --amp \
    --ema --ema_decay 0.999"

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
