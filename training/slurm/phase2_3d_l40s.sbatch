#!/bin/bash
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Phase 2: 3D training on Longleaf L40S (single GPU, batch=8)
# ============================================================================
# VRAM measured (canary v7, SegResNetDS 96^3 AMP, random tensors):
#   batch=8  → 10,579 MB (23%)  |  batch=12 → 15,732 MB (35%)
#   batch=16 → 20,882 MB (46%)  |  ~1,288 MB per sample
# batch=8 is the default: 4× throughput vs batch=2, 77% VRAM headroom.
# Override with BATCH_SIZE=12 or 16 if you want to push it.
# LR is scaled linearly: base 1e-4 at batch=2, so batch=8 → 4e-4.
#
# Crop size: 96^3 @ 8nm (768nm physical box) instead of 128^3 (1024nm).
# This nearly doubles usable training data: 78 → 152 datasets.
# Many CellMap crops (hela, jurkat, macrophage, fly) have min extent ~680-800nm,
# just under the 1024nm threshold but comfortably above 768nm.
# 96^3 still provides ample spatial context for all organelle classes.
#
# Memory leak fixes applied:
#   1. cellmap-data >= 2026.2.28.30 (PR #64): clears xarray DataArray cache
#   2. Monkey-patch in train.py: clears _current_coords (~48 MB/image/step)
#   3. glibc malloc tuning: MALLOC_ARENA_MAX=2 + aggressive trim/mmap thresholds
#
# Canary v5 (num_workers=0) measured ~3.1 GB steady-state, ~2 MB/step growth.
# With num_workers=4, forked workers get CoW copies of ~3 GB main process +
# their own TensorStore I/O buffers. 128G gives generous headroom.
# (L40S nodes have 1 TB RAM / 8 GPUs ≈ 125 GB per GPU slot.)
#
# Workers matter: with num_workers=0, each step is ~57s and data loading is
# ~47s of that. Workers pipeline data loading with GPU compute.
#
# NOTE: The old "1 worker" advice (class_weight_experiment.md) was written
# before the memory leak fix — the ~6 GB/worker was actually the leak.
# With fixes applied, workers are safe. malloc tuning env vars are inherited.
# ============================================================================

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?Must set EXPERIMENT_NAME}"
MODEL_NAME="${MODEL_NAME:-segresnet_3d}"
LOSS_NAME="${LOSS_NAME:?Must set LOSS_NAME}"
LOSS_KWARGS="${LOSS_KWARGS:-}"
USE_FG_MASK="${USE_FG_MASK:-true}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EPOCHS="${EPOCHS:-1000}"
ITERS="${ITERS:-300}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================"
echo "Phase 2 (L40S 3D): ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_NAME} | Loss: ${LOSS_NAME}"
echo "Cluster: Longleaf | Partition: l40-gpu"
echo "Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Batch: ${BATCH_SIZE} (single GPU) | LR will be scaled linearly"
echo "============================================"

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# --- Memory leak mitigations ---
# Limit TensorStore chunk cache (cellmap-data reads this env var)
export CELLMAP_TENSORSTORE_CACHE_BYTES=1

# Force glibc to return freed memory to OS aggressively.
# Without these, scipy/numpy temporary allocations fragment the heap and
# RSS grows ~940 MB/step even after Python-level leaks are fixed.
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536

# Linear LR scaling: base_lr=1e-4 at batch=2, scale proportionally.
# E.g. batch=4 → 2e-4, batch=6 → 3e-4, batch=8 → 4e-4.
BASE_LR="1e-4"
BASE_BATCH=2
SCALED_LR=$(python3 -c "print(f'{${BASE_LR} * ${BATCH_SIZE} / ${BASE_BATCH}:.1e}')")
echo "LR scaling: base=${BASE_LR} @ batch=${BASE_BATCH} → ${SCALED_LR} @ batch=${BATCH_SIZE}"

CMD="python -m training.train \
    --experiment_name ${EXPERIMENT_NAME} \
    --run_dir runs/ablation \
    --model ${MODEL_NAME} \
    --loss ${LOSS_NAME} \
    --batch_size ${BATCH_SIZE} \
    --input_shape 96 96 96 \
    --input_scale 8 8 8 \
    --epochs ${EPOCHS} \
    --iterations_per_epoch ${ITERS} \
    --learning_rate ${SCALED_LR} \
    --max_grad_norm 1.0 \
    --scheduler cosine \
    --warmup_epochs 34 \
    --validation_time_limit 600 \
    --val_every_n_epochs 5 \
    --best_metric val_dice \
    --num_workers 4 \
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
