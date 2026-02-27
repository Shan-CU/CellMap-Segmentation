#!/bin/bash
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=768G
#SBATCH --time=11-00:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Phase 2: 3D training on 4× A100 80GB via DDP (Longleaf a100-multi-gpu)
# ============================================================================
# Node: g180701 — 8× A100 SXM4 80GB, NVLink, 1TB RAM
# We request 2 GPUs so 4 jobs can run concurrently (8 GPUs / 2 = 4 slots).
# Effective batch = 2/GPU × 2 = 4 (sufficient for BatchNorm).
# Uses torchrun for DDP — same training code as H100 script.
#
# Key differences from H100 script:
#   - Longleaf partition/account/QoS
#   - Longleaf micromamba paths
#   - 192G RAM (node has 1TB total; 2 GPUs → fits 4 concurrent jobs)
#   - 11-day wall time (partition max)
#   - 16 CPUs (node has ~128 total; 2 GPUs → fits 4 concurrent jobs)
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
NUM_GPUS="${NUM_GPUS:-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================"
echo "Phase 2 DDP (A100): ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_NAME} | Loss: ${LOSS_NAME}"
echo "Node: $(hostname) | GPUs: ${NUM_GPUS}"
echo "Batch: ${BATCH_SIZE}/GPU × ${NUM_GPUS} = $((BATCH_SIZE * NUM_GPUS)) effective"
echo "Steps: ${EPOCHS} ep × ${ITERS} it = $((EPOCHS * ITERS)) steps"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Longleaf micromamba setup
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# Prevent host-RAM OOM from CellMapDataLoader.refresh() memory leak:
# refresh() recreates the PyTorch DataLoader each epoch. With
# persistent_workers=True (default), old worker processes stay alive and
# accumulate TensorStore chunk cache. Fix: --persistent_workers false.
export CELLMAP_TENSORSTORE_CACHE_BYTES=$((512 * 1024 * 1024))

# NCCL optimizations for NVLink (A100 SXM4 has NVLink)
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

CMD="torchrun \
    --standalone \
    --nproc_per_node=${NUM_GPUS} \
    -m training.train \
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
    --best_metric val_dice \
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
