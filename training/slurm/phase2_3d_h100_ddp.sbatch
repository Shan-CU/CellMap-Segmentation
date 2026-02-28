#!/bin/bash
#SBATCH --partition=h100_mn
#SBATCH --account=rc_alain_pi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=1200G
#SBATCH --time=5-00:00:00
#SBATCH --output=runs/ablation/logs/%x_%j.out
#SBATCH --error=runs/ablation/logs/%x_%j.err

# ============================================================================
# Phase 2: 3D training on 4× H100 via DDP (NVLink)
# ============================================================================
# Uses torchrun to launch 4 processes (one per GPU).
# Each process loads its own CellMapDataLoader with different random seeds.
# Model wrapped in DistributedDataParallel for gradient sync via NCCL+NVLink.
#
# Node specs: 4× H100 80GB, 256 cores, 1.5TB RAM, NVLink
# ============================================================================

set -euo pipefail

EXPERIMENT_NAME="${EXPERIMENT_NAME:?Must set EXPERIMENT_NAME}"
MODEL_NAME="${MODEL_NAME:-segresnet_3d}"
LOSS_NAME="${LOSS_NAME:?Must set LOSS_NAME}"
LOSS_KWARGS="${LOSS_KWARGS:-}"
USE_FG_MASK="${USE_FG_MASK:-true}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-300}"
ITERS="${ITERS:-1000}"
NUM_GPUS="${NUM_GPUS:-4}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "============================================"
echo "Phase 2 DDP: ${EXPERIMENT_NAME}"
echo "Model: ${MODEL_NAME} | Loss: ${LOSS_NAME}"
echo "Node: $(hostname) | GPUs: ${NUM_GPUS}"
echo "Batch: ${BATCH_SIZE}/GPU × ${NUM_GPUS} = $((BATCH_SIZE * NUM_GPUS)) effective"
echo "Steps: ${EPOCHS} ep × ${ITERS} it = $((EPOCHS * ITERS)) steps"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

eval "$(micromamba shell hook -s bash)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# NCCL optimizations for NVLink
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
    --warmup_epochs 10 \
    --validation_time_limit 300 \
    --val_every_n_epochs 10 \
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
