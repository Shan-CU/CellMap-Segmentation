#!/bin/bash
# ============================================================================
# Phase 2: Architecture Comparison — Launch all 9 experiments
# ============================================================================
# Recipe (from Phase 1 ablation):
#   Loss: dice_bce | EMA: 0.999 | FG mask: on | Weighted sampler: on
#   Intensity aug: OFF | Class-aware crop: OFF
#   2D: 100 epochs × 1000 iters (single GPU, Longleaf L40S)
#   3D: 300 epochs × 1000 iters (2-GPU DDP, A100 80GB)
#
# Usage:
#   bash launch_phase2.sh 2d       # submit 2D jobs
#   bash launch_phase2.sh 3d       # submit 3D jobs
#   bash launch_phase2.sh all      # submit both
#
# 2D models → Longleaf L40S (4 jobs, single GPU)
# 3D models → Longleaf A100 80GB (a100-multi-gpu) or Sycamore H100 (h100_mn)
# ============================================================================

set -euo pipefail

PHASE2_COMMON_ARGS="--ema --ema_decay 0.999"
LOSS="dice_bce"

echo "=============================================="
echo "Phase 2: Architecture Comparison"
echo "=============================================="
echo "Loss: ${LOSS} | EMA: 0.999 | FG mask: on"
echo ""

# ---- Parse arguments ----
LAUNCH_2D=false
LAUNCH_3D=false
for arg in "$@"; do
    case "$arg" in
        2d) LAUNCH_2D=true ;;
        3d) LAUNCH_3D=true ;;
        all) LAUNCH_2D=true; LAUNCH_3D=true ;;
        *) echo "Usage: $0 [2d|3d|all]"; exit 1 ;;
    esac
done
if ! $LAUNCH_2D && ! $LAUNCH_3D; then
    echo "Usage: $0 [2d|3d|all]"
    echo "  2d  — submit 2D jobs (L40S, single GPU)"
    echo "  3d  — submit 3D jobs (A100 DDP or H100 DDP)"
    echo "  all — submit both"
    exit 1
fi

# ---- Detect which cluster we're on ----
HOSTNAME=$(hostname)
if [[ "$HOSTNAME" == *longleaf* ]]; then
    CLUSTER="longleaf"
elif [[ "$HOSTNAME" == *sycamore* ]]; then
    CLUSTER="sycamore"
else
    echo "ERROR: Unknown cluster. Run from longleaf or sycamore login node."
    exit 1
fi

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# ============================================================================
# 2D MODELS (Longleaf L40S, single GPU)
# ============================================================================
launch_2d() {
    local MODEL=$1
    local NAME="arch_2d_${MODEL}"
    echo "  Submitting ${NAME} → L40S..."
    EXPERIMENT_NAME="${NAME}" \
    MODEL_NAME="${MODEL}_2d" \
    LOSS_NAME="${LOSS}" \
    USE_FG_MASK="true" \
    BATCH_SIZE=8 \
    EPOCHS=100 \
    ITERS=1000 \
    EXTRA_ARGS="${PHASE2_COMMON_ARGS}" \
    sbatch --job-name="${NAME}" training/slurm/ablation_2d_l40s.sbatch
}

# ============================================================================
# 3D MODELS — DDP (4 GPUs)
#   Longleaf: a100-multi-gpu (A100 80GB SXM4, NVLink)
#   Sycamore: h100_mn (H100 80GB, NVLink)  [kept for future access]
# ============================================================================
launch_3d() {
    local MODEL=$1
    local BATCH=$2
    local EXTRA=$3
    local NAME="arch_3d_${MODEL}"

    if [[ "$CLUSTER" == "longleaf" ]]; then
        echo "  Submitting ${NAME} → A100 DDP (2 GPUs)..."
        EXPERIMENT_NAME="${NAME}" \
        MODEL_NAME="${MODEL}_3d" \
        LOSS_NAME="${LOSS}" \
        USE_FG_MASK="true" \
        BATCH_SIZE="${BATCH}" \
        EPOCHS=300 \
        ITERS=1000 \
        NUM_GPUS=2 \
        EXTRA_ARGS="${PHASE2_COMMON_ARGS} ${EXTRA}" \
        sbatch --job-name="${NAME}" training/slurm/phase2_3d_a100_ddp.sbatch
    elif [[ "$CLUSTER" == "sycamore" ]]; then
        echo "  Submitting ${NAME} → H100 DDP (4 GPUs)..."
        EXPERIMENT_NAME="${NAME}" \
        MODEL_NAME="${MODEL}_3d" \
        LOSS_NAME="${LOSS}" \
        USE_FG_MASK="true" \
        BATCH_SIZE="${BATCH}" \
        EPOCHS=300 \
        ITERS=1000 \
        NUM_GPUS=4 \
        EXTRA_ARGS="${PHASE2_COMMON_ARGS} ${EXTRA}" \
        sbatch --job-name="${NAME}" training/slurm/phase2_3d_h100_ddp.sbatch
    fi
}

# ============================================================================
# LAUNCH
# ============================================================================

if $LAUNCH_2D; then
    if [[ "$CLUSTER" != "longleaf" ]]; then
        echo "WARNING: 2D jobs should be launched from longleaf (l40-gpu). Skipping."
    else
        echo ""
        echo "--- 2D Models (Longleaf L40S) ---"
        launch_2d "resnet"
        launch_2d "unet"
        launch_2d "swin"
        launch_2d "vit"
        echo ""
        echo "2D jobs submitted."
    fi
fi

if $LAUNCH_3D; then
    echo ""
    if [[ "$CLUSTER" == "longleaf" ]]; then
        echo "--- 3D Models (Longleaf A100 80GB, 2-GPU DDP) ---"
    elif [[ "$CLUSTER" == "sycamore" ]]; then
        echo "--- 3D Models (Sycamore H100 80GB, 4-GPU DDP) ---"
    fi
    launch_3d "segresnet" 2 "--deep_supervision"
    launch_3d "swinunetr" 2 ""
    launch_3d "unet"      2 ""
    launch_3d "resnet"    2 ""
    launch_3d "vitnet"    1 ""  # batch=1/GPU, global attention is memory hungry
    echo ""
    echo "3D jobs submitted."
fi

echo ""
echo "Monitor with: squeue -u gsgeorge"
echo "Done."
