#!/bin/bash
# ============================================================================
# Phase 2: Architecture Comparison — Clean Launch Script
# ============================================================================
# Submits all 9 experiments (4× 2D + 5× 3D) with the Phase 1 winning recipe:
#   dice_bce + EMA(0.999) + fg_mask ON + weighted_sampler ON
#
# Usage:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_phase2_clean.sh [--3d-only] [--2d-only]
#
# Run directory structure (all under runs/ablation/):
#   arch_2d_resnet/   → {config.json, checkpoints/, tensorboard/}
#   arch_2d_unet/
#   arch_2d_swin/
#   arch_2d_vit/
#   arch_3d_segresnet/
#   arch_3d_swinunetr/
#   arch_3d_unet/
#   arch_3d_resnet/
#   arch_3d_vitnet/
# ============================================================================

set -euo pipefail
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# Phase 2 recipe (from Phase 1 ablation winner)
LOSS="dice_bce"
COMMON_2D="--ema --ema_decay 0.999"
COMMON_3D="--ema --ema_decay 0.999"

# Parse args
LAUNCH_2D=true
LAUNCH_3D=true
for arg in "$@"; do
    case $arg in
        --3d-only) LAUNCH_2D=false ;;
        --2d-only) LAUNCH_3D=false ;;
    esac
done

echo "============================================"
echo "Phase 2: Architecture Comparison"
echo "Recipe: ${LOSS} + EMA(0.999) + fg_mask + weighted_sampler"
echo "============================================"
echo ""

# ============================================================================
# 3D MODELS (submitted first to monitor OOM fix)
# L40S single GPU, batch=2, 128³ patches, 1000ep × 300it
# persistent_workers=false to prevent refresh() memory leak
# ============================================================================
if $LAUNCH_3D; then
    echo "--- 3D Models (L40S, single GPU, batch=2) ---"

    for spec in \
        "segresnet:segresnet_3d:2:--deep_supervision" \
        "swinunetr:swinunetr_3d:2:" \
        "unet:unet_3d:2:" \
        "resnet:resnet_3d:2:" \
        "vitnet:vitnet_3d:1:"; do

        IFS=':' read -r NAME MODEL BATCH EXTRA <<< "$spec"
        JOB="arch_3d_${NAME}"
        echo "  Submitting ${JOB} (model=${MODEL}, batch=${BATCH})..."

        EXPERIMENT_NAME="${JOB}" \
        MODEL_NAME="${MODEL}" \
        LOSS_NAME="${LOSS}" \
        USE_FG_MASK="true" \
        BATCH_SIZE="${BATCH}" \
        EPOCHS=1000 \
        ITERS=300 \
        EXTRA_ARGS="${COMMON_3D} ${EXTRA}" \
        sbatch --job-name="${JOB}" training/slurm/phase2_3d_l40s.sbatch
    done

    echo ""
    echo "3D jobs submitted (5 total)."
    echo ""
fi

# ============================================================================
# 2D MODELS
# L40S single GPU, batch=8, 256×256 patches, 100ep × 1000it
# ============================================================================
if $LAUNCH_2D; then
    echo "--- 2D Models (L40S, single GPU, batch=8) ---"

    for spec in \
        "resnet:resnet_2d" \
        "unet:unet_2d" \
        "swin:swin_2d" \
        "vit:vit_2d"; do

        IFS=':' read -r NAME MODEL <<< "$spec"
        JOB="arch_2d_${NAME}"
        echo "  Submitting ${JOB} (model=${MODEL})..."

        EXPERIMENT_NAME="${JOB}" \
        MODEL_NAME="${MODEL}" \
        LOSS_NAME="${LOSS}" \
        USE_FG_MASK="true" \
        BATCH_SIZE=8 \
        EPOCHS=100 \
        ITERS=1000 \
        EXTRA_ARGS="${COMMON_2D}" \
        sbatch --job-name="${JOB}" training/slurm/phase2_2d_l40s.sbatch
    done

    echo ""
    echo "2D jobs submitted (4 total)."
    echo ""
fi

# ============================================================================
# TENSORBOARD
# ============================================================================
echo "--- TensorBoard ---"
sbatch --job-name="tb_phase2" training/slurm/tensorboard.sbatch
echo ""

echo "============================================"
echo "All jobs submitted. Monitor with:"
echo "  squeue -u gsgeorge"
echo "  ssh -L 6006:<tb_node>:6006 longleaf.unc.edu"
echo "============================================"
