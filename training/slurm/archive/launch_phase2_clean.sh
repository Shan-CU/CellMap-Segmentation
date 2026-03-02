#!/bin/bash
# ============================================================================
# Phase 2: Architecture Comparison — Clean Launch Script (v2)
# ============================================================================
# Submits all 9 experiments (4× 2D + 5× 3D) with the Phase 1 winning recipe:
#   dice_bce + EMA(0.999) + fg_mask ON + weighted_sampler ON
#
# v2 fixes (2026-02-28):
#   - EMA was missing from v1 launch (--ema flag not passed). Now baked into
#     both sbatch files. EXTRA_ARGS passes it too for safety.
#   - SegResNet switched from BST back to dice_bce + --deep_supervision.
#     Deep supervision prevents the logit explosion that made DiceBCE fail.
#   - ViT 2D batch reduced to 4 (CUDA kernel launch error at batch=8).
#   - All 3D models use batch=8, 96^3 crops, LR scaled linearly from 1e-4@b2.
#
# Usage:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_phase2_clean.sh [--3d-only] [--2d-only]
#
# Run directory structure (all under runs/ablation/):
#   p2_resnet_2d/   → {config.json, checkpoints/, tensorboard/}
#   p2_unet_2d/
#   p2_swin_2d/
#   p2_vit_2d/
#   p2_segresnet_3d/
#   p2_swinunetr_3d/
#   p2_unet_3d/
#   p2_resnet_3d/
#   p2_vitnet_3d/
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
# 3D MODELS
# L40S single GPU, batch=8, 96³ patches, 1000ep × 300it
# LR scaled linearly: base 1e-4 @ batch=2 → 4e-4 @ batch=8
# SegResNet uses dice_bce + deep_supervision (NOT BST).
# ============================================================================
if $LAUNCH_3D; then
    echo "--- 3D Models (L40S, single GPU, batch=8, 96³) ---"

    for spec in \
        "segresnet:segresnet_3d:8:--deep_supervision" \
        "swinunetr:swinunetr_3d:8:" \
        "unet:unet_3d:8:" \
        "resnet:resnet_3d:8:" \
        "vitnet:vitnet_3d:8:"; do

        IFS=':' read -r NAME MODEL BATCH EXTRA <<< "$spec"
        JOB="p2_${MODEL}"
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
# ViT uses batch=4 to avoid CUDA kernel launch error (cudaErrorInvalidConfiguration
# from BatchNorm2d backward pass at batch=8 + AMP on L40S Ada Lovelace).
# ============================================================================
if $LAUNCH_2D; then
    echo "--- 2D Models (L40S, single GPU) ---"

    for spec in \
        "resnet_2d:8" \
        "unet_2d:8" \
        "swin_2d:8" \
        "vit_2d:4"; do

        IFS=':' read -r MODEL BATCH <<< "$spec"
        JOB="p2_${MODEL}"
        echo "  Submitting ${JOB} (model=${MODEL}, batch=${BATCH})..."

        EXPERIMENT_NAME="${JOB}" \
        MODEL_NAME="${MODEL}" \
        LOSS_NAME="${LOSS}" \
        USE_FG_MASK="true" \
        BATCH_SIZE="${BATCH}" \
        EPOCHS=100 \
        ITERS=1000 \
        EXTRA_ARGS="${COMMON_2D}" \
        sbatch --job-name="${JOB}" training/slurm/phase2_2d_l40s.sbatch
    done

    echo ""
    echo "2D jobs submitted (4 total)."
    echo ""
fi

echo "============================================"
echo "All jobs submitted. Monitor with:"
echo "  squeue -u gsgeorge"
echo ""
echo "TensorBoard (run on sycamore):"
echo "  /nas/longleaf/home/gsgeorge/micromamba/envs/csc/bin/tensorboard \\"
echo "    --logdir /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/runs/ablation \\"
echo "    --port 6006 --host 0.0.0.0 --reload_interval 15 --reload_multifile true"
echo "============================================"
