#!/bin/bash
# ============================================================================
# Phase 2 v3: Bias initialization fix + enhanced transformer training on A100
# ============================================================================
# PROBLEM: Three of four 2D models (unet_2d, swin_2d, vit_2d) collapsed to
# zero dice in Phase 2 v2 due to narrow initial logit ranges + BCE on sparse
# targets. vitnet_3d has the same issue (logits ≈ ±0.001 at init).
#
# FIX 1: --bias_init -3.0
#   Sets final conv bias to -3.0 (sigmoid(-3)≈0.047). RetinaNet-style prior
#   probability initialization (Lin et al., 2017). Applied to all relaunched
#   models to prevent BCE-driven collapse to all-background predictions.
#
# FIX 2: Enhanced transformer training on A100 80GB
#   ViTs are data-hungry and benefit from stronger regularization
#   (Dosovitskiy et al. 2020, Touvron et al. 2020 "DeiT"):
#     - Larger batch size (16): 2-4× baseline → more diverse gradients per step
#     - Weight decay 0.05: Standard DeiT regularizer for transformers
#     - LR scaled linearly with batch size
#     - A100 Ampere avoids L40S Ada cudaErrorInvalidConfiguration bug
#       that forced ViT 2D to batch=4
#
# SwinUNETR 3D is NOT included — it has wide init logits [-4.3, +4.7] (no
# collapse risk) and is already running on A100 at batch=8 (job 33902468).
# At 62M params with 96³ crops, batch=16 would risk OOM even on 80GB.
#
# Naming convention:
#   p2v3_<model>          = L40S, bias_init only (same config as v2 otherwise)
#   p2v3_<model>_enhanced = A100, bias_init + larger batch + weight_decay
#
# Usage:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_phase2v3_biasinit.sh
#   bash training/slurm/launch_phase2v3_biasinit.sh --l40-only
#   bash training/slurm/launch_phase2v3_biasinit.sh --a100-only
# ============================================================================

set -euo pipefail
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

LOSS="dice_bce"
BIAS_INIT="--bias_init -3.0"

# Parse args
LAUNCH_L40=true
LAUNCH_A100=true
for arg in "$@"; do
    case $arg in
        --l40-only)  LAUNCH_A100=false ;;
        --a100-only) LAUNCH_L40=false ;;
    esac
done

echo "============================================"
echo "Phase 2 v3: bias_init fix + enhanced transformer training"
echo "Recipe: ${LOSS} + EMA(0.999) + fg_mask + bias_init=-3.0"
echo "============================================"
echo ""

# ============================================================================
# 2D MODELS on L40S — bias_init fix only, same batch as v2
# Only the 3 broken models (unet, swin, vit). ResNet already works in v2
# (dice=0.284) and doesn't need bias_init (wide init logits [-2, +2]).
# ============================================================================
if $LAUNCH_L40; then
    echo "--- 2D Models on L40S (bias_init fix) ---"

    for spec in \
        "unet_2d:8" \
        "swin_2d:8" \
        "vit_2d:4"; do

        IFS=':' read -r MODEL BATCH <<< "$spec"
        JOB="p2v3_${MODEL}"
        echo "  Submitting ${JOB} (model=${MODEL}, batch=${BATCH})..."

        EXPERIMENT_NAME="${JOB}" \
        MODEL_NAME="${MODEL}" \
        LOSS_NAME="${LOSS}" \
        USE_FG_MASK="true" \
        BATCH_SIZE="${BATCH}" \
        EPOCHS=100 \
        ITERS=1000 \
        EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT}" \
        sbatch --job-name="${JOB}" training/slurm/phase2_2d_l40s.sbatch
    done

    echo ""
    echo "L40S 2D jobs submitted (3 total)."
    echo ""
fi

# ============================================================================
# ENHANCED TRANSFORMERS on A100 80GB
# bias_init + larger batch + weight decay (DeiT-style regularization)
#   - 2D ViT:   batch 4→16 (4×), wd=0.05, lr scaled 1e-4→2e-4
#   - 2D Swin:  batch 8→16 (2×), wd=0.05, lr scaled 1e-4→2e-4
#   - 3D ViT:   batch 8→16 (2×), wd=0.05, lr scaled 4e-4→8e-4
# A100 node g180701: 8× A100 SXM4 80GB. Currently 3/8 GPUs in use
# (2× debman, 1× p2_swinunetr_3d). 5 GPUs available.
# ============================================================================
if $LAUNCH_A100; then
    echo "--- Enhanced transformers on A100 80GB (bias_init + batch=16 + wd=0.05) ---"

    # 2D transformers: batch=16, wd=0.05
    for spec in \
        "vit_2d:16" \
        "swin_2d:16"; do

        IFS=':' read -r MODEL BATCH <<< "$spec"
        JOB="p2v3_${MODEL}_enhanced"
        echo "  Submitting ${JOB} (model=${MODEL}, batch=${BATCH}, wd=0.05)..."

        EXPERIMENT_NAME="${JOB}" \
        MODEL_NAME="${MODEL}" \
        LOSS_NAME="${LOSS}" \
        USE_FG_MASK="true" \
        BATCH_SIZE="${BATCH}" \
        EPOCHS=100 \
        ITERS=1000 \
        EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT} --weight_decay 0.05" \
        sbatch --job-name="${JOB}" training/slurm/phase2v3_2d_a100.sbatch
    done

    # 3D ViT: batch=16, wd=0.05, 96^3 crops
    # vitnet_3d has init collapse (logits ≈ ±0.001) AND benefits from
    # transformer-specific regularization. A100 80GB accommodates batch=16.
    # LR scaled linearly: base 1e-4 @ batch=2 → 8e-4 @ batch=16.
    JOB="p2v3_vitnet_3d_enhanced"
    echo "  Submitting ${JOB} (vitnet_3d, batch=16, wd=0.05)..."

    EXPERIMENT_NAME="${JOB}" \
    MODEL_NAME="vitnet_3d" \
    LOSS_NAME="${LOSS}" \
    USE_FG_MASK="true" \
    BATCH_SIZE=16 \
    EPOCHS=1000 \
    ITERS=300 \
    EXTRA_ARGS="--ema --ema_decay 0.999 ${BIAS_INIT} --weight_decay 0.05" \
    sbatch --job-name="${JOB}" training/slurm/phase2_3d_a100.sbatch

    echo ""
    echo "A100 enhanced jobs submitted (3 total)."
    echo ""
fi

echo "============================================"
echo "Summary of v3 jobs (existing v2 jobs NOT affected):"
echo ""
echo "  L40S — bias_init fix (3 broken 2D models):"
echo "    p2v3_unet_2d              batch=8   (was zero dice in v2)"
echo "    p2v3_swin_2d              batch=8   (was zero dice in v2)"
echo "    p2v3_vit_2d               batch=4   (was zero dice in v2)"
echo ""
echo "  A100 — enhanced transformer training (bias_init + batch=16 + wd=0.05):"
echo "    p2v3_vit_2d_enhanced      batch=16  (4× v2 batch)"
echo "    p2v3_swin_2d_enhanced     batch=16  (2× v2 batch)"
echo "    p2v3_vitnet_3d_enhanced   batch=16  (2× v2 batch, init collapse fix)"
echo ""
echo "Monitor: ssh longleaf.unc.edu 'squeue -u gsgeorge'"
echo "============================================"
