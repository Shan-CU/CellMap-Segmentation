#!/bin/bash
# ============================================================================
# Launch Phase 2: Architecture Comparison on L40S (Longleaf)
#
# Uses the WINNING loss/masking from Phase 1 ablations.
# Edit BEST_LOSS and BEST_FG_MASK below based on Phase 1 results.
#
# Usage:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_arch_comparison.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ======== EDIT THESE BASED ON PHASE 1 RESULTS ========
BEST_LOSS="balanced_softmax_tversky"  # UPDATE after Phase 1
BEST_FG_MASK="true"                   # UPDATE after Phase 1
BEST_LOSS_KWARGS="{}"                 # UPDATE if needed
EPOCHS=100
# =====================================================

mkdir -p runs/arch_comparison/logs

submit_2d() {
    local model="$1"
    local name="arch_2d_${model}"

    echo "Submitting 2D: ${name}"
    sbatch \
        --export=ALL,EXPERIMENT_NAME="${name}",MODEL_NAME="${model}_2d",LOSS_NAME="${BEST_LOSS}",USE_FG_MASK="${BEST_FG_MASK}",LOSS_KWARGS="${BEST_LOSS_KWARGS}",EPOCHS="${EPOCHS}",ITERS=1000 \
        --job-name="arch_${name}" \
        --output="runs/arch_comparison/logs/%x_%j.out" \
        --error="runs/arch_comparison/logs/%x_%j.err" \
        "${SCRIPT_DIR}/ablation_2d.sbatch"
}

submit_3d() {
    local model="$1"
    local name="arch_3d_${model}"

    echo "Submitting 3D: ${name}"
    sbatch \
        --export=ALL,EXPERIMENT_NAME="${name}",MODEL_NAME="${model}_3d",LOSS_NAME="${BEST_LOSS}",USE_FG_MASK="${BEST_FG_MASK}",LOSS_KWARGS="${BEST_LOSS_KWARGS}",EPOCHS="${EPOCHS}",ITERS=500 \
        --job-name="arch_${name}" \
        --output="runs/arch_comparison/logs/%x_%j.out" \
        --error="runs/arch_comparison/logs/%x_%j.err" \
        "${SCRIPT_DIR}/ablation_3d.sbatch"
}

echo "============================================"
echo "Phase 2: Architecture Comparison"
echo "Best loss: ${BEST_LOSS}"
echo "Foreground mask: ${BEST_FG_MASK}"
echo "Epochs: ${EPOCHS}"
echo "============================================"

echo ""
echo "--- 2D Models ---"
submit_2d resnet
submit_2d unet
submit_2d swin
submit_2d vit

echo ""
echo "--- 3D Models ---"
submit_3d segresnet
submit_3d swinunetr
submit_3d unet
submit_3d resnet

echo ""
echo "============================================"
echo "All architecture comparison jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
