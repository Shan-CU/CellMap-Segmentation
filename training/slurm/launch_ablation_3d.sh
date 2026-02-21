#!/bin/bash
# ============================================================================
# Launch ALL Phase 1 3D ablation experiments on L40S (Longleaf)
#
# Each experiment runs as a separate SLURM job (1 GPU each).
# Total: 20 experiments × ~3h each ≈ 60 GPU-hours
#
# Usage:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_ablation_3d.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/ablation_3d.sbatch"

mkdir -p runs/ablation/logs

submit() {
    local name="$1"
    local model="$2"
    local loss="$3"
    local fg_mask="${4:-true}"
    local loss_kwargs="${5:-{}}"

    echo "Submitting: ${name} (model=${model}, loss=${loss}, fg=${fg_mask})"
    sbatch \
        --export=ALL,EXPERIMENT_NAME="${name}",MODEL_NAME="${model}",LOSS_NAME="${loss}",USE_FG_MASK="${fg_mask}",LOSS_KWARGS="${loss_kwargs}" \
        --job-name="abl_${name}" \
        "${SBATCH_FILE}"
}

echo "============================================"
echo "Phase 1A: Loss Function Sweep (3D)"
echo "============================================"
submit "loss_3d_bce"                        segresnet_3d bce
submit "loss_3d_focal"                      segresnet_3d focal
submit "loss_3d_dice_bce"                   segresnet_3d dice_bce
submit "loss_3d_tversky"                    segresnet_3d tversky
submit "loss_3d_balanced_softmax_tversky"   segresnet_3d balanced_softmax_tversky

echo ""
echo "============================================"
echo "Phase 1B: Tversky α/β Sweep (3D)"
echo "============================================"
submit "tversky_3d_balanced"                segresnet_3d tversky_balanced
submit "tversky_3d_precision_06_04"         segresnet_3d tversky         true '{"alpha": 0.6, "beta": 0.4}'
submit "tversky_3d_precision_07_03"         segresnet_3d tversky_precision
submit "tversky_3d_recall"                  segresnet_3d tversky_recall

echo ""
echo "============================================"
echo "Phase 1C: Class Weighting (τ) Sweep (3D)"
echo "============================================"
submit "tau_3d_0"                           segresnet_3d bst_tau0
submit "tau_3d_05"                          segresnet_3d bst_tau05
submit "tau_3d_10"                          segresnet_3d balanced_softmax_tversky
submit "tau_3d_15"                          segresnet_3d bst_tau15

echo ""
echo "============================================"
echo "Phase 1D: Masking Strategy Sweep (3D)"
echo "============================================"
submit "mask_3d_none"                       segresnet_3d bst_no_bbox     false
submit "mask_3d_fg_only"                    segresnet_3d bst_no_bbox     true
submit "mask_3d_bbox_only"                  segresnet_3d balanced_softmax_tversky false
submit "mask_3d_bbox_fg"                    segresnet_3d balanced_softmax_tversky true
submit "mask_3d_bbox_loose"                 segresnet_3d bst_bbox_loose  true
submit "mask_3d_masksup03"                  segresnet_3d bst_masksup03   true
submit "mask_3d_masksup03_no_bbox"          segresnet_3d bst_masksup03_no_bbox true

echo ""
echo "============================================"
echo "All 3D ablation jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
