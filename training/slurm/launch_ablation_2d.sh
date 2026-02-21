#!/bin/bash
# ============================================================================
# Launch ALL Phase 1 2D ablation experiments on L40S (Longleaf)
#
# Each experiment runs as a separate SLURM job (1 GPU each).
# Total: 20 experiments × ~1.5h each ≈ 30 GPU-hours
#
# Usage:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_ablation_2d.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/ablation_2d.sbatch"

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
echo "Phase 1A: Loss Function Sweep (2D)"
echo "============================================"
submit "loss_2d_bce"                        resnet_2d bce
submit "loss_2d_focal"                      resnet_2d focal
submit "loss_2d_dice_bce"                   resnet_2d dice_bce
submit "loss_2d_tversky"                    resnet_2d tversky
submit "loss_2d_balanced_softmax_tversky"   resnet_2d balanced_softmax_tversky

echo ""
echo "============================================"
echo "Phase 1B: Tversky α/β Sweep (2D)"
echo "============================================"
submit "tversky_2d_balanced"                resnet_2d tversky_balanced
submit "tversky_2d_precision_06_04"         resnet_2d tversky            true '{"alpha": 0.6, "beta": 0.4}'
submit "tversky_2d_precision_07_03"         resnet_2d tversky_precision
submit "tversky_2d_recall"                  resnet_2d tversky_recall

echo ""
echo "============================================"
echo "Phase 1C: Class Weighting (τ) Sweep (2D)"
echo "============================================"
submit "tau_2d_0"                           resnet_2d bst_tau0
submit "tau_2d_05"                          resnet_2d bst_tau05
submit "tau_2d_10"                          resnet_2d balanced_softmax_tversky
submit "tau_2d_15"                          resnet_2d bst_tau15

echo ""
echo "============================================"
echo "Phase 1D: Masking Strategy Sweep (2D)"
echo "============================================"
submit "mask_2d_none"                       resnet_2d bst_no_bbox       false
submit "mask_2d_fg_only"                    resnet_2d bst_no_bbox       true
submit "mask_2d_bbox_only"                  resnet_2d balanced_softmax_tversky  false
submit "mask_2d_bbox_fg"                    resnet_2d balanced_softmax_tversky  true
submit "mask_2d_bbox_loose"                 resnet_2d bst_bbox_loose    true
submit "mask_2d_masksup03"                  resnet_2d bst_masksup03     true
submit "mask_2d_masksup03_no_bbox"          resnet_2d bst_masksup03_no_bbox true

echo ""
echo "============================================"
echo "All 2D ablation jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
