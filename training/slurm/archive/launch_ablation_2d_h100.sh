#!/bin/bash
# ============================================================================
# Launch ALL Phase 1 2D ablation experiments on Sycamore H100
#
# Each experiment runs as a separate SLURM job (1× H100 each).
# Total: 29 experiments × ~45min each ≈ 22 GPU-hours
#
# Usage:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_ablation_2d_h100.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/ablation_2d_h100.sbatch"

mkdir -p runs/ablation/logs

submit() {
    local name="$1"
    local model="$2"
    local loss="$3"
    local fg_mask="${4:-true}"
    local loss_kwargs="${5:-}"
    local extra_args="${6:-}"

    echo "Submitting: ${name} (model=${model}, loss=${loss}, fg=${fg_mask})"
    EXPERIMENT_NAME="${name}" \
    MODEL_NAME="${model}" \
    LOSS_NAME="${loss}" \
    USE_FG_MASK="${fg_mask}" \
    LOSS_KWARGS="${loss_kwargs}" \
    EXTRA_ARGS="${extra_args}" \
    sbatch \
        --export=ALL \
        --job-name="abl_${name}" \
        "${SBATCH_FILE}"
}

echo "============================================"
echo "Phase 1A: Loss Function Sweep (2D) — H100"
echo "============================================"
submit "loss_2d_bce"                        resnet_2d bce
submit "loss_2d_focal"                      resnet_2d focal
submit "loss_2d_dice_bce"                   resnet_2d dice_bce
submit "loss_2d_tversky"                    resnet_2d tversky
submit "loss_2d_balanced_softmax_tversky"   resnet_2d balanced_softmax_tversky
submit "loss_2d_focal_tversky"              resnet_2d focal_tversky
submit "loss_2d_unified_focal"              resnet_2d unified_focal
submit "loss_2d_boundary_tversky"           resnet_2d boundary_tversky

echo ""
echo "============================================"
echo "Phase 1B: Tversky α/β Sweep (2D) — H100"
echo "============================================"
submit "tversky_2d_balanced"                resnet_2d tversky_balanced
submit "tversky_2d_precision_06_04"         resnet_2d tversky
submit "tversky_2d_precision_07_03"         resnet_2d tversky_precision
submit "tversky_2d_recall"                  resnet_2d tversky_recall
submit "tversky_2d_a08_b04"                 resnet_2d tversky_a08_b04
submit "tversky_2d_a08_b06"                 resnet_2d tversky_a08_b06

echo ""
echo "============================================"
echo "Phase 1C: Class Weighting (τ) Sweep (2D) — H100"
echo "============================================"
submit "tau_2d_0"                           resnet_2d bst_tau0
submit "tau_2d_05"                          resnet_2d bst_tau05
submit "tau_2d_10"                          resnet_2d balanced_softmax_tversky
submit "tau_2d_15"                          resnet_2d bst_tau15
submit "tau_2d_20"                          resnet_2d bst_tau20

echo ""
echo "============================================"
echo "Phase 1D: Masking Strategy Sweep (2D) — H100"
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
echo "Phase 1E: Training Technique Sweep (2D) — H100"
echo "============================================"
submit "tech_2d_ema"                        resnet_2d balanced_softmax_tversky  true "" "--ema --ema_decay 0.999"
submit "tech_2d_no_weighted_sampler"        resnet_2d balanced_softmax_tversky  true "" "--no_weighted_sampler"
submit "tech_2d_focal_tversky_mild"         resnet_2d focal_tversky_g05

echo ""
echo "============================================"
echo "All 29 2D ablation jobs submitted to H100!"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
