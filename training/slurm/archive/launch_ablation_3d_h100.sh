#!/bin/bash
# ============================================================================
# Launch ALL Phase 1 3D ablation experiments on Sycamore H100
#
# Each experiment runs as a separate SLURM job (1× H100 each).
# Total: 30 experiments × ~90min each ≈ 45 GPU-hours
#
# Usage:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/launch_ablation_3d_h100.sh
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_FILE="${SCRIPT_DIR}/ablation_3d_h100.sbatch"

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
echo "Phase 1A: Loss Function Sweep (3D) — H100"
echo "============================================"
submit "loss_3d_bce"                        segresnet_3d bce
submit "loss_3d_focal"                      segresnet_3d focal
submit "loss_3d_dice_bce"                   segresnet_3d dice_bce
submit "loss_3d_tversky"                    segresnet_3d tversky
submit "loss_3d_balanced_softmax_tversky"   segresnet_3d balanced_softmax_tversky
submit "loss_3d_focal_tversky"              segresnet_3d focal_tversky
submit "loss_3d_unified_focal"              segresnet_3d unified_focal
submit "loss_3d_boundary_tversky"           segresnet_3d boundary_tversky

echo ""
echo "============================================"
echo "Phase 1B: Tversky α/β Sweep (3D) — H100"
echo "============================================"
submit "tversky_3d_balanced"                segresnet_3d tversky_balanced
submit "tversky_3d_precision_06_04"         segresnet_3d tversky
submit "tversky_3d_precision_07_03"         segresnet_3d tversky_precision
submit "tversky_3d_recall"                  segresnet_3d tversky_recall
submit "tversky_3d_a08_b04"                 segresnet_3d tversky_a08_b04
submit "tversky_3d_a08_b06"                 segresnet_3d tversky_a08_b06

echo ""
echo "============================================"
echo "Phase 1C: Class Weighting (τ) Sweep (3D) — H100"
echo "============================================"
submit "tau_3d_0"                           segresnet_3d bst_tau0
submit "tau_3d_05"                          segresnet_3d bst_tau05
submit "tau_3d_10"                          segresnet_3d balanced_softmax_tversky
submit "tau_3d_15"                          segresnet_3d bst_tau15
submit "tau_3d_20"                          segresnet_3d bst_tau20

echo ""
echo "============================================"
echo "Phase 1D: Masking Strategy Sweep (3D) — H100"
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
echo "Phase 1E: Training Technique Sweep (3D) — H100"
echo "============================================"
submit "tech_3d_ema"                        segresnet_3d balanced_softmax_tversky  true "" "--ema --ema_decay 0.999"
submit "tech_3d_no_weighted_sampler"        segresnet_3d balanced_softmax_tversky  true "" "--no_weighted_sampler"
submit "tech_3d_focal_tversky_mild"         segresnet_3d focal_tversky_g05
submit "tech_3d_deep_supervision"           segresnet_3d balanced_softmax_tversky  true "" "--deep_supervision --model_kwargs '{\"dsdepth\":4}'"

echo ""
echo "============================================"
echo "All 30 3D ablation jobs submitted to H100!"
echo "Monitor with: squeue -u \$USER"
echo "============================================"
