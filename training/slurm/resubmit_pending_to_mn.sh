#!/bin/bash
# ============================================================================
# Resubmit pending jobs to h100_mn partition for faster throughput
#
# Strategy: Move the heaviest 3D pending jobs to h100_mn (7 free GPUs,
# 1.5TB RAM/node). Also move the 4 pending 2D jobs since they're light.
#
# This runs alongside the 8 jobs already running on h100_sn.
# ============================================================================

set -euo pipefail

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

SBATCH_3D="training/slurm/ablation_3d_h100_mn.sbatch"
SBATCH_2D="training/slurm/ablation_2d_h100_mn.sbatch"

submit_3d() {
    local name="$1"
    local model="$2"
    local loss="$3"
    local fg_mask="${4:-true}"
    local loss_kwargs="${5:-}"
    local extra_args="${6:-}"

    echo "Submitting 3D to h100_mn: ${name} (model=${model}, loss=${loss})"
    EXPERIMENT_NAME="${name}" \
    MODEL_NAME="${model}" \
    LOSS_NAME="${loss}" \
    USE_FG_MASK="${fg_mask}" \
    LOSS_KWARGS="${loss_kwargs}" \
    EXTRA_ARGS="${extra_args}" \
    sbatch \
        --export=ALL \
        --job-name="abl_${name}" \
        "${SBATCH_3D}"
}

submit_2d() {
    local name="$1"
    local model="$2"
    local loss="$3"
    local fg_mask="${4:-true}"
    local loss_kwargs="${5:-}"
    local extra_args="${6:-}"

    echo "Submitting 2D to h100_mn: ${name} (model=${model}, loss=${loss})"
    EXPERIMENT_NAME="${name}" \
    MODEL_NAME="${model}" \
    LOSS_NAME="${loss}" \
    USE_FG_MASK="${fg_mask}" \
    LOSS_KWARGS="${loss_kwargs}" \
    EXTRA_ARGS="${extra_args}" \
    sbatch \
        --export=ALL \
        --job-name="abl_${name}" \
        "${SBATCH_2D}"
}

# ============================================================================
# Step 1: Cancel the pending jobs we're moving to h100_mn
# ============================================================================
echo "Cancelling pending h100_sn jobs that will be resubmitted to h100_mn..."

# Heaviest 3D jobs (deep supervision, masksup, EMA, focal variants)
CANCEL_JOBS=(
    # Sweep E - Training techniques (heaviest/most complex)
    1793617   # tech_3d_deep_supervision
    1793614   # tech_3d_ema
    1793615   # tech_3d_no_weighted_sampler
    1793670   # tech_3d_focal_tversky_mild (fixed relaunch)
    # Sweep D - Masking strategy (complex loss computations)
    1793612   # mask_3d_masksup03
    1793613   # mask_3d_masksup03_no_bbox
    1793611   # mask_3d_bbox_loose
    # Sweep C - Tau sweep (BST variants)
    1793606   # tau_3d_20
    1793605   # tau_3d_15
    1793604   # tau_3d_10
    1793603   # tau_3d_05
    1793602   # tau_3d_0
    # Sweep D continued
    1793610   # mask_3d_bbox_fg
    1793609   # mask_3d_bbox_only
    1793608   # mask_3d_fg_only
    1793607   # mask_3d_none
    # Sweep B remaining
    1793601   # tversky_3d_a08_b06
    1793600   # tversky_3d_a08_b04
    1793599   # tversky_3d_recall
    1793598   # tversky_3d_precision_07_03
    # 3D focal relaunches
    1793668   # loss_3d_focal_tversky
    1793669   # loss_3d_unified_focal
    # 2D pending jobs (light, will finish fast on mn)
    1793621   # tech_2d_no_weighted_sampler
    1793665   # loss_2d_focal_tversky
    1793666   # loss_2d_unified_focal
    1793667   # tech_2d_focal_tversky_mild
)

for jid in "${CANCEL_JOBS[@]}"; do
    scancel "$jid" 2>/dev/null && echo "  Cancelled $jid" || echo "  Could not cancel $jid (may have already started)"
done

echo ""
sleep 2  # Give SLURM a moment to process cancellations

# ============================================================================
# Step 2: Resubmit ALL pending jobs to h100_mn
# SLURM will queue them and dispatch as GPUs become available
# ============================================================================

echo "============================================"
echo "Submitting 3D jobs to h100_mn"
echo "============================================"

# Sweep B remaining: Tversky α/β (3D)
submit_3d "tversky_3d_precision_07_03"      segresnet_3d tversky_precision
submit_3d "tversky_3d_recall"               segresnet_3d tversky_recall
submit_3d "tversky_3d_a08_b04"              segresnet_3d tversky_a08_b04
submit_3d "tversky_3d_a08_b06"              segresnet_3d tversky_a08_b06

# Sweep C: Tau sweep (3D)
submit_3d "tau_3d_0"                        segresnet_3d bst_tau0
submit_3d "tau_3d_05"                       segresnet_3d bst_tau05
submit_3d "tau_3d_10"                       segresnet_3d balanced_softmax_tversky
submit_3d "tau_3d_15"                       segresnet_3d bst_tau15
submit_3d "tau_3d_20"                       segresnet_3d bst_tau20

# Sweep D: Masking strategy (3D)
submit_3d "mask_3d_none"                    segresnet_3d bst_no_bbox     false
submit_3d "mask_3d_fg_only"                 segresnet_3d bst_no_bbox     true
submit_3d "mask_3d_bbox_only"               segresnet_3d balanced_softmax_tversky false
submit_3d "mask_3d_bbox_fg"                 segresnet_3d balanced_softmax_tversky true
submit_3d "mask_3d_bbox_loose"              segresnet_3d bst_bbox_loose  true
submit_3d "mask_3d_masksup03"               segresnet_3d bst_masksup03   true
submit_3d "mask_3d_masksup03_no_bbox"       segresnet_3d bst_masksup03_no_bbox true

# Sweep E: Training techniques (3D)
submit_3d "tech_3d_ema"                     segresnet_3d balanced_softmax_tversky  true "" "--ema --ema_decay 0.999"
submit_3d "tech_3d_no_weighted_sampler"     segresnet_3d balanced_softmax_tversky  true "" "--no_weighted_sampler"
submit_3d "tech_3d_focal_tversky_mild"      segresnet_3d focal_tversky_g05
submit_3d "tech_3d_deep_supervision"        segresnet_3d balanced_softmax_tversky  true "" "--deep_supervision --model_kwargs '{\"dsdepth\":4}'"

# 3D focal relaunches (fixed)
submit_3d "loss_3d_focal_tversky"           segresnet_3d focal_tversky
submit_3d "loss_3d_unified_focal"           segresnet_3d unified_focal

echo ""
echo "============================================"
echo "Submitting 2D jobs to h100_mn"
echo "============================================"

# 2D pending jobs
submit_2d "tech_2d_no_weighted_sampler"     resnet_2d balanced_softmax_tversky  true "" "--no_weighted_sampler"
submit_2d "loss_2d_focal_tversky"           resnet_2d focal_tversky
submit_2d "loss_2d_unified_focal"           resnet_2d unified_focal
submit_2d "tech_2d_focal_tversky_mild"      resnet_2d focal_tversky_g05

echo ""
echo "============================================"
echo "All 26 pending jobs resubmitted to h100_mn!"
echo ""
echo "h100_mn has 7 free GPUs across 2 nodes (1.5TB RAM each)"
echo "h100_sn still has 8 running jobs"
echo ""
echo "Max concurrent: ~7 on h100_mn + 8 on h100_sn = 15 GPUs!"
echo "(vs the previous 8 on h100_sn alone)"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "============================================"
