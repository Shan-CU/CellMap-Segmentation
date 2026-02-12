#!/bin/bash
# ============================================================
# Launch All 8 L40S Model Comparison Jobs
# ============================================================
# Submits all 8 models (UNet/ResNet/Swin/ViT × 2D/3D) to the
# Longleaf L40S partition.
#
# Usage:
#   cd CellMap-Segmentation
#   bash experiments/l40s_comparison/slurm/launch_all.sh
#
# Or submit selectively:
#   bash experiments/l40s_comparison/slurm/launch_all.sh 2d     # 2D models only
#   bash experiments/l40s_comparison/slurm/launch_all.sh 3d     # 3D models only
#   bash experiments/l40s_comparison/slurm/launch_all.sh unet   # UNet only
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FILTER="${1:-all}"

echo "=============================================="
echo "L40S Model Comparison — Job Launcher"
echo "=============================================="
echo "Filter: $FILTER"
echo "Script dir: $SCRIPT_DIR"
echo "=============================================="

# Create log directory
mkdir -p "$SCRIPT_DIR/logs"

declare -a JOBS_2D=(
    "train_unet_2d.sbatch"
    "train_resnet_2d.sbatch"
    "train_swin_2d.sbatch"
    "train_vit_2d.sbatch"
)

declare -a JOBS_3D=(
    "train_unet_3d.sbatch"
    "train_resnet_3d.sbatch"
    "train_swin_3d.sbatch"
    "train_vit_3d.sbatch"
)

submit_job() {
    local script="$1"
    local name=$(basename "$script" .sbatch)
    echo -n "  Submitting $name... "
    JOB_ID=$(sbatch "$SCRIPT_DIR/$script" 2>&1 | grep -oP '\d+')
    if [ -n "$JOB_ID" ]; then
        echo "Job $JOB_ID"
    else
        echo "FAILED"
        return 1
    fi
}

N_SUBMITTED=0

# Submit 2D models
if [[ "$FILTER" == "all" || "$FILTER" == "2d" || "$FILTER" == *"unet"* || "$FILTER" == *"resnet"* || "$FILTER" == *"swin"* || "$FILTER" == *"vit"* ]]; then
    echo ""
    echo "--- 2D Models ---"
    for script in "${JOBS_2D[@]}"; do
        model=$(echo "$script" | sed 's/train_\(.*\)_2d.sbatch/\1/')
        if [[ "$FILTER" == "all" || "$FILTER" == "2d" || "$FILTER" == *"$model"* ]]; then
            submit_job "$script" && ((N_SUBMITTED++))
            sleep 1  # Brief delay between submissions
        fi
    done
fi

# Submit 3D models
if [[ "$FILTER" == "all" || "$FILTER" == "3d" || "$FILTER" == *"unet"* || "$FILTER" == *"resnet"* || "$FILTER" == *"swin"* || "$FILTER" == *"vit"* ]]; then
    echo ""
    echo "--- 3D Models ---"
    for script in "${JOBS_3D[@]}"; do
        model=$(echo "$script" | sed 's/train_\(.*\)_3d.sbatch/\1/')
        if [[ "$FILTER" == "all" || "$FILTER" == "3d" || "$FILTER" == *"$model"* ]]; then
            submit_job "$script" && ((N_SUBMITTED++))
            sleep 1
        fi
    done
fi

echo ""
echo "=============================================="
echo "Submitted $N_SUBMITTED jobs"
echo "Check status: squeue -u \$USER"
echo "=============================================="
