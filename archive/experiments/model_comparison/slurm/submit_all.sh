#!/bin/bash
# ============================================================
# Submit All Model Comparison Jobs to blanca-biokem
# ============================================================
# This script submits all 8 model training jobs as separate SLURM jobs
# Each job uses 2x A100 80GB GPUs with 2-day time allocation
#
# Usage: ./submit_all.sh [2d|3d|all]
#   2d  - Submit only 2D models
#   3d  - Submit only 3D models
#   all - Submit all models (default)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "CellMap Model Comparison - Job Submission"
echo "=============================================="
echo "Cluster: blanca-biokem (2x A100 80GB per job)"
echo "Time allocation: 2 days per job"
echo "=============================================="
echo ""

MODE="${1:-all}"

submit_2d_jobs() {
    echo "Submitting 2D model jobs..."
    echo "----------------------------"
    
    echo -n "  UNet 2D:   "
    JOB1=$(sbatch train_unet_2d.sbatch | awk '{print $4}')
    echo "Job ID $JOB1"
    
    echo -n "  ResNet 2D: "
    JOB2=$(sbatch train_resnet_2d.sbatch | awk '{print $4}')
    echo "Job ID $JOB2"
    
    echo -n "  Swin 2D:   "
    JOB3=$(sbatch train_swin_2d.sbatch | awk '{print $4}')
    echo "Job ID $JOB3"
    
    echo -n "  ViT 2D:    "
    JOB4=$(sbatch train_vit_2d.sbatch | awk '{print $4}')
    echo "Job ID $JOB4"
    
    echo ""
    echo "2D Jobs submitted: $JOB1, $JOB2, $JOB3, $JOB4"
}

submit_3d_jobs() {
    echo "Submitting 3D model jobs..."
    echo "----------------------------"
    
    echo -n "  UNet 3D:   "
    JOB5=$(sbatch train_unet_3d.sbatch | awk '{print $4}')
    echo "Job ID $JOB5"
    
    echo -n "  ResNet 3D: "
    JOB6=$(sbatch train_resnet_3d.sbatch | awk '{print $4}')
    echo "Job ID $JOB6"
    
    echo -n "  Swin 3D:   "
    JOB7=$(sbatch train_swin_3d.sbatch | awk '{print $4}')
    echo "Job ID $JOB7"
    
    echo -n "  ViT 3D:    "
    JOB8=$(sbatch train_vit_3d.sbatch | awk '{print $4}')
    echo "Job ID $JOB8"
    
    echo ""
    echo "3D Jobs submitted: $JOB5, $JOB6, $JOB7, $JOB8"
}

case "$MODE" in
    2d)
        submit_2d_jobs
        ;;
    3d)
        submit_3d_jobs
        ;;
    all)
        submit_2d_jobs
        echo ""
        submit_3d_jobs
        ;;
    *)
        echo "Usage: $0 [2d|3d|all]"
        exit 1
        ;;
esac

echo ""
echo "=============================================="
echo "All jobs submitted successfully!"
echo "=============================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "Cancel all jobs with:"
echo "  scancel -u \$USER"
echo ""
echo "View logs in:"
echo "  $SCRIPT_DIR/logs/"
echo ""
echo "TensorBoard (after jobs start):"
echo "  tensorboard --logdir=../tensorboard"
echo "=============================================="
