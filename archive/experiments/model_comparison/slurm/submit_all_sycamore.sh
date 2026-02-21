#!/bin/bash
# ============================================================
# Submit All Model Comparison Jobs to UNC Sycamore h100_sn
# ============================================================
# This script submits all 8 model training jobs as separate SLURM jobs
# 
# Resource Allocation:
#   2D Models: 2x H100 80GB (I/O bound, 2 GPUs sufficient)
#   3D Models: 4x H100 80GB (compute bound, benefits from parallelism)
#
# Optimal Submission Order (fastest to slowest):
#   1. UNet 2D   (batch=64, ~2-3 days) - Fastest, simple architecture
#   2. UNet 3D   (batch=8,  ~3-4 days) - Fast, 3D convolutions
#   3. ResNet 2D (batch=48, ~3 days)   - Moderate, deeper network
#   4. ResNet 3D (batch=8,  ~4 days)   - Moderate, 3D ResNet blocks
#   5. Swin 2D   (batch=32, ~3-4 days) - Slower, window attention
#   6. Swin 3D   (batch=2,  ~4-5 days) - Very slow, 3D attention
#   7. ViT 2D    (batch=16, ~3-4 days) - Slow, global self-attention
#   8. ViT 3D    (batch=2,  ~5 days)   - Slowest, 3D global attention
#
# Usage: ./submit_all_sycamore.sh [2d|3d|all]
#   2d  - Submit only 2D models (4 jobs, 2 GPUs each)
#   3d  - Submit only 3D models (4 jobs, 4 GPUs each)
#   all - Submit all models (default)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create logs directory
mkdir -p logs

echo "=============================================="
echo "CellMap Model Comparison - Sycamore Job Submission"
echo "=============================================="
echo "Cluster: UNC Sycamore (h100_sn partition)"
echo "2D Models: 2x H100 80GB, 48 CPUs, 3-day limit"
echo "3D Models: 4x H100 80GB, 64 CPUs, 4-5 day limit"
echo "=============================================="
echo ""

MODE="${1:-all}"

# Submit in optimal order: fastest first to get early results
# This helps with debugging (fast failures) and gives intermediate results sooner

submit_all_optimal() {
    echo "Submitting all jobs in optimal order (fastest to slowest)..."
    echo "============================================================"
    
    echo ""
    echo "--- Fast Jobs (simple CNNs) ---"
    
    echo -n "  1. UNet 2D   (batch=64, ~2-3 days):  "
    JOB1=$(sbatch train_unet_2d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB1"
    
    echo -n "  2. UNet 3D   (batch=8,  ~3-4 days):  "
    JOB2=$(sbatch train_unet_3d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB2"
    
    echo ""
    echo "--- Moderate Jobs (ResNets) ---"
    
    echo -n "  3. ResNet 2D (batch=48, ~3 days):    "
    JOB3=$(sbatch train_resnet_2d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB3"
    
    echo -n "  4. ResNet 3D (batch=8,  ~4 days):    "
    JOB4=$(sbatch train_resnet_3d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB4"
    
    echo ""
    echo "--- Slow Jobs (Transformers) ---"
    
    echo -n "  5. Swin 2D   (batch=32, ~3-4 days):  "
    JOB5=$(sbatch train_swin_2d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB5"
    
    echo -n "  6. Swin 3D   (batch=2,  ~4-5 days):  "
    JOB6=$(sbatch train_swin_3d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB6"
    
    echo -n "  7. ViT 2D    (batch=16, ~3-4 days):  "
    JOB7=$(sbatch train_vit_2d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB7"
    
    echo -n "  8. ViT 3D    (batch=2,  ~5 days):    "
    JOB8=$(sbatch train_vit_3d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB8"
    
    echo ""
    echo "All 8 jobs submitted: $JOB1, $JOB2, $JOB3, $JOB4, $JOB5, $JOB6, $JOB7, $JOB8"
    echo "Total GPUs requested: 24 (2D: 4x2=8, 3D: 4x4=16)"
}

submit_2d_jobs() {
    echo "Submitting 2D model jobs (2x H100 each)..."
    echo "Order: UNet -> ResNet -> Swin -> ViT (fastest to slowest)"
    echo "-------------------------------------------"
    
    echo -n "  1. UNet 2D   (batch=64, ~2-3 days):  "
    JOB1=$(sbatch train_unet_2d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB1"
    
    echo -n "  2. ResNet 2D (batch=48, ~3 days):    "
    JOB2=$(sbatch train_resnet_2d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB2"
    
    echo -n "  3. Swin 2D   (batch=32, ~3-4 days):  "
    JOB3=$(sbatch train_swin_2d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB3"
    
    echo -n "  4. ViT 2D    (batch=16, ~3-4 days):  "
    JOB4=$(sbatch train_vit_2d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB4"
    
    echo ""
    echo "2D Jobs submitted: $JOB1, $JOB2, $JOB3, $JOB4"
    echo "Total GPUs requested: 8 (2 per job)"
}

submit_3d_jobs() {
    echo "Submitting 3D model jobs (4x H100 each)..."
    echo "Order: UNet -> ResNet -> Swin -> ViT (fastest to slowest)"
    echo "-------------------------------------------"
    
    echo -n "  1. UNet 3D   (batch=8,  ~3-4 days):  "
    JOB5=$(sbatch train_unet_3d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB5"
    
    echo -n "  2. ResNet 3D (batch=8,  ~4 days):    "
    JOB6=$(sbatch train_resnet_3d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB6"
    
    echo -n "  3. Swin 3D   (batch=2,  ~4-5 days):  "
    JOB7=$(sbatch train_swin_3d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB7"
    
    echo -n "  4. ViT 3D    (batch=2,  ~5 days):    "
    JOB8=$(sbatch train_vit_3d_sycamore.sbatch | awk '{print $4}')
    echo "Job ID $JOB8"
    
    echo ""
    echo "3D Jobs submitted: $JOB5, $JOB6, $JOB7, $JOB8"
    echo "Total GPUs requested: 16 (4 per job)"
}

case "$MODE" in
    2d)
        submit_2d_jobs
        ;;
    3d)
        submit_3d_jobs
        ;;
    all)
        submit_all_optimal
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
echo ""
echo "View logs with:"
echo "  tail -f logs/*_sycamore_*.out"
echo ""
echo "Cancel all jobs with:"
echo "  scancel -u \$USER"
echo "=============================================="
