#!/bin/bash

# ============================================================
# BLANCA BIOKEM COMPLETE MODEL COMPARISON LAUNCHER
# Launches all 8 models (4 architectures × 2D/3D) simultaneously
# Resource allocation optimized for blanca-biokem nodes
# ============================================================

set -e  # Exit on any error

echo "=========================================="
echo "BLANCA BIOKEM - 8 MODEL LAUNCHER"
echo "Date: $(date)"
echo "=========================================="

# Ensure we're in the correct directory
cd "$(dirname "$0")/.."
WORKDIR=$(pwd)
echo "Working directory: $WORKDIR"

# Load Blanca Slurm
echo "Loading slurm/blanca module..."
module purge
module load slurm/blanca

# Create logs directory
mkdir -p logs experiments/model_comparison/logs

echo ""
echo "=========================================="
echo "SUBMITTING 8 JOBS TO BLANCA-BIOKEM"
echo "=========================================="
echo ""
echo "2D Models (4 jobs):"
echo "===================="

# 2D Jobs - Use RTX 6000 nodes (bgpu-biokem1 & bgpu-biokem3)

# Job 1: ResNet 2D
echo -n "1. ResNet 2D (2x A100, 32 CPUs, 320GB)... "
RESNET_2D_JOB=$(sbatch --parsable experiments/model_comparison/slurm/train_resnet_2d.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $RESNET_2D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

# Job 2: UNet 2D
echo -n "2. UNet 2D (2x RTX 6000, 32 CPUs, 200GB)... "
UNET_2D_JOB=$(sbatch --parsable slurm/train_unet_blanca.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $UNET_2D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

# Job 3: Swin 2D
echo -n "3. Swin 2D (2x RTX 6000, 32 CPUs, 200GB)... "
SWIN_2D_JOB=$(sbatch --parsable slurm/train_swin_blanca.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $SWIN_2D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

# Job 4: ViT 2D
echo -n "4. ViT 2D (2x A100, 32 CPUs, 384GB)... "
VIT_2D_JOB=$(sbatch --parsable experiments/model_comparison/slurm/train_vit_2d.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $VIT_2D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

echo ""
echo "3D Models (4 jobs):"
echo "===================="

# 3D Jobs - Mix of A100 and RTX 6000 based on memory needs

# Job 5: ResNet 3D
echo -n "5. ResNet 3D (2x A100, 32 CPUs, 400GB)... "
RESNET_3D_JOB=$(sbatch --parsable experiments/model_comparison/slurm/train_resnet_3d.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $RESNET_3D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

# Job 6: UNet 3D
echo -n "6. UNet 3D (2x A100, 32 CPUs, 400GB)... "
UNET_3D_JOB=$(sbatch --parsable slurm/train_unet3d_blanca.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $UNET_3D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

# Job 7: Swin 3D
echo -n "7. Swin 3D (2x A100, 32 CPUs, 480GB)... "
SWIN_3D_JOB=$(sbatch --parsable experiments/model_comparison/slurm/train_swin_3d.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $SWIN_3D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

# Job 8: ViT 3D
echo -n "8. ViT 3D (2x A100, 40 CPUs, 400GB)... "
VIT_3D_JOB=$(sbatch --parsable slurm/train_vit3d_blanca.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $VIT_3D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "ALL 8 JOBS SUBMITTED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Job Summary:"
echo "============"
echo "2D Models:"
echo "  ResNet 2D: $RESNET_2D_JOB (2x A100)"
echo "  UNet 2D:   $UNET_2D_JOB (2x RTX 6000)"
echo "  Swin 2D:   $SWIN_2D_JOB (2x RTX 6000)"
echo "  ViT 2D:    $VIT_2D_JOB (2x A100)"
echo ""
echo "3D Models:"
echo "  ResNet 3D: $RESNET_3D_JOB (2x A100)"
echo "  UNet 3D:   $UNET_3D_JOB (2x A100)"
echo "  Swin 3D:   $SWIN_3D_JOB (2x A100)"
echo "  ViT 3D:    $VIT_3D_JOB (2x A100)"
echo ""
echo "Total Resources Requested:"
echo "  - A100 GPUs: 12 (6 jobs × 2 GPUs)"
echo "  - RTX 6000 GPUs: 4 (2 jobs × 2 GPUs)"
echo "  - Total GPUs: 16 across 8 jobs"
echo "  - Total CPUs: 264"
echo "  - Total RAM: ~2.7TB"
echo "  - Walltime: 3-7 days per job"
echo ""
echo "=========================================="
echo "MONITORING COMMANDS"
echo "=========================================="
echo ""
echo "Check all jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Watch jobs:"
echo "  watch -n 10 'squeue -u \$USER'"
echo ""
echo "Cancel all 8 jobs:"
echo "  scancel $RESNET_2D_JOB $UNET_2D_JOB $SWIN_2D_JOB $VIT_2D_JOB $RESNET_3D_JOB $UNET_3D_JOB $SWIN_3D_JOB $VIT_3D_JOB"
echo ""
echo "View logs:"
echo "  tail -f logs/*_blanca_*.out"
echo "  tail -f experiments/model_comparison/logs/*_*.out"
echo ""
echo "=========================================="

# Save job IDs to file for later reference
cat > logs/blanca_8job_ids.txt << EOF
ResNet_2D=$RESNET_2D_JOB
UNet_2D=$UNET_2D_JOB
Swin_2D=$SWIN_2D_JOB
ViT_2D=$VIT_2D_JOB
ResNet_3D=$RESNET_3D_JOB
UNet_3D=$UNET_3D_JOB
Swin_3D=$SWIN_3D_JOB
ViT_3D=$VIT_3D_JOB
Submitted=$(date)
EOF

echo "Job IDs saved to: logs/blanca_8job_ids.txt"
echo ""
echo "Happy training! 🚀🚀🚀"
