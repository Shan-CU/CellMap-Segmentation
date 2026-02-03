#!/bin/bash

# ============================================================
# BLANCA BIOKEM PARALLEL JOB LAUNCHER
# Launches all 4 baseline models simultaneously for maximum throughput
# Resource allocation optimized for blanca-biokem nodes:
#   - 2x RTX 6000 nodes (bgpu-biokem1 & bgpu-biokem3): 7 GPUs total
#   - 1x A100 node (bgpu-biokem2): 2 GPUs
# Total: 9 GPUs across 3 nodes running 4 jobs in parallel
# ============================================================

set -e  # Exit on any error

echo "=========================================="
echo "BLANCA BIOKEM JOB LAUNCHER"
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
mkdir -p logs

echo ""
echo "=========================================="
echo "SUBMITTING 4 JOBS TO BLANCA-BIOKEM"
echo "=========================================="

# Job 1: UNet 2D (2x RTX 6000)
echo -n "1. UNet 2D (2x RTX 6000, 32 CPUs, 200GB)... "
UNET_2D_JOB=$(sbatch --parsable slurm/train_unet_blanca.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $UNET_2D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

# Job 2: Swin 2D (2x RTX 6000)
echo -n "2. Swin 2D (2x RTX 6000, 32 CPUs, 200GB)... "
SWIN_2D_JOB=$(sbatch --parsable slurm/train_swin_blanca.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $SWIN_2D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

# Job 3: UNet 3D (3x RTX 6000) - MOST INTENSIVE
echo -n "3. UNet 3D (3x RTX 6000, 48 CPUs, 250GB)... "
UNET_3D_JOB=$(sbatch --parsable slurm/train_unet3d_blanca.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $UNET_3D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

# Job 4: ViT 3D (2x A100) - A100 EXCLUSIVE
echo -n "4. ViT 3D (2x A100, 40 CPUs, 400GB)... "
VIT_3D_JOB=$(sbatch --parsable slurm/train_vit3d_blanca.sbatch)
if [ $? -eq 0 ]; then
    echo "✓ Job ID: $VIT_3D_JOB"
else
    echo "✗ FAILED"
    exit 1
fi

echo ""
echo "=========================================="
echo "ALL JOBS SUBMITTED SUCCESSFULLY!"
echo "=========================================="
echo ""
echo "Job Summary:"
echo "  UNet 2D:  $UNET_2D_JOB (2x RTX 6000)"
echo "  Swin 2D:  $SWIN_2D_JOB (2x RTX 6000)"
echo "  UNet 3D:  $UNET_3D_JOB (3x RTX 6000)"
echo "  ViT 3D:   $VIT_3D_JOB (2x A100)"
echo ""
echo "Total Resources:"
echo "  - 9 GPUs (7x RTX 6000 + 2x A100)"
echo "  - 152 CPU cores"
echo "  - 1050 GB RAM"
echo "  - 4 parallel training jobs"
echo "  - 7 days walltime each"
echo ""
echo "=========================================="
echo "MONITORING COMMANDS"
echo "=========================================="
echo ""
echo "Check job status:"
echo "  squeue -u \$USER"
echo ""
echo "Watch all jobs:"
echo "  watch -n 10 'squeue -u \$USER'"
echo ""
echo "Check specific job:"
echo "  scontrol show job <JOB_ID>"
echo ""
echo "Monitor GPU usage (once running):"
echo "  ssh <node> nvidia-smi"
echo ""
echo "View logs in real-time:"
echo "  tail -f logs/unet_blanca_${UNET_2D_JOB}.out"
echo "  tail -f logs/swin_blanca_${SWIN_2D_JOB}.out"
echo "  tail -f logs/unet3d_blanca_${UNET_3D_JOB}.out"
echo "  tail -f logs/vit3d_blanca_${VIT_3D_JOB}.out"
echo ""
echo "Cancel all jobs:"
echo "  scancel $UNET_2D_JOB $SWIN_2D_JOB $UNET_3D_JOB $VIT_3D_JOB"
echo ""
echo "=========================================="
echo "TENSORBOARD ACCESS"
echo "=========================================="
echo ""
echo "Each job runs TensorBoard on a unique port (6006 + job_id % 1000)"
echo "To access, SSH tunnel from your local machine:"
echo ""
echo "For UNet 2D:"
echo "  Port: \$((6006 + $UNET_2D_JOB % 1000))"
echo ""
echo "For Swin 2D:"
echo "  Port: \$((6006 + $SWIN_2D_JOB % 1000))"
echo ""
echo "For UNet 3D:"
echo "  Port: \$((6006 + $UNET_3D_JOB % 1000))"
echo ""
echo "For ViT 3D:"
echo "  Port: \$((6006 + $VIT_3D_JOB % 1000))"
echo ""
echo "Once jobs are running, find the node with:"
echo "  squeue -u \$USER -o \"%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R\""
echo ""
echo "Then create SSH tunnel (example):"
echo "  ssh -L PORT:NODE:PORT gest9386@login.rc.colorado.edu"
echo ""
echo "=========================================="

# Save job IDs to file for later reference
cat > logs/blanca_job_ids.txt << EOF
UNet_2D=$UNET_2D_JOB
Swin_2D=$SWIN_2D_JOB
UNet_3D=$UNET_3D_JOB
ViT_3D=$VIT_3D_JOB
Submitted=$(date)
EOF

echo "Job IDs saved to: logs/blanca_job_ids.txt"
echo ""
echo "Happy training! 🚀"
