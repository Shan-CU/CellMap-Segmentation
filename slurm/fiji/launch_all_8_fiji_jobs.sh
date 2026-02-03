#!/bin/bash
# ============================================================
# Fiji 8-Model Training Launcher
# All Sycamore H100 optimizations applied for Fiji A100s
# ============================================================

echo "=========================================="
echo "FIJI - 8 MODEL TRAINING LAUNCHER"
echo "Date: $(date)"
echo "=========================================="
echo "Working directory: $(pwd)"
echo ""
echo "NOTE: Run this from project root:"
echo "  cd /scratch/Users/gest9386/CellMap-Segmentation"
echo "  bash slurm/fiji/launch_all_8_fiji_jobs.sh"
echo ""

# Verify we're in project root
if [ ! -d "experiments/model_comparison" ]; then
    echo "ERROR: Must run from CellMap-Segmentation project root!"
    echo "Current dir: $(pwd)"
    exit 1
fi

# Create logs directory
mkdir -p logs

# ============================================================
# SUBMIT ALL 8 JOBS
# ============================================================

echo "=========================================="
echo "SUBMITTING 8 JOBS TO FIJI"
echo "=========================================="
echo ""

echo "2D Models (4 jobs - 2 GPUs each):"
echo "=========================================="
echo -n "1. ResNet 2D (2x A100, 32 CPUs, 320GB)... "
RESNET_2D=$(sbatch --parsable slurm/fiji/train_resnet_2d.sbatch)
echo "✓ Job ID: $RESNET_2D"

echo -n "2. UNet 2D (2x A100, 32 CPUs, 200GB)... "
UNET_2D=$(sbatch --parsable slurm/fiji/train_unet_2d.sbatch)
echo "✓ Job ID: $UNET_2D"

echo -n "3. Swin 2D (2x A100, 32 CPUs, 200GB)... "
SWIN_2D=$(sbatch --parsable slurm/fiji/train_swin_2d.sbatch)
echo "✓ Job ID: $SWIN_2D"

echo -n "4. ViT 2D (2x A100, 32 CPUs, 384GB)... "
VIT_2D=$(sbatch --parsable slurm/fiji/train_vit_2d.sbatch)
echo "✓ Job ID: $VIT_2D"

echo ""
echo "3D Models (4 jobs - 4 GPUs each):"
echo "=========================================="
echo -n "5. ResNet 3D (4x A100, 64 CPUs, 400GB)... "
RESNET_3D=$(sbatch --parsable slurm/fiji/train_resnet_3d.sbatch)
echo "✓ Job ID: $RESNET_3D"

echo -n "6. UNet 3D (4x A100, 64 CPUs, 400GB)... "
UNET_3D=$(sbatch --parsable slurm/fiji/train_unet_3d.sbatch)
echo "✓ Job ID: $UNET_3D"

echo -n "7. Swin 3D (4x A100, 64 CPUs, 480GB)... "
SWIN_3D=$(sbatch --parsable slurm/fiji/train_swin_3d.sbatch)
echo "✓ Job ID: $SWIN_3D"

echo -n "8. ViT 3D (4x A100, 64 CPUs, 400GB)... "
VIT_3D=$(sbatch --parsable slurm/fiji/train_vit_3d.sbatch)
echo "✓ Job ID: $VIT_3D"

echo ""
echo "=========================================="
echo "ALL 8 JOBS SUBMITTED SUCCESSFULLY!"
echo "=========================================="
echo ""

# Save job IDs
cat > logs/fiji_8job_ids.txt << EOF
ResNet 2D: $RESNET_2D
UNet 2D: $UNET_2D
Swin 2D: $SWIN_2D
ViT 2D: $VIT_2D
ResNet 3D: $RESNET_3D
UNet 3D: $UNET_3D
Swin 3D: $SWIN_3D
ViT 3D: $VIT_3D
EOF

echo "Job Summary:"
echo "============"
echo "2D Models:"
echo "  ResNet 2D: $RESNET_2D (2x A100)"
echo "  UNet 2D:   $UNET_2D (2x A100)"
echo "  Swin 2D:   $SWIN_2D (2x A100)"
echo "  ViT 2D:    $VIT_2D (2x A100)"
echo ""
echo "3D Models:"
echo "  ResNet 3D: $RESNET_3D (4x A100)"
echo "  UNet 3D:   $UNET_3D (4x A100)"
echo "  Swin 3D:   $SWIN_3D (4x A100)"
echo "  ViT 3D:    $VIT_3D (4x A100)"
echo ""
echo "Total Resources Requested:"
echo "  - A100 GPUs: 24 across 8 jobs"
echo "    - 2D models: 8 GPUs (4 jobs × 2 GPUs)"
echo "    - 3D models: 16 GPUs (4 jobs × 4 GPUs)"
echo "  - Total CPUs: 320"
echo "  - Total RAM: ~2.6TB"
echo "  - Walltime: 7 days per job (Fiji's max)"
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
echo "  scancel $RESNET_2D $UNET_2D $SWIN_2D $VIT_2D $RESNET_3D $UNET_3D $SWIN_3D $VIT_3D"
echo ""
echo "View logs:"
echo "  tail -f logs/*_fiji_*.out"
echo "  tail -f experiments/model_comparison/logs/*_*.out"
echo ""
echo "TensorBoard ports (once running):"
echo "  ResNet 2D: \$((6006 + $RESNET_2D % 1000))"
echo "  UNet 2D:   \$((6006 + $UNET_2D % 1000))"
echo "  Swin 2D:   \$((6006 + $SWIN_2D % 1000))"
echo "  ViT 2D:    \$((6006 + $VIT_2D % 1000))"
echo "  ResNet 3D: \$((6006 + $RESNET_3D % 1000))"
echo "  UNet 3D:   \$((6006 + $UNET_3D % 1000))"
echo "  Swin 3D:   \$((6006 + $SWIN_3D % 1000))"
echo "  ViT 3D:    \$((6006 + $VIT_3D % 1000))"
echo ""
echo "To access TensorBoard, SSH tunnel:"
echo "  ssh -L PORT:NODE:PORT gest9386@fiji.colorado.edu"
echo "  Then open: http://localhost:PORT"
echo ""
echo "=========================================="
echo "Job IDs saved to: logs/fiji_8job_ids.txt"
echo ""
echo "Happy training! 🚀🚀🚀"
echo "=========================================="
