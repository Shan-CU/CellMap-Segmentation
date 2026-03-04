#!/bin/bash
# ============================================================================
# Phase 2 v5: Launch all optimized runs
# ============================================================================
# Usage: bash training/slurm/phase2v5/launch_all.sh
#
# Models launched:
#   L40 (single GPU):
#     - p2v5_resnet_2d  → 1× L40
#     - p2v5_unet_2d    → 1× L40
#
#   A100 (multi-GPU node):
#     - p2v5_segresnet_3d → 2× A100 DDP (port 29500)
#     - p2v5_unet_3d      → 2× A100 DDP (port 29501)
#     - p2v5_resnet_3d    → 1× A100
#
# Total GPU allocation: 2× L40, 5× A100
#
# v5 improvements over v4 (all models):
#   - AdamW optimizer with proper weight decay (1e-4 for UNet/ResNet, 1e-5 for SegResNet)
#   - eta_min=1e-6 to prevent learning rate collapse
#   - More training steps (500 iters/epoch for 3D, 1000 for 2D, 300-1000 epochs)
#   - Intensity augmentation for all models
#   - Tuned loss: dice_bce with bce_weight=0.4, smooth=1e-3
#   - Lower gradient clip (0.5)
#   - Proper warmup (5% of total epochs)
# ============================================================================

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "Launching Phase 2 v5 runs..."
echo "============================================"

# --- 2D models on L40 ---
echo ""
echo "--- 2D Models (L40, single GPU) ---"

JOB_RESNET2D=$(sbatch --parsable "$SLURM_DIR/resnet_2d.sbatch")
echo "  ResNet 2D: Job $JOB_RESNET2D"

JOB_UNET2D=$(sbatch --parsable "$SLURM_DIR/unet_2d.sbatch")
echo "  UNet 2D:   Job $JOB_UNET2D"

# --- 3D models on A100 ---
echo ""
echo "--- 3D Models (A100, multi-GPU node) ---"

JOB_SEGRESNET=$(sbatch --parsable "$SLURM_DIR/segresnet_3d.sbatch")
echo "  SegResNetDS 3D (2×A100 DDP): Job $JOB_SEGRESNET"

JOB_UNET3D=$(sbatch --parsable "$SLURM_DIR/unet_3d.sbatch")
echo "  UNet 3D (2×A100 DDP):        Job $JOB_UNET3D"

JOB_RESNET3D=$(sbatch --parsable "$SLURM_DIR/resnet_3d.sbatch")
echo "  ResNet 3D (1×A100):           Job $JOB_RESNET3D"

echo ""
echo "============================================"
echo "All v5 jobs submitted!"
echo "Monitor with: squeue -u \$USER"
echo "TensorBoard:  tensorboard --logdir runs/ablation"
echo "============================================"
