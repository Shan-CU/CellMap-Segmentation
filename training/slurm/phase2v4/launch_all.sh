#!/bin/bash
# ============================================================================
# Phase 2 v4: Launch all 9 architecture comparison jobs
# ============================================================================
# Usage: Run on Longleaf login node:
#   cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
#   bash training/slurm/phase2v4/launch_all.sh
#
# Each .sbatch is fully self-contained — open any one to see the exact
# hyperparameters, architecture details, and research justification.
#
# GPU Plan:
#   L40S 48GB (6 jobs): resnet_2d, unet_2d, swin_2d, vit_2d, resnet_3d, unet_3d
#   A100 80GB (3 jobs):  segresnet_3d, swinunetr_3d, vitnet_3d
# ============================================================================

set -euo pipefail
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

SCRIPT_DIR="training/slurm/phase2v4"

echo "============================================"
echo "Phase 2 v4: Submitting 9 architecture comparison jobs"
echo "============================================"
echo ""

# --- 2D models on L40S ---
echo "--- 2D Models (L40S 48GB) ---"
for model in resnet_2d unet_2d swin_2d vit_2d; do
    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/${model}.sbatch")
    echo "  Submitted ${model} → Job ${JOB_ID}"
done

echo ""

# --- 3D CNN models on L40S ---
echo "--- 3D CNN Models (L40S 48GB) ---"
for model in resnet_3d unet_3d; do
    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/${model}.sbatch")
    echo "  Submitted ${model} → Job ${JOB_ID}"
done

echo ""

# --- 3D MONAI + ViT models on A100 ---
echo "--- 3D MONAI + ViT Models (A100 80GB) ---"
for model in segresnet_3d swinunetr_3d vitnet_3d; do
    JOB_ID=$(sbatch --parsable "${SCRIPT_DIR}/${model}.sbatch")
    echo "  Submitted ${model} → Job ${JOB_ID}"
done

echo ""
echo "============================================"
echo "All 9 jobs submitted."
echo "============================================"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     runs/ablation/logs/p2v4_*"
echo "TBoard:   tensorboard --logdir runs/ablation --bind_all"
echo ""
echo "Summary:"
echo "  p2v4_resnet_2d     L40S   ResNet 2D       RAdam  lr=1e-4  wd=0"
echo "  p2v4_unet_2d       L40S   UNet 2D         RAdam  lr=1e-4  wd=0     +InstanceNorm +dropout"
echo "  p2v4_swin_2d       L40S   Swin V2 2D      AdamW  lr=1e-4  wd=0.05"
echo "  p2v4_vit_2d        L40S   ViT-V-Net 2D    AdamW  lr=1e-4  wd=0.01"
echo "  p2v4_resnet_3d     L40S   ResNet 3D       RAdam  lr=4e-4  wd=0"
echo "  p2v4_unet_3d       L40S   UNet 3D         RAdam  lr=4e-4  wd=0     +InstanceNorm +dropout"
echo "  p2v4_segresnet_3d  A100   SegResNetDS     AdamW  lr=2e-4  wd=1e-5  +DeepSup (batch=4)"
echo "  p2v4_swinunetr_3d  A100   SwinUNETR       AdamW  lr=4e-4  wd=1e-5  +GradCheckpoint"
echo "  p2v4_vitnet_3d     A100   ViT-V-Net 3D    RAdam  lr=4e-4  wd=0"
echo "  ALL: bias_init=-3.0  EMA(0.999)  dice_bce  fg_mask  val@5ep"
