#!/bin/bash
#SBATCH --job-name=seg2d_val
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --reservation=gsgeorge_9034
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200g
#SBATCH --time=06:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/seg2d_val_%a_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/seg2d_val_%a_%j.err
#SBATCH --array=0-3

# ============================================================
# Segment ALL 45 validation crops using trained 2D models
# ============================================================
# Reservation: gsgeorge_9034 on g181003 (8× L40S 48GB, 64 CPUs, ~1TB RAM)
# 4 array tasks run in parallel, one GPU each:
#   Array 0: ResNet 2D   (best: Dice 0.4819)
#   Array 1: UNet 2D     (best: Dice 0.2349)
#   Array 2: Swin 2D     (best: Dice 0.1225)
#   Array 3: ViT-V-Net   (best: Dice 0.1829)
#
# Each task: 1× L40S, 12 CPUs, 200G RAM
# Total: 4× L40S, 48 CPUs, 800G (leaves 4 GPUs free on the node)
#
# Output: /work/users/g/s/gsgeorge/cellmap/runs/monai_2d/val_volumes/<model>/<crop>/
#   - raw_image.nii.gz, ground_truth.nii.gz, prediction.nii.gz, dice_summary.json
# ============================================================

MODELS=(resnet_r3 unet_r3 swin_r3 vit_r3)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

set -euo pipefail

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/monai_cellmap

echo "=========================================="
echo "  Model: $MODEL"
echo "  SLURM Job: $SLURM_JOB_ID (array $SLURM_ARRAY_TASK_ID)"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  RAM: $(free -g | awk '/Mem:/{print $2}')G total"
echo "=========================================="

python segment_val_volumes_2d.py \
    --model "$MODEL" \
    --device cuda

echo "Done: $MODEL"
