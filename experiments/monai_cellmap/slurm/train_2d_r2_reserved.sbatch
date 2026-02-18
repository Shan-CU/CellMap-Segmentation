#!/bin/bash
#SBATCH --job-name=cellmap_2d_r2
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --mem=230g
#SBATCH --time=11-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/2d_r2_%a_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/2d_r2_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu
#SBATCH --array=0-3

# ============================================================
# 2D Models — Round 2 (35 classes) — Job Array on L40S Reservation
# ============================================================
# Node g181003: 8× L40S, 64 CPUs, ~1TB RAM
# 4 models, each gets: 1 GPU, 16 CPUs, 240g RAM (evenly spread)
#   Array 0: UNet 2D       (batch 32, ~8-10 GB VRAM)
#   Array 1: ResNet 2D     (batch 32, ~8-10 GB VRAM)
#   Array 2: Swin 2D       (batch 16, ~15-20 GB VRAM)
#   Array 3: ViT-V-Net 2D  (batch  4, ~20-25 GB VRAM)
#
# Reservation ends: 2026-03-17 (~28 days)
# ============================================================

set -euo pipefail

# Model configs indexed by SLURM_ARRAY_TASK_ID
CONFIGS=(
    "train_2d_unet_r2.py"
    "train_2d_resnet_r2.py"
    "train_2d_swin_r2.py"
    "train_2d_vit_r2.py"
)
MODEL_NAMES=(
    "UNet-2D"
    "ResNet-2D"
    "SwinTransformer-2D"
    "ViTVNet-2D"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo " CellMap 2D R2 — ${MODEL} (array task ${SLURM_ARRAY_TASK_ID})"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID})"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Config: ${CONFIG}"
echo "GPU assigned: ${CUDA_VISIBLE_DEVICES:-not set}"

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

# --- Fix libstdc++ for cellmap_data / IPython / sqlite3 ---
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# --- Verify ---
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# --- Performance ---
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVIDIA_TF32_OVERRIDE=1

# --- Paths ---
REPO_DIR="/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation"
CFG_DIR="$REPO_DIR/experiments/monai_cellmap/configs_2d"
cd "$REPO_DIR"

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
echo "=== RAM ==="
free -g

# --- Create output dirs ---
mkdir -p experiments/monai_cellmap/checkpoints_2d
mkdir -p experiments/monai_cellmap/tensorboard_2d

# ============================================================
# Run the selected model
# ============================================================
echo ""
echo "=========================================="
echo " Starting ${MODEL}: $(date)"
echo " Config: ${CFG_DIR}/${CONFIG}"
echo "=========================================="

python "${CFG_DIR}/${CONFIG}"
RC=$?

echo ""
echo "=========================================="
echo " ${MODEL} finished (exit $RC) at $(date)"
echo "=========================================="

exit $RC
