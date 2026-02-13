#!/bin/bash
#SBATCH --job-name=cellmap_2d
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --time=11-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/cellmap_2d_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/cellmap_2d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# 2D Model Training — Sequential on 1× L40S GPU
# ============================================================
# Runs 2D UNet then 2D SwinTransformer SEQUENTIALLY on a single GPU.
# Uses the CSC framework's native train() function (NOT our MONAI pipeline).
#
# Time budget: 11 days total
#   Model 1: 2D UNet          (~5.5 days)
#   Model 2: 2D SwinTransformer (~5.5 days)
#
# If the 3D monai_cellmap job is using GPUs 0-5 on the same node,
# this job should land on GPU 6 (or wherever Slurm places it).

set -euo pipefail

echo "============================================================"
echo " CellMap 2D Sequential Training (1 GPU)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "GPUs assigned: ${CUDA_VISIBLE_DEVICES:-not set}"

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
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVIDIA_TF32_OVERRIDE=1

# --- Paths ---
REPO_DIR="/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation"
CFG_DIR="$REPO_DIR/experiments/monai_cellmap/configs_2d"
LOG_DIR="$REPO_DIR/logs"
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
# Model 1: 2D UNet
# ============================================================
echo ""
echo "=========================================="
echo " [1/2] Starting 2D UNet: $(date)"
echo "=========================================="

python "$CFG_DIR/train_2d_unet.py" \
    > "$LOG_DIR/unet2d_${SLURM_JOB_ID}.out" 2>&1
RC_UNET=$?

echo "[$(date '+%H:%M:%S')] 2D UNet finished (exit $RC_UNET)"

# ============================================================
# Model 2: 2D SwinTransformer
# ============================================================
echo ""
echo "=========================================="
echo " [2/2] Starting 2D SwinTransformer: $(date)"
echo "=========================================="

python "$CFG_DIR/train_2d_swin.py" \
    > "$LOG_DIR/swin2d_${SLURM_JOB_ID}.out" 2>&1
RC_SWIN=$?

echo "[$(date '+%H:%M:%S')] 2D SwinTransformer finished (exit $RC_SWIN)"

# ============================================================
# Summary
# ============================================================
FAIL=0
[ $RC_UNET -ne 0 ] && FAIL=1
[ $RC_SWIN -ne 0 ] && FAIL=1

echo ""
echo "============================================================"
echo " All 2D training complete: $(date)"
echo " UNet exit code:  $RC_UNET"
echo " Swin exit code:  $RC_SWIN"
echo " Any failures: $([ $FAIL -eq 0 ] && echo 'NO' || echo 'YES')"
echo "============================================================"

exit $FAIL
