#!/bin/bash
#SBATCH --job-name=cellmap_2d_r3
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --reservation=gsgeorge_9034
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --mem=200g
#SBATCH --time=11-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/2d_r3_%a_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/2d_r3_%a_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu
#SBATCH --array=0-3

# ============================================================
# 2D Models — Round 3 (MONAI pipeline, all optimizations)
# ============================================================
# Reservation: gsgeorge_9034 on g181003 (expires 2026-03-17)
# L40S node: 8× L40S 48GB, 64 CPUs, ~1TB RAM
# SegResNet 3D also on g181003: 2 GPUs, 16 CPUs, 200G
# Remaining: 6 GPUs, 48 CPUs, ~800G for 4 × 2D jobs
# Each 2D job: 1 GPU, 12 CPUs, 200G RAM
#   Array 0: ResNet 2D     (4 vol×8 samp=32 slices/fwd, ~8-10 GB VRAM)
#   Array 1: UNet 2D       (4 vol×8 samp=32 slices/fwd, ~10-14 GB VRAM)
#   Array 2: Swin 2D       (2 vol×8 samp=16 slices/fwd, ~18-22 GB VRAM)
#   Array 3: ViT-V-Net 2D  (1 vol×8 samp= 8 slices/fwd, ~25-30 GB VRAM)
#
# R3 optimizations (from EXPERIMENT_FINDINGS.md):
#   - Tversky α=0.6/β=0.4 (+47%)
#   - Balanced Softmax τ=1.0 (+54%)
#   - Foreground masking (+110%)
#   - BBox spatial masking (+55%)
#   - RAM-cached dataset (microsecond slicing)
#   - Multi-axis slice sampling (axial+coronal+sagittal)
# ============================================================

set -euo pipefail

# Config modules indexed by SLURM_ARRAY_TASK_ID
CONFIGS=(
    "cfg_2d_resnet_r3"
    "cfg_2d_unet_r3"
    "cfg_2d_swin_r3"
    "cfg_2d_vit_r3"
)
MODEL_NAMES=(
    "ResNet-2D-R3"
    "UNet-2D-R3"
    "SwinTransformer-2D-R3"
    "ViTVNet-2D-R3"
)

CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
MODEL="${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}"

echo "============================================================"
echo " CellMap 2D R3 — ${MODEL} (array task ${SLURM_ARRAY_TASK_ID})"
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
cd "$REPO_DIR"

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""
echo "=== RAM ==="
free -g
echo ""
echo "=== Disk ==="
df -h /work/users/g/s/gsgeorge

# --- Create output dirs ---
mkdir -p /work/users/g/s/gsgeorge/cellmap/runs/monai_2d

# ============================================================
# Run training via the MONAI 2D pipeline (NOT CSC framework)
# ============================================================
echo ""
echo "=========================================="
echo " Starting ${MODEL}: $(date)"
echo " Config: ${CONFIG}"
echo "=========================================="

python experiments/monai_cellmap/train_2d.py -C "${CONFIG}"
RC=$?

echo ""
echo "=========================================="
echo " ${MODEL} finished (exit $RC) at $(date)"
echo "=========================================="

exit $RC
