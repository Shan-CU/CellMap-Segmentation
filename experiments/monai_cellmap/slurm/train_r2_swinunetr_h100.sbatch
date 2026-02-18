#!/bin/bash
#SBATCH --job-name=swinunetr_r2
#SBATCH --partition=h100_mn
#SBATCH --account=rc_alain_pi
#SBATCH --qos=h100_mn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=200g
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/swinunetr_r2_h100_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/swinunetr_r2_h100_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# SwinUNETR v2 — Round 2 — H100 80GB (single GPU)
# Moved from L40S reservation to free up GPU for 2D models
# 96³ b2, dropout 0.1, 35 classes
# Est. VRAM: ~42 GB / 80 GB (38 GB headroom)
# Will auto-resume from checkpoint_last.pth (ep19+)
# ============================================================
set -euo pipefail

echo "============================================================"
echo " SwinUNETR R2 (H100) — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo "============================================================"

# --- Environment (Sycamore) ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVIDIA_TF32_OVERRIDE=1

REPO_DIR="/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation"
EXP_DIR="$REPO_DIR/experiments/monai_cellmap"
cd "$REPO_DIR"

# --- Verify v2 data ---
V2_DIR="$REPO_DIR/auto3dseg/nifti_data_v2"
N_IMAGES=$(ls "$V2_DIR/images/" 2>/dev/null | wc -l)
[ "$N_IMAGES" -lt 200 ] && echo "ERROR: V2 data missing ($N_IMAGES images)" && exit 1
echo "V2 data OK: $N_IMAGES images"

# --- Clean start: remove L40S checkpoints (DDP 2-GPU, only 19 epochs) ---
CKPT_DIR="/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/swinunetr_r2"
if [ -d "$CKPT_DIR" ]; then
    echo "Cleaning old L40S checkpoints for fresh H100 start..."
    rm -rf "$CKPT_DIR"
fi
mkdir -p "$CKPT_DIR"

# --- Train (single GPU, no torchrun needed) ---
python "$EXP_DIR/train.py" -C cfg_swinunetr --fold -1

echo ""
echo "SwinUNETR R2 (H100) complete: $(date)"
