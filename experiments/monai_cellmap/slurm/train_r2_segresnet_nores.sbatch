#!/bin/bash
#SBATCH --job-name=segresnet_r2
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=200g
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/segresnet_r2_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/segresnet_r2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# SegResNet 32f — Round 2 (128³, batch 2, deep supervision)
# 144³ caused CUDA OOM (~45 GB); 128³ is safe (~32 GB)
# NO RESERVATION — standard l40-gpu queue
# ============================================================
set -euo pipefail

echo "============================================================"
echo " SegResNet R2 — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo "============================================================"

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

echo "Python: $(which python)"
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

# --- Train ---
torchrun --nproc_per_node=2 --master_port=29500 \
    "$EXP_DIR/train.py" -C cfg_segresnet --fold -1

echo ""
echo "SegResNet R2 complete: $(date)"
