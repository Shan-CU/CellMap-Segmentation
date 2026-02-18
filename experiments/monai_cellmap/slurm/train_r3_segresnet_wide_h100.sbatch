#!/bin/bash
#SBATCH --job-name=segwide_r3
#SBATCH --partition=h100_mn
#SBATCH --account=rc_alain_pi
#SBATCH --qos=h100_mn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=400g
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/segresnet_wide_r3_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/segresnet_wide_r3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# SegResNet-Wide 48f — Round 3 — H100 80GB
# FIX: Added box_class_mask_tight spatial masking (pad=0.05, bg=0.05)
# to prevent mode collapse from R2 (all models predicted everything positive)
# 160³ b2, deep supervision, 35 classes, BalSoftmax+Tversky+BboxMask
# ============================================================
set -euo pipefail

echo "============================================================"
echo " SegResNet-Wide R3 (H100) — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo " FIX: box_class_mask_tight spatial masking"
echo "============================================================"

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

echo "Python: $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export OMP_NUM_THREADS=8
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
torchrun --nproc_per_node=2 --master_port=29600 \
    "$EXP_DIR/train.py" -C cfg_segresnet_wide --fold -1

echo ""
echo "SegResNet-Wide R3 (H100) complete: $(date)"
