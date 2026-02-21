#!/bin/bash
#SBATCH --job-name=abl_mask
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=200g
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_mask_l40s_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_mask_l40s_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# Masking Strategy Ablation — L40S 48GB
#
# Same experiment as H100 version but with L40S resources.
# 128³ patches fit comfortably on L40S (~32 GB).
# ~6h per config, 11 configs → ~66h total.
#
# Split across two L40S jobs:
#   Job 1 (this): configs 0-5 (~36h)
#   Job 2: configs 6-10 (~30h)
# ============================================================
set -euo pipefail

echo "============================================================"
echo " Masking Ablation (L40S) — Job $SLURM_JOB_ID on $(hostname)"
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

# --- Parse optional args ---
export ABL_START=${1:-0}
export ABL_END=${2:--1}

echo "Running masking ablation configs [$ABL_START, $ABL_END)"

# --- Run ---
torchrun --nproc_per_node=2 --master_port=29611 \
    "$EXP_DIR/train_ablation.py" \
    --axis masking \
    --skip-completed --resume

echo ""
echo "Masking Ablation (L40S) complete: $(date)"
