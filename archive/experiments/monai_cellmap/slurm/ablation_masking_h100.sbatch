#!/bin/bash
#SBATCH --job-name=abl_mask
#SBATCH --partition=h100_mn
#SBATCH --account=rc_alain_pi
#SBATCH --qos=h100_mn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=400g
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_mask_h100_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_mask_h100_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# Masking Strategy Ablation — H100 80GB
#
# Runs 11 masking configs sequentially (100 epochs each, ~3.5h each)
# Total: ~38h (fits within 5-day limit with margin)
#
# Priority order:
#   0. no_mask           (R2-equivalent baseline)
#   1. fg_only           (isolate foreground masking)
#   2. bbox_tight_bg020  (hypothesis: sweet spot)
#   3. bbox_tight_bg050  (mild masking)
#   ...
#
# Use --start/--end to split across multiple jobs if needed:
#   --start 0 --end 4    → first 4 configs
#   --start 4 --end 8    → next 4 configs
#   --start 8 --end 11   → last 3 configs
# ============================================================
set -euo pipefail

echo "============================================================"
echo " Masking Ablation (H100) — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
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

# --- Parse optional args from sbatch command line ---
export ABL_START=${1:-0}
export ABL_END=${2:--1}

echo "Running masking ablation configs [$ABL_START, $ABL_END)"

# --- Run all configs sequentially (skips completed) ---
torchrun --nproc_per_node=2 --master_port=29610 \
    "$EXP_DIR/train_ablation.py" \
    --axis masking \
    --skip-completed --resume

echo ""
echo "Masking Ablation (H100) complete: $(date)"
