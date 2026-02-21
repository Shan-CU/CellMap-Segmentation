#!/bin/bash
#SBATCH --job-name=abl_B
#SBATCH --partition=h100_mn
#SBATCH --account=rc_alain_pi
#SBATCH --qos=h100_mn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=400g
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_B_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_B_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# ABLATION JOB B — H100 Node 2 (2 GPUs)
#
# Runs SECOND HALF of experiments:
#   1. Masking ablation configs 6-10 (5 configs × ~3.5h = ~17.5h)
#   2. Tversky ablation configs 4-7 (4 configs × ~3.5h = ~14h)
#   3. Weighting ablation all 6 (6 configs × ~3.5h = ~21h)
#   Total: ~52.5h (fits in 5-day limit)
# ============================================================
set -euo pipefail

echo "============================================================"
echo " ABLATION JOB B — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo " Phase 1: Masking 6-10, Phase 2: Tversky 4-7, Phase 3: Weighting all"
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

# --- Verify data ---
V2_DIR="$REPO_DIR/auto3dseg/nifti_data_v2"
N_IMAGES=$(ls "$V2_DIR/images/" 2>/dev/null | wc -l)
[ "$N_IMAGES" -lt 200 ] && echo "ERROR: V2 data missing ($N_IMAGES images)" && exit 1
echo "V2 data OK: $N_IMAGES images"

# --- Phase 1: Masking ablation (second half: configs 6-10) ---
echo ""
echo "========== PHASE 1: Masking Ablation (configs 6-10) =========="
export ABL_START=6
export ABL_END=11

torchrun --nproc_per_node=2 --master_port=29620 \
    "$EXP_DIR/train_ablation.py" \
    --axis masking \
    --skip-completed --resume

echo "Phase 1 complete: $(date)"

# --- Phase 2: Tversky ablation (second half: configs 4-7) ---
echo ""
echo "========== PHASE 2: Tversky Ablation (configs 4-7) =========="
export ABL_START=4
export ABL_END=8

torchrun --nproc_per_node=2 --master_port=29620 \
    "$EXP_DIR/train_ablation.py" \
    --axis tversky \
    --skip-completed --resume

echo "Phase 2 complete: $(date)"

# --- Phase 3: Weighting ablation (all 6 configs) ---
echo ""
echo "========== PHASE 3: Weighting Ablation (all 6) =========="
export ABL_START=0
export ABL_END=-1

torchrun --nproc_per_node=2 --master_port=29620 \
    "$EXP_DIR/train_ablation.py" \
    --axis weighting \
    --skip-completed --resume

echo ""
echo "ABLATION JOB B complete: $(date)"
