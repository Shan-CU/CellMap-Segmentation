#!/bin/bash
#SBATCH --job-name=abl_C
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --reservation=gsgeorge_9034
#SBATCH --nodelist=g181003
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=200g
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_C_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_C_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# ABLATION JOB C — L40S (reservation g181003, 2 GPUs)
#
# Weighting ablation: all 6 configs × ~6h = ~36h
# Then Tversky ablation first half: configs 0-3 × ~6h = ~24h
# Total: ~60h
# ============================================================
set -euo pipefail

echo "============================================================"
echo " ABLATION JOB C (L40S) — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo " Phase 1: Weighting (all 6), Phase 2: Tversky 0-3"
echo "============================================================"

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

V2_DIR="$REPO_DIR/auto3dseg/nifti_data_v2"
N_IMAGES=$(ls "$V2_DIR/images/" 2>/dev/null | wc -l)
[ "$N_IMAGES" -lt 200 ] && echo "ERROR: V2 data missing ($N_IMAGES images)" && exit 1
echo "V2 data OK: $N_IMAGES images"

# --- Phase 1: Weighting ablation (all 6) ---
echo ""
echo "========== PHASE 1: Weighting Ablation (all 6) =========="
export ABL_START=0
export ABL_END=-1

torchrun --nproc_per_node=2 --master_port=29630 \
    "$EXP_DIR/train_ablation.py" \
    --axis weighting \
    --skip-completed --resume

echo "Phase 1 complete: $(date)"

# --- Phase 2: Tversky ablation (first half: 0-3) ---
echo ""
echo "========== PHASE 2: Tversky Ablation (configs 0-3) =========="
export ABL_START=0
export ABL_END=4

torchrun --nproc_per_node=2 --master_port=29630 \
    "$EXP_DIR/train_ablation.py" \
    --axis tversky \
    --skip-completed --resume

echo ""
echo "ABLATION JOB C (L40S) complete: $(date)"
