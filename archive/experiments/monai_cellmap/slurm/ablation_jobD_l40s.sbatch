#!/bin/bash
#SBATCH --job-name=abl_D
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
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_D_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_D_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# ABLATION JOB D — L40S (reservation g181003, 2 GPUs)
#
# Tversky ablation second half: configs 4-7 × ~6h = ~24h
# Then Masking ablation overflow: configs 6-10 × ~6h = ~30h
# Total: ~54h
# ============================================================
set -euo pipefail

echo "============================================================"
echo " ABLATION JOB D (L40S) — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo " Phase 1: Tversky 4-7, Phase 2: Masking 6-10"
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

# --- Phase 1: Tversky ablation (second half: 4-7) ---
echo ""
echo "========== PHASE 1: Tversky Ablation (configs 4-7) =========="
export ABL_START=4
export ABL_END=8

torchrun --nproc_per_node=2 --master_port=29631 \
    "$EXP_DIR/train_ablation.py" \
    --axis tversky \
    --skip-completed --resume

echo "Phase 1 complete: $(date)"

# --- Phase 2: Masking ablation overflow (configs 6-10, skip-completed) ---
echo ""
echo "========== PHASE 2: Masking Ablation (configs 6-10) =========="
export ABL_START=6
export ABL_END=11

torchrun --nproc_per_node=2 --master_port=29631 \
    "$EXP_DIR/train_ablation.py" \
    --axis masking \
    --skip-completed --resume

echo ""
echo "ABLATION JOB D (L40S) complete: $(date)"
