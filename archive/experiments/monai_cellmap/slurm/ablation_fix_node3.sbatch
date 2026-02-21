#!/bin/bash
#SBATCH --job-name=abl_N3f
#SBATCH --partition=h100_sn
#SBATCH --account=rc_alain_pi
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=400g
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_N3f_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_N3f_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# ABLATION NODE 3 FIX — sequential, 2x H100
# Runs: tversky 0-7 (skip completed)
# ============================================================
set -uo pipefail

echo "============================================================"
echo " ABLATION NODE 3 FIX — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo " Sequential: tversky 0-7 (skip completed)"
echo "============================================================"

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

echo "Python: $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVIDIA_TF32_OVERRIDE=1
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8

REPO_DIR="/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation"
EXP_DIR="$REPO_DIR/experiments/monai_cellmap"
cd "$REPO_DIR"

V2_DIR="$REPO_DIR/auto3dseg/nifti_data_v2"
N_IMAGES=$(ls "$V2_DIR/images/" 2>/dev/null | wc -l)
[ "$N_IMAGES" -lt 200 ] && echo "ERROR: V2 data missing" && exit 1
echo "V2 data OK: $N_IMAGES images"

# --- tversky idx 0-7 (skip completed) ---
echo "tversky 0-7 — $(date)"
export ABL_START=0 ABL_END=8 MASTER_PORT=29650
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
    "$EXP_DIR/train_ablation.py" --axis tversky --skip-completed --resume
echo "tversky DONE — $(date)"

echo "NODE 3 FIX COMPLETE — $(date)"
