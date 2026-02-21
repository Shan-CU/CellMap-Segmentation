#!/bin/bash
#SBATCH --job-name=abl_N4
#SBATCH --partition=h100_sn
#SBATCH --account=rc_alain_pi
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=400g
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_N4_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_N4_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# FAST ABLATION NODE 4 — 2x H100, 1 config stream
#
# Slot F (GPUs 0,1): tversky idx 3-7 with --skip-completed
#   Already done: idx 4 (a06_b06), idx 5 (a07_b04)
#   Will run:     idx 3 (a05_b05), idx 6 (a04_b06), idx 7 (a08_b07)
# ============================================================
set -uo pipefail

echo "============================================================"
echo " ABLATION NODE 4 — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo " Slot F (GPU 0,1): tversky 3-7 (skip completed → runs 3,6,7)"
echo "============================================================"

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

echo "Python: $(which python)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVIDIA_TF32_OVERRIDE=1

REPO_DIR="/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation"
EXP_DIR="$REPO_DIR/experiments/monai_cellmap"
cd "$REPO_DIR"

V2_DIR="$REPO_DIR/auto3dseg/nifti_data_v2"
N_IMAGES=$(ls "$V2_DIR/images/" 2>/dev/null | wc -l)
[ "$N_IMAGES" -lt 200 ] && echo "ERROR: V2 data missing" && exit 1
echo "V2 data OK: $N_IMAGES images"

# --- Slot F: GPUs 0,1 — tversky 3-7 (skips completed idx 4,5) ---
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export ABL_START=3
export ABL_END=8
export MASTER_PORT=29660
echo "[Slot F] tversky idx 3-7 (skip completed) on GPUs 0,1 — $(date)"
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
    "$EXP_DIR/train_ablation.py" --axis tversky --skip-completed --resume
echo "[Slot F] DONE — $(date)"
echo "NODE 4 COMPLETE — $(date)"
