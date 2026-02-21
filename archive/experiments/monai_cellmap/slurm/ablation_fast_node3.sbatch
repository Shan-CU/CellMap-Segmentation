#!/bin/bash
#SBATCH --job-name=abl_N3
#SBATCH --partition=h100_sn
#SBATCH --account=rc_alain_pi
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=400g
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_N3_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_N3_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# FAST ABLATION NODE 3 — 2x H100, 1 config stream
#
# Slot E (GPUs 0,1): tversky idx 0-2 (3 configs)
# ============================================================
set -uo pipefail

echo "============================================================"
echo " ABLATION NODE 3 — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo " Slot E (GPU 0,1): tversky 0-2"
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

# --- Slot E: GPUs 0,1 — tversky 0-2 ---
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8
export ABL_START=0
export ABL_END=3
export MASTER_PORT=29650
echo "[Slot E] tversky idx 0-2 on GPUs 0,1 — $(date)"
torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
    "$EXP_DIR/train_ablation.py" --axis tversky --skip-completed --resume
echo "[Slot E] DONE — $(date)"
echo "NODE 3 COMPLETE — $(date)"
