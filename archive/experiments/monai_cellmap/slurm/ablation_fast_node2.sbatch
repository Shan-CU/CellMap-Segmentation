#!/bin/bash
#SBATCH --job-name=abl_N2
#SBATCH --partition=h100_sn
#SBATCH --account=rc_alain_pi
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=800g
#SBATCH --time=2-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_N2_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/abl_N2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# FAST ABLATION NODE 2 — 4x H100, 2 parallel config streams
#
# Slot C (GPUs 0,1): masking idx 8-10 (3 configs)
# Slot D (GPUs 2,3): weighting idx 2-5 (4 configs)
# ============================================================
set -uo pipefail

echo "============================================================"
echo " ABLATION NODE 2 — Job $SLURM_JOB_ID on $(hostname)"
echo " Start: $(date)"
echo " Slot C (GPU 0,1): masking 8-10 | Slot D (GPU 2,3): weighting 2-5"
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

# --- Slot C: GPUs 0,1 — masking 8-10 ---
(
  export CUDA_VISIBLE_DEVICES=0,1
  export OMP_NUM_THREADS=8
  export ABL_START=8
  export ABL_END=11
  export MASTER_PORT=29630
  echo "[Slot C] masking idx 8-10 on GPUs 0,1 — $(date)"
  torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
      "$EXP_DIR/train_ablation.py" --axis masking --skip-completed --resume
  echo "[Slot C] DONE — $(date)"
) &
SLOT_C=$!

# --- Slot D: GPUs 2,3 — weighting 2-5 ---
(
  export CUDA_VISIBLE_DEVICES=2,3
  export OMP_NUM_THREADS=8
  export ABL_START=2
  export ABL_END=6
  export MASTER_PORT=29640
  echo "[Slot D] weighting idx 2-5 on GPUs 2,3 — $(date)"
  torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
      "$EXP_DIR/train_ablation.py" --axis weighting --skip-completed --resume
  echo "[Slot D] DONE — $(date)"
) &
SLOT_D=$!

wait $SLOT_C; C_RC=$?
wait $SLOT_D; D_RC=$?
echo "NODE 2 COMPLETE — $(date) — C=$C_RC D=$D_RC"
