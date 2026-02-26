#!/bin/bash
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=384G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval_2d_perclass
#SBATCH --output=runs/ablation/logs/eval_2d_perclass_%j.out
#SBATCH --error=runs/ablation/logs/eval_2d_perclass_%j.err

# ============================================================================
# Per-class validation evaluation of all 29 2D ablation runs
# Loads each best.pth, runs inference on val set, computes per-class Dice/IoU
# ============================================================================

set -euo pipefail

echo "============================================"
echo "Per-class 2D Evaluation"
echo "Cluster: Longleaf | Partition: l40-gpu"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "============================================"

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

MAX_BATCHES="${MAX_BATCHES:-200}"

python -m training.eval_2d_perclass \
    --run_dir runs/ablation \
    --max_batches ${MAX_BATCHES} \
    --batch_size 8 \
    --num_workers 8

echo "Done!"
