#!/bin/bash
#SBATCH --partition=h100_sn
#SBATCH --account=rc_alain_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=eval_14c_3d
#SBATCH --output=runs/ablation/logs/eval_14c_3d_%j.out
#SBATCH --error=runs/ablation/logs/eval_14c_3d_%j.err

# ============================================================================
# Per-class Dice evaluation of ALL 3D ablation runs (Sycamore H100)
# Uses OLD 14-class datasplit for fair comparison
# ============================================================================

set -euo pipefail

echo "============================================"
echo "Per-class 3D Evaluation — ALL experiments (14-class)"
echo "Cluster: Sycamore | Partition: h100_sn"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Started: $(date)"
echo "============================================"

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

MAX_BATCHES="${MAX_BATCHES:-200}"
EVAL_CLASSES="ecs,pm,mito_mem,mito_lum,mito_ribo,golgi_mem,golgi_lum,ves_mem,ves_lum,endo_mem,endo_lum,er_mem,er_lum,nuc"

python -m training.eval_all_perclass \
    --run_dir runs/ablation \
    --filter 3d \
    --max_batches ${MAX_BATCHES} \
    --num_workers 8 \
    --datasplit_path datasplit_14class_backup.csv \
    --eval_classes ${EVAL_CLASSES} \
    --output runs/ablation/eval_14class_3d.json

echo "Done: $(date)"
