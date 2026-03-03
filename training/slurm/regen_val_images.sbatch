#!/bin/bash
#SBATCH --job-name=regen_val_imgs
#SBATCH --partition=a100-multi-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-06:00:00
#SBATCH --output=runs/ablation/logs/regen_val_imgs_%j.out
#SBATCH --error=runs/ablation/logs/regen_val_imgs_%j.err

# ============================================================================
# Retroactive Validation Image Regeneration
# ============================================================================
# Regenerates proper (non-empty) validation visualization images from ALL
# saved checkpoints across all p2v4_* models.
#
# Fixes the "black validation images" bug where the original training run
# uniformly sampled from 132K+ validation blocks (~98% empty padded regions).
#
# Runs on a100-gpu since l40-gpu is draining (cluster maintenance).
# All models (both 2D and 3D) fit on a single A100 80GB for inference.
#
# Expected: ~40 checkpoint-epoch pairs, ~2-4 hours total.
# ============================================================================

set -euo pipefail

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
mkdir -p runs/ablation/logs

# --- Memory leak mitigations ---
export CELLMAP_TENSORSTORE_CACHE_BYTES=1
export MALLOC_ARENA_MAX=2
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536

echo "========================================="
echo "Job $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Python: $(which python3)"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"
echo "Started: $(date)"
echo "========================================="

python3 -m training.regen_val_images \
    --run_dir runs/ablation \
    --max_val_batches 500

echo "========================================="
echo "Finished: $(date)"
echo "========================================="
