#!/bin/bash
#SBATCH --job-name=canary_v6_workers
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=runs/ablation/logs/canary_v6_workers_%j.out
#SBATCH --error=runs/ablation/logs/canary_v6_workers_%j.err

set -euo pipefail

echo "=== Canary v6: v5 fixes + num_workers=4 ==="
echo "Purpose: Verify memory is stable with multiple DataLoader workers"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs -I{} echo "GPU: {}"
echo "Start: $(date)"

export MEMORY_LOG_STEPS=1
export CELLMAP_TENSORSTORE_CACHE_BYTES=1

# glibc malloc tuning — inherited by forked workers
export MALLOC_TRIM_THRESHOLD_=65536
export MALLOC_MMAP_THRESHOLD_=65536
export MALLOC_ARENA_MAX=2

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

echo "cellmap-data version: $(python -c 'import cellmap_data; print(cellmap_data.__version__)')"
echo "MALLOC_ARENA_MAX=$MALLOC_ARENA_MAX"
echo "num_workers=4"

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

python -m training.train \
  --experiment_name canary_v6_workers \
  --run_dir runs/ablation \
  --model segresnet_3d \
  --loss dice_bce \
  --batch_size 2 \
  --input_shape 128 128 128 \
  --input_scale 8 8 8 \
  --epochs 1 \
  --iterations_per_epoch 12 \
  --learning_rate 1e-4 \
  --amp \
  --val_every_n_epochs 999 \
  --num_workers 4 \
  --debug_memory \
  --ema_decay 0.999

echo "End: $(date)"
