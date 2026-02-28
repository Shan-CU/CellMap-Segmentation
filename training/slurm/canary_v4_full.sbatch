#!/bin/bash
#SBATCH --job-name=canary_v4_full
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=runs/ablation/logs/canary_v4_full_%j.out
#SBATCH --error=runs/ablation/logs/canary_v4_full_%j.err

set -euo pipefail

echo "=== Canary v4: PR#64 + monkey-patch _clear_array_cache + CACHE=1 ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs -I{} echo "GPU: {}"
echo "Start: $(date)"

export MEMORY_LOG_STEPS=1
export CELLMAP_TENSORSTORE_CACHE_BYTES=1
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

echo "cellmap-data version: $(python -c 'import cellmap_data; print(cellmap_data.__version__)')"
echo "CELLMAP_TENSORSTORE_CACHE_BYTES=$CELLMAP_TENSORSTORE_CACHE_BYTES"

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

python -m training.train \
  --experiment_name canary_v4_full \
  --run_dir runs/ablation \
  --model segresnet_3d \
  --loss dice_bce \
  --batch_size 2 \
  --input_shape 128 128 128 \
  --input_scale 8 8 8 \
  --epochs 1 \
  --iterations_per_epoch 8 \
  --learning_rate 1e-4 \
  --amp \
  --val_every_n_epochs 999 \
  --num_workers 0 \
  --debug_memory \
  --ema_decay 0.999

echo "End: $(date)"
