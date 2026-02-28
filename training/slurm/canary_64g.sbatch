#!/bin/bash
#SBATCH --job-name=canary_64g_3d
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --account=rc_cburch_pi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/ablation/logs/canary_64g_3d_%j.out
#SBATCH --error=runs/ablation/logs/canary_64g_3d_%j.err

set -euo pipefail

echo "=== Canary: num_workers=0, 64G RAM, 3ep x 300iter, MEMORY_LOG_STEPS=1 ==="
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs -I{} echo "GPU: {}"
echo "Start: $(date)"

export MEMORY_LOG_STEPS=1
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

python -m training.train \
  --experiment_name canary_64g_3d \
  --run_dir runs/ablation \
  --model segresnet_3d \
  --loss dice_bce \
  --batch_size 2 \
  --input_shape 128 128 128 \
  --input_scale 8 8 8 \
  --epochs 3 \
  --iterations_per_epoch 300 \
  --learning_rate 1e-4 \
  --amp \
  --val_every_n_epochs 1 \
  --num_workers 0 \
  --debug_memory \
  --ema_decay 0.999

echo "End: $(date)"
