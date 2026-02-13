#!/bin/bash
#SBATCH --job-name=monai_probe
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=63
#SBATCH --gres=gpu:7
#SBATCH --mem=920g
#SBATCH --time=00:45:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/monai_probe_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/monai_probe_%j.err

# ============================================================
# Resource Probe — Launch SwinUNETR on 2 GPUs, measure everything
# SwinUNETR is the heaviest model (55M params, transformer attention)
# so it gives us the worst-case resource usage.
# ============================================================

set -euo pipefail

echo "=== RESOURCE PROBE ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVIDIA_TF32_OVERRIDE=1

REPO_DIR="/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation"
EXP_DIR="$REPO_DIR/experiments/monai_cellmap"
cd "$REPO_DIR"

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "=== RAM before training ==="
free -g

# Background resource logger: GPU mem + host RAM every 30s
(while true; do
    echo "[PROBE $(date +%H:%M:%S)] HOST RAM: $(free -g | awk '/Mem:/{print $3"/"$2" GB"}') | GPU:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
    echo "---"
    sleep 30
done) &
LOGGER_PID=$!

echo ""
echo "=== Launching SwinUNETR on GPUs 0,1 (2 GPUs) ==="
echo "Heaviest model — 55M params, transformer attention."

# Only use GPUs 0,1
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    "$EXP_DIR/train.py" \
    -C cfg_swinunetr \
    --fold -1 &
TRAIN_PID=$!

# Let it run for 30 minutes (cache warmup ~5-8 min + steady-state training)
sleep 1800

echo ""
echo "=== Final resource snapshot after 30 min ==="
echo "HOST RAM:"
free -g
echo ""
echo "GPU Memory:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""
echo "Per-process GPU memory:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# Kill training
kill $TRAIN_PID 2>/dev/null
wait $TRAIN_PID 2>/dev/null
kill $LOGGER_PID 2>/dev/null

echo ""
echo "=== Probe complete: $(date) ==="
