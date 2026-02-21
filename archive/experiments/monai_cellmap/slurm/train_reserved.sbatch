#!/bin/bash
#SBATCH --job-name=monai_cellmap
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:6
#SBATCH --mem=480g
#SBATCH --time=11-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/monai_cellmap_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/monai_cellmap_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# MONAI CellMap Training — 3 Models in Parallel on Reserved L40S
# ============================================================
# GPU layout:  SegResNet  → GPUs 0,1
#              FlexUNet   → GPUs 2,3
#              SwinUNETR  → GPUs 4,5
#              (GPU 6 left free for 2D job)
#
# Each model runs as a background torchrun process with its own
# CUDA_VISIBLE_DEVICES and unique MASTER_PORT to avoid DDP conflicts.
# Resource budget:
#   ~75 GB RAM per model  ×  3  =  ~225 GB  /  1007 GB
#   ~40 GB VRAM per GPU   ×  6  =  ~240 GB  /  336 GB
#
# Usage:
#   sbatch experiments/monai_cellmap/slurm/train_reserved.sbatch

set -euo pipefail

echo "============================================================"
echo " MONAI CellMap — Parallel Training (3 models × 2 GPUs)"
echo "============================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "GPUs: $SLURM_GPUS_ON_NODE"

# --- Environment ---
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

# --- Verify ---
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "MONAI: $(python -c 'import monai; print(monai.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"

# --- Performance ---
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVIDIA_TF32_OVERRIDE=1

# --- Paths ---
REPO_DIR="/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation"
EXP_DIR="$REPO_DIR/experiments/monai_cellmap"
LOG_DIR="$REPO_DIR/logs"
cd "$REPO_DIR"

echo ""
echo "=== GPU Info ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

echo ""
echo "=== RAM before training ==="
free -g

# --- Background resource monitor (every 5 min) ---
(while true; do
    echo "[MONITOR $(date '+%m/%d %H:%M')] RAM: $(free -g | awk '/Mem:/{print $3"/"$2" GB"}')"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
    echo "---"
    sleep 300
done) &
MONITOR_PID=$!

# ============================================================
# Launch all 3 models in parallel — each on 2 GPUs
# ============================================================
# Each torchrun gets its own:
#   - CUDA_VISIBLE_DEVICES (maps physical GPUs → local 0,1)
#   - MASTER_PORT (avoid DDP port collision)
#   - stdout/stderr log files

echo ""
echo "=========================================="
echo " Launching 3 models in parallel: $(date)"
echo "=========================================="

# --- Model 1: SegResNet (GPUs 0,1) ---
echo "[$(date '+%H:%M:%S')] Starting SegResNet on GPUs 0,1 (port 29500)"
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    "$EXP_DIR/train.py" \
    -C cfg_segresnet \
    --fold -1 \
    > "$LOG_DIR/segresnet_${SLURM_JOB_ID}.out" 2>&1 &
PID_SEGRESNET=$!

# --- Model 2: FlexibleUNet (GPUs 2,3) ---
echo "[$(date '+%H:%M:%S')] Starting FlexUNet on GPUs 2,3 (port 29501)"
CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    "$EXP_DIR/train.py" \
    -C cfg_flexunet_resnet \
    --fold -1 \
    > "$LOG_DIR/flexunet_${SLURM_JOB_ID}.out" 2>&1 &
PID_FLEXUNET=$!

# --- Model 3: SwinUNETR (GPUs 4,5) ---
echo "[$(date '+%H:%M:%S')] Starting SwinUNETR on GPUs 4,5 (port 29502)"
CUDA_VISIBLE_DEVICES=4,5 torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    "$EXP_DIR/train.py" \
    -C cfg_swinunetr \
    --fold -1 \
    > "$LOG_DIR/swinunetr_${SLURM_JOB_ID}.out" 2>&1 &
PID_SWINUNETR=$!

echo ""
echo "PIDs: SegResNet=$PID_SEGRESNET  FlexUNet=$PID_FLEXUNET  SwinUNETR=$PID_SWINUNETR"
echo "Logs:"
echo "  SegResNet : $LOG_DIR/segresnet_${SLURM_JOB_ID}.out"
echo "  FlexUNet  : $LOG_DIR/flexunet_${SLURM_JOB_ID}.out"
echo "  SwinUNETR : $LOG_DIR/swinunetr_${SLURM_JOB_ID}.out"
echo ""

# --- Wait for all 3 to finish ---
# Track exit codes independently so one failure doesn't kill the others.
FAIL=0

wait $PID_SEGRESNET
RC=$?
echo "[$(date '+%H:%M:%S')] SegResNet  finished (exit $RC)"
[ $RC -ne 0 ] && FAIL=1

wait $PID_FLEXUNET
RC=$?
echo "[$(date '+%H:%M:%S')] FlexUNet   finished (exit $RC)"
[ $RC -ne 0 ] && FAIL=1

wait $PID_SWINUNETR
RC=$?
echo "[$(date '+%H:%M:%S')] SwinUNETR  finished (exit $RC)"
[ $RC -ne 0 ] && FAIL=1

# --- Cleanup ---
kill $MONITOR_PID 2>/dev/null

echo ""
echo "============================================================"
echo " All training complete: $(date)"
echo " Any failures: $( [ $FAIL -eq 0 ] && echo 'NO' || echo 'YES — check per-model logs' )"
echo "============================================================"

exit $FAIL
