#!/bin/bash
#SBATCH --job-name=monai_r2
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --gres=gpu:8
#SBATCH --mem=600g
#SBATCH --time=5-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/monai_r2_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/logs/monai_r2_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# MONAI CellMap Round 2 Training — 35 classes, optimized
# ============================================================
# Round 2 improvements over Round 1 (based on 3-day training analysis):
#
#  1. 35 atomic classes (was 14) → covers all 48 tested challenge classes
#  2. Tversky α=0.6, β=0.4 (KEPT — experimentally validated)
#     loss_optimization exp: α=0.6/β=0.4 → Dice 0.370, α=0.3/β=0.7 → 0.028 (collapsed)
#     Balanced Softmax τ=1.0 for class weighting (class_weighting exp winner, 0.5711 Dice)
#  3. Mixup DISABLED for FlexUNet → was diluting unannotated channels
#  4. Bigger crops where GPU allows:
#     - SegResNet 32f:  144³ (was 128³) — 37 GB, 11 GB headroom
#     - SegResNet 48f:  128³ (NEW wide variant) — ~39 GB, 9 GB headroom
#     - FlexUNet:       128³ (was 96³) — 33 GB, 15 GB headroom
#     - SwinUNETR:      96³  (unchanged, was at 90% VRAM) — 42 GB, 6 GB headroom
#  5. 300 epochs (was 600) → best metrics hit at ep139-519, long tail wasted
#  6. SwinUNETR dropout 0.1 (was 0.0) → peaked at ep139 then overfit
#  7. Checkpoint only best+last+every50 (was every epoch = 480GB disk waste)
#  8. nifti_data_v2 with 35-class labels
#  9. 4th model (SegResNet-Wide 48f) for ensemble diversity on spare GPUs 6,7
#
# GPU layout:  SegResNet 32f  → GPUs 0,1
#              FlexUNet       → GPUs 2,3
#              SwinUNETR      → GPUs 4,5
#              SegResNet 48f  → GPUs 6,7  (NEW — uses spare GPUs)
#
# Output dirs (preserves Round 1 checkpoints):
#   $RUNS/segresnet_ds_r2/
#   $RUNS/flexunet_resnet34_r2/
#   $RUNS/swinunetr_r2/
#   $RUNS/segresnet_wide_r2/

set -euo pipefail

echo "============================================================"
echo " MONAI CellMap Round 2 — Parallel Training (4 × 2 GPUs)"
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

# --- Verify v2 data exists ---
V2_DIR="$REPO_DIR/auto3dseg/nifti_data_v2"
N_IMAGES=$(ls "$V2_DIR/images/" 2>/dev/null | wc -l)
N_LABELS=$(ls "$V2_DIR/labels/" 2>/dev/null | wc -l)
echo ""
echo "=== V2 Data Check ==="
echo "  Images: $N_IMAGES"
echo "  Labels: $N_LABELS"
if [ "$N_IMAGES" -lt 200 ] || [ "$N_LABELS" -lt 200 ]; then
    echo "ERROR: V2 conversion incomplete ($N_IMAGES images, $N_LABELS labels)"
    echo "       Need ~289 crops. Run convert_zarr_to_nifti_v2.py first."
    exit 1
fi
echo "  Datalist: $(ls -la "$V2_DIR/datalist.json" 2>/dev/null | awk '{print $5, "bytes"}' || echo 'MISSING')"

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
# Launch all 4 models in parallel — each on 2 GPUs
# ============================================================

echo ""
echo "=========================================="
echo " Launching 4 R2 models in parallel: $(date)"
echo "=========================================="

# --- Model 1: SegResNet 32f (GPUs 0,1) — 144³ crops ---
echo "[$(date '+%H:%M:%S')] Starting SegResNet R2 on GPUs 0,1 (port 29500)"
CUDA_VISIBLE_DEVICES=0,1 torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    "$EXP_DIR/train.py" \
    -C cfg_segresnet \
    --fold -1 \
    > "$LOG_DIR/segresnet_r2_${SLURM_JOB_ID}.out" 2>&1 &
PID_SEGRESNET=$!

# --- Model 2: FlexibleUNet (GPUs 2,3) — 128³ crops, no Mixup ---
echo "[$(date '+%H:%M:%S')] Starting FlexUNet R2 on GPUs 2,3 (port 29501)"
CUDA_VISIBLE_DEVICES=2,3 torchrun \
    --nproc_per_node=2 \
    --master_port=29501 \
    "$EXP_DIR/train.py" \
    -C cfg_flexunet_resnet \
    --fold -1 \
    > "$LOG_DIR/flexunet_r2_${SLURM_JOB_ID}.out" 2>&1 &
PID_FLEXUNET=$!

# --- Model 3: SwinUNETR (GPUs 4,5) — 96³ crops, with dropout ---
echo "[$(date '+%H:%M:%S')] Starting SwinUNETR R2 on GPUs 4,5 (port 29502)"
CUDA_VISIBLE_DEVICES=4,5 torchrun \
    --nproc_per_node=2 \
    --master_port=29502 \
    "$EXP_DIR/train.py" \
    -C cfg_swinunetr \
    --fold -1 \
    > "$LOG_DIR/swinunetr_r2_${SLURM_JOB_ID}.out" 2>&1 &
PID_SWINUNETR=$!

# --- Model 4: SegResNet-Wide 48f (GPUs 6,7) — 128³ crops, ensemble diversity ---
echo "[$(date '+%H:%M:%S')] Starting SegResNet-Wide R2 on GPUs 6,7 (port 29503)"
CUDA_VISIBLE_DEVICES=6,7 torchrun \
    --nproc_per_node=2 \
    --master_port=29503 \
    "$EXP_DIR/train.py" \
    -C cfg_segresnet_wide \
    --fold -1 \
    > "$LOG_DIR/segresnet_wide_r2_${SLURM_JOB_ID}.out" 2>&1 &
PID_SEGRESNET_WIDE=$!

echo ""
echo "PIDs: SegResNet=$PID_SEGRESNET  FlexUNet=$PID_FLEXUNET  SwinUNETR=$PID_SWINUNETR  SegResNet-Wide=$PID_SEGRESNET_WIDE"
echo "Logs:"
echo "  SegResNet      : $LOG_DIR/segresnet_r2_${SLURM_JOB_ID}.out"
echo "  FlexUNet       : $LOG_DIR/flexunet_r2_${SLURM_JOB_ID}.out"
echo "  SwinUNETR      : $LOG_DIR/swinunetr_r2_${SLURM_JOB_ID}.out"
echo "  SegResNet-Wide : $LOG_DIR/segresnet_wide_r2_${SLURM_JOB_ID}.out"
echo ""
echo "Round 2 optimizations active:"
echo "  Tversky: α=0.6, β=0.4 (precision-biased, experimentally validated)"
echo "  SegResNet 32f:  144³ (was 128³)"
echo "  SegResNet 48f:  128³ (NEW wide variant for ensemble diversity)"
echo "  FlexUNet:       128³ (was 96³), Mixup DISABLED"
echo "  SwinUNETR:      96³ unchanged, dropout=0.1"
echo "  Epochs: 300 (was 600)"
echo "  Classes: 35 atomic → 48 tested"
echo "  Models: 4 (was 3) — using all 8 GPUs"
echo ""

# --- Wait for all 4 to finish ---
FAIL=0

wait $PID_SEGRESNET
RC=$?
echo "[$(date '+%H:%M:%S')] SegResNet R2  finished (exit $RC)"
[ $RC -ne 0 ] && FAIL=1

wait $PID_FLEXUNET
RC=$?
echo "[$(date '+%H:%M:%S')] FlexUNet R2   finished (exit $RC)"
[ $RC -ne 0 ] && FAIL=1

wait $PID_SWINUNETR
RC=$?
echo "[$(date '+%H:%M:%S')] SwinUNETR R2  finished (exit $RC)"
[ $RC -ne 0 ] && FAIL=1

wait $PID_SEGRESNET_WIDE
RC=$?
echo "[$(date '+%H:%M:%S')] SegResNet-Wide R2 finished (exit $RC)"
[ $RC -ne 0 ] && FAIL=1

# --- Cleanup ---
kill $MONITOR_PID 2>/dev/null

echo ""
echo "============================================================"
echo " Round 2 training complete: $(date)"
echo " Any failures: $( [ $FAIL -eq 0 ] && echo 'NO' || echo 'YES — check per-model logs' )"
echo "============================================================"

exit $FAIL
