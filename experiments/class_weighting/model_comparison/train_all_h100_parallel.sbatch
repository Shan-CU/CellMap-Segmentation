#!/bin/bash
#SBATCH --job-name=mc_all_parallel
#SBATCH --qos=preemptable
#SBATCH --partition=blanca-bortz
#SBATCH --account=blanca-biokem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:3
#SBATCH --mem=750G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mc_all_parallel_%j.out
#SBATCH --error=logs/mc_all_parallel_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=shanmukhavamshi.kuruba@colorado.edu
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ============================================================
# All 4 Models — Parallel on 3× H100 (1 model per GPU)
#
# Strategy:
#   Phase 1: resnet(GPU0) + unet(GPU1) + swin(GPU2) in parallel
#   Phase 2: vit(GPU0) — runs on first available GPU after phase 1
#
# Each model gets a full H100 (100GB) — no DDP overhead.
# Batch size 128 for CNNs, 64 for ViT (single GPU, no splits).
# torch.compile disabled (stability issues with some ops).
# ============================================================

# ── Graceful preemption handler ───────────────────────────────
PREEMPTED=0
handle_preempt() {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  PREEMPTION SIGNAL RECEIVED                  ║"
    echo "║  Saving checkpoints and exiting gracefully...║"
    echo "╚══════════════════════════════════════════════╝"
    PREEMPTED=1
    # Kill all training processes — they save checkpoints on SIGTERM
    kill -TERM $PID0 $PID1 $PID2 $PID3 2>/dev/null
    wait $PID0 $PID1 $PID2 $PID3 2>/dev/null
    echo "Resubmitting job..."
    cd "$SLURM_SUBMIT_DIR"
    sbatch "${SLURM_SUBMIT_DIR}/$(basename "${BASH_SOURCE[0]:-$0}")"
    exit 0
}
trap 'handle_preempt' USR1

echo "╔══════════════════════════════════════════════╗"
echo "║  ALL MODELS — 3× H100 PARALLEL               ║"
echo "║  1 model per GPU, batch=128/64, no compile   ║"
echo "║  Phase 1: resnet + unet + swin                ║"
echo "║  Phase 2: vit                                  ║"
echo "╚══════════════════════════════════════════════╝"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# ── Navigate to experiment directory ──────────────────────────
cd "$SLURM_SUBMIT_DIR" || exit 1
echo "Working directory: $(pwd)"

# ── Load modules and conda ───────────────────────────────────
module purge
module load anaconda 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/software/anaconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/software/anaconda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/software/anaconda/bin/activate" ]; then
        source "$HOME/software/anaconda/bin/activate"
    fi
fi

conda activate /projects/$USER/software/anaconda/envs/cellmap || {
    echo "WARNING: Failed to activate cellmap env, trying by name"
    conda activate cellmap || { echo "FATAL: Cannot activate cellmap"; exit 2; }
}

# ── Verify GPU access ────────────────────────────────────────
echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs: $N_GPUS"
for i in $(seq 0 $((N_GPUS-1))); do
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name($i))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties($i).total_memory / 1e9:.1f} GB')")
    echo "  GPU $i: $GPU_NAME ($GPU_MEM)"
done

# ── Create output directories ────────────────────────────────
mkdir -p logs checkpoints runs results visualizations metrics features

# ── Environment variables ─────────────────────────────────────
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

# ── TensorBoard ──────────────────────────────────────────────
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
echo ""
echo "=========================================="
echo "TENSORBOARD: port $TB_PORT"
echo "SSH tunnel:  ssh -L $TB_PORT:$SLURM_NODELIST:$TB_PORT $USER@login.rc.colorado.edu"
echo "Browser:     http://localhost:$TB_PORT"
echo "=========================================="

tensorboard --logdir=runs --port=$TB_PORT --bind_all &
TB_PID=$!

# ── Compute class frequencies (if not already done) ─────────
if [ ! -f class_frequencies.json ]; then
    echo ""
    echo "Computing class frequencies for all 14 classes..."
    python compute_class_frequencies.py --n_batches 50
    echo "Class frequencies computed ✓"
else
    echo "class_frequencies.json exists ✓"
fi

# ── Training parameters ──────────────────────────────────────
BATCH=128  # Single H100 100GB — 128 for CNNs, 64 for ViT
BATCH_VIT=64
echo ""
echo "=========================================="
echo "  Single-GPU config: batch_size=$BATCH (ViT=$BATCH_VIT)"
echo "  torch.compile=OFF  (no DDP overhead)"
echo "=========================================="

# ── Helper function ───────────────────────────────────────────
run_model() {
    local MODEL=$1
    local GPU=$2
    local BS=$BATCH
    if [ "$MODEL" = "vit" ]; then
        BS=$BATCH_VIT
    fi
    local LOGFILE="logs/mc_${MODEL}_gpu${GPU}_${SLURM_JOB_ID}.log"

    echo "[$(date +%H:%M:%S)] Starting $MODEL on GPU $GPU (batch=$BS)"

    CUDA_VISIBLE_DEVICES=$GPU python train.py \
        --model $MODEL \
        --batch_size $BS \
        --resume \
        --no_compile \
        > "$LOGFILE" 2>&1

    local EXIT_CODE=$?
    echo "[$(date +%H:%M:%S)] $MODEL finished on GPU $GPU (exit=$EXIT_CODE)"
    return $EXIT_CODE
}

# ── Phase 1: 3 models in parallel (one per GPU) ──────────────
echo ""
echo "════════════════════════════════════════"
echo "  PHASE 1: resnet(GPU0) + unet(GPU1) + swin(GPU2)"
echo "════════════════════════════════════════"

run_model resnet 0 &
PID0=$!

run_model unet 1 &
PID1=$!

run_model swin 2 &
PID2=$!

# Wait for all 3 to finish
wait $PID0
EXIT0=$?
echo "  resnet exit: $EXIT0"

wait $PID1
EXIT1=$?
echo "  unet exit: $EXIT1"

wait $PID2
EXIT2=$?
echo "  swin exit: $EXIT2"

# ── Phase 2: vit on GPU 0 ────────────────────────────────────
echo ""
echo "════════════════════════════════════════"
echo "  PHASE 2: vit(GPU0)"
echo "════════════════════════════════════════"

run_model vit 0 &
PID3=$!
wait $PID3
EXIT3=$?
echo "  vit exit: $EXIT3"

# ── Cleanup ──────────────────────────────────────────────────
kill $TB_PID 2>/dev/null

echo ""
echo "=========================================="
echo "  ALL MODELS COMPLETE"
echo "  resnet: exit=$EXIT0"
echo "  unet:   exit=$EXIT1"
echo "  swin:   exit=$EXIT2"
echo "  vit:    exit=$EXIT3"
echo "  End time: $(date)"
if [ $PREEMPTED -eq 1 ]; then
    echo "  NOTE: Job was preempted — checkpoints saved, resubmitted"
fi
echo "=========================================="

# Exit with error if any model failed
[[ $EXIT0 -ne 0 || $EXIT1 -ne 0 || $EXIT2 -ne 0 || $EXIT3 -ne 0 ]] && exit 1
exit 0
