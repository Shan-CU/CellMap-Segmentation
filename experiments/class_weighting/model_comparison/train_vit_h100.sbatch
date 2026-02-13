#!/bin/bash
#SBATCH --job-name=mc_vit_h100
#SBATCH --qos=preemptable
#SBATCH --partition=blanca-bortz
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:h100:3
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mc_vit_h100_%j.out
#SBATCH --error=logs/mc_vit_h100_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=george.stephenson@colorado.edu
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --open-mode=append

# ============================================================
# Model Comparison: ViT-V-Net 2D — Preemptable H100 (blanca-bortz)
# Loss: Balanced Softmax Tversky τ=1.0 (class_weighting winner)
# Batch: 36 (3× H100 80GB each, 12 per GPU)
# DDP: 3× H100 (80GB each = 240GB total GPU memory)
# Preemptable: 24h wall, auto-requeue + signal handler
# Note: ViT is most memory-intensive — if OOM, reduce to 30
# ============================================================

# ── Graceful preemption handler ───────────────────────────────
PREEMPTED=0
handle_preempt() {
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║  PREEMPTION SIGNAL RECEIVED                  ║"
    echo "║  Saving checkpoint and exiting gracefully... ║"
    echo "╚══════════════════════════════════════════════╝"
    PREEMPTED=1
    if [ -n "$TRAIN_PID" ]; then
        kill -TERM "$TRAIN_PID" 2>/dev/null
        wait "$TRAIN_PID" 2>/dev/null
    fi
    echo "Resubmitting job..."
    cd "$SLURM_SUBMIT_DIR"
    sbatch "$0"
    exit 0
}
trap 'handle_preempt' USR1

echo "╔══════════════════════════════════════════════╗"
echo "║  MODEL COMPARISON — ViT-V-Net 2D (H100×3)    ║"
echo "║  Loss: Balanced Softmax Tversky τ=1.0        ║"
echo "║  Mode: PREEMPTABLE (24h, auto-requeue)       ║"
echo "╚══════════════════════════════════════════════╝"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# ── Navigate to experiment directory ──────────────────────────
cd "$SLURM_SUBMIT_DIR" || exit 1
EXPERIMENT_DIR="$(dirname "$(readlink -f "$0")")"
cd "$EXPERIMENT_DIR" || exit 1
echo "Working directory: $(pwd)"

# ── Load modules and conda ───────────────────────────────────
module purge
module load slurm/blanca
module load anaconda 2>/dev/null || true

if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/software/anaconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/software/anaconda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/software/anaconda/bin/activate" ]; then
        source "$HOME/software/anaconda/bin/activate"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found." >&2
    exit 2
fi

conda activate /projects/$USER/software/anaconda/envs/cellmap || {
    echo "WARNING: Failed to activate cellmap env, trying by name"
    conda activate cellmap || { echo "FATAL: Cannot activate cellmap"; exit 2; }
}

# ── Verify GPU access ────────────────────────────────────────
echo ""
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "GPUs: $N_GPUS"
for i in $(seq 0 $((N_GPUS-1))); do
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name($i))")
    GPU_MEM=$(python -c "import torch; print(f'{torch.cuda.get_device_properties($i).total_mem / 1e9:.1f} GB')")
    echo "  GPU $i: $GPU_NAME ($GPU_MEM)"
done

# ── Create output directories ────────────────────────────────
mkdir -p logs checkpoints runs results visualizations metrics features

# ── Environment variables ─────────────────────────────────────
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

# ── TensorBoard ──────────────────────────────────────────────
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
echo ""
echo "=========================================="
echo "TENSORBOARD: port $TB_PORT"
echo "SSH tunnel:  ssh -L $TB_PORT:$SLURM_NODELIST:$TB_PORT gest9386@login.rc.colorado.edu"
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

# ── Training ─────────────────────────────────────────────────
echo ""
echo "Starting ViT-V-Net 2D training with DDP ($N_GPUS× H100 GPUs)..."
echo "Batch size: 36 (12 per GPU)"
torchrun --standalone --nproc_per_node=$N_GPUS train.py \
    --model vit \
    --batch_size 36 \
    --resume \
    --no_compile &
TRAIN_PID=$!
wait $TRAIN_PID
TRAIN_EXIT=$?

# ── Cleanup ──────────────────────────────────────────────────
kill $TB_PID 2>/dev/null

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT"
if [ $PREEMPTED -eq 1 ]; then
    echo "NOTE: Job was preempted — checkpoint saved, resubmitted"
fi
echo "=========================================="

exit $TRAIN_EXIT
