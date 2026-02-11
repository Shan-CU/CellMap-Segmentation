#!/bin/bash
#SBATCH --job-name=mc_vit_blanca
#SBATCH --partition=blanca-biokem
#SBATCH --account=blanca-biokem
#SBATCH --qos=blanca-biokem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/mc_vit_%j.out
#SBATCH --error=logs/mc_vit_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# Model Comparison: ViT-V-Net 2D — Blanca Biokem A100
# Loss: Balanced Softmax Tversky τ=1.0 (class_weighting winner)
# Batch: 24 (uniform across all models)
# DDP: 2× A100 (80GB each)
# Note: ViT is the most memory-intensive model
# ============================================================

echo "╔══════════════════════════════════════════════╗"
echo "║  MODEL COMPARISON — ViT-V-Net 2D             ║"
echo "║  Loss: Balanced Softmax Tversky τ=1.0        ║"
echo "╚══════════════════════════════════════════════╝"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

# ── Navigate to experiment directory ──────────────────────────
cd "$SLURM_SUBMIT_DIR" || exit 1
EXPERIMENT_DIR="$(dirname "$0")"
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
    echo "  GPU $i: $(python -c "import torch; print(torch.cuda.get_device_name($i))")"
done

# ── Create output directories ────────────────────────────────
mkdir -p logs checkpoints runs results visualizations metrics features

# ── Environment variables ─────────────────────────────────────
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,max_split_size_mb:512,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY
export NCCL_DEBUG=INFO
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

# ── Training ─────────────────────────────────────────────────
echo ""
echo "Starting ViT-V-Net 2D training with DDP ($N_GPUS GPUs)..."
torchrun --standalone --nproc_per_node=$N_GPUS train.py \
    --model vit \
    --resume \
    --no_compile
TRAIN_EXIT=$?

# ── Cleanup ──────────────────────────────────────────────────
kill $TB_PID 2>/dev/null

echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT"
echo "=========================================="

exit $TRAIN_EXIT
