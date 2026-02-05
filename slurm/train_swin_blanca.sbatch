#!/bin/bash
#SBATCH --job-name=swin_baseline_blanca
#SBATCH --partition=blanca-biokem
#SBATCH --account=blanca-biokem
#SBATCH --qos=blanca-biokem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/swin_blanca_%j.out
#SBATCH --error=logs/swin_blanca_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# CellMap SwinTransformer 2D Training - BLANCA BIOKEM OPTIMIZED
# Hardware: blanca-biokem A100 nodes (bgpu-biokem1/2)
# Resource Request: 2x A100 (80GB each), 32 CPUs, 200GB RAM
# Walltime: 7 days (maximum for priority queue)
# Optimizations:
#   - 2 GPUs with DistributedDataParallel (DDP)
#   - OMP_NUM_THREADS=4 to prevent thread exhaustion
#   - NUM_WORKERS=0 (optimal for zarr datasets)
#   - --no_compile to avoid stdatomic.h issues
#   - Mixed precision (AMP) for TensorCore acceleration
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

cd $SLURM_SUBMIT_DIR

# --- Load Blanca-specific Slurm and modules ---
module purge
module load slurm/blanca
module load anaconda 2>/dev/null || true

# If the module system doesn't expose Anaconda on this node, fall back to a user install.
if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/software/anaconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/software/anaconda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/software/anaconda/bin/activate" ]; then
        # Older installs sometimes only ship the activate script.
        source "$HOME/software/anaconda/bin/activate"
    fi
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found. Load a conda module or install conda under ~/software/anaconda." >&2
    exit 2
fi

# Activate conda environment. Update the path if you install elsewhere.
conda activate /projects/$USER/software/anaconda/envs/cellmap || {
    echo "WARNING: Failed to activate /projects/$USER/software/anaconda/envs/cellmap"
    echo "Attempting to activate by name: cellmap"
    conda activate cellmap || true
}

# Verify GPU access
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
for i in $(seq 0 1); do
    echo "GPU $i: $(python -c "import torch; print(torch.cuda.get_device_name($i) if torch.cuda.device_count() > $i else 'N/A')")"
done

mkdir -p logs tensorboard checkpoints

# Set environment variables for optimal performance
# CRITICAL FIX: OMP_NUM_THREADS must be low (4) to prevent thread exhaustion with DDP
# Using $SLURM_CPUS_PER_TASK (32) causes libgomp: Thread creation failed
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=$OMP_NUM_THREADS

# NUM_WORKERS=0 is optimal for zarr datasets - avoids DataLoader overhead
NUM_WORKERS=0

# CRITICAL: garbage_collection_threshold=0.9 prevents memory fragmentation/leaks
# expandable_segments:False required for CUDA IPC compatibility with DDP
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False,max_split_size_mb:512,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
# Lazy module loading reduces initial memory overhead
export CUDA_MODULE_LOADING=LAZY
export NCCL_DEBUG=INFO
# A100/RTX optimizations
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
# Use job-specific cache directory to avoid conflicts
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

# Optional: stage data to local scratch for fastest I/O
# Blanca nodes have local scratch at $SLURM_SCRATCH (~400GB)
if [ -n "$SLURM_SCRATCH" ] && [ -d "/scratch/alpine/$USER/cellmap_data" ]; then
    echo "Staging data to local scratch for faster I/O..."
    rsync -a --info=progress2 "/scratch/alpine/$USER/cellmap_data" "$SLURM_SCRATCH/"
    export CELLMAP_DATA_DIR="$SLURM_SCRATCH/cellmap_data"
    echo "Data staged to: $CELLMAP_DATA_DIR"
fi

# ============================================================
# Start TensorBoard in background
# ============================================================
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
TB_LOGDIR="$SLURM_SUBMIT_DIR/tensorboard"

echo "=========================================="
echo "TENSORBOARD CONNECTION INFO"
echo "=========================================="
echo "TensorBoard starting on port: $TB_PORT"
echo "Node: $SLURM_NODELIST"
echo ""
echo "To connect, run this SSH tunnel from your LOCAL machine:"
echo "  ssh -L $TB_PORT:$SLURM_NODELIST:$TB_PORT gest9386@login.rc.colorado.edu"
echo ""
echo "Then open in browser: http://localhost:$TB_PORT"
echo "=========================================="

tensorboard --logdir=$TB_LOGDIR --port=$TB_PORT --bind_all &
TB_PID=$!
echo "TensorBoard PID: $TB_PID"

# Run training with torchrun for DDP
# torchrun handles process spawning and environment variable setup for DDP
echo "Starting SwinTransformer training with DDP (torchrun)..."
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $N_GPUS GPUs, launching $N_GPUS processes with torchrun"
echo "Using NUM_WORKERS=$NUM_WORKERS (0 is optimal for zarr datasets)"
torchrun --standalone --nproc_per_node=$N_GPUS experiments/model_comparison/train_comparison.py \
    --model_name swin_2d \
    --num_workers $NUM_WORKERS \
    --no_compile
TRAIN_EXIT_CODE=$?

# Cleanup TensorBoard
echo "Stopping TensorBoard..."
kill $TB_PID 2>/dev/null

echo "=========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
