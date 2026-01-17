#!/bin/bash

#SBATCH --job-name=unet_baseline
#SBATCH --partition=aa100
#SBATCH --account=ucb-general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/unet_%j.out
#SBATCH --error=logs/unet_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# CellMap UNet Baseline Training on Alpine (24-HOUR OPTIMIZED)
# Hardware: aa100 partition - 3x NVIDIA A100 80GB, 64 cores, 243GB RAM
# Resource Request: 3 GPUs, 32 CPUs, 192GB RAM
# Optimizations:
#   - 3 GPUs with DistributedDataParallel (DDP) for ~3x throughput
#   - DDP is preferred over DataParallel for better scaling
#   - 32 CPU cores for parallel data loading
#   - Local scratch ($SLURM_SCRATCH) for fastest I/O
#   - Mixed precision (AMP) for TensorCore acceleration
#   - Optimized for full 24-hour utilization
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Go to project directory
cd $SLURM_SUBMIT_DIR

# Load modules - use source for conda initialization
module purge
source /curc/sw/anaconda3/latest/etc/profile.d/conda.sh

# Activate your conda environment
# The environment should be at /projects/$USER/software/anaconda/envs/cellmap
# If this fails, check that the environment exists with: ls /projects/$USER/software/anaconda/envs/
conda activate /projects/$USER/software/anaconda/envs/cellmap || {
    echo "ERROR: Failed to activate conda environment"
    echo "Available envs: $(ls /projects/$USER/software/anaconda/envs/ 2>/dev/null || echo 'none found')"
    exit 1
}

# Verify GPU access
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
for i in $(seq 0 2); do
    echo "GPU $i: $(python -c "import torch; print(torch.cuda.get_device_name($i) if torch.cuda.device_count() > $i else 'N/A')")"
done

# Create logs directory if it doesn't exist
mkdir -p logs tensorboard checkpoints

# ============================================================
# ALPINE OPTIMIZATION: Stage data to local scratch for fastest I/O
# $SLURM_SCRATCH is a ~300GB local SSD - MUCH faster than network storage
# ============================================================
echo "Setting up local scratch for fast I/O..."
echo "Local scratch: $SLURM_SCRATCH"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Note: expandable_segments:True causes CUDA IPC errors on Alpine kernel
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# If data is on network storage, copy to local scratch
# Uncomment and modify the following if you want to stage data:
# DATA_SOURCE="/scratch/alpine/$USER/cellmap_data"
# if [ -d "$DATA_SOURCE" ]; then
#     echo "Staging data to local scratch..."
#     rsync -a --info=progress2 "$DATA_SOURCE" "$SLURM_SCRATCH/"
#     export CELLMAP_DATA_DIR="$SLURM_SCRATCH/cellmap_data"
#     echo "Data staged to: $CELLMAP_DATA_DIR"
# fi

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
echo "  ssh -L $TB_PORT:$SLURM_NODELIST:$TB_PORT george.stephenson@login.rc.colorado.edu"
echo ""
echo "Then open in browser: http://localhost:$TB_PORT"
echo "=========================================="

# Start TensorBoard in background
tensorboard --logdir=$TB_LOGDIR --port=$TB_PORT --bind_all &
TB_PID=$!
echo "TensorBoard PID: $TB_PID"

# Run training with torchrun for DDP
# torchrun handles process spawning and environment variable setup for DDP
echo "Starting UNet training with DDP (torchrun)..."
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $N_GPUS GPUs, launching $N_GPUS processes with torchrun"
torchrun --standalone --nproc_per_node=$N_GPUS examples/train_unet_baseline.py
TRAIN_EXIT_CODE=$?

# Cleanup TensorBoard
echo "Stopping TensorBoard..."
kill $TB_PID 2>/dev/null

echo "=========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
