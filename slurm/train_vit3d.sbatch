#!/bin/bash

#SBATCH --job-name=vit3d_baseline
#SBATCH --partition=aa100
#SBATCH --account=ucb-general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:3
#SBATCH --mem=235G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vit3d_%j.out
#SBATCH --error=logs/vit3d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# CellMap ViT-V-Net 3D Baseline Training on Alpine (24-HOUR OPTIMIZED)
# Hardware: aa100 partition - 3x NVIDIA A100 80GB, 64 cores, 243GB RAM
# Resource Request: 3 GPUs, 32 CPUs, 192GB RAM
# Optimizations:
#   - 3 GPUs with DistributedDataParallel (DDP) for ~3x throughput
#   - 3D Vision Transformer for volumetric context
#   - Mixed precision (AMP) essential for 3D ViT memory
#   - Gradient accumulation for large effective batch size
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
source /curc/sw/anaconda3/2023.09/etc/profile.d/conda.sh

# Activate conda environment
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

# Create directories
mkdir -p logs tensorboard checkpoints

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

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
echo "Starting ViT-V-Net 3D training with DDP (torchrun)..."
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $N_GPUS GPUs, launching $N_GPUS processes with torchrun"
torchrun --standalone --nproc_per_node=$N_GPUS examples/train_vit3d_baseline.py
TRAIN_EXIT_CODE=$?

# Cleanup
echo "Stopping TensorBoard..."
kill $TB_PID 2>/dev/null

echo "=========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
