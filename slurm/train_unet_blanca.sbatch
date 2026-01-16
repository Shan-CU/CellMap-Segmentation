#!/bin/bash

#SBATCH --job-name=unet_blanca
#SBATCH --qos=preemptable
#SBATCH --account=ucb-general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/unet_blanca_%j.out
#SBATCH --error=logs/unet_blanca_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu
#SBATCH --export=NONE

# ============================================================
# CellMap UNet Training on Blanca BioKEM A100 Nodes
# Hardware: bgpu-biokem2 - 2x NVIDIA A100 GPUs
# Time limit: 7 days (Blanca allows longer jobs!)
# ============================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "=========================================="

# Go to project directory
cd /projects/$USER/CellMap-Segmentation

# Load modules
module purge
module load anaconda

# Activate your conda environment
conda activate /projects/$USER/software/anaconda/envs/cellmap || {
    echo "ERROR: Failed to activate conda environment"
    exit 1
}

# Verify GPU access
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
for i in 0 1; do
    echo "GPU $i: $(python -c "import torch; print(torch.cuda.get_device_name($i) if torch.cuda.device_count() > $i else 'N/A')")"
done

# Create directories
mkdir -p logs tensorboard checkpoints

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Note: expandable_segments:True causes CUDA IPC errors
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# ============================================================
# Start TensorBoard in background
# ============================================================
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
TB_LOGDIR="$PWD/tensorboard"

echo "=========================================="
echo "TENSORBOARD CONNECTION INFO"
echo "=========================================="
echo "TensorBoard starting on port: $TB_PORT"
echo "Node: $SLURM_NODELIST"
echo ""
echo "To connect, run this SSH tunnel from your LOCAL machine:"
echo "  ssh -L $TB_PORT:$SLURM_NODELIST:$TB_PORT $USER@login.rc.colorado.edu"
echo ""
echo "Then open in browser: http://localhost:$TB_PORT"
echo "=========================================="

tensorboard --logdir=$TB_LOGDIR --port=$TB_PORT --bind_all &
TB_PID=$!
echo "TensorBoard PID: $TB_PID"

# Run training (uses 2 GPUs instead of 3)
echo "Starting UNet training on Blanca BioKEM..."
python examples/train_unet_baseline.py
TRAIN_EXIT_CODE=$?

# Cleanup TensorBoard
kill $TB_PID 2>/dev/null

echo "=========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
