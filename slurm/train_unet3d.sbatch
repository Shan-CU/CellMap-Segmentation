#!/bin/bash

#SBATCH --job-name=unet3d_baseline
#SBATCH --partition=aa100
#SBATCH --account=ucb-general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/unet3d_%j.out
#SBATCH --error=logs/unet3d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# CellMap UNet 3D Baseline Training on Alpine
# 3D model processes volumetric data (32×256×256 chunks)
# Uses more memory but captures 3D context of organelles
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

# Load modules
module purge
module load anaconda

# Activate conda environment
conda activate /projects/$USER/software/anaconda/envs/cellmap || {
    echo "ERROR: Failed to activate conda environment"
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

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

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

# Run training
echo "Starting UNet 3D training..."
python examples/train_unet3d_baseline.py
TRAIN_EXIT_CODE=$?

# Cleanup
echo "Stopping TensorBoard..."
kill $TB_PID 2>/dev/null

echo "=========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
