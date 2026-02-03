#!/bin/bash
#SBATCH --job-name=unet3d_alpine_test
#SBATCH --partition=aa100
#SBATCH --account=ucb-general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/unet3d_alpine_test_%j.out
#SBATCH --error=logs/unet3d_alpine_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=george.stephenson@colorado.edu

# ============================================================
# CellMap UNet3D Alpine TEST - Resource Optimization Validation
# Hardware: Alpine aa100 partition (2x A100 80GB)
# Purpose: Test memory leak fixes and resource utilization
# Optimizations: All sycamore improvements applied
# ============================================================

echo "========================================"
echo "ALPINE UNET3D OPTIMIZATION TEST"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Hostname: $(hostname)"
echo "Working Directory: $(pwd)"
echo "========================================"

# Environment setup
source ~/.bashrc
conda activate cellmap

# Load required modules
module load cuda/12.1.1

# ============================================================
# CRITICAL MEMORY LEAK FIXES (from Sycamore H100 testing)
# ============================================================
# 1. Aggressive garbage collection - prevents memory fragmentation in 3D
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512,expandable_segments:True"

# 2. Lazy CUDA module loading - reduces memory overhead
export CUDA_MODULE_LOADING=LAZY

# ============================================================
# PERFORMANCE OPTIMIZATIONS
# ============================================================
# Enable TensorFloat-32 for A100 acceleration
export NVIDIA_TF32_OVERRIDE=1
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# cuDNN optimizations
export TORCH_CUDNN_V8_API_ENABLED=1
export CUDNN_BENCHMARK=1

# NCCL tuning for 3D DDP
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=NVL
export NCCL_DEBUG=INFO

# Thread control for 3D volume processing
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Job-specific cache to avoid conflicts
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_unet3d_${SLURM_JOB_ID}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

# ============================================================
# SYSTEM INFO
# ============================================================
echo ""
echo "Environment Variables:"
echo "  PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "  CUDA_MODULE_LOADING: $CUDA_MODULE_LOADING"
echo "  NVIDIA_TF32_OVERRIDE: $NVIDIA_TF32_OVERRIDE"
echo "  NCCL_P2P_LEVEL: $NCCL_P2P_LEVEL"
echo ""
echo "System Resources:"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $(free -h | grep Mem | awk '{print $2}')"
echo "  GPUs: $SLURM_GPUS"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""
echo "PyTorch Configuration:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
for i in $(seq 0 1); do
    echo "GPU $i: $(python -c "import torch; print(torch.cuda.get_device_name($i) if torch.cuda.device_count() > $i else 'N/A')")"
done
echo ""

# ============================================================
# DATA MANAGEMENT
# ============================================================
echo "Setting up data access..."
if [ -n "$SLURM_SCRATCH" ] && [ -d "/scratch/alpine/$USER/cellmap_data" ]; then
    echo "Copying data to local scratch: $SLURM_SCRATCH"
    rsync -a --info=progress2 "/scratch/alpine/$USER/cellmap_data" "$SLURM_SCRATCH/"
    DATA_SOURCE="$SLURM_SCRATCH/cellmap_data"
else
    echo "Using project storage data"
    DATA_SOURCE="/projects/$USER/cellmap_data"
fi
echo "Data source: $DATA_SOURCE"
echo ""

# ============================================================
# TRAINING LAUNCH
# ============================================================
echo "========================================"
echo "LAUNCHING UNET3D TRAINING (DDP with 2 GPUs)"
echo "========================================"
echo "Command: torchrun --nproc_per_node=2 examples/train_unet3d_baseline.py"
echo ""

cd /projects/$USER/CellMap-Segmentation

# Launch training with DDP
torchrun --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=29500 \
    examples/train_unet3d_baseline.py \
    --data_source "$DATA_SOURCE" \
    --checkpoint_dir checkpoints \
    --tensorboard_dir tensorboard/unet3d_alpine_test \
    --log_dir logs

TRAIN_EXIT_CODE=$?

echo ""
echo "========================================"
echo "TRAINING COMPLETED"
echo "========================================"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "End Time: $(date)"
echo ""

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
    echo ""
    echo "Check resource utilization:"
    echo "  nvidia-smi"
    echo "  Logs: logs/unet3d_alpine_test_${SLURM_JOB_ID}.out"
    echo "  TensorBoard: tensorboard/unet3d_alpine_test"
else
    echo "✗ Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Check logs for details"
fi

echo "========================================"
exit $TRAIN_EXIT_CODE
