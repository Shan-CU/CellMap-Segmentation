#!/bin/bash
#SBATCH --job-name=test_syc_gpu
#SBATCH --partition=h100_sn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_sycamore_%j.out
#SBATCH --error=logs/test_sycamore_%j.err

# ============================================================
# Quick GPU Test for Sycamore H100
# ============================================================
# Tests all 8 models with a few iterations each to:
# 1. Verify environment and CUDA setup
# 2. Measure GPU memory usage per model
# 3. Identify scaling headroom
#
# Expected runtime: ~10-15 minutes
# ============================================================

echo "=============================================="
echo "Sycamore H100 GPU Test"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================="

cd $SLURM_SUBMIT_DIR/../../..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

mkdir -p logs
mkdir -p experiments/model_comparison/checkpoints
mkdir -p experiments/model_comparison/tensorboard

# Sycamore Environment Setup
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

echo ""
echo "=== Environment Check ==="
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA version: $(python -c 'import torch; print(torch.version.cuda)')"
echo "cuDNN version: $(python -c 'import torch; print(torch.backends.cudnn.version())')"
echo ""

echo "=== GPU Information ==="
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# H100 optimizations
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export NCCL_P2P_LEVEL=NVL

# Quick test function
test_model() {
    local MODEL=$1
    local DIM=$2
    local BATCH=$3
    local NGPUS=$4
    
    echo ""
    echo "=============================================="
    echo "Testing: $MODEL $DIM (batch=$BATCH, gpus=$NGPUS)"
    echo "=============================================="
    
    # Clear GPU memory
    python -c "import torch; torch.cuda.empty_cache()"
    
    # Run 10 iterations only (--debug mode)
    timeout 180 torchrun --standalone --nproc_per_node=$NGPUS \
        experiments/model_comparison/train_comparison.py \
        --model $MODEL \
        --dim $DIM \
        --epochs 1 \
        --batch_size $BATCH \
        --iterations_per_epoch 10 \
        --debug 2>&1 | head -100
    
    EXIT_CODE=$?
    
    # Report memory usage
    echo ""
    echo "--- GPU Memory After $MODEL $DIM ---"
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ $MODEL $DIM: PASSED"
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "⚠ $MODEL $DIM: TIMEOUT (but likely working)"
    else
        echo "✗ $MODEL $DIM: FAILED (exit code $EXIT_CODE)"
    fi
    
    return $EXIT_CODE
}

echo ""
echo "=============================================="
echo "PHASE 1: Testing 2D Models (2 GPUs)"
echo "=============================================="

# Test 2D models with planned batch sizes
test_model "unet" "2d" 48 2
UNET2D=$?

test_model "resnet" "2d" 48 2
RESNET2D=$?

test_model "swin" "2d" 40 2
SWIN2D=$?

test_model "vit" "2d" 40 2
VIT2D=$?

echo ""
echo "=============================================="
echo "PHASE 2: Testing 3D Models (4 GPUs)"
echo "=============================================="

# Test 3D models with planned batch sizes
test_model "unet" "3d" 8 4
UNET3D=$?

test_model "resnet" "3d" 6 4
RESNET3D=$?

test_model "swin" "3d" 3 4
SWIN3D=$?

test_model "vit" "3d" 2 4
VIT3D=$?

echo ""
echo "=============================================="
echo "TEST SUMMARY"
echo "=============================================="
echo "2D Models:"
echo "  UNet 2D (batch=48):   $([ $UNET2D -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  ResNet 2D (batch=48): $([ $RESNET2D -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  Swin 2D (batch=40):   $([ $SWIN2D -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  ViT 2D (batch=40):    $([ $VIT2D -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo ""
echo "3D Models:"
echo "  UNet 3D (batch=8):    $([ $UNET3D -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  ResNet 3D (batch=6):  $([ $RESNET3D -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  Swin 3D (batch=3):    $([ $SWIN3D -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo "  ViT 3D (batch=2):     $([ $VIT3D -eq 0 ] && echo '✓ PASS' || echo '✗ FAIL')"
echo ""
echo "=============================================="
echo "End time: $(date)"
echo "=============================================="

# Final GPU state
echo ""
echo "=== Final GPU Memory State ==="
nvidia-smi
