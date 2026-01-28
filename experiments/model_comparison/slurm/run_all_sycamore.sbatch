#!/bin/bash
#SBATCH --job-name=cellmap_all_sycamore
#SBATCH --partition=h100_sn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH --mem=640G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/comparison_sycamore_%j.out
#SBATCH --error=logs/comparison_sycamore_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@unc.edu

# ============================================================
# Model Comparison Experiment - UNC Sycamore Cluster
# ============================================================
# Trains all model architectures SEQUENTIALLY for comparison:
# - 2D Models: UNet, ResNet, Swin Transformer, ViT
# - 3D Models: UNet, ResNet, Swin Transformer, ViT
#
# NOTE: This runs models one at a time. For parallel training,
# use submit_all_sycamore.sh instead.
#
# Run with: sbatch run_all_sycamore.sbatch
# ============================================================

echo "=============================================="
echo "CellMap Model Comparison - Sycamore H100"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 4x H100 80GB"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=============================================="

# Navigate to project root
cd $SLURM_SUBMIT_DIR/../../..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

# Create necessary directories
mkdir -p logs
mkdir -p experiments/model_comparison/results
mkdir -p experiments/model_comparison/checkpoints
mkdir -p experiments/model_comparison/tensorboard
mkdir -p experiments/model_comparison/visualizations
mkdir -p experiments/model_comparison/metrics

# Sycamore Environment Setup (micromamba)
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# H100 Environment settings
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 4))
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export NCCL_P2P_LEVEL=NVL

N_GPUS=4
echo "Using $N_GPUS GPUs"

# Training epochs (optimized for Sycamore)
EPOCHS_2D=200
EPOCHS_3D=150
ITERS_2D=1000
ITERS_3D=500

# Start TensorBoard in background
TB_PORT=$((6006 + SLURM_JOB_ID % 1000))
tensorboard --logdir=experiments/model_comparison/tensorboard --port=$TB_PORT --bind_all &
TB_PID=$!
echo "TensorBoard started on port $TB_PORT (PID: $TB_PID)"

# ============================================================
# 2D Models Training
# ============================================================
echo ""
echo "=============================================="
echo "Training 2D Models"
echo "=============================================="

# 2D UNet
echo ""
echo "--- Training UNet 2D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model unet --dim 2d --epochs $EPOCHS_2D \
    --batch_size 48 --iterations_per_epoch $ITERS_2D \
    --save_features --vis_every 10
UNET2D_EXIT=$?
echo "UNet 2D finished with exit code: $UNET2D_EXIT"

# 2D ResNet
echo ""
echo "--- Training ResNet 2D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model resnet --dim 2d --epochs $EPOCHS_2D \
    --batch_size 48 --iterations_per_epoch $ITERS_2D \
    --save_features --vis_every 10
RESNET2D_EXIT=$?
echo "ResNet 2D finished with exit code: $RESNET2D_EXIT"

# 2D Swin Transformer
echo ""
echo "--- Training Swin Transformer 2D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model swin --dim 2d --epochs $EPOCHS_2D \
    --batch_size 40 --lr 5e-5 --iterations_per_epoch $ITERS_2D \
    --save_features --vis_every 10
SWIN2D_EXIT=$?
echo "Swin 2D finished with exit code: $SWIN2D_EXIT"

# 2D ViT
echo ""
echo "--- Training ViT 2D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model vit --dim 2d --epochs $EPOCHS_2D \
    --batch_size 40 --lr 5e-5 --iterations_per_epoch $ITERS_2D \
    --save_features --vis_every 10
VIT2D_EXIT=$?
echo "ViT 2D finished with exit code: $VIT2D_EXIT"

# ============================================================
# 3D Models Training
# ============================================================
echo ""
echo "=============================================="
echo "Training 3D Models"
echo "=============================================="

# 3D UNet
echo ""
echo "--- Training UNet 3D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model unet --dim 3d --epochs $EPOCHS_3D \
    --batch_size 8 --iterations_per_epoch $ITERS_3D \
    --save_features --vis_every 5
UNET3D_EXIT=$?
echo "UNet 3D finished with exit code: $UNET3D_EXIT"

# 3D ResNet
echo ""
echo "--- Training ResNet 3D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model resnet --dim 3d --epochs $EPOCHS_3D \
    --batch_size 6 --iterations_per_epoch $ITERS_3D \
    --save_features --vis_every 5
RESNET3D_EXIT=$?
echo "ResNet 3D finished with exit code: $RESNET3D_EXIT"

# 3D Swin Transformer
echo ""
echo "--- Training Swin Transformer 3D ---"
export GRADIENT_CHECKPOINTING=1
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model swin --dim 3d --epochs $EPOCHS_3D \
    --batch_size 3 --lr 3e-5 --iterations_per_epoch $ITERS_3D \
    --save_features --vis_every 5
SWIN3D_EXIT=$?
echo "Swin 3D finished with exit code: $SWIN3D_EXIT"

# 3D ViT
echo ""
echo "--- Training ViT 3D ---"
export GRADIENT_CHECKPOINTING=1
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model vit --dim 3d --epochs $EPOCHS_3D \
    --batch_size 2 --lr 3e-5 --iterations_per_epoch $ITERS_3D \
    --save_features --vis_every 5
VIT3D_EXIT=$?
echo "ViT 3D finished with exit code: $VIT3D_EXIT"

# ============================================================
# Summary
# ============================================================
echo ""
echo "=============================================="
echo "Training Complete - Summary"
echo "=============================================="
echo "UNet 2D:   Exit $UNET2D_EXIT"
echo "ResNet 2D: Exit $RESNET2D_EXIT"
echo "Swin 2D:   Exit $SWIN2D_EXIT"
echo "ViT 2D:    Exit $VIT2D_EXIT"
echo "UNet 3D:   Exit $UNET3D_EXIT"
echo "ResNet 3D: Exit $RESNET3D_EXIT"
echo "Swin 3D:   Exit $SWIN3D_EXIT"
echo "ViT 3D:    Exit $VIT3D_EXIT"
echo "=============================================="
echo "End time: $(date)"

# Cleanup TensorBoard
kill $TB_PID 2>/dev/null || true

# Return non-zero if any model failed
if [ $UNET2D_EXIT -ne 0 ] || [ $RESNET2D_EXIT -ne 0 ] || [ $SWIN2D_EXIT -ne 0 ] || [ $VIT2D_EXIT -ne 0 ] || \
   [ $UNET3D_EXIT -ne 0 ] || [ $RESNET3D_EXIT -ne 0 ] || [ $SWIN3D_EXIT -ne 0 ] || [ $VIT3D_EXIT -ne 0 ]; then
    echo "WARNING: One or more models failed!"
    exit 1
fi

exit 0
