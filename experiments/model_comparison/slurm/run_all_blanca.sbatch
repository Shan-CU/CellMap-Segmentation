#!/bin/bash
#SBATCH --job-name=cellmap_all_models_blanca
#SBATCH --partition=blanca-biokem
#SBATCH --account=blanca-biokem
#SBATCH --qos=blanca-biokem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=logs/comparison_blanca_%j.out
#SBATCH --error=logs/comparison_blanca_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@colorado.edu

# ============================================================
# Model Comparison Experiment - BLANCA Cluster
# ============================================================
# Trains all model architectures sequentially for comparison:
# - 2D Models: UNet, ResNet, Swin Transformer, ViT
# - 3D Models: UNet, ResNet, ViT-V-Net, Swin Transformer
#
# Run with: sbatch run_all_blanca.sbatch
# ============================================================

echo "=============================================="
echo "CellMap Model Comparison - BLANCA A100"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

# Load modules
module purge
module load anaconda 2>/dev/null || true

# Activate conda environment - handle different Blanca configurations
if ! command -v conda >/dev/null 2>&1; then
    if [ -f "$HOME/software/anaconda/etc/profile.d/conda.sh" ]; then
        source "$HOME/software/anaconda/etc/profile.d/conda.sh"
    elif [ -f "$HOME/software/anaconda/bin/activate" ]; then
        source "$HOME/software/anaconda/bin/activate"
    fi
fi

conda activate /projects/$USER/software/anaconda/envs/cellmap 2>/dev/null || \
    conda activate cellmap 2>/dev/null || \
    source activate cellmap

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null)"
echo "Number of GPUs: $(nvidia-smi -L | wc -l)"

# Environment settings
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Get number of GPUs
N_GPUS=$(nvidia-smi -L | wc -l)
echo "Using $N_GPUS GPUs"

# ============================================================
# Training Configuration
# ============================================================
# Blanca typically has fewer GPUs, so adjust epochs accordingly
EPOCHS_2D=100
EPOCHS_3D=50

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
    --model unet --dim 2d --epochs $EPOCHS_2D --save_features --vis_every 10
UNET2D_EXIT=$?
echo "UNet 2D finished with exit code: $UNET2D_EXIT"

# 2D ResNet
echo ""
echo "--- Training ResNet 2D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model resnet --dim 2d --epochs $EPOCHS_2D --save_features --vis_every 10
RESNET2D_EXIT=$?
echo "ResNet 2D finished with exit code: $RESNET2D_EXIT"

# 2D Swin Transformer
echo ""
echo "--- Training Swin Transformer 2D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model swin --dim 2d --epochs $EPOCHS_2D --save_features --vis_every 10
SWIN2D_EXIT=$?
echo "Swin 2D finished with exit code: $SWIN2D_EXIT"

# 2D ViT
echo ""
echo "--- Training ViT 2D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model vit --dim 2d --epochs $EPOCHS_2D --save_features --vis_every 10
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
    --model unet --dim 3d --epochs $EPOCHS_3D --save_features --vis_every 5
UNET3D_EXIT=$?
echo "UNet 3D finished with exit code: $UNET3D_EXIT"

# 3D ResNet
echo ""
echo "--- Training ResNet 3D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model resnet --dim 3d --epochs $EPOCHS_3D --save_features --vis_every 5
RESNET3D_EXIT=$?
echo "ResNet 3D finished with exit code: $RESNET3D_EXIT"

# 3D ViT-V-Net
echo ""
echo "--- Training ViT-V-Net 3D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model vit --dim 3d --epochs $EPOCHS_3D --save_features --vis_every 5
VIT3D_EXIT=$?
echo "ViT 3D finished with exit code: $VIT3D_EXIT"

# 3D Swin Transformer
echo ""
echo "--- Training Swin Transformer 3D ---"
torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model swin --dim 3d --epochs $EPOCHS_3D --save_features --vis_every 5
SWIN3D_EXIT=$?
echo "Swin 3D finished with exit code: $SWIN3D_EXIT"

# ============================================================
# Generate Visualizations and Analysis
# ============================================================
echo ""
echo "=============================================="
echo "Generating Visualizations and Analysis"
echo "=============================================="

cd experiments/model_comparison

python visualize_comparison.py \
    --metrics_dir metrics \
    --output figures

python analyze_results.py \
    --metrics_dir metrics \
    --output analysis

# ============================================================
# Cleanup and Summary
# ============================================================

echo ""
echo "=============================================="
echo "Training Summary"
echo "=============================================="
echo "UNet 2D:   Exit code $UNET2D_EXIT"
echo "ResNet 2D: Exit code $RESNET2D_EXIT"
echo "Swin 2D:   Exit code $SWIN2D_EXIT"
echo "ViT 2D:    Exit code $VIT2D_EXIT"
echo "UNet 3D:   Exit code $UNET3D_EXIT"
echo "ResNet 3D: Exit code $RESNET3D_EXIT"
echo "ViT 3D:    Exit code $VIT3D_EXIT"
echo "Swin 3D:   Exit code $SWIN3D_EXIT"
echo ""
echo "End time: $(date)"
echo "=============================================="

# Stop TensorBoard
kill $TB_PID 2>/dev/null || true

# Return overall status
if [ $UNET2D_EXIT -ne 0 ] || [ $RESNET2D_EXIT -ne 0 ] || [ $SWIN2D_EXIT -ne 0 ] || [ $VIT2D_EXIT -ne 0 ] || \
   [ $UNET3D_EXIT -ne 0 ] || [ $RESNET3D_EXIT -ne 0 ] || [ $VIT3D_EXIT -ne 0 ] || [ $SWIN3D_EXIT -ne 0 ]; then
    echo "Some models failed training!"
    exit 1
fi

exit 0
