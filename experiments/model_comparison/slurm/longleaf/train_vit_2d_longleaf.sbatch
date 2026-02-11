#!/bin/bash
#SBATCH --job-name=vit_2d_ll
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=128g
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/vit_2d_longleaf_%j.out
#SBATCH --error=logs/vit_2d_longleaf_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# ViT 2D Training - UNC Longleaf (A100 40GB)
# ============================================================
# attention_dropout=0.1 for transformer regularization
# ============================================================

MODEL="vit"
DIM="2d"

EPOCHS=200
BATCH_SIZE=16                    # Transformers need more VRAM
ITERATIONS_PER_EPOCH=1000
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION=4          # Effective batch = 16*4*2 = 128
NUM_WORKERS=8

echo "=============================================="
echo "CellMap Model Comparison - ViT 2D"
echo "=============================================="
echo "Cluster: UNC Longleaf (A100 40GB)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2x A100 40GB"
echo "Start time: $(date)"
echo "=============================================="

cd $SLURM_SUBMIT_DIR/../../../..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

mkdir -p logs
mkdir -p experiments/model_comparison/checkpoints
mkdir -p experiments/model_comparison/tensorboard
mkdir -p experiments/model_comparison/visualizations
mkdir -p experiments/model_comparison/slurm/longleaf/logs

module purge
module load cuda/12.2

# Use absolute paths for non-interactive batch environment
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
# Environment activation via micromamba run (works in non-interactive batch)
# Using micromamba run instead of activate for batch compatibility

echo "Python: $($MAMBA_EXE run -n csc which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

N_GPUS=2

$MAMBA_EXE run -n csc torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --iterations_per_epoch $ITERATIONS_PER_EPOCH \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --pin_memory \
    --amp \
    --compile \
    --save_features \
    --vis_every 10

EXIT_CODE=$?
echo "Training Complete - Exit code: $EXIT_CODE"
exit $EXIT_CODE
