#!/bin/bash
#SBATCH --job-name=vit_3d_ll
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256g
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/vit_3d_longleaf_%j.out
#SBATCH --error=logs/vit_3d_longleaf_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# ViT 3D Training - UNC Longleaf (A100 40GB)
# ============================================================
# Memory-intensive: 3D transformer
# num_workers=0 REQUIRED
# attention_dropout=0.1
# ============================================================

MODEL="vit"
DIM="3d"

EPOCHS=200
BATCH_SIZE=1                     # Minimal for 3D transformer
ITERATIONS_PER_EPOCH=500
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION=16         # Effective batch = 1*16*3 = 48
NUM_WORKERS=0                    # CRITICAL for 3D

echo "=============================================="
echo "CellMap Model Comparison - ViT 3D"
echo "=============================================="
echo "Cluster: UNC Longleaf (A100 40GB)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 2x A100 40GB"
echo "Start time: $(date)"
echo "=============================================="

cd $SLURM_SUBMIT_DIR/../../..
PROJECT_ROOT=$(pwd)
echo "Project root: $PROJECT_ROOT"

mkdir -p logs
mkdir -p experiments/model_comparison/checkpoints
mkdir -p experiments/model_comparison/tensorboard
mkdir -p experiments/model_comparison/visualizations
mkdir -p experiments/model_comparison/slurm/longleaf/logs

module purge
module load cuda/12.2

# Add micromamba to PATH
export PATH="$HOME/.local/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.7"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

N_GPUS=2

torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/model_comparison/train_comparison.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --iterations_per_epoch $ITERATIONS_PER_EPOCH \
    --lr $LEARNING_RATE \
    --num_workers $NUM_WORKERS \
    --amp \
    --compile \
    --save_features \
    --vis_every 10

EXIT_CODE=$?
echo "Training Complete - Exit code: $EXIT_CODE"
exit $EXIT_CODE
