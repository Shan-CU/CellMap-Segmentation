#!/bin/bash
#SBATCH --job-name=unet_2d_l40
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=512g
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/l40s_comparison/slurm/logs/unet_2d_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/l40s_comparison/slurm/logs/unet_2d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# UNet 2D — L40S Model Comparison
# ============================================================
# Loss: Balanced Softmax Tversky (τ=1.0, α=0.6, β=0.4)
# Batch: 26/GPU × 2 GPUs × 4 accum = 208 effective
# ============================================================

MODEL="unet"
DIM="2d"
EPOCHS=200
BATCH_SIZE=26
ITERATIONS_PER_EPOCH=1000
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION=4
NUM_WORKERS=8
N_GPUS=2

echo "=============================================="
echo "L40S Model Comparison — UNet 2D"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: ${N_GPUS}x L40S 48GB"
echo "Start: $(date)"
echo "Effective batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION * N_GPUS))"
echo "=============================================="

PROJECT_ROOT=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
cd $PROJECT_ROOT

mkdir -p experiments/l40s_comparison/{checkpoints,tensorboard,visualizations,metrics,slurm/logs}

module purge
module load cuda/12.2

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'

echo "Python: $($MAMBA_EXE run -n csc which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / 2))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.7"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

echo ""
echo "Starting training..."
echo ""

$MAMBA_EXE run -n csc torchrun --standalone --nproc_per_node=$N_GPUS \
    experiments/l40s_comparison/train.py \
    --model $MODEL \
    --dim $DIM \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --iterations_per_epoch $ITERATIONS_PER_EPOCH \
    --lr $LEARNING_RATE \
    --gradient_accumulation $GRADIENT_ACCUMULATION \
    --num_workers $NUM_WORKERS \
    --pin_memory \
    --persistent_workers \
    --amp \
    --compile \
    --vis_every 10 \
    --checkpoint_every 25

EXIT_CODE=$?
echo "=============================================="
echo "Complete — Exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=============================================="
exit $EXIT_CODE
