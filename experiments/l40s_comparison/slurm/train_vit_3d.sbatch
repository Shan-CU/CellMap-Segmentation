#!/bin/bash
#SBATCH --job-name=vit_3d_l40
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=512g
#SBATCH --time=7-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/l40s_comparison/slurm/logs/vit_3d_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/l40s_comparison/slurm/logs/vit_3d_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# ViT-V-Net 3D — L40S Model Comparison
# ============================================================
# Loss: Balanced Softmax Tversky (τ=1.0, α=0.6, β=0.4)
# Batch: 1/GPU × 2 GPUs × 8 accum = 16 effective
# Smallest batch size — most memory-hungry per-sample
# num_workers=0 REQUIRED for 3D
# ============================================================

MODEL="vit"
DIM="3d"
EPOCHS=200
BATCH_SIZE=1
ITERATIONS_PER_EPOCH=500
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION=8
NUM_WORKERS=0
N_GPUS=2

echo "=============================================="
echo "L40S Model Comparison — ViT-V-Net 3D"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: ${N_GPUS}x L40S 48GB"
echo "Start: $(date)"
echo "Effective batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION * N_GPUS))"
echo "num_workers: $NUM_WORKERS (0 required for 3D)"
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

# Fix glibc malloc arena fragmentation (google/tensorstore#223).
# Without this, RSS grows linearly because glibc retains freed memory
# in per-thread arenas (up to 8×N_CORES arenas by default).
export MALLOC_ARENA_MAX=1

# Bypass TensorStore — use dask/zarr backend to avoid unbounded RSS growth
# See: google/tensorstore#223, janelia-cellmap/cellmap-segmentation-challenge#183
export CELLMAP_DATA_BACKEND=dask

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
