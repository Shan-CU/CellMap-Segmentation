#!/bin/bash
#SBATCH --job-name=swin3d_oom_demo
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=63
#SBATCH --gres=gpu:7
#SBATCH --mem=920g
#SBATCH --time=11-00:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/l40s_comparison/slurm/logs/swin_3d_reserved_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/l40s_comparison/slurm/logs/swin_3d_reserved_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@ad.unc.edu

# ============================================================
# SwinUNETR 3D — OOM Demonstration on Reserved Node
# ============================================================
# Purpose: Show coworker that even with 920 GB RAM + 7× L40S,
#          the host-memory leak from 792 CellMapDataset objects
#          + TensorStore/dask cross-thread fragmentation still
#          causes OOM.  With all mitigations applied (tcmalloc,
#          dask synchronous scheduler, MALLOC_ARENA_MAX=1),
#          RSS still grows ~3.6 GB/min → 920 GB in ~4 hours.
#
# Loss: Balanced Softmax Tversky (τ=1.0, α=0.6, β=0.4)
# Batch: 2/GPU × 7 GPUs × 12 accum = 168 effective
# num_workers=0 REQUIRED for 3D (avoids fork-based dataset duplication)
# ============================================================

MODEL="swin"
DIM="3d"
EPOCHS=200
BATCH_SIZE=2
ITERATIONS_PER_EPOCH=500
LEARNING_RATE=5e-5
GRADIENT_ACCUMULATION=12
NUM_WORKERS=0
N_GPUS=7

echo "=============================================="
echo "L40S OOM Demo — SwinUNETR 3D (Reserved Node)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: ${N_GPUS}x L40S 48GB"
echo "RAM: 920 GB allocated"
echo "Start: $(date)"
echo "Effective batch: $((BATCH_SIZE * GRADIENT_ACCUMULATION * N_GPUS))"
echo "num_workers: $NUM_WORKERS (0 required for 3D)"
echo ""
echo "This run will demonstrate host-RAM OOM even with"
echo "920 GB + all mitigations (tcmalloc, dask synchronous,"
echo "MALLOC_ARENA_MAX=1). Expect OOM in ~4 hours."
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

# ============================================================
# Memory leak mitigations (all applied, still insufficient)
# ============================================================
# 1. tcmalloc: replaces glibc malloc, TransferCache handles
#    cross-thread alloc/free without arena fragmentation
export LD_PRELOAD=/nas/longleaf/home/gsgeorge/micromamba/envs/csc/lib/libtcmalloc.so.4
# 2. Limit glibc arenas (belt-and-suspenders with tcmalloc)
export MALLOC_ARENA_MAX=1
# 3. Dask backend: bypass TensorStore entirely
export CELLMAP_DATA_BACKEND=dask

export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK / N_GPUS))
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64,garbage_collection_threshold:0.7"
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export NVIDIA_TF32_OVERRIDE=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

# Log RSS every 60s in background so coworker can see the leak
(while true; do
    echo "[RSS $(date +%H:%M:%S)] $(ps -o rss= -p $$ 2>/dev/null || echo 'N/A') KB | $(free -g | awk '/Mem:/{print "Used: "$3"/"$2" GB"}')"
    sleep 60
done) &
RSS_LOGGER_PID=$!

echo ""
echo "Starting training at $(date)..."
echo "RSS logger PID: $RSS_LOGGER_PID"
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
    --no_compile \
    --vis_every 10 \
    --checkpoint_every 25

EXIT_CODE=$?

# Kill RSS logger
kill $RSS_LOGGER_PID 2>/dev/null

echo "=============================================="
echo "Complete — Exit code: $EXIT_CODE"
echo "End: $(date)"
echo "=============================================="

# If OOM killed, the exit code will be 137 (128 + SIGKILL=9)
if [ $EXIT_CODE -eq 137 ]; then
    echo "*** KILLED BY OOM (exit 137) — as predicted ***"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "*** FAILED (exit $EXIT_CODE) ***"
fi

exit $EXIT_CODE
