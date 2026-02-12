#!/bin/bash
#SBATCH --job-name=auto3dseg_train
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_l40s:2
#SBATCH --mem=350G
#SBATCH --time=11-00:00:00
#SBATCH --output=logs/auto3dseg_train_%j.out
#SBATCH --error=logs/auto3dseg_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@unc.edu

# ============================================================
# Step 3: Auto3DSeg Training (Train Only — patched bundles)
#
# Runs MONAI Auto3DSeg in train-only mode:
#   - Skips data analysis (already completed)
#   - Skips algorithm bundle generation (already completed & patched)
#   - Trains all 3 algorithms sequentially using patched bundles
#     with PartialAnnotationLossV2 (sigmoid per-class, masked by
#     annotated_classes per crop)
#
# The bundles have been patched with:
#   - PartialAnnotationLossV2: per-channel Dice+BCE, masked by
#     which classes each crop annotates (sigmoid mode)
#   - CropForegroundd removed (FIB-SEM crops are pre-cropped)
#   - target.float() cast for BCE compatibility
#   - Sub-batch mask slicing for swinunetr/dints patch permutation
#
# Hardware: 2x L40S (48GB each), 16 CPUs, 350GB RAM
# Walltime: 11 days (maximum for l40-gpu partition)
#
# RAM note: Reduced num_workers from 8→2 in all algorithm configs
# to keep peak RAM under 350G. With 2 workers × 2 prefetch × 2 DDP
# ranks = 8 images in flight. Worst-case 8 largest training images
# total ~228 GB after 15-channel label expansion in DiNTS/SwinUNETR.
# Plus ~80G overhead = ~310G peak. 350G gives safe margin.
# Previous attempts with 8 workers OOM'd at 512G (job 30343731).
#
# Uses --mode train which sets analyze=False, algo_gen=False,
# train=True, ensemble=False — so patched bundles are preserved.
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"

PROJECT_DIR=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
cd $PROJECT_DIR

# --- Environment setup ---
module purge

# Activate micromamba environment
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

# Fix: Ensure conda's libstdc++ (which has CXXABI_1.3.15) takes priority
# over the system /lib64/libstdc++ (which doesn't). Required by DiNTS → optuna → sqlite3.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch; [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
python -c "import monai; print(f'MONAI {monai.__version__}')"

mkdir -p logs auto3dseg/work_dir

# --- Performance tuning ---
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.9"
export CUDA_LAUNCH_BLOCKING=0
export CUDA_MODULE_LOADING=LAZY
export NCCL_DEBUG=INFO
export NVIDIA_TF32_OVERRIDE=1
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCHINDUCTOR_CACHE_DIR="/tmp/torch_cache_${SLURM_JOB_ID}"

# --- Verify datalist ---
DATALIST="auto3dseg/nifti_data/datalist.json"
if [ ! -f "$DATALIST" ]; then
    echo "ERROR: $DATALIST not found!"
    echo "Run the conversion step first: sbatch auto3dseg/auto3dseg_convert.sbatch"
    exit 1
fi

# --- Determine GPU count ---
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using $N_GPUS GPUs"

# ============================================================
# Configuration - ADJUST THESE FOR YOUR NEEDS
# ============================================================
# Algorithms to try: segresnet, swinunetr, dints, segresnet2d
# All 3 architectures: encoder-decoder, transformer, and NAS-discovered
ALGOS="segresnet swinunetr dints"

# Single fold since we have an explicit train/val split (not k-fold CV)
NUM_FOLD=1

# Let Auto3DSeg use its auto-computed epochs per algorithm:
#   SegResNet: 426, SwinUNETR: 354, DiNTS: 200 (+ NAS search)
# Override with a fixed value if desired (0 = use auto-computed):
NUM_EPOCHS=0

# ============================================================

echo ""
echo "Configuration:"
echo "  Algorithms: $ALGOS"
echo "  Folds:      $NUM_FOLD"
echo "  Epochs:     $NUM_EPOCHS (0 = use auto-computed per algorithm)"
echo "  GPUs:       $N_GPUS"
echo ""

# --- Build command ---
CMD="python auto3dseg/run_auto3dseg.py \
    --mode train \
    --datalist $DATALIST \
    --work_dir auto3dseg/work_dir \
    --algos $ALGOS \
    --num_fold $NUM_FOLD \
    --num_epochs $NUM_EPOCHS \
    --num_gpus $N_GPUS"

echo "Running: $CMD"
eval $CMD

echo ""
echo "Auto3DSeg training finished at $(date)"
echo "Results in: auto3dseg/work_dir/"
echo ""
echo "Key outputs to review:"
echo "  auto3dseg/work_dir/segresnet_0/model/   - SegResNet checkpoints"
echo "  auto3dseg/work_dir/swinunetr_0/model_fold0/ - SwinUNETR checkpoints"
echo "  auto3dseg/work_dir/dints_0/model_fold0/     - DiNTS checkpoints"
echo ""
echo "Next step: run ensemble or evaluate best model"
