#!/bin/bash
#SBATCH --job-name=auto3dseg_v2
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=200g
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/auto3dseg_train_%j.out
#SBATCH --error=logs/auto3dseg_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@unc.edu

# ============================================================
# Auto3DSeg v2 — Optimized Pipeline on Reserved Node (g181003)
#
# Fixes from v1:
#   - Outlier crop247 (8.2GB / 25.9GB uncompressed) REMOVED from datalist
#   - GPU optimization enabled (auto-tunes batch size per algo)
#   - Fresh DataAnalyzer run (stale datastats.yaml regenerated)
#   - Clean work_dir state (remove stale progress/cache)
#   - 2 GPUs for DDP training (downsized to free GPUs for Round 2)
#
# Pipeline:
#   Step 0: Re-analyze data (DataAnalyzer → fresh datastats.yaml)
#   Step 1: Generate bundles with GPU optimization (BundleGen)
#   Step 2: Patch bundles (PartialAnnotationLossV2 + fixes)
#   Step 3: Train all 3 algorithms sequentially via DDP
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

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

# Fix: Ensure conda's libstdc++ takes priority (required by DiNTS → optuna → sqlite3)
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
    exit 1
fi

echo "Datalist: $DATALIST"
python -c "import json; d=json.load(open('$DATALIST')); print(f'  {len(d[\"training\"])} train, {len(d.get(\"validation\",[]))} val')"

# Sanity check: outlier should be gone
python -c "
import json
dl = json.load(open('$DATALIST'))
for item in dl['training'] + dl.get('validation', []):
    if 'crop247' in item.get('image', ''):
        print('ERROR: outlier crop247 still in datalist!')
        exit(1)
print('  ✓ Outlier crop247 confirmed removed')
"
if [ $? -ne 0 ]; then exit 1; fi

# --- Determine GPU count ---
N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using $N_GPUS GPUs"

# ============================================================
# Configuration
# ============================================================
ALGOS="segresnet swinunetr dints"
NUM_FOLD=1
NUM_EPOCHS=0  # 0 = use auto-computed per algorithm

echo ""
echo "Configuration:"
echo "  Algorithms: $ALGOS"
echo "  Folds:      $NUM_FOLD"
echo "  Epochs:     $NUM_EPOCHS (0 = auto)"
echo "  GPUs:       $N_GPUS"
echo "  GPU Opt:    ENABLED (auto-tune batch sizes)"
echo ""

# ============================================================
# Step 0: Re-analyze data (fresh datastats.yaml)
# ============================================================
echo "============================================================"
echo "Step 0: Re-analyzing data (fresh datastats.yaml)..."
echo "============================================================"

# Remove stale datastats so DataAnalyzer re-scans
rm -f auto3dseg/work_dir/datastats.yaml
echo "  Removed stale datastats.yaml"

python auto3dseg/run_auto3dseg.py \
    --mode analyze \
    --datalist $DATALIST \
    --work_dir auto3dseg/work_dir \
    --device cuda \
    --workers 8

if [ $? -ne 0 ]; then
    echo "ERROR: Data analysis failed!"
    exit 1
fi

# Print summary of data analysis
python -c "
import yaml
with open('auto3dseg/work_dir/datastats.yaml') as f:
    stats = yaml.safe_load(f)
n_cases = len(stats.get('stats_by_cases', []))
summary = stats.get('stats_summary', {})
img_stats = summary.get('image_stats', {})
shape = img_stats.get('shape', {})
print(f'  Analyzed {n_cases} cases')
if shape:
    print(f'  Shape range: {shape.get(\"min\")} → {shape.get(\"max\")}')
    print(f'  Mean shape: {shape.get(\"mean\")}')
"

# ============================================================
# Step 1: Generate bundles with GPU optimization
# ============================================================
echo ""
echo "============================================================"
echo "Step 1: Generating algorithm bundles (GPU optimization ON)..."
echo "============================================================"

# Clear stale progress/cache so BundleGen creates fresh bundles
rm -f auto3dseg/work_dir/progress.yaml auto3dseg/work_dir/cache.yaml
echo "  Cleared stale progress/cache"

python auto3dseg/run_auto3dseg.py \
    --mode generate \
    --datalist $DATALIST \
    --work_dir auto3dseg/work_dir \
    --algos $ALGOS \
    --num_fold $NUM_FOLD \
    --num_gpus $N_GPUS

if [ $? -ne 0 ]; then
    echo "ERROR: Bundle generation failed!"
    exit 1
fi

# ============================================================
# Step 2: Patch bundles with PartialAnnotationLossV2
# ============================================================
echo ""
echo "============================================================"
echo "Step 2: Patching bundles with PartialAnnotationLossV2..."
echo "============================================================"
python auto3dseg/patch_templates.py --work_dir auto3dseg/work_dir

if [ $? -ne 0 ]; then
    echo "ERROR: Patching failed!"
    exit 1
fi

# Step 2b: Fix data_list_key / validation_key in all hyper_parameters.yaml files
# BundleGen sets data_list_key: null (expects fold-based splits), but our
# datalist uses explicit "training" and "validation" top-level keys.
echo ""
echo "Step 2b: Fixing data_list_key / validation_key in hyper_parameters configs..."
python -c "
import yaml, glob, os

for hp in glob.glob('auto3dseg/work_dir/*_0/configs/hyper_parameters*.yaml'):
    with open(hp) as f:
        cfg = yaml.safe_load(f)

    bundle_name = hp.split('/')[-3]  # e.g. segresnet_0, dints_0
    changed = False

    # --- Fix for dints / swinunetr: training#data_list_key ---
    if 'training' in cfg and isinstance(cfg['training'], dict):
        if cfg['training'].get('data_list_key') is None:
            cfg['training']['data_list_key'] = 'training'
            changed = True

    # --- Fix for dints NAS: searching#data_list_key ---
    if 'searching' in cfg and isinstance(cfg['searching'], dict):
        if cfg['searching'].get('data_list_key') is None:
            cfg['searching']['data_list_key'] = 'training'
            changed = True

    # --- Fix for dints / swinunetr: validate#data_list_key ---
    if 'validate' in cfg and isinstance(cfg['validate'], dict):
        if cfg['validate'].get('data_list_key') is None:
            cfg['validate']['data_list_key'] = 'validation'
            changed = True

    # --- Fix for segresnet: validation_key (top-level config) ---
    if 'segresnet' in bundle_name:
        if cfg.get('validation_key') is None:
            cfg['validation_key'] = 'validation'
            changed = True
            print(f'  Set validation_key=validation for {bundle_name}')

    if changed:
        with open(hp, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f'  Fixed: {hp}')
    else:
        print(f'  OK:    {hp}')
"

# Step 2c: Clear stale progress/cache so AutoRunner doesn't skip algorithms
echo ""
echo "Step 2c: Clearing stale progress cache..."
for f in auto3dseg/work_dir/progress.yaml auto3dseg/work_dir/cache.yaml; do
    if [ -f "$f" ]; then
        echo "  Removing $f"
        rm -f "$f"
    fi
done

# Step 2d: Print generated config summary for debugging
echo ""
echo "Step 2d: Config summary after generation + patching:"
python -c "
import yaml, glob

for hp in sorted(glob.glob('auto3dseg/work_dir/*_0/configs/hyper_parameters.yaml')):
    bundle = hp.split('/')[-3]
    with open(hp) as f:
        cfg = yaml.safe_load(f)
    
    # Extract key training params
    if 'training' in cfg and isinstance(cfg['training'], dict):
        t = cfg['training']
        print(f'{bundle}:')
        print(f'  num_epochs: {t.get(\"num_epochs\", \"N/A\")}')
        print(f'  num_images_per_batch: {t.get(\"num_images_per_batch\", \"N/A\")}')
        print(f'  num_sw_batch_size: {t.get(\"num_sw_batch_size\", \"N/A\")}')
        print(f'  data_list_key: {t.get(\"data_list_key\", \"N/A\")}')
    else:
        # segresnet uses different config structure
        print(f'{bundle}:')
        print(f'  num_epochs: {cfg.get(\"num_epochs\", \"N/A\")}')
        print(f'  batch_size: {cfg.get(\"batch_size\", \"N/A\")}')
        print(f'  num_sw_batch_size: {cfg.get(\"num_sw_batch_size\", \"N/A\")}')
        print(f'  validation_key: {cfg.get(\"validation_key\", \"N/A\")}')
    
    # Check roi_size
    roi = cfg.get('roi', cfg.get('roi_size', 'N/A'))
    if 'training' in cfg and isinstance(cfg['training'], dict):
        roi = cfg['training'].get('roi_size', roi)
    print(f'  roi_size: {roi}')
    print()
"

# ============================================================
# Step 3: Train (analyze=False, algo_gen=False to preserve patches)
# ============================================================
echo ""
echo "============================================================"
echo "Step 3: Training all algorithms..."
echo "============================================================"
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
