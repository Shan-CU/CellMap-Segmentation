#!/bin/bash
#SBATCH --job-name=auto3dseg_bundlegen
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/auto3dseg_bundlegen_%j.out
#SBATCH --error=logs/auto3dseg_bundlegen_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@unc.edu

# ============================================================
# Step 2: Auto3DSeg Bundle Generation (all 3 algorithms)
#
# Generates SegResNet, SwinUNETR, and DiNTS bundles.
# Requires GPU for auto_scale() memory detection.
# Requires datastats.yaml from the analysis step.
#
# Output: auto3dseg/work_dir/{segresnet_0,swinunetr_0,dints_0}/
# Time: ~1-2 minutes total
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

PROJECT_DIR=/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
cd $PROJECT_DIR

# --- Environment setup ---
module purge

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB')"
python -c "import monai; print(f'MONAI {monai.__version__}')"

# --- Verify prerequisites ---
DATALIST="auto3dseg/nifti_data/datalist.json"
DATASTATS="auto3dseg/work_dir/datastats.yaml"

if [ ! -f "$DATALIST" ]; then
    echo "ERROR: $DATALIST not found!"
    exit 1
fi

if [ ! -f "$DATASTATS" ]; then
    echo "ERROR: $DATASTATS not found!"
    echo "Run the analysis step first: sbatch auto3dseg/auto3dseg_analyze.sbatch"
    exit 1
fi

echo ""
echo "Datalist:"
python -c "import json; dl=json.load(open('$DATALIST')); print(f'  Training: {len(dl.get(\"training\",[]))}  Validation: {len(dl.get(\"validation\",[]))}  Sigmoid: {dl.get(\"sigmoid\")}')"

# --- Generate all 3 algorithm bundles ---
echo ""
echo "Generating bundles: segresnet, swinunetr, dints..."
python3 auto3dseg/run_auto3dseg.py --mode generate \
    --datalist "$DATALIST" \
    --work_dir auto3dseg/work_dir \
    --algos segresnet swinunetr dints \
    --num_fold 1

echo ""
echo "Bundle generation complete at $(date)"
echo ""
echo "Generated bundles:"
for algo in segresnet_0 swinunetr_0 dints_0; do
    if [ -d "auto3dseg/work_dir/$algo" ]; then
        echo "  ✓ $algo"
    else
        echo "  ✗ $algo (MISSING)"
    fi
done

echo ""
echo "Next step: sbatch auto3dseg/auto3dseg_train.sbatch"
