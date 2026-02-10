#!/bin/bash
#SBATCH --job-name=auto3dseg_analyze
#SBATCH --partition=batch
#SBATCH --account=rc_alain_pi
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/auto3dseg_analyze_%j.out
#SBATCH --error=logs/auto3dseg_analyze_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=gsgeorge@unc.edu

# ============================================================
# Step 2: Auto3DSeg Data Analysis
#
# Runs MONAI DataAnalyzer to compute comprehensive statistics
# about your CellMap dataset:
#   - Image intensity distributions
#   - Shape and spacing statistics
#   - Label class distributions and class imbalance
#   - Foreground/background ratios
#
# Output: auto3dseg/work_dir/datastats.yaml
# Time: ~10-30 minutes with 1 GPU
# ============================================================

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
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

echo "Python: $(which python)"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import monai; print(f'MONAI {monai.__version__}')"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

mkdir -p logs auto3dseg/work_dir

# --- Verify datalist exists ---
DATALIST="auto3dseg/nifti_data/datalist.json"
if [ ! -f "$DATALIST" ]; then
    echo "ERROR: $DATALIST not found!"
    echo "Run the conversion step first: sbatch auto3dseg/auto3dseg_convert.sbatch"
    exit 1
fi

echo "Datalist contents:"
python -c "import json; dl=json.load(open('$DATALIST')); print(f'  Training: {len(dl.get(\"training\",[]))} samples'); print(f'  Validation: {len(dl.get(\"validation\",[]))} samples')"

# --- Run data analysis ---
python auto3dseg/run_auto3dseg.py \
    --mode analyze \
    --datalist "$DATALIST" \
    --work_dir auto3dseg/work_dir \
    --device cpu \
    --workers 4

echo ""
echo "Analysis complete at $(date)"
echo "Review results: cat auto3dseg/work_dir/datastats.yaml"
echo ""
echo "Next step (generate algorithm configs):"
echo "  python auto3dseg/run_auto3dseg.py --mode generate \\"
echo "      --datalist $DATALIST --work_dir auto3dseg/work_dir"
echo ""
echo "Or run full pipeline:"
echo "  sbatch auto3dseg/auto3dseg_train.sbatch"
