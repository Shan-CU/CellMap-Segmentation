#!/bin/bash
#SBATCH --job-name=gen_dints
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/logs/gen_dints_%j.log

# Quick job: generate DiNTS bundle (needs GPU for auto_scale memory detection)

export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo ""
echo "Generating DiNTS bundle..."
python3 auto3dseg/run_auto3dseg.py --mode generate \
  --datalist auto3dseg/nifti_data/datalist.json \
  --work_dir auto3dseg/work_dir \
  --algos dints \
  --num_fold 1

echo ""
echo "Done! Listing generated bundles:"
ls -la auto3dseg/work_dir/dints_0/ 2>/dev/null || echo "dints_0 not found"
