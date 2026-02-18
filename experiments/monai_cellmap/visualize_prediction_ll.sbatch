#!/bin/bash
#SBATCH --job-name=viz_pred
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --reservation=gsgeorge_9034
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/logs/viz_pred_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/logs/viz_pred_%j.err

# Activate environment (Longleaf paths)
module purge
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
echo "Python: $(which python)"

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/experiments/monai_cellmap

MODEL=swinunetr_r2

# Crop 1: HeLa-3 (21 GT classes, 400^3 — richest annotations)
echo "============================================"
echo "Crop 1: jrc_hela-3_crop34"
echo "============================================"
python visualize_prediction.py --model $MODEL --crop jrc_hela-3_crop34

# Crop 2: Mouse kidney (14 GT classes, 200^3 — small, fast)
echo "============================================"
echo "Crop 2: jrc_mus-kidney_crop164"
echo "============================================"
python visualize_prediction.py --model $MODEL --crop jrc_mus-kidney_crop164

# Crop 3: Fly VNC (15 GT classes, 400^3 — different tissue type)
echo "============================================"
echo "Crop 3: jrc_fly-vnc-1_crop174"
echo "============================================"
python visualize_prediction.py --model $MODEL --crop jrc_fly-vnc-1_crop174

echo ""
echo "All done! Output at:"
echo "  experiments/monai_cellmap/visualizations/"
ls -la visualizations/
