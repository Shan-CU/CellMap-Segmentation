#!/bin/bash
#SBATCH --job-name=viz_pred
#SBATCH --partition=h100_mn
#SBATCH --qos=h100_mn
#SBATCH --account=rc_alain_pi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/logs/viz_pred_%j.out
#SBATCH --error=/work/users/g/s/gsgeorge/cellmap/logs/viz_pred_%j.err

# Activate environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

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
