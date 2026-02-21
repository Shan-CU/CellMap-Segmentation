#!/bin/bash
#SBATCH --job-name=download_crops
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/logs/download_crops_%j.log

echo "Starting download of 15 missing crops..."
echo "Start time: $(date)"

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# Activate micromamba environment
eval "$(micromamba shell hook --shell bash)"
micromamba activate csc

# Download missing crops with high parallelism
# 4 from jrc_zf-cardiac-1: 378,379,380,381
# 11 from jrc_ut21-1413-003: 198,199,200,214,220,222,224,225,226,227,228
csc fetch-data --crops "378,379,380,381,198,199,200,214,220,222,224,225,226,227,228" \
    -d data/ \
    --batch-size 1024 \
    --num-workers 128

echo "Download complete!"
echo "End time: $(date)"

# Verify the download
echo "=== Verifying jrc_zf-cardiac-1 ==="
ls data/jrc_zf-cardiac-1/jrc_zf-cardiac-1.zarr/recon-1/labels/groundtruth/ 2>/dev/null || echo "Dataset not found"

echo "=== Verifying jrc_ut21-1413-003 crops ==="
ls data/jrc_ut21-1413-003/jrc_ut21-1413-003.zarr/recon-1/labels/groundtruth/ | wc -l
