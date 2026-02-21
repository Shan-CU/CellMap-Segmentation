#!/bin/bash
#SBATCH --job-name=fetch_data
#SBATCH --partition=batch
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=64
#SBATCH --output=/work/users/g/s/gsgeorge/cellmap/logs/fetch_data_%j.log

source ~/.bashrc
micromamba activate csc
cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

echo "Starting download at $(date)"
csc fetch-data --crops all --num-workers 64 --batch-size 512
echo "Download complete at $(date)"
