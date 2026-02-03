#!/bin/bash
#SBATCH --job-name=fiji_check
#SBATCH --partition=nvidia-a100
#SBATCH --account=reguser
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=logs/fiji_check_%j.out
#SBATCH --error=logs/fiji_check_%j.err

echo "========================================"
echo "FIJI RESOURCE CHECK"
echo "========================================"
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo ""

echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,count --format=csv,noheader
echo ""

echo "CPU Info:"
lscpu | grep -E "^CPU\(s\)|Model name|Thread|Core"
echo ""

echo "Memory:"
free -h
echo ""

echo "Checking conda environment:"
source ~/.bashrc
conda activate cellmap
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
echo ""

echo "Checking data directories:"
ls -lh /projects/gest9386/cellmap_data/ | head -5 || echo "No data at /projects/gest9386/cellmap_data/"
ls -lh /projects/gest9386/CellMap-Segmentation/ | head -5 || echo "No project at /projects/gest9386/CellMap-Segmentation/"
