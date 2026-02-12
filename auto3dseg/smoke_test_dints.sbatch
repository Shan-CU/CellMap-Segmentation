#!/bin/bash
#SBATCH --job-name=smoke_dints
#SBATCH --partition=volta-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=logs/smoke_dints_%j.out
#SBATCH --error=logs/smoke_dints_%j.err

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"

echo "=== Smoke test: dints_0 ==="
echo "GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"
echo "VRAM: $(python -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory/1024**3:.0f}GB')")"

python auto3dseg/work_dir/dints_0/scripts/train.py run \
    --config_file "auto3dseg/work_dir/dints_0/configs/hyper_parameters.yaml,auto3dseg/work_dir/dints_0/configs/hyper_parameters_search.yaml,auto3dseg/work_dir/dints_0/configs/network.yaml,auto3dseg/work_dir/dints_0/configs/network_search.yaml,auto3dseg/work_dir/dints_0/configs/transforms_infer.yaml,auto3dseg/work_dir/dints_0/configs/transforms_train.yaml,auto3dseg/work_dir/dints_0/configs/transforms_validate.yaml" \
    --training#num_epochs 4 --training#num_epochs_per_validation 2
