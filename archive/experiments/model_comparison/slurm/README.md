# SLURM Job Scripts for Model Comparison

Optimized SLURM batch scripts for running the CellMap model comparison on **blanca-biokem** cluster with **2x A100 80GB GPUs**.

## Quick Start

```bash
# Submit all 8 jobs (4 2D + 4 3D models)
./submit_all.sh all

# Or submit by dimension
./submit_all.sh 2d   # Only 2D models
./submit_all.sh 3d   # Only 3D models

# Submit individual models
sbatch train_unet_2d.sbatch
sbatch train_swin_3d.sbatch
```

## Job Overview

### Resource Allocation (All Jobs)
- **Partition**: blanca-biokem
- **QoS**: blanca-biokem (high-priority)
- **GPUs**: 2x A100 80GB (`gres=gpu:a100:2`)
- **Memory**: 160 GB
- **CPUs**: 32 cores
- **Time Limit**: 2 days (48 hours)

### 2D Models

All 2D models share **iterations_per_epoch=500** for fair comparison.

| Model | Batch Size | Grad Accum | Effective Batch | Learning Rate | Est. Time |
|-------|------------|------------|-----------------|---------------|-----------|
| UNet 2D | 64 | 2 | 256 | 1e-4 | ~24-30h |
| ResNet 2D | 48 | 4 | 384 | 1e-4 | ~28-34h |
| Swin 2D | 24 | 8 | 384 | 5e-5 | ~36-40h |
| ViT 2D | 16 | 8 | 256 | 5e-5 | ~38-44h |

### 3D Models

All 3D models share **iterations_per_epoch=250** for fair comparison.

| Model | Batch Size | Grad Accum | Effective Batch | Learning Rate | Est. Time |
|-------|------------|------------|-----------------|---------------|-----------|
| UNet 3D | 8 | 4 | 64 | 1e-4 | ~30-40h |
| ResNet 3D | 6 | 6 | 72 | 1e-4 | ~36-44h |
| Swin 3D | 2 | 16 | 64 | 3e-5 | ~44-48h |
| ViT 3D | 1 | 32 | 64 | 3e-5 | ~46-48h |

## Design Choices

### Batch Size Strategy
- **CNNs (UNet, ResNet)**: Larger batches possible due to lower memory footprint
- **Transformers (Swin, ViT)**: Smaller batches required due to attention memory

### Learning Rate Strategy
- **CNNs**: Standard 1e-4 learning rate
- **Transformers**: Lower 3e-5 to 5e-5 for stable training

### Gradient Accumulation
- Compensates for smaller batch sizes
- Targets ~64-384 effective batch size across all models
- Ensures fair comparison despite memory constraints

### A100 Optimizations
- **TF32 enabled** for transformers (faster without accuracy loss)
- **Gradient checkpointing** for 3D transformers
- **Expandable segments** for CUDA memory management
- **NCCL optimizations** for multi-GPU communication

## Monitoring

### Job Status
```bash
# View your jobs
squeue -u $USER

# Watch job progress (updates every 30s)
watch -n 30 'squeue -u $USER'

# View job details
scontrol show job <JOB_ID>
```

### Logs
```bash
# Real-time log viewing
tail -f logs/unet_2d_<JOB_ID>.out

# Error logs
tail -f logs/unet_2d_<JOB_ID>.err
```

### TensorBoard
```bash
# From login node (after jobs start)
tensorboard --logdir=../tensorboard --port=6006

# Forward port to local machine
ssh -L 6006:localhost:6006 user@login.rc.colorado.edu
```

## Post-Training Analysis

After all jobs complete, run the analysis scripts:

```bash
cd ..  # experiments/model_comparison/

# Analyze metrics
python analyze_comparison.py

# Generate visualizations
python visualize_comparison.py

# Statistical tests (if multiple runs)
python statistical_analysis.py
```

## Troubleshooting

### Out of Memory (OOM)
If a transformer model runs out of memory:
1. Reduce `BATCH_SIZE` in the sbatch file
2. Increase `GRADIENT_ACCUMULATION` proportionally
3. Enable gradient checkpointing (already enabled for 3D)

### Job Timeout
If training doesn't complete in 48 hours:
1. Reduce `EPOCHS` 
2. Or increase time limit (max 7 days on blanca-biokem)
3. Or enable checkpoint resumption

### NCCL Errors
If multi-GPU communication fails:
```bash
export NCCL_DEBUG=INFO  # Enable debug logging
export NCCL_P2P_DISABLE=1  # Try disabling P2P if issues persist
```

## Files

```
slurm/
├── submit_all.sh           # Submit all jobs
├── train_unet_2d.sbatch    # UNet 2D training
├── train_resnet_2d.sbatch  # ResNet 2D training
├── train_swin_2d.sbatch    # Swin 2D training
├── train_vit_2d.sbatch     # ViT 2D training
├── train_unet_3d.sbatch    # UNet 3D training
├── train_resnet_3d.sbatch  # ResNet 3D training
├── train_swin_3d.sbatch    # Swin 3D training
├── train_vit_3d.sbatch     # ViT 3D training
├── logs/                   # Job output logs
└── README.md               # This file
```
