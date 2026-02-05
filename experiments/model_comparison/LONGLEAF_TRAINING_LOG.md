# Longleaf Training Log - Model Comparison Experiment

## Date: February 5, 2026

### Issue: SLURM Job Failures (Exit Code 127)

All GPU training jobs on Longleaf were failing immediately with exit code 127 ("command not found").

#### Root Cause
The micromamba environment activation was failing in non-interactive SLURM batch mode:
```
critical libmamba Shell not initialized
/var/spool/slurmd/job29489993/slurm_script: line 71: torchrun: command not found
```

The `eval "$("$MAMBA_EXE" shell hook ...)"` followed by `"$MAMBA_EXE" activate csc` pattern does not work in SLURM batch scripts because they run in non-interactive mode.

#### Solution
Replaced environment activation with `micromamba run -n csc` which executes commands directly within the conda environment without needing shell initialization:

```bash
# Before (broken):
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
"$MAMBA_EXE" activate csc
torchrun --standalone --nproc_per_node=$N_GPUS ...

# After (fixed):
$MAMBA_EXE run -n csc torchrun --standalone --nproc_per_node=$N_GPUS ...
```

#### Files Modified
- `slurm/longleaf/train_resnet_2d_longleaf.sbatch`
- `slurm/longleaf/train_resnet_3d_longleaf.sbatch`
- `slurm/longleaf/train_swin_2d_longleaf.sbatch`
- `slurm/longleaf/train_swin_3d_longleaf.sbatch`
- `slurm/longleaf/train_unet_2d_longleaf.sbatch`
- `slurm/longleaf/train_unet_3d_longleaf.sbatch`
- `slurm/longleaf/train_vit_2d_longleaf.sbatch`
- `slurm/longleaf/train_vit_3d_longleaf.sbatch`

---

### Jobs Resubmitted

| Job ID | Model | Partition | Status |
|--------|-------|-----------|--------|
| 29591894 | ResNet 2D | a100-gpu | Submitted |
| 29591895 | ResNet 3D | a100-gpu | Submitted |
| 29591896 | Swin 2D | a100-gpu | Submitted |
| 29591897 | Swin 3D | a100-gpu | Submitted |
| 29591898 | UNet 2D | a100-gpu | Submitted |
| 29591899 | UNet 3D | a100-gpu | Submitted |
| 29591900 | ViT 2D | a100-gpu | Submitted |
| 29591901 | ViT 3D | a100-gpu | Submitted |

---

### UNet 2D Training Progress (from previous Sycamore run)

Analysis from TensorBoard logs at step ~128,000:

| Metric | Current Value | Best Value | Best Step |
|--------|---------------|------------|-----------|
| Train Loss | ~0.44 | 0.050 | 12,573 |
| Val Loss | 0.502 | 0.396 | 12,000 |
| **Val Dice** | 0.483 | **0.515** | **121,000** |

#### Assessment
- **Model Status**: NOT collapsed, still learning
- **Overfitting**: Yes, validation loss degraded from 0.396 → 0.502
- **Best Checkpoint**: Step 121,000 (Dice = 0.515)
- **Recommendation**: Use checkpoint from step 121,000 for inference

#### Training Curves Summary
- Training loss was lowest early (step ~12k) suggesting good initial fit
- Validation Dice continued improving until step 121k
- After step 121k, model started to plateau/slightly overfit
- Gradient norms stable (~0.15-0.25), no exploding gradients
- Learning rate at 1.6e-5 (likely using cosine annealing schedule)

---

### References
- [Longleaf Getting Started](https://help.rc.unc.edu/getting-started-on-longleaf/)
- [Longleaf SLURM Examples](https://help.rc.unc.edu/longleaf-slurm-examples/)
- [Modules Documentation](https://help.rc.unc.edu/modules)
