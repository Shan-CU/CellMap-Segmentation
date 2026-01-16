# UNet 3D Baseline Training Config (Optimized for Alpine A100)
# Run with: python train_unet3d_baseline.py
# Monitor with: tensorboard --logdir tensorboard
#
# 3D MODEL ADVANTAGES:
#   - Sees 3D context (organelles span multiple slices)
#   - Better for volumetric EM data
#   - More parameters but captures 3D structure
#
# TRADE-OFF: Much higher memory usage - smaller batch sizes required

import torch
from upath import UPath
from cellmap_segmentation_challenge.models import UNet_3D
from cellmap_segmentation_challenge.utils import get_tested_classes

# %% Hyperparameters - Optimized for 3D training on Alpine A100
learning_rate = 1e-4  # Peak LR after warmup

# Batch size - 3D uses MUCH more memory than 2D
# A100 80GB can handle batch=4 per GPU for 32×256×256 volumes
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
batch_size_per_gpu = 4  # Small due to 3D memory requirements
batch_size = batch_size_per_gpu * n_gpus  # 12 total with 3 GPUs
gradient_accumulation_steps = 8  # Effective batch = 12 * 8 = 96
print(f"Using {n_gpus} GPU(s), batch_size={batch_size}, effective_batch={batch_size * gradient_accumulation_steps}")

# 3D input - 32 slices of 256×256
# This captures ~256nm of depth context (32 slices × 8nm/slice)
input_array_info = {
    "shape": (32, 256, 256),  # Depth × Height × Width
    "scale": (8, 8, 8),       # 8nm isotropic voxels
}
target_array_info = {
    "shape": (32, 256, 256),
    "scale": (8, 8, 8),
}

# Training schedule - adjusted for 3D (slower iterations)
# 3D is ~10-20x slower per iteration than 2D
# With 3 A100s and batch=12, ~10-15 seconds/iteration
# 24 hours = 86,400 seconds = ~6,000-8,000 iterations possible
epochs = 200  # Fewer epochs due to slower iterations
iterations_per_epoch = 50  # Checkpoint every ~10 minutes
warmup_steps = 200  # ~4 epochs of warmup
random_seed = 42

# Use all 14 classes
classes = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]
print(f"Training UNet 3D baseline with {len(classes)} classes: {classes}")

# Model - UNet 3D
model_name = "unet3d_baseline"
model_to_load = "unet3d_baseline"
_model = UNet_3D(1, len(classes))

# ============================================================
# MULTI-GPU: Wrap model with DataParallel if multiple GPUs available
# ============================================================
if n_gpus > 1:
    print(f"Using DataParallel with {n_gpus} GPUs")
    model = torch.nn.DataParallel(_model)
else:
    model = _model

load_model = "latest"

# Optimizer: AdamW with weight decay
optimizer = torch.optim.AdamW(
    _model.parameters(), 
    lr=learning_rate,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

# LR Scheduler: Cosine annealing with warmup
total_steps = epochs * iterations_per_epoch
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    total_steps=total_steps,
    pct_start=0.05,
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1000,
)

# Paths
logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# Data augmentation - 3D compatible
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.5}},  # Also flip in z
    "transpose": {"axes": ["x", "y"]},  # Only transpose x/y
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

# Training optimizations
max_grad_norm = 1.0
validation_time_limit = 60  # Longer validation for 3D
validation_batch_limit = 5  # Fewer batches due to 3D size
filter_by_scale = True

# Dataset handling
force_all_classes = False

# Dataloader - fewer workers for 3D (more memory per sample)
n_workers = 8  # Reduced from 16 due to 3D memory
print(f"Using {n_workers} dataloader workers")

dataloader_kwargs = {
    "num_workers": n_workers,
    "pin_memory": False,
    "persistent_workers": True if n_workers > 0 else False,
    "multiprocessing_context": "spawn" if n_workers > 0 else None,
}

# Mixed precision - critical for 3D to fit in memory
use_mixed_precision = True

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
