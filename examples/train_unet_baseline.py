# UNet Baseline Training Config (Optimized for Alpine A100)
# Run with: python train_unet_baseline.py
# Monitor with: tensorboard --logdir tensorboard
#
# ALPINE A100 OPTIMIZATIONS:
#   - 3x A100 80GB GPUs with DataParallel
#   - 16 CPU workers for parallel data loading
#   - Mixed precision (AMP) for TensorCore acceleration
#   - Larger batch sizes (A100 80GB can handle it)
#   - pin_memory + persistent_workers for fast data transfer
#
# Based on Lauenburg & eminorhan winning configs:
#   - Frequent validation for better learning signal
#   - LR warmup + cosine decay
#   - Gradient clipping for stability
#   - AdamW optimizer with weight decay

import os
import torch
from upath import UPath
from cellmap_segmentation_challenge.models import UNet_2D
from cellmap_segmentation_challenge.utils import get_tested_classes

# %% Hyperparameters - Optimized for 24-HOUR Alpine A100 training
learning_rate = 1e-4  # Peak LR after warmup

# Batch size optimized for multi-GPU A100 (80GB VRAM each)
# A100 80GB can handle large batches - UNet is memory efficient
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
batch_size_per_gpu = 32  # A100 80GB can handle 32 for 256x256 UNet
batch_size = batch_size_per_gpu * n_gpus  # 96 total with 3 GPUs
gradient_accumulation_steps = 2  # Effective batch = 96 * 2 = 192
print(f"Using {n_gpus} GPU(s), batch_size={batch_size}, effective_batch={batch_size * gradient_accumulation_steps}")

input_array_info = {
    "shape": (1, 256, 256),  # 256x256 is optimal for batch size
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}

# Training schedule - OPTIMIZED FOR 24-HOUR RUN
# With 3 A100s and batch=96, ~2 seconds/iteration
# 24 hours = 86,400 seconds = ~43,000 iterations possible
# We use 500 epochs × 100 iterations = 50,000 steps (with some validation overhead)
epochs = 500  # Maximum epochs for 24hr run
iterations_per_epoch = 100  # 100 iterations between validation (frequent checkpointing)
warmup_steps = 1000  # ~10 epochs of warmup for stability
random_seed = 42

# Use common classes that have good data coverage
classes = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]
print(f"Training UNet baseline with {len(classes)} classes: {classes}")

# Model - UNet baseline
model_name = "unet_baseline"
model_to_load = "unet_baseline"
_model = UNet_2D(1, len(classes))

# ============================================================
# MULTI-GPU: Wrap model with DataParallel if multiple GPUs available
# ============================================================
if n_gpus > 1:
    print(f"Using DataParallel with {n_gpus} GPUs")
    model = torch.nn.DataParallel(_model)
else:
    model = _model

load_model = "latest"

# Optimizer: AdamW with weight decay (better than RAdam for this task)
# Note: Use _model.parameters() to get actual model params (not DataParallel wrapper)
optimizer = torch.optim.AdamW(
    _model.parameters(), 
    lr=learning_rate,
    weight_decay=1e-4,  # Regularization
    betas=(0.9, 0.999)
)

# LR Scheduler: Cosine annealing with warmup
# OneCycleLR includes warmup (pct_start) and cosine decay
total_steps = epochs * iterations_per_epoch
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    total_steps=total_steps,
    pct_start=0.05,  # 5% warmup
    anneal_strategy='cos',
    div_factor=25,  # initial_lr = max_lr / 25
    final_div_factor=1000,  # final_lr = max_lr / 1000
)

# Paths
logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# Data augmentation - same as before
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

# Training optimizations
max_grad_norm = 1.0  # Gradient clipping for stability
validation_time_limit = 30  # Shorter validation since we run more often
validation_batch_limit = 10  # Limit validation batches
filter_by_scale = True  # Filter out datasets without data at required scale

# ============================================================
# DATASET HANDLING (from Lauenburg's winning solution)
# ============================================================
# force_all_classes=False allows training on datasets that only have SOME of
# the requested classes. This is critical because many crops don't have all
# organelle types. With False, a crop is included if it has at least one class.
# Options:
#   - False: Include crops with at least one requested class (most flexible)
#   - True: Require ALL classes to be present (strictest - may have no data!)
#   - "validate": Require all classes only for validation
#   - "train": Require all classes only for training
force_all_classes = False

# ============================================================
# ALPINE A100 DATALOADER OPTIMIZATIONS
# ============================================================
# Use 16 CPU cores for parallel data loading (set in SLURM script)
# pin_memory=False because cellmap-data moves data to GPU internally
# persistent_workers=True keeps workers alive between epochs (reduces overhead)
# multiprocessing_context="spawn" is REQUIRED for CUDA - fork doesn't work!
n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 8))
print(f"Using {n_workers} dataloader workers")

dataloader_kwargs = {
    "num_workers": n_workers,
    "pin_memory": False,  # cellmap-data moves data to GPU, can't pin CUDA tensors
    "persistent_workers": True if n_workers > 0 else False,
    "multiprocessing_context": "spawn" if n_workers > 0 else None,  # Required for CUDA!
}

# ============================================================
# MIXED PRECISION (AMP) - A100 TensorCores are optimized for FP16/BF16
# ============================================================
use_mixed_precision = True  # Enable AMP for ~2x speedup on A100

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
