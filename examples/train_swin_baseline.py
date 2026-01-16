# SwinTransformer Baseline Training Config (Optimized for Alpine A100)
# Run with: python train_swin_baseline.py
# Monitor with: tensorboard --logdir tensorboard
#
# ALPINE A100 OPTIMIZATIONS:
#   - 3x A100 80GB GPUs with DataParallel
#   - 16 CPU workers for parallel data loading
#   - Mixed precision (AMP) for TensorCore acceleration
#   - pin_memory + persistent_workers for fast data transfer
#
# Based on Lauenburg & eminorhan winning configs:
#   - Frequent validation for better learning signal
#   - LR warmup + cosine decay (CRITICAL for transformers)
#   - Gradient clipping for stability
#   - AdamW optimizer (essential for transformers)

import torch
from upath import UPath
from cellmap_segmentation_challenge.models import SwinTransformer
from cellmap_segmentation_challenge.utils import get_tested_classes

# %% Hyperparameters - Optimized for 24-HOUR transformer training on Alpine A100
learning_rate = 1e-4  # Peak LR after warmup

# Batch size optimized for multi-GPU A100 (80GB VRAM each)
# Transformers use more memory than UNets, but A100 80GB is generous
n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
batch_size_per_gpu = 16  # A100 80GB can handle 16 for SwinTransformer 256x256
batch_size = batch_size_per_gpu * n_gpus  # 48 total with 3 GPUs
gradient_accumulation_steps = 4  # Effective batch = 48 * 4 = 192
print(f"Using {n_gpus} GPU(s), batch_size={batch_size}, effective_batch={batch_size * gradient_accumulation_steps}")

input_array_info = {
    "shape": (1, 256, 256),  # Must be divisible by patch_size and window_size
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}

# Training schedule - OPTIMIZED FOR 24-HOUR RUN
# Transformers train slower than UNets, so same epochs but may complete fewer
# With 3 A100s and batch=48, ~3 seconds/iteration (transformers are slower)
# 24 hours = 86,400 seconds = ~28,000 iterations possible
# We use 400 epochs × 100 iterations = 40,000 steps (accounting for overhead)
epochs = 400  # Slightly fewer than UNet due to slower per-iteration time
iterations_per_epoch = 100  # Frequent validation/checkpoints
warmup_steps = 1000  # ~10 epochs warmup (CRITICAL for transformers!)
random_seed = 42

# Use common classes that have good data coverage
classes = [
    'ecs', 'pm', 'mito_mem', 'mito_lum', 'mito_ribo',
    'golgi_mem', 'golgi_lum', 'ves_mem', 'ves_lum',
    'endo_mem', 'endo_lum', 'er_mem', 'er_lum', 'nuc'
]
print(f"Training SwinTransformer baseline with {len(classes)} classes: {classes}")

# Model - SwinTransformer
model_name = "swin_baseline"
model_to_load = "swin_baseline"
_model = SwinTransformer(
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[7, 7],
    num_classes=len(classes)
)

# ============================================================
# MULTI-GPU: Wrap model with DataParallel if multiple GPUs available
# ============================================================
if n_gpus > 1:
    print(f"Using DataParallel with {n_gpus} GPUs")
    model = torch.nn.DataParallel(_model)
else:
    model = _model

load_model = "latest"

# Optimizer: AdamW with weight decay (essential for transformers)
# Note: Use _model.parameters() to get actual model params (not DataParallel wrapper)
optimizer = torch.optim.AdamW(
    _model.parameters(), 
    lr=learning_rate,
    weight_decay=1e-4,  # Regularization
    betas=(0.9, 0.999)
)

# LR Scheduler: Cosine annealing with warmup (critical for transformers)
total_steps = epochs * iterations_per_epoch
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,
    total_steps=total_steps,
    pct_start=0.05,  # 5% warmup
    anneal_strategy='cos',
    div_factor=25,
    final_div_factor=1000,
)

# Paths
logs_save_path = UPath("tensorboard/{model_name}").path
model_save_path = UPath("checkpoints/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# Data augmentation
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

# Training optimizations
max_grad_norm = 1.0  # Gradient clipping - CRITICAL for transformers
validation_time_limit = 30
validation_batch_limit = 10
filter_by_scale = True

# ============================================================
# DATASET HANDLING (from Lauenburg's winning solution)
# ============================================================
# force_all_classes=False allows training on datasets that only have SOME of
# the requested classes. This is critical because many crops don't have all
# organelle types. With False, a crop is included if it has at least one class.
force_all_classes = False

# ============================================================
# ALPINE A100 DATALOADER OPTIMIZATIONS
# ============================================================
# Use 16 CPU cores for parallel data loading (set in SLURM script)
# pin_memory=False because cellmap-data moves data to GPU internally
# persistent_workers=True keeps workers alive between epochs (reduces overhead)
# multiprocessing_context="spawn" is REQUIRED for CUDA - fork doesn't work!
n_workers = 16  # Hardcoded for Alpine (SLURM sets 32 CPUs, use half for dataloading)
print(f"Using {n_workers} dataloader workers")

dataloader_kwargs = {
    "num_workers": n_workers,
    "pin_memory": False,  # cellmap-data moves data to GPU, can't pin CUDA tensors
    "persistent_workers": True if n_workers > 0 else False,
    "multiprocessing_context": "spawn" if n_workers > 0 else None,  # Required for CUDA!
}

# ============================================================
# MIXED PRECISION (AMP) - A100 TensorCores are optimized for BF16
# BF16 is preferred for transformers (better numerical stability than FP16)
# ============================================================
use_mixed_precision = True  # Enable AMP for ~2x speedup on A100

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
