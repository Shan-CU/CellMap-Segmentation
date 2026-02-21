# 2D Swin Transformer Training Config — Optimized for L40S Single GPU (5.5-day run)
#
# Uses the CSC framework's native train() function.
# Improvements over examples/train_2D.py:
#   - All 14 tested classes (not just 2)
#   - AdamW optimizer with weight decay (essential for transformers)
#   - OneCycleLR cosine scheduler with warmup (critical for transformers)
#   - Gradient clipping for stability
#   - force_all_classes=False (critical for partial annotations)
#   - Longer training (5.5 days on single L40S)

import torch
from upath import UPath
from cellmap_segmentation_challenge.models import SwinTransformer

# ============================================================
# Hyperparameters
# ============================================================
learning_rate = 1e-4  # Peak LR (OneCycleLR will ramp up then decay)
batch_size = 16  # SwinTransformer uses more VRAM than UNet
gradient_accumulation_steps = 4  # Effective batch = 64

input_array_info = {
    "shape": (1, 256, 256),  # Must be divisible by patch_size × window_size
    "scale": (8, 8, 8),
}
target_array_info = {
    "shape": (1, 256, 256),
    "scale": (8, 8, 8),
}

# Training schedule
# SwinTransformer 2D is slower: ~3-5 s/iteration on L40S with batch=16
# 5.5 days = 475,200 s → ~95k-158k iterations possible
# Use 800 epochs × 150 iterations = 120,000 total steps
epochs = 800
iterations_per_epoch = 150
random_seed = 42

# All 14 classes from the challenge
classes = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
]

# ============================================================
# Model — SwinTransformer (Swin-T config with U-Net decoder)
# ============================================================
model_name = "2d_swin_l40s"
model_to_load = "2d_swin_l40s"
model = SwinTransformer(
    patch_size=[4, 4],
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=[7, 7],
    num_classes=len(classes),
    dropout=0.1,
    attention_dropout=0.1,
    stochastic_depth_prob=0.2,
)

load_model = "latest"  # Resume from latest checkpoint if exists

# ============================================================
# Optimizer — AdamW (essential for transformers)
# ============================================================
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=1e-4,
    betas=(0.9, 0.999),
)

# ============================================================
# LR Scheduler — CosineAnnealingLR (stepped per-epoch by CSC framework)
# Note: CSC train() calls scheduler.step() once per epoch.
# ============================================================
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,          # Full cosine cycle over all epochs
    eta_min=learning_rate / 1000,  # Minimum LR
)

# ============================================================
# Paths — save inside experiments/monai_cellmap/
# ============================================================
logs_save_path = UPath("experiments/monai_cellmap/tensorboard_2d/{model_name}").path
model_save_path = UPath("experiments/monai_cellmap/checkpoints_2d/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit.csv"

# ============================================================
# Data & Training Settings
# ============================================================
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

max_grad_norm = 1.0         # Gradient clipping — CRITICAL for transformers
validation_time_limit = 60  # 60s max per validation
validation_batch_limit = 20 # At most 20 val batches
filter_by_scale = True      # Only use data at matching resolution
force_all_classes = False   # Allow partial annotations
device = "cuda"

# Dataloader config (single GPU, no DDP)
dataloader_kwargs = {
    "num_workers": 2,
    "pin_memory": True,
}

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
