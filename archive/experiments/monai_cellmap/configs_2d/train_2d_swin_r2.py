# 2D Swin Transformer Training Config — Round 2 (35 atomic classes)
#
# Uses the CSC framework's native train() function.
# SwinTransformer uses shifted window attention — strong for dense prediction.
# R2 updates: 35 classes, get_tested_classes(), AdamW, CosineAnnealingLR,
# dropout=0.1, attention_dropout=0.1, stochastic_depth=0.2,
# gradient clipping, force_all_classes=False.
#
# Experimental findings applied:
#   - attention_dropout=0.1 (prevents attention collapse in transformers)
#   - stochastic_depth_prob=0.2 (regularization)
#   - Lower LR for transformer stability (5e-5)

import torch
from upath import UPath
from cellmap_segmentation_challenge.models import SwinTransformer
from cellmap_segmentation_challenge.utils import get_tested_classes

# ============================================================
# Classes — all 48 tested (35 atomic + 16 groups, handled by CSC)
# ============================================================
classes = get_tested_classes()

# ============================================================
# Hyperparameters
# ============================================================
learning_rate = 5e-5  # Lower LR for transformer stability
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

# Training schedule — 200 it/ep × 2000 ep = 400k total steps
# At ~2s/it: 400k × 2 = 800k s ≈ 9.3 days (within 11-day limit)
epochs = 2000
iterations_per_epoch = 200
random_seed = 42

# ============================================================
# Model — SwinTransformer (Swin-T config with U-Net decoder)
# ============================================================
model_name = "2d_swin_r2"
model_to_load = "2d_swin_r2"
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

load_model = "latest"

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
# LR Scheduler — CosineAnnealingLR
# ============================================================
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=learning_rate / 1000,
)

# ============================================================
# Paths
# ============================================================
logs_save_path = UPath("experiments/monai_cellmap/tensorboard_2d/{model_name}").path
model_save_path = UPath("experiments/monai_cellmap/checkpoints_2d/{model_name}_{epoch}.pth").path
datasplit_path = "datasplit_r2.csv"  # New CSV for 48 classes (old datasplit.csv has only 14)

# ============================================================
# Data & Training Settings
# ============================================================
spatial_transforms = {
    "mirror": {"axes": {"x": 0.5, "y": 0.5}},
    "transpose": {"axes": ["x", "y"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
}

max_grad_norm = 1.0  # CRITICAL for transformers
validation_time_limit = 60
validation_batch_limit = 20
filter_by_scale = True
force_all_classes = False
device = "cuda"

dataloader_kwargs = {
    "num_workers": 2,
    "pin_memory": True,
}

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train
    train(__file__)
