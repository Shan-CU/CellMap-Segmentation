# 2D UNet R3 Config — MONAI pipeline with all validated optimizations
#
# UNet_2D: 31M params, strong baseline architecture.
# R3 optimizations: Tversky α=0.6/β=0.4, Balanced Softmax τ=1.0,
# bbox masking, foreground masking, RAM-cached dataset
#
# VRAM: ~10-14 GB (batch 32 × 8 samples × 256×256) → fits on L40S 48GB

from copy import deepcopy
from .common_config_2d import basic_cfg_2d

cfg = deepcopy(basic_cfg_2d)

cfg.name = "2d_unet_r3"
cfg.backbone_type = "unet"
cfg.backbone_args = {
    "in_channels": 1,
}

# batch_size=4 volumes × num_samples=8 = 32 slices per forward pass
cfg.batch_size = 4
cfg.grad_accumulation = 2     # effective = 8 volumes × 8 samples = 64 slices per optimizer step
cfg.lr = 1e-4

cfg.epochs = 500
cfg.iterations_per_epoch = 1000
cfg.eval_epochs = 5
cfg.save_every_n_epochs = 50

cfg.output_dir = "/work/users/g/s/gsgeorge/cellmap/runs/monai_2d/unet_r3"
