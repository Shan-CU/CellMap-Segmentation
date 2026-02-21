# 2D SwinTransformer R3 Config — MONAI pipeline with all validated optimizations
#
# SwinTransformer: 36.3M params, shifted window attention.
# R3 optimizations: Tversky α=0.6/β=0.4, Balanced Softmax τ=1.0,
# bbox masking, foreground masking, RAM-cached dataset
#
# VRAM: ~18-22 GB (batch 16 × 8 samples × 256×256) → fits on L40S 48GB
# Lower LR for transformer stability (5e-5)
# attention_dropout=0.1 prevents attention collapse

from copy import deepcopy
from .common_config_2d import basic_cfg_2d

cfg = deepcopy(basic_cfg_2d)

cfg.name = "2d_swin_r3"
cfg.backbone_type = "swin"
cfg.backbone_args = {
    "patch_size": [4, 4],
    "embed_dim": 96,
    "depths": [2, 2, 6, 2],
    "num_heads": [3, 6, 12, 24],
    "window_size": [7, 7],
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "stochastic_depth_prob": 0.2,
}

# Swin uses more VRAM → smaller batch
# batch_size=2 volumes × num_samples=8 = 16 slices per forward pass
# (matches R2 Swin batch_size=16 individual slices)
cfg.batch_size = 2
cfg.grad_accumulation = 4     # effective = 8 volumes × 8 samples = 64 slices per optimizer step
cfg.lr = 5e-5                 # lower LR for transformer stability

cfg.epochs = 500
cfg.iterations_per_epoch = 1000
cfg.eval_epochs = 5
cfg.save_every_n_epochs = 50

cfg.output_dir = "/work/users/g/s/gsgeorge/cellmap/runs/monai_2d/swin_r3"
