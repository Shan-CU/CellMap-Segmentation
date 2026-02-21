# 2D ViT-V-Net R3 Config — MONAI pipeline with all validated optimizations
#
# ViTVNet2D: 105.2M params, most powerful 2D arch (global attention).
# R3 optimizations: Tversky α=0.6/β=0.4, Balanced Softmax τ=1.0,
# bbox masking, foreground masking, RAM-cached dataset
#
# VRAM: ~25-30 GB (batch 4 × 8 samples × 256×256) → fits on L40S 48GB
# Lower LR for transformer stability (5e-5)
# attention_dropout=0.1 prevents attention collapse

from copy import deepcopy
from .common_config_2d import basic_cfg_2d

cfg = deepcopy(basic_cfg_2d)

cfg.name = "2d_vit_r3"
cfg.backbone_type = "vit"
cfg.backbone_args = {
    "in_channels": 1,
    "vit_config": {
        "img_size": 192,
        "patch_size": 16,
        "hidden_size": 768,
        "num_layers": 12,
        "num_heads": 12,
        "mlp_dim": 3072,
        "decoder_channels": (256, 128, 64, 16),
        "dropout_rate": 0.1,
        "attention_dropout_rate": 0.1,
        "down_factor": 2,
    },
}

# ViT is VRAM-hungry → smallest batch
# batch_size=1 volume × num_samples=8 = 8 slices per forward pass
# (R2 ViT used batch_size=4 individual slices — we get 8 here)
cfg.batch_size = 1
cfg.grad_accumulation = 8     # effective = 8 volumes × 8 samples = 64 slices per optimizer step
cfg.lr = 5e-5                 # lower LR for transformer stability

cfg.epochs = 500
cfg.iterations_per_epoch = 1000
cfg.eval_epochs = 5
cfg.save_every_n_epochs = 50

# ViTVNet2D has layers incompatible with bf16/fp16 autocast on L40S
# (cudaErrorInvalidConfiguration during backward, even with SDPA attention).
# Use fp32 with smaller 192×192 ROI to fit in 46GB VRAM.
cfg.precision = "fp32"
cfg.roi_size_2d = [192, 192]

cfg.output_dir = "/work/users/g/s/gsgeorge/cellmap/runs/monai_2d/vit_r3"
