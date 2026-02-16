"""
SwinUNETR v2 config.

SwinUNETR uses shifted-window self-attention for the encoder with a
CNN-based decoder. Requires fixed spatial size matching img_size.

Reference: IMPLEMENTATION_SPEC.md §5.3
"""

from copy import deepcopy

from common_config import basic_cfg

cfg = deepcopy(basic_cfg)
cfg.name = "swinunetr"
cfg.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/{cfg.name}"

# --- Model ---
cfg.model = "mdl_cellmap"
cfg.backbone_type = "swinunetr"
cfg.backbone_args = dict(
    in_channels=cfg.in_channels,
    out_channels=cfg.num_classes,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_v2=True,
)
cfg.deep_supervision = False  # SwinUNETR doesn't have built-in DS

# --- Patches: must match img_size ---
cfg.roi_size = [96, 96, 96]
cfg.num_samples = 4
cfg.batch_size = 2

# --- Training ---
cfg.lr = 1e-4
cfg.epochs = 600
cfg.eval_epochs = 5
