"""
SwinUNETR v2 config.

SwinUNETR uses shifted-window self-attention for the encoder with a
CNN-based decoder. Requires fixed spatial size matching img_size.

Reference: IMPLEMENTATION_SPEC.md §5.3
"""

from copy import deepcopy

from common_config import basic_cfg

cfg = deepcopy(basic_cfg)
cfg.name = "swinunetr_r3"
cfg.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/{cfg.name}"

# --- Model ---
cfg.model = "mdl_cellmap"
cfg.backbone_type = "swinunetr"
cfg.backbone_args = dict(
    in_channels=cfg.in_channels,
    out_channels=cfg.num_classes,
    feature_size=48,
    drop_rate=0.1,             # Round 2: add dropout (was 0.0, model overfit early)
    attn_drop_rate=0.1,        # Round 2: attention dropout
    dropout_path_rate=0.1,     # Round 2: stochastic depth
    use_v2=True,
)
cfg.deep_supervision = False  # SwinUNETR doesn't have built-in DS

# --- Patches: 96³ b3 on H100 80GB (was b2 on L40S 48GB ~42GB VRAM → b3 ~63GB) ---
cfg.roi_size = [96, 96, 96]
cfg.num_samples = 4
cfg.batch_size = 3

# --- Training (Round 2: 300 epochs — best was at ep139/600, most volatile) ---
cfg.lr = 1e-4
cfg.epochs = 300
cfg.eval_epochs = 5
