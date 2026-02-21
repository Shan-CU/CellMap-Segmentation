"""
Base configuration for 2D MONAI CellMap training pipeline.

All per-model 2D configs import and deepcopy this, then override model-specific fields.
Inherits loss settings from the validated 3D common_config but with 2D-specific
data loading and training parameters.

Key 2D differences from 3D:
- ROI: [256, 256] (2D slices) vs [128, 128, 128] (3D patches)
- RAM-cached dataset with microsecond slicing vs MONAI CacheDataset
- Multi-axis slicing (axial/coronal/sagittal)
- Single GPU (no DDP)
- Higher batch sizes (2D models use far less VRAM)
- num_workers=4 with Linux COW sharing

Reference: EXPERIMENT_FINDINGS.md
"""

from types import SimpleNamespace


class Config(SimpleNamespace):
    """SimpleNamespace with .get() support for safe attribute access."""

    def get(self, key, default=None):
        return getattr(self, key, default)


cfg = Config()

# === Data ===
cfg.datalist = "/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/auto3dseg/nifti_data_v2/datalist.json"
cfg.dataroot = ""  # paths in datalist are absolute
cfg.num_classes = 35
cfg.class_names = [
    # ── Original 14 from Round 1 ──
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
    # ── New for Round 2 ──
    "lyso_mem", "lyso_lum", "ld_mem", "ld_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl",
    "mt_out", "cyto", "mt_in", "perox_mem", "perox_lum",
    "nhchrom", "nechrom", "nucleo",
]
cfg.in_channels = 1
cfg.sigmoid = True  # multi-label, not softmax

# === 2D Patches ===
cfg.roi_size_2d = [256, 256]       # crop size for random 2D slices
cfg.num_samples = 8                 # sub-slices per volume per __getitem__
cfg.multi_axis = True               # sample axial + coronal + sagittal slices
cfg.iterations_per_epoch = 1000     # effective dataset length per epoch

# === Training ===
cfg.epochs = 500
cfg.lr = 1e-4
cfg.optimizer = "AdamW"
cfg.weight_decay = 1e-5
cfg.schedule = "cosine"
cfg.warmup = 0.05          # fraction of total steps for linear warmup
cfg.batch_size = 4          # volumes per step; each yields num_samples slices
                            # → actual forward batch = batch_size × num_samples slices
cfg.grad_accumulation = 1
cfg.clip_grad = 1.0
cfg.seed = 42

# === Loss (All validated optimizations from EXPERIMENT_FINDINGS.md) ===
cfg.loss_type = "balanced_softmax_tversky"   # BalancedSoftmaxTverskyLoss
cfg.tversky_alpha = 0.6    # FP penalty — loss optimization winner (+47%)
cfg.tversky_beta = 0.4     # FN penalty — precision-biased works better with partial annotations
cfg.tau = 1.0              # Balanced Softmax temperature (+54%)
cfg.update_interval = 50   # steps between frequency re-estimates

# Spatial masking: box_class_mask_tight
# Eval Dice: 0.376 (+55% over no_mask baseline 0.243)
cfg.bbox_pad_fraction = 0.05   # fraction of bbox extent to pad
cfg.bbox_bg_weight = 0.05      # weight for voxels outside all class bboxes

# Mask-supervised reconstruction: masksup_r0.3
# Winner of masking_strategies experiment post-foreground-fix (§4.8)
# Eval Dice: 0.5711 (+12% over no_mask 0.511)
# Randomly masks 30% of annotated voxels → forces context learning
cfg.masksup_ratio = 0.3          # fraction of annotated voxels to mask
cfg.masksup_recon_weight = 0.5   # weight for reconstruction branch

# === Precision ===
cfg.bf16 = True             # bfloat16 autocast (L40S native support)
cfg.mixed_precision = False  # standard fp16 (mutually exclusive with bf16)

# === Resources ===
cfg.num_workers = 4          # Linux COW shares RAM-cached arrays across forked workers
cfg.pin_memory = False       # pin_memory=True causes epoch slowdown (MONAI #3116)

# === Checkpointing ===
cfg.save_checkpoint = True
cfg.save_weights_only = False
cfg.save_every_n_epochs = 50     # periodic epoch checkpoints
cfg.eval_epochs = 5              # validate every N epochs
cfg.output_dir = "/work/users/g/s/gsgeorge/cellmap/runs/monai_2d"

# === Model (defaults, overridden in per-model configs) ===
cfg.backbone_type = "resnet"
cfg.backbone_args = {}

# === Logging ===
cfg.disable_tqdm = False

basic_cfg_2d = cfg
