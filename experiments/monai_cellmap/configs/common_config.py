"""
Base configuration for MONAI CellMap training pipeline.

Uses a Config class (extends SimpleNamespace) that supports .get() method
to avoid AttributeError when accessing optional fields with defaults.
All per-model configs should deepcopy this and override specific fields.

Reference: IMPLEMENTATION_SPEC.md §5.1, §11.1, §11.6
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

# === Patches ===
cfg.roi_size = [128, 128, 128]  # patch size for random cropping
cfg.num_samples = 4  # sub-patches per volume per __getitem__ call

# === Training ===
cfg.epochs = 100
cfg.lr = 1e-3
cfg.optimizer = "AdamW"
cfg.weight_decay = 1e-5
cfg.schedule = "cosine"
cfg.warmup = 0.05  # fraction of total steps for warmup
cfg.batch_size = 2  # volumes per GPU per step (each yields num_samples patches)
cfg.grad_accumulation = 1
cfg.clip_grad = 1.0
cfg.seed = 42

# === Loss (Per-class Tversky + Balanced Softmax online weighting) ===
cfg.loss_type = "balanced_softmax_tversky"  # BalancedSoftmaxTverskyLoss
cfg.tversky_alpha = 0.6   # FP penalty weight — experimentally validated winner
                          # loss_optimization exp: α=0.6/β=0.4 → Dice 0.370, α=0.3/β=0.7 → 0.028 (collapsed)
cfg.tversky_beta = 0.4    # FN penalty weight — precision-biased works better with partial annotations
cfg.tau = 1.0             # Balanced Softmax temperature
cfg.update_interval = 50  # steps between frequency re-estimates

# === Augmentation (Mixup) ===
cfg.mixup_p = 0.0  # disabled by default (Round 2: Mixup dilutes unannotated channels)
cfg.mixup_beta = 1.0  # Beta distribution parameter

# === Precision ===
cfg.bf16 = True  # use bfloat16 autocast (L40S native support)
cfg.mixed_precision = False  # standard fp16 (mutually exclusive with bf16)

# === DDP ===
cfg.distributed = True
cfg.find_unused_parameters = False
cfg.syncbn = False

# === Resources ===
cfg.num_workers = 4   # 4 workers/rank × 6 ranks = 24 total; ~250 GB RAM headroom
cfg.pin_memory = False  # pin_memory=True causes epoch slowdown (MONAI #3116)
cfg.cache_rate = 1.0  # cache all qualifying crops in RAM
cfg.max_cache_file_size_mb = 500  # Strategy D: only cache crops <= this size
cfg.drop_last = True

# === Checkpointing ===
cfg.save_checkpoint = True
cfg.save_weights_only = False  # save full state (optimizer, epoch) for proper resume
cfg.save_only_last_ckpt = False
cfg.save_every_n_epochs = 50   # Round 2: only save epoch checkpoints every N epochs (was every epoch = 600GB wasted)
cfg.eval_epochs = 5  # validate every N epochs
cfg.output_dir = "/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap"

# === Model (defaults, overridden in per-model configs) ===
cfg.model = "mdl_cellmap"
cfg.backbone_type = "segresnet"
cfg.backbone_args = {}
cfg.deep_supervision = False
cfg.ds_weights = None
cfg.multi_scale_heads = False
cfg.lvl_weights = None

# === Logging ===
cfg.neptune_project = None  # or "workspace/project" for Neptune
cfg.disable_tqdm = False

basic_cfg = cfg
