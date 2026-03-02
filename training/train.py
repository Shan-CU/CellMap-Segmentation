"""
CellMap Ablation Training Script.

Uses CSC's cellmap-data for zarr data loading (correct multi-resolution handling)
with our custom partial annotation losses and model wrappers.

This replaces the old NIfTI-based MONAI pipeline. Data is loaded natively from
zarr at the correct resolution via cellmap-data's CellMapDataLoader.

Usage:
    python -m training.train --config training/configs/ablation_loss_2d.yaml
    python -m training.train --config training/configs/ablation_loss_3d.yaml

Or via SLURM:
    sbatch training/slurm/ablation_loss_2d.sbatch
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import logging
import os
import random
import resource
import sys
import time
from pathlib import Path
from typing import Optional

# Configure root logger so cellmap_data INFO messages (e.g. TensorStore
# cache bounding) are visible in stderr.
logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s: %(message)s",
    stream=sys.stderr,
)

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from cellmap_data.transforms.augment import NaNtoNum, Binarize
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cellmap_segmentation_challenge.utils import get_dataloader, get_tested_classes
from cellmap_segmentation_challenge.utils.ddp import (
    setup_ddp, cleanup_ddp, is_main_process, is_ddp_initialized,
    reduce_value, sync_across_processes, get_world_size, get_rank,
)
from training.models.model_zoo import build_model, MODEL_REGISTRY
from training.losses.loss_zoo import build_loss, LOSS_REGISTRY
from training.losses.partial_annotation import FG_THRESHOLD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CellMap Ablation Training")

    # Experiment identity
    parser.add_argument("--experiment_name", type=str, required=True,
                        help="Name for this experiment (used for logging/checkpoints)")
    parser.add_argument("--run_dir", type=str, default="runs",
                        help="Base directory for all run outputs")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model architecture name")
    parser.add_argument("--model_kwargs", type=json.loads, default="{}",
                        help="JSON dict of extra model kwargs")
    parser.add_argument("--bias_init", type=float, default=None,
                        help="Initialize final conv bias to this value. "
                        "Prevents BCE collapse on sparse targets. "
                        "Typical value: -3.0 (sigmoid(-3)≈0.047)")

    # Loss
    parser.add_argument("--loss", type=str, default="balanced_softmax_tversky",
                        choices=list(LOSS_REGISTRY.keys()),
                        help="Loss function name")
    parser.add_argument("--loss_kwargs", type=json.loads, default="{}",
                        help="JSON dict of extra loss kwargs")

    # Foreground masking
    parser.add_argument("--use_foreground_mask", action="store_true", default=True,
                        help="Enable foreground masking (default: True)")
    parser.add_argument("--no_foreground_mask", action="store_true", default=False,
                        help="Disable foreground masking")

    # Data
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--input_shape", type=int, nargs="+", default=[1, 256, 256],
                        help="Input shape (e.g., '1 256 256' for 2D, '128 128 128' for 3D)")
    parser.add_argument("--input_scale", type=float, nargs="+", default=[8, 8, 8],
                        help="Input voxel scale in nm")
    parser.add_argument("--target_shape", type=int, nargs="+", default=None,
                        help="Target shape (defaults to input_shape)")
    parser.add_argument("--target_scale", type=float, nargs="+", default=None,
                        help="Target scale (defaults to input_scale)")
    parser.add_argument("--datasplit_path", type=str, default="datasplit.csv")
    parser.add_argument("--filter_by_scale", action="store_true", default=True)

    # Training
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs (50 for ablation, 1000 for full training)")
    parser.add_argument("--iterations_per_epoch", type=int, default=500,
                        help="Iterations per epoch (500 for ablation, 1000 for full)")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)

    # EMA
    parser.add_argument("--ema", action="store_true", default=False,
                        help="Use exponential moving average of model weights")
    parser.add_argument("--ema_decay", type=float, default=0.999,
                        help="EMA decay rate")

    # Deep supervision
    parser.add_argument("--deep_supervision", action="store_true", default=False,
                        help="Enable deep supervision (model must support it)")
    parser.add_argument("--ds_weights", type=float, nargs="+", default=None,
                        help="Deep supervision weights per scale")

    # Data sampling
    parser.add_argument("--no_weighted_sampler", action="store_true", default=False,
                        help="Disable weighted crop sampler (use uniform)")

    # Data augmentation / sampling improvements
    parser.add_argument("--intensity_aug", action="store_true", default=False,
                        help="Enable intensity augmentation (brightness, contrast, noise)")
    parser.add_argument("--class_aware_sampling", action="store_true", default=False,
                        help="Use inverse-sqrt class-aware crop weighting sampler")

    # OHEM
    parser.add_argument("--ohem_ratio", type=float, default=0.0,
                        help="Online hard example mining: keep top-K%% hardest voxels (0=disabled)")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "step"],
                        help="LR scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Validation
    parser.add_argument("--validation_time_limit", type=int, default=600,
                        help="Max seconds for validation per epoch")
    parser.add_argument("--val_every_n_epochs", type=int, default=1,
                        help="Run validation every N epochs")
    parser.add_argument("--best_metric", type=str, default="val_dice",
                        choices=["val_loss", "val_dice"],
                        help="Metric for best checkpoint selection (default: val_dice)")

    # Mixed precision
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (default: True)")
    parser.add_argument("--no_amp", action="store_true", default=False)

    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)

    # Memory management
    parser.add_argument("--persistent_workers", type=str, default="auto",
                        choices=["true", "false", "auto"],
                        help="DataLoader persistent_workers. 'auto' = use PyTorch default. "
                             "Safe for both 2D and 3D now that cellmap-data bounds "
                             "TensorStore cache (default 2 GiB).")
    parser.add_argument("--debug_memory", action="store_true", default=False,
                        help="Enable memory debugging: log RSS and Python object growth "
                             "via objgraph every MEMORY_LOG_STEPS iterations (env var, "
                             "default 100). Requires 'pip install objgraph'.")

    args = parser.parse_args()

    # Resolve defaults
    if args.target_shape is None:
        args.target_shape = args.input_shape
    if args.target_scale is None:
        args.target_scale = args.input_scale
    if args.no_foreground_mask:
        args.use_foreground_mask = False
    if args.no_amp:
        args.amp = False
    args.weighted_sampler = not args.no_weighted_sampler

    return args


class ModelEMA:
    """Exponential Moving Average of model weights.

    Maintains a shadow copy of model parameters updated as:
        shadow = decay * shadow + (1 - decay) * param

    Used at validation/inference for smoother, more generalizable predictions.
    Standard in nnU-Net v2, MONAI Auto3DSeg, and most competitive medical seg pipelines.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        import copy
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, m_param in zip(self.shadow.parameters(), model.parameters()):
            s_param.data.mul_(self.decay).add_(m_param.data, alpha=1.0 - self.decay)
        for s_buf, m_buf in zip(self.shadow.buffers(), model.buffers()):
            s_buf.copy_(m_buf)

    def state_dict(self):
        return self.shadow.state_dict()

    def load_state_dict(self, state_dict):
        self.shadow.load_state_dict(state_dict)


def get_annotation_mask_from_targets(targets: torch.Tensor) -> torch.Tensor:
    """Derive per-sample, per-channel annotation mask from cellmap-data targets.

    cellmap-data uses NaN for unannotated classes. We convert this to a binary
    mask: 1.0 = annotated, 0.0 = unannotated.

    Args:
        targets: (B, C, *spatial) tensor, may contain NaN.

    Returns:
        (B, C) float tensor — annotation mask.
    """
    # A channel is annotated if it has any non-NaN values
    spatial_dims = tuple(range(2, targets.ndim))
    # isnan check: if ALL voxels in a channel are NaN, it's unannotated
    nan_count = targets.isnan().sum(dim=spatial_dims)  # (B, C)
    total_voxels = 1
    for d in spatial_dims:
        total_voxels *= targets.shape[d]
    # Channel is annotated if not ALL voxels are NaN
    annotation_mask = (nan_count < total_voxels).float()  # (B, C)
    return annotation_mask


def train(args: argparse.Namespace) -> None:
    """Main training loop (supports single-GPU and multi-GPU DDP via torchrun)."""

    # === DDP Setup ===
    local_rank, world_size = setup_ddp()
    use_ddp = world_size > 1
    main_process = is_main_process()

    # === Setup ===
    torch.backends.cudnn.deterministic = True
    # Offset seed per rank so each GPU sees different data
    rank_seed = args.random_seed + get_rank()
    torch.manual_seed(rank_seed)
    np.random.seed(rank_seed)
    random.seed(rank_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)

    device = args.device
    if device is None:
        if use_ddp:
            device = f"cuda:{local_rank}"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    if main_process:
        print(f"Device: {device}" + (f" (DDP: {world_size} GPUs)" if use_ddp else ""))

    # === Classes ===
    classes = get_tested_classes()
    num_classes = len(classes)
    if main_process:
        print(f"Training on {num_classes} classes: {classes[:5]}... (showing first 5)")

    # === Output directories (only rank 0 creates) ===
    run_dir = Path(args.run_dir) / args.experiment_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "tensorboard"
    val_img_dir = run_dir / "val_images"
    if main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        val_img_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = run_dir / "config.json"
        config_dict = vars(args).copy()
        config_dict["ddp_world_size"] = world_size
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
        print(f"Config saved to {config_path}")
    sync_across_processes()  # Ensure dirs exist before other ranks proceed

    # === Data ===
    input_array_info = {
        "shape": tuple(args.input_shape),
        "scale": tuple(args.input_scale),
    }
    target_array_info = {
        "shape": tuple(args.target_shape),
        "scale": tuple(args.target_scale),
    }

    # Determine spatial transforms based on dimensionality
    is_3d = all(s > 1 for s in args.input_shape)
    if is_3d:
        spatial_transforms = {
            "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.5}},
            "transpose": {"axes": ["x", "y", "z"]},
            "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180], "z": [-180, 180]}},
        }
    else:
        spatial_transforms = {
            "mirror": {"axes": {"x": 0.5, "y": 0.5}},
            "transpose": {"axes": ["x", "y"]},
            "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
        }

    if main_process:
        print("Loading data...")
    # Keep data on CPU during loading to avoid CUDA OOM from pre-moving
    # all ~784 dataset EmptyImage tensors to GPU. The training loop already
    # handles per-batch .to(device) for inputs and targets.

    # Intensity augmentation: append to train_raw_value_transforms
    extra_dl_kwargs = {}

    # Memory management: TensorStore cache bounding
    # cellmap-data >= 2026.2.27 has built-in tensorstore_cache_bytes on
    # CellMapDataLoader (default 2 GiB). It also reads the env var
    # CELLMAP_TENSORSTORE_CACHE_BYTES. No manual ts.Context needed.

    # persistent_workers control — safe for both 2D and 3D now that
    # TensorStore cache is bounded (no unbounded memory accumulation).
    if args.persistent_workers != "auto":
        extra_dl_kwargs["persistent_workers"] = (args.persistent_workers == "true")
        if main_process:
            print(f"DataLoader persistent_workers={extra_dl_kwargs['persistent_workers']}")
    # auto: let PyTorch DataLoader default (True when num_workers>0)

    if args.intensity_aug:
        from training.transforms.intensity import IntensityAugmentation
        intensity_aug = IntensityAugmentation()
        custom_train_transforms = T.Compose([
            T.ToDtype(torch.float, scale=True),
            NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            intensity_aug,
        ])
        extra_dl_kwargs["train_raw_value_transforms"] = custom_train_transforms
        if main_process:
            print(f"Intensity augmentation enabled: {intensity_aug}")

    # Class-aware sampling: build custom sampler callable
    # We need to create it AFTER the dataloader is built (needs access to
    # the dataset's class_counts). So we first build with default sampler,
    # then replace it.
    train_loader, val_loader = get_dataloader(
        datasplit_path=args.datasplit_path,
        classes=classes,
        batch_size=args.batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=spatial_transforms,
        iterations_per_epoch=args.iterations_per_epoch,
        random_validation=True,
        device="cpu",
        weighted_sampler=args.weighted_sampler,
        num_workers=args.num_workers,
        **extra_dl_kwargs,
    )

    # Replace sampler with class-aware version if requested
    if args.class_aware_sampling:
        from training.samplers.crop_weights import make_class_aware_sampler

    # --- Monkey-patch CellMapImage._clear_array_cache to also clear _current_coords ---
    # PR #64 (cellmap-data 2026.2.28.30) clears the xarray array cache in
    # __getitem__'s finally block, but _current_coords (128³×3 float64 meshgrid,
    # ~48 MB) is still set in apply_spatial_transforms() and never cleared.
    # This patch makes the cleanup happen automatically in every __getitem__ call.
    from cellmap_data.image import CellMapImage
    _orig_clear = CellMapImage._clear_array_cache
    def _patched_clear_array_cache(self):
        _orig_clear(self)
        self._current_coords = None
        self._current_spatial_transforms = None
    CellMapImage._clear_array_cache = _patched_clear_array_cache
    if main_process:
        print("[fix] Monkey-patched CellMapImage._clear_array_cache to also clear _current_coords")

    if args.class_aware_sampling:
        sampler_fn = make_class_aware_sampler(
            dataset=train_loader.dataset,
            iterations_per_epoch=args.iterations_per_epoch,
            batch_size=args.batch_size,
            blend_ratio=0.7,
        )
        train_loader.sampler = sampler_fn
        train_loader.refresh()
        if main_process:
            print("Class-aware crop weighting sampler enabled (blend=0.7)")
    if main_process:
        print(f"Train loader: {len(train_loader.loader)} batches/epoch")
        if val_loader is not None:
            print(f"Val loader: {len(val_loader.loader)} batches")

    # === Model ===
    model_kwargs = {"num_classes": num_classes, "in_channels": 1}
    model_kwargs.update(args.model_kwargs)
    if "img_size" not in model_kwargs and "3d" in args.model:
        model_kwargs["img_size"] = tuple(args.input_shape)

    # Deep supervision: set dsdepth for SegResNet-style models
    if args.deep_supervision and "segresnet" in args.model:
        model_kwargs.setdefault("dsdepth", 4)

    # Bias init: prevent BCE collapse on sparse targets (RetinaNet-style)
    if args.bias_init is not None:
        model_kwargs["bias_init"] = args.bias_init

    model = build_model(args.model, **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if main_process:
        if args.bias_init is not None:
            print(f"Model: {args.model} ({n_params:,} trainable params, bias_init={args.bias_init})")
        else:
            print(f"Model: {args.model} ({n_params:,} trainable params)")
    model = model.to(device)

    # === DDP model wrapping ===
    if use_ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if main_process:
            print(f"Model wrapped in DDP (device_ids=[{local_rank}])")

    # === EMA ===
    # EMA tracks the unwrapped model parameters
    ema_model = None
    if args.ema:
        unwrapped = model.module if use_ddp else model
        ema_model = ModelEMA(unwrapped, decay=args.ema_decay)
        if main_process:
            print(f"EMA enabled (decay={args.ema_decay})")

    # === Loss ===
    loss_kwargs = {"num_classes": num_classes}
    loss_kwargs.update(args.loss_kwargs)
    loss_fn = build_loss(args.loss, **loss_kwargs)

    # Wrap with deep supervision if enabled
    if args.deep_supervision:
        from training.losses.partial_annotation import PartialAnnotationDeepSupervisionLoss
        loss_fn = PartialAnnotationDeepSupervisionLoss(
            base_loss=loss_fn,
            weights=args.ds_weights,
        )
        if main_process:
            print(f"Deep supervision enabled (weights={args.ds_weights})")
    if hasattr(loss_fn, 'to'):
        loss_fn = loss_fn.to(device)
    if main_process:
        print(f"Loss: {args.loss}")

    # Check if loss supports annotation mask / foreground mask
    has_annotation_mask = hasattr(loss_fn, 'set_annotation_mask')
    has_foreground_mask = hasattr(loss_fn, 'set_foreground_mask')
    if main_process:
        print(f"  annotation_mask support: {has_annotation_mask}")
        print(f"  foreground_mask support: {has_foreground_mask}")
        print(f"  foreground_mask enabled: {args.use_foreground_mask}")

    # === Optimizer ===
    optimizer = torch.optim.RAdam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # === Scheduler ===
    scheduler = None
    if args.scheduler == "cosine":
        # Warmup + cosine annealing
        total_steps = args.epochs * args.iterations_per_epoch
        warmup_steps = args.warmup_epochs * args.iterations_per_epoch

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.epochs // 3), gamma=0.1
        )

    # === Mixed precision ===
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device == "cuda")

    # === Resume from checkpoint ===
    start_epoch = 0
    n_iter = 0
    best_val_loss = float("inf")
    best_val_dice = -1.0
    latest_ckpt = ckpt_dir / "latest.pth"
    if latest_ckpt.exists():
        if main_process:
            print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        # For DDP, load into the unwrapped model
        state_dict = ckpt["model_state_dict"]
        if use_ddp:
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        n_iter = ckpt["n_iter"]
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ema_model is not None and "ema_state_dict" in ckpt:
            ema_model.load_state_dict(ckpt["ema_state_dict"])
        if main_process:
            print(f"  Resumed at epoch {start_epoch}, iter {n_iter}")
            if "best_val_loss" in ckpt:
                best_val_loss = ckpt["best_val_loss"]
                print(f"  Restored best_val_loss={best_val_loss:.4f}")
            if "best_val_dice" in ckpt:
                best_val_dice = ckpt["best_val_dice"]
                print(f"  Restored best_val_dice={best_val_dice:.4f}")
        else:
            # Non-main ranks also need the restored metrics for consistent logic
            if "best_val_loss" in ckpt:
                best_val_loss = ckpt["best_val_loss"]
            if "best_val_dice" in ckpt:
                best_val_dice = ckpt["best_val_dice"]

    # === TensorBoard (rank 0 only) ===
    writer = SummaryWriter(str(log_dir)) if main_process else None

    # Get data keys
    input_keys = list(train_loader.dataset.input_arrays.keys())
    target_keys = list(train_loader.dataset.target_arrays.keys())

    # === Training ===
    if main_process:
        print(f"\n{'='*60}")
        print(f"Starting training: {args.experiment_name}")
        print(f"  Model: {args.model} | Loss: {args.loss}")
        print(f"  Epochs: {args.epochs} | Iters/epoch: {args.iterations_per_epoch}")
        print(f"  Batch size: {args.batch_size}" + (f" x {world_size} GPUs = {args.batch_size * world_size} effective" if use_ddp else "") + f" | LR: {args.learning_rate}")
        print(f"  Input shape: {args.input_shape} | Scale: {args.input_scale}")
        print(f"{'='*60}\n")

    # Initialize best metrics (only if not restored from checkpoint above)
    if best_val_loss == float("inf") and best_val_dice == -1.0:
        pass  # Already initialized at declaration
    # (If checkpoint restored values, they are already set)

    # --- Helper: clear leaked per-iteration state from CellMapImage objects ---
    # cellmap-data >=2026.2.28.30 (PR #64) clears the xarray `array` cache
    # in __getitem__'s finally block and keeps `_ts_store` alive (cheap).
    # However, `_current_coords` is still set in apply_spatial_transforms()
    # and never cleared.  With spatial_transforms including rotation, each
    # call stores a 128³×3 float64 meshgrid (~48 MB) that persists on the
    # CellMapImage instance.  Over 98 images/step × 48 MB = ~4.7 GB/step
    # of unbounded linear growth.  We clear it here after each training step.
    def _clear_image_leaks(dataset):
        """Clear _current_coords from all CellMapImage objects after each step."""
        from cellmap_data import CellMapImage
        try:
            datasets = dataset.datasets  # CellMapMultiDataset.datasets
        except AttributeError:
            datasets = [dataset]
        for ds in datasets:
            # Clear input sources
            for src in getattr(ds, 'input_sources', {}).values():
                if isinstance(src, CellMapImage):
                    src._current_coords = None
                    src._current_spatial_transforms = None
            # Clear target sources (dict of class→CellMapImage)
            for target_dict in getattr(ds, 'target_sources', {}).values():
                if isinstance(target_dict, dict):
                    for src in target_dict.values():
                        if isinstance(src, CellMapImage):
                            src._current_coords = None
                            src._current_spatial_transforms = None
                elif isinstance(target_dict, CellMapImage):
                    target_dict._current_coords = None
                    target_dict._current_spatial_transforms = None

    # --- Memory debugging (mirrors upstream CSC train.py debug_memory feature) ---
    debug_memory = args.debug_memory
    memory_log_steps = int(os.environ.get("MEMORY_LOG_STEPS", "100"))
    if debug_memory:
        try:
            import objgraph
        except ImportError:
            if main_process:
                print("WARNING: objgraph not installed. Install with 'pip install objgraph' "
                      "to enable full memory debugging. Falling back to RSS-only.")
            objgraph = None
        if main_process:
            rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB→MB on Linux
            print(f"\n[mem-debug] Baseline before training loop:", flush=True)
            print(f"[mem-debug]   RSS = {rss_mb:.0f} MB", flush=True)
            print(f"[mem-debug]   RSS = {rss_mb:.0f} MB", file=sys.stderr, flush=True)
            if objgraph is not None:
                # Establish baseline — capture current object counts
                objgraph.show_growth(limit=5, file=io.StringIO())  # prime baseline
                print(f"[mem-debug]   objgraph baseline primed ({memory_log_steps}-step interval)", flush=True)
            print(flush=True)
    else:
        objgraph = None  # ensure defined for later checks

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if hasattr(loss_fn, 'train'):
            loss_fn.train()

        train_loader.refresh()
        loader = iter(train_loader.loader)

        epoch_loss = 0.0
        optimizer.zero_grad()

        # Only show progress bar on rank 0
        step_iter = range(args.iterations_per_epoch)
        if main_process:
            epoch_bar = tqdm(
                step_iter,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                dynamic_ncols=True,
            )
        else:
            epoch_bar = step_iter

        for step in epoch_bar:
            batch = next(loader)
            n_iter += 1

            # Get inputs and targets
            inputs = batch[input_keys[0]].to(device)
            targets = batch[target_keys[0]].to(device)

            # Compute annotation mask from NaN pattern
            annotation_mask = get_annotation_mask_from_targets(targets)

            # Replace NaN with 0 in targets for loss computation
            targets_clean = targets.nan_to_num(0.0)

            # Set masks on loss function if supported
            if has_annotation_mask:
                loss_fn.set_annotation_mask(annotation_mask)

            if has_foreground_mask and args.use_foreground_mask:
                fg_mask = (inputs.abs().amax(dim=1, keepdim=True) > FG_THRESHOLD)
                loss_fn.set_foreground_mask(fg_mask)

            # Forward + backward with AMP
            with torch.amp.autocast('cuda', enabled=args.amp and device == "cuda"):
                outputs = model(inputs)

                # Deep supervision: pass full output list to DS loss,
                # otherwise take first element
                if args.deep_supervision and isinstance(outputs, (list, tuple)):
                    logits = outputs  # DS loss handles the list
                elif isinstance(outputs, (list, tuple)):
                    logits = outputs[0]
                else:
                    logits = outputs

                # For losses that don't support annotation mask (e.g., BCE),
                # apply NaN masking the CSC way
                if has_annotation_mask:
                    loss = loss_fn(logits, targets_clean)
                else:
                    # CSC-style: loss on non-NaN values only
                    raw_loss = loss_fn(logits, targets_clean)
                    if raw_loss.ndim > 0:
                        nan_mask = targets.isnan().logical_not()
                        raw_loss = (raw_loss * nan_mask).sum() / nan_mask.sum().clamp(min=1)
                    loss = raw_loss

                loss = loss / args.gradient_accumulation_steps

            # Skip NaN/Inf losses to prevent poisoning model weights
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()

            if args.max_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                # EMA update (track unwrapped model for DDP compatibility)
                if ema_model is not None:
                    unwrapped_m = model.module if use_ddp else model
                    ema_model.update(unwrapped_m)

            loss_val = loss.item() * args.gradient_accumulation_steps
            epoch_loss += loss_val

            if main_process:
                epoch_bar.set_postfix({
                    "loss": f"{loss_val:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })
                writer.add_scalar("train/loss", loss_val, n_iter)

            # Periodic memory monitoring & GPU cache clear
            if step % max(1, memory_log_steps) == 0:
                if debug_memory and main_process:
                    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                    # Read CURRENT (not peak) RSS from /proc
                    try:
                        with open("/proc/self/status") as _pf:
                            for _line in _pf:
                                if _line.startswith("VmRSS:"):
                                    cur_rss_mb = int(_line.split()[1]) / 1024  # kB→MB
                                    break
                            else:
                                cur_rss_mb = peak_mb
                    except OSError:
                        cur_rss_mb = peak_mb
                    gpu_mb = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
                    msg = (f"[mem-debug] iter={n_iter} (ep{epoch+1} step{step}) "
                           f"CurRSS={cur_rss_mb:.0f}MB  PeakRSS={peak_mb:.0f}MB  GPU_peak={gpu_mb:.0f}MB")
                    print(msg, flush=True)
                    print(msg, file=sys.stderr, flush=True)
                    if objgraph is not None:
                        objgraph.show_growth(limit=5)
                    writer.add_scalar("mem/rss_cur_mb", cur_rss_mb, n_iter)
                    writer.add_scalar("mem/rss_peak_mb", peak_mb, n_iter)
                    writer.add_scalar("mem/gpu_peak_mb", gpu_mb, n_iter)
                # Clear leaked _current_coords to prevent RSS growth
                _clear_image_leaks(train_loader.dataset)
                gc.collect()
                torch.cuda.empty_cache()

        avg_train_loss = epoch_loss / args.iterations_per_epoch
        # Reduce train loss across ranks for accurate logging
        if use_ddp:
            avg_train_loss = reduce_value(avg_train_loss, op="mean")
        if main_process:
            writer.add_scalar("train/epoch_loss", avg_train_loss, epoch + 1)
            writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # === Validation ===
        if val_loader is not None and (epoch + 1) % args.val_every_n_epochs == 0:
            # For eval, use EMA shadow (unwrapped) or the unwrapped model
            unwrapped = model.module if use_ddp else model
            eval_model = ema_model.shadow if ema_model is not None else unwrapped
            eval_model.eval()
            if hasattr(loss_fn, 'eval'):
                loss_fn.eval()

            val_loss = 0.0
            val_steps = 0
            val_start = time.time()

            # Per-class Dice accumulators: TP, FP, FN for each of num_classes channels
            dice_tp = torch.zeros(num_classes, device=device)
            dice_fp = torch.zeros(num_classes, device=device)
            dice_fn = torch.zeros(num_classes, device=device)

            val_loader.refresh()
            torch.cuda.empty_cache()

            # Capture first batch for visualization (rank 0 only)
            vis_sample = None  # Will hold (input, pred, gt) for first sample

            with torch.no_grad():
                for batch in val_loader.loader:
                    inputs = batch[input_keys[0]].to(device)
                    targets = batch[target_keys[0]].to(device)

                    annotation_mask = get_annotation_mask_from_targets(targets)
                    targets_clean = targets.nan_to_num(0.0)

                    if has_annotation_mask:
                        loss_fn.set_annotation_mask(annotation_mask)
                    if has_foreground_mask and args.use_foreground_mask:
                        fg_mask = (inputs.abs().amax(dim=1, keepdim=True) > FG_THRESHOLD)
                        loss_fn.set_foreground_mask(fg_mask)

                    with torch.amp.autocast('cuda', enabled=args.amp and device == "cuda"):
                        outputs = eval_model(inputs)

                        # Deep supervision: pass full output list to DS loss,
                        # otherwise take first element
                        if args.deep_supervision and isinstance(outputs, (list, tuple)):
                            logits = outputs  # DS loss handles the list
                        elif isinstance(outputs, (list, tuple)):
                            logits = outputs[0]
                        else:
                            logits = outputs

                        if has_annotation_mask:
                            vloss = loss_fn(logits, targets_clean)
                        else:
                            raw_loss = loss_fn(logits, targets_clean)
                            if raw_loss.ndim > 0:
                                nan_mask = targets.isnan().logical_not()
                                raw_loss = (raw_loss * nan_mask).sum() / nan_mask.sum().clamp(min=1)
                            vloss = raw_loss

                    # Skip NaN/Inf in validation too
                    if torch.isfinite(vloss):
                        val_loss += vloss.item()
                        val_steps += 1

                    # --- Per-class Dice accumulation ---
                    # Use the single-scale logits for Dice (not DS list)
                    dice_logits = logits[0] if isinstance(logits, (list, tuple)) else logits
                    preds = (torch.sigmoid(dice_logits.float()) > 0.5).float()  # (B, C, *spatial)
                    gt = targets_clean.float()  # (B, C, *spatial)
                    # Mask: only count annotated voxels (non-NaN in original targets)
                    # annotation_mask is (B, C) — per-channel, but we also need spatial NaN mask
                    valid = targets.isnan().logical_not().float()  # (B, C, *spatial)
                    spatial_dims = tuple(range(2, preds.ndim))
                    # TP/FP/FN per class, masked by valid voxels
                    dice_tp += (preds * gt * valid).sum(dim=(0, *spatial_dims))
                    dice_fp += (preds * (1 - gt) * valid).sum(dim=(0, *spatial_dims))
                    dice_fn += ((1 - preds) * gt * valid).sum(dim=(0, *spatial_dims))

                    # Capture first batch's first sample for visualization
                    if vis_sample is None and main_process:
                        vis_sample = (
                            inputs[0].detach().cpu(),         # (1, *spatial) or (1, D, H, W)
                            preds[0].detach().cpu(),          # (C, *spatial)
                            gt[0].detach().cpu(),             # (C, *spatial)
                        )

                    if time.time() - val_start > args.validation_time_limit:
                        break

            avg_val_loss = val_loss / max(val_steps, 1)

            # Reduce val loss and Dice accumulators across ranks
            if use_ddp:
                avg_val_loss = reduce_value(avg_val_loss, op="mean")
                # Sum TP/FP/FN across ranks
                torch.distributed.all_reduce(dice_tp)
                torch.distributed.all_reduce(dice_fp)
                torch.distributed.all_reduce(dice_fn)

            # Compute per-class Dice: 2*TP / (2*TP + FP + FN)
            denom = 2 * dice_tp + dice_fp + dice_fn
            per_class_dice = torch.where(
                denom > 0,
                2 * dice_tp / denom,
                torch.zeros_like(denom),  # 0 if no voxels seen for this class
            )  # (num_classes,)
            # Mean Dice over classes that had any ground truth or prediction
            has_voxels = denom > 0
            mean_dice = per_class_dice[has_voxels].mean().item() if has_voxels.any() else 0.0

            if main_process:
                writer.add_scalar("val/loss", avg_val_loss, epoch + 1)
                writer.add_scalar("val/mean_dice", mean_dice, epoch + 1)
                # Log per-class Dice
                for ci, cname in enumerate(classes):
                    writer.add_scalar(f"val_dice/{cname}", per_class_dice[ci].item(), epoch + 1)
                # Print summary
                n_active = has_voxels.sum().item()
                print(f"  Val loss: {avg_val_loss:.4f} | Mean Dice: {mean_dice:.4f} "
                      f"({n_active}/{num_classes} classes) [{val_steps} batches, "
                      f"{time.time() - val_start:.0f}s]")
                # Print top-5 and bottom-5 classes by Dice (among active)
                if n_active > 0:
                    active_indices = has_voxels.nonzero(as_tuple=True)[0]
                    active_dices = per_class_dice[active_indices]
                    sorted_idx = active_dices.argsort(descending=True)
                    top_n = min(5, len(sorted_idx))
                    top_str = ", ".join(
                        f"{classes[active_indices[sorted_idx[i]].item()]}={active_dices[sorted_idx[i]]:.3f}"
                        for i in range(top_n)
                    )
                    bot_str = ", ".join(
                        f"{classes[active_indices[sorted_idx[-i-1]].item()]}={active_dices[sorted_idx[-i-1]]:.3f}"
                        for i in range(top_n)
                    )
                    print(f"    Top-5: {top_str}")
                    print(f"    Bot-5: {bot_str}")

                # --- Log validation segmentation images to TensorBoard ---
                if vis_sample is not None:
                    try:
                        vis_input, vis_pred, vis_gt = vis_sample  # (1,*sp), (C,*sp), (C,*sp)
                        is_3d = vis_input.ndim == 4  # (1, D, H, W)

                        # For 3D: take center slice along depth axis
                        if is_3d:
                            mid = vis_input.shape[1] // 2
                            vis_input = vis_input[:, mid, :, :]   # (1, H, W)
                            vis_pred = vis_pred[:, mid, :, :]     # (C, H, W)
                            vis_gt = vis_gt[:, mid, :, :]         # (C, H, W)

                        # Normalize EM input to [0,1]
                        img_hw = vis_input[0]  # (H, W)
                        img_min, img_max = img_hw.min(), img_hw.max()
                        if img_max > img_min:
                            img_hw = (img_hw - img_min) / (img_max - img_min)

                        # --- Distinct color per organelle class ---
                        # 20 perceptually distinct colors (recycled if >20 classes)
                        CLASS_COLORS = [
                            (1.0, 0.0, 0.0),    # red - ecs
                            (0.0, 1.0, 0.0),    # green - pm
                            (0.0, 0.5, 1.0),    # blue - mito_mem
                            (1.0, 1.0, 0.0),    # yellow - mito_lum
                            (1.0, 0.0, 1.0),    # magenta - mito_ribo
                            (0.0, 1.0, 1.0),    # cyan - golgi_mem
                            (1.0, 0.5, 0.0),    # orange - golgi_lum
                            (0.5, 0.0, 1.0),    # purple - ves_mem
                            (0.0, 1.0, 0.5),    # spring green - ves_lum
                            (1.0, 0.0, 0.5),    # rose - endo_mem
                            (0.5, 1.0, 0.0),    # lime - endo_lum
                            (0.0, 0.5, 0.5),    # teal - lyso_mem
                            (0.8, 0.4, 0.0),    # brown - lyso_lum
                            (0.6, 0.0, 0.0),    # dark red - ld_mem
                            (0.0, 0.6, 0.0),    # dark green - ld_lum
                            (0.4, 0.4, 1.0),    # light blue - er_mem
                            (1.0, 0.8, 0.6),    # peach - er_lum
                            (0.6, 0.3, 0.6),    # mauve - eres_mem
                            (0.3, 0.8, 0.8),    # light teal - eres_lum
                            (0.9, 0.9, 0.0),    # gold - ne_mem
                        ]

                        H, W = img_hw.shape
                        alpha = 0.55  # overlay opacity

                        def _make_color_overlay(masks, label="pred"):
                            """Build (3,H,W) EM image with colored class masks overlaid."""
                            # Start with grayscale EM as RGB
                            canvas = torch.stack([img_hw, img_hw, img_hw], dim=0)  # (3,H,W)
                            for ci in range(masks.shape[0]):
                                m = masks[ci]  # (H,W) binary
                                if m.sum() == 0:
                                    continue
                                r, g, b = CLASS_COLORS[ci % len(CLASS_COLORS)]
                                color = torch.tensor([r, g, b], dtype=canvas.dtype).view(3, 1, 1)
                                mask_3c = m.unsqueeze(0).expand(3, -1, -1)  # (3,H,W)
                                canvas = torch.where(
                                    mask_3c > 0.5,
                                    canvas * (1 - alpha) + color * alpha,
                                    canvas,
                                )
                            return canvas.clamp(0, 1)

                        pred_overlay = _make_color_overlay(vis_pred, "pred")
                        gt_overlay = _make_color_overlay(vis_gt, "gt")

                        # Log to TensorBoard
                        writer.add_image("val_vis/input", img_hw.unsqueeze(0), epoch + 1)
                        writer.add_image("val_vis/prediction", pred_overlay, epoch + 1)
                        writer.add_image("val_vis/ground_truth", gt_overlay, epoch + 1)

                        # --- Save to disk as PNG ---
                        try:
                            from torchvision.utils import save_image
                            ep_dir = val_img_dir / f"epoch_{epoch+1:04d}"
                            ep_dir.mkdir(exist_ok=True)
                            save_image(img_hw.unsqueeze(0), ep_dir / "input.png")
                            save_image(pred_overlay, ep_dir / "prediction.png")
                            save_image(gt_overlay, ep_dir / "ground_truth.png")

                            # Also save a legend text file
                            legend_path = ep_dir / "legend.txt"
                            if not legend_path.exists():
                                with open(legend_path, "w") as lf:
                                    lf.write("Class Color Legend\n")
                                    lf.write("=" * 40 + "\n")
                                    for ci, cname in enumerate(classes):
                                        r, g, b = CLASS_COLORS[ci % len(CLASS_COLORS)]
                                        lf.write(f"{cname:20s}  RGB({r:.1f}, {g:.1f}, {b:.1f})\n")
                        except Exception as e2:
                            print(f"  [warn] Failed to save val images to disk: {e2}")

                    except Exception as e:
                        print(f"  [warn] Failed to log val images: {e}")

                # Save best model (EMA if available, else regular unwrapped)
                is_new_best = False
                if args.best_metric == "val_dice":
                    if val_steps > 0 and mean_dice > best_val_dice:
                        best_val_dice = mean_dice
                        is_new_best = True
                else:  # val_loss
                    if val_steps > 0 and avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        is_new_best = True
                # Always track both
                if val_steps > 0:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                    if mean_dice > best_val_dice:
                        best_val_dice = mean_dice
                if is_new_best:
                    best_state = ema_model.state_dict() if ema_model is not None else unwrapped.state_dict()
                    torch.save(best_state, ckpt_dir / "best.pth")
                    metric_str = f"mean_dice={mean_dice:.4f}" if args.best_metric == "val_dice" else f"val_loss={avg_val_loss:.4f}"
                    print(f"  New best model saved ({metric_str})")

        # === Save checkpoint (rank 0 only) ===
        if main_process:
            unwrapped = model.module if use_ddp else model
            ckpt_data = {
                "epoch": epoch + 1,
                "n_iter": n_iter,
                "model_state_dict": unwrapped.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train_loss,
                "best_val_loss": best_val_loss,
                "best_val_dice": best_val_dice,
            }
            if scheduler is not None:
                ckpt_data["scheduler_state_dict"] = scheduler.state_dict()
            if ema_model is not None:
                ckpt_data["ema_state_dict"] = ema_model.state_dict()
            torch.save(ckpt_data, ckpt_dir / "latest.pth")

            # Save periodic checkpoints
            if (epoch + 1) % 10 == 0:
                torch.save(ckpt_data, ckpt_dir / f"epoch_{epoch+1}.pth")

        # Sync all ranks after checkpoint save
        sync_across_processes()

    if main_process:
        writer.close()
        print(f"\nTraining complete.")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Best mean Dice: {best_val_dice:.4f}")
        print(f"  Best checkpoint metric: {args.best_metric}")
        print(f"  Outputs saved to: {run_dir}")

    # === DDP Cleanup ===
    cleanup_ddp()


if __name__ == "__main__":
    args = parse_args()
    train(args)
