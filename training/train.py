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
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Optional

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

    # OHEM
    parser.add_argument("--ohem_ratio", type=float, default=0.0,
                        help="Online hard example mining: keep top-K%% hardest voxels (0=disabled)")

    # Scheduler
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["none", "cosine", "step"],
                        help="LR scheduler type")
    parser.add_argument("--warmup_epochs", type=int, default=5)

    # Validation
    parser.add_argument("--validation_time_limit", type=int, default=120,
                        help="Max seconds for validation per epoch")
    parser.add_argument("--val_every_n_epochs", type=int, default=1,
                        help="Run validation every N epochs")

    # Mixed precision
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Use automatic mixed precision (default: True)")
    parser.add_argument("--no_amp", action="store_true", default=False)

    # Misc
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)

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
    """Main training loop."""

    # === Setup ===
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # === Classes ===
    classes = get_tested_classes()
    num_classes = len(classes)
    print(f"Training on {num_classes} classes: {classes[:5]}... (showing first 5)")

    # === Output directories ===
    run_dir = Path(args.run_dir) / args.experiment_name
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "tensorboard"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {config_path}")

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

    print("Loading data...")
    # Keep data on CPU during loading to avoid CUDA OOM from pre-moving
    # all ~784 dataset EmptyImage tensors to GPU. The training loop already
    # handles per-batch .to(device) for inputs and targets.
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
    )
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

    model = build_model(args.model, **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model} ({n_params:,} trainable params)")
    model = model.to(device)

    # === EMA ===
    ema_model = None
    if args.ema:
        ema_model = ModelEMA(model, decay=args.ema_decay)
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
        print(f"Deep supervision enabled (weights={args.ds_weights})")
    if hasattr(loss_fn, 'to'):
        loss_fn = loss_fn.to(device)
    print(f"Loss: {args.loss}")

    # Check if loss supports annotation mask / foreground mask
    has_annotation_mask = hasattr(loss_fn, 'set_annotation_mask')
    has_foreground_mask = hasattr(loss_fn, 'set_foreground_mask')
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
    latest_ckpt = ckpt_dir / "latest.pth"
    if latest_ckpt.exists():
        print(f"Resuming from {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"]
        n_iter = ckpt["n_iter"]
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ema_model is not None and "ema_state_dict" in ckpt:
            ema_model.load_state_dict(ckpt["ema_state_dict"])
        print(f"  Resumed at epoch {start_epoch}, iter {n_iter}")

    # === TensorBoard ===
    writer = SummaryWriter(str(log_dir))

    # Get data keys
    input_keys = list(train_loader.dataset.input_arrays.keys())
    target_keys = list(train_loader.dataset.target_arrays.keys())

    # === Training ===
    print(f"\n{'='*60}")
    print(f"Starting training: {args.experiment_name}")
    print(f"  Model: {args.model} | Loss: {args.loss}")
    print(f"  Epochs: {args.epochs} | Iters/epoch: {args.iterations_per_epoch}")
    print(f"  Batch size: {args.batch_size} | LR: {args.learning_rate}")
    print(f"  Input shape: {args.input_shape} | Scale: {args.input_scale}")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        if hasattr(loss_fn, 'train'):
            loss_fn.train()

        train_loader.refresh()
        loader = iter(train_loader.loader)

        epoch_loss = 0.0
        optimizer.zero_grad()

        epoch_bar = tqdm(
            range(args.iterations_per_epoch),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            dynamic_ncols=True,
        )

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
                # EMA update
                if ema_model is not None:
                    ema_model.update(model)

            loss_val = loss.item() * args.gradient_accumulation_steps
            epoch_loss += loss_val

            epoch_bar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            writer.add_scalar("train/loss", loss_val, n_iter)

        avg_train_loss = epoch_loss / args.iterations_per_epoch
        writer.add_scalar("train/epoch_loss", avg_train_loss, epoch + 1)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch + 1)

        # === Validation ===
        if val_loader is not None and (epoch + 1) % args.val_every_n_epochs == 0:
            eval_model = ema_model.shadow if ema_model is not None else model
            eval_model.eval()
            if hasattr(loss_fn, 'eval'):
                loss_fn.eval()

            val_loss = 0.0
            val_steps = 0
            val_start = time.time()

            val_loader.refresh()
            torch.cuda.empty_cache()

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

                    if time.time() - val_start > args.validation_time_limit:
                        break

            avg_val_loss = val_loss / max(val_steps, 1)
            writer.add_scalar("val/loss", avg_val_loss, epoch + 1)
            print(f"  Val loss: {avg_val_loss:.4f} ({val_steps} batches)")

            # Save best model (EMA if available, else regular)
            # Skip when val_steps=0 — avg_val_loss=0.0 is fake, not a real improvement
            if val_steps > 0 and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = ema_model.state_dict() if ema_model is not None else model.state_dict()
                torch.save(best_state, ckpt_dir / "best.pth")
                print(f"  New best model saved (val_loss={avg_val_loss:.4f})")

        # === Save checkpoint ===
        ckpt_data = {
            "epoch": epoch + 1,
            "n_iter": n_iter,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": avg_train_loss,
        }
        if scheduler is not None:
            ckpt_data["scheduler_state_dict"] = scheduler.state_dict()
        if ema_model is not None:
            ckpt_data["ema_state_dict"] = ema_model.state_dict()
        torch.save(ckpt_data, ckpt_dir / "latest.pth")

        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data, ckpt_dir / f"epoch_{epoch+1}.pth")

    writer.close()
    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Outputs saved to: {run_dir}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
