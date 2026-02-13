"""
Utility functions for MONAI CellMap training pipeline.

DDP setup/teardown, checkpointing, LR scheduling, seeding, metrics.

Reference: IMPLEMENTATION_SPEC.md §5.7
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


# ---------------------------------------------------------------------------
# DDP
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialize DDP process group via torchrun environment variables.

    Returns:
        (local_rank, world_size, global_rank)
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = int(os.environ.get("RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return local_rank, world_size, global_rank


def cleanup_ddp():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Check if this is the main (rank 0) process."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return True


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """All-reduce a tensor and average across processes."""
    if world_size <= 1:
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def set_seed(seed: int, rank: int = 0):
    """Set random seeds for reproducibility. Offset by rank for DDP."""
    s = seed + rank
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# LR Scheduling
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup.

    Args:
        optimizer: Wrapped optimizer.
        warmup_steps: Number of warmup steps (linear ramp from 0 to base_lr).
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate at the end of cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-7,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / max(self.warmup_steps, 1)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        return [
            max(self.min_lr, base_lr * scale)
            for base_lr in self.base_lrs
        ]


def build_scheduler(optimizer, cfg, steps_per_epoch: int):
    """Build LR scheduler from config.

    Args:
        optimizer: Wrapped optimizer.
        cfg: Config with schedule, warmup, epochs.
        steps_per_epoch: Number of optimizer steps per epoch.

    Returns:
        LR scheduler (step-level).
    """
    total_steps = getattr(cfg, "epochs", 100) * steps_per_epoch
    warmup_frac = getattr(cfg, "warmup", 0.05)
    warmup_steps = int(total_steps * warmup_frac)

    schedule_type = getattr(cfg, "schedule", "cosine")
    if schedule_type == "cosine":
        return CosineWarmupScheduler(
            optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")


def build_optimizer(model, cfg):
    """Build optimizer from config.

    Args:
        model: nn.Module (or DDP-wrapped).
        cfg: Config with optimizer, lr, weight_decay.

    Returns:
        Optimizer instance.
    """
    opt_name = getattr(cfg, "optimizer", "AdamW")
    lr = getattr(cfg, "lr", 1e-3)
    wd = getattr(cfg, "weight_decay", 1e-5)

    # Get parameters from the unwrapped model
    params = model.module.parameters() if isinstance(model, DDP) else model.parameters()

    if opt_name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    elif opt_name == "Adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    elif opt_name == "SGD":
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    step: int,
    best_metric: float,
    output_dir: str,
    is_best: bool = False,
    save_weights_only: bool = False,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model (possibly DDP-wrapped).
        optimizer: Optimizer state.
        scheduler: LR scheduler state.
        epoch: Current epoch.
        step: Current global step.
        best_metric: Best validation metric so far.
        output_dir: Directory to save to.
        is_best: Whether this is the best model so far.
        save_weights_only: If True, only save model weights.
    """
    if not is_main_process():
        return

    os.makedirs(output_dir, exist_ok=True)

    # Unwrap DDP
    state_dict = (
        model.module.state_dict()
        if isinstance(model, DDP)
        else model.state_dict()
    )

    if save_weights_only:
        checkpoint = {"model": state_dict}
    else:
        checkpoint = {
            "model": state_dict,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_metric": best_metric,
        }

    # Save latest
    path = os.path.join(output_dir, "checkpoint_last.pth")
    torch.save(checkpoint, path)

    # Save best
    if is_best:
        best_path = os.path.join(output_dir, "checkpoint_best.pth")
        torch.save(checkpoint, best_path)

    # Epoch checkpoint
    epoch_path = os.path.join(output_dir, f"checkpoint_epoch{epoch:04d}.pth")
    torch.save(checkpoint, epoch_path)


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    checkpoint_path: str = "",
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load checkpoint, returning metadata dict.

    Args:
        model: Model (possibly DDP-wrapped).
        optimizer: Optional optimizer to restore.
        scheduler: Optional scheduler to restore.
        checkpoint_path: Path to .pth file.
        device: Device to map tensors to.

    Returns:
        Dict with 'epoch', 'step', 'best_metric' (if available).
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"epoch": 0, "step": 0, "best_metric": 0.0}

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load model weights
    target = model.module if isinstance(model, DDP) else model
    target.load_state_dict(checkpoint["model"])

    # Load optimizer/scheduler if available
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "best_metric": checkpoint.get("best_metric", 0.0),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_per_channel_dice(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    sigmoid: bool = True,
    threshold: float = 0.5,
    smooth: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-channel Dice score with optional annotation masking.

    Args:
        pred: Logits (B, C, *spatial).
        target: Binary ground truth (B, C, *spatial).
        mask: Annotation mask (B, C) — 1 for annotated, 0 for not.
        sigmoid: Whether to apply sigmoid then threshold.
        threshold: Binarization threshold.
        smooth: Smoothing for Dice.

    Returns:
        (dice_per_channel, valid_counts) — dice_per_channel is (C,) averaged
        over batch, valid_counts is (C,) number of samples with that channel annotated.
    """
    B, C = pred.shape[:2]
    spatial_dims = tuple(range(2, pred.ndim))

    if sigmoid:
        pred_binary = (torch.sigmoid(pred) > threshold).float()
    else:
        pred_binary = (pred > threshold).float()

    target = target.float()

    # Per-sample per-channel Dice
    intersection = (pred_binary * target).sum(dim=spatial_dims)  # (B, C)
    union = pred_binary.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)  # (B, C)
    dice = (2.0 * intersection + smooth) / (union + smooth)  # (B, C)

    if mask is not None:
        mask = mask.to(pred.device)
        # Mask out unannotated channels
        dice = dice * mask  # (B, C)
        valid_counts = mask.sum(dim=0)  # (C,) — how many samples per channel
        # Sum over batch, divide by valid counts
        dice_sum = dice.sum(dim=0)  # (C,)
        dice_per_channel = dice_sum / valid_counts.clamp(min=1.0)
    else:
        valid_counts = torch.full((C,), B, dtype=torch.float32, device=pred.device)
        dice_per_channel = dice.mean(dim=0)  # (C,)

    return dice_per_channel, valid_counts
