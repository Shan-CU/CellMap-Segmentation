"""
Main training loop for MONAI CellMap segmentation pipeline.

Adapted from the CryoET 1st-place winner's training pattern:
- MONAI used as a library (not Auto3DSeg framework)
- DDP via torchrun
- bfloat16 autocast
- Per-channel partial annotation loss
- Cosine schedule with linear warmup

Usage:
    # Single GPU
    python train.py -C cfg_segresnet

    # Multi-GPU DDP
    torchrun --nproc_per_node=7 train.py -C cfg_segresnet

Reference: IMPLEMENTATION_SPEC.md §5.7
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Ensure the experiment directory is on the path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "configs"))

from data.ds_cellmap import CellMapDataset, load_datalist, flat_collate_fn
from data.ds_cellmap import batch_to_device
from models.mdl_cellmap import Net
from utils import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    reduce_tensor,
    set_seed,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    load_checkpoint,
    compute_per_channel_dice,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MONAI CellMap Training")
    parser.add_argument(
        "-C", "--config", type=str, required=True,
        help="Config module name (e.g., cfg_segresnet)",
    )
    parser.add_argument(
        "--fold", type=int, default=-1,
        help="Fold index (-1 = use datalist train/val split)",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def load_config(config_name: str):
    """Dynamically import a config module and return the cfg object."""
    mod = importlib.import_module(config_name)
    return mod.cfg


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.amp.GradScaler,
    cfg,
    epoch: int,
    global_step: int,
    device: torch.device,
    world_size: int,
) -> tuple[float, int]:
    """Run one training epoch.

    Returns:
        (avg_loss, global_step)
    """
    model.train()
    total_loss = 0.0
    n_steps = 0
    grad_accum = getattr(cfg, "grad_accumulation", 1)
    clip_grad = getattr(cfg, "clip_grad", 1.0)
    use_bf16 = getattr(cfg, "bf16", True)

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=not is_main_process())
    for step_in_epoch, batch in enumerate(pbar):
        batch = batch_to_device(batch, device)

        # Forward with autocast
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(batch)
            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            if grad_accum > 1:
                loss = loss / grad_accum

        # Backward
        scaler.scale(loss).backward()

        # Optimizer step after accumulation
        if (step_in_epoch + 1) % grad_accum == 0:
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # Logging
        loss_val = loss.item() * (grad_accum if grad_accum > 1 else 1)
        total_loss += loss_val
        n_steps += 1

        if is_main_process():
            lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

    avg_loss = total_loss / max(n_steps, 1)

    # All-reduce loss across processes
    if world_size > 1:
        loss_tensor = torch.tensor(avg_loss, device=device)
        avg_loss = reduce_tensor(loss_tensor, world_size).item()

    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg,
    device: torch.device,
    world_size: int,
) -> tuple[float, dict]:
    """Run validation and compute per-channel Dice.

    Returns:
        (mean_dice, per_channel_dice_dict)
    """
    model.eval()
    use_bf16 = getattr(cfg, "bf16", True)
    class_names = getattr(cfg, "class_names", [f"ch{i}" for i in range(14)])

    # Accumulators
    dice_sum = torch.zeros(cfg.num_classes, device=device)
    valid_sum = torch.zeros(cfg.num_classes, device=device)
    total_loss = 0.0
    n_steps = 0

    for batch in tqdm(loader, desc="Validation", disable=not is_main_process()):
        batch = batch_to_device(batch, device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            outputs = model(batch)
            logits = outputs["logits"]

        # Compute Dice
        mask = batch.get("annotation_mask", None)
        dice_pc, valid_pc = compute_per_channel_dice(
            logits, batch["target"], mask=mask, sigmoid=True,
        )
        dice_sum += dice_pc * valid_pc  # weighted sum
        valid_sum += valid_pc
        n_steps += 1

    # Average across all validated samples
    if world_size > 1:
        dist.all_reduce(dice_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_sum, op=dist.ReduceOp.SUM)

    per_channel_dice = dice_sum / valid_sum.clamp(min=1.0)

    # Only average over channels that had any annotated samples
    annotated_channels = (valid_sum > 0).float()
    n_annotated = annotated_channels.sum().clamp(min=1.0)
    mean_dice = (per_channel_dice * annotated_channels).sum() / n_annotated

    # Build per-class dict for logging
    dice_dict = {}
    for i, name in enumerate(class_names):
        dice_dict[name] = per_channel_dice[i].item()

    return mean_dice.item(), dice_dict


def main():
    args = parse_args()

    # --- DDP setup ---
    local_rank, world_size, global_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # --- Load config ---
    cfg = load_config(args.config)
    output_dir = getattr(cfg, "output_dir", "/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap")
    os.makedirs(output_dir, exist_ok=True)

    # --- Seed ---
    set_seed(getattr(cfg, "seed", 42), rank=global_rank)

    # --- Data ---
    if is_main_process():
        print(f"Loading datalist from: {cfg.datalist}")
    train_files, val_files = load_datalist(cfg)
    if is_main_process():
        print(f"Train: {len(train_files)} volumes, Val: {len(val_files)} volumes")

    train_dataset = CellMapDataset(train_files, cfg, mode="train")
    val_dataset = CellMapDataset(val_files, cfg, mode="val")

    # --- Samplers ---
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True)
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False)
        if world_size > 1
        else None
    )

    # --- DataLoaders ---
    batch_size = getattr(cfg, "batch_size", 2)
    num_workers = getattr(cfg, "num_workers", 4)
    drop_last = getattr(cfg, "drop_last", True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        collate_fn=flat_collate_fn,
        drop_last=drop_last,
        pin_memory=getattr(cfg, "pin_memory", False),
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=flat_collate_fn,
        drop_last=False,
        pin_memory=getattr(cfg, "pin_memory", False),
        persistent_workers=(num_workers > 0),
    )

    # --- Model ---
    model = Net(cfg).to(device)

    # Sync batch norm if configured
    if getattr(cfg, "syncbn", False) and world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP wrap
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=getattr(cfg, "find_unused_parameters", False),
        )

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader) // getattr(cfg, "grad_accumulation", 1)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    # --- GradScaler (for bf16 autocast — GradScaler is a no-op with bf16 but
    # keeps the code compatible if we switch to fp16) ---
    scaler = torch.amp.GradScaler("cuda", enabled=getattr(cfg, "mixed_precision", False))

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    best_metric = 0.0

    resume_path = args.resume
    if not resume_path:
        # Check for latest checkpoint
        last_ckpt = os.path.join(output_dir, "checkpoint_last.pth")
        if os.path.exists(last_ckpt):
            resume_path = last_ckpt

    if resume_path and os.path.exists(resume_path):
        if is_main_process():
            print(f"Resuming from: {resume_path}")
        meta = load_checkpoint(model, optimizer, scheduler, resume_path, device)
        start_epoch = meta["epoch"] + 1
        global_step = meta["step"]
        best_metric = meta["best_metric"]

    # --- Training Loop ---
    epochs = getattr(cfg, "epochs", 100)
    eval_epochs = getattr(cfg, "eval_epochs", 5)

    if is_main_process():
        print(f"\n{'='*60}")
        print(f"Training: {getattr(cfg, 'name', 'unknown')}")
        print(f"Backbone: {getattr(cfg, 'backbone_type', 'unknown')}")
        print(f"Epochs: {epochs}, Eval every: {eval_epochs}")
        print(f"Batch size: {batch_size} × {world_size} GPUs")
        print(f"Patch size: {getattr(cfg, 'roi_size', [128,128,128])}")
        print(f"Num samples: {getattr(cfg, 'num_samples', 4)}")
        print(f"LR: {getattr(cfg, 'lr', 1e-3)}, Schedule: {getattr(cfg, 'schedule', 'cosine')}")
        print(f"BF16: {getattr(cfg, 'bf16', True)}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")

    # --- TensorBoard ---
    writer = None
    if is_main_process():
        tb_dir = os.path.join(output_dir, "tb")
        writer = SummaryWriter(log_dir=tb_dir)
        print(f"TensorBoard: {tb_dir}")

    for epoch in range(start_epoch, epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()

        # Train
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            cfg, epoch, global_step, device, world_size,
        )

        epoch_time = time.time() - t0

        if is_main_process():
            print(
                f"Epoch {epoch}/{epochs-1} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, epoch)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
                writer.add_scalar("train/epoch_time_s", epoch_time, epoch)

        # Validation
        if (epoch + 1) % eval_epochs == 0 or epoch == epochs - 1:
            mean_dice, dice_dict = validate(
                model, val_loader, cfg, device, world_size,
            )

            is_best = mean_dice > best_metric
            if is_best:
                best_metric = mean_dice

            if is_main_process():
                print(f"  Val Dice: {mean_dice:.4f} (best: {best_metric:.4f})")
                for name, val in dice_dict.items():
                    print(f"    {name}: {val:.4f}")
                if writer is not None:
                    writer.add_scalar("val/mean_dice", mean_dice, epoch)
                    writer.add_scalar("val/best_dice", best_metric, epoch)
                    for name, val in dice_dict.items():
                        writer.add_scalar(f"val_dice/{name}", val, epoch)
                    writer.flush()

            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                output_dir=output_dir,
                is_best=is_best,
                save_weights_only=getattr(cfg, "save_weights_only", False),
            )
        elif getattr(cfg, "save_checkpoint", True):
            # Save latest checkpoint every epoch
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                output_dir=output_dir,
                is_best=False,
                save_weights_only=getattr(cfg, "save_weights_only", False),
            )

    # --- Done ---
    if is_main_process():
        print(f"\nTraining complete. Best val Dice: {best_metric:.4f}")
        print(f"Checkpoints saved to: {output_dir}")
        if writer is not None:
            writer.close()

    cleanup_ddp()


if __name__ == "__main__":
    main()
