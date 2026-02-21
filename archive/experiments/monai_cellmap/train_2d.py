"""
2D training loop for MONAI CellMap segmentation pipeline.

Adapted from the 3D train.py but optimized for 2D:
- Single GPU (no DDP needed — 1 L40S per model)
- RAM-cached dataset (all volumes loaded at init, microsecond slicing)
- bf16 autocast (L40S native support)
- All validated optimizations from EXPERIMENT_FINDINGS.md:
  - Tversky α=0.6/β=0.4, Balanced Softmax τ=1.0
  - Foreground masking (+110%), bbox masking (+55%)
  - ResNet is best 2D arch (+63% over UNet)

Usage:
    python train_2d.py -C cfg_2d_resnet_r3
    python train_2d.py -C cfg_2d_swin_r3 --resume /path/to/checkpoint.pth

Reference: EXPERIMENT_FINDINGS.md
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure the experiment directory is on the path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data.ds_cellmap_2d import CellMap2DDataset, load_datalist, flat_collate_fn
from data.ds_cellmap_2d import batch_to_device
from models.mdl_cellmap_2d import Net2D
from utils import (
    set_seed,
    build_optimizer,
    build_scheduler,
    save_checkpoint,
    load_checkpoint,
    compute_per_channel_dice,
)


def parse_args():
    parser = argparse.ArgumentParser(description="MONAI CellMap 2D Training")
    parser.add_argument(
        "-C", "--config", type=str, required=True,
        help="Config module name (e.g., cfg_2d_resnet_r3)",
    )
    parser.add_argument(
        "--resume", type=str, default="",
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def load_config(config_name: str):
    """Dynamically import a config module and return the cfg object.

    Config name can be bare (e.g. 'cfg_2d_resnet_r3') or qualified
    (e.g. 'configs_2d.cfg_2d_resnet_r3'). Bare names are looked up
    inside the configs_2d package.
    """
    if "." not in config_name:
        config_name = f"configs_2d.{config_name}"
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
    # Precision: "bf16" (default), "fp16", or "fp32"
    precision = getattr(cfg, "precision", "bf16" if getattr(cfg, "bf16", True) else "fp32")
    amp_enabled = precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}", disable=getattr(cfg, "disable_tqdm", False))
    for step_in_epoch, batch in enumerate(pbar):
        batch = batch_to_device(batch, device)

        # Forward with autocast
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
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

        pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    avg_loss = total_loss / max(n_steps, 1)
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    cfg,
    device: torch.device,
) -> tuple[float, dict]:
    """Run validation and compute per-channel Dice.

    Returns:
        (mean_dice, per_channel_dice_dict)
    """
    model.eval()
    precision = getattr(cfg, "precision", "bf16" if getattr(cfg, "bf16", True) else "fp32")
    amp_enabled = precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    class_names = getattr(cfg, "class_names", [f"ch{i}" for i in range(35)])

    # Accumulators
    dice_sum = torch.zeros(cfg.num_classes, device=device)
    valid_sum = torch.zeros(cfg.num_classes, device=device)

    for batch in tqdm(loader, desc="Validation", disable=getattr(cfg, "disable_tqdm", False)):
        batch = batch_to_device(batch, device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            outputs = model(batch)
            logits = outputs["logits"]

        # Compute Dice
        mask = batch.get("annotation_mask", None)
        dice_pc, valid_pc = compute_per_channel_dice(
            logits, batch["target"], mask=mask, sigmoid=True,
        )
        dice_sum += dice_pc * valid_pc  # weighted sum
        valid_sum += valid_pc

    per_channel_dice = dice_sum / valid_sum.clamp(min=1.0)

    # Only average over channels that had any annotated samples
    annotated_channels = (valid_sum > 0).float()
    n_annotated = annotated_channels.sum().clamp(min=1.0)
    mean_dice = (per_channel_dice * annotated_channels).sum() / n_annotated

    # Build per-class dict for logging
    dice_dict = {}
    for i, name in enumerate(class_names):
        if i < len(per_channel_dice):
            dice_dict[name] = per_channel_dice[i].item()

    return mean_dice.item(), dice_dict


def _overlay_masks(raw_hw: np.ndarray, mask_chw: np.ndarray,
                   colors: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend coloured binary masks on top of a greyscale image.

    Args:
        raw_hw: (H, W) float32 in [0, 1].
        mask_chw: (C, H, W) float32 binary.
        colors: (C, 3) float RGB per class.
        alpha: opacity of the overlay.

    Returns:
        (H, W, 3) float32 RGB in [0, 1].
    """
    rgb = np.stack([raw_hw] * 3, axis=-1)  # (H, W, 3)
    for c in range(mask_chw.shape[0]):
        fg = mask_chw[c] > 0.5
        if not fg.any():
            continue
        for ch in range(3):
            rgb[:, :, ch] = np.where(fg,
                                     rgb[:, :, ch] * (1 - alpha) + colors[c, ch] * alpha,
                                     rgb[:, :, ch])
    return np.clip(rgb, 0, 1)


def log_val_images(
    writer: SummaryWriter,
    model: torch.nn.Module,
    val_dataset,
    cfg,
    device: torch.device,
    epoch: int,
    class_names: list[str],
    class_colors: np.ndarray,
    n_samples: int = 4,
):
    """Log a few validation samples (raw | GT overlay | pred overlay) to TB."""
    model.eval()
    precision = getattr(cfg, "precision", "bf16" if getattr(cfg, "bf16", True) else "fp32")
    amp_enabled = precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # Pick n_samples evenly spaced volumes from val set
    n_vols = val_dataset.n_volumes
    indices = np.linspace(0, n_vols - 1, min(n_samples, n_vols), dtype=int)

    with torch.no_grad():
        for sample_idx, vol_idx in enumerate(indices):
            # Get a single slice (val mode returns 1 sample per volume)
            batch = val_dataset[int(vol_idx)]
            # batch["input"]: (1, 1, H, W), batch["target"]: (1, C, H, W)
            inp = batch["input"].to(device)  # (1, 1, H, W)
            tgt = batch["target"]            # (1, C, H, W)
            ann = batch["annotation_mask"]   # (1, C)

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model({"input": inp, "target": tgt.to(device),
                                 "annotation_mask": ann.to(device)})
                logits = outputs["logits"]    # (1, C, H, W)

            pred = torch.sigmoid(logits).cpu()[0].numpy()   # (C, H, W)
            pred_bin = (pred > 0.5).astype(np.float32)
            gt = tgt[0].numpy()                             # (C, H, W)
            raw = inp.cpu()[0, 0].numpy()                   # (H, W)

            # Mask out unannotated channels in GT display
            ann_np = ann[0].numpy()  # (C,)
            gt_display = gt * ann_np[:, None, None]

            # Build figure: 3 columns — Raw | GT overlay | Pred overlay
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)

            axes[0].imshow(raw, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Raw EM", fontsize=10)
            axes[0].axis("off")

            gt_overlay = _overlay_masks(raw, gt_display, class_colors)
            axes[1].imshow(gt_overlay)
            axes[1].set_title("Ground Truth", fontsize=10)
            axes[1].axis("off")

            pred_overlay = _overlay_masks(raw, pred_bin, class_colors)
            axes[2].imshow(pred_overlay)
            axes[2].set_title("Prediction", fontsize=10)
            axes[2].axis("off")

            # Add a small legend for present classes
            present = [i for i in range(len(class_names))
                       if gt_display[i].sum() > 0 or pred_bin[i].sum() > 0]
            if present:
                legend_patches = []
                for ci in present[:12]:  # cap at 12 to keep legend tidy
                    legend_patches.append(
                        plt.Line2D([0], [0], marker="s", color="w",
                                   markerfacecolor=class_colors[ci],
                                   markersize=8, label=class_names[ci])
                    )
                fig.legend(handles=legend_patches, loc="lower center",
                           ncol=min(6, len(legend_patches)), fontsize=7,
                           frameon=False)

            fig.tight_layout(rect=[0, 0.06, 1, 1])
            writer.add_figure(f"val_images/sample_{sample_idx}", fig, epoch)
            plt.close(fig)


def main():
    args = parse_args()

    # --- Device setup (single GPU) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load config ---
    cfg = load_config(args.config)
    output_dir = getattr(cfg, "output_dir", "/work/users/g/s/gsgeorge/cellmap/runs/monai_2d")
    os.makedirs(output_dir, exist_ok=True)

    # --- Seed ---
    set_seed(getattr(cfg, "seed", 42))

    # --- Class names + color map for visualization ---
    class_names = getattr(cfg, "class_names", [f"ch{i}" for i in range(35)])
    # Deterministic per-class colours (tab20 + tab20b for 35 classes)
    _tab20 = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
    _tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))[:, :3]
    CLASS_COLORS = np.concatenate([_tab20, _tab20b], axis=0)[:len(class_names)]

    # --- Data ---
    print(f"Loading datalist from: {cfg.datalist}")
    train_files, val_files = load_datalist(cfg)
    print(f"Train: {len(train_files)} volumes, Val: {len(val_files)} volumes")

    # RAM-cache all volumes (takes 1-3 minutes, then slicing is microseconds)
    print("\n=== Caching volumes into RAM ===")
    t0 = time.time()
    train_dataset = CellMap2DDataset(train_files, cfg, mode="train")
    val_dataset = CellMap2DDataset(val_files, cfg, mode="val")
    cache_time = time.time() - t0
    print(f"Cache complete in {cache_time:.0f}s\n")

    # --- DataLoaders ---
    batch_size = getattr(cfg, "batch_size", 32)
    num_workers = getattr(cfg, "num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=flat_collate_fn,
        drop_last=True,
        pin_memory=False,  # pin_memory=True causes slowdown (MONAI #3116)
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=flat_collate_fn,
        drop_last=False,
        pin_memory=False,
        persistent_workers=(num_workers > 0),
    )

    # --- Model ---
    model = Net2D(cfg).to(device)

    # --- Optimizer & Scheduler ---
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader) // getattr(cfg, "grad_accumulation", 1)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)

    # --- GradScaler (needed for fp16, no-op for bf16/fp32) ---
    precision = getattr(cfg, "precision", "bf16" if getattr(cfg, "bf16", True) else "fp32")
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))

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
        print(f"Resuming from: {resume_path}")
        meta = load_checkpoint(model, optimizer, scheduler, resume_path, device)
        start_epoch = meta["epoch"] + 1
        global_step = meta["step"]
        best_metric = meta["best_metric"]

    # --- Training Loop ---
    epochs = getattr(cfg, "epochs", 500)
    eval_epochs = getattr(cfg, "eval_epochs", 5)

    print(f"\n{'='*60}")
    print(f"Training: {getattr(cfg, 'name', 'unknown')}")
    print(f"Backbone: {getattr(cfg, 'backbone_type', 'unknown')}")
    print(f"Epochs: {epochs}, Eval every: {eval_epochs}")
    print(f"Batch size: {batch_size} (eff: {batch_size * getattr(cfg, 'grad_accumulation', 1)})")
    print(f"Num samples/volume: {getattr(cfg, 'num_samples', 8)}")
    print(f"ROI size: {getattr(cfg, 'roi_size_2d', [256, 256])}")
    print(f"Multi-axis: {getattr(cfg, 'multi_axis', True)}")
    print(f"LR: {getattr(cfg, 'lr', 1e-4)}, Schedule: {getattr(cfg, 'schedule', 'cosine')}")
    print(f"Precision: {precision}")
    print(f"Loss: {getattr(cfg, 'loss_type', 'balanced_softmax_tversky')}")
    print(f"  Tversky α={getattr(cfg, 'tversky_alpha', 0.6)}, β={getattr(cfg, 'tversky_beta', 0.4)}")
    print(f"  Balanced Softmax τ={getattr(cfg, 'tau', 1.0)}")
    print(f"  BBox pad={getattr(cfg, 'bbox_pad_fraction', 0.05)}, bg_weight={getattr(cfg, 'bbox_bg_weight', 0.05)}")
    print(f"Iterations/epoch: {len(train_loader)}")
    print(f"Optimizer steps/epoch: {steps_per_epoch}")
    print(f"Num workers: {num_workers}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    # --- TensorBoard ---
    tb_dir = os.path.join(output_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)
    print(f"TensorBoard: {tb_dir}")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        # Train
        avg_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            cfg, epoch, global_step, device,
        )

        epoch_time = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs-1} | "
            f"Loss: {avg_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/epoch_time_s", epoch_time, epoch)

        # Validation
        if (epoch + 1) % eval_epochs == 0 or epoch == epochs - 1:
            mean_dice, dice_dict = validate(model, val_loader, cfg, device)

            is_best = mean_dice > best_metric
            if is_best:
                best_metric = mean_dice

            print(f"  Val Dice: {mean_dice:.4f} (best: {best_metric:.4f})")
            for name, val in dice_dict.items():
                if val > 0.01:  # Only print non-trivial channels
                    print(f"    {name}: {val:.4f}")

            writer.add_scalar("val/mean_dice", mean_dice, epoch)
            writer.add_scalar("val/best_dice", best_metric, epoch)
            for name, val in dice_dict.items():
                writer.add_scalar(f"val_dice/{name}", val, epoch)

            # Log validation images (every 25 epochs to avoid bloating TB)
            vis_every = getattr(cfg, "vis_every_epochs", 25)
            if (epoch + 1) % vis_every == 0 or epoch == epochs - 1 or epoch == start_epoch:
                log_val_images(
                    writer, model, val_dataset, cfg, device, epoch,
                    class_names, CLASS_COLORS, n_samples=4,
                )

            writer.flush()

            # Save checkpoint
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                output_dir=output_dir,
                is_best=is_best,
                save_weights_only=False,
                save_every_n_epochs=getattr(cfg, "save_every_n_epochs", 50),
            )
        elif getattr(cfg, "save_checkpoint", True):
            save_checkpoint(
                model, optimizer, scheduler,
                epoch=epoch,
                step=global_step,
                best_metric=best_metric,
                output_dir=output_dir,
                is_best=False,
                save_weights_only=False,
                save_every_n_epochs=getattr(cfg, "save_every_n_epochs", 50),
            )

    # --- Done ---
    print(f"\nTraining complete. Best val Dice: {best_metric:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    writer.close()


if __name__ == "__main__":
    main()
