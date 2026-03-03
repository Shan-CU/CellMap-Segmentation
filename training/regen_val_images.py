#!/usr/bin/env python3
"""
Retroactive Validation Image Regeneration.

Iterates over all Phase 2 v4 (p2v4_*) model directories, loads config.json +
each saved checkpoint, reconstructs the model with EMA weights, runs a forward
pass on validation data, and saves properly non-empty visualization images to
val_images/epoch_XXXX/{input,prediction,ground_truth}.png.

This fixes the "black validation images" issue where the original training run
uniformly sampled from 132K+ validation blocks, ~98% of which were empty padded
regions. The new logic requires non-zero EM input AND ≥1% GT annotation fill.

Usage:
    # Single model:
    python -m training.regen_val_images --run_dir runs/ablation --model_dirs p2v4_unet_2d

    # All p2v4 models (default):
    python -m training.regen_val_images --run_dir runs/ablation

    # Process only specific checkpoints:
    python -m training.regen_val_images --run_dir runs/ablation --checkpoints best epoch_10

SLURM (a100-gpu):
    sbatch training/slurm/regen_val_images.sbatch
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add project root and src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cellmap_segmentation_challenge.utils import get_dataloader, get_tested_classes
from training.models.model_zoo import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)


# ── Color palette (same as train.py) ──────────────────────────────────────
# 48 perceptually distinct colors (3 HSV rings × 16 hues, interleaved)
CLASS_COLORS = [
    (0.9, 0.0, 0.0),      #  0 ecs
    (0.35, 1.0, 0.35),    #  1 pm
    (0.0, 0.0, 0.5),      #  2 mito_mem
    (0.9, 0.338, 0.0),    #  3 mito_lum
    (0.35, 1.0, 0.594),   #  4 mito_ribo
    (0.188, 0.0, 0.5),    #  5 golgi_mem
    (0.9, 0.675, 0.0),    #  6 golgi_lum
    (0.35, 1.0, 0.838),   #  7 ves_mem
    (0.375, 0.0, 0.5),    #  8 ves_lum
    (0.787, 0.9, 0.0),    #  9 endo_mem
    (0.35, 0.919, 1.0),   # 10 endo_lum
    (0.5, 0.0, 0.438),    # 11 lyso_mem
    (0.45, 0.9, 0.0),     # 12 lyso_lum
    (0.35, 0.675, 1.0),   # 13 ld_mem
    (0.5, 0.0, 0.25),     # 14 ld_lum
    (0.113, 0.9, 0.0),    # 15 er_mem
    (0.35, 0.431, 1.0),   # 16 er_lum
    (0.5, 0.0, 0.062),    # 17 eres_mem
    (0.0, 0.9, 0.225),    # 18 eres_lum
    (0.512, 0.35, 1.0),   # 19 ne_mem
    (0.5, 0.125, 0.0),    # 20 ne_lum
    (0.0, 0.9, 0.562),    # 21 np_out
    (0.756, 0.35, 1.0),   # 22 np_in
    (0.5, 0.312, 0.0),    # 23 hchrom
    (0.0, 0.9, 0.9),      # 24 echrom
    (1.0, 0.35, 1.0),     # 25 nucpl
    (0.5, 0.5, 0.0),      # 26 mt_out
    (0.0, 0.562, 0.9),    # 27 cyto
    (1.0, 0.35, 0.756),   # 28 mt_in
    (0.312, 0.5, 0.0),    # 29 nuc
    (0.0, 0.225, 0.9),    # 30 golgi
    (1.0, 0.35, 0.512),   # 31 ves
    (0.125, 0.5, 0.0),    # 32 endo
    (0.113, 0.0, 0.9),    # 33 lyso
    (1.0, 0.431, 0.35),   # 34 ld
    (0.0, 0.5, 0.062),    # 35 eres
    (0.45, 0.0, 0.9),     # 36 perox_mem
    (1.0, 0.675, 0.35),   # 37 perox_lum
    (0.0, 0.5, 0.25),     # 38 perox
    (0.787, 0.0, 0.9),    # 39 mito
    (1.0, 0.919, 0.35),   # 40 er
    (0.0, 0.5, 0.438),    # 41 ne
    (0.9, 0.0, 0.675),    # 42 np
    (0.838, 1.0, 0.35),   # 43 chrom
    (0.0, 0.375, 0.5),    # 44 mt
    (0.9, 0.0, 0.338),    # 45 cell
    (0.594, 1.0, 0.35),   # 46 er_mem_all
    (0.0, 0.188, 0.5),    # 47 ne_mem_all
]


def make_color_overlay(img_hw: torch.Tensor, masks: torch.Tensor,
                       alpha: float = 0.55) -> torch.Tensor:
    """Build (3,H,W) EM image with colored class masks overlaid.

    Args:
        img_hw: (H, W) normalized EM image [0, 1].
        masks: (C, H, W) binary mask per class.
        alpha: overlay opacity.

    Returns:
        (3, H, W) RGB tensor clamped to [0, 1].
    """
    canvas = torch.stack([img_hw, img_hw, img_hw], dim=0)  # (3,H,W)
    for ci in range(masks.shape[0]):
        m = masks[ci]
        if m.sum() == 0:
            continue
        r, g, b = CLASS_COLORS[ci % len(CLASS_COLORS)]
        color = torch.tensor([r, g, b], dtype=canvas.dtype).view(3, 1, 1)
        mask_3c = m.unsqueeze(0).expand(3, -1, -1)
        canvas = torch.where(mask_3c > 0.5,
                             canvas * (1 - alpha) + color * alpha,
                             canvas)
    return canvas.clamp(0, 1)


def find_valid_vis_sample(model, val_loader, device, amp_enabled,
                          deep_supervision=False,
                          max_batches=500, min_fill=0.01):
    """Find a non-empty validation sample and produce visualizations.

    Iterates through val_loader batches until finding a sample with
    non-zero EM input AND ≥min_fill annotated GT pixels.

    Returns:
        (vis_input, vis_pred, vis_gt) as CPU tensors, or None if not found.
    """
    input_keys = val_loader.dataset.input_arrays.keys()
    target_keys = val_loader.dataset.target_arrays.keys()
    input_keys = list(input_keys)
    target_keys = list(target_keys)

    model.eval()
    batch_count = 0
    with torch.no_grad():
        for batch in val_loader.loader:
            if batch_count >= max_batches:
                break
            batch_count += 1

            inputs = batch[input_keys[0]].to(device)
            targets = batch[target_keys[0]].to(device)
            targets_clean = targets.nan_to_num(0.0)

            with torch.amp.autocast('cuda', enabled=amp_enabled and device.startswith("cuda")):
                outputs = model(inputs)

            if deep_supervision and isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            elif isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs

            preds = (torch.sigmoid(logits.float()) > 0.5).float()
            gt = targets_clean.float()
            valid_mask = targets.isnan().logical_not().float()

            for bi in range(inputs.shape[0]):
                inp_bi = inputs[bi]   # (1, *spatial)
                gt_bi = gt[bi]        # (C, *spatial)

                # Require real EM signal
                if inp_bi.abs().max() < 1e-6:
                    continue
                frac = gt_bi.sum() / max(gt_bi.numel(), 1)
                if frac > min_fill:
                    return (
                        inp_bi.detach().cpu(),
                        preds[bi].detach().cpu(),
                        gt_bi.detach().cpu(),
                    )

    return None


def save_vis_images(vis_sample, epoch_dir: Path, classes: list[str]):
    """Save input, prediction, ground_truth PNGs and legend.txt."""
    from torchvision.utils import save_image

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

    pred_overlay = make_color_overlay(img_hw, vis_pred)
    gt_overlay = make_color_overlay(img_hw, vis_gt)

    epoch_dir.mkdir(parents=True, exist_ok=True)
    save_image(img_hw.unsqueeze(0), epoch_dir / "input.png")
    save_image(pred_overlay, epoch_dir / "prediction.png")
    save_image(gt_overlay, epoch_dir / "ground_truth.png")

    # Legend
    legend_path = epoch_dir / "legend.txt"
    with open(legend_path, "w") as lf:
        lf.write("Class Color Legend\n")
        lf.write("=" * 40 + "\n")
        for ci, cname in enumerate(classes):
            r, g, b = CLASS_COLORS[ci % len(CLASS_COLORS)]
            lf.write(f"{cname:20s}  RGB({r:.1f}, {g:.1f}, {b:.1f})\n")


def load_ema_or_model(checkpoint_path: str, model: nn.Module, device: str):
    """Load EMA weights if available, otherwise model_state_dict."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    epoch = ckpt.get("epoch", -1)

    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        model.load_state_dict(ckpt["ema_state_dict"])
        log.info(f"  Loaded EMA weights from epoch {epoch}")
    elif "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        log.info(f"  Loaded model weights from epoch {epoch} (no EMA)")
    else:
        raise ValueError(f"No model or EMA state dict in {checkpoint_path}")

    return epoch


def process_model_dir(model_dir: Path, device: str, classes: list[str],
                      checkpoint_names: list[str] | None = None,
                      max_val_batches: int = 500):
    """Process all checkpoints in a single model directory."""
    config_path = model_dir / "config.json"
    if not config_path.exists():
        log.warning(f"Skipping {model_dir.name}: no config.json")
        return

    cfg = json.load(open(config_path))
    model_name = cfg["model"]
    num_classes = len(classes)

    log.info(f"\n{'='*60}")
    log.info(f"Model: {model_dir.name} ({model_name})")
    log.info(f"{'='*60}")

    # --- Discover checkpoints ---
    ckpt_dir = model_dir / "checkpoints"
    if not ckpt_dir.exists():
        log.warning(f"  No checkpoints directory")
        return

    if checkpoint_names is not None:
        ckpt_files = []
        for name in checkpoint_names:
            fname = name if name.endswith('.pth') else f"{name}.pth"
            if (ckpt_dir / fname).exists():
                ckpt_files.append(fname)
            else:
                log.warning(f"  Checkpoint {fname} not found, skipping")
    else:
        ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith('.pth')])

    if not ckpt_files:
        log.warning(f"  No checkpoints to process")
        return

    log.info(f"  Checkpoints: {ckpt_files}")

    # --- Build model ---
    model_kwargs = {"num_classes": num_classes, "in_channels": 1}
    model_kwargs.update(cfg.get("model_kwargs", {}))
    if "img_size" not in model_kwargs and "3d" in model_name:
        model_kwargs["img_size"] = tuple(cfg["input_shape"])

    # Deep supervision
    if cfg.get("deep_supervision", False) and "segresnet" in model_name:
        model_kwargs.setdefault("dsdepth", 4)

    # Bias init
    bias_mode = cfg.get("bias_init_mode", "none")
    if bias_mode != "none":
        model_kwargs["bias_init_mode"] = bias_mode
        if cfg.get("bias_init") is not None:
            model_kwargs["bias_init"] = cfg["bias_init"]

    model = build_model(model_name, **model_kwargs)
    model = model.to(device)
    model.eval()

    # --- Build val dataloader (once per model, reuse across checkpoints) ---
    input_array_info = {
        "shape": tuple(cfg["input_shape"]),
        "scale": tuple(cfg["input_scale"]),
    }
    target_array_info = {
        "shape": tuple(cfg["target_shape"]),
        "scale": tuple(cfg["target_scale"]),
    }

    is_3d = all(s > 1 for s in cfg["input_shape"])
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

    log.info(f"  Loading validation dataloader...")
    _, val_loader = get_dataloader(
        datasplit_path=cfg.get("datasplit_path", "datasplit.csv"),
        classes=classes,
        batch_size=cfg.get("batch_size", 8),
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=spatial_transforms,
        iterations_per_epoch=cfg.get("iterations_per_epoch", 1000),
        random_validation=True,
        device="cpu",
        weighted_sampler=cfg.get("weighted_sampler", True),
        num_workers=4,  # enough for val-only
    )

    if val_loader is None:
        log.error(f"  No validation loader returned! Skipping model.")
        return

    log.info(f"  Val loader: {len(val_loader.loader)} batches")

    val_img_dir = model_dir / "val_images"
    val_img_dir.mkdir(exist_ok=True)
    amp_enabled = cfg.get("amp", True)
    deep_supervision = cfg.get("deep_supervision", False)

    # --- Process each checkpoint ---
    for ckpt_name in ckpt_files:
        ckpt_path = ckpt_dir / ckpt_name
        log.info(f"\n  --- {ckpt_name} ---")

        try:
            epoch = load_ema_or_model(str(ckpt_path), model, device)
        except Exception as e:
            log.error(f"  Failed to load {ckpt_name}: {e}")
            continue

        # Determine epoch label for directory name
        if ckpt_name.startswith("epoch_"):
            epoch_label = ckpt_name.replace(".pth", "")
        elif ckpt_name == "best.pth":
            epoch_label = f"best_epoch_{epoch}" if epoch >= 0 else "best"
        elif ckpt_name == "latest.pth":
            epoch_label = f"latest_epoch_{epoch}" if epoch >= 0 else "latest"
        else:
            epoch_label = ckpt_name.replace(".pth", "")

        ep_dir = val_img_dir / epoch_label

        # Skip if already generated
        if (ep_dir / "prediction.png").exists():
            log.info(f"  Already exists: {ep_dir}, skipping")
            continue

        t0 = time.time()
        vis_sample = find_valid_vis_sample(
            model, val_loader, device, amp_enabled,
            deep_supervision=deep_supervision,
            max_batches=max_val_batches,
        )

        if vis_sample is None:
            log.warning(f"  No valid vis sample found after {max_val_batches} batches!")
            # Save a marker file so we don't retry
            ep_dir.mkdir(parents=True, exist_ok=True)
            (ep_dir / "NO_VALID_SAMPLE.txt").write_text(
                f"No non-empty validation sample found after {max_val_batches} batches.\n"
            )
            continue

        save_vis_images(vis_sample, ep_dir, classes)
        elapsed = time.time() - t0
        log.info(f"  Saved to {ep_dir} ({elapsed:.1f}s)")

        # Clean up GPU cache between checkpoints
        torch.cuda.empty_cache()

    # Clean up model and dataloader between models
    del model, val_loader
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Retroactively regenerate validation images from saved checkpoints."
    )
    parser.add_argument("--run_dir", type=str, default="runs/ablation",
                        help="Base directory containing model run directories")
    parser.add_argument("--model_dirs", type=str, nargs="*", default=None,
                        help="Specific model directory names to process "
                             "(default: all p2v4_* dirs)")
    parser.add_argument("--checkpoints", type=str, nargs="*", default=None,
                        help="Specific checkpoint names to process "
                             "(e.g., 'best' 'epoch_10'). Default: all .pth files.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: cuda if available)")
    parser.add_argument("--max_val_batches", type=int, default=500,
                        help="Max validation batches to search for a non-empty sample")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    classes = get_tested_classes()
    log.info(f"Classes: {len(classes)}")

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        log.error(f"Run directory not found: {run_dir}")
        sys.exit(1)

    # Discover model directories
    if args.model_dirs:
        model_dirs = [run_dir / d for d in args.model_dirs]
    else:
        model_dirs = sorted([
            run_dir / d for d in os.listdir(run_dir)
            if d.startswith("p2v4_") and (run_dir / d).is_dir()
        ])

    log.info(f"Model directories to process: {[d.name for d in model_dirs]}")

    total_t0 = time.time()
    for model_dir in model_dirs:
        if not model_dir.exists():
            log.warning(f"Directory not found: {model_dir}")
            continue
        process_model_dir(
            model_dir, device, classes,
            checkpoint_names=args.checkpoints,
            max_val_batches=args.max_val_batches,
        )

    elapsed = time.time() - total_t0
    log.info(f"\nDone! Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
