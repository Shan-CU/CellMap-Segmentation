#!/usr/bin/env python3
"""
Generate validation image visualizations for already-trained 2D R3 models.

Loads best checkpoints, runs inference on a few val samples, and writes
image summaries to the existing TensorBoard log directories so they appear
alongside the existing scalar metrics.

Usage (on Longleaf with GPU):
    python generate_2d_val_images.py            # all 4 models
    python generate_2d_val_images.py --model resnet_r3  # single model
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure experiment dir is on path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from data.ds_cellmap_2d import CellMap2DDataset, load_datalist, batch_to_device
from models.mdl_cellmap_2d import Net2D


# ── Colour palette ──────────────────────────────────────────
_tab20 = plt.cm.tab20(np.linspace(0, 1, 20))[:, :3]
_tab20b = plt.cm.tab20b(np.linspace(0, 1, 20))[:, :3]
CLASS_COLORS = np.concatenate([_tab20, _tab20b], axis=0)


def _overlay_masks(raw_hw, mask_chw, colors, alpha=0.45):
    rgb = np.stack([raw_hw] * 3, axis=-1)
    for c in range(mask_chw.shape[0]):
        fg = mask_chw[c] > 0.5
        if not fg.any():
            continue
        for ch in range(3):
            rgb[:, :, ch] = np.where(
                fg,
                rgb[:, :, ch] * (1 - alpha) + colors[c, ch] * alpha,
                rgb[:, :, ch],
            )
    return np.clip(rgb, 0, 1)


MODEL_CONFIGS = {
    "resnet_r3": "cfg_2d_resnet_r3",
    "unet_r3": "cfg_2d_unet_r3",
    "swin_r3": "cfg_2d_swin_r3",
    "vit_r3": "cfg_2d_vit_r3",
}

RUNS_DIR = "/work/users/g/s/gsgeorge/cellmap/runs/monai_2d"


def load_config(config_name: str):
    if "." not in config_name:
        config_name = f"configs_2d.{config_name}"
    mod = importlib.import_module(config_name)
    return mod.cfg


def generate_images(model_name: str, n_samples: int = 6, device_str: str = "cuda"):
    print(f"\n{'='*60}")
    print(f"  Generating val images for: {model_name}")
    print(f"{'='*60}")

    config_name = MODEL_CONFIGS[model_name]
    cfg = load_config(config_name)
    device = torch.device(device_str)

    class_names = getattr(cfg, "class_names", [f"ch{i}" for i in range(35)])
    colors = CLASS_COLORS[: len(class_names)]

    # Load checkpoint
    run_dir = os.path.join(RUNS_DIR, model_name)
    ckpt_path = os.path.join(run_dir, "checkpoint_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  SKIP — no checkpoint at {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_epoch = ckpt.get("epoch", 0)
    best_metric = ckpt.get("best_metric", 0)
    print(f"  Best checkpoint: epoch {best_epoch}, Dice {best_metric:.4f}")

    # Build model + load weights
    model = Net2D(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load val dataset (cached)
    _, val_files = load_datalist(cfg)
    print(f"  Caching {len(val_files)} val volumes...")
    val_dataset = CellMap2DDataset(val_files, cfg, mode="val")

    # Precision setup
    precision = getattr(cfg, "precision", "bf16" if getattr(cfg, "bf16", True) else "fp32")
    amp_enabled = precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    # Pick evenly spaced volumes
    n_vols = val_dataset.n_volumes
    indices = np.linspace(0, n_vols - 1, min(n_samples, n_vols), dtype=int)

    # Write to existing TB dir
    tb_dir = os.path.join(run_dir, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

    with torch.no_grad():
        for sample_idx, vol_idx in enumerate(indices):
            batch = val_dataset[int(vol_idx)]
            inp = batch["input"].to(device)
            tgt = batch["target"]
            ann = batch["annotation_mask"]

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
                outputs = model({"input": inp, "target": tgt.to(device),
                                 "annotation_mask": ann.to(device)})
                logits = outputs["logits"]

            pred = torch.sigmoid(logits).float().cpu()[0].numpy()
            pred_bin = (pred > 0.5).astype(np.float32)
            gt = tgt[0].numpy()
            raw = inp.float().cpu()[0, 0].numpy()

            ann_np = ann[0].numpy()
            gt_display = gt * ann_np[:, None, None]

            # 3-panel figure
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=100)

            axes[0].imshow(raw, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("Raw EM", fontsize=10)
            axes[0].axis("off")

            gt_overlay = _overlay_masks(raw, gt_display, colors)
            axes[1].imshow(gt_overlay)
            axes[1].set_title("Ground Truth", fontsize=10)
            axes[1].axis("off")

            pred_overlay = _overlay_masks(raw, pred_bin, colors)
            axes[2].imshow(pred_overlay)
            axes[2].set_title("Prediction", fontsize=10)
            axes[2].axis("off")

            present = [i for i in range(len(class_names))
                       if gt_display[i].sum() > 0 or pred_bin[i].sum() > 0]
            if present:
                patches = []
                for ci in present[:12]:
                    patches.append(
                        plt.Line2D([0], [0], marker="s", color="w",
                                   markerfacecolor=colors[ci],
                                   markersize=8, label=class_names[ci])
                    )
                fig.legend(handles=patches, loc="lower center",
                           ncol=min(6, len(patches)), fontsize=7, frameon=False)

            fig.suptitle(f"{model_name} — val sample {sample_idx} "
                         f"(epoch {best_epoch}, Dice {best_metric:.4f})",
                         fontsize=11)
            fig.tight_layout(rect=[0, 0.06, 1, 0.95])

            # Write at the best epoch step so it aligns with the scalar curves
            writer.add_figure(f"val_images/sample_{sample_idx}", fig, best_epoch)
            plt.close(fig)
            print(f"  ✓ sample {sample_idx}")

    writer.flush()
    writer.close()
    print(f"  Done — images written to {tb_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="",
                        help="Single model name (e.g. resnet_r3). "
                             "Omit to generate for all 4.")
    parser.add_argument("--n-samples", type=int, default=6)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    models = [args.model] if args.model else list(MODEL_CONFIGS.keys())
    for m in models:
        if m not in MODEL_CONFIGS:
            print(f"Unknown model: {m}. Choose from {list(MODEL_CONFIGS.keys())}")
            continue
        generate_images(m, n_samples=args.n_samples, device_str=args.device)

    print("\n✅ All done. Refresh TensorBoard to see images.")


if __name__ == "__main__":
    main()
