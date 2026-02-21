#!/usr/bin/env python3
"""
Visualize GT label overlay on EM images for NIfTI quality assurance.

For each selected crop, produces a PNG showing the mid-slice in all 3 axes
(axial / coronal / sagittal) with:
  - Grayscale EM as background
  - Colored semi-transparent GT labels overlaid
  - Legend of class names + colors
  - Crop ID, shape, and annotation count in the title

Usage:
  python auto3dseg/visualize_gt_overlay.py [--crops N] [--output-dir DIR]

  --crops N       Number of crops to visualize (default: 20, 0 = all)
  --output-dir    Where to save PNGs (default: nifti_data_v2/qa_overlays/)
  --nifti-dir     NIfTI data directory (default: auto3dseg/nifti_data_v2/)
  --seed          Random seed for crop selection (default: 42)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

# ── Must match convert_zarr_to_nifti_v2.py ──
ATOMIC_CLASSES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
    "lyso_mem", "lyso_lum", "ld_mem", "ld_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl",
    "mt_out", "cyto", "mt_in", "perox_mem", "perox_lum",
    "nhchrom", "nechrom", "nucleo",
]
NUM_CLASSES = len(ATOMIC_CLASSES)

# Distinct colors for up to 35 classes (RGB 0-255).
# Using a curated palette that maximizes visual separation.
CLASS_COLORS = [
    (255, 0, 0),       #  1 ecs        — red
    (0, 255, 0),       #  2 pm         — green
    (0, 0, 255),       #  3 mito_mem   — blue
    (0, 255, 255),     #  4 mito_lum   — cyan
    (255, 0, 255),     #  5 mito_ribo  — magenta
    (255, 165, 0),     #  6 golgi_mem  — orange
    (255, 215, 0),     #  7 golgi_lum  — gold
    (128, 0, 128),     #  8 ves_mem    — purple
    (200, 150, 255),   #  9 ves_lum    — lavender
    (0, 128, 0),       # 10 endo_mem   — dark green
    (144, 238, 144),   # 11 endo_lum   — light green
    (139, 69, 19),     # 12 er_mem     — saddle brown
    (222, 184, 135),   # 13 er_lum     — burlywood
    (255, 255, 0),     # 14 nuc        — yellow
    (70, 130, 180),    # 15 lyso_mem   — steel blue
    (135, 206, 235),   # 16 lyso_lum   — sky blue
    (255, 99, 71),     # 17 ld_mem     — tomato
    (250, 128, 114),   # 18 ld_lum     — salmon
    (0, 100, 0),       # 19 eres_mem   — dark green
    (50, 205, 50),     # 20 eres_lum   — lime green
    (75, 0, 130),      # 21 ne_mem     — indigo
    (138, 43, 226),    # 22 ne_lum     — blue violet
    (0, 206, 209),     # 23 np_out     — dark turquoise
    (64, 224, 208),    # 24 np_in      — turquoise
    (220, 20, 60),     # 25 hchrom     — crimson
    (255, 105, 180),   # 26 echrom     — hot pink
    (255, 218, 185),   # 27 nucpl      — peach puff
    (128, 128, 0),     # 28 mt_out     — olive
    (192, 192, 192),   # 29 cyto       — silver
    (169, 169, 169),   # 30 mt_in      — dark gray
    (210, 105, 30),    # 31 perox_mem  — chocolate
    (244, 164, 96),    # 32 perox_lum  — sandy brown
    (178, 34, 34),     # 33 nhchrom    — firebrick
    (219, 112, 147),   # 34 nechrom    — pale violet red
    (255, 182, 193),   # 35 nucleo     — light pink
]


def make_overlay(em_slice: np.ndarray, label_slice: np.ndarray,
                 alpha: float = 0.45) -> np.ndarray:
    """Create an RGBA overlay of labels on EM.

    em_slice: 2D uint8 grayscale
    label_slice: 2D uint8 integer labels (0 = background)
    Returns: (H, W, 3) uint8 RGB image
    """
    # Normalize EM to [0, 255]
    em = em_slice.astype(np.float32)
    if em.max() > em.min():
        em = 255.0 * (em - em.min()) / (em.max() - em.min())
    else:
        em = np.zeros_like(em)

    # Build RGB from grayscale EM
    rgb = np.stack([em, em, em], axis=-1)  # (H, W, 3) float

    # Overlay colored labels
    for cls_id in range(1, NUM_CLASSES + 1):
        mask = label_slice == cls_id
        if not np.any(mask):
            continue
        color = CLASS_COLORS[cls_id - 1]
        for c in range(3):
            rgb[mask, c] = (1 - alpha) * rgb[mask, c] + alpha * color[c]

    return np.clip(rgb, 0, 255).astype(np.uint8)


def visualize_crop(image_path: str, label_path: str, crop_id: str,
                   annotated_classes: str, output_path: str):
    """Generate a 3-panel overlay PNG for one crop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Load NIfTI
    em_vol = np.asarray(nib.load(image_path).dataobj)    # (Z, Y, X)
    lbl_vol = np.asarray(nib.load(label_path).dataobj)   # (Z, Y, X)

    shape = em_vol.shape
    unique_labels = np.unique(lbl_vol)
    present_classes = sorted([int(x) for x in unique_labels if x > 0])

    # Parse annotated classes
    ann_str = annotated_classes if annotated_classes else ""
    n_annotated = len(ann_str.split(",")) if ann_str else 0

    # Mid-slices in 3 axes
    slices = [
        ("Axial (Z-mid)", em_vol[shape[0]//2, :, :], lbl_vol[shape[0]//2, :, :]),
        ("Coronal (Y-mid)", em_vol[:, shape[1]//2, :], lbl_vol[:, shape[1]//2, :]),
        ("Sagittal (X-mid)", em_vol[:, :, shape[2]//2], lbl_vol[:, :, shape[2]//2]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(
        f"{crop_id}  |  shape={shape}  |  {n_annotated} annotated  |  "
        f"{len(present_classes)} classes w/ foreground",
        fontsize=14, fontweight="bold",
    )

    for ax, (title, em_sl, lbl_sl) in zip(axes, slices):
        overlay = make_overlay(em_sl, lbl_sl)
        ax.imshow(overlay)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    # Legend of present classes
    legend_handles = []
    for cls_id in present_classes:
        color = tuple(c / 255.0 for c in CLASS_COLORS[cls_id - 1])
        name = ATOMIC_CLASSES[cls_id - 1]
        count = int(np.sum(lbl_vol == cls_id))
        legend_handles.append(
            Patch(facecolor=color, edgecolor="black",
                  label=f"{cls_id:2d} {name} ({count:,})")
        )

    if legend_handles:
        # Place legend outside right
        fig.legend(
            handles=legend_handles, loc="center right",
            fontsize=8, ncol=1, framealpha=0.9,
            bbox_to_anchor=(1.0, 0.5),
        )
        fig.subplots_adjust(right=0.82)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize GT overlay on EM for QA")
    parser.add_argument("--crops", type=int, default=20,
                        help="Number of crops to visualize (0 = all)")
    parser.add_argument("--nifti-dir", type=str, default=None,
                        help="NIfTI data directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for PNGs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for crop selection")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    nifti_dir = Path(args.nifti_dir) if args.nifti_dir else script_dir / "nifti_data_v2"
    output_dir = Path(args.output_dir) if args.output_dir else nifti_dir / "qa_overlays"
    datalist_path = nifti_dir / "datalist.json"

    if not datalist_path.exists():
        print(f"ERROR: {datalist_path} not found. Run conversion first.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    with open(datalist_path) as f:
        datalist = json.load(f)

    # Combine training + validation
    all_items = datalist.get("training", []) + datalist.get("validation", [])
    print(f"Found {len(all_items)} crops in datalist")

    # Select subset
    rng = np.random.RandomState(args.seed)
    if args.crops > 0 and args.crops < len(all_items):
        # Stratified selection: pick from diverse datasets
        datasets = {}
        for item in all_items:
            img_name = os.path.basename(item["image"])
            # Extract dataset name: jrc_xxx_cropYYY -> jrc_xxx
            parts = img_name.split("_crop")
            ds = parts[0] if parts else "unknown"
            datasets.setdefault(ds, []).append(item)

        # Pick at least 1 from each dataset, then fill remainder randomly
        selected = []
        ds_names = sorted(datasets.keys())
        rng.shuffle(ds_names)

        for ds in ds_names:
            if len(selected) >= args.crops:
                break
            items = datasets[ds]
            idx = rng.randint(0, len(items))
            selected.append(items[idx])

        # Fill remainder
        remaining = [it for it in all_items if it not in selected]
        rng.shuffle(remaining)
        while len(selected) < args.crops and remaining:
            selected.append(remaining.pop())

        all_items = selected

    print(f"Visualizing {len(all_items)} crops → {output_dir}\n")

    for i, item in enumerate(all_items, 1):
        img_path = item["image"]
        lbl_path = item["label"]
        ann_classes = item.get("annotated_classes", "")

        # Extract crop ID from filename
        crop_id = os.path.basename(lbl_path).replace(".nii.gz", "")

        out_png = output_dir / f"{crop_id}_overlay.png"

        print(f"[{i}/{len(all_items)}] {crop_id}")
        try:
            visualize_crop(img_path, lbl_path, crop_id, ann_classes, str(out_png))
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nDone! {len(all_items)} overlay PNGs saved to {output_dir}")


if __name__ == "__main__":
    main()
