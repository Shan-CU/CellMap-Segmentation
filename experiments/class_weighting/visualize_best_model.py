#!/usr/bin/env python3
"""
Visualize predictions from the best class weighting model
(Balanced Softmax Tversky tau=1.0) overlaid on raw EM images.

For each sample generates a figure with:
  Left:  Raw image with Ground Truth overlay
  Right: Raw image with Prediction overlay
  + a shared colour legend

Diversity is ensured by:
  - Skipping a random number of batches between accepted samples
  - Shuffling the validation loader (random_validation=True)
  - Requiring all 5 classes to be present in the GT

Usage:
    python visualize_best_model.py --num_samples 20 --min_classes 5
"""

import os
import sys
import argparse
import glob
import random
from pathlib import Path

# ── Path setup (before any local imports) ─────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))        # repo root
sys.path.insert(0, str(Path(__file__).parent))                       # this dir

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torchvision.transforms.v2 as T
from cellmap_data.transforms.augment import NaNtoNum

from config import (
    QUICK_TEST_CLASSES, CHECKPOINT_DIR,
    SPATIAL_TRANSFORMS_2D, DATALOADER_CONFIG,
)
from cellmap_segmentation_challenge.models import UNet_2D
from cellmap_segmentation_challenge.utils.dataloader import get_dataloader

# ── Constants ─────────────────────────────────────────────────────────
CLASSES = QUICK_TEST_CLASSES
N_CLASSES = len(CLASSES)

# RGBA overlay colours (with alpha for blending)
CLASS_COLORS = {
    'nuc':       np.array([1.0, 0.2, 0.2]),   # Red
    'mito_mem':  np.array([0.2, 1.0, 0.2]),   # Green
    'er_mem':    np.array([0.3, 0.5, 1.0]),   # Blue
    'pm':        np.array([1.0, 0.9, 0.1]),   # Yellow
    'golgi_mem': np.array([1.0, 0.3, 1.0]),   # Magenta
}

OVERLAY_ALPHA = 0.55  # blending strength for masks on raw image


# ======================================================================
# Model loading
# ======================================================================

def load_model(checkpoint_path, device):
    """Load a UNet_2D model from a training checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = UNet_2D(n_channels=1, n_classes=N_CLASSES)

    # Strip DDP ('module.') and torch.compile ('_orig_mod.') prefixes
    state_dict = checkpoint['model_state_dict']
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        cleaned[k] = v

    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    val_loss = checkpoint.get('val_loss', 'N/A')
    print(f"  epoch={epoch}  val_loss={val_loss}")
    return model


# ======================================================================
# Dataloader (mirrors train.py)
# ======================================================================

def create_val_loader(iterations_per_epoch=2000):
    """Create a randomised validation dataloader."""
    datasplit_path = Path(__file__).parent / "datasplit.csv"
    if not datasplit_path.exists():
        print(f"ERROR: {datasplit_path} not found. Run training first.")
        sys.exit(1)

    input_array_info  = {"shape": (1, 256, 256), "scale": (8, 8, 8)}
    target_array_info = {"shape": (1, 256, 256), "scale": (8, 8, 8)}

    def _normalize_to_float32(x):
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    raw_value_transforms = T.Compose([
        T.Lambda(_normalize_to_float32),
        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    ])

    dl_kwargs = DATALOADER_CONFIG.copy()
    dl_kwargs['num_workers'] = 0
    dl_kwargs['persistent_workers'] = False

    _train_loader, val_loader = get_dataloader(
        datasplit_path=str(datasplit_path),
        classes=CLASSES,
        batch_size=1,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=SPATIAL_TRANSFORMS_2D,
        iterations_per_epoch=iterations_per_epoch,
        train_raw_value_transforms=raw_value_transforms,
        val_raw_value_transforms=raw_value_transforms,
        random_validation=True,
        **dl_kwargs,
    )
    return val_loader


# ======================================================================
# Overlay helpers
# ======================================================================

def count_present_classes(mask_np):
    """Return list of class names with non-zero pixels."""
    return [name for c, name in enumerate(CLASSES) if mask_np[c].sum() > 0]


def build_overlay(raw_gray, mask_np, alpha=OVERLAY_ALPHA):
    """
    Blend multi-channel binary mask onto a grayscale image.

    Parameters
    ----------
    raw_gray : ndarray (H, W), float [0, 1]
    mask_np  : ndarray (C, H, W), binary
    alpha    : blending strength for the coloured mask

    Returns
    -------
    blended : ndarray (H, W, 3), float [0, 1]
    """
    H, W = raw_gray.shape
    # Start from the grayscale image as RGB
    base = np.stack([raw_gray] * 3, axis=-1)   # (H, W, 3)

    # Build combined colour mask
    colour_mask = np.zeros((H, W, 3), dtype=np.float64)
    any_class   = np.zeros((H, W), dtype=bool)

    for c, name in enumerate(CLASSES):
        m = mask_np[c] > 0.5
        if not m.any():
            continue
        any_class |= m
        for ch in range(3):
            colour_mask[m, ch] = np.maximum(colour_mask[m, ch],
                                            CLASS_COLORS[name][ch])

    # Alpha-blend only where masks are present
    blended = base.copy()
    blended[any_class] = ((1 - alpha) * base[any_class]
                          + alpha * colour_mask[any_class])
    return np.clip(blended, 0, 1)


def make_legend_handles():
    """Create matplotlib legend patches for each class."""
    handles = []
    for name in CLASSES:
        c = CLASS_COLORS[name]
        handles.append(mpatches.Patch(color=c, label=name))
    return handles


# ======================================================================
# Visualisation
# ======================================================================

def visualize_sample(raw_2d, gt_mask, pred_mask, sample_idx, output_dir):
    """
    Produce a single figure:
       [Raw + GT overlay]   [Raw + Pred overlay]
    with a shared legend below.
    """
    gt_blend   = build_overlay(raw_2d, gt_mask)
    pred_blend = build_overlay(raw_2d, pred_mask)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    axes[0].imshow(gt_blend)
    axes[0].set_title('Ground Truth', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(pred_blend)
    axes[1].set_title('Prediction', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    # Shared legend at the bottom
    handles = make_legend_handles()
    fig.legend(handles=handles, loc='lower center', ncol=N_CLASSES,
               fontsize=10, frameon=True, fancybox=True,
               edgecolor='#444444', facecolor='#f8f8f8')

    plt.suptitle(f'Sample {sample_idx}  —  Balanced Softmax Tversky (tau=1.0)',
                 fontsize=14, y=0.97)
    plt.tight_layout(rect=[0, 0.06, 1, 0.94])

    out_path = os.path.join(output_dir, f'sample_{sample_idx:03d}.png')
    plt.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> saved {out_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize best class-weighting model predictions')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--min_classes', type=int, default=5)
    parser.add_argument('--checkpoint_dir', type=str,
                        default=str(CHECKPOINT_DIR))
    parser.add_argument('--output_dir', type=str,
                        default='visualizations/best_model')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Find checkpoint ──────────────────────────────────────────────
    pattern = os.path.join(args.checkpoint_dir,
                           'cw_balanced_softmax_tau_1.0_*_best.pt')
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        print(f"ERROR: no checkpoint matching {pattern}")
        sys.exit(1)
    ckpt_path = ckpts[-1]
    print(f"Checkpoint: {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(ckpt_path, device)

    # ── Dataloader ───────────────────────────────────────────────────
    # Use a large iteration count so the random sampler draws from many
    # different crops / datasets, increasing diversity.
    print("Creating validation dataloader...")
    val_loader = create_val_loader(iterations_per_epoch=5000)
    print(f"Scanning for diverse samples with >= {args.min_classes} classes...\n")

    # ── Collect qualifying samples with diversity spacing ────────────
    # Instead of taking the first N qualifying samples (which tend to
    # cluster in the same region), we collect ALL qualifying samples
    # into a pool, then pick the most spread-out subset.
    MAX_SCAN = 3000          # max batches to scan
    MIN_SKIP = 20            # minimum batches to skip between accepted
    candidates = []          # list of (raw, gt, pred) numpy tuples
    checked = 0
    last_accepted_idx = -MIN_SKIP  # allow accepting from the start

    print("Phase 1: Scanning for qualifying samples...")
    with torch.no_grad():
        for batch in val_loader:
            if checked >= MAX_SCAN:
                break
            if len(candidates) >= args.num_samples * 3:
                break  # enough candidates gathered

            images = batch['input'].to(device)
            masks  = batch['output'].to(device)
            if images.dim() == 5 and images.shape[1] == 1:
                images = images.squeeze(1)

            checked += 1

            # Quick class check on GT (CPU)
            gt_np = masks[0].cpu().numpy()
            gt_np = np.nan_to_num(gt_np, nan=0.0)
            present = count_present_classes(gt_np)

            if len(present) < args.min_classes:
                continue

            # Enforce spacing for diversity
            if (checked - last_accepted_idx) < MIN_SKIP:
                continue

            # Forward pass
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()

            img_np  = images[0].cpu().numpy()    # (1, H, W)
            pred_np = preds[0].cpu().numpy()     # (C, H, W)

            candidates.append((img_np[0], gt_np, pred_np))
            last_accepted_idx = checked

            if checked % 200 == 0:
                print(f"  scanned {checked}, found {len(candidates)} candidates")

    print(f"Phase 1 done: scanned {checked} batches, "
          f"found {len(candidates)} candidates\n")

    if len(candidates) == 0:
        print("ERROR: No qualifying samples found!")
        sys.exit(1)

    # ── Phase 2: Pick diverse subset ─────────────────────────────────
    # Shuffle candidates and take the first num_samples
    random.shuffle(candidates)
    selected = candidates[:args.num_samples]

    print(f"Phase 2: Generating {len(selected)} visualizations...\n")
    for i, (raw_2d, gt_np, pred_np) in enumerate(selected, start=1):
        visualize_sample(raw_2d, gt_np, pred_np, i, args.output_dir)

    print(f"\nDone! Saved {len(selected)} visualizations to {args.output_dir}")


if __name__ == '__main__':
    main()
