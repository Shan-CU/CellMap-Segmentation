#!/usr/bin/env python3
"""
Visualize predictions from ALL masking-strategy checkpoints.

Each model gets its own subfolder under visualizations/.
The SAME set of diverse validation samples is reused across all models
so predictions are directly comparable.

Also prints per-class sigmoid activation stats and prediction counts
to help diagnose missing classes (e.g. golgi_mem).

Usage:
    python visualize_all_models.py --num_samples 20 --min_classes 5
"""

import os
import sys
import argparse
import glob
import random
import re
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(REPO_ROOT))
sys.path.insert(2, str(REPO_ROOT / "src"))

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
    DATASPLIT_CSV,
)
from cellmap_segmentation_challenge.models import UNet_2D
from cellmap_segmentation_challenge.utils.dataloader import get_dataloader

# ── Constants ─────────────────────────────────────────────────────────
CLASSES = QUICK_TEST_CLASSES
N_CLASSES = len(CLASSES)

CLASS_COLORS = {
    'nuc':       np.array([1.0, 0.2, 0.2]),
    'mito_mem':  np.array([0.2, 1.0, 0.2]),
    'er_mem':    np.array([0.3, 0.5, 1.0]),
    'pm':        np.array([1.0, 0.9, 0.1]),
    'golgi_mem': np.array([1.0, 0.3, 1.0]),
}

OVERLAY_ALPHA = 0.55


# ======================================================================
# Model helpers
# ======================================================================

def load_model(checkpoint_path, device):
    """Load UNet_2D from checkpoint, stripping compile/DDP prefixes."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = UNet_2D(n_channels=1, n_classes=N_CLASSES)

    state_dict = checkpoint['model_state_dict']
    cleaned = {k.replace('module.', '').replace('_orig_mod.', ''): v
               for k, v in state_dict.items()}

    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    best_dice = checkpoint.get('best_dice', 'N/A')
    strategy = checkpoint.get('strategy_name', '?')
    return model, epoch, best_dice, strategy


def ckpt_to_model_name(ckpt_path):
    """Extract a clean model name from checkpoint filename.
    e.g. 'mask_regional_g16_20260215_143022_best.pt'
      -> 'regional_g16'
    """
    fname = Path(ckpt_path).stem                    # drop .pt
    fname = re.sub(r'_best$', '', fname)            # drop _best
    fname = re.sub(r'_\d{8}_\d{6}$', '', fname)    # drop _YYYYMMDD_HHMMSS
    fname = re.sub(r'^mask_', '', fname)            # drop mask_ prefix
    return fname


# ======================================================================
# Dataloader
# ======================================================================

def create_diverse_val_loader(num_samples_per_dataset=10, iterations_per_epoch=1000):
    """Create validation dataloader sampling from ALL available datasets.
    
    This ensures maximum diversity in visualization samples by explicitly
    sampling from multiple different datasets instead of randomly drawing
    from a single datasplit.
    
    Args:
        num_samples_per_dataset: Samples to attempt from each dataset
        iterations_per_epoch: Iterations for dataloader
    """
    from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv
    import tempfile
    
    # Available datasets (from data directory)
    diverse_datasets = [
        'jrc_hela-2',         # HeLa cells
        'jrc_jurkat-1',       # Jurkat T-cells
        'jrc_cos7-1a',        # COS-7 cells
        'jrc_mus-liver',      # Mouse liver
        'jrc_mus-kidney',     # Mouse kidney
        'jrc_macrophage-2',   # Macrophages
    ]
    
    # Create a temporary datasplit with all diverse datasets
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=True) as f:
        datasplit_path = Path(f.name)
    
    # Now the file doesn't exist, safe to create
    make_datasplit_csv(
        classes=CLASSES,
        csv_path=str(datasplit_path),
        validation_prob=1.0,  # All samples go to validation
        force_all_classes=False,
        datasets=diverse_datasets,
    )
    
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

    _, val_loader = get_dataloader(
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


def create_val_loader(dataset_name=None, iterations_per_epoch=5000):
    """Create validation dataloader.
    
    Args:
        dataset_name: If provided, use only this dataset (e.g. 'jrc_hela-2').
                      If None, uses the training datasplit.csv.
        iterations_per_epoch: Number of iterations
    """
    if dataset_name:
        # Use a specific dataset instead of the training split
        from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv
        import tempfile
        
        # Create a temporary datasplit with just this dataset
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            datasplit_path = Path(f.name)
        
        # Force all samples from this dataset into validation
        make_datasplit_csv(
            classes=CLASSES,
            csv_path=str(datasplit_path),
            validation_prob=1.0,  # All samples go to validation
            force_all_classes=False,
            datasets=[dataset_name],  # Only this dataset
        )
    else:
        datasplit_path = DATASPLIT_CSV
        if not datasplit_path.exists():
            print(f"ERROR: {datasplit_path} not found.")
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

    _, val_loader = get_dataloader(
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
# Visualisation helpers
# ======================================================================

def count_present_classes(mask_np):
    return [name for c, name in enumerate(CLASSES) if mask_np[c].sum() > 0]


def build_overlay(raw_gray, mask_np, alpha=OVERLAY_ALPHA):
    H, W = raw_gray.shape
    base = np.stack([raw_gray] * 3, axis=-1)
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

    blended = base.copy()
    blended[any_class] = (1 - alpha) * base[any_class] + alpha * colour_mask[any_class]
    return np.clip(blended, 0, 1)


def make_legend_handles():
    return [mpatches.Patch(color=CLASS_COLORS[n], label=n) for n in CLASSES]


def visualize_sample(raw_2d, gt_mask, pred_mask, pred_logits,
                     sample_idx, model_name, output_dir):
    """
    Figure layout:
       [Raw + GT overlay]   [Raw + Pred overlay]
    Title includes per-class sigmoid stats.
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

    handles = make_legend_handles()
    fig.legend(handles=handles, loc='lower center', ncol=N_CLASSES,
               fontsize=10, frameon=True, fancybox=True,
               edgecolor='#444444', facecolor='#f8f8f8')

    # Build subtitle with per-class prediction pixel counts
    pred_counts = []
    for c, name in enumerate(CLASSES):
        n_px = int(pred_mask[c].sum())
        pred_counts.append(f"{name}:{n_px}")
    subtitle = '  |  '.join(pred_counts)

    plt.suptitle(f'{model_name}  —  Sample {sample_idx}\n'
                 f'Pred px: {subtitle}',
                 fontsize=11, y=0.99)
    plt.tight_layout(rect=[0, 0.06, 1, 0.90])

    out_path = os.path.join(output_dir, f'sample_{sample_idx:03d}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize ALL masking-strategy model predictions')
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--min_classes', type=int, default=2,
                        help='Minimum number of classes present in sample (default: 2)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset to use (e.g., jrc_hela-2, jrc_jurkat-1). '
                             'If not provided, uses training datasplit.')
    parser.add_argument('--checkpoint_dir', type=str,
                        default=str(CHECKPOINT_DIR))
    parser.add_argument('--output_root', type=str,
                        default=str(SCRIPT_DIR / 'visualizations'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # ── Discover all checkpoints ─────────────────────────────────────
    all_ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'mask_*_best.pt')))
    if not all_ckpts:
        print("ERROR: No checkpoints found matching 'mask_*_best.pt' in "
              f"{args.checkpoint_dir}")
        sys.exit(1)

    # De-duplicate: keep BEST dice for each strategy name
    ckpt_by_model = {}
    for ckpt in all_ckpts:
        name = ckpt_to_model_name(ckpt)
        # Load checkpoint to check dice score
        try:
            checkpoint = torch.load(ckpt, map_location='cpu', weights_only=False)
            dice = checkpoint.get('best_dice', 0)
        except Exception:
            continue
        
        # Keep checkpoint with highest dice
        if name not in ckpt_by_model or dice > ckpt_by_model[name][1]:
            ckpt_by_model[name] = (ckpt, dice)

    model_names = sorted(ckpt_by_model.keys())
    print(f"Found {len(model_names)} models (best checkpoint per strategy):")
    for name in model_names:
        ckpt_path, dice = ckpt_by_model[name]
        print(f"  - {name:<25s} (dice={dice:.4f})")
    print()

    # ── Phase 1: Cache diverse validation samples ────────────────────
    if args.dataset:
        print(f"Phase 1: Caching samples from specific dataset: {args.dataset}...")
        val_loader = create_val_loader(dataset_name=args.dataset, iterations_per_epoch=5000)
    else:
        print("Phase 1: Caching DIVERSE samples from multiple datasets...")
        print("  (HeLa, Jurkat, COS-7, mouse liver, mouse kidney, macrophage)")
        val_loader = create_diverse_val_loader(iterations_per_epoch=2000)

    MAX_SCAN = 3000
    MIN_SKIP = 20
    cached_samples = []     # list of (raw_2d, gt_np, input_tensor)
    checked = 0
    last_accepted = -MIN_SKIP

    with torch.no_grad():
        for batch in val_loader:
            if checked >= MAX_SCAN or len(cached_samples) >= args.num_samples * 3:
                break

            images = batch['input'].to(device)
            masks  = batch['output'].to(device)
            if images.dim() == 5 and images.shape[1] == 1:
                images = images.squeeze(1)

            checked += 1

            gt_np = masks[0].cpu().numpy()
            gt_np = np.nan_to_num(gt_np, nan=0.0)
            present = count_present_classes(gt_np)

            if len(present) < args.min_classes:
                continue
            if (checked - last_accepted) < MIN_SKIP:
                continue

            cached_samples.append((
                images[0].cpu().numpy()[0],   # raw_2d  (H, W)
                gt_np,                        # gt      (C, H, W)
                images[0].cpu(),              # input tensor for inference
            ))
            last_accepted = checked

    print(f"  Scanned {checked} batches, cached {len(cached_samples)} candidates")

    # Shuffle and select final set
    random.shuffle(cached_samples)
    selected = cached_samples[:args.num_samples]
    print(f"  Selected {len(selected)} diverse samples\n")

    if not selected:
        print("ERROR: No qualifying samples found!")
        sys.exit(1)

    # ── Phase 2: Run every model on the cached samples ───────────────
    print(f"Phase 2: Generating visualizations for {len(model_names)} models...\n")
    print("=" * 80)

    for model_name in model_names:
        ckpt_path, _ = ckpt_by_model[model_name]
        out_dir = os.path.join(args.output_root, model_name)
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}")
        print(f"  Checkpoint: {Path(ckpt_path).name}")

        model, epoch, best_dice, strategy = load_model(ckpt_path, device)
        print(f"  Epoch: {epoch}  Best Dice: {best_dice}  Strategy: {strategy}")

        # Per-model aggregate stats
        total_pred_px = {name: 0 for name in CLASSES}
        total_gt_px   = {name: 0 for name in CLASSES}
        sigmoid_stats  = {name: [] for name in CLASSES}

        with torch.no_grad():
            for i, (raw_2d, gt_np, input_tensor) in enumerate(selected, start=1):
                inp = input_tensor.unsqueeze(0).to(device)  # (1, 1, H, W)
                logits = model(inp)                          # (1, C, H, W)
                sig = torch.sigmoid(logits)
                pred = (sig > 0.5).float()

                pred_np   = pred[0].cpu().numpy()
                logits_np = logits[0].cpu().numpy()
                sig_np    = sig[0].cpu().numpy()

                # Accumulate stats
                for c, name in enumerate(CLASSES):
                    total_pred_px[name] += int(pred_np[c].sum())
                    total_gt_px[name]   += int(gt_np[c].sum())
                    sigmoid_stats[name].append({
                        'mean': float(sig_np[c].mean()),
                        'max':  float(sig_np[c].max()),
                        'min':  float(sig_np[c].min()),
                        'pct_above_0.5': float((sig_np[c] > 0.5).mean() * 100),
                    })

                visualize_sample(raw_2d, gt_np, pred_np, logits_np,
                                 i, model_name, out_dir)

        # Print per-class summary for this model
        print(f"\n  Per-class summary ({len(selected)} samples):")
        print(f"  {'Class':<12} {'GT px':>10} {'Pred px':>10} {'Pred/GT':>8} "
              f"{'Sig mean':>10} {'Sig max':>10} {'% > 0.5':>8}")
        print(f"  {'─' * 72}")
        for c, name in enumerate(CLASSES):
            gt_total   = total_gt_px[name]
            pred_total = total_pred_px[name]
            ratio = pred_total / max(gt_total, 1)
            s = sigmoid_stats[name]
            avg_mean = np.mean([x['mean'] for x in s])
            avg_max  = np.mean([x['max'] for x in s])
            avg_pct  = np.mean([x['pct_above_0.5'] for x in s])
            flag = " <<<" if pred_total == 0 and gt_total > 0 else ""
            print(f"  {name:<12} {gt_total:>10,} {pred_total:>10,} {ratio:>8.3f} "
                  f"{avg_mean:>10.4f} {avg_max:>10.4f} {avg_pct:>7.2f}%{flag}")

        print(f"  Saved {len(selected)} images to {out_dir}/")
        del model
        torch.cuda.empty_cache()

    print(f"\n{'=' * 80}")
    print(f"All done! {len(model_names)} models x {len(selected)} samples "
          f"= {len(model_names) * len(selected)} visualizations")
    print(f"Output root: {args.output_root}/")


if __name__ == '__main__':
    main()
