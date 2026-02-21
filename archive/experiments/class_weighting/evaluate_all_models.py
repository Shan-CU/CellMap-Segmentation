#!/usr/bin/env python3
"""
Evaluate ALL class-weighting checkpoints on the full validation set.

Computes per-class and mean:
  - Precision, Recall, F1
  - Dice coefficient
  - IoU (Jaccard index)

Outputs:
  1. Console summary tables (per-model)
  2. CSV with all results: results/evaluation_metrics.csv
  3. A ranked summary table at the end

Usage:
    python evaluate_all_models.py
    python evaluate_all_models.py --num_batches 500   # quick test
"""

import os
import sys
import argparse
import csv
import glob
import re
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
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
THRESHOLD = 0.5


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
    val_loss = checkpoint.get('val_loss', 'N/A')
    return model, epoch, val_loss


def ckpt_to_model_name(ckpt_path):
    """Extract a clean model name from checkpoint filename."""
    fname = Path(ckpt_path).stem
    fname = re.sub(r'_best$', '', fname)
    fname = re.sub(r'_\d{8}_\d{6}$', '', fname)
    fname = re.sub(r'^cw_', '', fname)
    return fname


# ======================================================================
# Dataloader
# ======================================================================

def create_val_loader(iterations_per_epoch=5000):
    datasplit_path = Path(__file__).parent / "datasplit.csv"
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
# Metric accumulators
# ======================================================================

class MetricsAccumulator:
    """Accumulates TP, FP, FN per class for efficient metric computation."""

    def __init__(self, class_names):
        self.class_names = class_names
        self.n_classes = len(class_names)
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.n_classes, dtype=np.float64)
        self.fp = np.zeros(self.n_classes, dtype=np.float64)
        self.fn = np.zeros(self.n_classes, dtype=np.float64)
        self.n_samples = 0

    def update(self, pred_binary, gt_binary):
        """
        Update accumulators.
        pred_binary: (C, H, W) binary numpy array
        gt_binary:   (C, H, W) binary numpy array
        """
        for c in range(self.n_classes):
            p = pred_binary[c].astype(bool)
            g = gt_binary[c].astype(bool)
            self.tp[c] += np.sum(p & g)
            self.fp[c] += np.sum(p & ~g)
            self.fn[c] += np.sum(~p & g)
        self.n_samples += 1

    def precision(self):
        """Per-class precision: TP / (TP + FP)"""
        denom = self.tp + self.fp
        return np.where(denom > 0, self.tp / denom, 0.0)

    def recall(self):
        """Per-class recall: TP / (TP + FN)"""
        denom = self.tp + self.fn
        return np.where(denom > 0, self.tp / denom, 0.0)

    def f1(self):
        """Per-class F1: 2 * P * R / (P + R)"""
        p = self.precision()
        r = self.recall()
        denom = p + r
        return np.where(denom > 0, 2 * p * r / denom, 0.0)

    def dice(self):
        """Per-class Dice: 2*TP / (2*TP + FP + FN)  (equivalent to F1 at pixel level)"""
        denom = 2 * self.tp + self.fp + self.fn
        return np.where(denom > 0, 2 * self.tp / denom, 0.0)

    def iou(self):
        """Per-class IoU: TP / (TP + FP + FN)"""
        denom = self.tp + self.fp + self.fn
        return np.where(denom > 0, self.tp / denom, 0.0)

    def summary_dict(self):
        """Return a dict with all metrics per class + means."""
        prec = self.precision()
        rec  = self.recall()
        f1   = self.f1()
        dice = self.dice()
        iou  = self.iou()

        results = {}
        for c, name in enumerate(self.class_names):
            results[name] = {
                'precision': prec[c],
                'recall':    rec[c],
                'f1':        f1[c],
                'dice':      dice[c],
                'iou':       iou[c],
                'tp':        self.tp[c],
                'fp':        self.fp[c],
                'fn':        self.fn[c],
            }

        # Macro averages
        results['mean'] = {
            'precision': np.mean(prec),
            'recall':    np.mean(rec),
            'f1':        np.mean(f1),
            'dice':      np.mean(dice),
            'iou':       np.mean(iou),
        }

        return results


# ======================================================================
# Printing
# ======================================================================

def print_model_table(model_name, epoch, val_loss, metrics, n_samples):
    """Print a formatted per-class metrics table for one model."""
    print(f"\n{'─' * 90}")
    print(f"Model: {model_name}  |  Epoch: {epoch}  |  Val loss: {val_loss}  |  "
          f"Samples: {n_samples}")
    print(f"{'─' * 90}")
    hdr = (f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} "
           f"{'Dice':>10} {'IoU':>10}")
    print(hdr)
    print(f"  {'─' * 64}")

    for name in CLASSES:
        m = metrics[name]
        print(f"  {name:<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
              f"{m['f1']:>10.4f} {m['dice']:>10.4f} {m['iou']:>10.4f}")

    m = metrics['mean']
    print(f"  {'─' * 64}")
    print(f"  {'MEAN':<12} {m['precision']:>10.4f} {m['recall']:>10.4f} "
          f"{m['f1']:>10.4f} {m['dice']:>10.4f} {m['iou']:>10.4f}")


def print_ranked_summary(all_results):
    """Print a final ranked summary across all models."""
    print(f"\n{'=' * 100}")
    print("RANKED SUMMARY  (sorted by mean Dice, descending)")
    print(f"{'=' * 100}")

    # Collect rows
    rows = []
    for model_name, info in all_results.items():
        m = info['metrics']['mean']
        per_class_dice = {c: info['metrics'][c]['dice'] for c in CLASSES}
        rows.append({
            'model': model_name,
            'mean_prec': m['precision'],
            'mean_rec':  m['recall'],
            'mean_f1':   m['f1'],
            'mean_dice': m['dice'],
            'mean_iou':  m['iou'],
            **{f'dice_{c}': per_class_dice[c] for c in CLASSES},
        })

    rows.sort(key=lambda r: r['mean_dice'], reverse=True)

    # Header
    class_cols = ''.join(f' {c:>10}' for c in CLASSES)
    print(f"\n  {'Rank':<5} {'Model':<28} {'mPrec':>7} {'mRec':>7} {'mF1':>7} "
          f"{'mDice':>7} {'mIoU':>7} |{class_cols}")
    print(f"  {'─' * (74 + 11 * len(CLASSES))}")

    for rank, row in enumerate(rows, 1):
        dcols = ''.join(f" {row[f'dice_{c}']:>10.4f}" for c in CLASSES)
        print(f"  {rank:<5} {row['model']:<28} {row['mean_prec']:>7.4f} "
              f"{row['mean_rec']:>7.4f} {row['mean_f1']:>7.4f} "
              f"{row['mean_dice']:>7.4f} {row['mean_iou']:>7.4f} |{dcols}")

    print()


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate ALL class-weighting models on validation set')
    parser.add_argument('--num_batches', type=int, default=None,
                        help='Limit evaluation to N batches (default: full val set)')
    parser.add_argument('--iterations_per_epoch', type=int, default=5000,
                        help='Validation loader iterations_per_epoch')
    parser.add_argument('--checkpoint_dir', type=str,
                        default=str(CHECKPOINT_DIR))
    parser.add_argument('--output_csv', type=str,
                        default='results/evaluation_metrics.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Discover checkpoints ─────────────────────────────────────────
    all_ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'cw_*_best.pt')))
    if not all_ckpts:
        print("ERROR: No checkpoints found!")
        sys.exit(1)

    # De-duplicate: keep latest timestamp for same config
    ckpt_by_model = {}
    for ckpt in all_ckpts:
        name = ckpt_to_model_name(ckpt)
        ckpt_by_model[name] = ckpt

    model_names = sorted(ckpt_by_model.keys())
    print(f"\nFound {len(model_names)} models:")
    for name in model_names:
        print(f"  - {name}")

    # ── Create validation loader ─────────────────────────────────────
    print(f"\nCreating validation dataloader (iterations_per_epoch={args.iterations_per_epoch})...")
    val_loader = create_val_loader(iterations_per_epoch=args.iterations_per_epoch)

    # ── Cache ALL validation batches once ─────────────────────────────
    # This avoids re-creating the dataloader for each model and ensures
    # every model sees the exact same data.
    print("Caching validation batches...")
    cached_batches = []
    max_batches = args.num_batches if args.num_batches else float('inf')

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            images = batch['input']
            masks  = batch['output']
            # Squeeze depth dim if present (1, 1, H, W) -> (1, H, W) for 2D
            if images.dim() == 5 and images.shape[1] == 1:
                images = images.squeeze(1)
            cached_batches.append((images, masks))

    n_batches = len(cached_batches)
    print(f"Cached {n_batches} validation batches\n")

    # ── Evaluate each model ──────────────────────────────────────────
    all_results = {}
    csv_rows = []

    for model_name in model_names:
        ckpt_path = ckpt_by_model[model_name]
        print(f"Evaluating: {model_name} ...")
        t0 = time.time()

        model, epoch, val_loss = load_model(ckpt_path, device)
        acc = MetricsAccumulator(CLASSES)

        with torch.no_grad():
            for images, masks in cached_batches:
                images_dev = images.to(device)
                logits = model(images_dev)
                probs = torch.sigmoid(logits)
                preds = (probs > THRESHOLD).float()

                pred_np = preds[0].cpu().numpy()
                gt_np   = masks[0].cpu().numpy()
                gt_np   = np.nan_to_num(gt_np, nan=0.0)
                gt_bin  = (gt_np > 0.5).astype(np.float32)

                acc.update(pred_np, gt_bin)

        elapsed = time.time() - t0
        metrics = acc.summary_dict()

        all_results[model_name] = {
            'epoch':    epoch,
            'val_loss': val_loss,
            'metrics':  metrics,
        }

        print_model_table(model_name, epoch, val_loss, metrics, acc.n_samples)
        print(f"  ({elapsed:.1f}s)")

        # Collect CSV rows
        for class_name in CLASSES + ['mean']:
            m = metrics[class_name]
            row = {
                'model':     model_name,
                'epoch':     epoch,
                'class':     class_name,
                'precision': f"{m['precision']:.6f}",
                'recall':    f"{m['recall']:.6f}",
                'f1':        f"{m['f1']:.6f}",
                'dice':      f"{m['dice']:.6f}",
                'iou':       f"{m['iou']:.6f}",
            }
            if class_name != 'mean':
                row['tp'] = int(m['tp'])
                row['fp'] = int(m['fp'])
                row['fn'] = int(m['fn'])
            csv_rows.append(row)

        del model
        torch.cuda.empty_cache()

    # ── Save CSV ─────────────────────────────────────────────────────
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ['model', 'epoch', 'class', 'precision', 'recall',
                  'f1', 'dice', 'iou', 'tp', 'fp', 'fn']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to: {csv_path}")

    # ── Ranked summary ───────────────────────────────────────────────
    print_ranked_summary(all_results)


if __name__ == '__main__':
    main()
