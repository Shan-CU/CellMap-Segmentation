#!/usr/bin/env python3
"""
Experiment 1: Threshold Tuning
Sweep prediction thresholds on ALL existing checkpoints.

No retraining required — this evaluates the same saved models at
different sigmoid thresholds to find the optimal precision/recall
trade-off.

Outputs:
  - Console: per-model × per-threshold summary tables
  - CSV:     results/threshold_sweep.csv
  - Best:    ranked table of (model, threshold) combos by mean Dice

Usage:
    python evaluate_threshold_sweep.py
    python evaluate_threshold_sweep.py --thresholds 0.3 0.5 0.7 0.8 0.9
    python evaluate_threshold_sweep.py --num_batches 500  # quick test
"""

import os
import sys
import argparse
import csv
import glob
import re
import time
from pathlib import Path

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

CLASSES = QUICK_TEST_CLASSES
N_CLASSES = len(CLASSES)

DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ======================================================================
# Model helpers
# ======================================================================

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = UNet_2D(n_channels=1, n_classes=N_CLASSES)
    state_dict = checkpoint['model_state_dict']
    cleaned = {k.replace('module.', '').replace('_orig_mod.', ''): v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()
    epoch = checkpoint.get('epoch', '?')
    return model, epoch


def ckpt_to_model_name(ckpt_path):
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
# Multi-threshold accumulator
# ======================================================================

class MultiThresholdAccumulator:
    """Accumulates TP/FP/FN for multiple thresholds simultaneously.

    Stores raw sigmoid outputs and GT so we can compute metrics at
    any threshold without re-running inference.
    """

    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.n_classes = N_CLASSES
        # Per-threshold accumulators
        self.tp = {t: np.zeros(self.n_classes, dtype=np.float64) for t in thresholds}
        self.fp = {t: np.zeros(self.n_classes, dtype=np.float64) for t in thresholds}
        self.fn = {t: np.zeros(self.n_classes, dtype=np.float64) for t in thresholds}
        self.n_samples = 0

    def update(self, sigmoid_np, gt_binary):
        """
        sigmoid_np: (C, H, W) float in [0, 1]
        gt_binary:  (C, H, W) binary
        """
        for t in self.thresholds:
            pred = sigmoid_np > t
            for c in range(self.n_classes):
                p = pred[c]
                g = gt_binary[c]
                self.tp[t][c] += np.sum(p & g)
                self.fp[t][c] += np.sum(p & ~g)
                self.fn[t][c] += np.sum(~p & g)
        self.n_samples += 1

    def metrics_at(self, threshold):
        """Return dict with precision, recall, f1, dice, iou per class + mean."""
        tp, fp, fn = self.tp[threshold], self.fp[threshold], self.fn[threshold]

        prec_denom = tp + fp
        precision = np.where(prec_denom > 0, tp / prec_denom, 0.0)

        rec_denom = tp + fn
        recall = np.where(rec_denom > 0, tp / rec_denom, 0.0)

        f1_denom = precision + recall
        f1 = np.where(f1_denom > 0, 2 * precision * recall / f1_denom, 0.0)

        dice_denom = 2 * tp + fp + fn
        dice = np.where(dice_denom > 0, 2 * tp / dice_denom, 0.0)

        iou_denom = tp + fp + fn
        iou = np.where(iou_denom > 0, tp / iou_denom, 0.0)

        result = {}
        for c, name in enumerate(CLASSES):
            result[name] = {
                'precision': precision[c],
                'recall':    recall[c],
                'f1':        f1[c],
                'dice':      dice[c],
                'iou':       iou[c],
            }
        result['mean'] = {
            'precision': np.mean(precision),
            'recall':    np.mean(recall),
            'f1':        np.mean(f1),
            'dice':      np.mean(dice),
            'iou':       np.mean(iou),
        }
        return result


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Threshold sweep on existing checkpoints')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=DEFAULT_THRESHOLDS)
    parser.add_argument('--num_batches', type=int, default=None,
                        help='Limit to N validation batches (default: all)')
    parser.add_argument('--iterations_per_epoch', type=int, default=5000)
    parser.add_argument('--checkpoint_dir', type=str, default=str(CHECKPOINT_DIR))
    parser.add_argument('--output_csv', type=str,
                        default='results/threshold_sweep.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Thresholds: {args.thresholds}\n")

    # ── Discover checkpoints ─────────────────────────────────────
    all_ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, 'cw_*_best.pt')))
    if not all_ckpts:
        print("ERROR: No checkpoints found!")
        sys.exit(1)

    ckpt_by_model = {}
    for ckpt in all_ckpts:
        name = ckpt_to_model_name(ckpt)
        ckpt_by_model[name] = ckpt

    model_names = sorted(ckpt_by_model.keys())
    print(f"Found {len(model_names)} models:")
    for name in model_names:
        print(f"  - {name}")

    # ── Cache validation data ────────────────────────────────────
    print(f"\nCreating validation dataloader...")
    val_loader = create_val_loader(iterations_per_epoch=args.iterations_per_epoch)

    print("Caching validation batches...")
    cached_batches = []
    max_b = args.num_batches if args.num_batches else float('inf')

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_b:
                break
            images = batch['input']
            masks  = batch['output']
            if images.dim() == 5 and images.shape[1] == 1:
                images = images.squeeze(1)
            cached_batches.append((images, masks))

    print(f"Cached {len(cached_batches)} validation batches\n")

    # ── Evaluate ─────────────────────────────────────────────────
    csv_rows = []
    all_combos = []  # (model, threshold, mean_dice, metrics)

    for model_name in model_names:
        ckpt_path = ckpt_by_model[model_name]
        print(f"{'─' * 80}")
        print(f"Model: {model_name}")
        t0 = time.time()

        model, epoch = load_model(ckpt_path, device)
        acc = MultiThresholdAccumulator(args.thresholds)

        with torch.no_grad():
            for images, masks in cached_batches:
                images_dev = images.to(device)
                logits = model(images_dev)
                sigmoid = torch.sigmoid(logits)

                sig_np = sigmoid[0].cpu().numpy()
                gt_np  = masks[0].cpu().numpy()
                gt_np  = np.nan_to_num(gt_np, nan=0.0)
                gt_bin = (gt_np > 0.5).astype(bool)

                acc.update(sig_np, gt_bin)

        elapsed = time.time() - t0
        print(f"  Inference: {elapsed:.1f}s  ({acc.n_samples} samples)\n")

        # Print compact table: threshold → mean metrics
        print(f"  {'Thresh':>7} {'mPrec':>8} {'mRec':>8} {'mF1':>8} "
              f"{'mDice':>8} {'mIoU':>8}  | "
              + ''.join(f'{c:>10}' for c in CLASSES))
        print(f"  {'─' * (50 + 10 * len(CLASSES))}")

        for t in args.thresholds:
            m = acc.metrics_at(t)
            mm = m['mean']
            class_dice = ''.join(f"{m[c]['dice']:>10.4f}" for c in CLASSES)
            print(f"  {t:>7.2f} {mm['precision']:>8.4f} {mm['recall']:>8.4f} "
                  f"{mm['f1']:>8.4f} {mm['dice']:>8.4f} {mm['iou']:>8.4f}  | "
                  f"{class_dice}")

            all_combos.append((model_name, t, mm['dice'], m))

            # CSV rows
            for class_name in CLASSES + ['mean']:
                row = {
                    'model':     model_name,
                    'epoch':     epoch,
                    'threshold': t,
                    'class':     class_name,
                    'precision': f"{m[class_name]['precision']:.6f}",
                    'recall':    f"{m[class_name]['recall']:.6f}",
                    'f1':        f"{m[class_name]['f1']:.6f}",
                    'dice':      f"{m[class_name]['dice']:.6f}",
                    'iou':       f"{m[class_name]['iou']:.6f}",
                }
                csv_rows.append(row)

        # Mark best threshold for this model
        best_t = max(args.thresholds,
                     key=lambda t: acc.metrics_at(t)['mean']['dice'])
        best_d = acc.metrics_at(best_t)['mean']['dice']
        print(f"\n  ★ Best threshold: {best_t}  (mean Dice = {best_d:.4f})\n")

        del model
        torch.cuda.empty_cache()

    # ── Save CSV ─────────────────────────────────────────────────
    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['model', 'epoch', 'threshold', 'class',
                  'precision', 'recall', 'f1', 'dice', 'iou']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to: {csv_path}")

    # ── Final ranked summary ─────────────────────────────────────
    all_combos.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'=' * 110}")
    print("TOP 20 (model, threshold) COMBOS — ranked by mean Dice")
    print(f"{'=' * 110}")
    print(f"  {'Rank':<5} {'Model':<30} {'Thresh':>7} {'mPrec':>8} {'mRec':>8} "
          f"{'mF1':>8} {'mDice':>8} {'mIoU':>8}  | "
          + ''.join(f'{c:>10}' for c in CLASSES))
    print(f"  {'─' * (76 + 10 * len(CLASSES))}")

    for rank, (model_name, t, mean_dice, m) in enumerate(all_combos[:20], 1):
        mm = m['mean']
        class_dice = ''.join(f"{m[c]['dice']:>10.4f}" for c in CLASSES)
        print(f"  {rank:<5} {model_name:<30} {t:>7.2f} {mm['precision']:>8.4f} "
              f"{mm['recall']:>8.4f} {mm['f1']:>8.4f} {mm['dice']:>8.4f} "
              f"{mm['iou']:>8.4f}  | {class_dice}")

    # ── Per-model best threshold summary ─────────────────────────
    print(f"\n{'=' * 80}")
    print("BEST THRESHOLD PER MODEL")
    print(f"{'=' * 80}")
    print(f"  {'Model':<30} {'Best T':>7} {'mDice':>8} {'mPrec':>8} {'mRec':>8}")
    print(f"  {'─' * 55}")

    seen_models = set()
    for model_name, t, mean_dice, m in all_combos:
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        mm = m['mean']
        print(f"  {model_name:<30} {t:>7.2f} {mm['dice']:>8.4f} "
              f"{mm['precision']:>8.4f} {mm['recall']:>8.4f}")

    print()


if __name__ == '__main__':
    main()
