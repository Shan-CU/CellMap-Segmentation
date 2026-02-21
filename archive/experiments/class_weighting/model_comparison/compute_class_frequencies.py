#!/usr/bin/env python3
"""
Compute per-class voxel frequencies for all 14 organelle classes.

Scans the training dataset and outputs class_frequencies.json with:
  - Raw positive/total counts per class
  - Frequency ratios
  - Weight strategies: inv_freq, sqrt_inv, log_inv, effective_num
  - estimated_voxel_counts (used by Balanced Softmax loss)

Usage:
    python compute_class_frequencies.py                 # 50 batches (default)
    python compute_class_frequencies.py --n_batches 100 # more samples
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.v2 as T
from cellmap_data.transforms.augment import NaNtoNum

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from config import CLASSES, INPUT_SHAPE, SCALE, SPATIAL_TRANSFORMS_2D, DATALOADER_CONFIG


def compute_frequencies(classes, n_batches=50, batch_size=24,
                        input_shape=INPUT_SHAPE):
    """Scan training data and count per-class positive voxels."""
    from cellmap_segmentation_challenge.utils.dataloader import get_dataloader

    datasplit_path = SCRIPT_DIR / "datasplit.csv"
    if not datasplit_path.exists():
        from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv
        print("[Phase 1/3] Creating datasplit.csv (indexes all zarr datasets) ...")
        t0 = time.time()
        make_datasplit_csv(
            classes=classes,
            csv_path=str(datasplit_path),
            validation_prob=0.15,
            force_all_classes=False,
        )
        print(f"  datasplit.csv created in {time.time()-t0:.1f}s")
    else:
        print("[Phase 1/3] datasplit.csv already exists ✓")

    input_array_info = {"shape": input_shape, "scale": SCALE}
    target_array_info = {"shape": input_shape, "scale": SCALE}

    def _normalize(x):
        x = x.float()
        if x.max() > 1.5:
            x = x / 255.0
        return x.clamp(0.0, 1.0)

    raw_value_transforms = T.Compose([
        T.Lambda(_normalize),
        NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
    ])

    dl_kwargs = {k: v for k, v in DATALOADER_CONFIG.items()}
    dl_kwargs.setdefault('num_workers', 4)
    if dl_kwargs['num_workers'] == 0:
        dl_kwargs['persistent_workers'] = False
    if dl_kwargs.get('prefetch_factor') is None:
        dl_kwargs.pop('prefetch_factor', None)

    print(f"[Phase 2/3] Initializing dataloader ...")
    t0 = time.time()
    train_loader, _ = get_dataloader(
        datasplit_path=str(datasplit_path),
        classes=classes,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=SPATIAL_TRANSFORMS_2D,
        iterations_per_epoch=n_batches,
        train_raw_value_transforms=raw_value_transforms,
        val_raw_value_transforms=raw_value_transforms,
        random_validation=True,
        **dl_kwargs,
    )
    print(f"  Dataloader ready in {time.time()-t0:.1f}s")

    print(f"[Phase 3/3] Scanning {n_batches} batches (batch_size={batch_size}, "
          f"{n_batches * batch_size} total crops) ...")
    t0 = time.time()
    class_positive = defaultdict(int)
    class_total = defaultdict(int)

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        targets = batch['output']
        for c_idx, c_name in enumerate(classes):
            channel = targets[:, c_idx]
            valid = ~channel.isnan()
            n_valid = valid.sum().item()
            n_positive = (channel[valid] > 0.5).sum().item()
            class_positive[c_name] += int(n_positive)
            class_total[c_name] += int(n_valid)

        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (batch_idx + 1) / elapsed
            eta = (n_batches - batch_idx - 1) / rate
            print(f"  Batch {batch_idx+1}/{n_batches}  "
                  f"({rate:.1f} batch/s, ~{eta:.0f}s remaining)")

    scan_time = time.time() - t0
    print(f"  Scan complete in {scan_time:.1f}s "
          f"({n_batches * batch_size} crops processed)")

    print("\n--- Raw counts ---")
    freqs = {}
    for c in classes:
        pos = class_positive[c]
        tot = class_total[c]
        freq = pos / max(tot, 1)
        freqs[c] = freq
        print(f"  {c:20s}:  positive={pos:>12,}  total={tot:>12,}  freq={freq:.6f}")

    return dict(class_positive), dict(class_total), freqs


def compute_weight_strategies(class_positive, classes, beta=0.999):
    """From raw positive counts, compute normalized weight strategies."""
    counts = np.array([class_positive[c] for c in classes], dtype=np.float64)
    counts = np.maximum(counts, 1)

    total_voxels = counts.sum()
    freq = counts / total_voxels

    def _normalize(w):
        w = np.array(w, dtype=np.float64)
        return (w / w.max()).tolist()

    inv = 1.0 / freq
    sqrt_inv = 1.0 / np.sqrt(freq)
    log_inv = 1.0 / np.log1p(freq)
    effective_n = (1 - beta ** counts) / (1 - beta)

    strategies = {
        'inv_freq': {c: round(w, 6) for c, w in zip(classes, _normalize(inv))},
        'sqrt_inv': {c: round(w, 6) for c, w in zip(classes, _normalize(sqrt_inv))},
        'log_inv': {c: round(w, 6) for c, w in zip(classes, _normalize(log_inv))},
        'effective_num': {c: round(w, 6) for c, w in zip(classes, _normalize(1.0 / effective_n))},
    }
    return strategies


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-class frequencies for all 14 organelle classes')
    parser.add_argument('--n_batches', type=int, default=50,
                        help='Number of training batches to scan')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: class_frequencies.json)')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='Beta for effective number computation')
    args = parser.parse_args()

    classes = list(CLASSES)
    print(f"Computing frequencies for {len(classes)} classes: {classes}")

    class_positive, class_total, freqs = compute_frequencies(
        classes, n_batches=args.n_batches, batch_size=args.batch_size)

    strategies = compute_weight_strategies(class_positive, classes, beta=args.beta)

    print("\n--- Weight strategies (max-normalized) ---")
    for strat, weights in strategies.items():
        print(f"\n  {strat}:")
        for c, w in weights.items():
            print(f"    {c:20s}: {w:.6f}")

    output_path = args.output or str(SCRIPT_DIR / "class_frequencies.json")

    result = {
        'classes': classes,
        'n_classes': len(classes),
        'n_batches_scanned': args.n_batches,
        'raw_positive_counts': class_positive,
        'raw_total_counts': class_total,
        'frequencies': {c: round(f, 8) for c, f in freqs.items()},
        'weight_strategies': strategies,
        'estimated_voxel_counts': class_positive,
        'beta': args.beta,
    }
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Saved to {output_path}")
    print(f"\nESTIMATED_VOXEL_COUNTS = {json.dumps(class_positive)}")


if __name__ == '__main__':
    main()
