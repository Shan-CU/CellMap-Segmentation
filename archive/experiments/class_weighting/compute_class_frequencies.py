#!/usr/bin/env python3
"""
Compute per-class voxel frequencies from the training data.

Scans the training dataset, counts the number of positive voxels per class,
and outputs six weight strategies to class_frequencies.json:

    inv_freq          – 1 / freq(c), normalized so max = 1
    sqrt_inv          – 1 / sqrt(freq(c)), normalized
    log_inv           – 1 / log(1 + freq(c)), normalized
    effective_num     – (1 - β^n) / (1 - β), β=0.999, normalized
    manual (passthrough of raw counts for reference)

Usage:
    python compute_class_frequencies.py                 # 100 batches (default)
    python compute_class_frequencies.py --n_batches 200 # more samples
    python compute_class_frequencies.py --output freqs.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.v2 as T
from cellmap_data.transforms.augment import NaNtoNum

# ── Path setup ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent.parent))        # repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "loss_optimization"))
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    QUICK_TEST_CLASSES, DATA_ROOT, SPATIAL_TRANSFORMS_2D,
    DATALOADER_CONFIG, MODEL_CONFIG,
)


def compute_frequencies(classes, n_batches=50, batch_size=24,
                        input_shape=(1, 256, 256)):
    """Scan training data and count per-class positive voxels."""
    import time as _time
    from cellmap_segmentation_challenge.utils.dataloader import get_dataloader

    datasplit_path = Path(__file__).parent / "datasplit.csv"
    if not datasplit_path.exists():
        from cellmap_segmentation_challenge.utils.datasplit import make_datasplit_csv
        print("[Phase 1/3] Creating datasplit.csv (indexes all zarr datasets — ~5 min first time) ...")
        t0 = _time.time()
        make_datasplit_csv(
            classes=classes,
            csv_path=str(datasplit_path),
            validation_prob=0.15,
            force_all_classes=False,
        )
        print(f"  datasplit.csv created in {_time.time()-t0:.1f}s")
    else:
        print("[Phase 1/3] datasplit.csv already exists ✓")

    input_array_info = {"shape": input_shape, "scale": (8, 8, 8)}
    target_array_info = {"shape": (1, 256, 256), "scale": (8, 8, 8)}

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
    # Use workers for parallel loading (much faster than num_workers=0)
    dl_kwargs.setdefault('num_workers', 4)
    if dl_kwargs['num_workers'] == 0:
        dl_kwargs['persistent_workers'] = False

    print(f"[Phase 2/3] Initializing dataloader (opens zarr arrays) ...")
    t0 = _time.time()
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
    print(f"  Dataloader ready in {_time.time()-t0:.1f}s")

    print(f"[Phase 3/3] Scanning {n_batches} batches (batch_size={batch_size}, "
          f"{n_batches * batch_size} total crops) ...")
    t0 = _time.time()
    class_positive = defaultdict(int)
    class_total = defaultdict(int)

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= n_batches:
            break
        targets = batch['output']            # (B, C, D, H, W) or (B, C, H, W)
        for c_idx, c_name in enumerate(classes):
            channel = targets[:, c_idx]
            valid = ~channel.isnan()
            n_valid = valid.sum().item()
            n_positive = (channel[valid] > 0.5).sum().item()
            class_positive[c_name] += int(n_positive)
            class_total[c_name] += int(n_valid)

        if (batch_idx + 1) % 10 == 0:
            elapsed = _time.time() - t0
            rate = (batch_idx + 1) / elapsed
            eta = (n_batches - batch_idx - 1) / rate
            print(f"  Batch {batch_idx+1}/{n_batches}  "
                  f"({rate:.1f} batch/s, ~{eta:.0f}s remaining)")

    scan_time = _time.time() - t0
    print(f"  Scan complete in {scan_time:.1f}s "
          f"({n_batches * batch_size} crops processed)")

    # ── Derive frequencies ────────────────────────────────────────
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
    """
    From raw positive counts, compute normalized weight strategies.

    Returns dict of {strategy_name: {class: weight}}.
    """
    counts = np.array([class_positive[c] for c in classes], dtype=np.float64)
    counts = np.maximum(counts, 1)          # avoid div-by-zero

    total_voxels = counts.sum()
    freq = counts / total_voxels            # per-class frequency

    def _normalize(w):
        """Normalize so max = 1."""
        w = np.array(w, dtype=np.float64)
        return (w / w.max()).tolist()

    # 1) Inverse frequency
    inv = 1.0 / freq
    inv_weights = _normalize(inv)

    # 2) Sqrt inverse frequency
    sqrt_inv = 1.0 / np.sqrt(freq)
    sqrt_weights = _normalize(sqrt_inv)

    # 3) Log inverse frequency
    log_inv = 1.0 / np.log1p(freq)
    log_weights = _normalize(log_inv)

    # 4) Effective number (β = 0.999)
    effective_n = (1 - beta ** counts) / (1 - beta)
    eff_weights = _normalize(1.0 / effective_n)

    strategies = {
        'inv_freq': {c: round(w, 6) for c, w in zip(classes, inv_weights)},
        'sqrt_inv': {c: round(w, 6) for c, w in zip(classes, sqrt_weights)},
        'log_inv': {c: round(w, 6) for c, w in zip(classes, log_weights)},
        'effective_num': {c: round(w, 6) for c, w in zip(classes, eff_weights)},
    }
    return strategies


def main():
    parser = argparse.ArgumentParser(
        description='Compute per-class frequencies and weight strategies')
    parser.add_argument('--n_batches', type=int, default=50,
                        help='Number of training batches to scan (50 × 24 = 1200 crops)')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON path (default: class_frequencies.json)')
    parser.add_argument('--beta', type=float, default=0.999,
                        help='Beta for effective number computation')
    args = parser.parse_args()

    classes = list(QUICK_TEST_CLASSES)
    class_positive, class_total, freqs = compute_frequencies(
        classes, n_batches=args.n_batches, batch_size=args.batch_size)

    strategies = compute_weight_strategies(class_positive, classes, beta=args.beta)

    print("\n--- Weight strategies (max-normalized) ---")
    for strat, weights in strategies.items():
        print(f"\n  {strat}:")
        for c, w in weights.items():
            print(f"    {c:20s}: {w:.6f}")

    # ── Also emit the dict format for config.py ───────────────────
    print("\n--- Copy-paste into config.py LOSS_CONFIGS ---")
    for strat, weights in strategies.items():
        print(f"\n  # {strat}")
        print(f"  'class_weights': {json.dumps(weights)},")

    # ── Save to JSON ──────────────────────────────────────────────
    output_path = args.output or str(
        Path(__file__).parent / "class_frequencies.json")

    result = {
        'classes': classes,
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

    # ── Also print the ESTIMATED_VOXEL_COUNTS dict ────────────────
    print("\n--- Copy-paste for config.py ESTIMATED_VOXEL_COUNTS ---")
    print(f"ESTIMATED_VOXEL_COUNTS = {json.dumps(class_positive)}")


if __name__ == '__main__':
    main()
