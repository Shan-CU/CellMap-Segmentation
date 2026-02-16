#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive evaluation: test all trained models on all validation datasets.

This script:
1. Loads each trained masking strategy checkpoint (best dice)
2. Evaluates on each validation dataset separately
3. Generates per-dataset metrics (Dice, Precision, Recall, IoU)
4. Produces comparison tables and CSV outputs

Usage:
    python evaluate_all_datasets.py
    python evaluate_all_datasets.py --checkpoint_dir ./checkpoints
    python evaluate_all_datasets.py --datasets jrc_hela-2 jrc_jurkat-1
    python evaluate_all_datasets.py --batch_limit 50  # faster evaluation
"""

import os
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'

import argparse
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from cellmap_data.transforms.augment import NaNtoNum, Binarize
from tqdm import tqdm

# Path setup
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(REPO_ROOT))
sys.path.insert(2, str(REPO_ROOT / "src"))

from config import (
    MASKING_CONFIGS, QUICK_TEST_CLASSES,
    DATA_ROOT, DATASPLIT_CSV,
    SPATIAL_TRANSFORMS_2D, DATALOADER_CONFIG,
    MODEL_CONFIG, USE_AMP,
)

# Import the REAL model used in training
from cellmap_segmentation_challenge.models import UNet_2D
from cellmap_segmentation_challenge.utils.dataloader import get_dataloader


# ============================================================
# Dataset Utilities
# ============================================================

def get_available_validation_datasets():
    """Extract list of datasets with validation data from the master datasplit."""
    with open(DATASPLIT_CSV, 'r') as f:
        lines = f.readlines()

    datasets = set()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Each line starts with "train" or "validate" (quoted)
        if 'validate' in line.split(',')[0].strip('"').lower():
            # Extract dataset name from path
            for part in line.split(','):
                part = part.strip('"')
                if '.zarr' in part:
                    # path like .../jrc_hela-2/jrc_hela-2.zarr/...
                    zarr_idx = part.find('.zarr')
                    prefix = part[:zarr_idx]
                    dataset_name = prefix.split('/')[-1]
                    datasets.add(dataset_name)
                    break

    return sorted(datasets)


def create_per_dataset_datasplit(dataset_name):
    """Create a temporary datasplit CSV for evaluating a specific dataset.

    Includes ALL original training rows (so CellMapDataSplit doesn't crash)
    but only validation rows from the target dataset.
    """
    with open(DATASPLIT_CSV, 'r') as f:
        lines = f.readlines()

    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        first_field = stripped.split(',')[0].strip('"')
        if 'train' in first_field.lower():
            # Keep all training rows
            kept.append(stripped)
        elif 'validate' in first_field.lower():
            # Only keep validation rows for this dataset
            if dataset_name in stripped:
                kept.append(stripped)

    if not any('validate' in l.split(',')[0].strip('"').lower() for l in kept):
        return None

    tmp = tempfile.NamedTemporaryFile(
        mode='w', suffix='.csv', delete=False, prefix=f'eval_{dataset_name}_'
    )
    tmp.write('\n'.join(kept) + '\n')
    tmp.close()
    return tmp.name


def create_val_loader_for_dataset(dataset_name, batch_size=1,
                                   iterations_per_epoch=200):
    """Create a validation dataloader for a specific dataset.

    Uses the same get_dataloader() as train.py so batch format,
    transforms, and array shapes are identical.
    """
    split_csv = create_per_dataset_datasplit(dataset_name)
    if split_csv is None:
        return None, None

    input_shape = MODEL_CONFIG['input_shape']      # (1, 256, 256)
    input_array_info = {"shape": input_shape, "scale": (8, 8, 8)}
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
    # Force single-process for eval to avoid issues
    dl_kwargs['num_workers'] = 2
    dl_kwargs['persistent_workers'] = True

    try:
        _train_loader, val_loader = get_dataloader(
            datasplit_path=split_csv,
            classes=QUICK_TEST_CLASSES,
            batch_size=batch_size,
            input_array_info=input_array_info,
            target_array_info=target_array_info,
            spatial_transforms=SPATIAL_TRANSFORMS_2D,
            iterations_per_epoch=iterations_per_epoch,
            train_raw_value_transforms=raw_value_transforms,
            val_raw_value_transforms=raw_value_transforms,
            random_validation=True,
            **dl_kwargs,
        )
        return val_loader, split_csv
    except Exception as e:
        print(f"    ⚠️  Failed to create loader for {dataset_name}: {e}")
        # Clean up
        if Path(split_csv).exists():
            Path(split_csv).unlink()
        return None, None


# ============================================================
# Model Loading
# ============================================================

def load_model(checkpoint_path, device='cuda'):
    """Load model from checkpoint, matching train.py's create_model()."""
    n_classes = len(QUICK_TEST_CLASSES)
    input_channels = MODEL_CONFIG['input_channels']  # 1

    model = UNet_2D(input_channels, n_classes)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']

    # Strip DDP / torch.compile prefixes
    new_state = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        new_state[k] = v

    model.load_state_dict(new_state)
    model = model.to(device)
    model.eval()
    return model


def find_best_checkpoint(checkpoint_dir, strategy_name):
    """Find best checkpoint for a strategy (highest dice score)."""
    checkpoint_dir = Path(checkpoint_dir)

    # Pattern: mask_<strategy>_<timestamp>_best.pt
    candidates = list(checkpoint_dir.glob(f"mask_{strategy_name}_*_best.pt"))

    if not candidates:
        return None, -1.0

    best_ckpt = None
    best_dice = -1.0

    for ckpt_path in candidates:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Checkpoints store 'best_dice' (some older ones may use 'val_dice')
            dice = ckpt.get('best_dice', ckpt.get('val_dice', -1.0))
            if dice > best_dice:
                best_dice = dice
                best_ckpt = ckpt_path
        except Exception:
            continue

    return best_ckpt, best_dice


# ============================================================
# Metrics  (same logic as train.py: sigmoid > 0.5, NaN-masked)
# ============================================================

def compute_batch_counts(pred, target):
    """Compute TP/FP/FN per class, respecting NaN mask.

    Identical to train.py's compute_batch_counts.
    - pred:   raw logits  [B, C, H, W]
    - target: float with NaN for unannotated  [B, C, H, W]
    """
    pred_binary = (torch.sigmoid(pred) > 0.5).float()
    valid_mask = ~target.isnan()
    target_clean = target.nan_to_num(0)

    tp_list, fp_list, fn_list = [], [], []
    for c in range(pred.shape[1]):
        p = pred_binary[:, c] * valid_mask[:, c]
        t = target_clean[:, c] * valid_mask[:, c]
        tp_list.append((p * t).sum().item())
        fp_list.append((p * (1 - t)).sum().item())
        fn_list.append(((1 - p) * t).sum().item())
    return {'tp': tp_list, 'fp': fp_list, 'fn': fn_list}


@torch.no_grad()
def evaluate_model_on_dataset(model, val_loader, device='cuda', batch_limit=200):
    """Evaluate model on a validation dataset.

    Uses the same metric logic as train.py's validate():
    - Accumulate global TP/FP/FN per class across all batches
    - Compute Dice/Precision/Recall/IoU from totals
    """
    model.eval()
    classes = QUICK_TEST_CLASSES
    n_classes = len(classes)

    global_tp = [0.0] * n_classes
    global_fp = [0.0] * n_classes
    global_fn = [0.0] * n_classes
    n_batches = 0

    pbar = tqdm(enumerate(val_loader), total=min(batch_limit, len(val_loader)),
                desc="    Evaluating", leave=False)

    for batch_idx, batch in pbar:
        if batch_idx >= batch_limit:
            break

        # Same input handling as train.py
        inputs = batch['input'].to(device, non_blocking=True)
        if inputs.dim() == 5 and inputs.shape[1] == 1:
            inputs = inputs.squeeze(1)
        targets = batch['output'].to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=USE_AMP):
            outputs = model(inputs)

        counts = compute_batch_counts(outputs.detach(), targets.detach())
        for i in range(n_classes):
            global_tp[i] += counts['tp'][i]
            global_fp[i] += counts['fp'][i]
            global_fn[i] += counts['fn'][i]

        n_batches += 1
        del inputs, targets, outputs

    if n_batches == 0:
        return None

    # Compute per-class metrics (same formulas as train.py validate())
    per_class = {}
    for i, c in enumerate(classes):
        tp, fp, fn = global_tp[i], global_fp[i], global_fn[i]
        denom_dice = 2 * tp + fp + fn
        dice = (2 * tp / denom_dice) if denom_dice > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec  = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        iou_denom = tp + fp + fn
        iou  = (tp / iou_denom) if iou_denom > 0 else 0.0
        per_class[c] = {
            'dice': dice, 'precision': prec, 'recall': rec, 'iou': iou,
            'tp': tp, 'fp': fp, 'fn': fn,
        }

    mean_dice = sum(m['dice'] for m in per_class.values()) / n_classes
    mean_prec = sum(m['precision'] for m in per_class.values()) / n_classes
    mean_rec  = sum(m['recall'] for m in per_class.values()) / n_classes
    mean_iou  = sum(m['iou'] for m in per_class.values()) / n_classes

    return {
        'dice_mean': mean_dice,
        'precision_mean': mean_prec,
        'recall_mean': mean_rec,
        'iou_mean': mean_iou,
        'per_class': per_class,
        'n_batches': n_batches,
    }


# ============================================================
# Pretty Printing
# ============================================================

def print_per_class_table(results_matrix, datasets):
    """Print per-class metrics for each strategy (averaged across datasets)."""
    classes = QUICK_TEST_CLASSES

    print(f"\n{'═'*90}")
    print(f"  PER-CLASS BREAKDOWN (averaged across datasets)")
    print(f"{'═'*90}")

    for metric_name in ['dice', 'precision', 'recall', 'iou']:
        print(f"\n  ── {metric_name.upper()} per class ──")
        header = f"  {'Strategy':<27}"
        for c in classes:
            header += f" {c:>10}"
        header += f" {'MEAN':>10}"
        print(header)
        print(f"  {'─'*87}")

        for strategy in sorted(results_matrix.keys()):
            ds_results = results_matrix[strategy]
            if not ds_results:
                continue

            # Average per-class values across datasets
            class_vals = {c: [] for c in classes}
            for ds_metrics in ds_results.values():
                if ds_metrics is None:
                    continue
                for c in classes:
                    class_vals[c].append(ds_metrics['per_class'][c][metric_name])

            row = f"  {strategy:<27}"
            vals = []
            for c in classes:
                if class_vals[c]:
                    v = np.mean(class_vals[c])
                    vals.append(v)
                    row += f" {v:>10.4f}"
                else:
                    row += f" {'N/A':>10}"
            if vals:
                row += f" {np.mean(vals):>10.4f}"
            print(row)


# ============================================================
# Main Evaluation Pipeline
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate all models on all datasets')
    parser.add_argument('--checkpoint_dir', type=str,
                       default=str(SCRIPT_DIR / 'checkpoints'),
                       help='Directory containing checkpoints')
    parser.add_argument('--output_dir', type=str,
                       default=str(SCRIPT_DIR / 'evaluation_results'),
                       help='Output directory for results')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific datasets to evaluate (default: all)')
    parser.add_argument('--strategies', nargs='+', default=None,
                       help='Specific strategies to evaluate (default: all with checkpoints)')
    parser.add_argument('--batch_limit', type=int, default=200,
                       help='Max batches per dataset (default: 200)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (default: cuda)')

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # ── Discover validation datasets ──
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = get_available_validation_datasets()

    print(f"{'═'*70}")
    print(f"  Comprehensive Evaluation: all models × all datasets")
    print(f"{'═'*70}")
    print(f"\n  Validation datasets ({len(datasets)}):")
    for ds in datasets:
        print(f"    • {ds}")

    # ── Discover strategies with checkpoints ──
    if args.strategies:
        strategies = args.strategies
    else:
        strategies = list(MASKING_CONFIGS.keys())

    # Pre-scan checkpoints and report
    strategy_ckpts = {}
    print(f"\n  Strategies ({len(strategies)}):")
    for strat in strategies:
        ckpt_path, ckpt_dice = find_best_checkpoint(args.checkpoint_dir, strat)
        if ckpt_path is not None:
            strategy_ckpts[strat] = (ckpt_path, ckpt_dice)
            print(f"    ✅ {strat:<27} best_dice={ckpt_dice:.4f}  ({ckpt_path.name})")
        else:
            print(f"    ⬜ {strat:<27} no checkpoint found")

    if not strategy_ckpts:
        print("\n❌ No checkpoints found! Nothing to evaluate.")
        return

    print(f"\n  Batch limit: {args.batch_limit} per dataset")
    print(f"  Total evaluations: {len(strategy_ckpts)} strategies × {len(datasets)} datasets"
          f" = {len(strategy_ckpts) * len(datasets)}")
    print()

    # ── Main evaluation loop ──
    results_matrix = {}  # {strategy: {dataset: metrics_dict}}

    for strat_idx, (strategy, (ckpt_path, ckpt_dice)) in enumerate(strategy_ckpts.items()):
        print(f"\n{'═'*70}")
        print(f"  [{strat_idx+1}/{len(strategy_ckpts)}] Strategy: {strategy}")
        print(f"  Checkpoint: {ckpt_path.name}  (dice={ckpt_dice:.4f})")
        print(f"{'═'*70}")

        # Load model once per strategy
        try:
            model = load_model(ckpt_path, device=args.device)
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            continue

        results_matrix[strategy] = {}

        for ds_idx, dataset in enumerate(datasets):
            print(f"\n  [{ds_idx+1}/{len(datasets)}] {dataset}", end="", flush=True)

            val_loader, split_csv = create_val_loader_for_dataset(
                dataset,
                batch_size=1,
                iterations_per_epoch=args.batch_limit,
            )

            if val_loader is None:
                print("  ⚠️  skipped (no val data)")
                continue

            try:
                metrics = evaluate_model_on_dataset(
                    model, val_loader,
                    device=args.device,
                    batch_limit=args.batch_limit,
                )

                if metrics is None:
                    print("  ⚠️  no batches evaluated")
                    continue

                results_matrix[strategy][dataset] = metrics

                print(f"  →  Dice={metrics['dice_mean']:.4f}  "
                      f"Prec={metrics['precision_mean']:.4f}  "
                      f"Rec={metrics['recall_mean']:.4f}  "
                      f"IoU={metrics['iou_mean']:.4f}  "
                      f"({metrics['n_batches']} batches)")

            except Exception as e:
                print(f"  ❌ {e}")
                import traceback
                traceback.print_exc()

            finally:
                # Clean up temp CSV
                if split_csv and Path(split_csv).exists():
                    try:
                        Path(split_csv).unlink()
                    except OSError:
                        pass

        # Free GPU memory between strategies
        del model
        torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════════
    # Save results
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  Saving results...")
    print(f"{'═'*70}")

    # Convert to JSON-serializable
    json_results = {}
    for strategy, ds_results in results_matrix.items():
        json_results[strategy] = {}
        for dataset, metrics in ds_results.items():
            json_results[strategy][dataset] = {
                'dice_mean': metrics['dice_mean'],
                'precision_mean': metrics['precision_mean'],
                'recall_mean': metrics['recall_mean'],
                'iou_mean': metrics['iou_mean'],
                'n_batches': metrics['n_batches'],
                'per_class': {
                    c: {k: float(v) for k, v in cls_m.items()}
                    for c, cls_m in metrics['per_class'].items()
                },
            }

    results_json = output_dir / 'all_results.json'
    with open(results_json, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  ✅ {results_json}")

    # ══════════════════════════════════════════════════════════
    # Table 1: Overall performance (averaged across datasets)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  OVERALL PERFORMANCE (averaged across all datasets)")
    print(f"{'═'*70}")
    print(f"  {'Strategy':<27} {'Dice':>8} {'Prec':>8} {'Rec':>8} {'IoU':>8}  {'#DS':>4}")
    print(f"  {'─'*67}")

    overall_summary = []
    for strategy in sorted(results_matrix.keys()):
        ds_results = results_matrix[strategy]
        if not ds_results:
            continue

        dice_vals = [m['dice_mean'] for m in ds_results.values()]
        prec_vals = [m['precision_mean'] for m in ds_results.values()]
        rec_vals  = [m['recall_mean'] for m in ds_results.values()]
        iou_vals  = [m['iou_mean'] for m in ds_results.values()]

        mean_dice = np.mean(dice_vals)
        mean_prec = np.mean(prec_vals)
        mean_rec  = np.mean(rec_vals)
        mean_iou  = np.mean(iou_vals)

        print(f"  {strategy:<27} {mean_dice:>8.4f} {mean_prec:>8.4f} "
              f"{mean_rec:>8.4f} {mean_iou:>8.4f}  {len(ds_results):>4}")

        overall_summary.append({
            'strategy': strategy,
            'dice': mean_dice, 'precision': mean_prec,
            'recall': mean_rec, 'iou': mean_iou,
            'n_datasets': len(ds_results),
        })

    # Sort by dice descending
    overall_summary.sort(key=lambda x: x['dice'], reverse=True)
    pd.DataFrame(overall_summary).to_csv(output_dir / 'overall_summary.csv', index=False)

    # Highlight best
    if overall_summary:
        best = overall_summary[0]
        print(f"\n  🏆 Best overall: {best['strategy']}  "
              f"(Dice={best['dice']:.4f}, IoU={best['iou']:.4f})")

    # ══════════════════════════════════════════════════════════
    # Table 2: Per-dataset breakdown
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  PER-DATASET PERFORMANCE")
    print(f"{'═'*70}")

    per_dataset_summary = []
    for dataset in datasets:
        # Check if any strategy has results for this dataset
        has_results = any(
            dataset in results_matrix.get(s, {}) for s in results_matrix
        )
        if not has_results:
            continue

        print(f"\n  Dataset: {dataset}")
        print(f"  {'Strategy':<27} {'Dice':>8} {'Prec':>8} {'Rec':>8} {'IoU':>8}")
        print(f"  {'-'*67}")

        for strategy in sorted(results_matrix.keys()):
            if dataset not in results_matrix[strategy]:
                continue

            m = results_matrix[strategy][dataset]
            print(f"  {strategy:<27} {m['dice_mean']:>8.4f} "
                  f"{m['precision_mean']:>8.4f} "
                  f"{m['recall_mean']:>8.4f} "
                  f"{m['iou_mean']:>8.4f}")

            per_dataset_summary.append({
                'dataset': dataset,
                'strategy': strategy,
                'dice': m['dice_mean'],
                'precision': m['precision_mean'],
                'recall': m['recall_mean'],
                'iou': m['iou_mean'],
            })

    pd.DataFrame(per_dataset_summary).to_csv(
        output_dir / 'per_dataset_summary.csv', index=False
    )

    # ══════════════════════════════════════════════════════════
    # Table 3: Per-class breakdown (averaged across datasets)
    # ══════════════════════════════════════════════════════════
    print_per_class_table(results_matrix, datasets)

    # ══════════════════════════════════════════════════════════
    # Done
    # ══════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print("  ✅ Evaluation complete!")
    print(f"{'═'*70}")
    print(f"  Results saved to: {output_dir}")
    print(f"    • all_results.json         — full raw results")
    print(f"    • overall_summary.csv      — mean performance across datasets")
    print(f"    • per_dataset_summary.csv  — per-dataset breakdown")
    print()


if __name__ == '__main__':
    main()
