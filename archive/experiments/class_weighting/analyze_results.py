#!/usr/bin/env python3
"""
Post-experiment analysis for the class-weighting comparison.

Reads results JSONs from experiments/class_weighting/results/ and generates:
  1. Per-class Dice grouped bar chart
  2. Training/validation curves
  3. Strategy-family comparison (static vs CB vs BalancedSoftmax vs Seesaw)
  4. Console summary table

Usage:
    python analyze_results.py
    python analyze_results.py --results_dir /path/to/results
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import LOSS_CONFIGS, RESULTS_DIR


# ── Colour palette ────────────────────────────────────────────────────
FAMILY_COLORS = {
    'static': '#4C72B0',
    'class_balanced': '#DD8452',
    'balanced_softmax': '#55A868',
    'seesaw': '#C44E52',
}

def classify_family(loss_name):
    if loss_name.startswith('cb_'):
        return 'class_balanced'
    elif loss_name.startswith('balanced_softmax'):
        return 'balanced_softmax'
    elif loss_name.startswith('seesaw'):
        return 'seesaw'
    return 'static'


# ======================================================================
# Loaders
# ======================================================================

def load_all_results(results_dir: Path):
    results = {}
    for rfile in sorted(results_dir.glob("cw_*_results.json")):
        with open(rfile) as f:
            data = json.load(f)
        loss_name = data.get('loss_name', rfile.stem)
        results[loss_name] = data
    return results


# ======================================================================
# Plot 1: Per-class Dice grouped bars
# ======================================================================

def plot_per_class_dice(results, out_path):
    if not results:
        return

    loss_names = list(results.keys())
    # Gather class names from first result
    first = next(iter(results.values()))
    classes = list(first['history'][-1]['per_class'].keys()) if first['history'] else []
    if not classes:
        return

    n_losses = len(loss_names)
    n_classes = len(classes)
    x = np.arange(n_classes)
    width = 0.8 / n_losses

    fig, ax = plt.subplots(figsize=(max(14, n_losses * 1.5), 7))

    for i, lname in enumerate(loss_names):
        hist = results[lname].get('history', [])
        if not hist:
            continue
        last = hist[-1]['per_class']
        dices = [last.get(c, {}).get('dice', 0) for c in classes]
        family = classify_family(lname)
        ax.bar(x + i * width, dices, width, label=lname,
               color=FAMILY_COLORS.get(family, '#999999'), alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Dice Score', fontsize=12)
    ax.set_title('Per-Class Dice by Weighting Strategy', fontsize=14)
    ax.set_xticks(x + width * (n_losses - 1) / 2)
    ax.set_xticklabels(classes, rotation=30, ha='right')
    ax.legend(fontsize=7, ncol=3, loc='upper right')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ======================================================================
# Plot 2: Training / validation curves
# ======================================================================

def plot_training_curves(results, out_path):
    if not results:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for lname, data in results.items():
        hist = data.get('history', [])
        if not hist:
            continue
        epochs = [h['epoch'] for h in hist]
        family = classify_family(lname)
        color = FAMILY_COLORS.get(family, '#999999')
        alpha = 0.8

        axes[0].plot(epochs, [h['train_loss'] for h in hist],
                     label=lname, color=color, alpha=alpha)
        axes[1].plot(epochs, [h['val_loss'] for h in hist],
                     label=lname, color=color, alpha=alpha)
        axes[2].plot(epochs, [h['dice_mean'] for h in hist],
                     label=lname, color=color, alpha=alpha)

    axes[0].set_title('Training Loss')
    axes[1].set_title('Validation Loss')
    axes[2].set_title('Mean Dice')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ======================================================================
# Plot 3: Strategy-family comparison box / bar
# ======================================================================

def plot_family_comparison(results, out_path):
    if not results:
        return

    family_scores = {}
    for lname, data in results.items():
        fam = classify_family(lname)
        best = data.get('best_dice', 0)
        family_scores.setdefault(fam, []).append((lname, best))

    families = list(family_scores.keys())
    fig, ax = plt.subplots(figsize=(10, 6))

    x_offset = 0
    tick_positions = []
    tick_labels = []
    for fam in families:
        entries = family_scores[fam]
        color = FAMILY_COLORS.get(fam, '#999999')
        for lname, dice in entries:
            ax.bar(x_offset, dice, color=color, alpha=0.85,
                   edgecolor='white', linewidth=0.5)
            ax.text(x_offset, dice + 0.005, f"{dice:.3f}",
                    ha='center', va='bottom', fontsize=7)
            tick_positions.append(x_offset)
            tick_labels.append(lname)
            x_offset += 1
        x_offset += 0.5  # gap between families

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Best Mean Dice', fontsize=12)
    ax.set_title('Best Dice by Weighting Strategy Family', fontsize=14)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


# ======================================================================
# Console summary
# ======================================================================

def print_summary(results):
    print(f"\n{'='*72}")
    print("CLASS WEIGHTING COMPARISON — SUMMARY")
    print(f"{'='*72}")

    ranked = sorted(results.items(),
                    key=lambda x: x[1].get('best_dice', 0), reverse=True)

    header = f"{'Rank':<5} {'Loss Config':<30} {'Family':<18} {'Best Dice':<10} {'Time (m)':<8}"
    print(header)
    print('-' * len(header))

    for i, (lname, data) in enumerate(ranked, 1):
        fam = classify_family(lname)
        dice = data.get('best_dice', 0)
        elapsed = data.get('elapsed_min', 0)
        print(f"{i:<5} {lname:<30} {fam:<18} {dice:<10.4f} {elapsed:<8.1f}")

    # ── Per-class breakdown for top 5 ────────────────────────────
    print(f"\n{'='*72}")
    print("TOP-5 PER-CLASS BREAKDOWN")
    print(f"{'='*72}")
    for i, (lname, data) in enumerate(ranked[:5], 1):
        hist = data.get('history', [])
        if not hist:
            continue
        last_pc = hist[-1]['per_class']
        classes_str = '  '.join(f"{c}: {v['dice']:.3f}" for c, v in last_pc.items())
        print(f"  {i}. {lname}")
        print(f"     {classes_str}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyse class-weighting results')
    parser.add_argument('--results_dir', type=str, default=None)
    args = parser.parse_args()

    rdir = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    results = load_all_results(rdir)

    if not results:
        print(f"No result files found in {rdir}")
        sys.exit(1)

    print(f"Loaded {len(results)} experiment result(s) from {rdir}")
    plots_dir = rdir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots ...")
    plot_per_class_dice(results, plots_dir / "per_class_dice.png")
    plot_training_curves(results, plots_dir / "training_curves.png")
    plot_family_comparison(results, plots_dir / "family_comparison.png")

    print_summary(results)


if __name__ == '__main__':
    main()
