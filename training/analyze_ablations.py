"""
Analyze ablation experiment results.

Reads TensorBoard logs from all ablation runs and produces a comparison
summary with best validation loss, training curves, and rankings.

Usage:
    python -m training.analyze_ablations --run_dir runs/ablation
    python -m training.analyze_ablations --run_dir runs/ablation --plot
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


def load_tensorboard_scalars(log_dir: Path, tag: str) -> list[tuple[int, float]]:
    """Load scalar values from TensorBoard event files."""
    if not HAS_TB:
        return []

    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={event_accumulator.SCALARS: 0},  # load all
    )
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        return []

    events = ea.Scalars(tag)
    return [(e.step, e.value) for e in events]


def analyze_experiment(run_path: Path) -> dict:
    """Analyze a single experiment run."""
    result = {
        "name": run_path.name,
        "path": str(run_path),
    }

    # Load config
    config_path = run_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            result["config"] = json.load(f)
    else:
        result["config"] = {}

    # Load TensorBoard data
    tb_dir = run_path / "tensorboard"
    if tb_dir.exists() and HAS_TB:
        # Training loss
        train_loss = load_tensorboard_scalars(tb_dir, "train/epoch_loss")
        if train_loss:
            result["final_train_loss"] = train_loss[-1][1]
            result["best_train_loss"] = min(v for _, v in train_loss)
            result["train_loss_curve"] = train_loss

        # Validation loss
        val_loss = load_tensorboard_scalars(tb_dir, "val/loss")
        if val_loss:
            result["final_val_loss"] = val_loss[-1][1]
            result["best_val_loss"] = min(v for _, v in val_loss)
            result["best_val_epoch"] = min(val_loss, key=lambda x: x[1])[0]
            result["val_loss_curve"] = val_loss
    else:
        # Try to read from checkpoint
        ckpt_path = run_path / "checkpoints" / "latest.pth"
        if ckpt_path.exists():
            result["has_checkpoint"] = True

    return result


def print_comparison_table(results: list[dict], group_name: str = ""):
    """Print a formatted comparison table."""
    # Filter to experiments that have validation results
    valid = [r for r in results if "best_val_loss" in r]
    if not valid:
        print(f"\n{group_name}: No completed experiments found.")
        return

    # Sort by best validation loss
    valid.sort(key=lambda r: r["best_val_loss"])

    print(f"\n{'='*80}")
    print(f"  {group_name}")
    print(f"{'='*80}")
    print(f"  {'Rank':>4} {'Experiment':<40} {'Best Val':>10} {'Final Val':>10} {'Best Epoch':>10}")
    print(f"  {'-'*4} {'-'*40} {'-'*10} {'-'*10} {'-'*10}")

    for i, r in enumerate(valid):
        marker = " ★" if i == 0 else "  "
        print(
            f"{marker}{i+1:>3} {r['name']:<40} "
            f"{r['best_val_loss']:>10.4f} "
            f"{r.get('final_val_loss', float('nan')):>10.4f} "
            f"{r.get('best_val_epoch', 'N/A'):>10}"
        )

    if valid:
        print(f"\n  Winner: {valid[0]['name']} (best_val_loss={valid[0]['best_val_loss']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation results")
    parser.add_argument("--run_dir", type=str, default="runs/ablation")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    # Find all experiment directories
    experiments = sorted(run_dir.iterdir())
    experiments = [d for d in experiments if d.is_dir() and (d / "config.json").exists()]

    if not experiments:
        print(f"No experiments found in {run_dir}")
        return

    print(f"Found {len(experiments)} experiments in {run_dir}")

    # Analyze all
    results = [analyze_experiment(d) for d in experiments]

    # Group by experiment type
    groups = {
        "Loss Function Sweep (2D)": [r for r in results if r["name"].startswith("loss_2d_")],
        "Tversky α/β Sweep (2D)": [r for r in results if r["name"].startswith("tversky_2d_")],
        "Class Weighting (τ) Sweep (2D)": [r for r in results if r["name"].startswith("tau_2d_")],
        "Masking Strategy Sweep (2D)": [r for r in results if r["name"].startswith("mask_2d_")],
        "Loss Function Sweep (3D)": [r for r in results if r["name"].startswith("loss_3d_")],
        "Tversky α/β Sweep (3D)": [r for r in results if r["name"].startswith("tversky_3d_")],
        "Class Weighting (τ) Sweep (3D)": [r for r in results if r["name"].startswith("tau_3d_")],
        "Masking Strategy Sweep (3D)": [r for r in results if r["name"].startswith("mask_3d_")],
        "Architecture Comparison (2D)": [r for r in results if r["name"].startswith("arch_2d_")],
        "Architecture Comparison (3D)": [r for r in results if r["name"].startswith("arch_3d_")],
    }

    for group_name, group_results in groups.items():
        if group_results:
            print_comparison_table(group_results, group_name)

    # Overall summary
    all_valid = [r for r in results if "best_val_loss" in r]
    if all_valid:
        print_comparison_table(all_valid, "ALL EXPERIMENTS (Overall Ranking)")

    # Plot if requested
    if args.plot and all_valid:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            for group_name, group_results in groups.items():
                valid = [r for r in group_results if "val_loss_curve" in r]
                if not valid:
                    continue

                fig, ax = plt.subplots(1, 1, figsize=(12, 6))
                for r in valid:
                    steps, values = zip(*r["val_loss_curve"])
                    ax.plot(steps, values, label=r["name"], marker="o", markersize=3)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Validation Loss")
                ax.set_title(group_name)
                ax.legend(fontsize=8, loc="upper right")
                ax.grid(True, alpha=0.3)

                fname = group_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                fig.savefig(run_dir / f"{fname}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Plot saved: {run_dir / f'{fname}.png'}")

        except ImportError:
            print("matplotlib not available, skipping plots")

    # Save JSON summary
    summary_path = run_dir / "ablation_summary.json"
    summary = {}
    for r in results:
        entry = {k: v for k, v in r.items() if k not in ("train_loss_curve", "val_loss_curve")}
        summary[r["name"]] = entry
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
