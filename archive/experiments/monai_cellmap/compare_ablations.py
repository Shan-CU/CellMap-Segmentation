#!/usr/bin/env python3
"""
Compare ablation experiment results.

Reads ablation_summary.json from each completed run and produces a
ranked comparison table.

Usage:
    python compare_ablations.py --axis masking
    python compare_ablations.py --axis weighting
    python compare_ablations.py --axis tversky
    python compare_ablations.py --all
"""

import argparse
import json
import os
import glob
from pathlib import Path


ABLATION_DIR = "/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/ablations"


def load_results(axis=None):
    """Load all ablation summary JSONs, optionally filtered by axis."""
    results = []
    pattern = os.path.join(ABLATION_DIR, "*/ablation_summary.json")

    for path in sorted(glob.glob(pattern)):
        with open(path) as f:
            data = json.load(f)
        data["_path"] = str(Path(path).parent)

        # Determine axis from name prefix
        name = data.get("name", "")
        if name.startswith("abl_mask_"):
            data["_axis"] = "masking"
        elif name.startswith("abl_weight_"):
            data["_axis"] = "weighting"
        elif name.startswith("abl_tversky_"):
            data["_axis"] = "tversky"
        else:
            data["_axis"] = "unknown"

        if axis is None or data["_axis"] == axis:
            results.append(data)

    return results


def print_table(results, title="Ablation Results"):
    """Print a ranked comparison table."""
    if not results:
        print(f"\n{title}: No results found.\n")
        return

    # Sort by best_dice descending
    results.sort(key=lambda x: x.get("best_dice", 0), reverse=True)

    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"{'Rank':>4}  {'Name':30s}  {'Dice':>8}  {'Loss':25s}  {'Key Params'}")
    print(f"{'─'*4}  {'─'*30}  {'─'*8}  {'─'*25}  {'─'*30}")

    best_dice = results[0].get("best_dice", 0)
    for i, r in enumerate(results):
        dice = r.get("best_dice", 0)
        delta = dice - best_dice if i > 0 else 0
        delta_str = f" ({delta:+.4f})" if i > 0 else " (best)"

        name = r.get("name", "?")
        loss = r.get("loss_type", "?")
        alpha = r.get("tversky_alpha", "?")
        beta = r.get("tversky_beta", "?")
        tau = r.get("tau", "N/A")
        bg_w = r.get("bbox_bg_weight", "?")
        pad = r.get("bbox_pad_fraction", "?")
        msup = r.get("masksup_ratio", 0)

        params = f"bg={bg_w}, pad={pad}, τ={tau}"
        if msup > 0:
            params += f", msup={msup}"

        print(f"{i+1:>4}  {name:30s}  {dice:.4f}{delta_str:>12s}  "
              f"α={alpha}/β={beta} {loss[:15]:>15s}  {params}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Compare ablation results")
    parser.add_argument("--axis", type=str, default=None,
                        choices=["masking", "weighting", "tversky"],
                        help="Filter by ablation axis")
    parser.add_argument("--all", action="store_true",
                        help="Show all axes")
    args = parser.parse_args()

    if args.all or args.axis is None:
        for axis in ["masking", "weighting", "tversky"]:
            results = load_results(axis)
            if results:
                print_table(results, f"{axis.upper()} Ablation Results")
    else:
        results = load_results(args.axis)
        print_table(results, f"{args.axis.upper()} Ablation Results")


if __name__ == "__main__":
    main()
