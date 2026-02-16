"""
Threshold tuning for per-class sigmoid cutoffs.

Runs inference on validation set with saved probability maps,
then sweeps thresholds per class to maximize Dice.

Usage:
    # First: run inference with --save-probs to get probability maps
    python inference.py --ensemble --split val --save-probs

    # Then: tune thresholds
    python tune_thresholds.py --probs-dir /path/to/predictions --split val
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "configs"))

import nibabel as nib

from data.ds_cellmap import load_datalist

CLASS_NAMES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
]
NUM_CLASSES = 14
RUNS_DIR = "/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap"


def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    intersection = (pred * target).sum()
    return float((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


def tune_thresholds(args):
    """Sweep thresholds per class on validation probability maps."""

    # Load datalist to get ground truth paths and annotation masks
    import importlib
    cfg = importlib.import_module("cfg_flexunet_resnet").cfg
    _, val_files = load_datalist(cfg)

    probs_dir = Path(args.probs_dir)
    thresholds_to_try = np.arange(0.1, 0.9, 0.05)

    print(f"Tuning thresholds on {len(val_files)} validation volumes")
    print(f"Threshold range: {thresholds_to_try[0]:.2f} to {thresholds_to_try[-1]:.2f}")
    print(f"Probability maps from: {probs_dir}")

    # Collect per-class dice at each threshold
    # shape: {class_name: {threshold: [dice_scores]}}
    results = {
        name: {f"{t:.2f}": [] for t in thresholds_to_try}
        for name in CLASS_NAMES
    }

    for file_entry in tqdm(val_files, desc="Volumes"):
        vol_name = Path(file_entry["image"]).stem.replace("_0000", "")
        prob_path = probs_dir / vol_name / "probabilities.nii.gz"

        if not prob_path.exists():
            print(f"  Skipping {vol_name} — no probability map found")
            continue

        # Load probabilities and ground truth
        probs = nib.load(str(prob_path)).get_fdata()  # (C, D, H, W)
        label_path = file_entry.get("label")
        if not label_path or not os.path.exists(label_path):
            continue

        gt_nii = nib.load(label_path)
        gt_data = gt_nii.get_fdata().astype(np.int64)

        # Parse annotation mask
        ann_str = file_entry.get("annotated_classes", "")
        annotated = set()
        if ann_str:
            for idx_str in str(ann_str).split(","):
                idx_str = idx_str.strip()
                if idx_str:
                    annotated.add(int(idx_str))

        # Sweep thresholds per class
        for c, class_name in enumerate(CLASS_NAMES):
            if annotated and c not in annotated:
                continue

            gt_binary = (gt_data == (c + 1)).astype(np.float32)

            # Skip if no ground truth for this class
            if gt_binary.sum() == 0 and c in annotated:
                # Class is annotated but has no positive voxels — still valid
                pass

            for t in thresholds_to_try:
                pred_binary = (probs[c] > t).astype(np.float32)
                d = compute_dice(pred_binary, gt_binary)
                results[class_name][f"{t:.2f}"].append(d)

    # Find best threshold per class
    print("\n" + "=" * 60)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 60)

    best_thresholds = {}
    for class_name in CLASS_NAMES:
        best_t = 0.5
        best_dice = -1.0

        for t_str, scores in results[class_name].items():
            if scores:
                mean_dice = np.mean(scores)
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    best_t = float(t_str)

        best_thresholds[class_name] = best_t
        n = len(results[class_name].get(f"{best_t:.2f}", []))
        print(f"  {class_name:12s}: threshold={best_t:.2f}  dice={best_dice:.4f}  (n={n})")

    # Save thresholds
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"\nThresholds saved to: {output_path}")

    # Also save detailed results
    detail_path = output_path.parent / "threshold_sweep_detail.json"
    detail = {}
    for class_name in CLASS_NAMES:
        detail[class_name] = {}
        for t_str, scores in results[class_name].items():
            if scores:
                detail[class_name][t_str] = {
                    "mean_dice": float(np.mean(scores)),
                    "std_dice": float(np.std(scores)),
                    "n": len(scores),
                }
    with open(detail_path, "w") as f:
        json.dump(detail, f, indent=2)
    print(f"Detailed sweep saved to: {detail_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune per-class thresholds")
    parser.add_argument("--probs-dir", type=str,
                        default=f"{RUNS_DIR}/predictions",
                        help="Directory with probability maps")
    parser.add_argument("--output", type=str,
                        default=f"{RUNS_DIR}/predictions/thresholds.json",
                        help="Output path for thresholds JSON")
    args = parser.parse_args()
    tune_thresholds(args)
