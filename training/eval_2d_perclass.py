"""
Per-class validation evaluation for all 2D ablation runs.

Loads each best.pth checkpoint, runs inference on the validation set,
and computes per-class Dice and IoU scores. Results are saved to JSON
and a summary CSV.

Usage:
    python -m training.eval_2d_perclass [--run_dir runs/ablation] [--max_batches 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cellmap_segmentation_challenge.utils import get_dataloader, get_tested_classes
from training.models.model_zoo import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Per-class validation evaluation")
    parser.add_argument("--run_dir", type=str, default="runs/ablation",
                        help="Base directory containing experiment folders")
    parser.add_argument("--max_batches", type=int, default=200,
                        help="Max validation batches per experiment (0=all)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Evaluate only this experiment (default: all *2d*)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: {run_dir}/eval_2d_perclass.json)")
    return parser.parse_args()


def compute_perclass_metrics(
    model: nn.Module,
    val_loader,
    classes: list[str],
    device: str,
    input_key: str = "input",
    target_key: str = "target",
    max_batches: int = 0,
    use_amp: bool = True,
) -> dict:
    """
    Compute per-class Dice and IoU on the validation set.
    """
    model.eval()
    num_classes = len(classes)

    # Accumulators
    tp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fn = torch.zeros(num_classes, dtype=torch.float64, device=device)
    annotated_voxels = torch.zeros(num_classes, dtype=torch.float64, device=device)

    val_loader.refresh()
    n_batches = 0
    t0 = time.time()

    with torch.no_grad():
        for batch in val_loader.loader:
            inputs = batch[input_key].to(device)
            targets = batch[target_key].to(device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]

            # Predictions: sigmoid > 0.5
            preds = (torch.sigmoid(outputs.float()) > 0.5)  # (B, C, H, W) bool

            # Annotation mask: where targets are not NaN
            not_nan = ~targets.isnan()  # (B, C, H, W)
            targets_clean = targets.nan_to_num(0.0)
            targets_bool = (targets_clean > 0.5)  # (B, C, H, W) bool

            # Masked predictions and targets (only where annotated)
            p = preds & not_nan       # predicted positive AND annotated
            t = targets_bool & not_nan  # ground truth positive AND annotated

            # TP: predicted positive AND ground truth positive (within annotated)
            batch_tp = (p & t).sum(dim=(0, 2, 3)).to(torch.float64)
            # FP: predicted positive AND ground truth negative (within annotated)
            batch_fp = (p & ~t).sum(dim=(0, 2, 3)).to(torch.float64)
            # FN: predicted negative AND ground truth positive (within annotated)
            batch_fn = (~p & t).sum(dim=(0, 2, 3)).to(torch.float64)

            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            annotated_voxels += not_nan.sum(dim=(0, 2, 3)).to(torch.float64)

            n_batches += 1
            if max_batches > 0 and n_batches >= max_batches:
                break

            if n_batches % 50 == 0:
                elapsed = time.time() - t0
                print(f"  Batch {n_batches} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"  Done: {n_batches} batches in {elapsed:.1f}s")

    # Compute metrics
    tp_np = tp.cpu().numpy()
    fp_np = fp.cpu().numpy()
    fn_np = fn.cpu().numpy()
    ann_np = annotated_voxels.cpu().numpy()

    per_class = {}
    valid_dices = []
    valid_ious = []

    for i, cls in enumerate(classes):
        denom_dice = 2 * tp_np[i] + fp_np[i] + fn_np[i]
        denom_iou = tp_np[i] + fp_np[i] + fn_np[i]

        dice = (2 * tp_np[i] / denom_dice) if denom_dice > 0 else 0.0
        iou = (tp_np[i] / denom_iou) if denom_iou > 0 else 0.0

        per_class[cls] = {
            "dice": float(dice),
            "iou": float(iou),
            "tp": int(tp_np[i]),
            "fp": int(fp_np[i]),
            "fn": int(fn_np[i]),
            "annotated_voxels": int(ann_np[i]),
        }

        # Only count classes with actual annotation data
        if ann_np[i] > 0 and (tp_np[i] + fn_np[i]) > 0:
            valid_dices.append(dice)
            valid_ious.append(iou)

    mean_dice = float(np.mean(valid_dices)) if valid_dices else 0.0
    mean_iou = float(np.mean(valid_ious)) if valid_ious else 0.0

    return {
        "per_class": per_class,
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "n_batches": n_batches,
        "n_classes_evaluated": len(valid_dices),
        "eval_time_s": elapsed,
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # Find 2D experiments
    if args.experiment:
        exp_dirs = [run_dir / args.experiment]
    else:
        exp_dirs = sorted([
            d for d in run_dir.iterdir()
            if d.is_dir() and "2d" in d.name and (d / "checkpoints" / "best.pth").exists()
        ])

    print(f"Found {len(exp_dirs)} 2D experiments to evaluate")
    for d in exp_dirs:
        print(f"  {d.name}")

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    classes = get_tested_classes()
    num_classes = len(classes)
    print(f"Classes: {num_classes}")

    # Data loader — create once, reuse for all experiments
    input_array_info = {"shape": (1, 256, 256), "scale": (8.0, 8.0, 8.0)}
    target_array_info = {"shape": (1, 256, 256), "scale": (8.0, 8.0, 8.0)}

    print("Loading validation data...")
    _, val_loader = get_dataloader(
        datasplit_path="datasplit.csv",
        classes=classes,
        batch_size=args.batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms={},
        iterations_per_epoch=10,  # doesn't matter for val
        random_validation=True,
        device="cpu",
        weighted_sampler=False,
        num_workers=args.num_workers,
    )
    print(f"Val loader: {len(val_loader.loader)} batches")

    # Discover batch keys
    input_key = list(val_loader.dataset.input_arrays.keys())[0]
    target_key = list(val_loader.dataset.target_arrays.keys())[0]
    print(f"Batch keys: input='{input_key}', target='{target_key}'")

    # Evaluate each experiment
    all_results = {}

    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        ckpt_path = exp_dir / "checkpoints" / "best.pth"
        config_path = exp_dir / "config.json"

        print(f"\n{'='*60}")
        print(f"  Evaluating: {exp_name}")
        print(f"{'='*60}")

        # Load config to get model name
        with open(config_path) as f:
            config = json.load(f)

        model_name = config.get("model", "resnet_2d")
        model_kwargs = config.get("model_kwargs", {})

        # Build model
        model = build_model(
            model_name,
            num_classes=num_classes,
            in_channels=1,
            **model_kwargs,
        )

        # Load best checkpoint
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        # Handle both raw state_dict and wrapped checkpoint formats
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        model = model.to(device)
        print(f"  Loaded {ckpt_path.name} ({model_name})")

        # Evaluate
        results = compute_perclass_metrics(
            model=model,
            val_loader=val_loader,
            classes=classes,
            device=device,
            input_key=input_key,
            target_key=target_key,
            max_batches=args.max_batches,
            use_amp=True,
        )

        results["model"] = model_name
        results["experiment"] = exp_name
        results["checkpoint"] = str(ckpt_path)
        all_results[exp_name] = results

        # Print summary for this experiment
        print(f"\n  Mean Dice: {results['mean_dice']:.4f}")
        print(f"  Mean IoU:  {results['mean_iou']:.4f}")
        print(f"  Classes evaluated: {results['n_classes_evaluated']}/{num_classes}")

        # Show top 5 and bottom 5 by Dice
        pc = results["per_class"]
        ranked = sorted(
            [(c, v["dice"]) for c, v in pc.items() if v["annotated_voxels"] > 0 and (v["tp"] + v["fn"]) > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        if ranked:
            print(f"\n  Top 5:")
            for c, d in ranked[:5]:
                print(f"    {c:20s}: Dice={d:.4f}")
            print(f"  Bottom 5:")
            for c, d in ranked[-5:]:
                print(f"    {c:20s}: Dice={d:.4f}")

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        # Save intermediate results (in case of crash)
        output_path = Path(args.output) if args.output else run_dir / "eval_2d_perclass.json"
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # === Final summary CSV ===
    csv_path = output_path.with_suffix(".csv")
    print(f"\n\n{'='*80}")
    print(f"  FINAL SUMMARY — {len(all_results)} experiments")
    print(f"{'='*80}")

    # Header
    header = ["experiment", "mean_dice", "mean_iou", "n_classes"]
    # Add per-class dice columns for key classes
    key_classes = [c for c in classes if any(
        all_results[list(all_results.keys())[0]]["per_class"][c]["annotated_voxels"] > 0
        for _ in [1]
    )] if all_results else classes

    with open(csv_path, "w") as f:
        # Write header
        cols = ["experiment", "mean_dice", "mean_iou"]
        for c in classes:
            cols.append(f"dice_{c}")
        f.write(",".join(cols) + "\n")

        # Write rows sorted by mean_dice
        for exp_name in sorted(all_results, key=lambda x: all_results[x]["mean_dice"], reverse=True):
            r = all_results[exp_name]
            row = [exp_name, f"{r['mean_dice']:.4f}", f"{r['mean_iou']:.4f}"]
            for c in classes:
                d = r["per_class"][c]["dice"]
                row.append(f"{d:.4f}")
            f.write(",".join(row) + "\n")

    print(f"\nResults saved to:")
    print(f"  JSON: {output_path}")
    print(f"  CSV:  {csv_path}")

    # Print leaderboard
    print(f"\n{'Experiment':<45s} {'Mean Dice':>10s} {'Mean IoU':>10s}")
    print("-" * 65)
    for exp_name in sorted(all_results, key=lambda x: all_results[x]["mean_dice"], reverse=True):
        r = all_results[exp_name]
        print(f"{exp_name:<45s} {r['mean_dice']:>10.4f} {r['mean_iou']:>10.4f}")


if __name__ == "__main__":
    main()
