"""
Per-class validation evaluation for ALL ablation runs (2D and 3D).

Loads each best.pth checkpoint, reads config.json to determine model type
and input shape, runs inference on the validation set, and computes per-class
Dice and IoU scores. Results are saved to JSON and summary CSVs.

Usage:
    # Evaluate all experiments (2D and 3D)
    python -m training.eval_all_perclass --run_dir runs/ablation --max_batches 200

    # Evaluate only 2D experiments
    python -m training.eval_all_perclass --run_dir runs/ablation --filter 2d

    # Evaluate only 3D experiments
    python -m training.eval_all_perclass --run_dir runs/ablation --filter 3d

    # Single experiment
    python -m training.eval_all_perclass --run_dir runs/ablation --experiment loss_2d_dice_bce
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
    parser = argparse.ArgumentParser(description="Per-class validation evaluation (2D & 3D)")
    parser.add_argument("--run_dir", type=str, default="runs/ablation",
                        help="Base directory containing experiment folders")
    parser.add_argument("--max_batches", type=int, default=200,
                        help="Max validation batches per experiment (0=all)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Evaluate only this experiment")
    parser.add_argument("--filter", type=str, default=None, choices=["2d", "3d"],
                        help="Filter to only 2D or 3D experiments")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: {run_dir}/eval_all_perclass.json)")
    parser.add_argument("--eval_classes", type=str, default=None,
                        help="Comma-separated list of classes to include in mean Dice. "
                             "If not set, all classes with annotations are included.")
    parser.add_argument("--datasplit_path", type=str, default="datasplit.csv",
                        help="Path to datasplit CSV for loading val data")
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
    ndim: int = 2,
    eval_classes: set[str] | None = None,
) -> dict:
    """Compute per-class Dice and IoU on the validation set.
    
    Args:
        eval_classes: If provided, only these classes are included in mean Dice/IoU.
                      Per-class stats are still computed for all classes.
    """
    model.eval()
    num_classes = len(classes)

    tp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fp = torch.zeros(num_classes, dtype=torch.float64, device=device)
    fn = torch.zeros(num_classes, dtype=torch.float64, device=device)
    annotated_voxels = torch.zeros(num_classes, dtype=torch.float64, device=device)

    # Sum dimensions: batch + spatial dims (skip channel dim 1)
    if ndim == 3:
        sum_dims = (0, 2, 3, 4)  # (B, C, D, H, W)
    else:
        sum_dims = (0, 2, 3)     # (B, C, H, W)

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

            preds = (torch.sigmoid(outputs.float()) > 0.5)
            not_nan = ~targets.isnan()
            targets_clean = targets.nan_to_num(0.0)
            targets_bool = (targets_clean > 0.5)

            p = preds & not_nan
            t = targets_bool & not_nan

            batch_tp = (p & t).sum(dim=sum_dims).to(torch.float64)
            batch_fp = (p & ~t).sum(dim=sum_dims).to(torch.float64)
            batch_fn = (~p & t).sum(dim=sum_dims).to(torch.float64)

            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            annotated_voxels += not_nan.sum(dim=sum_dims).to(torch.float64)

            n_batches += 1
            if max_batches > 0 and n_batches >= max_batches:
                break

            if n_batches % 50 == 0:
                elapsed = time.time() - t0
                print(f"    Batch {n_batches} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"    Done: {n_batches} batches in {elapsed:.1f}s")

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

        # Only count class toward mean if it has annotations AND
        # it's in the eval_classes filter (if specified)
        if ann_np[i] > 0 and (tp_np[i] + fn_np[i]) > 0:
            if eval_classes is None or cls in eval_classes:
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


def detect_dimensionality(config: dict) -> tuple[int, dict, int]:
    """
    From a config.json, determine ndim, array_info, and batch_size.
    Returns (ndim, array_info_dict, batch_size).
    """
    input_shape = config.get("input_shape", [1, 256, 256])
    input_scale = config.get("input_scale", [8.0, 8.0, 8.0])
    target_shape = config.get("target_shape", input_shape)
    target_scale = config.get("target_scale", input_scale)

    if len(input_shape) == 3 and input_shape[0] > 1:
        ndim = 3
        batch_size = 1  # 3D needs small batch
    else:
        ndim = 2
        batch_size = 8

    input_array_info = {"shape": tuple(input_shape), "scale": tuple(input_scale)}
    target_array_info = {"shape": tuple(target_shape), "scale": tuple(target_scale)}

    return ndim, input_array_info, target_array_info, batch_size


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)

    # Find experiments
    if args.experiment:
        exp_dirs = [run_dir / args.experiment]
    else:
        exp_dirs = sorted([
            d for d in run_dir.iterdir()
            if d.is_dir()
            and d.name != "logs"
            and (d / "checkpoints" / "best.pth").exists()
            and (d / "config.json").exists()
        ])

    # Apply filter by reading config to detect dimensionality
    if args.filter:
        filtered_dirs = []
        for d in exp_dirs:
            with open(d / "config.json") as f:
                cfg = json.load(f)
            shape = cfg.get("input_shape", [1, 256, 256])
            is_3d = len(shape) == 3 and shape[0] > 1
            if (args.filter == "3d" and is_3d) or (args.filter == "2d" and not is_3d):
                filtered_dirs.append(d)
        exp_dirs = filtered_dirs

    print(f"Found {len(exp_dirs)} experiments to evaluate")
    for d in exp_dirs:
        print(f"  {d.name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    classes = get_tested_classes()
    num_classes = len(classes)
    print(f"Model classes: {num_classes}")

    # Parse eval_classes filter
    eval_classes = None
    if args.eval_classes:
        eval_classes = set(args.eval_classes.split(","))
        print(f"Eval classes filter: {len(eval_classes)} classes")
        for c in sorted(eval_classes):
            print(f"  {c}")
    else:
        print("Eval classes: all with annotations")

    # Group experiments by dimensionality to reuse data loaders
    exp_groups = {"2d": [], "3d": []}
    for exp_dir in exp_dirs:
        with open(exp_dir / "config.json") as f:
            config = json.load(f)
        ndim, input_info, target_info, batch_size = detect_dimensionality(config)
        key = f"{ndim}d"
        exp_groups[key].append((exp_dir, config, ndim, input_info, target_info, batch_size))

    # Load results if they already exist (for incremental evaluation)
    output_path = Path(args.output) if args.output else run_dir / "eval_all_perclass.json"
    all_results = {}
    if output_path.exists():
        with open(output_path) as f:
            all_results = json.load(f)
        print(f"\nLoaded {len(all_results)} existing results from {output_path}")

    # Process each group
    for group_key in ["2d", "3d"]:
        group = exp_groups[group_key]
        if not group:
            continue

        print(f"\n{'='*70}")
        print(f"  EVALUATING {len(group)} {group_key.upper()} EXPERIMENTS")
        print(f"{'='*70}")

        # Use first experiment's shape info for the data loader
        _, _, ndim, input_info, target_info, batch_size = group[0]

        print(f"\n  Loading {group_key.upper()} validation data...")
        print(f"  Input shape: {input_info['shape']}, scale: {input_info['scale']}")

        _, val_loader = get_dataloader(
            datasplit_path=args.datasplit_path,
            classes=classes,
            batch_size=batch_size,
            input_array_info=input_info,
            target_array_info=target_info,
            spatial_transforms={},
            iterations_per_epoch=10,
            random_validation=True,
            device="cpu",
            weighted_sampler=False,
            num_workers=args.num_workers,
        )
        print(f"  Val loader: {len(val_loader.loader)} batches")

        input_key = list(val_loader.dataset.input_arrays.keys())[0]
        target_key = list(val_loader.dataset.target_arrays.keys())[0]
        print(f"  Batch keys: input='{input_key}', target='{target_key}'")

        for exp_dir, config, exp_ndim, _, _, _ in group:
            exp_name = exp_dir.name
            ckpt_path = exp_dir / "checkpoints" / "best.pth"

            print(f"\n  {'─'*60}")
            print(f"  Evaluating: {exp_name}")
            print(f"  {'─'*60}")

            model_name = config.get("model", "resnet_2d")
            model_kwargs = config.get("model_kwargs", {})

            model = build_model(
                model_name,
                num_classes=num_classes,
                in_channels=1,
                **model_kwargs,
            )

            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict)
            model = model.to(device)
            print(f"    Loaded {ckpt_path.name} ({model_name})")

            results = compute_perclass_metrics(
                model=model,
                val_loader=val_loader,
                classes=classes,
                device=device,
                input_key=input_key,
                target_key=target_key,
                max_batches=args.max_batches,
                use_amp=True,
                ndim=ndim,
                eval_classes=eval_classes,
            )

            results["model"] = model_name
            results["experiment"] = exp_name
            results["checkpoint"] = str(ckpt_path)
            results["ndim"] = ndim
            all_results[exp_name] = results

            print(f"\n    Mean Dice: {results['mean_dice']:.4f}")
            print(f"    Mean IoU:  {results['mean_iou']:.4f}")
            print(f"    Classes evaluated: {results['n_classes_evaluated']}/{num_classes}")

            # Top and bottom 5
            pc = results["per_class"]
            ranked = sorted(
                [(c, v["dice"]) for c, v in pc.items()
                 if v["annotated_voxels"] > 0 and (v["tp"] + v["fn"]) > 0],
                key=lambda x: x[1], reverse=True,
            )
            if ranked:
                print(f"\n    Top 5:")
                for c, d in ranked[:5]:
                    print(f"      {c:20s}: Dice={d:.4f}")
                print(f"    Bottom 5:")
                for c, d in ranked[-5:]:
                    print(f"      {c:20s}: Dice={d:.4f}")

            del model
            torch.cuda.empty_cache()

            # Save intermediate
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2)

        # Clean up data loader
        del val_loader
        torch.cuda.empty_cache()

    # === Write summary CSVs ===
    _write_summary(all_results, classes, output_path)


def _write_summary(all_results: dict, classes: list[str], output_path: Path):
    """Write leaderboard CSVs (separate for 2D and 3D, plus combined)."""

    for suffix, filter_fn in [
        ("_2d", lambda r: r.get("ndim", 2) == 2),
        ("_3d", lambda r: r.get("ndim", 2) == 3),
        ("", lambda r: True),
    ]:
        filtered = {k: v for k, v in all_results.items() if filter_fn(v)}
        if not filtered:
            continue

        csv_path = output_path.with_name(f"eval_all_perclass{suffix}.csv")
        with open(csv_path, "w") as f:
            cols = ["experiment", "ndim", "model", "mean_dice", "mean_iou", "n_classes"]
            for c in classes:
                cols.append(f"dice_{c}")
            f.write(",".join(cols) + "\n")

            for exp_name in sorted(filtered, key=lambda x: filtered[x]["mean_dice"], reverse=True):
                r = filtered[exp_name]
                row = [
                    exp_name,
                    str(r.get("ndim", 2)),
                    r.get("model", "unknown"),
                    f"{r['mean_dice']:.4f}",
                    f"{r['mean_iou']:.4f}",
                    str(r.get("n_classes_evaluated", 0)),
                ]
                for c in classes:
                    d = r["per_class"][c]["dice"]
                    row.append(f"{d:.4f}")
                f.write(",".join(row) + "\n")

        print(f"\nSaved: {csv_path}")

    # Print leaderboards
    for label, filter_fn in [("2D", lambda r: r.get("ndim", 2) == 2),
                              ("3D", lambda r: r.get("ndim", 2) == 3)]:
        filtered = {k: v for k, v in all_results.items() if filter_fn(v)}
        if not filtered:
            continue

        print(f"\n{'='*75}")
        print(f"  {label} LEADERBOARD ({len(filtered)} experiments)")
        print(f"{'='*75}")
        print(f"  {'Rank':<5s} {'Experiment':<40s} {'Mean Dice':>10s} {'Mean IoU':>10s}")
        print(f"  {'-'*65}")
        for rank, exp_name in enumerate(
            sorted(filtered, key=lambda x: filtered[x]["mean_dice"], reverse=True), 1
        ):
            r = filtered[exp_name]
            marker = " ★" if rank == 1 else ""
            print(f"  {rank:<5d} {exp_name:<40s} {r['mean_dice']:>10.4f} {r['mean_iou']:>10.4f}{marker}")


if __name__ == "__main__":
    main()
