#!/usr/bin/env python3
"""
Interpret Auto3DSeg Results for CellMap Training Optimization.

After running Auto3DSeg data analysis (datastats.yaml), this script reads the
results and provides actionable recommendations for optimizing your existing
CellMap training pipeline.

Usage:
    python interpret_results.py --datastats auto3dseg/work_dir/datastats.yaml
"""

import argparse
import json
import os
import sys

try:
    import yaml
except ImportError:
    print("Install PyYAML: pip install pyyaml")
    sys.exit(1)


def load_datastats(path: str) -> dict:
    """Load the datastats.yaml file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def analyze_shapes(stats: dict) -> dict:
    """Analyze shape statistics and recommend patch sizes."""
    summary = stats.get("stats_summary", {})
    image_stats = summary.get("image_stats", {})
    shape_info = image_stats.get("shape", {})

    recommendations = {}

    if shape_info:
        min_shape = shape_info.get("min", [0, 0, 0])
        max_shape = shape_info.get("max", [0, 0, 0])
        mean_shape = shape_info.get("mean", [0, 0, 0])

        # Recommend patch size based on shape distribution
        # Rule: patch should be <= min shape, and a power of 2
        patch_candidates = [32, 64, 96, 128, 160, 192, 224, 256]

        if isinstance(min_shape, list) and len(min_shape) >= 3:
            min_dim = min(min_shape[:3]) if all(isinstance(s, (int, float)) for s in min_shape[:3]) else 128
            # Find largest patch that fits
            best_patch = 64
            for p in patch_candidates:
                if p <= min_dim:
                    best_patch = p

            recommendations["patch_size"] = best_patch
            recommendations["min_crop_shape"] = min_shape[:3]
            recommendations["max_crop_shape"] = max_shape[:3]
            recommendations["mean_crop_shape"] = [round(s, 1) if isinstance(s, float) else s for s in mean_shape[:3]]

    return recommendations


def analyze_intensity(stats: dict) -> dict:
    """Analyze intensity statistics for normalization strategy."""
    summary = stats.get("stats_summary", {})
    image_stats = summary.get("image_stats", {})
    fg_stats = summary.get("image_foreground_stats", {})

    recommendations = {}

    # Overall intensity
    intensity = image_stats.get("intensity", {})
    if intensity:
        recommendations["intensity_range"] = [
            intensity.get("min", 0),
            intensity.get("max", 255),
        ]
        recommendations["intensity_mean"] = round(intensity.get("mean", 127), 2)
        recommendations["intensity_std"] = round(intensity.get("stdev", 50), 2)

        # Recommend normalization
        int_min = intensity.get("min", 0)
        int_max = intensity.get("max", 255)
        p005 = intensity.get("percentile_00_5", int_min)
        p995 = intensity.get("percentile_99_5", int_max)

        recommendations["recommended_normalization"] = {
            "method": "percentile_clip_and_scale",
            "clip_min": round(p005, 1) if isinstance(p005, float) else p005,
            "clip_max": round(p995, 1) if isinstance(p995, float) else p995,
            "note": "Clip to [0.5th, 99.5th percentile], then scale to [0, 1]",
        }

    # Foreground intensity
    fg_intensity = fg_stats.get("intensity", {})
    if fg_intensity:
        recommendations["foreground_intensity_mean"] = round(fg_intensity.get("mean", 0), 2)
        recommendations["foreground_intensity_std"] = round(fg_intensity.get("stdev", 0), 2)

    return recommendations


def analyze_labels(stats: dict) -> dict:
    """Analyze label statistics for class balancing."""
    summary = stats.get("stats_summary", {})
    label_stats = summary.get("label_stats", {})

    recommendations = {}

    if isinstance(label_stats, dict) and "label" in label_stats:
        labels = label_stats["label"]
        if isinstance(labels, list):
            class_info = []
            total_fg = 0
            for i, lbl in enumerate(labels):
                if isinstance(lbl, dict):
                    pct = lbl.get("percentage", 0)
                    if isinstance(pct, (int, float)):
                        class_info.append({"class_id": i, "percentage": round(pct, 4)})
                        if i > 0:  # skip background
                            total_fg += pct

            if class_info:
                recommendations["class_distribution"] = class_info
                recommendations["total_foreground_pct"] = round(total_fg, 4)
                recommendations["background_pct"] = round(100 - total_fg, 4)

                # Identify class imbalance
                fg_classes = [c for c in class_info if c["class_id"] > 0 and c["percentage"] > 0]
                if fg_classes:
                    min_pct = min(c["percentage"] for c in fg_classes)
                    max_pct = max(c["percentage"] for c in fg_classes)
                    imbalance_ratio = max_pct / min_pct if min_pct > 0 else float("inf")
                    recommendations["class_imbalance_ratio"] = round(imbalance_ratio, 1)

                    if imbalance_ratio > 10:
                        recommendations["loss_recommendation"] = (
                            "SEVERE class imbalance detected! Use weighted loss. "
                            "Consider: DiceFocalLoss, or class-weighted CrossEntropyLoss. "
                            "Current CellMapLossWrapper may need class weights."
                        )
                    elif imbalance_ratio > 3:
                        recommendations["loss_recommendation"] = (
                            "Moderate class imbalance. Consider: DiceCELoss with "
                            "class weights, or focal loss. Weighted sampling may also help."
                        )
                    else:
                        recommendations["loss_recommendation"] = (
                            "Class distribution is relatively balanced. Standard "
                            "DiceCELoss should work well."
                        )

    return recommendations


def analyze_spacing(stats: dict) -> dict:
    """Analyze spacing statistics."""
    summary = stats.get("stats_summary", {})
    image_stats = summary.get("image_stats", {})
    spacing = image_stats.get("spacing", {})

    recommendations = {}

    if spacing:
        mean_spacing = spacing.get("mean", [8, 8, 8])
        recommendations["mean_spacing"] = mean_spacing

        if isinstance(mean_spacing, list) and len(mean_spacing) >= 3:
            # Check if spacing is anisotropic
            spacings = [s for s in mean_spacing[:3] if isinstance(s, (int, float))]
            if spacings:
                ratio = max(spacings) / min(spacings) if min(spacings) > 0 else 1
                recommendations["anisotropy_ratio"] = round(ratio, 2)

                if ratio > 2:
                    recommendations["anisotropy_note"] = (
                        "Data is anisotropic. Consider using anisotropic patch sizes "
                        "(e.g., smaller in the coarser dimension) or resampling to "
                        "isotropic spacing."
                    )
                else:
                    recommendations["anisotropy_note"] = (
                        "Data is approximately isotropic. Cubic patches are appropriate."
                    )

    return recommendations


def generate_training_config(shape_rec, intensity_rec, label_rec, spacing_rec) -> dict:
    """Generate recommended training configuration for CellMap pipeline."""
    config = {
        "# RECOMMENDED TRAINING CONFIGURATION": "Based on Auto3DSeg data analysis",
    }

    # Patch size
    patch = shape_rec.get("patch_size", 128)
    config["input_array_info"] = {
        "shape": [patch, patch, patch],
        "scale": spacing_rec.get("mean_spacing", [8, 8, 8])[:3],
    }

    # Batch size suggestion based on patch size
    if patch <= 64:
        config["batch_size"] = 8
    elif patch <= 128:
        config["batch_size"] = 4
    elif patch <= 192:
        config["batch_size"] = 2
    else:
        config["batch_size"] = 1

    # Learning rate
    config["learning_rate"] = 0.0002
    config["optimizer"] = "AdamW"
    config["scheduler"] = "OneCycleLR (recommended by Auto3DSeg)"

    # Loss function
    if label_rec.get("class_imbalance_ratio", 1) > 10:
        config["loss"] = "DiceFocalLoss (severe imbalance)"
    elif label_rec.get("class_imbalance_ratio", 1) > 3:
        config["loss"] = "DiceCELoss with class weights"
    else:
        config["loss"] = "DiceCELoss"

    # Data augmentation
    config["augmentations"] = [
        "RandomFlip(axes=(0,1,2))",
        "RandomRotate90(axes=(1,2))",
        "RandScaleIntensity(factors=0.1, prob=0.5)",
        "RandShiftIntensity(offsets=0.1, prob=0.5)",
        "RandGaussianNoise(std=0.05, prob=0.3)",
    ]

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Interpret Auto3DSeg results for CellMap optimization"
    )
    parser.add_argument(
        "--datastats",
        type=str,
        default="auto3dseg/work_dir/datastats.yaml",
        help="Path to datastats.yaml from Auto3DSeg analysis",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save recommendations to a YAML file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.datastats):
        print(f"ERROR: {args.datastats} not found!")
        print("Run Auto3DSeg data analysis first.")
        sys.exit(1)

    print("=" * 70)
    print("Auto3DSeg Results Interpreter for CellMap")
    print("=" * 70)

    stats = load_datastats(args.datastats)

    # Analyze each aspect
    shape_rec = analyze_shapes(stats)
    intensity_rec = analyze_intensity(stats)
    label_rec = analyze_labels(stats)
    spacing_rec = analyze_spacing(stats)

    # Print recommendations
    print("\n📐 SHAPE ANALYSIS")
    print("-" * 40)
    if shape_rec:
        print(f"  Min crop shape:  {shape_rec.get('min_crop_shape', 'N/A')}")
        print(f"  Max crop shape:  {shape_rec.get('max_crop_shape', 'N/A')}")
        print(f"  Mean crop shape: {shape_rec.get('mean_crop_shape', 'N/A')}")
        print(f"  ✅ Recommended patch size: {shape_rec.get('patch_size', 128)}³")

    print("\n🔬 INTENSITY ANALYSIS")
    print("-" * 40)
    if intensity_rec:
        print(f"  Range:     [{intensity_rec.get('intensity_range', ['?', '?'])[0]}, "
              f"{intensity_rec.get('intensity_range', ['?', '?'])[1]}]")
        print(f"  Mean±Std:  {intensity_rec.get('intensity_mean', '?')} ± "
              f"{intensity_rec.get('intensity_std', '?')}")
        print(f"  FG Mean:   {intensity_rec.get('foreground_intensity_mean', '?')}")
        norm = intensity_rec.get("recommended_normalization", {})
        if norm:
            print(f"  ✅ Normalization: clip to [{norm.get('clip_min', '?')}, "
                  f"{norm.get('clip_max', '?')}] then scale to [0, 1]")

    print("\n🏷️  LABEL ANALYSIS")
    print("-" * 40)
    if label_rec:
        print(f"  Background:  {label_rec.get('background_pct', '?')}%")
        print(f"  Foreground:  {label_rec.get('total_foreground_pct', '?')}%")
        print(f"  Imbalance:   {label_rec.get('class_imbalance_ratio', '?')}x")
        if "loss_recommendation" in label_rec:
            print(f"  ✅ {label_rec['loss_recommendation']}")

        if "class_distribution" in label_rec:
            from convert_zarr_to_nifti import BASE_CLASSES
            print(f"\n  Per-class distribution:")
            for c in label_rec["class_distribution"]:
                cid = c["class_id"]
                pct = c["percentage"]
                name = BASE_CLASSES[cid - 1] if 0 < cid <= len(BASE_CLASSES) else f"class_{cid}"
                if cid == 0:
                    name = "background"
                bar = "█" * max(1, int(pct * 2))
                print(f"    {name:>12s} ({cid:2d}): {pct:6.2f}% {bar}")

    print("\n📏 SPACING ANALYSIS")
    print("-" * 40)
    if spacing_rec:
        print(f"  Mean spacing:    {spacing_rec.get('mean_spacing', 'N/A')}")
        print(f"  Anisotropy:      {spacing_rec.get('anisotropy_ratio', '?')}x")
        print(f"  ✅ {spacing_rec.get('anisotropy_note', 'N/A')}")

    # Generate recommended config
    print("\n" + "=" * 70)
    print("📋 RECOMMENDED CELLMAP TRAINING CONFIG")
    print("=" * 70)
    config = generate_training_config(shape_rec, intensity_rec, label_rec, spacing_rec)
    for key, val in config.items():
        if isinstance(val, dict):
            print(f"\n  {key}:")
            for k, v in val.items():
                print(f"    {k}: {v}")
        elif isinstance(val, list):
            print(f"\n  {key}:")
            for item in val:
                print(f"    - {item}")
        else:
            print(f"  {key}: {val}")

    # Save to file
    if args.output:
        all_recommendations = {
            "shape_analysis": shape_rec,
            "intensity_analysis": intensity_rec,
            "label_analysis": label_rec,
            "spacing_analysis": spacing_rec,
            "recommended_config": config,
        }
        with open(args.output, "w") as f:
            yaml.dump(all_recommendations, f, default_flow_style=False)
        print(f"\n\nRecommendations saved to: {args.output}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review the statistics and recommendations above")
    print("2. Apply the recommended patch size and loss function to your")
    print("   existing training configs (e.g., examples/train_3D.py)")
    print("3. If running the full Auto3DSeg pipeline, compare the")
    print("   Auto3DSeg-trained models against your current baselines")
    print("4. Use Auto3DSeg's best architecture/hyperparameters as a")
    print("   starting point for your CellMap pipeline")


if __name__ == "__main__":
    main()
