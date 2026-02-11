#!/usr/bin/env python3
"""
Run MONAI Auto3DSeg on CellMap FIB-SEM segmentation data.

This script provides three modes:
  1. analyze  - Run only the DataAnalyzer to get dataset statistics
  2. generate - Run DataAnalyzer + BundleGen to create algorithm bundles
  3. full     - Run the complete Auto3DSeg pipeline (analyze → generate → train → ensemble)

The output includes:
  - datastats.yaml: Dataset statistics report
  - Algorithm bundle folders with optimized configs
  - Trained model checkpoints (in full mode)
  - Ensemble predictions (in full mode)

Usage:
    # Quick analysis (10-30 min, 1 GPU)
    python run_auto3dseg.py --mode analyze --datalist ./nifti_data/datalist.json

    # Generate algorithm bundles without training
    python run_auto3dseg.py --mode generate --datalist ./nifti_data/datalist.json

    # Full pipeline (multi-day, multi-GPU recommended)
    python run_auto3dseg.py --mode full --datalist ./nifti_data/datalist.json \
        --num_epochs 100 --algos segresnet swinunetr
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path


def check_monai_auto3dseg():
    """Verify MONAI Auto3DSeg is available."""
    try:
        import monai
        from monai.apps.auto3dseg import AutoRunner, DataAnalyzer, BundleGen

        print(f"MONAI version: {monai.__version__}")
        return True
    except ImportError as e:
        print(f"ERROR: MONAI Auto3DSeg not available: {e}")
        print("Install with: pip install 'monai[all]'")
        print("Or minimum: pip install monai nibabel fire")
        return False


def run_data_analysis(
    datalist: str,
    dataroot: str,
    work_dir: str,
    device: str = "cuda",
    worker: int = 4,
    do_ccp: bool = False,
) -> str:
    """
    Run MONAI DataAnalyzer on the CellMap dataset.

    This analyzes:
    - Image intensity distributions (per-channel stats)
    - Image shapes and spacing
    - Label class distributions (voxel counts per class)
    - Foreground/background ratios
    - Connected component statistics (optional)

    Returns path to datastats.yaml
    """
    from monai.apps.auto3dseg import DataAnalyzer

    os.makedirs(work_dir, exist_ok=True)
    output_path = os.path.join(work_dir, "datastats.yaml")

    print("=" * 70)
    print("MONAI Auto3DSeg - Data Analysis")
    print("=" * 70)
    print(f"  Datalist:  {datalist}")
    print(f"  Data root: {dataroot}")
    print(f"  Output:    {output_path}")
    print(f"  Device:    {device}")
    print(f"  Workers:   {worker}")
    print(f"  CCP:       {do_ccp}")
    print("=" * 70)

    analyzer = DataAnalyzer(
        datalist=datalist,
        dataroot=dataroot,
        output_path=output_path,
        average=True,
        do_ccp=do_ccp,
        device=device,
        worker=worker,
        image_key="image",
        label_key="label",
    )

    result = analyzer.get_all_case_stats()

    print("\n" + "=" * 70)
    print("DATA ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {output_path}")

    # Print summary
    if os.path.exists(output_path):
        print_datastats_summary(output_path)

    return output_path


def print_datastats_summary(datastats_path: str):
    """Print a human-readable summary of the data analysis results."""
    with open(datastats_path, "r") as f:
        stats = yaml.safe_load(f)

    if not stats or "stats_summary" not in stats:
        print("  (Could not parse stats summary)")
        return

    summary = stats["stats_summary"]

    print("\n--- Image Statistics ---")
    if "image_stats" in summary:
        img = summary["image_stats"]
        if "shape" in img:
            print(f"  Shape range: {img['shape'].get('min')} → {img['shape'].get('max')}")
            print(f"  Mean shape:  {img['shape'].get('mean')}")
        if "spacing" in img:
            print(f"  Spacing:     {img['spacing'].get('mean')}")
        if "intensity" in img:
            inten = img["intensity"]
            print(f"  Intensity:   [{inten.get('min'):.1f}, {inten.get('max'):.1f}]")
            print(f"               mean={inten.get('mean'):.1f}, std={inten.get('stdev'):.1f}")

    print("\n--- Foreground Statistics ---")
    if "image_foreground_stats" in summary:
        fg = summary["image_foreground_stats"]
        if "intensity" in fg:
            inten = fg["intensity"]
            print(f"  FG Intensity: [{inten.get('min'):.1f}, {inten.get('max'):.1f}]")
            print(f"                mean={inten.get('mean'):.1f}, std={inten.get('stdev'):.1f}")

    print("\n--- Label Statistics ---")
    if "label_stats" in summary:
        label = summary["label_stats"]
        if isinstance(label, dict) and "label" in label:
            for i, lbl in enumerate(label["label"]):
                if isinstance(lbl, dict):
                    pct = lbl.get("percentage", "N/A")
                    print(f"  Class {i}: {pct}% of volume")

    # Print per-case info
    if "stats_by_cases" in stats:
        n_cases = len(stats["stats_by_cases"])
        print(f"\n--- Per-case Stats: {n_cases} cases analyzed ---")


def run_bundle_generation(
    datalist: str,
    dataroot: str,
    work_dir: str,
    datastats_path: str,
    algos: list[str] | None = None,
    num_fold: int = 5,
    gpu_customization: bool = False,
):
    """
    Run BundleGen to create algorithm bundles based on data statistics.

    This creates self-contained algorithm folders (MONAI bundles) with:
    - Training configs optimized for the dataset
    - Network architecture configs
    - Training/inference scripts
    """
    from monai.apps.auto3dseg import BundleGen

    print("\n" + "=" * 70)
    print("MONAI Auto3DSeg - Algorithm Bundle Generation")
    print("=" * 70)

    # Create data source config
    data_src_cfg = os.path.join(work_dir, "data_src_cfg.yaml")

    # Load datalist to check modality
    with open(datalist, "r") as f:
        dl = json.load(f)

    # MONAI algorithm templates only accept "CT" or "MRI" as modality.
    # EM (electron microscopy) is single-channel grayscale like CT, so we use "CT".
    data_src = {
        "modality": "CT",
        "datalist": os.path.abspath(datalist),
        "dataroot": os.path.abspath(dataroot) if dataroot else "",
    }

    # Add class_names if available
    if "class_names" in dl:
        data_src["class_names"] = dl["class_names"]

    with open(data_src_cfg, "w") as f:
        yaml.dump(data_src, f)

    bundle_gen = BundleGen(
        algo_path=work_dir,
        algos=algos,
        data_stats_filename=datastats_path,
        data_src_cfg_name=data_src_cfg,
    )

    bundle_gen.generate(
        output_folder=work_dir,
        num_fold=num_fold,
        gpu_customization=gpu_customization,
    )

    history = bundle_gen.get_history()
    print(f"\nGenerated {len(history)} algorithm bundles:")
    for h in history:
        algo_name = h.get("name", "unknown")
        print(f"  - {algo_name}")

    return history


def run_full_pipeline(
    datalist: str,
    dataroot: str,
    work_dir: str,
    algos: list[str] | None = None,
    num_fold: int = 5,
    num_epochs: int = 100,
    num_gpus: int = 1,
    gpu_customization: bool = True,
    ensemble: bool = True,
    train_params: dict | None = None,
):
    """
    Run the complete Auto3DSeg pipeline using AutoRunner.

    Steps:
    1. Data analysis → datastats.yaml
    2. Algorithm generation → bundle folders
    3. Training → checkpoints
    4. Ensemble → predictions
    """
    from monai.apps.auto3dseg import AutoRunner

    print("\n" + "=" * 70)
    print("MONAI Auto3DSeg - Full Pipeline")
    print("=" * 70)

    # Create input config
    # Load datalist to extract class names
    with open(datalist, "r") as f:
        dl = json.load(f)

    input_cfg = {
        "modality": "CT",  # EM is single-channel grayscale like CT; MONAI only accepts CT/MRI
        "datalist": os.path.abspath(datalist),
        "dataroot": os.path.abspath(dataroot) if dataroot else "",
    }

    if "class_names" in dl:
        input_cfg["class_names"] = dl["class_names"]

    input_cfg_path = os.path.join(work_dir, "input.yaml")
    os.makedirs(work_dir, exist_ok=True)
    with open(input_cfg_path, "w") as f:
        yaml.dump(input_cfg, f)

    # Create and configure AutoRunner
    runner = AutoRunner(
        work_dir=work_dir,
        input=input_cfg_path,
        algos=algos,
        ensemble=ensemble,
    )

    # Set number of folds
    runner.set_num_fold(num_fold)

    # Set training parameters
    # If num_epochs <= 0, let each algorithm use its auto-computed epoch count
    default_train_params = {
        "num_epochs_per_validation": 5,
        "num_images_per_batch": 2,
        "num_warmup_epochs": 5,
    }
    if num_epochs > 0:
        default_train_params["num_epochs"] = num_epochs
    if train_params:
        default_train_params.update(train_params)

    runner.set_training_params(params=default_train_params)

    # Configure GPU
    if num_gpus > 1:
        gpu_ids = ",".join(str(i) for i in range(num_gpus))
        runner.set_device_info(
            cuda_visible_devices=gpu_ids,
            num_nodes=1,
        )

    # Enable GPU customization for memory optimization
    if gpu_customization:
        runner.set_gpu_customization(
            gpu_customization=True,
            gpu_customization_specs={
                "universal": {
                    "num_trials": 3,
                    "range_num_images_per_batch": [1, 4],
                    "range_num_sw_batch_size": [1, 8],
                },
            },
        )

    # Run the full pipeline
    print("\nStarting Auto3DSeg pipeline...")
    print("  This may take several hours to days depending on dataset size")
    print("  and number of algorithms/folds.\n")

    runner.run()

    print("\n" + "=" * 70)
    print("AUTO3DSEG PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Results in: {work_dir}")
    print("  - datastats.yaml: Dataset analysis report")
    print("  - algorithm_templates/: Generated algorithm templates")
    print("  - <algo>_<fold>/: Trained model bundles")
    if ensemble:
        print("  - ensemble_output/: Ensemble predictions")


def main():
    parser = argparse.ArgumentParser(
        description="Run MONAI Auto3DSeg on CellMap FIB-SEM data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick data analysis only
  python run_auto3dseg.py --mode analyze --datalist nifti_data/datalist.json

  # Generate algorithm bundles (no training)
  python run_auto3dseg.py --mode generate --datalist nifti_data/datalist.json

  # Full pipeline with specific algorithms
  python run_auto3dseg.py --mode full --datalist nifti_data/datalist.json \\
      --algos segresnet swinunetr --num_epochs 100
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["analyze", "generate", "full"],
        default="analyze",
        help="Pipeline mode: analyze, generate, or full",
    )
    parser.add_argument(
        "--datalist",
        type=str,
        required=True,
        help="Path to datalist.json (created by convert_zarr_to_nifti.py)",
    )
    parser.add_argument(
        "--dataroot",
        type=str,
        default="",
        help="Root directory for data files (if datalist uses relative paths)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default="auto3dseg/work_dir",
        help="Working directory for Auto3DSeg output",
    )
    parser.add_argument(
        "--algos",
        type=str,
        nargs="*",
        default=None,
        help="Algorithms to use (e.g., segresnet swinunetr dints). "
             "Default: all available",
    )
    parser.add_argument(
        "--num_fold",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=100,
        help="Number of training epochs (full mode only)",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for data analysis (cuda or cpu)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--do_ccp",
        action="store_true",
        help="Run connected component analysis (slower but more detailed)",
    )
    parser.add_argument(
        "--no_ensemble",
        action="store_true",
        help="Skip ensemble step in full mode",
    )
    parser.add_argument(
        "--no_gpu_customization",
        action="store_true",
        help="Disable automatic GPU memory optimization",
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_monai_auto3dseg():
        sys.exit(1)

    # Validate datalist
    if not os.path.exists(args.datalist):
        print(f"ERROR: Datalist not found: {args.datalist}")
        print("Run convert_zarr_to_nifti.py first to create the datalist.")
        sys.exit(1)

    # Run the selected mode
    if args.mode == "analyze":
        run_data_analysis(
            datalist=args.datalist,
            dataroot=args.dataroot,
            work_dir=args.work_dir,
            device=args.device,
            worker=args.workers,
            do_ccp=args.do_ccp,
        )

    elif args.mode == "generate":
        # First analyze, then generate
        datastats_path = os.path.join(args.work_dir, "datastats.yaml")
        if not os.path.exists(datastats_path):
            print("Running data analysis first...")
            datastats_path = run_data_analysis(
                datalist=args.datalist,
                dataroot=args.dataroot,
                work_dir=args.work_dir,
                device=args.device,
                worker=args.workers,
                do_ccp=args.do_ccp,
            )

        run_bundle_generation(
            datalist=args.datalist,
            dataroot=args.dataroot,
            work_dir=args.work_dir,
            datastats_path=datastats_path,
            algos=args.algos,
            num_fold=args.num_fold,
        )

    elif args.mode == "full":
        run_full_pipeline(
            datalist=args.datalist,
            dataroot=args.dataroot,
            work_dir=args.work_dir,
            algos=args.algos,
            num_fold=args.num_fold,
            num_epochs=args.num_epochs,
            num_gpus=args.num_gpus,
            gpu_customization=not args.no_gpu_customization,
            ensemble=not args.no_ensemble,
        )


if __name__ == "__main__":
    main()
