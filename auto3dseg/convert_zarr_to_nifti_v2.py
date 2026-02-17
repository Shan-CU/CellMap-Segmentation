#!/usr/bin/env python3
"""
Convert CellMap zarr groundtruth + EM to NIfTI for MONAI training (v2).

Major changes from v1:
- 31 atomic classes (up from 14) — all leaf classes that appear in zarr GT
- Group classes (nuc, mito, ves, etc.) are NOT stored in NIfTI — they are
  composed at inference time via union of sub-classes
- All 289 zarr crops processed (v1 missed 12)
- Per-crop annotation audit: records which classes have non-zero voxels
- Parallel conversion via multiprocessing
- Outputs to nifti_data_v2/ (v1 data in nifti_data/ renamed to nifti_data_old/)

Output format:
  nifti_data_v2/images/<dataset>_<crop>_0000.nii.gz   — uint8 EM
  nifti_data_v2/labels/<dataset>_<crop>.nii.gz         — uint8 integer labels (0-31)
  nifti_data_v2/datalist.json                          — MONAI datalist with annotations

Integer encoding:
  0 = background / unannotated
  1..31 = atomic class ID (see ATOMIC_CLASSES below)

Usage:
  python auto3dseg/convert_zarr_to_nifti_v2.py [--workers 16] [--dry-run]

Requires: zarr, nibabel, numpy
"""

from __future__ import annotations

import argparse
import gc
import csv
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr


# ═══════════════════════════════════════════════════════════════════════════
# CLASS DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════

# All 47 tested classes, split into two sets:
#
# A) TRAINABLE classes (29 atomic leaf classes from zarr groundtruth)
#    - These get integer labels 1..29 in the NIfTI
#    - The model predicts these directly
#
# B) GROUP classes (18 composites)
#    - Composed at inference time by union of atomic predictions
#    - NOT stored in NIfTI labels — zero training cost
#
# Why not train on groups directly? Because the zarr stores both the leaf
# and the pre-composed group. Training on both would double-count voxels
# (e.g., mito_mem voxels appear in both mito_mem channel AND mito channel).
# Instead: predict 29 leaves → compose 18 groups → submit all 47.
#
# Exception: "nuc" — the challenge defines nuc = union(ne_mem..nucleo), but
# zarr stores "nuc" as a direct annotation that may differ from the union of
# sub-components (some crops have "nuc" but NOT ne_mem/ne_lum/etc.). So we
# read "nuc" directly from zarr as a trainable class.

ATOMIC_CLASSES = [
    # ── Original 14 from Round 1 ──
    "ecs",          #  1
    "pm",           #  2
    "mito_mem",     #  3
    "mito_lum",     #  4
    "mito_ribo",    #  5
    "golgi_mem",    #  6
    "golgi_lum",    #  7
    "ves_mem",      #  8
    "ves_lum",      #  9
    "endo_mem",     # 10
    "endo_lum",     # 11
    "er_mem",       # 12
    "er_lum",       # 13
    "nuc",          # 14  — read directly from zarr (see note above)
    # ── New for Round 2 ──
    "lyso_mem",     # 15
    "lyso_lum",     # 16
    "ld_mem",       # 17
    "ld_lum",       # 18
    "eres_mem",     # 19
    "eres_lum",     # 20
    "ne_mem",       # 21
    "ne_lum",       # 22
    "np_out",       # 23
    "np_in",        # 24
    "hchrom",       # 25
    "echrom",       # 26
    "nucpl",        # 27
    "mt_out",       # 28
    "cyto",         # 29
    "mt_in",        # 30
    "perox_mem",    # 31
    "perox_lum",    # 32
]

NUM_CLASSES = len(ATOMIC_CLASSES)  # 32
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(ATOMIC_CLASSES)}

# Group classes composed at inference time via union of atomic predictions.
# Maps group_name -> list of ATOMIC class names whose predictions are OR'd.
GROUP_CLASSES = {
    "mito":       ["mito_mem", "mito_lum", "mito_ribo"],
    "golgi":      ["golgi_mem", "golgi_lum"],
    "ves":        ["ves_mem", "ves_lum"],
    "endo":       ["endo_mem", "endo_lum"],
    "lyso":       ["lyso_mem", "lyso_lum"],
    "ld":         ["ld_mem", "ld_lum"],
    "eres":       ["eres_mem", "eres_lum"],
    "perox":      ["perox_mem", "perox_lum"],
    "ne":         ["ne_mem", "ne_lum", "np_out", "np_in"],
    "np":         ["np_out", "np_in"],
    "chrom":      ["hchrom", "echrom"],
    "mt":         ["mt_out", "mt_in"],
    "er":         ["er_mem", "er_lum", "eres_mem", "eres_lum",
                   "ne_mem", "ne_lum", "np_out", "np_in"],
    "er_mem_all": ["er_mem", "eres_mem", "ne_mem"],
    "cell":       ["pm", "mito_mem", "mito_lum", "mito_ribo",
                   "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
                   "endo_mem", "endo_lum", "lyso_mem", "lyso_lum",
                   "ld_mem", "ld_lum", "er_mem", "er_lum",
                   "eres_mem", "eres_lum", "ne_mem", "ne_lum",
                   "np_out", "np_in", "hchrom", "echrom", "nucpl",
                   "mt_out", "cyto", "mt_in", "perox_mem", "perox_lum"],
    # "nuc" is both a tested group AND an atomic zarr class — we train it
    # directly (ATOMIC_CLASSES[13]), so no group composition needed.
}

# 47 tested classes for reference
TESTED_CLASSES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "lyso_mem", "lyso_lum",
    "ld_mem", "ld_lum", "er_mem", "er_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl",
    "mt_out", "cyto", "mt_in",
    "nuc", "golgi", "ves", "endo", "lyso", "ld", "eres",
    "perox_mem", "perox_lum", "perox",
    "mito", "er", "ne", "np", "chrom", "mt",
    "cell", "er_mem_all",
]


# ═══════════════════════════════════════════════════════════════════════════
# CONVERSION LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def get_zarr_array(zarr_path: str, class_name: str, crop_name: str) -> np.ndarray | None:
    """Load a zarr groundtruth array at s0 resolution.

    Tries multiple scale keys (s0, s1, etc.) and picks the highest resolution.
    Returns None if the class directory doesn't exist or has no data.
    """
    gt_path = os.path.join(zarr_path, "recon-1", "labels", "groundtruth", crop_name, class_name)
    if not os.path.isdir(gt_path):
        return None

    # Try s0 first (highest resolution)
    for scale in ["s0", "s1", "s2"]:
        scale_path = os.path.join(gt_path, scale)
        if os.path.isdir(scale_path):
            try:
                arr = zarr.open(scale_path, mode="r")
                data = np.asarray(arr)
                if data.size > 0:
                    return data
            except Exception:
                continue
    return None


def get_em_array(zarr_path: str) -> np.ndarray | None:
    """Load EM (fibsem-uint8) array at s0 resolution."""
    em_base = os.path.join(zarr_path, "recon-1", "em", "fibsem-uint8")
    for scale in ["s0", "s1"]:
        scale_path = os.path.join(em_base, scale)
        if os.path.isdir(scale_path):
            try:
                arr = zarr.open(scale_path, mode="r")
                return np.asarray(arr)
            except Exception:
                continue
    return None


def convert_one_crop(
    dataset_name: str,
    crop_name: str,
    zarr_path: str,
    output_images_dir: str,
    output_labels_dir: str,
    dry_run: bool = False,
) -> dict | None:
    """Convert a single zarr crop to NIfTI.

    Returns a dict with metadata, or None if the crop has no usable data.
    """
    crop_id = f"{dataset_name}_{crop_name}"
    img_out = os.path.join(output_images_dir, f"{crop_id}_0000.nii.gz")
    lbl_out = os.path.join(output_labels_dir, f"{crop_id}.nii.gz")

    # Skip if already converted
    if os.path.exists(img_out) and os.path.exists(lbl_out):
        # Still need to compute annotation info — read label back
        try:
            lbl_nii = nib.load(lbl_out)
            lbl_data = np.asarray(lbl_nii.dataobj)
            annotated_indices = []
            for c in range(NUM_CLASSES):
                if np.any(lbl_data == (c + 1)):
                    annotated_indices.append(c)
            return {
                "crop_id": crop_id,
                "image": img_out,
                "label": lbl_out,
                "annotated_indices": annotated_indices,
                "shape": list(lbl_data.shape),
                "status": "exists",
            }
        except Exception as e:
            # Re-convert if existing file is corrupt
            pass

    if dry_run:
        return {
            "crop_id": crop_id,
            "status": "dry_run",
            "annotated_indices": [],
            "shape": [],
        }

    # Load EM
    em_data = get_em_array(zarr_path)
    if em_data is None:
        print(f"  SKIP {crop_id}: no EM data")
        return None

    shape = em_data.shape  # (Z, Y, X)

    # Build integer label volume
    label_vol = np.zeros(shape, dtype=np.uint8)
    annotated_indices = []
    annotated_names = []

    for cls_name in ATOMIC_CLASSES:
        cls_id = CLASS_TO_ID[cls_name]
        gt_data = get_zarr_array(zarr_path, cls_name, crop_name)

        if gt_data is None:
            continue

        # Resize if shape mismatch (different scales)
        if gt_data.shape != shape:
            from scipy.ndimage import zoom
            zoom_factors = tuple(s / g for s, g in zip(shape, gt_data.shape))
            gt_data = zoom(gt_data.astype(np.float32), zoom_factors, order=0).astype(np.uint8)

        # Check for non-zero content
        mask = gt_data > 0
        if not np.any(mask):
            continue

        # Write into label volume (later classes overwrite earlier — shouldn't
        # overlap much for atomic classes, but if they do, last wins)
        label_vol[mask] = cls_id
        annotated_indices.append(cls_id - 1)  # 0-indexed
        annotated_names.append(cls_name)

    if not annotated_indices:
        print(f"  SKIP {crop_id}: all classes empty")
        return None

    # Save NIfTI — use identity affine (isotropic 1mm, doesn't matter for MONAI)
    affine = np.eye(4)

    em_nii = nib.Nifti1Image(em_data.astype(np.uint8), affine)
    nib.save(em_nii, img_out)

    lbl_nii = nib.Nifti1Image(label_vol, affine)
    nib.save(lbl_nii, lbl_out)

    # Free large arrays before returning
    del em_data, label_vol, em_nii, lbl_nii
    gc.collect()

    size_mb = (os.path.getsize(img_out) + os.path.getsize(lbl_out)) / (1024 * 1024)
    print(f"  OK {crop_id}: shape={shape}, classes={len(annotated_indices)}/{NUM_CLASSES} "
          f"({','.join(annotated_names[:5])}{'...' if len(annotated_names) > 5 else ''}), "
          f"size={size_mb:.1f}MB")

    return {
        "crop_id": crop_id,
        "image": img_out,
        "label": lbl_out,
        "annotated_indices": annotated_indices,
        "annotated_names": annotated_names,
        "shape": list(shape),
        "status": "converted",
    }


def _convert_wrapper(args):
    """Wrapper for multiprocessing — unpacks args and catches exceptions."""
    try:
        return convert_one_crop(*args)
    except Exception as e:
        crop_id = f"{args[0]}_{args[1]}"
        print(f"  ERROR {crop_id}: {e}")
        traceback.print_exc()
        return None


# ═══════════════════════════════════════════════════════════════════════════
# DATALIST GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def build_datalist(results: list[dict], datasplit_path: str, output_path: str):
    """Build datalist.json from conversion results + datasplit.csv for train/val split."""

    # Parse datasplit.csv for train/val assignment
    split_map = {}  # crop_id -> "train" or "validate"
    with open(datasplit_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            split = row[0].strip().strip('"')
            if split not in ("train", "validate"):
                continue
            raw_path = row[1].strip().strip('"')
            label_key = row[4].strip().strip('"')

            ds_match = re.search(r"(jrc_[^/]+)", raw_path)
            crop_match = re.search(r"(crop\d+)", label_key)
            if ds_match and crop_match:
                crop_id = f"{ds_match.group(1)}_{crop_match.group(1)}"
                if crop_id not in split_map:
                    split_map[crop_id] = split

    training = []
    validation = []

    for r in results:
        crop_id = r["crop_id"]
        ann_indices = sorted(r["annotated_indices"])
        ann_str = ",".join(str(i) for i in ann_indices) if ann_indices else ""

        item = {
            "image": r["image"],
            "label": r["label"],
            "annotated_classes": ann_str,
        }

        split = split_map.get(crop_id, "train")
        if split == "validate":
            validation.append(item)
        else:
            training.append(item)

    class_names_cfg = [
        {"name": name, "index": [idx + 1]}
        for idx, name in enumerate(ATOMIC_CLASSES)
    ]

    datalist = {
        "name": "CellMap FIB-SEM Segmentation Challenge (v2 — 32 atomic classes)",
        "description": "3D FIB-SEM volumes with integer labels 0-32 and partial annotation masking. "
                       "Group classes (mito, nuc, ves, etc.) composed at inference.",
        "modality": "CT",
        "sigmoid": True,
        "num_classes": NUM_CLASSES,
        "class_names": class_names_cfg,
        "group_classes": {k: v for k, v in GROUP_CLASSES.items()},
        "tested_classes": TESTED_CLASSES,
        "training": training,
    }
    if validation:
        datalist["validation"] = validation

    with open(output_path, "w") as f:
        json.dump(datalist, f, indent=2)

    print(f"\nDatalist saved: {output_path}")
    print(f"  Training:   {len(training)}")
    print(f"  Validation: {len(validation)}")

    # Per-class coverage summary
    print(f"\n  Per-class annotation coverage ({NUM_CLASSES} classes):")
    all_results = training + validation
    for idx, cls_name in enumerate(ATOMIC_CLASSES):
        n = sum(1 for item in all_results
                if str(idx) in item["annotated_classes"].split(","))
        pct = 100 * n / len(all_results) if all_results else 0
        print(f"    {idx+1:2d} {cls_name:<12}: {n:>3}/{len(all_results)} ({pct:5.1f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Convert CellMap zarr to NIfTI (v2, 32 atomic classes)")
    parser.add_argument("--workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually write files")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    repo_dir = script_dir.parent
    data_dir = Path(args.data_dir) if args.data_dir else repo_dir / "data"
    output_dir = script_dir / "nifti_data_v2"
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    datasplit_path = repo_dir / "datasplit.csv"

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"CellMap Zarr → NIfTI v2 Converter")
    print(f"  Data dir:    {data_dir}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Workers:     {args.workers}")
    print(f"  Atomic classes: {NUM_CLASSES}")
    print(f"  Dry run:     {args.dry_run}")
    print()

    # Discover all zarr crops
    jobs = []
    for ds_entry in sorted(os.listdir(data_dir)):
        ds_path = data_dir / ds_entry
        zarr_dir = ds_path / f"{ds_entry}.zarr"
        gt_dir = zarr_dir / "recon-1" / "labels" / "groundtruth"
        if not gt_dir.is_dir():
            continue
        for crop_entry in sorted(os.listdir(gt_dir)):
            if not crop_entry.startswith("crop"):
                continue
            jobs.append((
                ds_entry,          # dataset_name
                crop_entry,        # crop_name (e.g., "crop234")
                str(zarr_dir),     # zarr_path
                str(images_dir),   # output_images_dir
                str(labels_dir),   # output_labels_dir
                args.dry_run,      # dry_run
            ))

    print(f"Found {len(jobs)} zarr crops to convert\n")

    # Run conversions in parallel
    t0 = time.time()
    results = []

    if args.workers <= 1:
        for job in jobs:
            r = _convert_wrapper(job)
            if r is not None:
                results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_convert_wrapper, job): job for job in jobs}
            for future in as_completed(futures):
                r = future.result()
                if r is not None:
                    results.append(r)

    elapsed = time.time() - t0
    n_converted = sum(1 for r in results if r.get("status") == "converted")
    n_existed = sum(1 for r in results if r.get("status") == "exists")
    n_total = len(results)

    print(f"\n{'='*60}")
    print(f"Conversion complete in {elapsed:.1f}s")
    print(f"  Total crops:   {n_total}")
    print(f"  New converts:  {n_converted}")
    print(f"  Already exist: {n_existed}")
    print(f"  Skipped/error: {len(jobs) - n_total}")

    if not args.dry_run and results:
        # Build datalist
        datalist_path = str(output_dir / "datalist.json")
        build_datalist(results, str(datasplit_path), datalist_path)

    print(f"\nDone!")


if __name__ == "__main__":
    main()
