#!/usr/bin/env python3
"""
Regenerate datalist.json for Auto3DSeg from existing NIfTI files.

This script does NOT re-convert any data. It reads datasplit.csv to determine
the train/validate split and class annotations, checks zarr chunk directories
to determine which classes are truly annotated, and writes a fresh datalist.json.

Usage:
    python auto3dseg/regenerate_datalist.py
"""

import csv
import json
import os
import re
import sys
from pathlib import Path

# The 14 base label classes used in CellMap ground truth
BASE_CLASSES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
]

CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(BASE_CLASSES)}


def has_zarr_chunks(zarr_path: str, array_key: str) -> bool:
    """Check if a zarr array has any actual chunk data on disk."""
    chunk_dir = os.path.join(zarr_path, array_key)
    if not os.path.isdir(chunk_dir):
        return False
    for entry in os.listdir(chunk_dir):
        if not entry.startswith(".") and entry not in (".zarray", ".zattrs", ".zgroup"):
            return True
    return False


def parse_label_key(label_key: str) -> tuple[str, list[str]]:
    """Parse label key to extract crop path and class names."""
    match = re.search(r"\[([^\]]+)\]", label_key)
    if not match:
        raise ValueError(f"Could not parse classes from label key: {label_key}")
    classes = match.group(1).split(",")
    crop_path = label_key[: match.start()].rstrip("/")
    return crop_path, classes


def main():
    script_dir = Path(__file__).parent
    repo_dir = script_dir.parent
    output_dir = str(script_dir / "nifti_data")
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    # Find datasplit.csv
    datasplit_path = str(repo_dir / "datasplit.csv")
    if not os.path.exists(datasplit_path):
        print(f"ERROR: datasplit.csv not found at {datasplit_path}")
        sys.exit(1)

    print(f"Reading datasplit from: {datasplit_path}")
    print(f"NIfTI directory: {output_dir}")

    # Parse datasplit.csv (deduplicate by crop_id)
    entries = []
    seen = set()
    with open(datasplit_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            split = row[0].strip().strip('"')
            raw_path = row[1].strip().strip('"')
            label_path = row[3].strip().strip('"')
            label_key = row[4].strip().strip('"')

            if split not in ("train", "validate"):
                continue

            dataset_match = re.search(r"(jrc_[^/]+)", raw_path)
            dataset_name = dataset_match.group(1) if dataset_match else "unknown"
            crop_match = re.search(r"(crop\d+)", label_key)
            crop_name = crop_match.group(1) if crop_match else "unknown"
            crop_id = f"{dataset_name}_{crop_name}"

            if crop_id in seen:
                continue
            seen.add(crop_id)

            entries.append({
                "split": split,
                "crop_id": crop_id,
                "label_zarr_path": label_path,
                "label_key": label_key,
            })

    print(f"Found {len(entries)} unique crops in datasplit.csv")

    # Build image_paths, label_paths, annotated_classes_map from existing files
    image_paths = {}
    label_paths = {}
    annotated_classes_map = {}
    missing = 0

    for entry in entries:
        crop_id = entry["crop_id"]
        img_path = os.path.join(images_dir, f"{crop_id}_0000.nii.gz")
        lbl_path = os.path.join(labels_dir, f"{crop_id}.nii.gz")

        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            missing += 1
            continue

        image_paths[crop_id] = img_path
        label_paths[crop_id] = lbl_path

        # Determine annotated classes from zarr chunks
        try:
            crop_path, class_names = parse_label_key(entry["label_key"])
            scale_match = re.search(r"/(s\d+)$", entry["label_key"])
            scale_suffix = scale_match.group(1) if scale_match else "s0"

            annotated = []
            for cn in class_names:
                if cn not in CLASS_TO_ID:
                    continue
                class_key = f"{crop_path}/{cn}/{scale_suffix}"
                if has_zarr_chunks(entry["label_zarr_path"], class_key):
                    annotated.append(cn)
            annotated_classes_map[crop_id] = annotated
        except Exception as e:
            print(f"  WARNING: Could not determine annotated classes for {crop_id}: {e}")
            annotated_classes_map[crop_id] = []

    print(f"Found {len(image_paths)} NIfTI image/label pairs")
    if missing > 0:
        print(f"  ({missing} crops in datasplit.csv missing NIfTI files)")

    # Build datalist.json
    training = []
    validation = []

    for entry in entries:
        crop_id = entry["crop_id"]
        if crop_id not in image_paths:
            continue

        ann_names = annotated_classes_map.get(crop_id, [])
        ann_indices = sorted(
            CLASS_TO_ID[name] - 1
            for name in ann_names
            if name in CLASS_TO_ID
        )
        ann_str = ",".join(str(i) for i in ann_indices) if ann_indices else ""

        item = {
            "image": image_paths[crop_id],
            "label": label_paths[crop_id],
            "annotated_classes": ann_str,
        }

        if entry["split"] == "train":
            training.append(item)
        else:
            validation.append(item)

    class_names_cfg = [
        {"name": name, "index": [idx + 1]}
        for idx, name in enumerate(BASE_CLASSES)
    ]

    datalist = {
        "name": "CellMap FIB-SEM Segmentation Challenge",
        "description": "3D FIB-SEM volumes with single-channel integer labels and partial annotation masking",
        "modality": "CT",
        "sigmoid": True,
        "num_classes": len(BASE_CLASSES),
        "class_names": class_names_cfg,
        "training": training,
    }

    if validation:
        datalist["validation"] = validation

    datalist_path = os.path.join(output_dir, "datalist.json")
    with open(datalist_path, "w") as f:
        json.dump(datalist, f, indent=2)

    print(f"\nDatalist saved to {datalist_path}")
    print(f"  Training samples:   {len(training)}")
    print(f"  Validation samples: {len(validation)}")

    # Print annotation coverage summary
    print(f"\n  Per-class annotation coverage:")
    for cls_idx, cls_name in enumerate(BASE_CLASSES):
        n_annotated = sum(
            1 for cid, classes in annotated_classes_map.items()
            if cls_name in classes
        )
        n_total = len(annotated_classes_map)
        pct = 100 * n_annotated / n_total if n_total > 0 else 0
        print(f"    {cls_idx:2d} {cls_name:<12}: {n_annotated:>3}/{n_total} crops ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
