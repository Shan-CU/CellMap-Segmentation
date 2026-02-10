#!/usr/bin/env python3
"""
Convert CellMap Zarr data to NIfTI format for MONAI Auto3DSeg.

This script reads the CellMap datasplit.csv, extracts 3D crops from the zarr
volumes, and saves them as paired image/label NIfTI files suitable for
Auto3DSeg's DataAnalyzer and AutoRunner.

Label encoding:
  - Background = 0
  - Each of the 14 base classes gets a unique integer ID (1-14)
  - Auto3DSeg treats this as a standard multi-class segmentation problem

Usage:
    python convert_zarr_to_nifti.py \
        --datasplit ../datasplit.csv \
        --output_dir ./nifti_data \
        --scale s0 \
        --target_spacing 8 8 8
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    print("ERROR: nibabel is required. Install with: pip install nibabel")
    sys.exit(1)

try:
    import zarr
except ImportError:
    print("ERROR: zarr is required. Install with: pip install zarr")
    sys.exit(1)


# The 14 base label classes used in CellMap ground truth
BASE_CLASSES = [
    "ecs",
    "pm",
    "mito_mem",
    "mito_lum",
    "mito_ribo",
    "golgi_mem",
    "golgi_lum",
    "ves_mem",
    "ves_lum",
    "endo_mem",
    "endo_lum",
    "er_mem",
    "er_lum",
    "nuc",
]

# Class name -> integer ID mapping (1-indexed, 0 = background)
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(BASE_CLASSES)}


def parse_label_key(label_key: str) -> tuple[str, list[str]]:
    """
    Parse a label array key like:
      'recon-1/labels/groundtruth/crop234/[ecs,pm,mito_mem,...]/s0'
    Returns (crop_path, list_of_class_names).
    """
    # Extract the bracket-enclosed class list
    match = re.search(r"\[([^\]]+)\]", label_key)
    if not match:
        raise ValueError(f"Could not parse classes from label key: {label_key}")

    classes = match.group(1).split(",")

    # Extract crop path (everything before the bracket)
    crop_path = label_key[: match.start()].rstrip("/")

    return crop_path, classes


def read_zarr_array(zarr_path: str, array_key: str) -> np.ndarray:
    """Read a zarr array, handling both local and nested paths."""
    store = zarr.open(zarr_path, mode="r")
    arr = store[array_key]
    # Read into memory (handles chunked data)
    return np.array(arr)


def convert_crop(
    raw_zarr_path: str,
    raw_array_key: str,
    label_zarr_path: str,
    label_array_key: str,
    output_dir: str,
    crop_id: str,
    target_spacing: tuple[float, float, float] = (8.0, 8.0, 8.0),
) -> tuple[str, str] | None:
    """
    Convert a single crop from zarr to NIfTI.

    Returns (image_path, label_path) or None if conversion fails.
    """
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    image_path = os.path.join(images_dir, f"{crop_id}_0000.nii.gz")
    label_path = os.path.join(labels_dir, f"{crop_id}.nii.gz")

    # Skip if already converted
    if os.path.exists(image_path) and os.path.exists(label_path):
        print(f"  [SKIP] {crop_id} already converted")
        return image_path, label_path

    try:
        # Parse the label key to get crop path and class names
        crop_path, class_names = parse_label_key(label_array_key)

        # Determine which scale suffix to use for the raw data
        # The raw_array_key is like "recon-1/em/fibsem-uint8"
        # We need to append the scale level
        # Check if label_array_key ends with /s0, /s1, etc.
        scale_match = re.search(r"/(s\d+)$", label_array_key)
        scale_suffix = scale_match.group(1) if scale_match else "s0"

        # Read the raw EM image at the matching scale
        raw_key = f"{raw_array_key}/{scale_suffix}"
        print(f"  Reading raw EM from {raw_zarr_path} :: {raw_key}")

        raw_store = zarr.open(raw_zarr_path, mode="r")
        label_store = zarr.open(label_zarr_path, mode="r")

        # Read the first label class to get the shape and offset
        first_class = class_names[0]
        first_label_key = f"{crop_path}/{first_class}/{scale_suffix}"

        # Get label array info for spatial coordinates
        label_arr = label_store[first_label_key]
        label_shape = label_arr.shape
        print(f"  Label shape: {label_shape}, classes: {class_names}")

        # Try to get the spatial offset from zarr attributes
        # The label crop has translation offsets in its .zattrs
        crop_attrs_path = crop_path + "/" + first_class
        try:
            attrs = label_store[crop_attrs_path].attrs
            # Look for multiscales metadata with coordinateTransformations
            if "multiscales" in attrs:
                multiscales = attrs["multiscales"]
                if isinstance(multiscales, list) and len(multiscales) > 0:
                    datasets = multiscales[0].get("datasets", [])
                    for ds in datasets:
                        if ds.get("path") == scale_suffix:
                            transforms = ds.get("coordinateTransformations", [])
                            for t in transforms:
                                if t.get("type") == "translation":
                                    offset = t["translation"]
                                    print(f"  Spatial offset: {offset}")
                                    break
        except (KeyError, IndexError):
            pass

        # Get the raw EM data corresponding to this label crop region
        # We need to find the raw data that corresponds to the label's spatial
        # location. For simplicity (and since Auto3DSeg mainly cares about
        # intensity stats and spatial properties), we'll read the raw data
        # at the same array indices as the label crop.

        # The label crops are stored as separate arrays with translation offsets.
        # To find the corresponding raw region, we compute pixel coordinates from
        # the translation metadata. However, this requires parsing the full
        # multiscale metadata chain which varies per dataset.

        # PRACTICAL APPROACH: Read the raw EM at the label crop's coordinate space.
        # The zarr multiscale metadata has coordinateTransformations that define
        # the mapping. For the common case (s0 at 2nm), we can compute the
        # offset in voxels.

        raw_arr = raw_store[raw_key]

        # Try to find the voxel offset of the crop in the raw EM volume
        try:
            crop_attrs = label_store[crop_path + "/" + first_class].attrs
            multiscales = crop_attrs.get("multiscales", [{}])
            label_transforms = multiscales[0].get("datasets", [{}])

            # Get raw EM transforms
            raw_attrs = raw_store[raw_array_key].attrs
            raw_multiscales = raw_attrs.get("multiscales", [{}])
            raw_datasets = raw_multiscales[0].get("datasets", [{}])

            # Find scale and translation for label
            label_scale = None
            label_translation = None
            for ds in label_transforms:
                if ds.get("path") == scale_suffix:
                    for t in ds.get("coordinateTransformations", []):
                        if t["type"] == "scale":
                            label_scale = t["scale"]
                        elif t["type"] == "translation":
                            label_translation = t["translation"]

            # Find scale for raw
            raw_scale = None
            for ds in raw_datasets:
                if ds.get("path") == scale_suffix:
                    for t in ds.get("coordinateTransformations", []):
                        if t["type"] == "scale":
                            raw_scale = t["scale"]

            if label_scale and label_translation and raw_scale:
                # Convert translation from physical coords to voxel coords in raw
                # translation is in the same unit as scale (nanometers)
                voxel_offset = [
                    int(round(label_translation[i] / raw_scale[i]))
                    for i in range(len(raw_scale))
                ]
                # Read the corresponding raw region
                slices = tuple(
                    slice(voxel_offset[i], voxel_offset[i] + label_shape[i])
                    for i in range(len(label_shape))
                )
                print(f"  Raw region: offset={voxel_offset}, shape={label_shape}")

                # Check bounds
                valid = all(
                    voxel_offset[i] >= 0
                    and voxel_offset[i] + label_shape[i] <= raw_arr.shape[i]
                    for i in range(len(label_shape))
                )

                if valid:
                    raw_data = np.array(raw_arr[slices])
                else:
                    print(
                        f"  WARNING: Raw region out of bounds, using zeros. "
                        f"Offset={voxel_offset}, Label shape={label_shape}, "
                        f"Raw shape={raw_arr.shape}"
                    )
                    raw_data = np.zeros(label_shape, dtype=np.uint8)
            else:
                print(
                    "  WARNING: Could not determine spatial mapping, "
                    "reading raw from origin"
                )
                slices = tuple(slice(0, s) for s in label_shape)
                raw_data = np.array(raw_arr[slices])

        except (KeyError, IndexError, TypeError) as e:
            print(f"  WARNING: Metadata parsing failed ({e}), reading from origin")
            slices = tuple(
                slice(0, min(s, raw_arr.shape[i]))
                for i, s in enumerate(label_shape)
            )
            raw_data = np.array(raw_arr[slices])
            # Pad if needed
            if raw_data.shape != label_shape:
                padded = np.zeros(label_shape, dtype=raw_data.dtype)
                pad_slices = tuple(slice(0, s) for s in raw_data.shape)
                padded[pad_slices] = raw_data
                raw_data = padded

        # Build multi-class label volume
        # 0 = background, 1-14 = class IDs
        label_data = np.zeros(label_shape, dtype=np.uint8)

        for class_name in class_names:
            if class_name not in CLASS_TO_ID:
                print(f"  WARNING: Unknown class '{class_name}', skipping")
                continue

            class_id = CLASS_TO_ID[class_name]
            class_key = f"{crop_path}/{class_name}/{scale_suffix}"

            try:
                class_arr = np.array(label_store[class_key])
                # In CellMap: 0 = absent, 255 = unknown, other values = instance IDs
                # For semantic segmentation: nonzero and not 255 = class present
                mask = (class_arr > 0) & (class_arr != 255)
                # Only assign if not already assigned (first class wins for overlaps)
                unassigned = label_data == 0
                label_data[mask & unassigned] = class_id
            except KeyError:
                print(f"  WARNING: Class array not found: {class_key}")

        # Create NIfTI images with proper spacing
        # Spacing is in nanometers, but we'll use it as-is for Auto3DSeg
        # (the actual unit doesn't matter as long as it's consistent)
        affine = np.eye(4)
        affine[0, 0] = target_spacing[0]
        affine[1, 1] = target_spacing[1]
        affine[2, 2] = target_spacing[2]

        # Save image (add channel dim for MONAI: shape goes from ZYX to ZYX)
        img_nii = nib.Nifti1Image(raw_data.astype(np.float32), affine)
        nib.save(img_nii, image_path)

        # Save label
        lbl_nii = nib.Nifti1Image(label_data.astype(np.uint8), affine)
        nib.save(lbl_nii, label_path)

        print(f"  [OK] Saved {crop_id}: image={raw_data.shape}, "
              f"label classes={np.unique(label_data).tolist()}")

        return image_path, label_path

    except Exception as e:
        print(f"  [ERROR] Failed to convert {crop_id}: {e}")
        import traceback
        traceback.print_exc()
        return None


def parse_datasplit(datasplit_path: str) -> list[dict]:
    """
    Parse the CellMap datasplit.csv into a list of crop entries.
    Returns unique crops (deduplicates by dataset+crop combination).
    """
    entries = []
    seen = set()

    with open(datasplit_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue

            split = row[0].strip().strip('"')
            raw_path = row[1].strip().strip('"')
            raw_key = row[2].strip().strip('"')
            label_path = row[3].strip().strip('"')
            label_key = row[4].strip().strip('"')

            # Skip non-train/validate entries
            if split not in ("train", "validate"):
                continue

            # Extract dataset name and crop ID for deduplication
            # Dataset: e.g., jrc_cos7-1a
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
                "raw_path": raw_path,
                "raw_key": raw_key,
                "label_path": label_path,
                "label_key": label_key,
                "crop_id": crop_id,
                "dataset": dataset_name,
            })

    return entries


def create_datalist(
    entries: list[dict],
    output_dir: str,
    image_paths: dict[str, str],
    label_paths: dict[str, str],
) -> str:
    """
    Create the datalist.json required by Auto3DSeg.
    Returns path to the datalist file.
    """
    training = []
    validation = []

    for entry in entries:
        crop_id = entry["crop_id"]
        if crop_id not in image_paths or crop_id not in label_paths:
            continue

        item = {
            "image": image_paths[crop_id],
            "label": label_paths[crop_id],
        }

        if entry["split"] == "train":
            training.append(item)
        else:
            validation.append(item)

    # If no explicit validation split, Auto3DSeg will do cross-validation
    datalist = {
        "name": "CellMap FIB-SEM Segmentation Challenge",
        "description": "3D FIB-SEM volumes with organelle annotations",
        "modality": "EM",
        "num_classes": len(BASE_CLASSES),
        "class_names": BASE_CLASSES,
        "training": training,
    }

    if validation:
        datalist["validation"] = validation

    datalist_path = os.path.join(output_dir, "datalist.json")
    with open(datalist_path, "w") as f:
        json.dump(datalist, f, indent=2)

    print(f"\nDatalist saved to {datalist_path}")
    print(f"  Training samples: {len(training)}")
    print(f"  Validation samples: {len(validation)}")

    return datalist_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert CellMap zarr data to NIfTI for Auto3DSeg"
    )
    parser.add_argument(
        "--datasplit",
        type=str,
        default="datasplit.csv",
        help="Path to datasplit.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="auto3dseg/nifti_data",
        help="Output directory for NIfTI files",
    )
    parser.add_argument(
        "--target_spacing",
        type=float,
        nargs=3,
        default=[8.0, 8.0, 8.0],
        help="Target voxel spacing in nm (z y x)",
    )
    parser.add_argument(
        "--max_crops",
        type=int,
        default=0,
        help="Max crops to convert (0=all, useful for testing)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Only convert specific datasets (e.g., jrc_cos7-1a jrc_hela-2)",
    )

    args = parser.parse_args()

    # Find the datasplit file
    datasplit_path = args.datasplit
    if not os.path.exists(datasplit_path):
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        datasplit_path = str(script_dir / "datasplit.csv")

    if not os.path.exists(datasplit_path):
        print(f"ERROR: datasplit.csv not found at {args.datasplit} or {datasplit_path}")
        sys.exit(1)

    print(f"Reading datasplit from: {datasplit_path}")
    entries = parse_datasplit(datasplit_path)
    print(f"Found {len(entries)} unique crops")

    # Filter by dataset if specified
    if args.datasets:
        entries = [e for e in entries if e["dataset"] in args.datasets]
        print(f"Filtered to {len(entries)} crops from datasets: {args.datasets}")

    # Limit crops if specified
    if args.max_crops > 0:
        entries = entries[: args.max_crops]
        print(f"Limited to {args.max_crops} crops")

    # Convert each crop
    os.makedirs(args.output_dir, exist_ok=True)
    target_spacing = tuple(args.target_spacing)

    image_paths: dict[str, str] = {}
    label_paths: dict[str, str] = {}

    for i, entry in enumerate(entries):
        crop_id = entry["crop_id"]
        print(f"\n[{i+1}/{len(entries)}] Converting {crop_id} ({entry['split']})...")

        result = convert_crop(
            raw_zarr_path=entry["raw_path"],
            raw_array_key=entry["raw_key"],
            label_zarr_path=entry["label_path"],
            label_array_key=entry["label_key"],
            output_dir=args.output_dir,
            crop_id=crop_id,
            target_spacing=target_spacing,
        )

        if result:
            image_paths[crop_id] = result[0]
            label_paths[crop_id] = result[1]

    # Create datalist.json
    print(f"\n{'='*60}")
    print("Creating Auto3DSeg datalist...")
    datalist_path = create_datalist(entries, args.output_dir, image_paths, label_paths)

    # Summary
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print(f"  NIfTI files: {args.output_dir}/")
    print(f"  Datalist: {datalist_path}")
    print(f"  Converted: {len(image_paths)}/{len(entries)} crops")
    print(f"\nClass mapping:")
    for name, idx in CLASS_TO_ID.items():
        print(f"  {idx:2d} = {name}")
    print(f"\nNext step: run Auto3DSeg analysis:")
    print(f"  python auto3dseg/run_auto3dseg.py --mode analyze \\")
    print(f"      --datalist {datalist_path}")


if __name__ == "__main__":
    main()
