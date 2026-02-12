#!/usr/bin/env python3
"""
Re-convert the 43 crops that got blank (all-zero) images due to a strict
bounds check in the original zarr→NIfTI conversion.

The labels for these crops are already correct — only the images need
to be regenerated with clip+pad logic.

Root cause:
    convert_zarr_to_nifti.py used an all-or-nothing bounds check.  If the
    annotation crop extended even 1 voxel past the raw EM boundary, the
    entire image was written as zeros.  In reality 50-100% of each crop
    overlaps with valid raw EM data.

Fix:
    Read the overlapping region, zero-pad the out-of-bounds edges.

Usage:
    python auto3dseg/reconvert_blank_images.py          # from repo root
    python auto3dseg/reconvert_blank_images.py --dry-run # preview only
"""

import argparse
import os
import re
import sys

import nibabel as nib
import numpy as np
import zarr

# ---------------------------------------------------------------------------
# 43 crops whose images are all-zeros
# ---------------------------------------------------------------------------
BLANK_CROPS = [
    "jrc_cos7-1a_crop254",
    "jrc_cos7-1a_crop257",
    "jrc_cos7-1b_crop238",
    "jrc_hela-3_crop62",
    "jrc_hela-3_crop63",
    "jrc_hela-3_crop64",
    "jrc_hela-3_crop65",
    "jrc_jurkat-1_crop67",
    "jrc_mus-heart-1_crop423",
    "jrc_mus-kidney_crop179",
    "jrc_mus-kidney_crop184",
    "jrc_mus-kidney_crop230",
    "jrc_mus-kidney_crop231",
    "jrc_mus-liver_crop177",
    "jrc_mus-liver-3_crop473",
    "jrc_mus-liver-zon-1_crop282",
    "jrc_mus-liver-zon-1_crop289",
    "jrc_mus-liver-zon-1_crop324",
    "jrc_mus-liver-zon-1_crop337",
    "jrc_mus-liver-zon-1_crop346",
    "jrc_mus-liver-zon-1_crop349",
    "jrc_mus-liver-zon-1_crop351",
    "jrc_mus-liver-zon-1_crop386",
    "jrc_mus-liver-zon-1_crop410",
    "jrc_mus-liver-zon-1_crop412",
    "jrc_mus-liver-zon-1_crop413",
    "jrc_mus-liver-zon-2_crop355",
    "jrc_mus-liver-zon-2_crop356",
    "jrc_mus-liver-zon-2_crop357",
    "jrc_mus-liver-zon-2_crop358",
    "jrc_mus-liver-zon-2_crop367",
    "jrc_mus-liver-zon-2_crop387",
    "jrc_mus-nacc-1_crop115",
    "jrc_sum159-4_crop202",
    "jrc_sum159-4_crop203",
    "jrc_sum159-4_crop206",
    "jrc_sum159-4_crop213",
    "jrc_sum159-4_crop218",
    "jrc_sum159-4_crop219",
    "jrc_ut21-1413-003_crop197",
    "jrc_ut21-1413-003_crop227",
    "jrc_zf-cardiac-1_crop379",
    "jrc_zf-cardiac-1_crop380",
]


def reconvert_image(crop_id: str, data_dir: str, nifti_dir: str, dry_run: bool = False):
    """Re-read raw EM from zarr with clip+pad and overwrite the NIfTI image."""
    parts = crop_id.rsplit("_crop", 1)
    dataset_name = parts[0]
    crop_num = parts[1]

    zarr_path = os.path.join(data_dir, dataset_name, f"{dataset_name}.zarr")
    image_path = os.path.join(nifti_dir, "images", f"{crop_id}_0000.nii.gz")

    if not os.path.exists(zarr_path):
        print(f"  [SKIP] {crop_id}: zarr not found at {zarr_path}")
        return False

    store = zarr.open(zarr_path, mode="r")

    # --- Determine the scale suffix used during original conversion ---
    # Read the existing NIfTI to get the spacing (tells us which scale level)
    existing = nib.load(image_path)
    spacing = existing.header.get_zooms()
    label_shape = existing.shape
    # spacing[0] == 8.0 → s2 (2nm base * 2^2 = 8nm)... but the original
    # converter used whatever scale_suffix matched the label key.
    # We'll determine it from the label metadata.

    crop_path = f"recon-1/labels/groundtruth/crop{crop_num}"
    crop_group = store[crop_path]
    first_cls = None
    for cls_name in crop_group.keys():
        first_cls = cls_name
        break
    if first_cls is None:
        print(f"  [SKIP] {crop_id}: no classes in label crop")
        return False

    label_cls_group = crop_group[first_cls]
    # Find which scale level has the matching shape
    scale_suffix = None
    for key in sorted(label_cls_group.keys()):
        if label_cls_group[key].shape == tuple(label_shape):
            scale_suffix = key
            break
    if scale_suffix is None:
        # Default to s0
        scale_suffix = "s0"
        label_shape = label_cls_group["s0"].shape

    # --- Get coordinate transforms ---
    ms = label_cls_group.attrs.get("multiscales", [{}])[0]
    label_translation = None
    label_scale = None
    for ds in ms.get("datasets", []):
        if ds.get("path") == scale_suffix:
            for t in ds.get("coordinateTransformations", []):
                if t["type"] == "translation":
                    label_translation = t["translation"]
                elif t["type"] == "scale":
                    label_scale = t["scale"]

    raw_path = "recon-1/em/fibsem-uint8"
    raw_group = store[raw_path]
    raw_arr = raw_group[scale_suffix]

    raw_ms = raw_group.attrs.get("multiscales", [{}])[0]
    raw_scale = None
    for ds in raw_ms.get("datasets", []):
        if ds.get("path") == scale_suffix:
            for t in ds.get("coordinateTransformations", []):
                if t["type"] == "scale":
                    raw_scale = t["scale"]

    if label_translation is None or raw_scale is None:
        print(f"  [SKIP] {crop_id}: missing coordinate metadata")
        return False

    # --- Compute voxel offset and clipped region ---
    ndim = len(label_shape)
    voxel_offset = [
        int(round(label_translation[i] / raw_scale[i])) for i in range(ndim)
    ]

    clipped_start = [max(voxel_offset[i], 0) for i in range(ndim)]
    clipped_end = [
        min(voxel_offset[i] + label_shape[i], raw_arr.shape[i]) for i in range(ndim)
    ]
    clipped_shape = [clipped_end[i] - clipped_start[i] for i in range(ndim)]

    overlap_voxels = 1
    for s in clipped_shape:
        overlap_voxels *= max(s, 0)
    total_voxels = 1
    for s in label_shape:
        total_voxels *= s
    pct = 100.0 * overlap_voxels / total_voxels if total_voxels else 0

    if overlap_voxels <= 0:
        print(f"  [SKIP] {crop_id}: no overlap with raw EM ({pct:.1f}%)")
        return False

    print(
        f"  {crop_id}: {scale_suffix} offset={voxel_offset} "
        f"clip_shape={clipped_shape} raw={raw_arr.shape} "
        f"overlap={pct:.1f}%"
    )

    if dry_run:
        return True

    # --- Read clipped region and assemble padded image ---
    clipped_slices = tuple(
        slice(clipped_start[i], clipped_end[i]) for i in range(ndim)
    )
    clipped_data = np.array(raw_arr[clipped_slices])

    raw_data = np.zeros(label_shape, dtype=clipped_data.dtype)
    dest_slices = tuple(
        slice(
            max(0, -voxel_offset[i]),
            max(0, -voxel_offset[i]) + clipped_shape[i],
        )
        for i in range(ndim)
    )
    raw_data[dest_slices] = clipped_data

    # --- Save NIfTI with same affine as original ---
    affine = existing.affine.copy()
    img_nii = nib.Nifti1Image(raw_data.astype(np.float32), affine)
    nib.save(img_nii, image_path)

    # Sanity check
    print(
        f"    → Saved: min={raw_data.min()}, max={raw_data.max()}, "
        f"nonzero_pct={100.0 * np.count_nonzero(raw_data) / raw_data.size:.1f}%"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Re-convert blank crop images")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Path to the data/ directory with zarr stores",
    )
    parser.add_argument(
        "--nifti-dir",
        default="auto3dseg/nifti_data",
        help="Path to NIfTI output directory",
    )
    args = parser.parse_args()

    print(f"Re-converting {len(BLANK_CROPS)} crops with blank images")
    print(f"  Data dir:  {args.data_dir}")
    print(f"  NIfTI dir: {args.nifti_dir}")
    print(f"  Dry run:   {args.dry_run}")
    print()

    success = 0
    skipped = 0
    for crop_id in BLANK_CROPS:
        ok = reconvert_image(crop_id, args.data_dir, args.nifti_dir, args.dry_run)
        if ok:
            success += 1
        else:
            skipped += 1

    print(f"\nDone: {success} reconverted, {skipped} skipped")


if __name__ == "__main__":
    main()
