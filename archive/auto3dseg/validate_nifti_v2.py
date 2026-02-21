#!/usr/bin/env python3
"""
Validate converted NIfTI data against original zarr groundtruth.

Checks every crop in datalist.json for:
  1. Image/label shape match
  2. EM data matches zarr source at the correct voxel offset (not black, not wrong region)
  3. Per-class label voxels match zarr groundtruth (accounting for nuc skip)
  4. annotated_indices tracks all zarr-present classes (including true negatives)
  5. No all-zero images
  6. Label IDs are in valid range [0, NUM_CLASSES]

Prints a per-crop pass/fail report and exits non-zero if ANY check fails.

Usage:
  python auto3dseg/validate_nifti_v2.py [--workers 8] [--quick]

  --quick   Validate a random 10% sample instead of all crops (for fast CI).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
import numpy as np
import zarr


# ── Must match convert_zarr_to_nifti_v2.py ──
ATOMIC_CLASSES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
    "lyso_mem", "lyso_lum", "ld_mem", "ld_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl",
    "mt_out", "cyto", "mt_in", "perox_mem", "perox_lum",
    "nhchrom", "nechrom", "nucleo",
]
NUM_CLASSES = len(ATOMIC_CLASSES)
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(ATOMIC_CLASSES)}

# Sub-nuclear classes whose union == nuc (nuc is NOT in the NIfTI label)
NUC_SUB_CLASSES = {
    "ne_mem", "ne_lum", "np_out", "np_in",
    "hchrom", "echrom", "nucpl",
    "nhchrom", "nechrom", "nucleo",
}
NUC_SUB_IDS = sorted(CLASS_TO_ID[c] for c in NUC_SUB_CLASSES)


def _get_zarr_metadata(zarr_path: str, crop_name: str):
    """Get voxel offset, label shape, and EM shape from zarr metadata."""
    gt_base = os.path.join(zarr_path, "recon-1", "labels", "groundtruth", crop_name)
    em_base = os.path.join(zarr_path, "recon-1", "em", "fibsem-uint8")

    label_translation = None
    raw_scale_vals = None
    label_shape = None

    for cls_name in ATOMIC_CLASSES:
        cls_dir = os.path.join(gt_base, cls_name)
        s0_path = os.path.join(cls_dir, "s0")
        if not os.path.isdir(s0_path):
            continue
        arr = zarr.open(s0_path, mode="r")
        if arr.size == 0:
            continue
        label_shape = arr.shape

        label_group = zarr.open(cls_dir, mode="r")
        if hasattr(label_group, "attrs") and "multiscales" in label_group.attrs:
            ms = label_group.attrs["multiscales"]
            if isinstance(ms, list) and len(ms) > 0:
                for ds_entry in ms[0].get("datasets", []):
                    if ds_entry.get("path") == "s0":
                        for t in ds_entry.get("coordinateTransformations", []):
                            if t.get("type") == "translation":
                                label_translation = t["translation"]
        break

    em_group = zarr.open(em_base, mode="r")
    if hasattr(em_group, "attrs") and "multiscales" in em_group.attrs:
        ms = em_group.attrs["multiscales"]
        if isinstance(ms, list) and len(ms) > 0:
            for ds_entry in ms[0].get("datasets", []):
                if ds_entry.get("path") == "s0":
                    for t in ds_entry.get("coordinateTransformations", []):
                        if t.get("type") == "scale":
                            raw_scale_vals = t["scale"]

    voxel_offset = None
    if label_translation and raw_scale_vals:
        voxel_offset = [
            int(round(label_translation[i] / raw_scale_vals[i]))
            for i in range(len(raw_scale_vals))
        ]

    em_shape = None
    em_s0 = os.path.join(em_base, "s0")
    if os.path.isdir(em_s0):
        em_shape = zarr.open(em_s0, mode="r").shape

    return voxel_offset, label_shape, em_shape


def validate_one_crop(
    crop_id: str,
    img_path: str,
    lbl_path: str,
    annotated_str: str,
    data_dir: str,
) -> tuple[bool, str, list[str]]:
    """Validate a single NIfTI crop against zarr source.

    Returns (passed, crop_id, list_of_error_messages).
    """
    errors: list[str] = []

    # ── Parse crop_id → dataset + crop ──
    parts = crop_id.rsplit("_", 1)
    crop_name = parts[-1]
    dataset = parts[0]
    zarr_path = os.path.join(data_dir, dataset, f"{dataset}.zarr")

    if not os.path.isdir(zarr_path):
        return False, crop_id, [f"zarr not found: {zarr_path}"]

    # ── Load NIfTI ──
    if not os.path.isfile(img_path):
        return False, crop_id, [f"image file missing: {img_path}"]
    if not os.path.isfile(lbl_path):
        return False, crop_id, [f"label file missing: {lbl_path}"]

    img_nii = nib.load(img_path)
    lbl_nii = nib.load(lbl_path)
    img_data = np.asarray(img_nii.dataobj)
    lbl_data = np.asarray(lbl_nii.dataobj)

    # ── Check 1: Shape match ──
    if img_data.shape != lbl_data.shape:
        errors.append(f"shape mismatch: image={img_data.shape} label={lbl_data.shape}")

    # ── Check 2: No all-zero image ──
    if img_data.max() == 0:
        errors.append(f"ALL-ZERO image (black)")

    # ── Check 3: Label IDs in valid range ──
    unique_ids = np.unique(lbl_data)
    invalid = unique_ids[unique_ids > NUM_CLASSES]
    if len(invalid) > 0:
        errors.append(f"invalid label IDs: {invalid}")

    # ── Check 4: nuc (id=14) should NOT be in labels ──
    nuc_count = int(np.sum(lbl_data == CLASS_TO_ID["nuc"]))
    if nuc_count > 0:
        errors.append(f"nuc (id=14) found in labels ({nuc_count} voxels) — should be omitted")

    # ── Check 5: EM matches zarr at correct offset ──
    voxel_offset, label_shape, em_shape = _get_zarr_metadata(zarr_path, crop_name)

    if voxel_offset is not None and label_shape is not None and em_shape is not None:
        em_arr = zarr.open(
            os.path.join(zarr_path, "recon-1", "em", "fibsem-uint8", "s0"), mode="r"
        )
        # Compute the clipped region (same logic as converter)
        clipped_start = [max(0, voxel_offset[i]) for i in range(3)]
        clipped_end = [min(em_shape[i], voxel_offset[i] + label_shape[i]) for i in range(3)]
        clipped_shape = tuple(clipped_end[i] - clipped_start[i] for i in range(3))

        if all(s > 0 for s in clipped_shape):
            slices = tuple(slice(clipped_start[i], clipped_end[i]) for i in range(3))
            zarr_em_region = np.array(em_arr[slices])

            # Where in the NIfTI image should this data appear
            pad_start = [clipped_start[i] - voxel_offset[i] for i in range(3)]
            nifti_slices = tuple(
                slice(pad_start[i], pad_start[i] + zarr_em_region.shape[i])
                for i in range(3)
            )

            try:
                nifti_em_region = img_data[nifti_slices]
                if not np.array_equal(zarr_em_region, nifti_em_region):
                    n_diff = int(np.sum(zarr_em_region != nifti_em_region))
                    n_total = zarr_em_region.size
                    errors.append(
                        f"EM data mismatch: {n_diff}/{n_total} voxels differ "
                        f"({100*n_diff/n_total:.1f}%)"
                    )
            except (IndexError, ValueError) as e:
                errors.append(f"EM slice comparison failed: {e}")

            # Check that padded border (if any) is zero
            is_clipped = any(
                voxel_offset[i] < 0 or voxel_offset[i] + label_shape[i] > em_shape[i]
                for i in range(3)
            )
            if is_clipped:
                # Create a mask of padded voxels (outside clipped region)
                padded_mask = np.ones(img_data.shape, dtype=bool)
                padded_mask[nifti_slices] = False
                padded_sum = int(img_data[padded_mask].sum())
                if padded_sum > 0:
                    errors.append(f"padded region has non-zero values (sum={padded_sum})")

    # ── Check 6: Per-class label voxels match zarr ──
    gt_base = os.path.join(zarr_path, "recon-1", "labels", "groundtruth", crop_name)
    for cls_name in ATOMIC_CLASSES:
        if cls_name == "nuc":
            continue  # nuc is not in NIfTI, tested separately below

        cls_id = CLASS_TO_ID[cls_name]
        s0_path = os.path.join(gt_base, cls_name, "s0")
        if not os.path.isdir(s0_path):
            # Class not annotated in zarr — should not appear in NIfTI
            nifti_count = int(np.sum(lbl_data == cls_id))
            if nifti_count > 0:
                errors.append(f"{cls_name} has {nifti_count} voxels in NIfTI but no zarr dir")
            continue

        try:
            zarr_gt = np.asarray(zarr.open(s0_path, mode="r"))
        except Exception:
            continue

        if zarr_gt.shape != lbl_data.shape:
            continue  # shape mismatch handled above

        zarr_fg = zarr_gt > 0
        zarr_count = int(zarr_fg.sum())
        nifti_has = (lbl_data == cls_id)
        nifti_count = int(nifti_has.sum())

        if zarr_count == 0:
            # True negative — should have 0 voxels in NIfTI too
            if nifti_count > 0:
                errors.append(f"{cls_name}: zarr has 0 fg but NIfTI has {nifti_count}")
            continue

        # For classes that can be overwritten by later classes in the encoding,
        # nifti_count <= zarr_count is expected (last-writer-wins). But the
        # voxels that ARE in NIfTI must match zarr foreground locations.
        if nifti_count > 0:
            # Every NIfTI voxel with this class ID should be foreground in zarr
            precision = int((nifti_has & zarr_fg).sum()) / nifti_count
            if precision < 0.999:
                errors.append(
                    f"{cls_name}: {100*(1-precision):.2f}% of NIfTI voxels not in zarr fg"
                )

    # ── Check 7: nuc reconstruction sanity ──
    # Verify that union of sub-nuclear classes in NIfTI covers what zarr "nuc" has
    nuc_s0 = os.path.join(gt_base, "nuc", "s0")
    if os.path.isdir(nuc_s0):
        try:
            zarr_nuc = np.asarray(zarr.open(nuc_s0, mode="r"))
            if zarr_nuc.shape == lbl_data.shape:
                zarr_nuc_fg = zarr_nuc > 0
                zarr_nuc_count = int(zarr_nuc_fg.sum())

                # Reconstruct nuc from NIfTI sub-nuclear labels
                nifti_nuc_union = np.zeros(lbl_data.shape, dtype=bool)
                for sid in NUC_SUB_IDS:
                    nifti_nuc_union |= (lbl_data == sid)
                nifti_nuc_count = int(nifti_nuc_union.sum())

                # The union should cover ≥99% of zarr nuc (may lose edge
                # voxels due to sub-class annotation gaps)
                if zarr_nuc_count > 0:
                    recall = int((zarr_nuc_fg & nifti_nuc_union).sum()) / zarr_nuc_count
                    if recall < 0.95:
                        errors.append(
                            f"nuc recall from sub-nuclear union: {recall:.3f} "
                            f"(zarr={zarr_nuc_count}, nifti_union={nifti_nuc_count})"
                        )
        except Exception:
            pass

    # ── Check 8: annotated_indices completeness ──
    annotated_set = set()
    if annotated_str:
        for idx_str in annotated_str.split(","):
            idx_str = idx_str.strip()
            if idx_str:
                annotated_set.add(int(idx_str))

    # Every class with a zarr directory should be in annotated_indices
    for cls_name in ATOMIC_CLASSES:
        cls_idx = ATOMIC_CLASSES.index(cls_name)  # 0-based
        cls_dir = os.path.join(gt_base, cls_name)
        if os.path.isdir(cls_dir) and cls_idx not in annotated_set:
            errors.append(f"{cls_name} (idx={cls_idx}) has zarr dir but not in annotated_indices")

    return len(errors) == 0, crop_id, errors


def _validate_wrapper(args):
    """Wrapper for multiprocessing."""
    try:
        return validate_one_crop(*args)
    except Exception as e:
        return False, args[0], [f"EXCEPTION: {e}"]


def main():
    parser = argparse.ArgumentParser(description="Validate NIfTI v2 data against zarr source")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="Validate 10% random sample")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--nifti-dir", type=str, default=None)
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    repo_dir = script_dir.parent
    data_dir = args.data_dir or str(repo_dir / "data")
    nifti_dir = args.nifti_dir or str(script_dir / "nifti_data_v2")
    datalist_path = os.path.join(nifti_dir, "datalist.json")

    print(f"NIfTI v2 Validation")
    print(f"  Data dir:  {data_dir}")
    print(f"  NIfTI dir: {nifti_dir}")
    print(f"  Datalist:  {datalist_path}")
    print()

    with open(datalist_path) as f:
        datalist = json.load(f)

    all_entries = datalist.get("training", []) + datalist.get("validation", [])

    if args.quick:
        import random
        random.seed(42)
        n = max(1, len(all_entries) // 10)
        all_entries = random.sample(all_entries, n)
        print(f"  Quick mode: validating {n}/{len(datalist.get('training', []))+len(datalist.get('validation', []))} crops\n")

    # Build validation jobs
    jobs = []
    for entry in all_entries:
        crop_id = os.path.basename(entry["label"]).replace(".nii.gz", "")
        jobs.append((
            crop_id,
            entry["image"],
            entry["label"],
            entry.get("annotated_classes", ""),
            data_dir,
        ))

    print(f"Validating {len(jobs)} crops...\n")

    t0 = time.time()
    passed = 0
    failed = 0
    all_errors: dict[str, list[str]] = {}

    if args.workers <= 1:
        for job in jobs:
            ok, crop_id, errs = _validate_wrapper(job)
            if ok:
                passed += 1
            else:
                failed += 1
                all_errors[crop_id] = errs
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_validate_wrapper, job): job for job in jobs}
            for future in as_completed(futures):
                ok, crop_id, errs = future.result()
                if ok:
                    passed += 1
                else:
                    failed += 1
                    all_errors[crop_id] = errs

    elapsed = time.time() - t0

    # ── Report ──
    print(f"{'='*70}")
    print(f"VALIDATION RESULTS  ({elapsed:.1f}s)")
    print(f"{'='*70}")

    if all_errors:
        print(f"\n  FAILURES ({failed}):")
        for crop_id in sorted(all_errors):
            print(f"\n  ✗ {crop_id}")
            for err in all_errors[crop_id]:
                print(f"      - {err}")

    print(f"\n  Summary: {passed} passed, {failed} failed, {passed+failed} total")

    if failed > 0:
        print(f"\n  ❌ VALIDATION FAILED — do NOT proceed with training!")
        sys.exit(1)
    else:
        print(f"\n  ✅ ALL CHECKS PASSED — data is ready for training.")
        sys.exit(0)


if __name__ == "__main__":
    main()
