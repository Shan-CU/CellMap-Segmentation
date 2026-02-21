#!/usr/bin/env python3
"""
Segment ALL validation crops into full 3D NIfTI volumes using trained 2D models.

For each 2D model (resnet, unet, swin, vit), loads the best checkpoint and runs
slice-by-slice inference along all three axes (axial, coronal, sagittal), then
averages the three orientation probabilities to produce a final 3D segmentation.

Outputs per model per crop:
    <out>/<model>/<crop>/raw_image.nii.gz       — original EM volume (uint8)
    <out>/<model>/<crop>/ground_truth.nii.gz    — integer label map (0=bg, 1-35)
    <out>/<model>/<crop>/prediction.nii.gz      — integer label map from 2D model
    <out>/<model>/<crop>/prediction_ml.nii.gz   — multi-label 4D volume (D,H,W,C) uint8
    <out>/<model>/<crop>/dice_summary.json      — per-class Dice scores

The integer label maps assign each voxel to at most one class (argmax of
probabilities, restricted to annotated classes only). The multi-label volume
stores per-class binary masks as the 4th dimension — open in ImageJ with
Image → Hyperstack to browse individual class channels.

All NIfTI files preserve original affine → can be opened directly in ImageJ
(File → Open, or drag-and-drop).  In ImageJ use Image → Lookup Tables → Spectrum
for coloured label maps.

Validation volumes are processed in order of decreasing annotation count so the
most informative crops finish first.

Usage:
    python segment_val_volumes_2d.py                       # all 4 models, all 45 val crops
    python segment_val_volumes_2d.py --model resnet_r3     # single model
    python segment_val_volumes_2d.py --top-k 10            # only top-10 most annotated
    python segment_val_volumes_2d.py --save-probs          # also save 35-ch probability volume

Reference: EXPERIMENT_FINDINGS.md §5 (Model Architecture Comparison)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from models.mdl_cellmap_2d import Net2D

# ── Constants ─────────────────────────────────────────────────────────
RUNS_DIR = "/work/users/g/s/gsgeorge/cellmap/runs/monai_2d"
OUTPUT_ROOT = "/work/users/g/s/gsgeorge/cellmap/runs/monai_2d/val_volumes"

MODEL_CONFIGS = {
    "resnet_r3": "cfg_2d_resnet_r3",
    "unet_r3":   "cfg_2d_unet_r3",
    "swin_r3":   "cfg_2d_swin_r3",
    "vit_r3":    "cfg_2d_vit_r3",
}

CLASS_NAMES = [
    # Original 14 from Round 1
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
    # New for Round 2
    "lyso_mem", "lyso_lum", "ld_mem", "ld_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl",
    "mt_out", "cyto", "mt_in", "perox_mem", "perox_lum",
    "nhchrom", "nechrom", "nucleo",
]

FG_THRESHOLD = 0.01  # match training foreground masking


# ── Helpers ───────────────────────────────────────────────────────────

def load_config(config_name: str):
    """Import a config module from configs_2d/."""
    if "." not in config_name:
        config_name = f"configs_2d.{config_name}"
    mod = importlib.import_module(config_name)
    return mod.cfg


def load_model(model_name: str, device: torch.device):
    """Load a 2D model from its best checkpoint."""
    config_name = MODEL_CONFIGS[model_name]
    cfg = load_config(config_name)

    run_dir = os.path.join(RUNS_DIR, model_name)
    ckpt_path = os.path.join(run_dir, "checkpoint_best.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No best checkpoint at {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    best_epoch = ckpt.get("epoch", "?")
    best_metric = ckpt.get("best_metric", 0.0)

    model = Net2D(cfg)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded {model_name}: epoch {best_epoch}, val Dice {best_metric:.4f}, "
          f"{n_params:.1f}M params")
    return model, cfg


def get_sorted_val_entries(datalist_path: str) -> list[dict]:
    """Load validation entries sorted by annotation count (descending)."""
    with open(datalist_path) as f:
        datalist = json.load(f)
    val = datalist.get("validation", [])

    def count_annotations(entry):
        ann = entry.get("annotated_classes", "")
        if isinstance(ann, str) and ann.strip():
            return len([x for x in ann.split(",") if x.strip()])
        elif isinstance(ann, (list, tuple)):
            return len(ann)
        return 0

    # Sort by annotation count descending
    val_sorted = sorted(val, key=count_annotations, reverse=True)
    return val_sorted


def crop_name_from_path(image_path: str) -> str:
    """Extract crop name: 'jrc_hela-2_crop14_0000.nii.gz' → 'jrc_hela-2_crop14'."""
    return Path(image_path).name.replace("_0000.nii.gz", "").replace(".nii.gz", "")


def infer_axis(
    model: torch.nn.Module,
    volume: np.ndarray,
    axis: int,
    roi_size: list[int],
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    num_classes: int = 35,
) -> np.ndarray:
    """
    Run 2D inference on all slices along a given axis.

    For each slice:
      1. Pad to roi_size if needed.
      2. Tile into non-overlapping roi_size patches (with overlap=0 for speed).
      3. Run the model, get sigmoid probabilities.
      4. Zero out predictions on black-padding pixels (foreground masking).
      5. Assemble back into the full slice.

    Args:
        model: Trained Net2D in eval mode.
        volume: (D, H, W) float32 normalized [0, 1].
        axis: 0 (axial), 1 (coronal), 2 (sagittal).
        roi_size: [rh, rw] — model's expected input size.
        device: torch device.
        amp_dtype: autocast dtype.
        amp_enabled: whether to use autocast.
        num_classes: number of output channels.

    Returns:
        probs: (C, D, H, W) float32 sigmoid probabilities.
    """
    D, H, W = volume.shape
    rh, rw = roi_size

    # Determine slice dimensions for this axis
    if axis == 0:
        n_slices = D
    elif axis == 1:
        n_slices = H
    else:
        n_slices = W

    # Output accumulator
    probs_out = np.zeros((num_classes, D, H, W), dtype=np.float32)

    for si in range(n_slices):
        # Extract 2D slice
        if axis == 0:
            slc = volume[si]          # (H, W)
        elif axis == 1:
            slc = volume[:, si]       # (D, W)
        else:
            slc = volume[:, :, si]    # (D, H)

        sh, sw = slc.shape

        # Pad slice to at least roi_size
        pad_h = max(0, rh - sh)
        pad_w = max(0, rw - sw)
        if pad_h > 0 or pad_w > 0:
            slc_padded = np.pad(slc, ((0, pad_h), (0, pad_w)),
                                mode="constant", constant_values=0)
        else:
            slc_padded = slc

        ph, pw = slc_padded.shape

        # Tile the padded slice into roi_size patches
        # Use step = roi_size for non-overlapping (fast); could use overlap for quality
        ys = list(range(0, ph - rh + 1, rh))
        xs = list(range(0, pw - rw + 1, rw))
        # Ensure we cover the last patch
        if ys[-1] + rh < ph:
            ys.append(ph - rh)
        if xs[-1] + rw < pw:
            xs.append(pw - rw)

        # Assemble all patches for this slice into a batch
        patches = []
        coords = []
        for y0 in ys:
            for x0 in xs:
                patch = slc_padded[y0:y0+rh, x0:x0+rw]
                patches.append(patch)
                coords.append((y0, x0))

        # Stack into batch: (N, 1, rh, rw)
        batch = np.stack(patches)[:, np.newaxis].astype(np.float32)
        batch_t = torch.from_numpy(batch).to(device)

        # Inference
        with torch.no_grad():
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=amp_dtype):
                    logits = model.backbone(batch_t)
            else:
                logits = model.backbone(batch_t)
            sig = torch.sigmoid(logits).float().cpu().numpy()  # (N, C, rh, rw)

        # Foreground masking: zero predictions on black padding
        fg = (batch > FG_THRESHOLD).astype(np.float32)  # (N, 1, rh, rw)
        sig = sig * fg

        # Reassemble into full padded slice
        slice_probs = np.zeros((num_classes, ph, pw), dtype=np.float32)
        slice_count = np.zeros((1, ph, pw), dtype=np.float32)

        for (y0, x0), patch_prob in zip(coords, sig):
            slice_probs[:, y0:y0+rh, x0:x0+rw] += patch_prob
            slice_count[:, y0:y0+rh, x0:x0+rw] += 1.0

        # Average overlapping regions
        slice_count = np.maximum(slice_count, 1.0)
        slice_probs /= slice_count

        # Crop back to original slice size
        slice_probs = slice_probs[:, :sh, :sw]

        # Place back into 3D volume
        if axis == 0:
            probs_out[:, si, :, :] = slice_probs
        elif axis == 1:
            probs_out[:, :, si, :] = slice_probs
        else:
            probs_out[:, :, :, si] = slice_probs

    return probs_out


def segment_volume(
    model: torch.nn.Module,
    image_path: str,
    cfg,
    device: torch.device,
    multi_axis: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment a full 3D volume using a 2D model with optional multi-axis averaging.

    Returns:
        probs: (C, D, H, W) float32 averaged sigmoid probabilities.
        raw_data: (D, H, W) original uint8 data.
        affine: (4, 4) NIfTI affine.
    """
    nii = nib.load(image_path)
    raw_data = nii.get_fdata()
    affine = nii.affine

    # Normalize to [0, 1]
    volume = raw_data.astype(np.float32) / 255.0

    roi_size = getattr(cfg, "roi_size_2d", [256, 256])
    num_classes = getattr(cfg, "num_classes", 35)

    # Precision setup
    precision = getattr(cfg, "precision", "bf16" if getattr(cfg, "bf16", True) else "fp32")
    amp_enabled = precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    if multi_axis:
        axes = [0, 1, 2]
        axis_names = ["axial", "coronal", "sagittal"]
    else:
        axes = [0]
        axis_names = ["axial"]

    probs_sum = np.zeros((num_classes,) + volume.shape, dtype=np.float32)
    for ax, ax_name in zip(axes, axis_names):
        print(f"    {ax_name} inference ({volume.shape[ax]} slices)...", end=" ", flush=True)
        t0 = time.time()
        probs_ax = infer_axis(
            model, volume, ax, roi_size, device,
            amp_dtype, amp_enabled, num_classes,
        )
        probs_sum += probs_ax
        print(f"{time.time() - t0:.1f}s")

    probs_avg = probs_sum / len(axes)

    return probs_avg, raw_data.astype(np.uint8), affine


def parse_annotated_classes(ann_str) -> set[int]:
    """Parse annotated_classes string → set of 0-indexed class indices."""
    if isinstance(ann_str, str) and ann_str.strip():
        return {int(x.strip()) for x in ann_str.split(",") if x.strip()}
    elif isinstance(ann_str, (list, tuple)):
        return {int(x) for x in ann_str}
    return set()


def probs_to_multilabel(probs: np.ndarray, threshold: float = 0.5,
                        annotated_classes: set[int] | None = None) -> np.ndarray:
    """Convert (C, D, H, W) probs → (C, D, H, W) binary multi-label masks.

    Only keeps predictions for annotated classes (if provided) to suppress
    spurious activations on classes the model was never supervised on.
    """
    binary = (probs > threshold).astype(np.uint8)
    if annotated_classes is not None:
        mask = np.zeros(probs.shape[0], dtype=np.uint8)
        for c in annotated_classes:
            if 0 <= c < probs.shape[0]:
                mask[c] = 1
        # Zero out un-annotated channels
        binary *= mask[:, None, None, None]
    return binary


def multilabel_to_label_map(binary: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Convert (C, D, H, W) binary → (D, H, W) integer label map.

    For voxels with multiple classes predicted, the class with the highest
    probability wins. Label 0 = background (no class predicted).
    """
    # Mask probabilities to only predicted classes
    masked_probs = probs * binary.astype(np.float32)
    any_pred = binary.any(axis=0)  # (D, H, W)
    label_map = np.zeros(probs.shape[1:], dtype=np.int16)
    label_map[any_pred] = masked_probs.argmax(axis=0)[any_pred] + 1
    return label_map


def compute_dice_scores(gt_labels: np.ndarray, pred_binary: np.ndarray,
                        annotated_classes: set[int] | None = None,
                        num_classes: int = 35) -> dict:
    """Compute per-class Dice: multi-label prediction vs integer GT.

    GT is an integer label map (0=bg, k=class k).
    Pred is (C, D, H, W) binary multi-label masks.
    Only evaluates annotated classes (if provided).
    """
    results = {}
    eval_classes = annotated_classes if annotated_classes else set(range(num_classes))

    # Sub-nuclear class IDs (1-based) whose union == nuc.
    # nuc is no longer stored in single-label NIfTI, so we reconstruct it.
    _NUC_SUB_IDS = [21, 22, 23, 24, 25, 26, 27, 33, 34, 35]  # ne_mem..nucleo

    for c in sorted(eval_classes):
        if c < 0 or c >= num_classes:
            continue

        if c == 13:  # nuc — reconstruct from sub-nuclear classes
            gt_mask = np.zeros(gt_labels.shape, dtype=bool)
            for sid in _NUC_SUB_IDS:
                gt_mask |= (gt_labels == sid)
        else:
            gt_mask = (gt_labels == (c + 1))
        pred_mask = pred_binary[c].astype(bool)
        gt_sum = int(gt_mask.sum())
        pred_sum = int(pred_mask.sum())
        if gt_sum + pred_sum == 0:
            continue  # skip classes absent in both GT and pred
        inter = int((gt_mask & pred_mask).sum())
        dice = float(2 * inter / (gt_sum + pred_sum))
        results[CLASS_NAMES[c]] = {
            "dice": round(dice, 4),
            "gt_voxels": gt_sum,
            "pred_voxels": pred_sum,
        }
    return results


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Segment validation crops into 3D NIfTI volumes using trained 2D models")
    parser.add_argument("--model", type=str, default="",
                        help="Single model (e.g. resnet_r3). Omit for all 4.")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Process only top-K most-annotated crops (0 = all 45).")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary predictions.")
    parser.add_argument("--no-multi-axis", action="store_true",
                        help="Use only axial slices (faster, lower quality).")
    parser.add_argument("--save-probs", action="store_true",
                        help="Also save 35-channel probability NIfTI (large!).")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_ROOT,
                        help=f"Output root directory. Default: {OUTPUT_ROOT}")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    multi_axis = not args.no_multi_axis

    # Select models
    if args.model:
        if args.model not in MODEL_CONFIGS:
            print(f"Unknown model: {args.model}. Choose from {list(MODEL_CONFIGS.keys())}")
            return
        model_names = [args.model]
    else:
        model_names = list(MODEL_CONFIGS.keys())

    # Load datalist and sort val entries by annotation count
    datalist_path = "/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/auto3dseg/nifti_data_v2/datalist.json"
    val_entries = get_sorted_val_entries(datalist_path)

    if args.top_k > 0:
        val_entries = val_entries[:args.top_k]

    print(f"{'='*70}")
    print(f"  2D Model → 3D Volume Segmentation")
    print(f"  Models: {model_names}")
    print(f"  Validation crops: {len(val_entries)} (sorted by annotation count)")
    print(f"  Multi-axis averaging: {multi_axis}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}\n")

    for model_name in model_names:
        print(f"\n{'#'*70}")
        print(f"  MODEL: {model_name}")
        print(f"{'#'*70}")

        try:
            model, cfg = load_model(model_name, device)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            continue

        all_dice = {}  # crop_name → per-class dice

        for vi, entry in enumerate(val_entries):
            crop = crop_name_from_path(entry["image"])
            ann_str = entry.get("annotated_classes", "")
            n_ann = len([x for x in ann_str.split(",") if x.strip()]) if isinstance(ann_str, str) else 0

            out_dir = Path(args.output_dir) / model_name / crop
            pred_path = out_dir / "prediction.nii.gz"

            # Skip if already done
            if pred_path.exists():
                print(f"\n  [{vi+1}/{len(val_entries)}] {crop} (ann={n_ann}) — SKIP (exists)")
                # Load existing dice summary if available
                dice_json = out_dir / "dice_summary.json"
                if dice_json.exists():
                    with open(dice_json) as f:
                        all_dice[crop] = json.load(f)
                continue

            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{vi+1}/{len(val_entries)}] {crop} (ann={n_ann})")
            t0 = time.time()

            # Parse annotation mask for this crop
            ann_classes = parse_annotated_classes(ann_str)

            # Run inference
            probs, raw_data, affine = segment_volume(
                model, entry["image"], cfg, device, multi_axis=multi_axis,
            )

            # Save raw image
            raw_nii = nib.Nifti1Image(raw_data, affine)
            nib.save(raw_nii, str(out_dir / "raw_image.nii.gz"))

            # Save ground truth label map
            gt_nii = nib.load(entry["label"])
            gt_labels = gt_nii.get_fdata().astype(np.int16)
            gt_out = nib.Nifti1Image(gt_labels, affine)
            nib.save(gt_out, str(out_dir / "ground_truth.nii.gz"))

            # Multi-label prediction: threshold + restrict to annotated classes
            pred_binary = probs_to_multilabel(
                probs, threshold=args.threshold,
                annotated_classes=ann_classes if ann_classes else None,
            )

            # Save integer label map (argmax among predicted classes)
            pred_labels = multilabel_to_label_map(pred_binary, probs)
            pred_nii = nib.Nifti1Image(pred_labels, affine)
            nib.save(pred_nii, str(out_dir / "prediction.nii.gz"))

            # Save multi-label 4D volume (D, H, W, C) uint8 — each channel is a class
            # Open in ImageJ → Image → Hyperstack to browse channels
            ml_vol = np.moveaxis(pred_binary, 0, -1)  # (C,D,H,W) → (D,H,W,C)
            ml_nii = nib.Nifti1Image(ml_vol, affine)
            nib.save(ml_nii, str(out_dir / "prediction_ml.nii.gz"))

            # Optional: save full probabilities
            if args.save_probs:
                prob_vol = np.moveaxis(probs, 0, -1).astype(np.float32)
                prob_nii = nib.Nifti1Image(prob_vol, affine)
                nib.save(prob_nii, str(out_dir / "probabilities.nii.gz"))

            # Compute per-class Dice (multi-label: per-channel binary vs GT)
            dice_scores = compute_dice_scores(
                gt_labels, pred_binary,
                annotated_classes=ann_classes if ann_classes else None,
            )
            all_dice[crop] = dice_scores

            # Save per-crop dice summary
            with open(out_dir / "dice_summary.json", "w") as f:
                json.dump(dice_scores, f, indent=2)

            elapsed = time.time() - t0

            # Print summary for this crop
            valid_dice = [v["dice"] for v in dice_scores.values()]
            mean_dice = np.mean(valid_dice) if valid_dice else 0.0

            print(f"    Shape: {raw_data.shape} | Time: {elapsed:.1f}s | "
                  f"Mean Dice: {mean_dice:.4f} ({len(valid_dice)} classes)")

            # Per-class Dice table
            if dice_scores:
                print(f"    {'Class':<15} {'Dice':>6} {'GT vox':>10} {'Pred vox':>10}")
                print(f"    {'─'*45}")
                for name, vals in sorted(dice_scores.items(),
                                          key=lambda x: -x[1]["dice"]):
                    flag = " ◄" if vals["gt_voxels"] > 0 and vals["pred_voxels"] == 0 else ""
                    print(f"    {name:<15} {vals['dice']:>6.4f} "
                          f"{vals['gt_voxels']:>10,} {vals['pred_voxels']:>10,}{flag}")

        # ── Model-level summary ───────────────────────────────────────
        print(f"\n{'='*70}")
        print(f"  MODEL SUMMARY: {model_name}")
        print(f"{'='*70}")

        if all_dice:
            # Aggregate: mean Dice per class across all crops
            class_dice_lists = {}
            for crop, scores in all_dice.items():
                for cls_name, vals in scores.items():
                    if vals["gt_voxels"] > 0:  # only count when GT exists
                        class_dice_lists.setdefault(cls_name, []).append(vals["dice"])

            print(f"\n  Per-class mean Dice (across {len(all_dice)} crops):")
            print(f"  {'Class':<15} {'Mean Dice':>10} {'Crops w/ GT':>12}")
            print(f"  {'─'*40}")
            all_means = []
            for name in CLASS_NAMES:
                if name in class_dice_lists:
                    vals = class_dice_lists[name]
                    m = np.mean(vals)
                    all_means.append(m)
                    print(f"  {name:<15} {m:>10.4f} {len(vals):>12}")

            overall_mean = np.mean(all_means) if all_means else 0.0
            print(f"  {'─'*40}")
            print(f"  {'OVERALL':<15} {overall_mean:>10.4f} "
                  f"{len(class_dice_lists):>12} classes")

            # Save model-level summary
            summary = {
                "model": model_name,
                "overall_mean_dice": round(float(overall_mean), 4),
                "per_class_mean_dice": {
                    name: round(float(np.mean(vals)), 4)
                    for name, vals in class_dice_lists.items()
                },
                "per_crop_dice": all_dice,
            }
            summary_path = Path(args.output_dir) / model_name / "summary.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n  Summary saved: {summary_path}")

        # Clean up model
        del model
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  ✅ All done!")
    print(f"  Output: {args.output_dir}/")
    print(f"\n  To view in ImageJ/Fiji:")
    print(f"    File → Open → raw_image.nii.gz")
    print(f"    File → Open → prediction.nii.gz")
    print(f"    File → Open → ground_truth.nii.gz")
    print(f"    Then: Image → Lookup Tables → Spectrum (for label maps)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
