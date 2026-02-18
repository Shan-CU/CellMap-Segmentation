#!/usr/bin/env python3
"""
Visualize 3D model predictions on validation crops.

Runs sliding window inference on a NIfTI validation volume, saves:
  - raw_image.nii.gz         : original EM volume
  - ground_truth.nii.gz      : integer-labeled ground truth (0=bg, 1-35=classes)
  - prediction.nii.gz        : integer-labeled predictions (0=bg, 1-35=classes)
  - probabilities.nii.gz     : multi-channel sigmoid probabilities (optional, large)

These can be opened in 3D Slicer / ITK-SNAP for visualization.

Usage:
    python visualize_prediction.py --model swinunetr_r2 [--crop jrc_hela-2_crop14] [--save-probs]

Requirements: run from experiments/monai_cellmap/ or set PYTHONPATH.
"""

import argparse
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference

# Ensure configs/ and models/ are importable
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "configs"))

# ── Model name → config module mapping ──
MODEL_CONFIGS = {
    "swinunetr_r2": "cfg_swinunetr",
    "flexunet_resnet34_r2": "cfg_flexunet_resnet",
    "segresnet_ds_r2": "cfg_segresnet",
    "segresnet_wide_r2": "cfg_segresnet_wide",
}

CLASS_NAMES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
    "lyso_mem", "lyso_lum", "ld_mem", "ld_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl",
    "mt_out", "cyto", "mt_in", "perox_mem", "perox_lum",
    "nhchrom", "nechrom", "nucleo",
]


def load_model(model_name: str, device: str = "cuda"):
    """Load a trained model from its best checkpoint."""
    config_module = MODEL_CONFIGS[model_name]
    cfg = importlib.import_module(config_module).cfg

    from models.mdl_cellmap import Net
    model = Net(cfg)

    ckpt_path = Path(cfg.output_dir) / "checkpoint_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No best checkpoint at {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    # Handle DDP prefix
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)

    epoch = ckpt.get("epoch", "?")
    best_metric = ckpt.get("best_metric", "?")
    print(f"  Epoch: {epoch}, Best val Dice: {best_metric}")

    model = model.to(device).eval()
    return model, cfg


def find_validation_crop(crop_name: str, datalist_path: str):
    """Find a validation crop's image and label paths from the datalist."""
    with open(datalist_path) as f:
        datalist = json.load(f)

    val_entries = datalist.get("validation", datalist.get("val", []))

    if crop_name is None:
        # Pick first validation crop
        entry = val_entries[0]
        crop_name = Path(entry["image"]).stem.replace("_0000", "")
        print(f"Auto-selected validation crop: {crop_name}")
    else:
        entry = None
        for e in val_entries:
            if crop_name in e["image"]:
                entry = e
                break
        if entry is None:
            print(f"Available validation crops:")
            for e in val_entries:
                print(f"  {Path(e['image']).stem.replace('_0000', '')}")
            raise ValueError(f"Crop '{crop_name}' not found in validation set")

    return entry, crop_name


def run_inference(model, cfg, image_path: str, device: str = "cuda"):
    """Run sliding window inference on a NIfTI volume."""
    print(f"Loading image: {image_path}")
    nii = nib.load(image_path)
    data = nii.get_fdata().astype(np.float32)
    print(f"  Volume shape: {data.shape}")

    # Normalize (zero mean, unit variance)
    data = (data - data.mean()) / (data.std() + 1e-8)
    volume = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # Sliding window inference on backbone directly
    def fwd(x):
        out = model.backbone(x)
        return out[0] if isinstance(out, (list, tuple)) else out

    roi_size = cfg.roi_size
    print(f"  Running sliding window inference (roi={roi_size}, overlap=0.5)...")

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = sliding_window_inference(
            inputs=volume,
            roi_size=roi_size,
            sw_batch_size=2,
            predictor=fwd,
            overlap=0.5,
            mode="gaussian",
            padding_mode="replicate",
            sw_device=device,
            device="cpu",
        )

    probs = torch.sigmoid(logits).numpy()[0]  # (C, D, H, W)
    print(f"  Output shape: {probs.shape}")
    return probs, nii.affine


def probs_to_label_map(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Convert multi-channel probabilities to a single integer label map.
    For overlapping predictions, highest probability wins.
    Returns shape (D, H, W) with values 0 (background) to num_classes.
    """
    binary = probs > threshold
    # If no class predicted, label = 0 (background)
    # If multiple classes predicted, take argmax of probabilities
    label_map = np.zeros(probs.shape[1:], dtype=np.int16)

    # For each voxel, assign the class with highest probability (if above threshold)
    max_prob = probs.max(axis=0)
    has_prediction = max_prob > threshold
    # argmax gives 0-indexed class, add 1 for 1-indexed labels
    label_map[has_prediction] = probs.argmax(axis=0)[has_prediction] + 1

    return label_map


def load_ground_truth(label_path: str) -> np.ndarray:
    """Load ground truth NIfTI — already an integer label map."""
    nii = nib.load(label_path)
    return nii.get_fdata().astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D model predictions")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                        help="Model name")
    parser.add_argument("--crop", default=None,
                        help="Crop name (e.g., jrc_hela-2_crop14). Default: first val crop")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory. Default: experiments/monai_cellmap/visualizations/")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Sigmoid threshold for binary predictions")
    parser.add_argument("--save-probs", action="store_true",
                        help="Also save full probability volume (large!)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load model
    model, cfg = load_model(args.model, device=args.device)

    # Find validation crop
    entry, crop_name = find_validation_crop(args.crop, cfg.datalist)

    # Setup output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = SCRIPT_DIR / "visualizations" / f"{args.model}_{crop_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Run inference
    probs, affine = run_inference(model, cfg, entry["image"], device=args.device)

    # Save raw image (copy for convenience)
    raw_nii = nib.load(entry["image"])
    nib.save(raw_nii, str(out_dir / "raw_image.nii.gz"))
    print(f"Saved: raw_image.nii.gz")

    # Save ground truth
    gt = load_ground_truth(entry["label"])
    gt_nii = nib.Nifti1Image(gt, affine)
    nib.save(gt_nii, str(out_dir / "ground_truth.nii.gz"))
    print(f"Saved: ground_truth.nii.gz")

    # Save prediction label map
    pred_labels = probs_to_label_map(probs, threshold=args.threshold)
    pred_nii = nib.Nifti1Image(pred_labels, affine)
    nib.save(pred_nii, str(out_dir / "prediction.nii.gz"))
    print(f"Saved: prediction.nii.gz")

    # Print per-class summary
    print(f"\n{'Class':<15} {'GT voxels':>12} {'Pred voxels':>12} {'Dice':>8}")
    print("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        gt_mask = (gt == (i + 1))
        pred_mask = (pred_labels == (i + 1))
        gt_count = gt_mask.sum()
        pred_count = pred_mask.sum()
        if gt_count + pred_count > 0:
            intersection = (gt_mask & pred_mask).sum()
            dice = 2 * intersection / (gt_count + pred_count)
        else:
            dice = float('nan')
        if gt_count > 0 or pred_count > 0:
            print(f"{name:<15} {gt_count:>12,} {pred_count:>12,} {dice:>8.4f}")

    # Optionally save probabilities
    if args.save_probs:
        prob_nii = nib.Nifti1Image(probs.astype(np.float32), affine)
        nib.save(prob_nii, str(out_dir / "probabilities.nii.gz"))
        print(f"Saved: probabilities.nii.gz")

    print(f"\n✅ Done! Open in 3D Slicer:")
    print(f"   1. File → Add Data → {out_dir / 'raw_image.nii.gz'}")
    print(f"   2. File → Add Data → {out_dir / 'prediction.nii.gz'} (as LabelMap)")
    print(f"   3. File → Add Data → {out_dir / 'ground_truth.nii.gz'} (as LabelMap)")


if __name__ == "__main__":
    main()
