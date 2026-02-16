"""
Inference pipeline for MONAI CellMap segmentation.

Loads trained checkpoints, runs sliding-window inference with TTA and
per-class ensemble on validation or test volumes.

Features:
- Sliding window inference with Gaussian importance weighting
- Test-time augmentation (8 flip combinations)
- Per-class model ensemble (select best model per class based on val Dice)
- Class-specific sigmoid thresholds
- Post-processing per class (connected components, size filtering)
- Outputs NIfTI predictions for evaluation

Usage:
    # Single model inference
    python inference.py --model segresnet --checkpoint /path/to/best.pth

    # Full ensemble with TTA
    python inference.py --ensemble --tta --split val

    # Custom thresholds
    python inference.py --ensemble --tta --threshold-json thresholds.json

Reference: AGENT_CONTEXT.md §8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add experiment directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "configs"))

from monai.inferers import sliding_window_inference
import monai.transforms as mt
import nibabel as nib

from models.mdl_cellmap import Net
from data.ds_cellmap import (
    IntegerLabelToMultiChanneld,
    ParseAnnotationMaskd,
    load_datalist,
)


# ─── Constants ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
]
NUM_CLASSES = 14

# Default per-class thresholds (start at 0.5, tune on val later)
DEFAULT_THRESHOLDS = {name: 0.5 for name in CLASS_NAMES}

# Default per-class model selection based on Round 1 training results
# (best validation Dice per class — see AGENT_CONTEXT.md §4)
DEFAULT_ENSEMBLE_MAP = {
    "ecs": "swinunetr",
    "pm": "swinunetr",
    "mito_mem": "segresnet",
    "mito_lum": "swinunetr",
    "mito_ribo": "swinunetr",
    "golgi_mem": "segresnet",
    "golgi_lum": "segresnet",
    "ves_mem": "flexunet",
    "ves_lum": "flexunet",
    "endo_mem": "swinunetr",
    "endo_lum": "swinunetr",
    "er_mem": "swinunetr",
    "er_lum": "swinunetr",
    "nuc": "flexunet",
}

# Model checkpoint paths
RUNS_DIR = "/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap"
MODEL_CONFIGS = {
    "segresnet": {
        "config": "cfg_segresnet",
        "checkpoint": f"{RUNS_DIR}/segresnet_ds/checkpoint_best.pth",
    },
    "flexunet": {
        "config": "cfg_flexunet_resnet",
        "checkpoint": f"{RUNS_DIR}/flexunet_resnet34/checkpoint_best.pth",
    },
    "swinunetr": {
        "config": "cfg_swinunetr",
        "checkpoint": f"{RUNS_DIR}/swinunetr/checkpoint_best.pth",
    },
}


# ─── TTA ──────────────────────────────────────────────────────────────────

class TTAPredictor:
    """Test-time augmentation with flips.

    Applies all 8 combinations of flipping along x, y, z axes,
    runs the model on each, flips predictions back, and averages.
    """

    # All 8 combinations of (flip_x, flip_y, flip_z)
    FLIP_COMBOS = [
        (),
        (2,),
        (3,),
        (4,),
        (2, 3),
        (2, 4),
        (3, 4),
        (2, 3, 4),
    ]

    def __init__(self, model: torch.nn.Module, sw_params: dict):
        """
        Args:
            model: The segmentation model.
            sw_params: Dict of params for sliding_window_inference
                       (roi_size, sw_batch_size, overlap, mode, etc.)
                       Should include sw_device (GPU) and device (CPU).
        """
        self.model = model
        self.sw_params = sw_params

    @torch.no_grad()
    def predict(self, volume: torch.Tensor) -> torch.Tensor:
        """Run TTA prediction on a single volume.

        Args:
            volume: (1, 1, D, H, W) input tensor (can be on CPU;
                    sw_device handles moving patches to GPU).

        Returns:
            (1, C, D, H, W) averaged logits on CPU (NOT sigmoid'd).
        """
        accum = None
        n = 0

        for flip_dims in self.FLIP_COMBOS:
            # Flip input
            x = volume
            if flip_dims:
                x = torch.flip(x, dims=flip_dims)

            # Sliding window inference — patches run on GPU (sw_device),
            # output accumulates on CPU (device) via MONAI internals
            logits = sliding_window_inference(
                inputs=x,
                predictor=self._forward_fn,
                **self.sw_params,
            )

            # Flip prediction back (already on CPU from MONAI's device param)
            if flip_dims:
                logits = torch.flip(logits, dims=flip_dims)

            if accum is None:
                accum = logits
            else:
                accum = accum + logits
            n += 1

        return accum / n

    def _forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function for sliding_window_inference.

        Must return logits (not a dict). Uses model.backbone directly
        since Net.forward() expects a batch dict, not raw tensors.
        Handles deep supervision (list output) by taking first element.
        """
        out = self.model.backbone(x)
        if isinstance(out, (list, tuple)):
            return out[0]
        return out


# ─── Model Loading ────────────────────────────────────────────────────────

def load_model(
    model_name: str,
    device: torch.device,
    checkpoint_path: Optional[str] = None,
) -> tuple[torch.nn.Module, object]:
    """Load a trained model from checkpoint.

    Args:
        model_name: One of 'segresnet', 'flexunet', 'swinunetr'.
        device: Device to load model onto.
        checkpoint_path: Override checkpoint path (default: use MODEL_CONFIGS).

    Returns:
        (model, cfg) tuple.
    """
    import importlib

    config_name = MODEL_CONFIGS[model_name]["config"]
    ckpt_path = checkpoint_path or MODEL_CONFIGS[model_name]["checkpoint"]

    # Load config
    cfg = importlib.import_module(config_name).cfg

    # Build model (Net wraps backbone + loss)
    model = Net(cfg)

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model"]

    # Handle DDP state dict (keys prefixed with "module.")
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} from {ckpt_path} (epoch {checkpoint.get('epoch', '?')})")
    return model, cfg


# ─── Pre/Post Processing ─────────────────────────────────────────────────

def load_volume(file_entry: dict) -> tuple[torch.Tensor, np.ndarray, tuple]:
    """Load a NIfTI volume for inference.

    Args:
        file_entry: Dict with 'image' and optionally 'label' keys.

    Returns:
        (image_tensor, affine, original_shape)
        image_tensor: (1, 1, D, H, W) normalized float32 tensor.
    """
    img_path = file_entry["image"]
    nii = nib.load(img_path)
    data = nii.get_fdata().astype(np.float32)
    affine = nii.affine

    # Normalize (same as training: zero-mean, unit-var)
    mean = data.mean()
    std = data.std()
    if std > 0:
        data = (data - mean) / std

    # Convert to tensor: (1, 1, D, H, W)
    tensor = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
    return tensor, affine, data.shape


def load_ground_truth(file_entry: dict, num_classes: int = 14) -> Optional[torch.Tensor]:
    """Load ground truth labels for evaluation.

    Returns:
        (1, C, D, H, W) multi-channel binary tensor, or None.
    """
    label_path = file_entry.get("label")
    if not label_path or not os.path.exists(label_path):
        return None

    nii = nib.load(label_path)
    data = nii.get_fdata().astype(np.int64)

    # Integer label → multi-channel binary
    multi = np.zeros((num_classes,) + data.shape, dtype=np.float32)
    for c in range(num_classes):
        multi[c] = (data == (c + 1)).astype(np.float32)

    return torch.from_numpy(multi).unsqueeze(0)  # (1, C, D, H, W)


def apply_postprocessing(
    pred: np.ndarray,
    class_idx: int,
    min_size: int = 0,
) -> np.ndarray:
    """Apply per-class post-processing to a binary prediction.

    Args:
        pred: (D, H, W) binary numpy array.
        class_idx: Index into CLASS_NAMES.
        min_size: Minimum connected component size (voxels). 0 = no filtering.

    Returns:
        Post-processed binary array.
    """
    if min_size <= 0:
        return pred

    try:
        from scipy import ndimage
    except ImportError:
        return pred

    # Connected component analysis
    labeled, n_components = ndimage.label(pred)
    if n_components == 0:
        return pred

    # Remove small components
    component_sizes = ndimage.sum(pred, labeled, range(1, n_components + 1))
    small_mask = np.zeros_like(pred, dtype=bool)
    for i, size in enumerate(component_sizes):
        if size < min_size:
            small_mask |= (labeled == (i + 1))

    pred[small_mask] = 0
    return pred


# Default minimum component sizes per class (in voxels)
# Larger structures → larger minimum, tiny structures → 0 (no filtering)
MIN_COMPONENT_SIZE = {
    "ecs": 1000,       # large extracellular space
    "pm": 0,           # thin membrane, don't filter
    "mito_mem": 100,
    "mito_lum": 100,
    "mito_ribo": 0,    # very small, don't filter
    "golgi_mem": 100,
    "golgi_lum": 100,
    "ves_mem": 0,       # tiny vesicles
    "ves_lum": 0,
    "endo_mem": 50,
    "endo_lum": 50,
    "er_mem": 50,
    "er_lum": 50,
    "nuc": 5000,        # nuclei are large
}


# ─── Evaluation ───────────────────────────────────────────────────────────

def compute_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Compute Dice score between two binary arrays."""
    intersection = (pred * target).sum()
    return float((2 * intersection + smooth) / (pred.sum() + target.sum() + smooth))


# ─── Main Inference ───────────────────────────────────────────────────────

def run_inference(args):
    n_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {n_gpus}")

    # ── Load thresholds ──
    if args.threshold_json and os.path.exists(args.threshold_json):
        with open(args.threshold_json) as f:
            thresholds = json.load(f)
        print(f"Loaded thresholds from {args.threshold_json}")
    else:
        thresholds = DEFAULT_THRESHOLDS.copy()

    # ── Load ensemble map ──
    if args.ensemble_json and os.path.exists(args.ensemble_json):
        with open(args.ensemble_json) as f:
            ensemble_map = json.load(f)
        print(f"Loaded ensemble map from {args.ensemble_json}")
    else:
        ensemble_map = DEFAULT_ENSEMBLE_MAP.copy()

    # ── Determine which models to load ──
    if args.ensemble:
        model_names = list(set(ensemble_map.values()))
    elif args.model:
        model_names = [args.model]
    else:
        model_names = ["flexunet"]  # default to best single model

    # ── Load each model onto its own GPU ──
    models = {}
    configs = {}
    model_devices = {}  # track which GPU each model is on
    for i, name in enumerate(model_names):
        gpu_id = i % max(n_gpus, 1)
        dev = torch.device(f"cuda:{gpu_id}" if n_gpus > 0 else "cpu")
        ckpt = args.checkpoint if (args.model == name and args.checkpoint) else None
        model, cfg = load_model(name, dev, checkpoint_path=ckpt)
        models[name] = model
        configs[name] = cfg
        model_devices[name] = dev
        print(f"  {name} → {dev}")

    # ── Sliding window parameters ──
    # sw_device = GPU for patch computation, device = CPU for output accumulation.
    # This lets us handle 800³ volumes (29 GB output) without GPU OOM.
    sw_roi = [int(x) for x in args.sw_roi_size.split(",")]
    print(f"Sliding window: roi={sw_roi}, overlap={args.overlap}, batch={args.sw_batch_size}")

    def make_sw_params(gpu_device):
        return {
            "roi_size": sw_roi,
            "sw_batch_size": args.sw_batch_size,
            "overlap": args.overlap,
            "mode": "gaussian",
            "padding_mode": "replicate",
            "sw_device": gpu_device,  # patches computed on GPU
            "device": "cpu",          # output accumulated on CPU
        }

    # ── Build TTA predictors if enabled ──
    if args.tta:
        predictors = {
            name: TTAPredictor(model, make_sw_params(model_devices[name]))
            for name, model in models.items()
        }
        print(f"TTA enabled: 8 flip combinations per model")
    else:
        predictors = None

    # ── Load data list ──
    # Use any config's datalist (they're all the same)
    any_cfg = list(configs.values())[0]
    train_files, val_files = load_datalist(any_cfg)

    if args.split == "val":
        file_list = val_files
        print(f"Running inference on {len(val_files)} validation volumes")
    elif args.split == "train":
        file_list = train_files
        print(f"Running inference on {len(train_files)} training volumes")
    else:
        # Could be a path to a custom file list
        raise ValueError(f"Unknown split: {args.split}")

    # Limit volumes if specified
    if args.max_volumes > 0:
        file_list = file_list[:args.max_volumes]
        print(f"Limited to {len(file_list)} volumes")

    # ── Output directory ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {output_dir}")

    # ── Run inference ──
    all_dice = {name: [] for name in CLASS_NAMES}
    all_mean_dice = []

    for vol_idx, file_entry in enumerate(tqdm(file_list, desc="Volumes")):
        vol_name = Path(file_entry["image"]).stem.replace("_0000", "")
        t0 = time.time()

        # Load volume — stays on CPU; MONAI's sw_device pulls patches to GPU
        volume, affine, orig_shape = load_volume(file_entry)
        print(f"\n[{vol_idx+1}/{len(file_list)}] {vol_name}: shape={orig_shape}")

        # ── Get logits from each model (parallel across GPUs) ──
        def _run_model(name):
            model = models[name]
            dev = model_devices[name]
            sw_p = make_sw_params(dev)
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                if args.tta:
                    return name, predictors[name].predict(volume)
                else:
                    def _fwd(x, _backbone=model.backbone):
                        out = _backbone(x)
                        if isinstance(out, (list, tuple)):
                            return out[0]
                        return out
                    return name, sliding_window_inference(
                        inputs=volume,
                        predictor=_fwd,
                        **sw_p,
                    )

        model_logits = {}  # all on CPU already (device="cpu" in sw_params)
        if len(model_names) > 1 and n_gpus > 1:
            # Run models in parallel — each on its own GPU
            with ThreadPoolExecutor(max_workers=len(model_names)) as pool:
                futures = [pool.submit(_run_model, name) for name in model_names]
                for fut in as_completed(futures):
                    name, logits = fut.result()
                    model_logits[name] = logits
                    print(f"    {name} done")
        else:
            # Sequential fallback
            for name in model_names:
                _, logits = _run_model(name)
                model_logits[name] = logits
                print(f"    {name} done")

        # ── Per-class ensemble (on CPU) ──
        if args.ensemble:
            # Build final prediction by selecting best model per class
            final_logits = torch.zeros_like(list(model_logits.values())[0])
            for c, class_name in enumerate(CLASS_NAMES):
                best_model = ensemble_map.get(class_name, model_names[0])
                if best_model in model_logits:
                    final_logits[:, c] = model_logits[best_model][:, c]
                else:
                    # Fallback: average all available models
                    for ml in model_logits.values():
                        final_logits[:, c] += ml[:, c]
                    final_logits[:, c] /= len(model_logits)
        else:
            final_logits = list(model_logits.values())[0]

        # Free model logits
        del model_logits

        # ── Sigmoid + threshold (on CPU) ──
        probs = torch.sigmoid(final_logits).numpy()[0]  # (C, D, H, W)
        del final_logits

        predictions = np.zeros_like(probs, dtype=np.uint8)
        for c, class_name in enumerate(CLASS_NAMES):
            thresh = thresholds.get(class_name, 0.5)
            binary = (probs[c] > thresh).astype(np.uint8)

            # Post-processing
            min_size = MIN_COMPONENT_SIZE.get(class_name, 0)
            if min_size > 0 and args.postprocess:
                binary = apply_postprocessing(binary, c, min_size=min_size)

            predictions[c] = binary

        # ── Save predictions ──
        vol_out_dir = output_dir / vol_name
        vol_out_dir.mkdir(parents=True, exist_ok=True)

        # Save as multi-channel NIfTI
        # Also save individual class NIfTIs for inspection
        combined = nib.Nifti1Image(predictions.astype(np.float32), affine)
        nib.save(combined, str(vol_out_dir / "prediction_multichannel.nii.gz"))

        # Save probability maps (for threshold tuning later)
        if args.save_probs:
            prob_nii = nib.Nifti1Image(probs, affine)
            nib.save(prob_nii, str(vol_out_dir / "probabilities.nii.gz"))

        # ── Evaluate against ground truth (if available) ──
        gt = load_ground_truth(file_entry)
        if gt is not None:
            gt_np = gt.numpy()[0]  # (C, D, H, W)

            # Parse annotation mask
            ann_str = file_entry.get("annotated_classes", "")
            annotated = set()
            if ann_str:
                for idx_str in str(ann_str).split(","):
                    idx_str = idx_str.strip()
                    if idx_str:
                        annotated.add(int(idx_str))

            vol_dice = {}
            for c, class_name in enumerate(CLASS_NAMES):
                if annotated and c not in annotated:
                    continue  # skip unannotated classes
                d = compute_dice(predictions[c], gt_np[c])
                vol_dice[class_name] = d
                all_dice[class_name].append(d)

            if vol_dice:
                mean_d = np.mean(list(vol_dice.values()))
                all_mean_dice.append(mean_d)
                print(f"  Dice: {mean_d:.4f} | " +
                      " ".join(f"{k}:{v:.3f}" for k, v in sorted(vol_dice.items())))

            # Save per-volume results
            with open(vol_out_dir / "dice_scores.json", "w") as f:
                json.dump(vol_dice, f, indent=2)

        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

        # Free memory
        del volume
        for dev in set(model_devices.values()):
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("INFERENCE SUMMARY")
    print("=" * 60)

    if all_mean_dice:
        print(f"\nOverall Mean Dice: {np.mean(all_mean_dice):.4f} ± {np.std(all_mean_dice):.4f}")
        print(f"\nPer-Class Dice (mean ± std across volumes):")
        summary = {}
        for class_name in CLASS_NAMES:
            scores = all_dice[class_name]
            if scores:
                mean = np.mean(scores)
                std = np.std(scores)
                print(f"  {class_name:12s}: {mean:.4f} ± {std:.4f}  (n={len(scores)})")
                summary[class_name] = {"mean": float(mean), "std": float(std), "n": len(scores)}

        # Save summary
        summary_path = output_dir / "evaluation_summary.json"
        with open(summary_path, "w") as f:
            json.dump({
                "overall_mean_dice": float(np.mean(all_mean_dice)),
                "overall_std_dice": float(np.std(all_mean_dice)),
                "per_class": summary,
                "settings": {
                    "models": model_names,
                    "ensemble": args.ensemble,
                    "tta": args.tta,
                    "postprocess": args.postprocess,
                    "sw_roi_size": sw_roi,
                    "overlap": args.overlap,
                    "thresholds": thresholds,
                },
            }, f, indent=2)
        print(f"\nSummary saved to: {summary_path}")

    print(f"\nPredictions saved to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="MONAI CellMap Inference")

    # Model selection
    parser.add_argument("--model", type=str, default="flexunet",
                        choices=["segresnet", "flexunet", "swinunetr"],
                        help="Single model to use (ignored if --ensemble)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path for --model")
    parser.add_argument("--ensemble", action="store_true",
                        help="Use per-class ensemble (all 3 models)")

    # TTA and post-processing
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (8 flips)")
    parser.add_argument("--postprocess", action="store_true",
                        help="Enable per-class post-processing")

    # Sliding window
    parser.add_argument("--sw-roi-size", type=str, default="128,128,128",
                        help="Sliding window ROI size (comma-separated)")
    parser.add_argument("--sw-batch-size", type=int, default=2,
                        help="Sliding window batch size")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap fraction")

    # Data
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "train"],
                        help="Data split to run on")
    parser.add_argument("--max-volumes", type=int, default=0,
                        help="Max volumes to process (0 = all)")

    # Output
    parser.add_argument("--output-dir", type=str,
                        default=f"{RUNS_DIR}/predictions",
                        help="Output directory for predictions")
    parser.add_argument("--save-probs", action="store_true",
                        help="Also save probability maps (for threshold tuning)")

    # Config overrides
    parser.add_argument("--threshold-json", type=str, default=None,
                        help="JSON file with per-class thresholds")
    parser.add_argument("--ensemble-json", type=str, default=None,
                        help="JSON file mapping class → model name")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)
