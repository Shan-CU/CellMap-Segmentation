"""
Quick validation evaluation — run each model on the validation set
and report per-class Dice, then compute the per-class ensemble score.

This is a lightweight version of inference.py that uses the same
crop-based evaluation as training (no sliding window) for speed.
Use this for fast model comparison; use inference.py for final predictions.

Usage:
    python evaluate_ensemble.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "configs"))

from data.ds_cellmap import load_datalist, CellMapDataset, flat_collate_fn
from models.mdl_cellmap import Net
from utils import compute_per_channel_dice

CLASS_NAMES = [
    # ── Original 14 from Round 1 ──
    "ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
    "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
    "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc",
    # ── New for Round 2 ──
    "lyso_mem", "lyso_lum", "ld_mem", "ld_lum",
    "eres_mem", "eres_lum", "ne_mem", "ne_lum",
    "np_out", "np_in", "hchrom", "echrom", "nucpl",
    "mt_out", "cyto", "mt_in", "perox_mem", "perox_lum",
    "nhchrom", "nechrom", "nucleo",
]
RUNS_DIR = "/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap"

MODELS = {
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


def evaluate_model(model_name: str, device: torch.device) -> dict:
    """Evaluate a single model on validation set, return per-class Dice."""
    info = MODELS[model_name]
    cfg = importlib.import_module(info["config"]).cfg
    _, val_files = load_datalist(cfg)

    # Build model and load checkpoint
    model = Net(cfg)
    ckpt = torch.load(info["checkpoint"], map_location=device, weights_only=False)
    state_dict = ckpt["model"]
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device).eval()
    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name} (epoch {ckpt.get('epoch', '?')})")
    print(f"Checkpoint best metric: {ckpt.get('best_metric', '?'):.4f}")
    print(f"{'='*50}")

    # Build validation dataset
    val_ds = CellMapDataset(val_files, cfg, mode="val")
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        collate_fn=flat_collate_fn, num_workers=2,
    )

    # Run validation
    num_classes = len(CLASS_NAMES)
    dice_sum = torch.zeros(num_classes, device=device)
    valid_sum = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=model_name):
            for k in ["input", "target", "annotation_mask"]:
                if k in batch and isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(batch)
                logits = outputs["logits"]

            mask = batch.get("annotation_mask", None)
            dice_pc, valid_pc = compute_per_channel_dice(
                logits, batch["target"], mask=mask, sigmoid=True,
            )
            dice_sum += dice_pc * valid_pc
            valid_sum += valid_pc

    per_channel = dice_sum / valid_sum.clamp(min=1.0)
    dice_dict = {}
    for i, name in enumerate(CLASS_NAMES):
        dice_dict[name] = per_channel[i].item()

    annotated = (valid_sum > 0).float()
    mean_dice = (per_channel * annotated).sum() / annotated.sum().clamp(min=1)

    print(f"\nMean Dice: {mean_dice.item():.4f}")
    for name, val in dice_dict.items():
        print(f"  {name:12s}: {val:.4f}")

    return dice_dict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = {}
    for model_name in MODELS:
        all_results[model_name] = evaluate_model(model_name, device)

    # Per-class ensemble: pick best model per class
    print(f"\n{'='*60}")
    print("PER-CLASS ENSEMBLE")
    print(f"{'='*60}")

    ensemble_dice = {}
    ensemble_map = {}
    for class_name in CLASS_NAMES:
        best_model = max(
            all_results.keys(),
            key=lambda m: all_results[m].get(class_name, 0),
        )
        best_dice = all_results[best_model][class_name]
        ensemble_dice[class_name] = best_dice
        ensemble_map[class_name] = best_model
        print(f"  {class_name:12s}: {best_dice:.4f}  ← {best_model}")

    mean_ensemble = np.mean(list(ensemble_dice.values()))
    print(f"\nEnsemble Mean Dice: {mean_ensemble:.4f}")

    # Save results
    output = {
        "per_model": all_results,
        "ensemble_dice": ensemble_dice,
        "ensemble_map": ensemble_map,
        "ensemble_mean_dice": float(mean_ensemble),
    }
    output_path = f"{RUNS_DIR}/ensemble_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Also save ensemble map for use by inference.py
    map_path = f"{RUNS_DIR}/ensemble_map.json"
    with open(map_path, "w") as f:
        json.dump(ensemble_map, f, indent=2)
    print(f"Ensemble map saved to: {map_path}")


if __name__ == "__main__":
    main()
