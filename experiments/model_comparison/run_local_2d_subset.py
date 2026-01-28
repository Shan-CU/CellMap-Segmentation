#!/usr/bin/env python3
"""Run all 2D models locally on 2x RTX 3090 with a curated 5-class subset.

This script wraps the existing `train_comparison.py` to run the four 2D models
(`unet`, `resnet`, `swin`, `vit`) with DDP (torchrun) using GPU-friendly
hyperparameters for 24GB GPUs. It selects 5 classes by coverage / structure:

- `nuc`       : abundant / large structure (group)
- `mito_mem` : moderate availability, mitochondria membrane
- `mito_ribo`: rare class (very few pixels)
- `pm`        : plasma membrane (thin boundary)
- `ves`       : vesicle (small circular structures)

The script:
 - creates a temporary config override file listing the classes
 - launches each model with `torchrun --nproc_per_node=2` so DDP runs on both GPUs
 - sets batch sizes and debug/iteration counts suitable for quick local runs
 - enables `--save_features` and `--vis_every` to capture feature visualizations

Usage:
    python3 run_local_2d_subset.py --outdir /path/to/results --quick

Notes:
 - Requires the repository root as current working dir or will adjust paths.
 - Uses `train_comparison.py` in the same experiments folder.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parent
TRAIN_SCRIPT = ROOT / "train_comparison.py"


DEFAULT_CLASSES = [
    'nuc',        # abundant
    'mito_mem',   # moderate
    'mito_ribo',  # rare
    'pm',         # thin boundary / different structure
    'ves',        # small circular structures
]


def write_temp_classes_file(classes, outdir: Path):
    cfg = {
        'CLASSES': classes
    }
    path = outdir / 'classes_override.json'
    with open(path, 'w') as f:
        json.dump(cfg, f)
    return path


def build_cmd(model_name: str, classes_file: Path, epochs: int, iterations: int, batch_size: int, vis_every: int, save_features: bool):
    cmd = [
        'torchrun',
        '--nproc_per_node=2',
        '--standalone',
        str(TRAIN_SCRIPT),
        '--model', model_name,
        '--dim', '2d',
        '--classes_file', str(classes_file),
        '--epochs', str(epochs),
        '--iterations_per_epoch', str(iterations),
        '--batch_size', str(batch_size),
        '--vis_every', str(vis_every),
        '--save_features',
    ]
    if not save_features:
        # train_comparison.py accepts --no-save_features? If not, we still keep explicit flag
        pass
    return cmd


def run_models(args):
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    classes_file = write_temp_classes_file(args.classes or DEFAULT_CLASSES, outdir)

    # Choose conservative batch sizes for 24GB RTX 3090 per-GPU memory and 2 GPUs
    # We'll pass per-process batch size; torchrun will spawn 2 processes
    batch_sizes = {
        'unet': 16,
        'resnet': 12,
        'swin': 8,
        # vit is memory-hungry; use 1 as a safe default for 24GB GPUs
        'vit': 1,
    }

    # For quick local tests use fewer epochs/iterations unless user requests full
    if args.quick:
        epochs = 3
        iterations = 20
        vis_every = 1
    else:
        epochs = 25
        iterations = 100
        vis_every = 5

    models = ['unet', 'resnet', 'swin', 'vit']

    env = os.environ.copy()
    env.update({
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:128',
        'PYTHONUNBUFFERED': '1',
    })

    for model in models:
        bs = batch_sizes.get(model, 4)
        cmd = build_cmd(model, classes_file, epochs, iterations, bs, vis_every, save_features=True)
        print('\nRunning:', ' '.join(shlex.quote(c) for c in cmd))
        # Run and stream output
        process = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
        ret = process.wait()
        if ret != 0:
            print(f"Model {model} exited with code {ret}; continuing to next model.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='local_runs', help='Directory to save class override and outputs')
    p.add_argument('--quick', action='store_true', help='Use short runs for testing')
    p.add_argument('--classes', type=str, nargs='*', help='Optional override classes list')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_models(args)
