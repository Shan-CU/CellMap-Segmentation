#!/usr/bin/env python3
"""Run longer local DDP experiments for model comparison.

This wrapper launches DDP runs (torchrun) for multiple models with
user-configurable total epochs and checkpoint frequency. It is intended
for longer validation runs (not just quick smoke tests) so you can
inspect loss curves and checkpoints in TensorBoard and the checkpoints
folder.

Usage examples:

# Run all 2D models for 100 epochs, checkpoint every 10 epochs
python3 run_local_ddp_long.py --outdir local_long_runs --epochs 100 --checkpoint_every 10

# Run a single model (vit) for 50 epochs
python3 run_local_ddp_long.py --models vit --epochs 50 --checkpoint_every 5

Notes:
- Uses train_comparison.py CLI options (passes --epochs and --checkpoint_every)
- TensorBoard logs are written to the experiment TENSORBOARD_PATH by train_comparison
- Checkpoints are saved to CHECKPOINTS_PATH by train_comparison
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
    'nuc', 'mito_mem', 'mito_ribo', 'pm', 'ves'
]

DEFAULT_BATCH_SIZES = {
    'unet': 16,
    'resnet': 16,
    'swin': 16,
    'vit': 16,   # memory-hungry
}

# Map epochs -> sensible checkpoint frequency if user doesn't provide one
def default_checkpoint_every(epochs: int) -> int:
    if epochs <= 10:
        return 1
    if epochs <= 50:
        return 5
    if epochs <= 200:
        return 10
    return 20


def write_classes_file(outdir: Path, classes):
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / 'classes_override.json'
    with open(path, 'w') as f:
        json.dump({'CLASSES': classes}, f)
    return path


def build_cmd(model, classes_file: Path, epochs: int, iterations: int, batch_size: int, checkpoint_every: int, save_features: bool, dim='2d'):
    cmd = [
        'torchrun',
        '--nproc_per_node=2',
        '--standalone',
        str(TRAIN_SCRIPT),
        '--model', model,
        '--dim', dim,
        '--epochs', str(epochs),
        '--iterations_per_epoch', str(iterations),
        '--batch_size', str(batch_size),
        '--checkpoint_every', str(checkpoint_every),
    ]
    if save_features:
        cmd.append('--save_features')
    return cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='local_long_runs')
    p.add_argument('--models', nargs='+', default=['unet','resnet','swin','vit'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--iterations', type=int, default=200, help='Iterations per epoch')
    p.add_argument('--batch_sizes', nargs='*', help='Optional per-model batch sizes like unet=16 vit=1')
    p.add_argument('--checkpoint_every', type=int, default=None, help='Override checkpoint frequency')
    p.add_argument('--classes', nargs='*', help='Optional class list override')
    p.add_argument('--save_features', action='store_true')
    p.add_argument('--dim', choices=['2d','3d'], default='2d')
    args = p.parse_args()

    outdir = Path(args.outdir).resolve()
    classes_file = write_classes_file(outdir, args.classes or DEFAULT_CLASSES)

    # Parse batch size overrides
    batch_sizes = DEFAULT_BATCH_SIZES.copy()
    if args.batch_sizes:
        for entry in args.batch_sizes:
            if '=' in entry:
                k, v = entry.split('=', 1)
                batch_sizes[k.strip()] = int(v)

    # checkpoint frequency
    checkpoint_every = args.checkpoint_every or default_checkpoint_every(args.epochs)

    env = os.environ.copy()
    env.update({
        'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True,max_split_size_mb:128',
        'PYTHONUNBUFFERED': '1',
    })

    for model in args.models:
        bs = batch_sizes.get(model, 4)
        cmd = build_cmd(model, classes_file, args.epochs, args.iterations, bs, checkpoint_every, args.save_features, dim=args.dim)
        print('\nRunning:', ' '.join(shlex.quote(c) for c in cmd))
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env)
        ret = proc.wait()
        if ret != 0:
            print(f"Model {model} exited with code {ret}; continuing to next model.")


if __name__ == '__main__':
    main()
