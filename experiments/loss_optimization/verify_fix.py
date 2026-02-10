#!/usr/bin/env python3
"""Verify the normalization fix: both train and val should be in [0, 1]."""
import os, sys
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['SINGLE_GPU_MODE'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.dirname(__file__))

import torch
torch.set_num_threads(4)

import logging
logging.disable(logging.INFO)

from config_shenron import QUICK_TEST_CLASSES
from train_local import create_dataloaders

classes = QUICK_TEST_CLASSES

print("Creating dataloaders with fixed normalization...")
train_loader, val_loader = create_dataloaders(
    classes=classes, batch_size=4, iterations_per_epoch=10, input_shape=(1, 256, 256)
)

print("\n=== TRAINING DATA ===")
for i, batch in enumerate(train_loader):
    if i >= 3: break
    inp = batch['input']
    tgt = batch['output']
    print(f"  Batch {i}: input range=[{inp.min():.4f}, {inp.max():.4f}], "
          f"input mean={inp.mean():.4f}, input dtype={inp.dtype}")

print("\n=== VALIDATION DATA ===")
val_sums = []
for i in range(3):
    batch = next(iter(val_loader))
    inp = batch['input']
    tgt = batch['output']
    val_sums.append(inp.sum().item())
    print(f"  Iter {i}: input range=[{inp.min():.4f}, {inp.max():.4f}], "
          f"input mean={inp.mean():.4f}, input_sum={inp.sum().item():.2f}")

if len(set([round(s, 2) for s in val_sums])) == 1:
    print("\n  WARNING: Val loader still returns same data every time!")
else:
    print("\n  OK: Val loader returns different data each time.")

print("\nDone!")
