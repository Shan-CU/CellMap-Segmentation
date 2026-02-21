"""
Model wrappers for CellMap training.

Wraps both CSC 2D models and MONAI 3D models with a unified interface:
- Forward returns (logits, annotation_mask, foreground_mask)
- Handles annotation masking from cellmap-data's NaN convention
- Computes foreground mask from EM input

These wrappers are used by the training loop in training/train.py.
The loss function handles partial annotation via set_annotation_mask().
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .model_zoo import build_model, MODEL_REGISTRY

__all__ = [
    "CellMapModelWrapper",
    "build_model",
    "MODEL_REGISTRY",
]
