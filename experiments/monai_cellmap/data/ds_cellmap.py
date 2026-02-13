"""
CellMap NIfTI Dataset for MONAI-based training.

Loads NIfTI images + labels from datalist.json, converts integer labels (0-14)
to 14-channel binary masks, parses partial annotation masks, and provides
random cropping + augmentation for training.

Key design: "crop-first" on-the-fly loading
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CellMap volumes range from a few MB to 84 GB when expanded to float32 multi-
channel.  Caching full volumes (PersistentDataset / CacheDataset) is
infeasible — even a single 84 GB .pt file exceeds what PersistentDataset can
handle efficiently, and building the cache across 231 volumes would need
>1 TB of disk + hundreds of GB of concurrent worker RAM.

Instead we use a plain MONAI ``Dataset`` with an all-in-one transform pipeline
that does:

    Load NIfTI  →  Random Crop (at integer-label resolution)
                →  Multi-channel expand (only on the small crop)
                →  Normalize  →  Augment

This means:
- Peak RAM per DataLoader worker ≈ 1 decompressed NIfTI volume (image + int
  label) — typically a few hundred MB, the 8 GB giant only briefly.
- The 14-channel float32 expansion happens *after* cropping → 14 × 96³ × 4 B
  ≈ 50 MB instead of 14 × full-volume × 4 B ≈ 84 GB.
- No disk cache, no cross-rank duplication, no first-epoch warmup.
- DataLoader ``num_workers`` controls I/O parallelism.  Use 2 to keep peak RAM
  ≤ 2 concurrent decompressed volumes ≈ 20 GB.

Informed by MONAI GitHub issues #3843, #5930, #6585, #1589, #3116.

Reference: IMPLEMENTATION_SPEC.md §5.5
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

import monai.data as md
import monai.transforms as mt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def batch_to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensors in a batch dict to device."""
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def collate_fn(batch: list[dict]) -> dict:
    """Stack batch items into tensors.

    Each item is a dict with keys 'input', 'target', 'annotation_mask'.
    """
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out


# Train and val use the same collate
tr_collate_fn = collate_fn
val_collate_fn = collate_fn


# ---------------------------------------------------------------------------
# Custom MONAI Transforms
# ---------------------------------------------------------------------------

class IntegerLabelToMultiChanneld(mt.MapTransform):
    """Convert integer label (1, Z, Y, X) with values 0-14 to multi-channel
    binary mask (C, Z, Y, X).

    Value 0 = background/unannotated (ignored).
    Values 1..num_classes map to channels 0..num_classes-1.
    """

    def __init__(self, keys="label", num_classes: int = 14):
        super().__init__(keys)
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]  # shape: (1, Z, Y, X) integer tensor or ndarray

            # Convert to numpy for efficient one-hot
            if isinstance(label, torch.Tensor):
                label_np = label.numpy()
            else:
                label_np = np.asarray(label)

            # Squeeze channel dim: (1, Z, Y, X) -> (Z, Y, X)
            if label_np.ndim == 4 and label_np.shape[0] == 1:
                label_np = label_np[0]

            # Build multi-channel binary: channel c = (label == c+1)
            multi = np.zeros(
                (self.num_classes,) + label_np.shape, dtype=np.float32
            )
            for c in range(self.num_classes):
                multi[c] = (label_np == (c + 1)).astype(np.float32)

            d[key] = torch.from_numpy(multi)
        return d


class ParseAnnotationMaskd(mt.Transform):
    """Parse 'annotated_classes' string into a float tensor of shape (C,).

    Handles:
    - Normal: "0,1,2,3,7,8" → [1,1,1,1,0,0,0,1,1,0,0,0,0,0]
    - Empty string or missing key → all-zero mask (§11.2)
    - MONAI path mangling: os.path.basename strips prefixed basedir
    """

    def __init__(
        self,
        source_key: str = "annotated_classes",
        mask_key: str = "annotation_mask",
        num_classes: int = 14,
    ):
        super().__init__()
        self.source_key = source_key
        self.mask_key = mask_key
        self.num_classes = num_classes

    def __call__(self, data):
        d = dict(data)
        ann_str = d.get(self.source_key, "")

        mask = torch.zeros(self.num_classes, dtype=torch.float32)

        if isinstance(ann_str, str) and ann_str.strip():
            # Handle MONAI's path mangling (os.path.join with basedir)
            ann_str = os.path.basename(ann_str)
            for idx_str in ann_str.split(","):
                idx_str = idx_str.strip()
                if idx_str:
                    idx = int(idx_str)
                    if 0 <= idx < self.num_classes:
                        mask[idx] = 1.0
        elif isinstance(ann_str, (list, tuple)):
            for idx in ann_str:
                idx = int(idx)
                if 0 <= idx < self.num_classes:
                    mask[idx] = 1.0

        d[self.mask_key] = mask
        return d


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def load_datalist(cfg) -> tuple[list[dict], list[dict]]:
    """Load training and validation file lists from datalist.json.

    Returns:
        (train_files, val_files) — each a list of dicts with keys
        'image', 'label', 'annotated_classes'.
    """
    datalist_path = getattr(cfg, "datalist", "")
    with open(datalist_path, "r") as f:
        datalist = json.load(f)

    train_files = datalist.get("training", [])
    val_files = datalist.get("validation", [])

    return train_files, val_files


class CellMapDataset(Dataset):
    """
    On-the-fly "crop-first" NIfTI dataset for CellMap segmentation.

    For each ``__getitem__`` call:
      1. Load one NIfTI volume (image uint8 + label uint8/16) from disk.
      2. Random-crop ``num_samples`` patches at roi_size *before* expanding
         labels to 14-channel float32 — this keeps peak RAM per worker at
         one decompressed NIfTI volume (~tens of MB to ~8 GB for the
         biggest crop) rather than 14× float32 expansion of the full volume
         (up to 84 GB).
      3. Expand integer labels to multi-channel binary on the *small crop*.
      4. Normalize intensity, apply spatial augmentations.
      5. Return stacked sub-patches ready for flat_collate_fn.

    This replaces the previous PersistentDataset approach which attempted to
    cache full expanded volumes to disk — infeasible when a single .pt file
    reaches 84 GB (MONAI issues #3843, #5930, #6585).
    """

    def __init__(self, file_list: list[dict], cfg, mode: str = "train"):
        self.cfg = cfg
        self.mode = mode
        self.file_list = file_list
        self.num_classes = cfg.num_classes
        self.num_samples = getattr(cfg, "num_samples", 4) if mode == "train" else 1
        roi_size = getattr(cfg, "roi_size", [128, 128, 128])

        # ── Phase 1: Load + orient + pad (both image & integer label) ────
        # Label stays as integer (1, Z, Y, X) — NOT expanded yet.
        # SpatialPadd ensures every dimension ≥ roi_size so that
        # RandSpatialCropSamplesd never receives a too-small volume
        # (some CellMap volumes have a dimension < 128).
        self.load_transforms = mt.Compose([
            mt.LoadImaged(keys=["image", "label"]),
            mt.EnsureChannelFirstd(keys=["image", "label"]),
            mt.Orientationd(keys=["image", "label"], axcodes="RAS"),
            mt.SpatialPadd(keys=["image", "label"], spatial_size=roi_size),
            # Parse annotation mask BEFORE crop (it's metadata, not spatial)
            ParseAnnotationMaskd(
                source_key="annotated_classes",
                mask_key="annotation_mask",
                num_classes=cfg.num_classes,
            ),
        ])

        # ── Phase 2: Random crop at integer-label resolution ─────────────
        # Crop BEFORE multi-channel expansion = huge RAM savings.
        if mode == "train":
            self.crop_transform = mt.RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=roi_size,
                num_samples=self.num_samples,
                random_center=True,
                random_size=False,
            )
        else:
            self.crop_transform = mt.RandSpatialCropSamplesd(
                keys=["image", "label"],
                roi_size=roi_size,
                num_samples=1,
                random_center=True,
                random_size=False,
            )

        # ── Phase 3: Post-crop transforms (on small patches only) ───────
        # Now it's safe to expand to 14 channels — 14 × 96³ × 4B ≈ 50 MB.
        post_crop_list = [
            IntegerLabelToMultiChanneld(
                keys=["label"], num_classes=cfg.num_classes
            ),
            mt.NormalizeIntensityd(keys=["image"]),
        ]
        if mode == "train":
            post_crop_list.extend([
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                mt.RandRotate90d(
                    keys=["image", "label"], prob=0.75, max_k=3,
                    spatial_axes=(0, 1),
                ),
            ])
        self.post_crop_transforms = mt.Compose(post_crop_list)

        self.n_volumes = len(file_list)
        self.length = self.n_volumes

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> dict:
        vol_idx = idx % self.n_volumes

        # Phase 1: load NIfTI + orient (integer label, small footprint)
        item = self.load_transforms(self.file_list[vol_idx])

        # Grab annotation mask before crop (per-volume metadata)
        annotation_mask = item["annotation_mask"]  # (C,)

        # Phase 2: random crop → list of dicts (still integer label)
        patches = self.crop_transform(item)

        # Free the full volume immediately — only small crops survive
        del item

        # Phase 3: expand labels + normalize + augment (on small patches)
        patches = [self.post_crop_transforms(p) for p in patches]

        # Stack sub-patches into a single dict with batch dim
        images = torch.stack([p["image"] for p in patches])      # (S, 1, D, H, W)
        targets = torch.stack([p["label"] for p in patches])     # (S, C, D, H, W)
        masks = annotation_mask.unsqueeze(0).expand(len(patches), -1)  # (S, C)

        return {
            "input": images,                # (S, 1, D, H, W)
            "target": targets,              # (S, C, D, H, W)
            "annotation_mask": masks,       # (S, C)
        }


def flat_collate_fn(batch: list[dict]) -> dict:
    """Collate that concatenates along dim=0 (flattening sub-patches into batch).

    Each item in batch has shape (S, ...) from CellMapDataset.__getitem__.
    Output has shape (B*S, ...).
    """
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [item[k] for item in batch]
        if isinstance(vals[0], torch.Tensor):
            out[k] = torch.cat(vals, dim=0)  # (B*S, ...)
        else:
            # Flatten lists
            flat = []
            for v in vals:
                if isinstance(v, list):
                    flat.extend(v)
                else:
                    flat.append(v)
            out[k] = flat
    return out


# Override the default collate functions for train/val
tr_collate_fn = flat_collate_fn
val_collate_fn = flat_collate_fn
