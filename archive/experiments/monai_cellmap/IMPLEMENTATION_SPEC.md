# Custom MONAI Training Pipeline for CellMap Segmentation Challenge

## Implementation Specification v1.0

**Date**: February 13, 2026
**Branch**: `feature/l40s-model-comparison`
**Inspired by**: [CryoET 1st Place Solution](https://github.com/ChristofHenkel/kaggle-cryoet-1st-place-segmentation) by Christof Henkel

---

## 1. Objective

Build a custom MONAI-based training pipeline for the [Janelia CellMap Segmentation Challenge](https://github.com/janelia-cellmap/cellmap-segmentation-challenge). The pipeline uses MONAI as a **library** (not Auto3DSeg as a framework), giving us full control over loss functions, augmentations, partial annotation handling, and model configuration.

The CryoET 1st place winners proved this approach works — they used MONAI's `FlexibleUNet`, `CacheDataset`, and transforms as building blocks inside a custom training loop with DDP, bfloat16, Mixup, weighted CrossEntropy, and deep supervision. We adapt their architecture for CellMap's multi-label organelle segmentation with partial annotations.

---

## 2. Environment & Hardware

| Item | Value |
|------|-------|
| **Cluster** | UNC Longleaf (RHEL9, glibc 2.38, Slurm) |
| **Node** | g181003 (reserved until Feb 23, 2026) |
| **GPUs** | 7× NVIDIA L40S 48GB (1 GPU held by another user's job) |
| **CPUs** | 63 |
| **RAM** | ~920 GB available |
| **Python** | 3.11 |
| **PyTorch** | 2.10.0+cu128 |
| **MONAI** | 1.5.2 |
| **CUDA** | 12.8 |
| **Conda env** | `csc` (micromamba) |
| **Partition** | `l40-gpu`, reservation `gsgeorge_9034` |
| **SSH path** | Sycamore → Longleaf → compute node. `sbatch` from longleaf login. |

**Micromamba activation sequence** (required in all sbatch scripts):
```bash
export MAMBA_EXE='/nas/longleaf/home/gsgeorge/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/nas/longleaf/home/gsgeorge/micromamba'
eval "$($MAMBA_EXE shell hook --shell bash --root-prefix $MAMBA_ROOT_PREFIX 2>/dev/null)"
micromamba activate csc
```

---

## 3. Dataset Description

### 3.1 Data Format

Data has already been converted from CellMap zarr format to NIfTI for training. Files are at:

```
/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/auto3dseg/nifti_data/
├── images/   # 277 files: {dataset}_{cropID}_0000.nii.gz (float32)
├── labels/   # 277 files: {dataset}_{cropID}.nii.gz (uint8, values 0-14)
└── datalist.json  # MONAI-format data manifest
```

### 3.2 Image Properties

| Property | Value |
|----------|-------|
| **Format** | NIfTI (.nii.gz), gzip compressed |
| **Image dtype** | float32 |
| **Label dtype** | uint8 |
| **Spacing** | (8.0, 8.0, 8.0) nm isotropic |
| **Intensity range** | ~[0, 252] (varies per dataset) |
| **Typical shape** | 400×400×400 (median) |
| **Min shape** | 100×124×180 |
| **Max shape** | 1796×1500×3000 (a few giant crops) |
| **Total images** | 277 (231 train, 46 validation) |

### 3.3 Memory Requirements

| Subset | Count | Total RAM (img+label) |
|--------|-------|----------------------|
| All crops | 277 | ~206 GB |
| Crops ≤ 500 MB each | 232 | ~60 GB |
| Crops > 500 MB each | 45 | ~146 GB |

**⚠ CRITICAL — CacheDataset × DDP Memory Duplication:**

`torchrun` spawns N independent processes. If each process independently creates a `CacheDataset(cache_rate=1.0)`, the 277 crops (~206 GB) are cached **N times** → 7 × 206 = **1.4 TB**, which exceeds 920 GB and will OOM immediately. This is the single most important memory decision in the pipeline.

**Recommended caching strategies (pick one):**

| Strategy | RAM Usage | Complexity | Recommended? |
|----------|-----------|------------|-------------|
| **A. `num_workers=0` in CacheDataset** | 206 GB × 7 = 1.4 TB ❌ | Low | ❌ Still duplicates per-process |
| **B. Cache in rank-0, broadcast via shared memory** | ~206 GB shared | High | ✅ Ideal but requires custom code |
| **C. `PersistentDataset` (disk cache)** | ~206 GB disk + small RAM | Medium | ✅ Good fallback |
| **D. Cache only ≤500 MB crops** | 60 GB × 7 = 420 GB ✅ | Low | ✅ Conservative, fits in 920 GB |
| **E. `cache_rate=1.0` + Linux fork COW** | ~206 GB if read-only | Medium | ⚠ Only works with `mp.set_start_method('fork')` |

**Strategy D is the safest starting point**: set `cache_rate=1.0` but only pass the 232 crops ≤500 MB to CacheDataset (60 GB × 7 = 420 GB fits). For the 45 large crops, load them on-demand with a second `Dataset(cache_rate=0.0)` and `ConcatDataset` the two. This leaves ~500 GB headroom for model weights, activations, and gradients.

**Strategy B is the ideal end-state**: Cache all 277 crops in a single process using `torch.multiprocessing.set_start_method('fork')` so child processes inherit the parent's memory via copy-on-write. Since cached tensors are read-only (only random crops are written), COW pages are never dirtied and RAM stays at ~206 GB total. **However**: `fork` + CUDA is unsafe if any CUDA context exists before fork. The safe pattern is:
1. Cache datasets in main process **before** any `torch.cuda` call
2. Call `torch.multiprocessing.spawn()` (or use `torchrun` which uses `spawn`)
3. Each spawned worker inherits read-only cached data

Since `torchrun` uses `spawn` (not `fork`), Strategy B requires explicitly using `multiprocessing.spawn()` in `train.py` instead of `torchrun`, or serializing the cache to shared memory (`/dev/shm`) and mmapping it.

**Validate early**: Before full training, run a quick test caching 10 crops with 7 DDP workers and monitor RSS per process via `ps -eo pid,rss,comm | grep python`.

### 3.4 Classes (14 organelles)

| Index | Label Value | Name | Annotated in N crops | % of 277 |
|-------|-------------|------|---------------------|----------|
| 0 | 1 | ecs (extracellular space) | 155 | 56% |
| 1 | 2 | pm (plasma membrane) | 125 | 45% |
| 2 | 3 | mito_mem (mitochondria membrane) | 133 | 48% |
| 3 | 4 | mito_lum (mitochondria lumen) | 133 | 48% |
| 4 | 5 | mito_ribo (mito ribosomes) | 62 | 22% |
| 5 | 6 | golgi_mem (Golgi membrane) | 19 | 7% |
| 6 | 7 | golgi_lum (Golgi lumen) | 19 | 7% |
| 7 | 8 | ves_mem (vesicle membrane) | 177 | 64% |
| 8 | 9 | ves_lum (vesicle lumen) | 178 | 64% |
| 9 | 10 | endo_mem (endosome membrane) | 170 | 61% |
| 10 | 11 | endo_lum (endosome lumen) | 170 | 61% |
| 11 | 12 | er_mem (ER membrane) | 185 | 67% |
| 12 | 13 | er_lum (ER lumen) | 186 | 67% |
| 13 | 14 | nuc (nucleus) | 78 | 28% |

**Critical**: Labels are integers 0-14 in the NIfTI files. Value 0 = background/unannotated. Values 1-14 map to the 14 classes above. The label must be converted to 14-channel binary masks before training (channel `c` = `(label == c+1)`).

### 3.5 Partial Annotations

**Not all 14 classes are annotated in every crop.** Each crop in `datalist.json` has an `annotated_classes` field — a comma-separated string of 0-indexed channel indices that are annotated in that crop.

Example: `"annotated_classes": "0,1,2,3,7,8,9,10,11,12"` means channels 0,1,2,3,7,8,9,10,11,12 are annotated; channels 4,5,6,13 (mito_ribo, golgi_mem, golgi_lum, nuc) are NOT annotated in this crop.

**For unannotated channels**:
- The label NIfTI will have 0s everywhere for those classes
- But 0 does NOT mean "this class is absent" — it means "we don't know"
- The loss MUST ignore unannotated channels (no gradient signal)
- Golgi is the rarest (only 7% of crops), so ignoring it when absent is critical

### 3.6 Datalist Format

The file `auto3dseg/nifti_data/datalist.json` has this structure:

```json
{
    "name": "CellMap FIB-SEM Segmentation",
    "description": "...",
    "modality": "EM",
    "sigmoid": true,
    "num_classes": 14,
    "class_names": [
        {"name": "ecs", "index": [1]},
        {"name": "pm", "index": [2]},
        ...
    ],
    "training": [
        {
            "image": "/abs/path/to/images/jrc_cos7-1a_crop234_0000.nii.gz",
            "label": "/abs/path/to/labels/jrc_cos7-1a_crop234.nii.gz",
            "annotated_classes": "0,1,2,3,7,8,9,10,11,12"
        },
        ...
    ],
    "validation": [
        {
            "image": "/abs/path/to/images/jrc_cos7-1a_crop252_0000.nii.gz",
            "label": "/abs/path/to/labels/jrc_cos7-1a_crop252.nii.gz",
            "annotated_classes": "0,1,2,3,4,7,8,9,10,11,12"
        },
        ...
    ]
}
```

All image/label paths are **absolute**. The `dataroot` / `data_file_base_dir` should be set to `""` (empty string) since paths are absolute.

---

## 4. Architecture Design

### 4.1 Directory Structure

```
experiments/monai_cellmap/
├── IMPLEMENTATION_SPEC.md      # This file
├── configs/
│   ├── common_config.py        # Base config (SimpleNamespace)
│   ├── cfg_segresnet.py        # SegResNet with deep supervision
│   ├── cfg_swinunetr.py        # SwinUNETR (96³ patches)
│   └── cfg_flexunet_resnet.py  # FlexibleUNet + ResNet34 encoder
├── models/
│   └── mdl_cellmap.py          # Net class: backbone + partial annotation loss
├── data/
│   └── ds_cellmap.py           # Dataset: NIfTI CacheDataset + transforms
├── losses/
│   └── partial_annotation.py   # PartialAnnotationLossV2 (copy from auto3dseg/)
├── train.py                    # Main training loop (adapted from CryoET winner)
├── utils.py                    # DDP, checkpointing, scheduling, etc.
└── slurm/
    └── train_reserved.sbatch   # Slurm submission script
```

### 4.2 Key Design Decisions

1. **MONAI as library, not framework**: We import MONAI components directly — no Auto3DSeg, no AutoRunner, no BundleGen.

2. **NIfTI data via CacheDataset**: Use `monai.data.CacheDataset` to avoid re-reading NIfTI files every epoch. **⚠ See Section 3.3 and Section 11.3 for the critical DDP memory duplication pitfall.** The safe starting point is Strategy D: cache only the 232 crops ≤500 MB each (60 GB per worker × 7 workers = 420 GB, fits in 920 GB). Load the 45 large crops on demand.

3. **Multi-label sigmoid** (not softmax): Each of the 14 channels is an independent binary segmentation. Uses `sigmoid` activation, not `softmax`. This is because organelles can potentially overlap at boundaries.

4. **Partial annotation masking**: Compute per-channel Dice + BCE losses separately, then zero out unannotated channels and average only over annotated ones. This is implemented in `PartialAnnotationLossV2` (already written, at `auto3dseg/partial_annotation.py`).

5. **Sub-epoch random cropping**: Like the CryoET winner, define `train_sub_epochs` (number of random patch re-samples per volume per epoch). With 231 training volumes and `sub_epochs=10`, each epoch has 2310 training steps. Patches are randomly cropped from cached volumes each time.

6. **bfloat16**: Use `torch.autocast('cuda', dtype=torch.bfloat16)` for training. L40S supports bf16 natively. More numerically stable than fp16 for training.

7. **DDP via torchrun**: Launch multi-GPU training with `torchrun --nproc_per_node=N train.py -C <config>`.

---

## 5. Component Specifications

### 5.1 `configs/common_config.py` — Base Configuration

```python
from types import SimpleNamespace
from monai import transforms as mt

cfg = SimpleNamespace(**{})

# === Data ===
cfg.datalist = "/work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation/auto3dseg/nifti_data/datalist.json"
cfg.dataroot = ""  # paths in datalist are absolute
cfg.num_classes = 14
cfg.class_names = ["ecs", "pm", "mito_mem", "mito_lum", "mito_ribo",
                   "golgi_mem", "golgi_lum", "ves_mem", "ves_lum",
                   "endo_mem", "endo_lum", "er_mem", "er_lum", "nuc"]
cfg.in_channels = 1
cfg.sigmoid = True  # multi-label, not softmax

# === Patches ===
cfg.roi_size = [128, 128, 128]  # patch size for random cropping
cfg.num_samples = 4             # sub-patches per volume per __getitem__ call

# === Training ===
cfg.epochs = 100
cfg.lr = 1e-3
cfg.optimizer = "AdamW"
cfg.weight_decay = 1e-5
cfg.schedule = "cosine"
cfg.warmup = 0.05              # fraction of total steps for warmup
cfg.batch_size = 2             # volumes per GPU per step (each yields num_samples patches)
cfg.grad_accumulation = 1
cfg.clip_grad = 1.0
cfg.seed = 42

# === Loss ===
cfg.loss_type = "partial_dice_ce"  # our custom PartialAnnotationLossV2
cfg.dice_weight = 1.0
cfg.ce_weight = 1.0
cfg.class_weights = None  # or np.array([...]) for weighted CE per class

# === Augmentation (Mixup) ===
cfg.mixup_p = 0.5             # probability of applying mixup
cfg.mixup_beta = 1.0          # Beta distribution parameter

# === Precision ===
cfg.bf16 = True               # use bfloat16 autocast
cfg.mixed_precision = False    # standard fp16 (mutually exclusive with bf16)

# === DDP ===
cfg.distributed = True
cfg.find_unused_parameters = False
cfg.syncbn = False

# === Resources ===
cfg.num_workers = 4
cfg.pin_memory = False          # not needed with CacheDataset
cfg.cache_rate = 1.0            # cache all crops in RAM
cfg.drop_last = True

# === Checkpointing ===
cfg.save_checkpoint = True
cfg.save_weights_only = True
cfg.save_only_last_ckpt = False
cfg.eval_epochs = 5             # validate every N epochs
cfg.output_dir = "/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap"

# === Logging ===
cfg.neptune_project = None      # or "workspace/project" for Neptune
cfg.disable_tqdm = False

basic_cfg = cfg
```

### 5.2 `configs/cfg_segresnet.py` — SegResNet Config

```python
from common_config import basic_cfg
from copy import copy
import numpy as np

cfg = copy(basic_cfg)
cfg.name = "segresnet_ds"
cfg.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/{cfg.name}"

# Model
cfg.model = "mdl_cellmap"
cfg.backbone_type = "segresnet"
cfg.backbone_args = dict(
    spatial_dims=3,
    in_channels=cfg.in_channels,
    out_channels=cfg.num_classes,
    init_filters=32,
    blocks_down=[1, 2, 2, 4, 4],
    norm="INSTANCE",
    # deep supervision is built into SegResNetDS
)
cfg.deep_supervision = True
cfg.ds_weights = [1.0, 0.5, 0.25, 0.125]  # per level

# Loss - class weights: higher weight for rare classes
# Golgi (5,6) and mito_ribo (4) are rare → upweight
cfg.class_weights = np.array([1, 1, 1, 1, 4, 8, 8, 1, 1, 1, 1, 1, 1, 2], dtype=np.float32)

# Patches
cfg.roi_size = [128, 128, 128]
cfg.num_samples = 4
cfg.batch_size = 2

# Training
cfg.lr = 2e-4
cfg.epochs = 200
cfg.eval_epochs = 10
```

### 5.3 `configs/cfg_swinunetr.py` — SwinUNETR Config

```python
from common_config import basic_cfg
from copy import copy
import numpy as np

cfg = copy(basic_cfg)
cfg.name = "swinunetr"
cfg.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/{cfg.name}"

# Model
cfg.model = "mdl_cellmap"
cfg.backbone_type = "swinunetr"
cfg.backbone_args = dict(
    img_size=[96, 96, 96],  # SwinUNETR requires fixed spatial size
    in_channels=cfg.in_channels,
    out_channels=cfg.num_classes,
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_v2=True,
)
cfg.deep_supervision = False  # SwinUNETR doesn't have built-in DS

# Loss
cfg.class_weights = np.array([1, 1, 1, 1, 4, 8, 8, 1, 1, 1, 1, 1, 1, 2], dtype=np.float32)

# Patches — must match img_size
cfg.roi_size = [96, 96, 96]
cfg.num_samples = 4
cfg.batch_size = 2

# Training
cfg.lr = 1e-4
cfg.epochs = 200
cfg.eval_epochs = 10
```

### 5.4 `configs/cfg_flexunet_resnet.py` — FlexibleUNet + ResNet34 (CryoET winner style)

```python
from common_config import basic_cfg
from copy import copy
import numpy as np

cfg = copy(basic_cfg)
cfg.name = "flexunet_resnet34"
cfg.output_dir = f"/work/users/g/s/gsgeorge/cellmap/runs/monai_cellmap/{cfg.name}"

# Model — directly from MONAI, same as CryoET winner
cfg.model = "mdl_cellmap"
cfg.backbone_type = "flexunet"
cfg.backbone_args = dict(
    spatial_dims=3,
    in_channels=cfg.in_channels,
    out_channels=cfg.num_classes,
    backbone="resnet34",
    pretrained=False,  # no ImageNet pretrained for 3D
)
cfg.deep_supervision = False  # FlexibleUNet has multi-scale heads via PatchedUNetDecoder
cfg.multi_scale_heads = True  # use segmentation heads at multiple decoder scales
cfg.lvl_weights = np.array([0, 0, 0, 1], dtype=np.float32)  # only use final scale

# Loss — CryoET winner used 256:1 class weights; we adapt for CellMap
cfg.class_weights = np.array([1, 1, 1, 1, 4, 8, 8, 1, 1, 1, 1, 1, 1, 2], dtype=np.float32)

# Mixup — CryoET winner used this aggressively
cfg.mixup_p = 1.0
cfg.mixup_beta = 1.0

# Patches
cfg.roi_size = [96, 96, 96]
cfg.num_samples = 4
cfg.batch_size = 4  # smaller patches → can fit more per batch

# Training
cfg.lr = 1e-3
cfg.optimizer = "Adam"
cfg.weight_decay = 0.0
cfg.epochs = 100
cfg.eval_epochs = 5
```

### 5.5 `data/ds_cellmap.py` — Dataset

This is the most important adaptation. It must:
1. Load NIfTI images + labels from datalist.json
2. Convert integer labels (0-14) to 14-channel binary masks
3. Parse `annotated_classes` strings into binary channel masks
4. Use MONAI `CacheDataset` to cache everything in RAM
5. Apply random spatial crops + augmentations at training time
6. Return `{"input": tensor, "target": tensor, "annotation_mask": tensor}`

```python
import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import monai.data as md
import monai.transforms as mt
from tqdm import tqdm


def batch_to_device(batch, device):
    """Move all tensors in a batch dict to device."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def collate_fn(batch):
    """Stack batch items into tensors."""
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch]) for k in keys}


tr_collate_fn = collate_fn
val_collate_fn = collate_fn


class IntegerLabelToMultiChanneld(mt.MapTransform):
    """Convert integer label (1, Z, Y, X) with values 0-14 to multi-channel binary (C, Z, Y, X).
    
    Value 0 = background (ignored). Values 1..num_classes map to channels 0..num_classes-1.
    """
    def __init__(self, keys="label", num_classes=14):
        super().__init__(keys)
        self.num_classes = num_classes
    
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            label = d[key]  # shape: (1, Z, Y, X) integer
            if isinstance(label, torch.Tensor):
                label_np = label.numpy()
            else:
                label_np = np.asarray(label)
            
            # Squeeze channel dim if present
            if label_np.ndim == 4 and label_np.shape[0] == 1:
                label_np = label_np[0]
            
            # Create multi-channel binary: (C, Z, Y, X)
            multi = np.zeros((self.num_classes,) + label_np.shape, dtype=np.float32)
            for c in range(self.num_classes):
                multi[c] = (label_np == (c + 1)).astype(np.float32)
            
            d[key] = torch.from_numpy(multi)
        return d


class ParseAnnotationMaskd(mt.MapTransform):
    """Parse 'annotated_classes' string into a float tensor of shape (C,)."""
    def __init__(self, source_key="annotated_classes", mask_key="annotation_mask", num_classes=14):
        super().__init__(keys=[])  # We don't transform standard keys
        self.source_key = source_key
        self.mask_key = mask_key
        self.num_classes = num_classes
    
    def __call__(self, data):
        d = dict(data)
        ann_str = d.get(self.source_key, "")
        mask = torch.zeros(self.num_classes, dtype=torch.float32)
        if isinstance(ann_str, str) and ann_str.strip():
            for idx_str in ann_str.split(","):
                idx = int(idx_str.strip())
                if 0 <= idx < self.num_classes:
                    mask[idx] = 1.0
        d[self.mask_key] = mask
        return d


class CellMapDataset(Dataset):
    """
    NIfTI-based dataset for CellMap segmentation.
    
    Mirrors the CryoET winner's pattern:
    1. Load all volumes at init using CacheDataset (static transforms only)
    2. At __getitem__, apply random augmentations (crop, flip, rotate)
    3. Return dict with input, target, annotation_mask
    
    For training: random crops → multiple sub-patches per volume
    For validation: grid patches over full volume (or just random crops)
    """
    
    def __init__(self, file_list, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode
        self.num_classes = cfg.num_classes
        
        # Static transforms: load NIfTI, ensure channel first, normalize, convert labels
        static_transforms = mt.Compose([
            mt.LoadImaged(keys=["image", "label"]),
            mt.EnsureChannelFirstd(keys=["image"]),
            mt.EnsureChannelFirstd(keys=["label"]),
            IntegerLabelToMultiChanneld(keys=["label"], num_classes=cfg.num_classes),
            ParseAnnotationMaskd(source_key="annotated_classes", mask_key="annotation_mask",
                                 num_classes=cfg.num_classes),
            mt.NormalizeIntensityd(keys=["image"]),  # zero mean, unit std
        ])
        
        # Cache all volumes in RAM
        print(f"  Caching {len(file_list)} {mode} volumes (cache_rate={cfg.cache_rate})...")
        self.cached_data = md.CacheDataset(
            data=file_list,
            transform=static_transforms,
            cache_rate=cfg.cache_rate,
            num_workers=cfg.num_workers,
        )
        
        # Random augmentations applied at __getitem__ time
        if mode == "train":
            self.random_transforms = mt.Compose([
                mt.RandSpatialCropSamplesd(
                    keys=["image", "label"],
                    roi_size=cfg.roi_size,
                    num_samples=cfg.num_samples,
                    random_center=True,
                    random_size=False,
                ),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
                mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
                mt.RandRotate90d(keys=["image", "label"], prob=0.75, max_k=3, spatial_axes=(0, 1)),
            ])
            self.sub_epochs = getattr(cfg, "train_sub_epochs", 10)  # re-sample patches N times per epoch
            self.length = len(self.cached_data) * self.sub_epochs
        else:
            # Validation: random crop (or grid patch for full-volume eval)
            self.random_transforms = mt.Compose([
                mt.RandSpatialCropd(
                    keys=["image", "label"],
                    roi_size=cfg.roi_size,
                    random_center=True,
                    random_size=False,
                ),
            ])
            self.sub_epochs = 1
            self.length = len(self.cached_data)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Map idx to a cached volume (cycle through sub_epochs)
        vol_idx = idx % len(self.cached_data)
        cached_item = self.cached_data[vol_idx]
        
        if self.mode == "train":
            # Apply random transforms → returns list of num_samples dicts
            augmented = self.random_transforms(cached_item)
            # Stack sub-patches into a batch
            feature_dict = {
                "input": torch.stack([item["image"] for item in augmented]),   # (num_samples, 1, D, H, W)
                "target": torch.stack([item["label"] for item in augmented]),  # (num_samples, C, D, H, W)
                "annotation_mask": cached_item["annotation_mask"].unsqueeze(0).expand(
                    len(augmented), -1
                ),  # (num_samples, C)
            }
        else:
            augmented = self.random_transforms(cached_item)
            feature_dict = {
                "input": augmented["image"].unsqueeze(0),    # (1, 1, D, H, W)
                "target": augmented["label"].unsqueeze(0),   # (1, C, D, H, W)
                "annotation_mask": cached_item["annotation_mask"].unsqueeze(0),  # (1, C)
            }
        
        return feature_dict
```

### 5.6 `models/mdl_cellmap.py` — Network + Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.nets import SegResNetDS, SwinUNETR, FlexibleUNet


class PartialAnnotationLossV2(nn.Module):
    """Per-channel Dice + BCE with annotation masking.
    
    Copy from: auto3dseg/partial_annotation.py (PartialAnnotationLossV2 class)
    Already tested and working. See that file for full docstring.
    """
    # ... (copy from auto3dseg/partial_annotation.py, lines 247-360)


class Mixup(nn.Module):
    """Beta-distribution mixup for 3D volumes."""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta_dist = torch.distributions.Beta(beta, beta)
    
    def forward(self, x, y, mask=None):
        bs = x.shape[0]
        perm = torch.randperm(bs)
        lam = self.beta_dist.rsample((bs,)).to(x.device)
        lam_x = lam.view(-1, *([1] * (x.ndim - 1)))
        lam_y = lam.view(-1, *([1] * (y.ndim - 1)))
        
        x_mixed = lam_x * x + (1 - lam_x) * x[perm]
        y_mixed = lam_y * y + (1 - lam_y) * y[perm]
        
        if mask is not None:
            # For partial annotations: mixup the masks too (take union)
            mask_mixed = torch.max(mask, mask[perm])
            return x_mixed, y_mixed, mask_mixed
        return x_mixed, y_mixed, mask


class Net(nn.Module):
    """Main network module with backbone + loss computation.
    
    Following the CryoET winner's pattern: the forward() method returns
    a dict with 'loss' (during training) and 'logits' (during eval).
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.num_classes
        
        # Build backbone based on config
        if cfg.backbone_type == "segresnet":
            self.backbone = SegResNetDS(**cfg.backbone_args)
        elif cfg.backbone_type == "swinunetr":
            self.backbone = SwinUNETR(**cfg.backbone_args)
        elif cfg.backbone_type == "flexunet":
            self.backbone = FlexibleUNet(**cfg.backbone_args)
        else:
            raise ValueError(f"Unknown backbone_type: {cfg.backbone_type}")
        
        # Loss
        # NOTE: The existing PartialAnnotationLossV2 does NOT accept class_weights.
        # To add per-class weighting, you must extend the class — see Section 11.4.
        self.loss_fn = PartialAnnotationLossV2(
            sigmoid=cfg.sigmoid,
            squared_pred=True,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            ce_weight=cfg.ce_weight,
            dice_weight=cfg.dice_weight,
            num_classes=cfg.num_classes,
        )
        # Store class weights for manual application in forward()
        if cfg.class_weights is not None:
            self.register_buffer("class_weights", torch.from_numpy(cfg.class_weights))
        else:
            self.class_weights = None
        
        # Mixup
        if hasattr(cfg, 'mixup_p') and cfg.mixup_p > 0:
            self.mixup = Mixup(cfg.mixup_beta)
        else:
            self.mixup = None
        
        # Deep supervision weights
        self.deep_supervision = getattr(cfg, 'deep_supervision', False)
        self.ds_weights = getattr(cfg, 'ds_weights', [1.0, 0.5, 0.25, 0.125])
        
        # Print param count
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Net [{cfg.backbone_type}]: {n_params:,} trainable parameters")
    
    def forward(self, batch):
        x = batch["input"]          # (B, 1, D, H, W)
        y = batch.get("target")     # (B, C, D, H, W) or None
        mask = batch.get("annotation_mask")  # (B, C) or None
        
        # Mixup during training
        if self.training and y is not None and self.mixup is not None:
            if torch.rand(1).item() < self.cfg.mixup_p:
                x, y, mask = self.mixup(x, y, mask)
        
        # Forward through backbone
        out = self.backbone(x)
        
        outputs = {}
        
        if y is not None:
            # Compute loss
            if self.deep_supervision and isinstance(out, (list, tuple)):
                total_loss = 0.0
                total_w = 0.0
                for i, pred in enumerate(out):
                    w = self.ds_weights[i] if i < len(self.ds_weights) else self.ds_weights[-1]
                    if w <= 0:
                        continue
                    # Resize target if needed
                    if pred.shape[2:] != y.shape[2:]:
                        y_resized = F.interpolate(y, size=pred.shape[2:], mode="nearest")
                    else:
                        y_resized = y
                    self.loss_fn.set_annotation_mask(mask)
                    total_loss += w * self.loss_fn(pred, y_resized)
                    total_w += w
                outputs["loss"] = total_loss / total_w
            else:
                pred = out[-1] if isinstance(out, (list, tuple)) else out
                self.loss_fn.set_annotation_mask(mask)
                outputs["loss"] = self.loss_fn(pred, y)
        
        if not self.training:
            pred = out[-1] if isinstance(out, (list, tuple)) else out
            outputs["logits"] = pred
        
        return outputs
```

### 5.7 `train.py` — Training Loop

Adapt directly from the CryoET winner's `train.py`. Key structure:

```
if __name__ == "__main__":
    parse args (-C config, --fold, etc.)
    load config module
    
    if distributed:
        init_process_group("nccl")
    
    load datalist.json
    split into train_files / val_files
    
    train_dataset = CellMapDataset(train_files, cfg, mode="train")
    val_dataset = CellMapDataset(val_files, cfg, mode="val")
    train_loader = DataLoader(train_dataset, sampler=DistributedSampler, ...)
    val_loader = DataLoader(val_dataset, ...)
    
    model = Net(cfg)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = AdamW(model.parameters(), lr=cfg.lr)
    scheduler = CosineAnnealingLR(...)
    
    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            batch = batch_to_device(batch, device)
            with autocast('cuda', dtype=torch.bfloat16):
                output = model(batch)
            loss = output["loss"]
            loss.backward()
            clip_grad_norm_(...)
            optimizer.step()
            scheduler.step()
        
        if (epoch + 1) % cfg.eval_epochs == 0:
            model.eval()
            run_validation(model, val_loader, cfg)
        
        save_checkpoint(model, optimizer, epoch)
```

### 5.8 `slurm/train_reserved.sbatch` — Submission Script

```bash
#!/bin/bash
#SBATCH --job-name=monai_cellmap
#SBATCH --reservation=gsgeorge_9034
#SBATCH --partition=l40-gpu
#SBATCH --account=rc_cburch_pi
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=63
#SBATCH --gres=gpu:7
#SBATCH --mem=920g
#SBATCH --time=11-00:00:00
#SBATCH --output=logs/monai_cellmap_%j.out
#SBATCH --error=logs/monai_cellmap_%j.err

# Environment
# ... (micromamba activation as in Section 2)

cd /work/users/g/s/gsgeorge/cellmap/repo/CellMap-Segmentation

# Performance
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NVIDIA_TF32_OVERRIDE=1

N_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")

# Train SegResNet
torchrun --nproc_per_node=$N_GPUS experiments/monai_cellmap/train.py \
    -C cfg_segresnet --fold -1

# Then train FlexibleUNet+ResNet34
torchrun --nproc_per_node=$N_GPUS experiments/monai_cellmap/train.py \
    -C cfg_flexunet_resnet --fold -1
```

---

## 6. Validation & Metrics

### 6.1 Validation During Training

- Compute per-channel Dice score on validation set every `eval_epochs` epochs
- Only compute Dice for channels that are annotated in each crop
- Report mean Dice across all annotated channel-crop pairs
- Save best model checkpoint based on mean validation Dice

### 6.2 Full Evaluation

After training, use the CellMap challenge's built-in evaluation:
```bash
csc evaluate --predictions <pred_dir> --ground_truth <gt_dir>
```

---

## 7. Differences from CryoET Winner

| Aspect | CryoET Winner | Our CellMap Adaptation | Why |
|--------|--------------|----------------------|-----|
| **Task** | Object detection (sparse point targets) | Dense segmentation (voxel-level) | Different downstream task |
| **Activation** | Softmax (mutually exclusive classes + background) | Sigmoid (multi-label, organelles can overlap) | Organelle membranes overlap with lumens |
| **Loss** | `DenseCrossEntropy` (softmax + class weights) | `PartialAnnotationLossV2` (sigmoid + per-channel Dice+BCE + annotation mask) | Must handle partial annotations |
| **Background class** | Explicit background channel (class N+1) | No explicit background (sigmoid per-class) | Different formulation |
| **Deep supervision** | Multi-scale seg heads with lvl_weights | SegResNetDS built-in or manual | Similar concept |
| **Data format** | Zarr volumes, loaded as numpy | NIfTI volumes (pre-converted from zarr) | Already converted |
| **Post-processing** | NMS + reconstruction for object detection | Thresholding + connected components for instance seg | Different task |

---

## 8. Existing Code to Reuse

These files already exist in the repository and should be copied or imported:

1. **`auto3dseg/partial_annotation.py`** — Contains `PartialAnnotationLossV2` and `AnnotatedClassMaskd`. Copy the `PartialAnnotationLossV2` class into `losses/partial_annotation.py`.

2. **`auto3dseg/nifti_data/datalist.json`** — The data manifest. Read this directly; do NOT regenerate.

3. **`auto3dseg/nifti_data/images/` and `labels/`** — The NIfTI files. Use absolute paths from datalist.json.

---

## 9. Implementation Priority

1. **First**: `data/ds_cellmap.py` + `configs/common_config.py` — get data loading working
2. **Second**: `models/mdl_cellmap.py` + `losses/partial_annotation.py` — network + loss
3. **Third**: `train.py` + `utils.py` — training loop with DDP
4. **Fourth**: `configs/cfg_segresnet.py` — first model config to test
5. **Fifth**: `slurm/train_reserved.sbatch` — submit to cluster
6. **Last**: Additional configs (swinunetr, flexunet_resnet) once SegResNet baseline works

---

## 10. Testing Checklist

Before submitting the full training job, verify each component:

- [ ] `ds_cellmap.py`: Load one crop, verify shapes (`image`: `(1, D, H, W)`, `label`: `(14, D, H, W)`, `annotation_mask`: `(14,)`)
- [ ] `ds_cellmap.py`: Verify `IntegerLabelToMultiChanneld` correctly converts label value 3 → channel 2 = 1.0
- [ ] `ds_cellmap.py`: Verify `ParseAnnotationMaskd` correctly parses `"0,1,2,3,7,8"` → `[1,1,1,1,0,0,0,1,1,0,0,0,0,0]`
- [ ] `ds_cellmap.py`: Verify `CacheDataset` caches without OOM (try with 5 crops first)
- [ ] `ds_cellmap.py`: Verify `jrc_zf-cardiac-1_crop380` (empty annotated_classes) produces all-zero mask and 0.0 loss
- [ ] `ds_cellmap.py`: Monitor per-process RSS with 7 DDP workers caching 10 crops — ensure no N× duplication
- [ ] `mdl_cellmap.py`: Forward pass with dummy input `(2, 1, 128, 128, 128)` produces output of correct shape
- [ ] `mdl_cellmap.py`: Loss backward works with partial annotation mask
- [ ] `mdl_cellmap.py`: Loss with all-zero annotation mask returns 0.0 (not NaN)
- [ ] `mdl_cellmap.py`: Class weights multiply per-channel loss correctly (if extended)
- [ ] `train.py`: Single-GPU training runs for 1 epoch without error
- [ ] `train.py`: Multi-GPU DDP training runs for 1 epoch without error
- [ ] Checkpoint saving/loading works correctly
- [ ] Validation loop computes per-channel Dice correctly

---

## 11. Known Pitfalls & Gotchas

These are sharp edges discovered during design review. **Every one of these will cause a crash or silent bug if not handled.**

### 11.1 `SimpleNamespace` has no `.get()` method

The config object is a `types.SimpleNamespace`, not a `dict`. Calling `cfg.get("key", default)` will raise `AttributeError: 'SimpleNamespace' object has no attribute 'get'`.

**Rule**: All config access throughout the codebase must use:
```python
getattr(cfg, "key", default_value)
```
not `cfg.get("key", default_value)`. This applies to `ds_cellmap.py`, `mdl_cellmap.py`, `train.py`, and `utils.py`.

Alternatively, the config could be changed from `SimpleNamespace` to a custom class with a `.get()` method:
```python
class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)
```
But `getattr` is the safer convention since it works with any object.

### 11.2 Empty `annotated_classes` → zero-denominator loss

One crop (`jrc_zf-cardiac-1_crop380`) has an empty `annotated_classes` string, meaning **no channels are annotated**. The annotation mask will be all zeros.

**Is this safe?** Yes — `PartialAnnotationLossV2` handles it correctly:
```python
# Line 345 of auto3dseg/partial_annotation.py
num_annotated = mask.sum(dim=1).clamp(min=1.0)  # (B,)
per_sample_loss = per_channel_loss.sum(dim=1) / num_annotated  # (B,)
```
The `.clamp(min=1.0)` prevents division by zero. When all mask values are 0, `per_channel_loss * mask` is all zeros, so the numerator is 0.0 and `0.0 / 1.0 = 0.0`. No NaN, no gradient signal — correct behavior.

**However**, verify this with an explicit unit test:
```python
def test_zero_annotation_loss():
    loss_fn = PartialAnnotationLossV2(sigmoid=True, num_classes=14)
    pred = torch.randn(2, 14, 32, 32, 32)
    target = torch.zeros(2, 14, 32, 32, 32)
    mask = torch.zeros(2, 14)  # no channels annotated
    loss_fn.set_annotation_mask(mask)
    loss = loss_fn(pred, target)
    assert loss.item() == 0.0
    assert not torch.isnan(loss)
```

Also verify the **Mixup** path: if a zero-annotation crop is mixed with a normal crop, `mask_mixed = torch.max(mask, mask[perm])` correctly preserves the annotated channels from the other crop.

### 11.3 CacheDataset × DDP = N× memory duplication

See Section 3.3 for the full analysis. **TL;DR**: `torchrun` spawns independent processes. Each one that calls `CacheDataset(cache_rate=1.0)` independently caches all data → 7 × 206 GB = 1.4 TB OOM.

**Mandatory validation step**: Before any full training run, launch 7 DDP workers with 10 crops each and watch RSS:
```bash
# In the sbatch script, add before training:
watch -n 5 'ps -eo pid,rss,comm | grep python | awk "{sum+=\$2; print} END {print \"Total: \" sum/1024/1024 \" GB\"}"'
```

If RSS shows 7 independent copies, switch to Strategy D (cache only ≤500 MB crops) or Strategy B (shared memory) from Section 3.3.

### 11.4 `PartialAnnotationLossV2` does not accept `class_weights`

The existing `PartialAnnotationLossV2` in `auto3dseg/partial_annotation.py` has this constructor:
```python
def __init__(self, sigmoid, squared_pred, smooth_nr, smooth_dr,
             ce_weight, dice_weight, num_classes):
```
There is **no `class_weights` parameter**. The config files define `cfg.class_weights` (e.g., `[1,1,1,1,4,8,8,1,1,1,1,1,1,2]` to upweight rare golgi/mito_ribo), but the loss function cannot use them.

**Two options to implement per-class weighting:**

**Option A — Extend `PartialAnnotationLossV2`** (recommended):
```python
class WeightedPartialAnnotationLossV2(PartialAnnotationLossV2):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.register_buffer("class_weights", torch.as_tensor(class_weights, dtype=torch.float32))
        else:
            self.class_weights = None
    
    def forward(self, input, target):
        # ... (same as parent up to per_channel_loss computation)
        # Then apply class weights before masking:
        if self.class_weights is not None:
            per_channel_loss = per_channel_loss * self.class_weights.unsqueeze(0)  # (B, C)
        # ... (continue with masking and averaging)
```

**Option B — Apply weights externally in `Net.forward()`**:
```python
# In mdl_cellmap.py Net.forward():
self.loss_fn.set_annotation_mask(mask)
raw_loss = self.loss_fn(pred, y)  # scalar
# Problem: can't apply per-channel weights after the loss is already averaged
```
Option B doesn't work because the loss is already averaged over channels inside `forward()`. **You must use Option A** (extend the class) or modify the loss to return per-channel losses before averaging.

### 11.5 `RandSpatialCropSamplesd` returns a list, not a dict

When `num_samples > 1`, `RandSpatialCropSamplesd` returns a **list of dicts**, not a single dict. The `ds_cellmap.py` code correctly handles this (stacking into tensors), but downstream `RandFlipd` / `RandRotate90d` transforms will fail because they expect a dict, not a list.

**Fix**: Split the transform pipeline — apply `RandSpatialCropSamplesd` first, then apply flip/rotate to each sub-patch individually:
```python
self.crop_transform = mt.RandSpatialCropSamplesd(
    keys=["image", "label"], roi_size=cfg.roi_size,
    num_samples=cfg.num_samples, random_center=True, random_size=False,
)
self.augment_transforms = mt.Compose([
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    mt.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    mt.RandRotate90d(keys=["image", "label"], prob=0.75, max_k=3, spatial_axes=(0, 1)),
])

# In __getitem__:
patches = self.crop_transform(cached_item)  # list of num_samples dicts
augmented = [self.augment_transforms(p) for p in patches]  # augment each patch
```

### 11.6 `copy()` of `SimpleNamespace` is shallow

The per-model config files use `cfg = copy(basic_cfg)`. Since `SimpleNamespace` attributes that are mutable (like `dict` for `backbone_args`) are **not deep-copied**, modifying `cfg.backbone_args["in_channels"]` in one config will mutate the original `basic_cfg`. This is unlikely to cause problems when configs are loaded one at a time, but if multiple configs are ever imported in the same process, use `deepcopy`:
```python
from copy import deepcopy
cfg = deepcopy(basic_cfg)
```
