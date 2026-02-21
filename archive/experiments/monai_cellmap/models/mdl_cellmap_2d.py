"""
2D Network module for CellMap segmentation.

Wraps CSC's existing 2D architectures (ResNet, UNet, Swin, ViT) with:
- MONAI-style partial annotation loss computation (same as 3D pipeline)
- Foreground masking (biggest single gain: +110% baseline Dice)
- Per-class bounding-box spatial masking (box_class_mask_tight)
- Balanced Softmax Tversky loss (τ=1.0, α=0.6, β=0.4)
- Forward returns dict with 'loss' (train) and 'logits' (eval)

This bridges the CSC model zoo with the MONAI training infrastructure,
applying all validated optimizations from EXPERIMENT_FINDINGS.md.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Add the CSC source directory to path so we can import CSC models
CSC_SRC = Path(__file__).resolve().parents[3] / "src"
if str(CSC_SRC) not in sys.path:
    sys.path.insert(0, str(CSC_SRC))

from losses.partial_annotation import build_partial_annotation_loss, FG_THRESHOLD


class Net2D(nn.Module):
    """
    2D network: CSC backbone + MONAI-style loss computation.

    The forward() method returns a dict:
    - Training: {"loss": scalar, "logits": tensor}
    - Eval: {"logits": tensor}

    Supported backbone_type values:
    - "resnet": CSC's ResNet 2D (7.8M params, +63% over UNet)
    - "unet": CSC's UNet 2D (31M params)
    - "swin": CSC's SwinTransformer 2D (36.3M params)
    - "vit": CSC's ViTVNet2D (105.2M params)

    Args:
        cfg: Config namespace with backbone_type, backbone_args, num_classes, etc.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone_type = getattr(cfg, "backbone_type", "resnet")
        num_classes = getattr(cfg, "num_classes", 35)

        # --- Build 2D backbone from CSC models ---
        if backbone_type == "resnet":
            from cellmap_segmentation_challenge.models import ResNet
            backbone_args = getattr(cfg, "backbone_args", {})
            self.backbone = ResNet(
                ndims=2,
                input_nc=backbone_args.get("in_channels", 1),
                output_nc=num_classes,
                ngf=backbone_args.get("ngf", 64),
                n_blocks=backbone_args.get("n_blocks", 6),
                n_downsampling=backbone_args.get("n_downsampling", 2),
            )

        elif backbone_type == "unet":
            from cellmap_segmentation_challenge.models import UNet_2D
            backbone_args = getattr(cfg, "backbone_args", {})
            # UNet_2D uses n_channels / n_classes (not in_channels / out_channels)
            self.backbone = UNet_2D(
                n_channels=backbone_args.get("in_channels", 1),
                n_classes=num_classes,
                **{k: v for k, v in backbone_args.items()
                   if k not in ("in_channels", "out_channels")},
            )

        elif backbone_type == "swin":
            from cellmap_segmentation_challenge.models import SwinTransformer
            backbone_args = getattr(cfg, "backbone_args", {})
            self.backbone = SwinTransformer(
                num_classes=num_classes,
                **backbone_args,
            )

        elif backbone_type == "vit":
            from cellmap_segmentation_challenge.models import ViTVNet2D
            backbone_args = getattr(cfg, "backbone_args", {})
            vit_config = backbone_args.get("vit_config", {
                "img_size": 256,
                "patch_size": 16,
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "mlp_dim": 3072,
                "decoder_channels": (256, 128, 64, 16),
                "dropout_rate": 0.1,
                "attention_dropout_rate": 0.1,
                "down_factor": 2,
            })
            self.backbone = ViTVNet2D(
                config=vit_config,
                in_channels=backbone_args.get("in_channels", 1),
                num_classes=num_classes,
            )

        else:
            raise ValueError(f"Unknown 2D backbone_type: {backbone_type}")

        # --- Build loss (same as 3D pipeline) ---
        # All validated optimizations from EXPERIMENT_FINDINGS.md:
        # - Tversky α=0.6, β=0.4 (loss optimization winner: +47%)
        # - Balanced Softmax τ=1.0 (class weighting winner: +54%)
        # - bbox spatial masking (masking_strategies: +55% pre-fix)
        # - foreground masking (+110% baseline, biggest single gain)
        # - masksup_r0.3 (masking_strategies winner post-fix: +12%)
        self.loss_fn = build_partial_annotation_loss(
            num_classes=num_classes,
            loss_type=getattr(cfg, "loss_type", "balanced_softmax_tversky"),
            tversky_alpha=getattr(cfg, "tversky_alpha", 0.6),
            tversky_beta=getattr(cfg, "tversky_beta", 0.4),
            tau=getattr(cfg, "tau", 1.0),
            update_interval=getattr(cfg, "update_interval", 50),
            bbox_pad_fraction=getattr(cfg, "bbox_pad_fraction", 0.05),
            bbox_bg_weight=getattr(cfg, "bbox_bg_weight", 0.05),
            masksup_ratio=getattr(cfg, "masksup_ratio", 0.3),
            masksup_recon_weight=getattr(cfg, "masksup_recon_weight", 0.5),
            deep_supervision=False,  # no DS for 2D models
        )

        # --- Parameter count ---
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Net2D [{backbone_type}]: {n_params:,} trainable parameters")

    def forward(self, batch: dict) -> dict:
        """
        Forward pass.

        Args:
            batch: Dict with keys:
                - 'input': (B, 1, H, W) image tensor
                - 'target': (B, C, H, W) multi-channel binary labels
                - 'annotation_mask': (B, C) per-sample annotation mask

        Returns:
            Dict with 'logits' and optionally 'loss' (if training).
        """
        x = batch["input"]          # (B, 1, H, W)
        y = batch["target"]         # (B, C, H, W)
        mask = batch.get("annotation_mask", None)  # (B, C) or None

        outputs = {}

        if self.training:
            # Forward through backbone
            logits = self.backbone(x)
            outputs["logits"] = logits

            # Set annotation mask
            if mask is not None:
                self.loss_fn.set_annotation_mask(mask)

            # Foreground mask: True where EM image has real data, False on black padding.
            # This was the single biggest gain in 2D experiments: +110% baseline Dice.
            # For 2D slices, black padding comes from: (1) zarr crop boundaries,
            # (2) padding during crop when slice < roi_size.
            fg_mask = (x.abs().amax(dim=1, keepdim=True) > FG_THRESHOLD)  # (B, 1, H, W)
            if hasattr(self.loss_fn, 'set_foreground_mask'):
                self.loss_fn.set_foreground_mask(fg_mask)

            outputs["loss"] = self.loss_fn(logits, y)

        else:
            # Evaluation: just forward pass
            with torch.no_grad():
                logits = self.backbone(x)
            outputs["logits"] = logits

        return outputs
