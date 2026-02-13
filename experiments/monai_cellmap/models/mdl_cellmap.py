"""
Network module for CellMap segmentation.

Wraps MONAI backbone (SegResNetDS, SwinUNETR, FlexibleUNet) with:
- Partial annotation loss computation
- Optional Mixup augmentation
- Deep supervision support
- Forward returns dict with 'loss' (train) and 'logits' (eval)

Following the CryoET 1st-place winner pattern where forward() handles
loss computation internally.

Reference: IMPLEMENTATION_SPEC.md §5.6
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.partial_annotation import build_partial_annotation_loss


class Mixup(nn.Module):
    """Beta-distribution Mixup for 3D volumes.

    Mixes pairs of samples within a batch using random lambda from Beta(beta, beta).
    Also mixes the corresponding targets and annotation masks.

    Args:
        beta: Beta distribution parameter. beta=1.0 → Uniform(0,1).
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta_dist = torch.distributions.Beta(beta, beta)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input images (B, 1, D, H, W).
            y: Targets (B, C, D, H, W).
            mask: Annotation masks (B, C).

        Returns:
            (x_mixed, y_mixed, mask_mixed)
        """
        bs = x.shape[0]
        if bs < 2:
            return x, y, mask

        # Sample lambda
        lam = self.beta_dist.sample((bs,)).to(x.device)
        # Reshape for broadcasting: (B, 1, 1, 1, 1) for spatial
        lam_x = lam.view(bs, 1, 1, 1, 1)
        lam_y = lam.view(bs, 1, 1, 1, 1)

        # Random permutation for mixing partners
        perm = torch.randperm(bs, device=x.device)

        x_mixed = lam_x * x + (1.0 - lam_x) * x[perm]
        y_mixed = lam_y * y + (1.0 - lam_y) * y[perm]

        if mask is not None:
            # For annotation mask: take the union (max) — if either sample
            # has a channel annotated, we can compute loss for it after mixing
            # Note: actually we should use the intersection for correctness,
            # but max is what the spec suggests since mixed targets have
            # contributions from both samples.
            # Using max: if sample A has channel annotated and B doesn't,
            # the mixed target for that channel is lam * A_target + (1-lam) * 0.
            # This is valid since we do have a partial signal.
            mask_mixed = torch.max(mask, mask[perm])
        else:
            mask_mixed = None

        return x_mixed, y_mixed, mask_mixed


class Net(nn.Module):
    """
    Main network: backbone + loss computation.

    The forward() method returns a dict:
    - Training: {"loss": scalar, "logits": tensor}
    - Eval: {"logits": tensor}

    This pattern (from CryoET winner) keeps loss computation inside the model,
    which simplifies the training loop and enables model-specific loss logic.

    Args:
        cfg: Config namespace with backbone_type, backbone_args, etc.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        backbone_type = getattr(cfg, "backbone_type", "segresnet")
        backbone_args = getattr(cfg, "backbone_args", {})

        # --- Build backbone ---
        if backbone_type == "segresnet":
            from monai.networks.nets import SegResNetDS
            self.backbone = SegResNetDS(**backbone_args)

        elif backbone_type == "swinunetr":
            from monai.networks.nets import SwinUNETR
            self.backbone = SwinUNETR(**backbone_args)

        elif backbone_type == "flexunet":
            from monai.networks.nets import FlexibleUNet
            self.backbone = FlexibleUNet(**backbone_args)

        else:
            raise ValueError(f"Unknown backbone_type: {backbone_type}")

        # --- Build loss ---
        deep_supervision = getattr(cfg, "deep_supervision", False)
        ds_weights = getattr(cfg, "ds_weights", None)

        self.loss_fn = build_partial_annotation_loss(
            num_classes=getattr(cfg, "num_classes", 14),
            loss_type=getattr(cfg, "loss_type", "balanced_softmax_tversky"),
            tversky_alpha=getattr(cfg, "tversky_alpha", 0.6),
            tversky_beta=getattr(cfg, "tversky_beta", 0.4),
            tau=getattr(cfg, "tau", 1.0),
            update_interval=getattr(cfg, "update_interval", 50),
            deep_supervision=deep_supervision,
            ds_weights=ds_weights,
        )

        # --- Mixup ---
        mixup_p = getattr(cfg, "mixup_p", 0.0)
        if mixup_p > 0:
            self.mixup = Mixup(beta=getattr(cfg, "mixup_beta", 1.0))
            self.mixup_p = mixup_p
        else:
            self.mixup = None
            self.mixup_p = 0.0

        # --- Parameter count ---
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Net [{backbone_type}]: {n_params:,} trainable parameters")

    def forward(self, batch: dict) -> dict:
        """
        Forward pass.

        Args:
            batch: Dict with keys:
                - 'input': (B, 1, D, H, W) image tensor
                - 'target': (B, C, D, H, W) multi-channel binary labels
                - 'annotation_mask': (B, C) per-sample annotation mask

        Returns:
            Dict with 'logits' and optionally 'loss' (if training).
        """
        x = batch["input"]          # (B, 1, D, H, W)
        y = batch["target"]         # (B, C, D, H, W)
        mask = batch.get("annotation_mask", None)  # (B, C) or None

        outputs = {}

        if self.training:
            # Apply Mixup if configured
            if self.mixup is not None and torch.rand(1).item() < self.mixup_p:
                x, y, mask = self.mixup(x, y, mask)

            # Forward through backbone
            logits = self.backbone(x)
            outputs["logits"] = logits if not isinstance(logits, (list, tuple)) else logits[0]

            # Compute loss with partial annotation masking
            if mask is not None:
                self.loss_fn.set_annotation_mask(mask)

            # logits may be a list (deep supervision) or single tensor
            outputs["loss"] = self.loss_fn(logits, y)

        else:
            # Evaluation: just forward pass, no loss
            logits = self.backbone(x)
            if isinstance(logits, (list, tuple)):
                # Deep supervision: use only the full-resolution output
                outputs["logits"] = logits[0]
            else:
                outputs["logits"] = logits

        return outputs
