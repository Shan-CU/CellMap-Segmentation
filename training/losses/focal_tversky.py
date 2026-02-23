"""
Focal Tversky Loss and Asymmetric Unified Focal Loss.

Focal Tversky Loss (Abraham & Khan, ISBI 2019):
    Applies focal modulation to per-class Tversky:
    loss_c = (1 - Tversky_c)^γ
    When γ < 1, easy (high-Tversky) classes are down-weighted;
    hard (low-Tversky) classes receive relatively more gradient.

Asymmetric Unified Focal Loss (Yeung et al., Medical Image Analysis 2022):
    Combines two complementary losses:
    1. Distribution-based: Asymmetric Focal Tversky (per-voxel focal)
    2. Region-based: Dice Focal (per-class focal on Dice)
    L = w * L_dist + (1-w) * L_region

Both support partial annotation masking via set_annotation_mask().
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss — focuses training on hard classes.

    FTL(c) = (1 - Tversky(c))^γ

    When γ < 1: suppresses loss from easy (high-overlap) classes,
    amplifies gradient for hard (low-overlap) classes.
    When γ = 1: reduces to standard Tversky loss.
    When γ > 1: opposite effect (focus on easy classes — not typical).

    Best γ values from literature: 0.5-0.75 for imbalanced segmentation.

    Args:
        gamma: Focal parameter. Recommended 0.5-0.75.
        alpha: FP weight for Tversky index.
        beta: FN weight for Tversky index.
        smooth: Smoothing constant.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        gamma: float = 0.75,
        alpha: float = 0.6,
        beta: float = 0.4,
        smooth: float = 1e-6,
        num_classes: int = 35,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.num_classes = num_classes
        self._annotation_mask: Optional[torch.Tensor] = None

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        self._annotation_mask = mask

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._annotation_mask
        self._annotation_mask = None

        # Cast to float32 before sigmoid to avoid AMP float16 gradient issues:
        # d/dx[x^γ] = γ·x^(γ-1) → ∞ as x→0 when γ<1.
        pred = torch.sigmoid(input.float())
        target = target.float()
        spatial_dims = tuple(range(2, input.ndim))

        # Per-channel Tversky

        tp = (pred * target).sum(dim=spatial_dims)
        fp = (pred * (1.0 - target)).sum(dim=spatial_dims)
        fn = ((1.0 - pred) * target).sum(dim=spatial_dims)

        denom = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)

        # Focal modulation: (1 - TI)^γ
        # Clamp to [eps,1] before fractional power — eps prevents the
        # backward gradient γ·x^(γ-1) from exploding to inf near x=0.
        eps = 1e-6
        focal_loss = (1.0 - tversky).clamp(min=eps, max=1.0).pow(self.gamma)  # (B, C)

        if mask is not None:
            mask = mask.to(input.device)
            focal_loss = focal_loss * mask
            num_annotated = mask.sum(dim=1).clamp(min=1.0)
            per_sample = focal_loss.sum(dim=1) / num_annotated
        else:
            per_sample = focal_loss.mean(dim=1)

        return per_sample.mean()


class AsymmetricUnifiedFocalLoss(nn.Module):
    """Asymmetric Unified Focal Loss (Yeung et al., MedIA 2022).

    Combines two complementary losses:

    1. **Asymmetric Focal Tversky** (distribution-based):
       Per-voxel focal cross-entropy with asymmetric δ:
       L_dist = mean_c[ (1 - Tversky_c)^γ_dist ]

    2. **Dice Focal** (region-based):
       Per-class Dice with focal modulation:
       L_region = mean_c[ (1 - Dice_c)^γ_region ]

    Final loss = weight_dist * L_dist + (1 - weight_dist) * L_region

    The asymmetry (δ > 0.5) penalises FP more than FN, which addresses
    the low-precision problem we observed in all prior rounds.

    Args:
        delta: Asymmetry parameter (like α in Tversky). δ > 0.5 = precision bias.
        gamma_dist: Focal γ for distribution-based component.
        gamma_region: Focal γ for region-based component.
        weight_dist: Weight for distribution component (region gets 1-weight_dist).
        smooth: Smoothing constant.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        delta: float = 0.6,
        gamma_dist: float = 0.75,
        gamma_region: float = 0.75,
        weight_dist: float = 0.5,
        smooth: float = 1e-6,
        num_classes: int = 35,
    ) -> None:
        super().__init__()
        self.delta = delta
        self.gamma_dist = gamma_dist
        self.gamma_region = gamma_region
        self.weight_dist = weight_dist
        self.smooth = smooth
        self.num_classes = num_classes
        self._annotation_mask: Optional[torch.Tensor] = None

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        self._annotation_mask = mask

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._annotation_mask
        self._annotation_mask = None

        # Compute in float32 to avoid AMP float16 gradient issues:
        # d/dx[x^γ] = γ·x^(γ-1) → ∞ as x→0 when γ<1.
        pred = torch.sigmoid(input.float())
        target = target.float()
        spatial_dims = tuple(range(2, input.ndim))
        eps = 1e-6

        # === Distribution-based: Asymmetric Focal Tversky ===
        tp = (pred * target).sum(dim=spatial_dims)
        fp = (pred * (1.0 - target)).sum(dim=spatial_dims)
        fn = ((1.0 - pred) * target).sum(dim=spatial_dims)

        # Tversky with asymmetric delta (= alpha)
        denom_tv = tp + self.delta * fp + (1.0 - self.delta) * fn + self.smooth
        tversky = (tp + self.smooth) / denom_tv.clamp(min=self.smooth)
        # Clamp to [eps,1] before fractional power — eps prevents the
        # backward gradient γ·x^(γ-1) from exploding to inf near x=0.
        focal_tversky = (1.0 - tversky).clamp(min=eps, max=1.0).pow(self.gamma_dist)  # (B, C)

        # === Region-based: Dice Focal ===
        intersection = (pred * target).sum(dim=spatial_dims)
        union = pred.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        focal_dice = (1.0 - dice).clamp(min=eps, max=1.0).pow(self.gamma_region)  # (B, C)

        # === Combine ===
        combined = self.weight_dist * focal_tversky + (1.0 - self.weight_dist) * focal_dice

        if mask is not None:
            mask = mask.to(input.device)
            combined = combined * mask
            num_annotated = mask.sum(dim=1).clamp(min=1.0)
            per_sample = combined.sum(dim=1) / num_annotated
        else:
            per_sample = combined.mean(dim=1)

        return per_sample.mean()
