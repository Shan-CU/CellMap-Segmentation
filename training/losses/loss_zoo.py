"""
Loss function zoo for ablation experiments.

All losses support partial annotation via NaN masking (the CSC default)
and/or explicit annotation_mask + foreground_mask for our custom losses.

Registry pattern: each loss is registered with a name and builder function.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .partial_annotation import (
    PartialTverskyLoss,
    BalancedSoftmaxTverskyLoss,
    build_partial_annotation_loss,
)
from .focal_tversky import FocalTverskyLoss, AsymmetricUnifiedFocalLoss
from .boundary_loss import BoundaryWeightedTverskyLoss

LOSS_REGISTRY: Dict[str, dict] = {}


def register_loss(name: str, description: str = ""):
    """Decorator to register a loss builder function."""
    def wrapper(fn):
        LOSS_REGISTRY[name] = {"builder": fn, "description": description}
        return fn
    return wrapper


# ============================================================================
# 1. LOSS FUNCTION ABLATION
# ============================================================================

@register_loss("bce", "Binary cross-entropy with logits (CSC default)")
def build_bce(**kwargs) -> nn.Module:
    """Standard BCEWithLogitsLoss — the CSC default baseline."""
    return nn.BCEWithLogitsLoss(reduction="none")


@register_loss("focal", "Focal loss for hard example mining")
def build_focal(gamma: float = 2.0, **kwargs) -> nn.Module:
    """Focal loss — down-weights easy examples, focuses on hard ones."""
    class FocalLoss(nn.Module):
        def __init__(self, gamma: float = 2.0):
            super().__init__()
            self.gamma = gamma

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            bce = F.binary_cross_entropy_with_logits(input, target, reduction="none")
            pt = torch.exp(-bce)
            focal = ((1 - pt) ** self.gamma) * bce
            return focal.mean()

    return FocalLoss(gamma=gamma)


@register_loss("dice_bce", "Dice + BCE combination loss")
def build_dice_bce(bce_weight: float = 0.5, smooth: float = 1e-6, **kwargs) -> nn.Module:
    """Dice + BCE combination — common in medical segmentation."""
    class DiceBCELoss(nn.Module):
        def __init__(self, bce_weight: float = 0.5, smooth: float = 1e-6):
            super().__init__()
            self.bce_weight = bce_weight
            self.smooth = smooth

        def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            bce = F.binary_cross_entropy_with_logits(input, target, reduction="mean")
            pred = torch.sigmoid(input.float())  # float32 for AMP safety
            spatial_dims = tuple(range(2, input.ndim))
            intersection = (pred * target).sum(dim=spatial_dims)
            union = pred.sum(dim=spatial_dims) + target.sum(dim=spatial_dims)
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = 1 - dice.mean()
            return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss

    return DiceBCELoss(bce_weight=bce_weight, smooth=smooth)


@register_loss("tversky", "Per-channel Tversky with partial annotation masking")
def build_tversky(
    alpha: float = 0.6,
    beta: float = 0.4,
    num_classes: int = 35,
    **kwargs,
) -> nn.Module:
    return PartialTverskyLoss(alpha=alpha, beta=beta, num_classes=num_classes)


@register_loss(
    "balanced_softmax_tversky",
    "Logit-adjusted Tversky + bbox masking + foreground masking (our best)",
)
def build_balanced_softmax_tversky(
    tau: float = 1.0,
    alpha: float = 0.6,
    beta: float = 0.4,
    num_classes: int = 35,
    bbox_pad_fraction: float = 0.05,
    bbox_bg_weight: float = 0.05,
    masksup_ratio: float = 0.0,
    **kwargs,
) -> nn.Module:
    return BalancedSoftmaxTverskyLoss(
        tau=tau,
        alpha=alpha,
        beta=beta,
        num_classes=num_classes,
        bbox_pad_fraction=bbox_pad_fraction,
        bbox_bg_weight=bbox_bg_weight,
        masksup_ratio=masksup_ratio,
    )


# ============================================================================
# 2. TVERSKY ALPHA/BETA ABLATION
# ============================================================================

@register_loss("tversky_balanced", "Tversky α=0.5, β=0.5 (= Dice)")
def build_tversky_balanced(num_classes: int = 35, **kwargs) -> nn.Module:
    return PartialTverskyLoss(alpha=0.5, beta=0.5, num_classes=num_classes)


@register_loss("tversky_precision", "Tversky α=0.7, β=0.3 (precision bias)")
def build_tversky_precision(num_classes: int = 35, **kwargs) -> nn.Module:
    return PartialTverskyLoss(alpha=0.7, beta=0.3, num_classes=num_classes)


@register_loss("tversky_recall", "Tversky α=0.3, β=0.7 (recall bias)")
def build_tversky_recall(num_classes: int = 35, **kwargs) -> nn.Module:
    return PartialTverskyLoss(alpha=0.3, beta=0.7, num_classes=num_classes)


# ============================================================================
# 3. WEIGHTING ABLATION (within balanced softmax framework)
# ============================================================================

@register_loss("bst_tau0", "BalancedSoftmaxTversky with τ=0 (no adjustment)")
def build_bst_tau0(num_classes: int = 35, **kwargs) -> nn.Module:
    return BalancedSoftmaxTverskyLoss(tau=0.0, num_classes=num_classes)


@register_loss("bst_tau05", "BalancedSoftmaxTversky with τ=0.5")
def build_bst_tau05(num_classes: int = 35, **kwargs) -> nn.Module:
    return BalancedSoftmaxTverskyLoss(tau=0.5, num_classes=num_classes)


@register_loss("bst_tau15", "BalancedSoftmaxTversky with τ=1.5")
def build_bst_tau15(num_classes: int = 35, **kwargs) -> nn.Module:
    return BalancedSoftmaxTverskyLoss(tau=1.5, num_classes=num_classes)


# ============================================================================
# 4. MASKING ABLATION (within balanced softmax framework)
# ============================================================================

@register_loss("bst_no_bbox", "BalancedSoftmaxTversky WITHOUT bbox masking")
def build_bst_no_bbox(num_classes: int = 35, **kwargs) -> nn.Module:
    """bbox_bg_weight=1.0 effectively disables spatial masking."""
    return BalancedSoftmaxTverskyLoss(
        num_classes=num_classes,
        bbox_bg_weight=1.0,
        bbox_pad_fraction=0.0,
    )


@register_loss("bst_bbox_loose", "BalancedSoftmaxTversky with loose bbox (pad=0.2, bg=0.1)")
def build_bst_bbox_loose(num_classes: int = 35, **kwargs) -> nn.Module:
    return BalancedSoftmaxTverskyLoss(
        num_classes=num_classes,
        bbox_pad_fraction=0.2,
        bbox_bg_weight=0.1,
    )


@register_loss("bst_masksup03", "BalancedSoftmaxTversky + masksup ratio=0.3")
def build_bst_masksup03(num_classes: int = 35, **kwargs) -> nn.Module:
    return BalancedSoftmaxTverskyLoss(
        num_classes=num_classes,
        masksup_ratio=0.3,
    )


@register_loss("bst_masksup03_no_bbox", "masksup=0.3 but NO bbox masking")
def build_bst_masksup03_no_bbox(num_classes: int = 35, **kwargs) -> nn.Module:
    return BalancedSoftmaxTverskyLoss(
        num_classes=num_classes,
        masksup_ratio=0.3,
        bbox_bg_weight=1.0,
        bbox_pad_fraction=0.0,
    )


# ============================================================================
# 5. HIGH-α TVERSKY (precision-boosting, from R2 findings)
# ============================================================================

@register_loss("tversky_a08_b04", "Tversky α=0.8, β=0.4 (strong precision bias)")
def build_tversky_a08_b04(num_classes: int = 35, **kwargs) -> nn.Module:
    return PartialTverskyLoss(alpha=0.8, beta=0.4, num_classes=num_classes)


@register_loss("tversky_a08_b06", "Tversky α=0.8, β=0.6 (precision + high FN penalty)")
def build_tversky_a08_b06(num_classes: int = 35, **kwargs) -> nn.Module:
    return PartialTverskyLoss(alpha=0.8, beta=0.6, num_classes=num_classes)


# ============================================================================
# 6. τ=2.0 (strong logit adjustment)
# ============================================================================

@register_loss("bst_tau20", "BalancedSoftmaxTversky with τ=2.0 (strong)")
def build_bst_tau20(num_classes: int = 35, **kwargs) -> nn.Module:
    return BalancedSoftmaxTverskyLoss(tau=2.0, num_classes=num_classes)


# ============================================================================
# 7. FOCAL TVERSKY LOSS (Abraham & Khan, 2019)
#    (1 - Tversky)^γ — focuses training on hard classes/voxels
# ============================================================================

@register_loss("focal_tversky", "Focal Tversky (γ=0.75) — down-weights easy classes")
def build_focal_tversky(
    gamma: float = 0.75,
    alpha: float = 0.6,
    beta: float = 0.4,
    num_classes: int = 35,
    **kwargs,
) -> nn.Module:
    return FocalTverskyLoss(
        gamma=gamma, alpha=alpha, beta=beta, num_classes=num_classes,
    )


@register_loss("focal_tversky_g05", "Focal Tversky (γ=0.5) — mild focal")
def build_focal_tversky_g05(num_classes: int = 35, **kwargs) -> nn.Module:
    return FocalTverskyLoss(gamma=0.5, alpha=0.6, beta=0.4, num_classes=num_classes)


# ============================================================================
# 8. ASYMMETRIC UNIFIED FOCAL LOSS (Yeung et al., MedIA 2022)
#    Combines distribution-based (Focal Tversky) + region-based (Dice Focal)
# ============================================================================

@register_loss(
    "unified_focal",
    "Asymmetric Unified Focal Loss (Yeung 2022) — SOTA compound loss",
)
def build_unified_focal(
    delta: float = 0.6,
    gamma_dist: float = 0.75,
    gamma_region: float = 0.75,
    weight_dist: float = 0.5,
    num_classes: int = 35,
    **kwargs,
) -> nn.Module:
    return AsymmetricUnifiedFocalLoss(
        delta=delta,
        gamma_dist=gamma_dist,
        gamma_region=gamma_region,
        weight_dist=weight_dist,
        num_classes=num_classes,
    )


# ============================================================================
# 9. BOUNDARY-WEIGHTED TVERSKY (distance-transform upweighting near boundaries)
# ============================================================================

@register_loss(
    "boundary_tversky",
    "Boundary-weighted Tversky — upweights loss near membrane boundaries",
)
def build_boundary_tversky(
    alpha: float = 0.6,
    beta: float = 0.4,
    boundary_weight: float = 5.0,
    boundary_sigma: float = 3.0,
    num_classes: int = 35,
    **kwargs,
) -> nn.Module:
    return BoundaryWeightedTverskyLoss(
        alpha=alpha,
        beta=beta,
        boundary_weight=boundary_weight,
        boundary_sigma=boundary_sigma,
        num_classes=num_classes,
    )


def build_loss(name: str, **kwargs) -> nn.Module:
    """Build a loss function by name.

    Args:
        name: Registered loss name.
        **kwargs: Arguments forwarded to the builder function.

    Returns:
        nn.Module loss function.
    """
    if name not in LOSS_REGISTRY:
        available = ", ".join(sorted(LOSS_REGISTRY.keys()))
        raise ValueError(f"Unknown loss '{name}'. Available: {available}")
    return LOSS_REGISTRY[name]["builder"](**kwargs)


def list_losses() -> None:
    """Print all registered losses."""
    print(f"{'Name':<30} {'Description'}")
    print("-" * 80)
    for name, info in sorted(LOSS_REGISTRY.items()):
        print(f"{name:<30} {info['description']}")
