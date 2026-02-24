"""
Boundary-Weighted Tversky Loss.

Upweights loss near ground-truth boundaries using a Gaussian distance transform.
Specifically designed for membrane segmentation tasks where boundary accuracy
is critical (er_mem, mito_mem, pm, ne_mem, etc.).

For each annotated class with foreground:
1. Compute binary erosion to find boundary voxels
2. Apply Gaussian blur to create soft boundary weight map
3. Weight map = 1 + (boundary_weight - 1) * boundary_map

The Tversky loss is then computed with this spatial weight map.

This avoids scipy dependency by using morphological operations in pure PyTorch.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryWeightedTverskyLoss(nn.Module):
    """Tversky loss with boundary upweighting via distance transform.

    Args:
        alpha: FP weight for Tversky.
        beta: FN weight for Tversky.
        boundary_weight: Multiplier at exact boundary (decays with distance).
        boundary_sigma: Gaussian sigma for boundary decay (in voxels).
        smooth: Smoothing constant.
        num_classes: Number of output classes.
    """

    def __init__(
        self,
        alpha: float = 0.6,
        beta: float = 0.4,
        boundary_weight: float = 5.0,
        boundary_sigma: float = 3.0,
        smooth: float = 1e-6,
        num_classes: int = 35,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.boundary_weight = boundary_weight
        self.boundary_sigma = boundary_sigma
        self.smooth = smooth
        self.num_classes = num_classes
        self._annotation_mask: Optional[torch.Tensor] = None

        # Pre-build Gaussian kernel for boundary smoothing
        self._kernel = None
        self._ndim = None

    def set_annotation_mask(self, mask: torch.Tensor) -> None:
        self._annotation_mask = mask

    def _get_gaussian_kernel(self, ndim: int, device: torch.device) -> torch.Tensor:
        """Build Gaussian smoothing kernel (cached)."""
        if self._kernel is not None and self._ndim == ndim:
            return self._kernel.to(device)

        sigma = self.boundary_sigma
        ksize = int(2 * sigma + 1) | 1  # ensure odd
        coords = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        g1d = torch.exp(-0.5 * (coords / sigma) ** 2)
        g1d = g1d / g1d.sum()

        if ndim == 2:
            kernel = g1d.unsqueeze(1) * g1d.unsqueeze(0)
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        else:  # 3D
            kernel = g1d.unsqueeze(1).unsqueeze(2) * g1d.unsqueeze(0).unsqueeze(2) * g1d.unsqueeze(0).unsqueeze(1)
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        self._kernel = kernel
        self._ndim = ndim
        return kernel.to(device)

    def _compute_boundary_weights(self, target: torch.Tensor) -> torch.Tensor:
        """Compute per-voxel boundary weight map.

        Uses morphological boundary detection:
        boundary = target XOR erode(target)
        Then smooths with Gaussian to create soft weight falloff.

        Args:
            target: (B, C, *spatial) binary targets.

        Returns:
            Weight map (B, C, *spatial), values in [1.0, boundary_weight].
        """
        ndim = target.ndim - 2  # spatial dims
        kernel = self._get_gaussian_kernel(ndim, target.device)
        pad_size = kernel.shape[-1] // 2

        B, C = target.shape[:2]
        # Process all channels as a batch for efficiency
        flat = target.view(B * C, 1, *target.shape[2:]).float()

        # Erosion via min-pool
        if ndim == 2:
            eroded = -F.max_pool2d(-flat, kernel_size=3, stride=1, padding=1)
        else:
            eroded = -F.max_pool3d(-flat, kernel_size=3, stride=1, padding=1)

        # Boundary = foreground pixels that are NOT interior
        boundary = (flat - eroded).clamp(0, 1)

        # Smooth boundary with Gaussian
        if ndim == 2:
            smooth_boundary = F.conv2d(boundary, kernel, padding=pad_size)
        else:
            smooth_boundary = F.conv3d(boundary, kernel, padding=pad_size)

        # Normalize to [0, 1] and scale
        smooth_boundary = smooth_boundary.clamp(0, 1)

        # Weight map: 1.0 everywhere + extra weight near boundaries
        weight_map = 1.0 + (self.boundary_weight - 1.0) * smooth_boundary
        return weight_map.view(B, C, *target.shape[2:])

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = self._annotation_mask
        self._annotation_mask = None

        pred = torch.sigmoid(input.float())  # float32 for AMP safety
        target = target.float()
        spatial_dims = tuple(range(2, input.ndim))

        # Compute boundary weight map
        with torch.no_grad():
            w = self._compute_boundary_weights(target)

        # Weighted Tversky
        tp = (w * pred * target).sum(dim=spatial_dims)
        fp = (w * pred * (1.0 - target)).sum(dim=spatial_dims)
        fn = (w * (1.0 - pred) * target).sum(dim=spatial_dims)

        denom = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky = (tp + self.smooth) / denom.clamp(min=self.smooth)
        per_channel_loss = 1.0 - tversky  # (B, C)

        if mask is not None:
            mask = mask.to(input.device)
            per_channel_loss = per_channel_loss * mask
            num_annotated = mask.sum(dim=1).clamp(min=1.0)
            per_sample = per_channel_loss.sum(dim=1) / num_annotated
        else:
            per_sample = per_channel_loss.mean(dim=1)

        return per_sample.mean()
