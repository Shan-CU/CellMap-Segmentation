"""
Intensity augmentation transforms for EM data.

Cherry-picked from OrganelleSeg's augmentation pipeline. Applied to raw EM
inputs via cellmap-data's `train_raw_value_transforms` hook.

All transforms operate on float tensors in [0, 1] range and are compatible
with torchvision.transforms.v2.
"""

from __future__ import annotations

import torch
import torchvision.transforms.v2 as T


class RandomBrightness(torch.nn.Module):
    """Random additive brightness shift.

    Adds a uniform random offset in [-max_delta, +max_delta] to all voxels.
    Clamps output to [0, 1].

    Args:
        max_delta: Maximum brightness shift. Default: 0.1.
        p: Probability of applying the transform. Default: 0.5.
    """

    def __init__(self, max_delta: float = 0.1, p: float = 0.5):
        super().__init__()
        self.max_delta = max_delta
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        delta = (torch.rand(1).item() * 2 - 1) * self.max_delta
        return (x + delta).clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_delta={self.max_delta}, p={self.p})"


class RandomContrast(torch.nn.Module):
    """Random multiplicative contrast adjustment.

    Multiplies voxels by a uniform random factor in [lower, upper],
    recentered around the mean. Clamps output to [0, 1].

    Args:
        lower: Lower bound of contrast factor. Default: 0.8.
        upper: Upper bound of contrast factor. Default: 1.2.
        p: Probability of applying the transform. Default: 0.5.
    """

    def __init__(self, lower: float = 0.8, upper: float = 1.2, p: float = 0.5):
        super().__init__()
        self.lower = lower
        self.upper = upper
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        factor = self.lower + torch.rand(1).item() * (self.upper - self.lower)
        mean = x.mean()
        return ((x - mean) * factor + mean).clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(lower={self.lower}, "
            f"upper={self.upper}, p={self.p})"
        )


class RandomGaussianNoise(torch.nn.Module):
    """Additive Gaussian noise.

    Adds zero-mean Gaussian noise with σ sampled uniformly from
    [sigma_min, sigma_max]. Clamps output to [0, 1].

    Args:
        sigma_min: Minimum noise standard deviation. Default: 0.01.
        sigma_max: Maximum noise standard deviation. Default: 0.05.
        p: Probability of applying the transform. Default: 0.5.
    """

    def __init__(
        self, sigma_min: float = 0.01, sigma_max: float = 0.05, p: float = 0.5
    ):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        sigma = self.sigma_min + torch.rand(1).item() * (self.sigma_max - self.sigma_min)
        noise = torch.randn_like(x) * sigma
        return (x + noise).clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(sigma_min={self.sigma_min}, "
            f"sigma_max={self.sigma_max}, p={self.p})"
        )


class IntensityAugmentation(torch.nn.Module):
    """Composed intensity augmentation pipeline for EM data.

    Applies brightness, contrast, and Gaussian noise augmentations in sequence.
    Designed to be prepended/appended to cellmap-data's train_raw_value_transforms.

    Usage:
        from training.transforms.intensity import IntensityAugmentation
        aug = IntensityAugmentation()
        train_raw_value_transforms = T.Compose([
            T.ToDtype(torch.float, scale=True),
            NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            aug,
        ])
    """

    def __init__(
        self,
        brightness_delta: float = 0.1,
        contrast_lower: float = 0.8,
        contrast_upper: float = 1.2,
        noise_sigma_min: float = 0.01,
        noise_sigma_max: float = 0.05,
        p: float = 0.5,
    ):
        super().__init__()
        self.pipeline = T.Compose([
            RandomBrightness(max_delta=brightness_delta, p=p),
            RandomContrast(lower=contrast_lower, upper=contrast_upper, p=p),
            RandomGaussianNoise(sigma_min=noise_sigma_min, sigma_max=noise_sigma_max, p=p),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pipeline(x)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(\n  {self.pipeline}\n)"
