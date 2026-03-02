"""
Model zoo for ablation experiments.

Registry of all model architectures available for training.
Supports both 2D (CSC) and 3D (MONAI) architectures.

All models take (in_channels, num_classes, **kwargs) and return logits.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import math

import torch
import torch.nn as nn

# Add CSC source to path
CSC_SRC = Path(__file__).resolve().parents[2] / "src"
if str(CSC_SRC) not in sys.path:
    sys.path.insert(0, str(CSC_SRC))

MODEL_REGISTRY: Dict[str, dict] = {}


def register_model(name: str, ndim: int, description: str = ""):
    """Decorator to register a model builder function."""
    def wrapper(fn):
        MODEL_REGISTRY[name] = {
            "builder": fn,
            "ndim": ndim,
            "description": description,
        }
        return fn
    return wrapper


# ============================================================================
# 2D MODELS (from CSC model zoo)
# ============================================================================

@register_model("resnet_2d", ndim=2, description="CSC ResNet 2D (~7.8M params)")
def build_resnet_2d(num_classes: int = 35, in_channels: int = 1, **kwargs) -> nn.Module:
    from cellmap_segmentation_challenge.models import ResNet
    return ResNet(
        ndims=2,
        input_nc=in_channels,
        output_nc=num_classes,
        ngf=kwargs.get("ngf", 64),
        n_blocks=kwargs.get("n_blocks", 6),
        n_downsampling=kwargs.get("n_downsampling", 2),
    )


@register_model("unet_2d", ndim=2, description="CSC UNet 2D (~31M params)")
def build_unet_2d(num_classes: int = 35, in_channels: int = 1, **kwargs) -> nn.Module:
    from cellmap_segmentation_challenge.models import UNet_2D
    return UNet_2D(
        n_channels=in_channels,
        n_classes=num_classes,
        use_instancenorm=kwargs.get("use_instancenorm", False),
        dropout=kwargs.get("dropout", 0.0),
    )


@register_model("swin_2d", ndim=2, description="CSC SwinTransformer 2D (~36M params)")
def build_swin_2d(num_classes: int = 35, in_channels: int = 1, **kwargs) -> nn.Module:
    from cellmap_segmentation_challenge.models import SwinTransformer
    return SwinTransformer(
        patch_size=kwargs.get("patch_size", [4, 4]),
        embed_dim=kwargs.get("embed_dim", 96),
        depths=kwargs.get("depths", [2, 2, 6, 2]),
        num_heads=kwargs.get("num_heads", [3, 6, 12, 24]),
        window_size=kwargs.get("window_size", [8, 8]),  # 8 aligns with 256→64→32→16→8
        num_classes=num_classes,
        dropout=kwargs.get("dropout", 0.1),
        attention_dropout=kwargs.get("attention_dropout", 0.0),  # Paper uses 0.0; stochastic_depth is the primary regularizer
        stochastic_depth_prob=kwargs.get("stochastic_depth_prob", 0.2),
    )


@register_model("vit_2d", ndim=2, description="CSC ViTVNet 2D (~105M params)")
def build_vit_2d(num_classes: int = 35, in_channels: int = 1, **kwargs) -> nn.Module:
    from cellmap_segmentation_challenge.models import ViTVNet2D, get_vit_config_2d
    config = get_vit_config_2d("base")
    config["img_size"] = kwargs.get("img_size", 256)
    # Keep default patch_size=16 — Embeddings2D divides by down_factor(2) internally
    # so actual patch conv is 8×8 on the 64×64 CNN output → 64 tokens
    # (Previous bug: setting patch_size=64 created a 32×32 conv → 201M params, OOM)
    return ViTVNet2D(config=config, in_channels=in_channels, num_classes=num_classes)


# ============================================================================
# 3D MODELS (from MONAI)
# ============================================================================

@register_model("segresnet_3d", ndim=3, description="MONAI SegResNetDS 3D (~87M params with 5-level encoder)")
def build_segresnet_3d(
    num_classes: int = 35, in_channels: int = 1, **kwargs
) -> nn.Module:
    from monai.networks.nets import SegResNetDS
    # MONAI Auto3DSeg official defaults: 5-level encoder, instance norm, deep supervision.
    # See: research-contributions/auto3dseg/algorithm_templates/segresnet/configs/hyper_parameters.yaml
    return SegResNetDS(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=num_classes,
        init_filters=kwargs.get("init_filters", 32),
        blocks_down=kwargs.get("blocks_down", (1, 2, 2, 4, 4)),
        blocks_up=kwargs.get("blocks_up", (1, 1, 1, 1)),
        dsdepth=kwargs.get("dsdepth", 4),  # 4 = deep supervision (3 aux outputs)
        norm=kwargs.get("norm", "INSTANCE"),  # instance norm standard for small-batch medical seg
    )


@register_model("swinunetr_3d", ndim=3, description="MONAI SwinUNETR 3D (~62M params)")
def build_swinunetr_3d(
    num_classes: int = 35, in_channels: int = 1, **kwargs
) -> nn.Module:
    from monai.networks.nets import SwinUNETR
    # img_size was removed in MONAI >= 1.5; pop it so it doesn't get forwarded
    kwargs.pop("img_size", None)
    return SwinUNETR(
        in_channels=in_channels,
        out_channels=num_classes,
        feature_size=kwargs.get("feature_size", 48),
        drop_rate=kwargs.get("drop_rate", 0.0),
        attn_drop_rate=kwargs.get("attn_drop_rate", 0.0),
        use_checkpoint=kwargs.get("use_checkpoint", False),  # gradient checkpointing saves VRAM
        spatial_dims=3,
    )


@register_model("unet_3d", ndim=3, description="CSC UNet 3D (~31M params)")
def build_unet_3d(num_classes: int = 35, in_channels: int = 1, **kwargs) -> nn.Module:
    from cellmap_segmentation_challenge.models import UNet_3D
    return UNet_3D(
        n_channels=in_channels,
        n_classes=num_classes,
        use_instancenorm=kwargs.get("use_instancenorm", False),
        dropout=kwargs.get("dropout", 0.0),
    )


@register_model("resnet_3d", ndim=3, description="CSC ResNet 3D (~7.8M params)")
def build_resnet_3d(num_classes: int = 35, in_channels: int = 1, **kwargs) -> nn.Module:
    from cellmap_segmentation_challenge.models import ResNet
    return ResNet(
        ndims=3,
        input_nc=in_channels,
        output_nc=num_classes,
        ngf=kwargs.get("ngf", 64),
        n_blocks=kwargs.get("n_blocks", 6),
        n_downsampling=kwargs.get("n_downsampling", 2),
    )


@register_model("vitnet_3d", ndim=3, description="CSC ViTVNet 3D (~28M params)")
def build_vitnet_3d(num_classes: int = 35, in_channels: int = 1, **kwargs) -> nn.Module:
    from cellmap_segmentation_challenge.models import ViTVNet
    return ViTVNet(
        out_channels=num_classes,
        img_size=kwargs.get("img_size", (128, 128, 128)),
    )


def _find_last_conv(model: nn.Module):
    """Find the last Conv2d or Conv3d layer in a model (DFS order).

    Returns the layer and its parent module, or (None, None).
    """
    last_conv = None
    last_parent = None
    last_attr = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d)):
            last_conv = module
            # Walk the name to find parent
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, attr = parts
                last_parent = dict(model.named_modules())[parent_name]
            else:
                last_parent = model
                attr = parts[0]
            last_attr = attr
    return last_conv, last_parent, last_attr


def init_output_bias(model: nn.Module, bias_value: float) -> nn.Module:
    """Initialize the bias of the model's final conv layer.

    This implements the "prior probability" initialization from RetinaNet
    (Lin et al., 2017). For sigmoid outputs on sparse/imbalanced targets,
    initializing the final bias to log(p/(1-p)) where p is the expected
    foreground fraction prevents BCE-driven collapse to all-background
    predictions in the early training phase.

    For p≈0.05 (typical organelle sparsity), bias = log(0.05/0.95) ≈ -2.94.
    A value of -3.0 works well in practice (sigmoid(-3) ≈ 0.047).

    If the final conv has no bias parameter (bias=False), this function
    replaces it with an equivalent layer that has bias enabled.

    Args:
        model: The model to modify (in-place).
        bias_value: Value to set for all output biases (e.g., -3.0).

    Returns:
        The model (modified in-place).
    """
    last_conv, parent, attr = _find_last_conv(model)
    if last_conv is None:
        raise RuntimeError("Could not find any Conv2d/Conv3d in model")

    # If the layer has no bias, replace it with one that does
    if last_conv.bias is None:
        ConvClass = type(last_conv)  # nn.Conv2d or nn.Conv3d
        new_conv = ConvClass(
            in_channels=last_conv.in_channels,
            out_channels=last_conv.out_channels,
            kernel_size=last_conv.kernel_size,
            stride=last_conv.stride,
            padding=last_conv.padding,
            dilation=last_conv.dilation,
            groups=last_conv.groups,
            bias=True,
            padding_mode=last_conv.padding_mode,
        )
        # Copy the existing weights
        new_conv.weight.data.copy_(last_conv.weight.data)
        # Replace in parent module
        setattr(parent, attr, new_conv)
        last_conv = new_conv

    # Initialize bias
    nn.init.constant_(last_conv.bias, bias_value)
    return model


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a model by name.

    Args:
        name: Registered model name.
        **kwargs: Arguments forwarded to the builder.
            bias_init (float, optional): If provided, initialize the final
                conv layer's bias to this value. Useful for preventing
                BCE collapse on sparse targets. Typical value: -3.0.

    Returns:
        nn.Module model.
    """
    # Pop bias_init before forwarding to builder (it's not a model arg)
    bias_init = kwargs.pop("bias_init", None)

    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    model = MODEL_REGISTRY[name]["builder"](**kwargs)

    if bias_init is not None:
        init_output_bias(model, bias_init)

    return model


def list_models() -> None:
    """Print all registered models."""
    print(f"{'Name':<20} {'Dim':>3} {'Description'}")
    print("-" * 80)
    for name, info in sorted(MODEL_REGISTRY.items()):
        print(f"{name:<20} {info['ndim']:>2}D  {info['description']}")
