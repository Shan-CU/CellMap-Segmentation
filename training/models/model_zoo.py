"""
Model zoo for ablation experiments.

Registry of all model architectures available for training.
Supports both 2D (CSC) and 3D (MONAI) architectures.

All models take (in_channels, num_classes, **kwargs) and return logits.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import math

import torch
import torch.nn as nn

# Add CSC source to path
CSC_SRC = Path(__file__).resolve().parents[2] / "src"
if str(CSC_SRC) not in sys.path:
    sys.path.insert(0, str(CSC_SRC))

MODEL_REGISTRY: Dict[str, dict] = {}


# ============================================================================
# Per-class prior probability bias initialization (RetinaNet-style)
# ============================================================================
# Voxel counts from CLASS_REFERENCE.md (42.3B total annotated voxels).
# Order MUST match get_tested_classes() from cellmap_segmentation_challenge.
# Group classes use their pre-computed OR voxel counts (Section 6).
#
# bias_c = log(π_c / (1 - π_c))  where π_c = voxels_c / total_voxels
# Clamped to [-10, 0] to avoid numerical extremes.
#
# References:
#   - Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
#   - Isensee et al., nnU-Net v2 (2024) — same principle
# ============================================================================

_TOTAL_VOXELS = 42_341_742_368

# fmt: off
# (class_name, voxel_count) — 48 entries, order matches get_tested_classes()
_CLASS_VOXEL_COUNTS: list[tuple[str, int]] = [
    # === Atomic classes (0–28) ===
    ("ecs",        3_750_644_301),   # 0  — extracellular space
    ("pm",           365_966_884),   # 1  — plasma membrane
    ("mito_mem",     740_946_964),   # 2  — mitochondrial membrane
    ("mito_lum",     910_826_941),   # 3  — mitochondrial lumen
    ("mito_ribo",      1_643_455),   # 4  — mitochondrial ribosomes
    ("golgi_mem",     63_358_310),   # 5  — Golgi membrane
    ("golgi_lum",     90_271_273),   # 6  — Golgi lumen
    ("ves_mem",       20_218_267),   # 7  — vesicle membrane
    ("ves_lum",       11_410_160),   # 8  — vesicle lumen
    ("endo_mem",      97_744_082),   # 9  — endosome membrane
    ("endo_lum",     203_076_684),   # 10 — endosome lumen
    ("lyso_mem",      20_217_077),   # 11 — lysosome membrane
    ("lyso_lum",      64_241_359),   # 12 — lysosome lumen
    ("ld_mem",        14_896_191),   # 13 — lipid droplet membrane
    ("ld_lum",       120_758_049),   # 14 — lipid droplet lumen
    ("er_mem",       523_064_850),   # 15 — ER membrane
    ("er_lum",       654_688_548),   # 16 — ER lumen
    ("eres_mem",       5_456_843),   # 17 — ER exit site membrane
    ("eres_lum",       5_252_100),   # 18 — ER exit site lumen
    ("ne_mem",        56_357_675),   # 19 — nuclear envelope membrane
    ("ne_lum",        50_414_607),   # 20 — nuclear envelope lumen
    ("np_out",         4_086_527),   # 21 — nuclear pore outer ring
    ("np_in",          2_798_532),   # 22 — nuclear pore inner ring
    ("hchrom",       302_119_910),   # 23 — heterochromatin
    ("echrom",         5_585_955),   # 24 — euchromatin
    ("nucpl",        604_553_039),   # 25 — nucleoplasm
    ("mt_out",        38_248_677),   # 26 — microtubule outer wall
    ("cyto",       7_557_206_558),   # 27 — cytoplasm
    ("mt_in",         17_017_019),   # 28 — microtubule lumen
    # === Group / composite classes (29–47) ===
    ("nuc",        3_533_146_803),   # 29 — nucleus (OR of 10 nuclear atomics)
    ("golgi",        153_629_583),   # 30 — all Golgi
    ("ves",           31_628_427),   # 31 — all vesicles
    ("endo",         300_820_766),   # 32 — all endosomes
    ("lyso",          84_458_436),   # 33 — all lysosomes
    ("ld",           142_488_418),   # 34 — all lipid droplets
    ("eres",          10_708_943),   # 35 — all ER exit sites
    ("perox_mem",      5_857_060),   # 36 — peroxisome membrane
    ("perox_lum",     16_430_634),   # 37 — peroxisome lumen
    ("perox",         24_342_923),   # 38 — all peroxisomes
    ("mito",       1_735_204_611),   # 39 — all mitochondria
    ("er",         1_302_119_682),   # 40 — entire ER system (incl. NE)
    ("ne",           113_657_341),   # 41 — entire nuclear envelope
    ("np",             6_885_059),   # 42 — nuclear pores
    ("chrom",        324_920_357),   # 43 — all chromatin
    ("mt",            55_265_696),   # 44 — all microtubules
    ("cell",      12_589_379_127),   # 45 — everything intracellular
    ("er_mem_all",   584_879_368),   # 46 — all ER-related membranes
    ("ne_mem_all",    63_242_734),   # 47 — NE membrane + pores
]
# fmt: on

_BIAS_CLAMP_MIN = -10.0
_BIAS_CLAMP_MAX = 0.0


def get_per_class_bias_init(num_classes: int = 48) -> list[float]:
    """Compute per-class bias init from dataset voxel frequencies.

    Each bias is: b_c = clamp(log(π_c / (1 - π_c)), -10, 0)
    where π_c is the global voxel fraction for class c.

    This ensures each sigmoid output starts near the true class prior,
    preventing BCE-driven collapse on both common classes (cyto/ecs start
    near their true ~18-30% prior) and rare classes (np_in starts at
    ~0.007% instead of the 4.7% from a uniform -3.0 bias).

    Args:
        num_classes: Number of output channels (must be <= 48).

    Returns:
        List of bias values, one per output channel.
    """
    if num_classes > len(_CLASS_VOXEL_COUNTS):
        raise ValueError(
            f"num_classes={num_classes} exceeds the {len(_CLASS_VOXEL_COUNTS)} "
            f"known classes. Cannot compute per-class bias."
        )
    biases = []
    for i in range(num_classes):
        _name, voxels = _CLASS_VOXEL_COUNTS[i]
        frac = voxels / _TOTAL_VOXELS
        if frac <= 0:
            b = _BIAS_CLAMP_MIN
        else:
            b = math.log(frac / (1.0 - frac))
        b = max(_BIAS_CLAMP_MIN, min(_BIAS_CLAMP_MAX, b))
        biases.append(b)
    return biases


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
        dropout=kwargs.get("dropout", 0.0),  # Was 0.1; removed — over-regularized with wd+stochastic_depth
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


def init_output_bias(
    model: nn.Module,
    bias_value: Union[float, Sequence[float]],
) -> nn.Module:
    """Initialize the bias of the model's final conv layer.

    Implements "prior probability" initialization (RetinaNet, Lin et al. 2017).

    For sigmoid outputs on imbalanced multi-label targets, initializing each
    output channel's bias to log(π_c / (1 - π_c)) — where π_c is the expected
    foreground fraction for class c — prevents BCE-driven gradient collapse.

    Supports two modes:
      - Scalar: all output biases set to the same value (e.g., -3.0).
      - Per-class vector: each output channel gets its own bias from the
        dataset frequency distribution. Use get_per_class_bias_init().

    If the final conv has no bias parameter (bias=False), this function
    replaces it with an equivalent layer that has bias enabled.

    Args:
        model: The model to modify (in-place).
        bias_value: Either a single float (uniform) or a sequence of floats
                    (one per output channel). Length must match out_channels.

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

    # Initialize bias — scalar or per-channel vector
    if isinstance(bias_value, (list, tuple)):
        if len(bias_value) != last_conv.out_channels:
            raise ValueError(
                f"Per-class bias vector length ({len(bias_value)}) != "
                f"output channels ({last_conv.out_channels})"
            )
        with torch.no_grad():
            last_conv.bias.copy_(torch.tensor(bias_value, dtype=last_conv.bias.dtype))
    else:
        nn.init.constant_(last_conv.bias, float(bias_value))

    return model


def build_model(name: str, **kwargs) -> nn.Module:
    """Build a model by name.

    Args:
        name: Registered model name.
        **kwargs: Arguments forwarded to the builder.
            bias_init_mode (str, optional): How to initialize the final conv bias.
                "none"      – skip bias init (default if omitted).
                "uniform"   – set all output biases to a single scalar; requires
                              ``bias_init`` kwarg (e.g., -3.0).
                "per_class" – set each output channel's bias to
                              log(π_c / (1 - π_c)) from the dataset frequency
                              distribution (see get_per_class_bias_init).
            bias_init (float, optional): Scalar value for mode="uniform".

    Returns:
        nn.Module model.
    """
    # Pop bias-init kwargs before forwarding to builder (not model args)
    bias_init_mode = kwargs.pop("bias_init_mode", "none")
    bias_init = kwargs.pop("bias_init", None)

    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model '{name}'. Available: {available}")

    model = MODEL_REGISTRY[name]["builder"](**kwargs)

    # Apply bias initialisation
    if bias_init_mode == "uniform":
        if bias_init is None:
            raise ValueError("bias_init_mode='uniform' requires --bias_init <float>")
        init_output_bias(model, float(bias_init))
    elif bias_init_mode == "per_class":
        # Builders use num_classes (CSC models) or out_channels (MONAI).
        num_classes = kwargs.get("num_classes", kwargs.get("out_channels", 48))
        per_class_biases = get_per_class_bias_init(num_classes)
        init_output_bias(model, per_class_biases)
    elif bias_init_mode != "none":
        raise ValueError(
            f"Unknown bias_init_mode '{bias_init_mode}'. "
            f"Choose from: none, uniform, per_class"
        )

    return model


def list_models() -> None:
    """Print all registered models."""
    print(f"{'Name':<20} {'Dim':>3} {'Description'}")
    print("-" * 80)
    for name, info in sorted(MODEL_REGISTRY.items()):
        print(f"{name:<20} {info['ndim']:>2}D  {info['description']}")
