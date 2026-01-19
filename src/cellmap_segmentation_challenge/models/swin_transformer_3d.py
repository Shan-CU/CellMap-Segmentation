# 3D Swin Transformer for Volumetric Segmentation
# Extends the 2D Swin Transformer to handle 3D volumetric data

import math
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision.ops import StochasticDepth


class Permute3D(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class MLP3D(nn.Module):
    def __init__(self, in_features, hidden_features=None, activation_layer=nn.GELU, inplace=None, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features[0])
        self.act = activation_layer()
        self.fc2 = nn.Linear(hidden_features[0], hidden_features[1])
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _get_relative_position_bias_3d(
    relative_position_bias_table: torch.Tensor,
    relative_position_index: torch.Tensor,
    window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1] * window_size[2]
    relative_position_bias = relative_position_bias_table[relative_position_index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


def _patch_merging_pad_3d(x: torch.Tensor) -> torch.Tensor:
    D, H, W, _ = x.shape[-4:]
    x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2, 0, D % 2))
    
    x0 = x[..., 0::2, 0::2, 0::2, :]
    x1 = x[..., 1::2, 0::2, 0::2, :]
    x2 = x[..., 0::2, 1::2, 0::2, :]
    x3 = x[..., 1::2, 1::2, 0::2, :]
    x4 = x[..., 0::2, 0::2, 1::2, :]
    x5 = x[..., 1::2, 0::2, 1::2, :]
    x6 = x[..., 0::2, 1::2, 1::2, :]
    x7 = x[..., 1::2, 1::2, 1::2, :]
    
    x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
    return x


class PatchMerging3D(nn.Module):
    """Patch Merging Layer for 3D Swin Transformer."""
    
    def __init__(self, dim: int, norm_layer: Callable[..., nn.Module] = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x: Tensor):
        x = _patch_merging_pad_3d(x)
        x = self.reduction(x)
        x = self.norm(x)
        return x


def shifted_window_attention_3d(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
    training: bool = True,
) -> Tensor:
    """3D Window based multi-head self attention with relative position bias."""
    B, D, H, W, C = input.shape
    
    # Pad feature maps to multiples of window size
    pad_d = (window_size[0] - D % window_size[0]) % window_size[0]
    pad_h = (window_size[1] - H % window_size[1]) % window_size[1]
    pad_w = (window_size[2] - W % window_size[2]) % window_size[2]
    x = F.pad(input, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
    _, pad_D, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    if window_size[0] >= pad_D:
        shift_size[0] = 0
    if window_size[1] >= pad_H:
        shift_size[1] = 0
    if window_size[2] >= pad_W:
        shift_size[2] = 0

    # Cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    # Partition windows
    num_windows = (pad_D // window_size[0]) * (pad_H // window_size[1]) * (pad_W // window_size[2])
    x = x.view(B, pad_D // window_size[0], window_size[0],
               pad_H // window_size[1], window_size[1],
               pad_W // window_size[2], window_size[2], C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(B * num_windows, window_size[0] * window_size[1] * window_size[2], C)

    # Multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()
    
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    if logit_scale is not None:
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    
    attn = attn + relative_position_bias

    # Attention mask for shifted windows
    if sum(shift_size) > 0:
        attn_mask = x.new_zeros((pad_D, pad_H, pad_W))
        d_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        h_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        w_slices = ((0, -window_size[2]), (-window_size[2], -shift_size[2]), (-shift_size[2], None))
        count = 0
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    attn_mask[d[0]:d[1], h[0]:h[1], w[0]:w[1]] = count
                    count += 1
        
        attn_mask = attn_mask.view(pad_D // window_size[0], window_size[0],
                                    pad_H // window_size[1], window_size[1],
                                    pad_W // window_size[2], window_size[2])
        attn_mask = attn_mask.permute(0, 2, 4, 1, 3, 5).reshape(num_windows, window_size[0] * window_size[1] * window_size[2])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # Reverse windows
    x = x.view(B, pad_D // window_size[0], pad_H // window_size[1], pad_W // window_size[2],
               window_size[0], window_size[1], window_size[2], C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, pad_D, pad_H, pad_W, C)

    # Reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

    # Unpad
    x = x[:, :D, :H, :W, :].contiguous()
    return x


class ShiftedWindowAttention3D(nn.Module):
    """3D Shifted Window Attention."""
    
    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if len(window_size) != 3 or len(shift_size) != 3:
            raise ValueError("window_size and shift_size must be of length 3")
        
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False)
        )
        
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length : 2 * length].data.zero_()

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        relative_coords_d = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_h = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[2] - 1), self.window_size[2], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_d, relative_coords_h, relative_coords_w], indexing="ij")
        )
        relative_coords_table = relative_coords_table.permute(1, 2, 3, 0).contiguous().unsqueeze(0)

        relative_coords_table[:, :, :, :, 0] /= (self.window_size[0] - 1)
        relative_coords_table[:, :, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table[:, :, :, :, 2] /= (self.window_size[2] - 1)

        relative_coords_table *= 8
        relative_coords_table = (
            torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        )
        self.register_buffer("relative_coords_table", relative_coords_table)

    def define_relative_position_index(self):
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * self.window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1).flatten()
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias_3d(
            self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads),
            self.relative_position_index,
            self.window_size,
        )
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    def forward(self, x: Tensor) -> Tensor:
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention_3d(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            logit_scale=self.logit_scale,
            training=self.training,
        )


class SwinTransformerBlock3D(nn.Module):
    """3D Swin Transformer Block."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ShiftedWindowAttention3D(
            dim,
            window_size,
            shift_size,
            num_heads,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.norm2 = norm_layer(dim)
        self.mlp = MLP3D(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)

        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: Tensor):
        x = x + self.stochastic_depth(self.norm1(self.attn(x)))
        x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
        return x


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()
        if not hidden_channels:
            hidden_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UnetBlockUp3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, skip):
        x1 = self.up(x1)
        x = torch.cat([skip, x1], dim=1)
        return self.conv(x)


class SegmentationHead3D(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size=(2, 4, 4)):
        super().__init__()
        self.hidden_channels = in_channels // 2
        self.patch_size = patch_size
        
        # Calculate total upsampling needed based on patch size
        # patch_size = (D, H, W) determines how much we downsampled
        # We need to upsample by patch_size to get back to original resolution
        # But we do it in two steps
        
        # First upsample: (1, 2, 2) - handles H and W partially
        self.up1 = nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # Second upsample: (patch_size[0], 2, 2) - handles D and remaining H/W
        self.up2 = nn.Upsample(scale_factor=(patch_size[0], 2, 2), mode="trilinear", align_corners=True)
        self.conv2 = nn.Sequential(
            nn.Conv3d(self.hidden_channels, self.hidden_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(self.hidden_channels // 2),
            nn.ReLU(inplace=True),
        )
        
        self.final = nn.Conv3d(self.hidden_channels // 2, out_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.final(x)
        return x


class SwinTransformer3D(nn.Module):
    """
    3D Swin Transformer for Volumetric Segmentation.
    
    Extends the 2D Swin Transformer to handle 3D volumetric data with
    3D window attention and 3D patch merging.
    
    Parameters
    ----------
    patch_size : List[int]
        Patch size in each dimension [D, H, W].
    embed_dim : int
        Patch embedding dimension.
    depths : List[int]
        Depth of each Swin Transformer stage.
    num_heads : List[int]
        Number of attention heads in each stage.
    window_size : List[int]
        Window size for 3D attention [D, H, W].
    in_channels : int
        Number of input channels. Default: 1.
    mlp_ratio : float
        Ratio of MLP hidden dim to embedding dim. Default: 4.0.
    dropout : float
        Dropout rate. Default: 0.0.
    attention_dropout : float
        Attention dropout rate. Default: 0.0.
    stochastic_depth_prob : float
        Stochastic depth rate. Default: 0.1.
    num_classes : int
        Number of output classes. Default: 14.
    """
    
    def __init__(
        self,
        patch_size: List[int] = [2, 4, 4],
        embed_dim: int = 96,
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: List[int] = [4, 7, 7],
        in_channels: int = 1,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 14,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depths = depths

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.encoder_layers = nn.ModuleList()
        
        # Patch embedding: split volume into non-overlapping patches
        self.encoder_layers.append(
            nn.Sequential(
                nn.Conv3d(
                    in_channels, embed_dim,
                    kernel_size=tuple(patch_size),
                    stride=tuple(patch_size)
                ),
                Permute3D([0, 2, 3, 4, 1]),  # B, C, D, H, W -> B, D, H, W, C
                norm_layer(embed_dim),
            )
        )

        total_stage_blocks = sum(depths)
        stage_block_id = 0
        
        # Build Swin Transformer stages
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            
            for i_layer in range(depths[i_stage]):
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    SwinTransformerBlock3D(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                    )
                )
                stage_block_id += 1
            
            self.encoder_layers.append(nn.Sequential(*stage))
            
            # Add patch merging layer (except for last stage)
            if i_stage < (len(depths) - 1):
                self.encoder_layers.append(PatchMerging3D(dim, norm_layer))

        num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(num_features)

        # Decoder
        decoder_dims = [embed_dim * 2 ** i for i in range(len(depths))]
        decoder_dims = list(reversed(decoder_dims))

        self.up_blocks = nn.ModuleList()
        for i in range(len(decoder_dims) - 1):
            self.up_blocks.append(UnetBlockUp3D(decoder_dims[i], decoder_dims[i + 1]))

        self.head = SegmentationHead3D(decoder_dims[-1], self.num_classes, patch_size=patch_size)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out_features = []
        
        for layer in self.encoder_layers:
            x = layer(x)
            if not isinstance(layer, PatchMerging3D):
                # Convert from B, D, H, W, C to B, C, D, H, W
                out_features.append(x.permute(0, 4, 1, 2, 3))

        # Remove patch embedding output
        del out_features[0]

        # Reverse for decoder
        out_features = list(reversed(out_features))

        # Decoder with skip connections
        x = out_features[0]
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, out_features[i + 1])

        out = self.head(x)
        return out


if __name__ == "__main__":
    # Test the 3D Swin Transformer
    print("Testing SwinTransformer3D...")
    model = SwinTransformer3D(
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[4, 7, 7],
        num_classes=14
    )
    
    # Test input: B=1, C=1, D=32, H=128, W=128
    x = torch.randn(1, 1, 32, 128, 128)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
