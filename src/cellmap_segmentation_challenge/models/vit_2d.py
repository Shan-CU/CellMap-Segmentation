# 2D ViT-V-Net for Image Segmentation
# Adapts the 3D ViT-V-Net architecture to handle 2D images

import math
from typing import Optional, List
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2dReLU(nn.Sequential):
    """2D Convolution followed by ReLU activation."""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batchnorm,
        )
        relu = nn.LeakyReLU(inplace=True)
        
        if use_batchnorm:
            bn = nn.BatchNorm2d(out_channels)
        else:
            bn = nn.Identity()
        
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Embeddings2D(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """
    
    def __init__(self, config, in_channels=512):
        super(Embeddings2D, self).__init__()
        self.config = config
        down_factor = config.get("down_factor", 2)
        patch_size = (config["patch_size"] // down_factor, config["patch_size"] // down_factor)
        n_patches = int((config["img_size"] // down_factor // patch_size[0]) * 
                        (config["img_size"] // down_factor // patch_size[1]))
        self.hybrid_model = CNNEncoder2D(config, in_channels=in_channels)
        in_channels = self.hybrid_model.out_channels
        self.patch_embeddings = nn.Conv2d(
            in_channels=in_channels,
            out_channels=config["hidden_size"],
            kernel_size=patch_size,
            stride=patch_size
        )
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config["hidden_size"]))
        self.dropout = nn.Dropout(config.get("dropout_rate", 0.1))

    def forward(self, x):
        x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Attention(nn.Module):
    """Multi-head attention module."""
    
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config["num_heads"]
        self.attention_head_size = int(config["hidden_size"] / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(config["hidden_size"], self.all_head_size)
        self.key = nn.Linear(config["hidden_size"], self.all_head_size)
        self.value = nn.Linear(config["hidden_size"], self.all_head_size)
        
        self.out = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.attn_dropout = nn.Dropout(config.get("attention_dropout_rate", 0.0))
        self.proj_dropout = nn.Dropout(config.get("dropout_rate", 0.1))
        
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class MLP(nn.Module):
    """Feed-forward network in transformer."""
    
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config["hidden_size"], config["mlp_dim"])
        self.fc2 = nn.Linear(config["mlp_dim"], config["hidden_size"])
        self.act_fn = nn.GELU()
        self.dropout = nn.Dropout(config.get("dropout_rate", 0.1))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.attention_norm = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        self.ffn = MLP(config)
        self.attn = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    """Transformer encoder."""
    
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(config["hidden_size"], eps=1e-6)
        for _ in range(config["num_layers"]):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    """Transformer module combining embeddings and encoder."""
    
    def __init__(self, config, in_channels=512):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings2D(config, in_channels=in_channels)
        self.encoder = Encoder(config)
        # Store CNN feature channels for decoder skip connections
        self.feature_channels = self.embeddings.hybrid_model.feature_channels

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded = self.encoder(embedding_output)
        return encoded, features


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection."""
    
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
        use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    """Cascaded UNet-style decoder."""
    
    def __init__(self, config, feature_channels):
        """
        Args:
            config: Configuration dict with hidden_size and decoder_channels
            feature_channels: List of channel sizes from CNN encoder features (in reverse order)
        """
        super().__init__()
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config["hidden_size"],
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        
        decoder_channels = config.get("decoder_channels", (256, 128, 64, 16))
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        
        # Build skip channels list based on actual feature channels
        # feature_channels is already in reverse order (highest resolution first)
        skip_channels = []
        for i in range(len(decoder_channels)):
            if i < len(feature_channels):
                skip_channels.append(feature_channels[i])
            else:
                skip_channels.append(0)
        
        self.blocks = nn.ModuleList([
            DecoderBlock(in_ch, out_ch, skip_ch)
            for in_ch, out_ch, skip_ch in zip(in_channels, out_channels, skip_channels)
        ])

    def forward(self, hidden_states, features):
        B, n_patch, hidden = hidden_states.size()
        
        # Reshape to spatial dimensions
        h = w = int(math.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        
        # Decode with skip connections
        for i, decoder_block in enumerate(self.blocks):
            if i < len(features):
                skip = features[i]
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        
        return x


class CNNEncoder2D(nn.Module):
    """2D CNN Encoder for extracting features."""
    
    def __init__(self, config, in_channels=1):
        super(CNNEncoder2D, self).__init__()
        down_factor = config.get("down_factor", 2)
        
        self.down_num = int(math.log2(down_factor))
        self.base_channels = 64
        self.out_channels = self.base_channels * (2 ** (self.down_num + 1))
        
        # Track feature channels for decoder
        self.feature_channels = []
        
        # Build encoder stages
        self.enc_stages = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        
        # Initial layer
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.base_channels),
            nn.LeakyReLU(inplace=True),
        )
        
        # Encoder stages
        current_channels = self.base_channels
        for i in range(self.down_num + 1):
            out_ch = current_channels * 2
            
            self.enc_stages.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(inplace=True),
                )
            )
            
            if i < self.down_num:
                self.feature_channels.append(out_ch)
                self.down_convs.append(
                    nn.Conv2d(out_ch, out_ch, kernel_size=2, stride=2)
                )
            
            current_channels = out_ch
        
        self.out_channels = current_channels

    def forward(self, x):
        features = []
        x = self.init_conv(x)
        
        for i, enc_stage in enumerate(self.enc_stages):
            x = enc_stage(x)
            if i < self.down_num:
                features.insert(0, x)
                x = self.down_convs[i](x)
        
        return x, features


class SegmentationHead(nn.Sequential):
    """Final segmentation head."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.Upsample(scale_factor=upsampling, mode='bilinear', align_corners=True) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class ViTVNet2D(nn.Module):
    """
    2D Vision Transformer V-Net for Image Segmentation.
    
    Adapts the 3D ViT-V-Net architecture for 2D images, combining:
    - CNN encoder for hierarchical feature extraction
    - Vision Transformer for global context modeling
    - V-Net style decoder with skip connections
    
    Parameters
    ----------
    config : dict
        Configuration dictionary with the following keys:
        - img_size: Input image size (assumes square images)
        - patch_size: Size of patches for transformer
        - hidden_size: Transformer hidden dimension
        - num_layers: Number of transformer blocks
        - num_heads: Number of attention heads
        - mlp_dim: MLP hidden dimension
        - decoder_channels: Decoder channel dimensions
        - dropout_rate: Dropout rate
    in_channels : int
        Number of input channels. Default: 1.
    num_classes : int
        Number of output classes. Default: 14.
    """
    
    def __init__(
        self,
        config: Optional[dict] = None,
        in_channels: int = 1,
        num_classes: int = 14,
    ):
        super(ViTVNet2D, self).__init__()
        
        # Default configuration
        if config is None:
            config = {
                "img_size": 128,
                "patch_size": 16,
                "hidden_size": 768,
                "num_layers": 12,
                "num_heads": 12,
                "mlp_dim": 3072,
                "decoder_channels": (256, 128, 64, 16),
                "dropout_rate": 0.1,
                "attention_dropout_rate": 0.0,
                "down_factor": 2,
            }
        
        self.config = config
        self.num_classes = num_classes
        
        # Transformer encoder
        self.transformer = Transformer(config, in_channels=in_channels)
        
        # Decoder - pass feature channels for skip connection sizing
        self.decoder = DecoderCup(config, self.transformer.feature_channels)
        
        # Segmentation head
        self.seg_head = SegmentationHead(
            in_channels=config["decoder_channels"][-1],
            out_channels=num_classes,
            kernel_size=3,
        )

    def forward(self, x):
        # Pass through transformer
        encoded, features = self.transformer(x)
        
        # Decode
        decoded = self.decoder(encoded, features)
        
        # Upsample to original size if needed
        if decoded.shape[2:] != x.shape[2:]:
            decoded = F.interpolate(decoded, size=x.shape[2:], mode='bilinear', align_corners=True)
        
        # Segmentation head
        out = self.seg_head(decoded)
        
        return out


def get_vit_config_2d(name="base"):
    """Get predefined ViT configuration for 2D."""
    configs = {
        "tiny": {
            "img_size": 128,
            "patch_size": 16,
            "hidden_size": 192,
            "num_layers": 6,
            "num_heads": 3,
            "mlp_dim": 768,
            "decoder_channels": (128, 64, 32, 16),
            "dropout_rate": 0.1,
            "attention_dropout_rate": 0.0,
            "down_factor": 2,
        },
        "small": {
            "img_size": 128,
            "patch_size": 16,
            "hidden_size": 384,
            "num_layers": 6,
            "num_heads": 6,
            "mlp_dim": 1536,
            "decoder_channels": (192, 96, 48, 16),
            "dropout_rate": 0.1,
            "attention_dropout_rate": 0.0,
            "down_factor": 2,
        },
        "base": {
            "img_size": 128,
            "patch_size": 16,
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "mlp_dim": 3072,
            "decoder_channels": (256, 128, 64, 16),
            "dropout_rate": 0.1,
            "attention_dropout_rate": 0.0,
            "down_factor": 2,
        },
    }
    return configs.get(name, configs["base"])


if __name__ == "__main__":
    # Test the 2D ViT-V-Net
    print("Testing ViTVNet2D...")
    
    config = get_vit_config_2d("base")
    model = ViTVNet2D(
        config=config,
        in_channels=1,
        num_classes=14,
    )
    
    # Test input: B=1, C=1, H=128, W=128
    x = torch.randn(1, 1, 128, 128)
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    print(f"Output shape: {y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
