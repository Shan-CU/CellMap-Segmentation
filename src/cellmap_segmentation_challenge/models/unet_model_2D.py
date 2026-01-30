# Adapted from:
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
# By Emma Avetissian, @aemmav

""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [Norm] => ReLU => [Dropout]) * 2
    
    Supports BatchNorm or InstanceNorm with optional Dropout for regularization.
    InstanceNorm is recommended for segmentation tasks with small batch sizes
    as it normalizes per-sample rather than per-batch.
    """

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        mid_channels: int = None,
        use_instancenorm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Choose normalization layer
        norm_layer = nn.InstanceNorm2d if use_instancenorm else nn.BatchNorm2d
        
        # Build sequential with optional dropout
        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        
        layers.extend([
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        ])
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        use_instancenorm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(
                in_channels, out_channels,
                use_instancenorm=use_instancenorm,
                dropout=dropout
            )
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        bilinear: bool = True,
        use_instancenorm: bool = False,
        dropout: float = 0.0
    ):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, in_channels // 2,
                use_instancenorm=use_instancenorm,
                dropout=dropout
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(
                in_channels, out_channels,
                use_instancenorm=use_instancenorm,
                dropout=dropout
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# Original source code from:
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py


""" Full assembly of the parts to form the complete network """


class UNet_2D(nn.Module):
    """
    2D U-Net model.

    Adapted from:
        https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    By Emma Avetissian, @aemmav

    Parameters
    ----------
    n_channels : int
        Number of input channels.
    n_classes : int
        Number of output channels.
    trilinear : bool
        Whether to use trilinear interpolation or not.
    use_instancenorm : bool
        If True, use InstanceNorm instead of BatchNorm.
        Recommended for segmentation with small batch sizes.
    dropout : float
        Dropout probability (0.0 = no dropout). Recommended: 0.1-0.2.
    """

    def __init__(
        self, 
        n_channels: int, 
        n_classes: int, 
        trilinear: bool = False,
        use_instancenorm: bool = False,
        dropout: float = 0.0
    ):
        super(UNet_2D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.trilinear = trilinear
        self.use_instancenorm = use_instancenorm
        self.dropout = dropout

        self.inc = DoubleConv(
            n_channels, 64, 
            use_instancenorm=use_instancenorm, 
            dropout=dropout
        )
        self.down1 = Down(64, 128, use_instancenorm=use_instancenorm, dropout=dropout)
        self.down2 = Down(128, 256, use_instancenorm=use_instancenorm, dropout=dropout)
        self.down3 = Down(256, 512, use_instancenorm=use_instancenorm, dropout=dropout)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor, use_instancenorm=use_instancenorm, dropout=dropout)
        self.up1 = Up(1024, 512 // factor, trilinear, use_instancenorm=use_instancenorm, dropout=dropout)
        self.up2 = Up(512, 256 // factor, trilinear, use_instancenorm=use_instancenorm, dropout=dropout)
        self.up3 = Up(256, 128 // factor, trilinear, use_instancenorm=use_instancenorm, dropout=dropout)
        self.up4 = Up(128, 64, trilinear, use_instancenorm=use_instancenorm, dropout=dropout)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
