"""Classic four-level U-Net with a trainable RGB encoder."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, base_channels=64):
        super().__init__()
        widths = [base_channels * 2**i for i in range(5)]
        self.encoder = nn.ModuleList(
            DoubleConv(in_channels if i == 0 else widths[i - 1], width)
            for i, width in enumerate(widths)
        )
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList(
            nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2)
            for i in range(4, 0, -1)
        )
        self.decoder = nn.ModuleList(
            DoubleConv(widths[i], widths[i - 1]) for i in range(4, 0, -1)
        )
        self.head = nn.Conv2d(base_channels, num_classes, 1)

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.encoder):
            x = block(x if i == 0 else self.pool(x))
            skips.append(x)
        for up, block, skip in zip(self.up, self.decoder, reversed(skips[:-1])):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = block(torch.cat([skip, x], dim=1))
        return self.head(x)
