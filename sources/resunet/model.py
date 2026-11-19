"""ResUNet: U-Net built from residual blocks.

Every stage is a residual block (two convolutions plus an identity shortcut,
with a 1x1 projection when the shape changes).  Downsampling uses strided
convolutions and upsampling uses transposed convolutions, so the network is
fully convolutional and keeps the U-Net skip structure.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CONFIG: Dict[str, object] = {
    "in_channels": 1,
    "out_channels": 1,
    "base_channels": 16,
    "depth": 4,
    "dropout": 0.0,
}


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        needs_projection = stride != 1 or in_channels != out_channels
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            if needs_projection else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + self.shortcut(x))


class ResUNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 16, depth: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.depth = depth
        self.multiple = 2 ** depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.stem = ResidualBlock(in_channels, channels[0])
        self.downs = nn.ModuleList(
            [ResidualBlock(channels[i], channels[i + 1], stride=2) for i in range(depth)]
        )
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(channels[k + 1], channels[k], 2, stride=2)
             for k in reversed(range(depth))]
        )
        self.decoders = nn.ModuleList(
            [ResidualBlock(channels[k] * 2, channels[k]) for k in reversed(range(depth))]
        )

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.multiple
        pad_w = (-width) % self.multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        skips = []
        features = self.stem(x)
        skips.append(features)
        for index, down in enumerate(self.downs):
            features = down(features)
            if index < self.depth - 1:
                skips.append(features)

        for index, (up, decoder) in enumerate(zip(self.ups, self.decoders)):
            skip = skips[len(skips) - 1 - index]
            features = up(features)
            if features.shape[-2:] != skip.shape[-2:]:
                features = F.interpolate(
                    features, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            features = decoder(torch.cat([features, skip], dim=1))

        logits = self.head(self.dropout(features))
        if pad_h or pad_w:
            logits = logits[:, :, :height, :width]
        return logits


def build_model(config: Optional[Dict[str, object]] = None) -> ResUNet:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    return ResUNet(
        in_channels=int(merged["in_channels"]),
        out_channels=int(merged["out_channels"]),
        base_channels=int(merged["base_channels"]),
        depth=int(merged["depth"]),
        dropout=float(merged.get("dropout", 0.0)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
