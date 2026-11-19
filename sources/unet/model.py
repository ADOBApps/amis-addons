"""U-Net (Ronneberger, Fischer & Brox, 2015).

The reference architecture: a contracting path of double convolutions with
max-pooling, a bottleneck, and an expanding path that concatenates the encoder
features at each resolution.  Kept deliberately plain - it is the baseline the
other methods in this catalogue are compared against.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_CONFIG: Dict[str, object] = {
    "in_channels": 1,
    "out_channels": 1,
    "base_channels": 32,
    "depth": 4,
    "dropout": 0.0,
}


class DoubleConv(nn.Module):
    """(3x3 conv -> BN -> ReLU) twice."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 32, depth: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.depth = depth
        self.multiple = 2 ** depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in channels:
            self.encoders.append(DoubleConv(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(channels[-1], channels[-1] * 2)

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        previous = channels[-1] * 2
        for index in reversed(range(depth)):
            self.ups.append(nn.ConvTranspose2d(previous, channels[index], 2, stride=2))
            self.decoders.append(DoubleConv(channels[index] * 2, channels[index]))
            previous = channels[index]

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.multiple
        pad_w = (-width) % self.multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        skips = []
        features = x
        for index, encoder in enumerate(self.encoders):
            features = encoder(features)
            if index < len(self.encoders) - 1:
                skips.append(features)
                features = self.pool(features)

        features = self.bottleneck(features)

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


def build_model(config: Optional[Dict[str, object]] = None) -> UNet:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    return UNet(
        in_channels=int(merged["in_channels"]),
        out_channels=int(merged["out_channels"]),
        base_channels=int(merged["base_channels"]),
        depth=int(merged["depth"]),
        dropout=float(merged.get("dropout", 0.0)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
