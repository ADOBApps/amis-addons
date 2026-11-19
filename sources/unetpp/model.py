"""UNet++ (Zhou et al., 2018).

Nested U-Net: between every encoder level and the decoder there is a dense grid
of convolution nodes, X[i][j], each receiving all previous nodes of its row and
an upsampled node from the row below.  The dense paths reduce the semantic gap
between the encoder features and the decoder features without any extra
supervision.
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


class ConvBlock(nn.Module):
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


class UNetPP(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 base_channels: int = 16, depth: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")
        self.depth = depth
        self.multiple = 2 ** depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in channels:
            self.encoders.append(ConvBlock(previous, width))
            previous = width
        self.pool = nn.MaxPool2d(2)

        # node (i, j): concatenates j nodes of row i and one upsampled node
        self.nodes = nn.ModuleDict()
        for j in range(1, depth + 1):
            for i in range(0, depth + 1 - j):
                in_channels_node = j * channels[i] + channels[i + 1]
                self.nodes[f"{i}_{j}"] = ConvBlock(in_channels_node, channels[i])

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(channels[0], out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.multiple
        pad_w = (-width) % self.multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        nodes: Dict[tuple, torch.Tensor] = {}
        features = x
        for i, encoder in enumerate(self.encoders):
            features = encoder(features)
            nodes[(i, 0)] = features
            if i < self.depth:
                features = self.pool(features)

        for j in range(1, self.depth + 1):
            for i in range(0, self.depth + 1 - j):
                target = nodes[(i, 0)].shape[-2:]
                parts = [nodes[(i, k)] for k in range(j)]
                upsampled = F.interpolate(
                    nodes[(i + 1, j - 1)], size=target,
                    mode="bilinear", align_corners=False)
                nodes[(i, j)] = self.nodes[f"{i}_{j}"](
                    torch.cat(parts + [upsampled], dim=1))

        logits = self.head(self.dropout(nodes[(0, self.depth)]))
        if pad_h or pad_w:
            logits = logits[:, :, :height, :width]
        return logits


def build_model(config: Optional[Dict[str, object]] = None) -> UNetPP:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    return UNetPP(
        in_channels=int(merged["in_channels"]),
        out_channels=int(merged["out_channels"]),
        base_channels=int(merged["base_channels"]),
        depth=int(merged["depth"]),
        dropout=float(merged.get("dropout", 0.0)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
