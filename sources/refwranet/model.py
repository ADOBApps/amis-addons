"""refWRANet: wavelet residual attention U-Net for retinal layer segmentation.

Reference implementation for the thesis proposal *"Multilayer retinal
segmentation"*: a U-Net whose downsampling is a discrete wavelet transform, so
the thin, low-contrast boundaries between retinal layers are not destroyed by
pooling.

The layer stack (OCTAVE's 14 classes, Kermany et al. 2025):

    background | SRM | HRM | FLU | HTD | RPE | RET | CHO | VIT | HYA | RHS | ART | ERM | SES

backbone: WRANet (Zhao et al., 2023, doi:10.1007/s40747-023-01119-y)
* downsampling: fixed analysis DWT, LL sub-band only (no pooling, no parameters)
* skip gating: residual attention, ``x_out = alpha * x_skip + x_skip``
* upsampling: bilinear x2, concat with the gated skip, double 3x3 conv
* output: one logit per class (``num_classes``), argmax at inference

The wavelet taps come from :mod:`wavelib`, which carries the exact filters for
Haar, Daubechies (2, 4, 6, 8), and Symlets (4, 8); biorthogonal families are
available in the DWT explorer tab of the addon.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelib import FILTERS, MODEL_WAVELETS

DEFAULT_CONFIG: Dict[str, object] = {
    "in_channels": 1,
    "out_channels": 14,          # one logit per class
    "num_classes": 14,
    "base_channels": 16,
    "depth": 4,
    "wavelet": "haar",
    "dropout": 0.0,
    "repeats": 2,
}


class WaveletDecompose(nn.Module):
    """Fixed separable 2-D analysis DWT with stride 2 (no trainable weights).

    ``forward`` returns the LL sub-band; :meth:`full` returns
    ``[LL, LH, HL, HH]`` when the high bands are wanted for an ablation.
    Padding is ``L - 2`` samples (split evenly, symmetric) so an even input
    yields exactly half the resolution.
    """

    def __init__(self, channels: int, wavelet: str = "haar") -> None:
        super().__init__()
        if wavelet not in FILTERS:
            raise ValueError(f"unknown wavelet '{wavelet}'; available: {sorted(FILTERS)}")
        if wavelet not in MODEL_WAVELETS:
            raise ValueError(
                f"{wavelet} is not orthogonal; the network accepts {MODEL_WAVELETS}. "
                "Use the 2-D DWT tab to inspect the biorthogonal families.")
        spec = FILTERS[wavelet]
        self.channels = channels
        self.length = len(spec["dec_lo"])
        total_pad = max(0, self.length - 2)
        self.pad_before = total_pad // 2
        self.pad_after = total_pad - self.pad_before

        self.register_buffer("lo_v", self._kernel(spec["dec_lo"], True), persistent=False)
        self.register_buffer("lo_h", self._kernel(spec["dec_lo"], False), persistent=False)
        self.register_buffer("hi_v", self._kernel(spec["dec_hi"], True), persistent=False)
        self.register_buffer("hi_h", self._kernel(spec["dec_hi"], False), persistent=False)

    def _kernel(self, taps: List[float], vertical: bool) -> torch.Tensor:
        weights = torch.tensor(taps, dtype=torch.float32)
        shape = (1, 1, -1, 1) if vertical else (1, 1, 1, -1)
        return weights.view(shape).repeat(self.channels, 1, 1, 1)

    def _transform(self, x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        vertical = (0, 0, self.pad_before, self.pad_after)
        horizontal = (self.pad_before, self.pad_after, 0, 0)
        y = F.conv2d(F.pad(x, vertical, mode="replicate"), rows,
                     stride=(2, 1), groups=self.channels)
        return F.conv2d(F.pad(y, horizontal, mode="replicate"), cols,
                        stride=(1, 2), groups=self.channels)

    def full(self, x: torch.Tensor) -> List[torch.Tensor]:
        ll = self._transform(x, self.lo_v, self.lo_h)
        lh = self._transform(x, self.lo_v, self.hi_h)
        hl = self._transform(x, self.hi_v, self.lo_h)
        hh = self._transform(x, self.hi_v, self.hi_h)
        return [ll, lh, hl, hh]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._transform(x, self.lo_v, self.lo_h)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, repeats: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for index in range(max(1, repeats)):
            layers += [
                nn.Conv2d(in_channels if index == 0 else out_channels,
                          out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveletDown(nn.Module):
    """DWT downsample -> double conv."""

    def __init__(self, in_channels: int, out_channels: int, wavelet: str = "haar",
                 keep_high: bool = False, repeats: int = 2) -> None:
        super().__init__()
        self.dwt = WaveletDecompose(in_channels, wavelet)
        self.keep_high = keep_high
        conv_in = in_channels * 4 if keep_high else in_channels
        self.conv = DoubleConv(conv_in, out_channels, repeats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reduced = torch.cat(self.dwt.full(x), dim=1) if self.keep_high else self.dwt(x)
        return self.conv(reduced)


class ResidualAttention(nn.Module):
    """Paper Eqs. 7-9: attention-gated skip plus residual add."""

    def __init__(self, low_channels: int, gate_channels: int) -> None:
        super().__init__()
        mid = max(1, low_channels)
        self.w_low = nn.Conv2d(low_channels, mid, 1)
        self.w_gate = nn.Conv2d(gate_channels, mid, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.psi = nn.Conv2d(mid, 1, 1)

    def forward(self, low: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != low.shape[-2:]:
            gate = F.interpolate(gate, size=low.shape[-2:], mode="bilinear",
                                 align_corners=False)
        energy = self.bn(self.w_low(low) + self.w_gate(gate))
        alpha = torch.sigmoid(self.psi(F.relu(energy, inplace=True)))
        return alpha * low + low


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 repeats: int = 2) -> None:
        super().__init__()
        self.attention = ResidualAttention(skip_channels, in_channels)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, repeats)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([up, self.attention(skip, up)], dim=1))


class RefWRANet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 14,
                 base_channels: int = 16, depth: int = 4, wavelet: str = "haar",
                 dropout: float = 0.0, repeats: int = 2) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        self.depth = depth
        self.multiple = 2 ** depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, channels[0], repeats)
        self.downs = nn.ModuleList(
            [WaveletDown(channels[i], channels[i + 1], wavelet, False, repeats)
             for i in range(depth)])
        self.ups = nn.ModuleList(
            [UpBlock(channels[i + 1], channels[i], channels[i], repeats)
             for i in reversed(range(depth))])

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels[0], out_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.multiple
        pad_w = (-width) % self.multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        skips: List[torch.Tensor] = [self.stem(x)]
        features = skips[0]
        for down in self.downs:
            features = down(features)
            skips.append(features)

        for index, up in enumerate(self.ups):
            features = up(features, skips[len(skips) - 2 - index])

        logits = self.head(self.dropout(features))
        if pad_h or pad_w:
            logits = logits[:, :, :height, :width]
        return logits


def build_model(config: Optional[Dict[str, object]] = None) -> RefWRANet:
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    return RefWRANet(
        in_channels=int(merged["in_channels"]),
        out_channels=int(merged.get("out_channels", merged.get("num_classes", 14))),
        base_channels=int(merged["base_channels"]),
        depth=int(merged["depth"]),
        wavelet=str(merged.get("wavelet", "haar")),
        dropout=float(merged.get("dropout", 0.0)),
        repeats=int(merged.get("repeats", 2)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
