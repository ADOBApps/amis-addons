"""WRANet Layers - wavelet residual attention U-Net for retinal layer segmentation.

Same backbone as WRANet (Zhao et al., 2023), but with one output channel per
class instead of a single sigmoid logit, so the network labels the retinal
layers and the pathological spaces in one pass:

    background | SRM | HRM | FLU | HTD | RPE | RET | CHO | VIT | HYA | RHS | ART | ERM | SES

The wavelet downsampling is what makes this viable: the DWT keeps the
low-frequency sub-band and discards noisy high-frequency detail, so the thin
boundaries between layers survive downsampling, and the residual attention gates
re-weight the skips at the resolution where those boundaries live.

Differences from the binary `wranet` addon:

* ``out_channels`` defaults to the number of classes (14 for OCTAVE);
* training uses cross-entropy + macro soft Dice and per-class Dice validation;
* prediction takes ``argmax`` over the class axis, not a sigmoid threshold.

The architecture is a copy of the reference implementation on purpose: an addon
must be self-contained, and a method's backbone is part of the method.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------- filters
#: Quadrature-mirror low-pass analysis filters (sum = sqrt(2)).
WAVELET_TAPS: Dict[str, List[float]] = {
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db2": [
        0.48296291314453416,
        0.8365163037378079,
        0.22414386804201339,
        -0.12940952255126037,
    ],
    "db4": [
        0.2303778133088964,
        0.7148465705529154,
        0.6308807679298587,
        -0.027983769416859854,
        -0.18703481171888114,
        0.030841381835560764,
        0.0328830116668852,
        -0.010597401785069032,
    ],
}


def highpass_from_lowpass(taps: List[float]) -> List[float]:
    """QMF relation ``h[i] = (-1)^i l[L-1-i]`` (paper Eq. 1)."""
    length = len(taps)
    return [((-1) ** i) * taps[length - 1 - i] for i in range(length)]


class WaveletDecompose(nn.Module):
    """Fixed separable 2-D DWT with stride 2 (analysis only).

    ``forward`` returns the LL sub-band (what WRANet propagates);
    :meth:`full` returns ``[LL, LH, HL, HH]`` for ablations.

    Padding (``L - 2`` samples, split evenly, replicated) is chosen so that an
    even-sized input yields exactly half the resolution - no off-by-one surprises
    when the decoder upsamples back to the skip size.
    """

    def __init__(self, channels: int, wavelet: str = "haar") -> None:
        super().__init__()
        if wavelet not in WAVELET_TAPS:
            raise ValueError(
                f"unknown wavelet '{wavelet}'; available: {sorted(WAVELET_TAPS)}"
            )
        low = WAVELET_TAPS[wavelet]
        high = highpass_from_lowpass(low)
        self.channels = channels
        self.length = len(low)
        total_pad = max(0, self.length - 2)
        self.pad_before = total_pad // 2
        self.pad_after = total_pad - self.pad_before

        self.register_buffer("lo_v", self._kernel(low, vertical=True), persistent=False)
        self.register_buffer("lo_h", self._kernel(low, vertical=False), persistent=False)
        self.register_buffer("hi_v", self._kernel(high, vertical=True), persistent=False)
        self.register_buffer("hi_h", self._kernel(high, vertical=False), persistent=False)

    def _kernel(self, taps: List[float], vertical: bool) -> torch.Tensor:
        weights = torch.tensor(taps, dtype=torch.float32)
        shape = (1, 1, -1, 1) if vertical else (1, 1, 1, -1)
        return weights.view(shape).repeat(self.channels, 1, 1, 1)

    def _transform(self, x: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
        vertical = (0, 0, self.pad_before, self.pad_after)
        horizontal = (self.pad_before, self.pad_after, 0, 0)
        y = F.conv2d(
            F.pad(x, vertical, mode="replicate"),
            rows,
            stride=(2, 1),
            groups=self.channels,
        )
        return F.conv2d(
            F.pad(y, horizontal, mode="replicate"),
            cols,
            stride=(1, 2),
            groups=self.channels,
        )

    def full(self, x: torch.Tensor) -> List[torch.Tensor]:
        """``[LL, LH, HL, HH]`` (paper Eq. 6 ordering)."""
        ll = self._transform(x, self.lo_v, self.lo_h)
        lh = self._transform(x, self.lo_v, self.hi_h)
        hl = self._transform(x, self.hi_v, self.lo_h)
        hh = self._transform(x, self.hi_v, self.hi_h)
        return [ll, lh, hl, hh]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._transform(x, self.lo_v, self.lo_h)


class DoubleConv(nn.Module):
    """``(conv 3x3 -> BN -> ReLU)`` repeated ``repeats`` times.

    The paper's parameter budget (34.6 M for the full model) only closes with
    two convolutions per stage, i.e. the classic U-Net block; ``repeats`` is a
    config knob so the count can be matched against the published table.
    """

    def __init__(self, in_channels: int, out_channels: int, repeats: int = 2) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for index in range(max(1, repeats)):
            layers += [
                nn.Conv2d(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveletDown(nn.Module):
    """DWT downsample -> double conv (BN/ReLU inside)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        wavelet: str = "haar",
        keep_high: bool = False,
        repeats: int = 2,
    ) -> None:
        super().__init__()
        self.dwt = WaveletDecompose(in_channels, wavelet)
        self.keep_high = keep_high
        conv_in = in_channels * 4 if keep_high else in_channels
        self.conv = DoubleConv(conv_in, out_channels, repeats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.keep_high:
            reduced = torch.cat(self.dwt.full(x), dim=1)
        else:
            reduced = self.dwt(x)
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
            gate = F.interpolate(
                gate, size=low.shape[-2:], mode="bilinear", align_corners=False
            )
        energy = self.bn(self.w_low(low) + self.w_gate(gate))
        alpha = torch.sigmoid(self.psi(F.relu(energy, inplace=True)))
        return alpha * low + low


class UpBlock(nn.Module):
    """Paper Eq. 10: bilinear x2 -> concat with gated skip -> double conv."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        repeats: int = 2,
    ) -> None:
        super().__init__()
        self.attention = ResidualAttention(skip_channels, in_channels)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels, repeats)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        merged = torch.cat([up, self.attention(skip, up)], dim=1)
        return self.conv(merged)


class WRANetLayers(nn.Module):
    """Wavelet residual attention U-Net (2-D, single input channel by default)."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        wavelet: str = "haar",
        keep_high: bool = False,
        dropout: float = 0.0,
        repeats: int = 2,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.depth = depth
        self.multiple = 2 ** depth
        channels = [base_channels * (2 ** i) for i in range(depth + 1)]

        self.stem = DoubleConv(in_channels, channels[0], repeats)

        self.downs = nn.ModuleList(
            [
                WaveletDown(channels[i], channels[i + 1], wavelet, keep_high, repeats)
                for i in range(depth)
            ]
        )
        self.ups = nn.ModuleList(
            [
                UpBlock(channels[i + 1], channels[i], channels[i], repeats)
                for i in reversed(range(depth))
            ]
        )

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


# --------------------------------------------------------------------- config
DEFAULT_CONFIG: Dict[str, object] = {
    "in_channels": 1,
    "out_channels": 14,          # one logit per class (see classes.json)
    "num_classes": 14,
    "base_channels": 64,
    "depth": 4,
    "wavelet": "haar",
    "keep_high": False,
    "dropout": 0.0,
    "repeats": 2,
}


def build_model(config: Optional[Dict[str, object]] = None) -> WRANetLayers:
    """Factory used by the core's trainer, exporter and segmenter."""
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    return WRANetLayers(
        in_channels=int(merged["in_channels"]),
        out_channels=int(merged.get("out_channels", merged.get("num_classes", 14))),
        base_channels=int(merged["base_channels"]),
        depth=int(merged["depth"]),
        wavelet=str(merged["wavelet"]),
        keep_high=bool(merged["keep_high"]),
        dropout=float(merged["dropout"]),
        repeats=int(merged["repeats"]),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_summary(config: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Cheap description for the GUI (no forward pass, no allocation)."""
    merged = dict(DEFAULT_CONFIG)
    merged.update(config or {})
    base = int(merged["base_channels"])
    depth = int(merged["depth"])
    model = build_model(merged)
    return {
        "name": "WRANetLayers",
        "channels": [base * (2 ** i) for i in range(depth + 1)],
        "downsample_factor": 2 ** depth,
        "parameters_m": round(count_parameters(model) / 1e6, 2),
        "wavelet": merged["wavelet"],
        "keep_high": bool(merged["keep_high"]),
    }


__all__ = [
    "WRANetLayers",
    "WaveletDown",
    "WaveletDecompose",
    "ResidualAttention",
    "UpBlock",
    "WAVELET_TAPS",
    "DEFAULT_CONFIG",
    "build_model",
    "model_summary",
    "count_parameters",
]
