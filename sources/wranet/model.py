"""WRANet 2.0: encoder-decoder with attention skip connections (RAM).

Same data contract, panel and training recipe as :mod:`refwranet`, so the two
addons form a clean ablation pair; the difference is the topology: MaxPool2d
downsampling (with the LL sub-band of a 2-D DWT selectable as an ablation)
instead of the wavelet U-Net, channel + spatial residual attention, transposed
convolution decoding and a single 1x1 head.

The 1.1.x WRANet checkpoint and ONNX export are **not** compatible with this
graph (different blocks, channels and head); 1.1.4 remains published for pinning.

* **encoder** - ``Conv3x3 + BN + ReLU`` per stage, ``MaxPool2d(2, 2)`` between
  stages (or, as an ablation, the LL sub-band of a fixed 2-D DWT, which has the
  same H/2 x W/2 output), channels doubling from ``base_channels``
  (16 -> 32 -> 64 -> 128) and a bottleneck at ``base_channels * 2**depth``
  (256);
* **bottleneck** - the deepest ``Conv3x3 + BN + ReLU`` at H/16;
* **skip gating** - every skip passes through a *residual attention model*
  (RAM) before concatenation: a channel-attention vector (squeeze-and-excite on
  average + max statistics) times a gate-driven spatial attention map, sent
  through a residual block and added back to the untouched features;
* **decoder** - ``ConvTranspose2d(2, 2) + Conv3x3 + BN + ReLU`` (or bilinear
  upsampling), concatenated with ``RAM(encoder_feature)``;
* **head** - a single ``1x1`` convolution to ``num_classes`` logits.

With ``base_channels=16`` and ``depth=4`` the tensor ladder is exactly the one
in the spec::

    input      [1, 256, 256]
    enc 1      16 @ 256x256
    enc 2      32 @ 128x128
    enc 3      64 @  64x64
    enc 4     128 @  32x32
    bottleneck 256 @  16x16
    dec 1     256 @  32x32   (128 upsampled + 128 gated skip)
    dec 2     128 @  64x64   ( 64 upsampled +  64 gated skip)
    dec 3      64 @ 128x128  ( 32 upsampled +  32 gated skip)
    dec 4      32 @ 256x256  ( 16 upsampled +  16 gated skip)
    head       14 @ 256x256

``channel_plan()`` recomputes that table for any width, depth or input size and
is what the panel prints, so the GUI cannot drift from the code.

Downsampling is selectable (``downsample="maxpool"`` by default, as in the spec;
``"dwt"`` swaps the pooling for the LL sub-band of a wavelet transform and is
what connects this variant to the DWT explorer). Both produce exactly half the
resolution, so the ladder and the decoder are identical either way - which is
what makes the comparison an ablation rather than a different network.

Odd input sizes are handled: the tensor is replicate-padded up to a multiple of
``2**depth`` and the logits are cropped back to the original size.
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
    "channel_multiplier": 2,
    "dropout": 0.0,
    "repeats": 2,                # conv+BN+ReLU units inside each RAM residual block
    "img_size": 256,
    "attn_reduction": 8,         # channel-attention squeeze ratio
    "upsample": "convtranspose",  # "convtranspose" or "bilinear"
    "downsample": "maxpool",     # "maxpool" (spec) or "dwt" (LL sub-band, ablation)
    "wavelet": "haar",           # only used when downsample == "dwt"
}


class ConvBNReLU(nn.Module):
    """The block named in the spec: Conv3x3 + BatchNorm + ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveletReduce(nn.Module):
    """Fixed separable 2-D analysis DWT with stride 2, LL sub-band only.

    No trainable weights: the taps come from :mod:`wavelib`. Padding is
    ``L - 2`` samples, split evenly and replicated, so an even input yields
    exactly half the resolution - the same contract as ``MaxPool2d(2, 2)``.
    """

    def __init__(self, channels: int, wavelet: str = "haar") -> None:
        super().__init__()
        if wavelet not in FILTERS:
            raise ValueError(f"unknown wavelet '{wavelet}'; available: {sorted(FILTERS)}")
        if wavelet not in MODEL_WAVELETS:
            raise ValueError(
                f"{wavelet} is not orthogonal; the network accepts {list(MODEL_WAVELETS)}. "
                "Use the 2-D DWT tab to inspect the biorthogonal families.")
        spec = FILTERS[wavelet]
        self.channels = channels
        total_pad = max(0, len(spec["dec_lo"]) - 2)
        self.pad_before = total_pad // 2
        self.pad_after = total_pad - self.pad_before
        self.register_buffer("lo_v", self._kernel(spec["dec_lo"], True), persistent=False)
        self.register_buffer("lo_h", self._kernel(spec["dec_lo"], False), persistent=False)

    def _kernel(self, taps: List[float], vertical: bool) -> torch.Tensor:
        weights = torch.tensor(taps, dtype=torch.float32)
        shape = (1, 1, -1, 1) if vertical else (1, 1, 1, -1)
        return weights.view(shape).repeat(self.channels, 1, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        vertical = (0, 0, self.pad_before, self.pad_after)
        horizontal = (self.pad_before, self.pad_after, 0, 0)
        rows = F.conv2d(F.pad(x, vertical, mode="replicate"), self.lo_v,
                        stride=(2, 1), groups=self.channels)
        return F.conv2d(F.pad(rows, horizontal, mode="replicate"), self.lo_h,
                        stride=(1, 2), groups=self.channels)


class Downsample(nn.Module):
    """One encoder step: ``MaxPool2d(2, 2)`` or the DWT LL sub-band.

    ``"maxpool"`` is the architecture in the addon spec; ``"dwt"`` keeps the
    same output shape and channel count, so the rest of the network (and the
    printed ladder) is unchanged - that is the ablation.
    """

    def __init__(self, channels: int, kind: str = "maxpool", wavelet: str = "haar") -> None:
        super().__init__()
        if kind not in ("maxpool", "dwt"):
            raise ValueError(f"unknown downsample '{kind}'; use 'maxpool' or 'dwt'")
        self.kind = kind
        self.wavelet = wavelet
        if kind == "dwt":
            self.op: nn.Module = WaveletReduce(channels, wavelet)
        else:
            self.op = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class ChannelAttention(nn.Module):
    """Squeeze-and-excite gate on the skip features (average + max statistics)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(1, channels // max(1, reduction))
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        average = x.mean(dim=(2, 3), keepdim=True)
        maximum = x.amax(dim=(2, 3), keepdim=True)
        return torch.sigmoid(self.mlp(average) + self.mlp(maximum))


class ResidualBlock(nn.Module):
    """``repeats`` x (Conv3x3 + BN + ReLU [+ Dropout]) plus the identity path."""

    def __init__(self, channels: int, repeats: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        for _ in range(max(1, repeats)):
            layers += [
                nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResidualAttention(nn.Module):
    """RAM: spatial + channel attention on a skip, then a residual block.

    ``low`` is the encoder feature (the skip) and ``gate`` the upsampled decoder
    feature. The attention map is the additive attention gate of the WRANet
    paper (Eqs. 7-9) multiplied by the channel descriptor; the residual path
    preserves whatever the gate suppresses.
    """

    def __init__(self, low_channels: int, gate_channels: int, repeats: int = 2,
                 dropout: float = 0.0, reduction: int = 8) -> None:
        super().__init__()
        mid = max(1, low_channels // 2)
        self.channel = ChannelAttention(low_channels, reduction)
        self.w_low = nn.Conv2d(low_channels, mid, 1, bias=False)
        self.w_gate = nn.Conv2d(gate_channels, mid, 1, bias=False)
        self.bn = nn.BatchNorm2d(mid)
        self.psi = nn.Conv2d(mid, 1, 1)
        self.residual = ResidualBlock(low_channels, repeats, dropout)

    def forward(self, low: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if gate.shape[-2:] != low.shape[-2:]:
            gate = F.interpolate(gate, size=low.shape[-2:], mode="bilinear",
                                 align_corners=False)
        channel = self.channel(low)
        energy = self.bn(self.w_low(low) + self.w_gate(gate))
        spatial = torch.sigmoid(self.psi(F.relu(energy, inplace=True)))
        attended = low * channel * spatial
        return self.residual(attended) + low


class UpBlock(nn.Module):
    """Upsample + Conv + BN + ReLU, then concat with the RAM-gated skip."""

    def __init__(self, in_channels: int, up_channels: int, skip_channels: int,
                 repeats: int = 2, dropout: float = 0.0, upsample: str = "convtranspose",
                 reduction: int = 8) -> None:
        super().__init__()
        if upsample == "bilinear":
            self.up: Optional[nn.Module] = None
            # interpolation keeps the channel count; reduce it with a 1x1 conv
            self.project: Optional[nn.Module] = nn.Conv2d(
                in_channels, up_channels, 1, bias=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, up_channels, kernel_size=2, stride=2)
            self.project = None
        self.conv = ConvBNReLU(up_channels, up_channels)
        self.ram = ResidualAttention(skip_channels, up_channels, repeats, dropout,
                                     reduction)
        self.out_channels = up_channels + skip_channels

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.up is not None:
            up = self.up(x)
        else:
            up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            up = self.project(up)
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode="bilinear",
                               align_corners=False)
        up = self.conv(up)
        return torch.cat([up, self.ram(skip, up)], dim=1)


class WRANet(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 14,
                 base_channels: int = 16, depth: int = 4, channel_multiplier: int = 2,
                 dropout: float = 0.0, repeats: int = 2, upsample: str = "convtranspose",
                 attn_reduction: int = 8, downsample: str = "maxpool",
                 wavelet: str = "haar") -> None:
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")
        if channel_multiplier < 1:
            raise ValueError("channel_multiplier must be >= 1")
        if downsample not in ("maxpool", "dwt"):
            raise ValueError("downsample must be 'maxpool' or 'dwt'")
        self.depth = depth
        self.downsample = downsample
        self.wavelet = wavelet
        self.multiple = 2 ** depth
        channels = [base_channels * (channel_multiplier ** index)
                    for index in range(depth + 1)]

        #: encoder stage 1 (full resolution) -> ... -> bottleneck
        self.stem = ConvBNReLU(in_channels, channels[0])
        self.encoder = nn.ModuleList(
            [ConvBNReLU(channels[index], channels[index + 1])
             for index in range(depth)])
        #: one downsampler per encoder step, matching the features it consumes
        self.downs = nn.ModuleList(
            [Downsample(channels[index], downsample, wavelet)
             for index in range(depth)])

        self.ups = nn.ModuleList()
        in_channels_up = channels[depth]
        for level in reversed(range(depth)):
            block = UpBlock(in_channels_up, channels[level], channels[level],
                            repeats, dropout, upsample, attn_reduction)
            self.ups.append(block)
            in_channels_up = block.out_channels

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.head = nn.Conv2d(in_channels_up, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[-2:]
        pad_h = (-height) % self.multiple
        pad_w = (-width) % self.multiple
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        features = self.stem(x)
        skips: List[torch.Tensor] = [features]
        for index, block in enumerate(self.encoder):
            features = block(self.downs[index](features))
            if index < self.depth - 1:      # the last one is the bottleneck
                skips.append(features)

        for block, skip in zip(self.ups, reversed(skips)):
            features = block(features, skip)

        logits = self.head(self.dropout(features))
        if pad_h or pad_w:
            logits = logits[:, :, :height, :width]
        return logits


def _resolve(config: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Merge a caller's config over the defaults, ignoring unknown keys.

    ``num_classes`` is the friendlier alias: it sets ``out_channels`` only when
    the caller did not pass ``out_channels`` explicitly (the default already
    contains one, so a plain ``dict.update`` would let it win silently).
    """
    given = dict(config or {})
    merged = dict(DEFAULT_CONFIG)
    merged.update({key: value for key, value in given.items() if key in DEFAULT_CONFIG})
    if "num_classes" in given and "out_channels" not in given:
        merged["out_channels"] = given["num_classes"]
    return merged


def channel_plan(config: Optional[Dict[str, object]] = None,
                 size: Optional[int] = None) -> Dict[str, object]:
    """The tensor ladder the model builds, for labels, logs and tests."""
    merged = _resolve(config)
    depth = int(merged["depth"])
    base = int(merged["base_channels"])
    multiplier = int(merged["channel_multiplier"])
    in_channels = int(merged["in_channels"])
    downsample = str(merged.get("downsample", "maxpool"))
    wavelet = str(merged.get("wavelet", "haar"))
    if downsample not in ("maxpool", "dwt"):
        raise ValueError(f"unknown downsample '{downsample}'; use 'maxpool' or 'dwt'")
    if downsample == "dwt" and wavelet not in MODEL_WAVELETS:
        raise ValueError(
            f"{wavelet} is not orthogonal; the network accepts {list(MODEL_WAVELETS)}")
    size = int(size if size is not None else merged.get("img_size", 256))
    multiple = 2 ** depth
    padded = size + (-size) % multiple      # replicate padding used by forward()
    channels = [base * (multiplier ** index) for index in range(depth + 1)]

    stages = []
    op = "DWT-" + wavelet if downsample == "dwt" else "MaxPool2d"
    for index in range(depth):
        side = max(1, padded // (2 ** index))
        stages.append({
            "stage": index + 1,
            "channels": channels[index],
            "size": [side, side],
            "op": f"Conv3x3+BN+ReLU + {op}",
        })
    bottleneck_side = max(1, padded // multiple)
    bottleneck = {"channels": channels[depth], "size": [bottleneck_side, bottleneck_side]}

    decoder = []
    in_channels_up = channels[depth]
    for index, level in enumerate(reversed(range(depth))):
        side = max(1, padded // (2 ** level))
        up = channels[level]
        skip = channels[level]
        decoder.append({
            "stage": index + 1,
            "in_channels": in_channels_up,
            "upsampled_channels": up,
            "skip_channels": skip,
            "out_channels": up + skip,
            "size": [side, side],
        })
        in_channels_up = up + skip

    return {
        "input": [in_channels, size, size],
        "padded_input": [in_channels, padded, padded],
        "downsample": downsample,
        "wavelet": wavelet if downsample == "dwt" else "",
        "encoder": stages,
        "bottleneck": bottleneck,
        "decoder": decoder,
        "head": {"in_channels": in_channels_up,
                 "out_channels": int(merged.get("out_channels",
                                                 merged.get("num_classes", 14))),
                 "size": [size, size]},
    }


def build_model(config: Optional[Dict[str, object]] = None) -> WRANet:
    merged = _resolve(config)
    return WRANet(
        in_channels=int(merged["in_channels"]),
        out_channels=int(merged.get("out_channels", merged.get("num_classes", 14))),
        base_channels=int(merged["base_channels"]),
        depth=int(merged["depth"]),
        channel_multiplier=int(merged["channel_multiplier"]),
        dropout=float(merged.get("dropout", 0.0)),
        repeats=int(merged.get("repeats", 2)),
        upsample=str(merged.get("upsample", "convtranspose")),
        attn_reduction=int(merged.get("attn_reduction", 8)),
        downsample=str(merged.get("downsample", "maxpool")),
        wavelet=str(merged.get("wavelet", "haar")),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
