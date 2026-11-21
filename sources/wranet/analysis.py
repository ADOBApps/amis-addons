"""Training-free analysis of an image through the addon's architecture outline.

No weights, no optimisation, no classes: every operation here is a fixed,
data-driven rule over the wavelet coefficients, which is what makes the tab work
without training.

    outline element      training-free reading
    -----------------    -----------------------------------------------------
    encoder stages       forward DWT cascade, LL kept at each level
    skip connections     LH / HL / HH of every level
    RAM gate             g = |c| / (|c| + T),  T = k * sigma * sqrt(2 ln n),
                         sigma = median(|c|) / 0.6745 (MAD noise estimate)
    decoder              inverse DWT with the synthesis taps (exact to ~1e-10)
    head                 maps: reconstruction, detail energy, gate map

The result is a deterministic analysis of the transform (denoised
reconstruction, edge/texture energy, where the gate acts, per-band statistics).
It is **not** a segmentation: class labels need a trained model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from . import wavelib

BANDS = ("LL", "LH", "HL", "HH")
DETAILS = ("LH", "HL", "HH")
#: median(|N(0, 1)|); converts a median absolute deviation into a sigma
MAD_SCALE = 0.6744897501960817
EPS = 1e-12


def _threshold(values: np.ndarray, threshold: float, mode: str) -> np.ndarray:
    if mode == "soft":
        return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)
    if mode == "hard":
        return np.where(np.abs(values) > threshold, values, 0.0)
    return values


def _to_image_size(array: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour x2 upsample (coefficient grid is half the image)."""
    big = np.repeat(np.repeat(array, 2, axis=0), 2, axis=1)
    return big[:shape[0], :shape[1]]


@dataclass
class BandStat:
    level: int
    band: str
    sigma: float
    threshold: float
    kept: float
    energy_share: float
    max_abs: float


@dataclass
class AnalysisResult:
    wavelet: str
    levels: int
    mode: str
    k: float
    recon: np.ndarray
    detail: np.ndarray
    gate: np.ndarray
    stats: List[BandStat] = field(default_factory=list)
    rmse: float = 0.0
    psnr: float = 0.0
    report: str = ""
    requested_levels: int = 0

    def csv_rows(self) -> List[Tuple[object, ...]]:
        rows: List[Tuple[object, ...]] = []
        for stat in self.stats:
            rows.append((stat.level, stat.band, f"{stat.sigma:.6f}",
                         f"{stat.threshold:.6f}", f"{100.0 * stat.kept:.2f}",
                         f"{100.0 * stat.energy_share:.2f}",
                         f"{stat.max_abs:.6f}"))
        return rows

    @staticmethod
    def csv_header() -> List[str]:
        return ["level", "band", "sigma", "threshold", "kept_pct",
                "energy_pct", "max_abs"]


def _safe_levels(shape: Tuple[int, int], taps: int, requested: int) -> int:
    """Stop the cascade while the coarsest input is still >= the filter length.

    Below that the circular filter bank is no longer invertible, so the extra
    levels would only add aliasing (and no analysis value).
    """
    levels = 1
    height, width = shape
    while levels < requested:
        height, width = (height + 1) // 2, (width + 1) // 2
        if min(height, width) < taps:
            break
        levels += 1
    return levels


def run_analysis(image: np.ndarray, wavelet: str = "haar", levels: int = 2,
                 k: float = 3.0, mode: str = "soft") -> AnalysisResult:
    """Analyse one image with fixed wavelet rules (see the module docstring)."""
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 2:
        raise ValueError("run_analysis expects a 2-D grayscale image")
    if wavelet not in wavelib.FILTERS:
        raise ValueError(f"unknown wavelet '{wavelet}'")
    taps = len(wavelib.FILTERS[wavelet]["dec_lo"])
    if min(image.shape) < taps:
        raise ValueError(
            f"image {image.shape[1]}x{image.shape[0]} is smaller than the "
            f"{wavelet} filter ({taps} taps); choose 'haar' or a larger image")
    requested = max(1, int(levels))
    levels = _safe_levels(image.shape, taps, requested)
    k = max(0.0, float(k))
    mode = mode if mode in ("soft", "hard", "none") else "soft"

    chain = wavelib.cascade_periodic(image, wavelet, levels)

    stats: List[BandStat] = []
    gated_chain = []
    for level, bands, shape in chain:
        total = sum(float(np.sum(np.asarray(bands[band]) ** 2))
                    for band in BANDS) or 1.0
        out_bands = dict(bands)
        for band in BANDS:
            values = np.abs(bands[band])
            sigma = float(np.median(values) / MAD_SCALE) if values.size else 0.0
            threshold = 0.0
            if values.size:
                threshold = k * sigma * math.sqrt(2.0 * math.log(max(values.size, 2)))
            kept = float(np.mean(values > threshold)) if threshold > 0 else 1.0
            stats.append(BandStat(
                level=level,
                band=band,
                sigma=sigma,
                threshold=threshold,
                kept=kept,
                energy_share=float(np.sum(values ** 2) / total),
                max_abs=float(values.max()) if values.size else 0.0,
            ))
            if band != "LL" and mode != "none" and threshold > 0:
                out_bands[band] = _threshold(bands[band], threshold, mode)
        gated_chain.append((level, out_bands, shape))

    recon = wavelib.reconstruct_periodic(gated_chain, wavelet)
    recon = np.clip(recon[:image.shape[0], :image.shape[1]], 0.0, 1.0)

    # maps from the finest level (the unthresholded coefficients)
    finest = chain[0][1]
    thresholds = {stat.band: stat.threshold for stat in stats if stat.level == 1}
    detail = np.sqrt(sum(finest[band] ** 2 for band in DETAILS))
    gate = np.maximum.reduce([
        np.abs(finest[band]) / (np.abs(finest[band]) + thresholds[band] + EPS)
        for band in DETAILS
    ]) if thresholds else np.ones_like(finest["LL"])

    peak = float(np.percentile(detail, 99.5)) if detail.size else 0.0
    detail_display = np.clip(detail / peak, 0.0, 1.0) if peak > 0 else np.zeros_like(detail)

    difference = recon - image
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    psnr = float(20.0 * math.log10(1.0 / rmse)) if rmse > 0 else float("inf")

    result = AnalysisResult(
        wavelet=wavelet,
        levels=levels,
        mode=mode,
        k=k,
        recon=recon,
        detail=_to_image_size(detail_display, image.shape),
        gate=_to_image_size(gate, image.shape),
        stats=stats,
        rmse=rmse,
        psnr=psnr,
        requested_levels=requested,
    )
    result.report = _report(result, image.shape)
    return result


def _report(result: AnalysisResult, shape: Tuple[int, int]) -> str:
    lines = [
        "Training-free wavelet analysis - no trained weights, no classes",
        f"image {shape[1]}x{shape[0]} | wavelet {result.wavelet} | "
        f"{result.levels} level(s) | gate {result.mode} | k {result.k:.2f}",
    ]
    if result.requested_levels and result.requested_levels != result.levels:
        lines.append(
            f"levels capped at {result.levels} (asked for "
            f"{result.requested_levels}): the coarsest input must stay at "
            f"least as large as the filter")
    lines += [
        "",
        "outline -> training-free reading:",
        "  encoder stages    forward DWT cascade (LL kept at each level)",
        "  skip connections  LH / HL / HH bands of every level",
        "  RAM gate          |c| / (|c| + k*sigma), sigma = median(|c|) / 0.6745",
        "  decoder           inverse DWT with the synthesis taps (~1e-10 exact)",
        "  head              maps: reconstruction, detail energy, gate",
        "",
        "per level / band:",
    ]
    for stat in result.stats:
        lines.append(
            f"  L{stat.level} {stat.band}  sigma {stat.sigma:.6f}  "
            f"T {stat.threshold:.6f}  kept {100.0 * stat.kept:5.1f}%  "
            f"energy {100.0 * stat.energy_share:5.1f}%")
    lines += [
        "",
        f"reconstruction: RMSE {result.rmse:.6f}  PSNR {result.psnr:.2f} dB",
    ]
    if result.mode == "none":
        lines.append("threshold mode 'none': every coefficient is kept "
                     "(the reconstruction is the inverse transform itself)")
    lines += [
        "",
        "Deterministic: the same image and settings give the same numbers. The",
        "image is loaded as float32 in [0, 1] (the AMIS convention), so the RMSE",
        "is in that scale. This analyses the transform, not the anatomy - class",
        "labels need a trained model (Layers tab).",
    ]
    return "\n".join(lines)
