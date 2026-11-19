"""Flat-field threshold: relative contrast against a local background.

A global threshold fails under uneven illumination; a purely local threshold
(Sauvola, Niblack) instead marks smooth regions as foreground, because its
threshold sits below the local mean.  This method takes the middle road, which
is what most microscopy and angiography pipelines do:

1. estimate the illumination field B as a large-window box mean of the image;
2. divide it out: I' = I / B, so a flat background becomes 1.0 everywhere while
   an object that is brighter than its surroundings stays above 1;
3. keep the pixels at least ``bias`` times brighter than their background
   (1.05 by default, i.e. 5 % above the local level);
4. two majority-filter passes fill the object interior and remove speckle.

The result is a filled mask that survives a gradient background, which neither a
global threshold (too high for the dark side) nor a strictly local one such as
Sauvola (its threshold falls below the local mean, so smooth regions become
foreground) manages on its own.  Otsu on the corrected image is *not* used: the
background dominates the histogram and the threshold lands inside the gradient
tail.
"""

from __future__ import annotations

import numpy as np

DEFAULT_RADIUS_FRACTION = 0.12
DEFAULT_BIAS = 1.05
EPS = 1e-6


def describe() -> str:
    return "Flat-field threshold"


def _integral(array: np.ndarray) -> np.ndarray:
    padded = np.pad(array, ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(axis=0).cumsum(axis=1)


def _box_mean(image: np.ndarray, radius: int) -> np.ndarray:
    """Mean in a (2r+1)^2 window, computed from an integral image."""
    radius = max(1, int(radius))
    padded = np.pad(image, radius, mode="reflect")
    integral = _integral(padded)
    height, width = image.shape
    size = 2 * radius + 1
    rows = np.arange(height) + size
    cols = np.arange(width) + size
    top = rows - size
    left = cols - size
    total = (
        integral[rows][:, cols]
        - integral[top][:, cols]
        - integral[rows][:, left]
        + integral[top][:, left]
    )
    return total / float(size * size)


def segment(image: np.ndarray, radius_fraction: float = DEFAULT_RADIUS_FRACTION,
            bias: float = DEFAULT_BIAS, invert: bool = False) -> np.ndarray:
    """Return a binary mask as float32 in {0, 1}."""
    img = np.asarray(image, dtype=np.float64)
    peak = float(np.max(img)) if img.size else 0.0
    if peak > 1.5:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    height, width = img.shape
    radius = max(8, int(min(height, width) * float(radius_fraction)))
    background = _box_mean(img, radius) + EPS
    corrected = img / background

    mask = corrected > float(bias)

    smooth_radius = max(2, radius // 8)
    for _ in range(2):
        mask = _box_mean(mask.astype(np.float64), smooth_radius) > 0.5

    if invert:
        mask = ~mask
    return mask.astype(np.float32)
