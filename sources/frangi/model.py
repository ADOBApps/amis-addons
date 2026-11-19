"""Frangi multiscale Hessian vesselness (Frangi et al., 1998).

Tubular structures - vessels, and by extension the neck of an aneurysm - are
characterised by one large and one small principal curvature.  For each scale
the image is smoothed with a Gaussian, the Hessian is computed, and its two
eigenvalues (|l1| <= |l2|) give

    Rb = |l1| / |l2|                       (blobness, small for tubes)
    S  = sqrt(l1^2 + l2^2)                 (structure strength)
    V  = exp(-Rb^2 / (2 beta^2)) * (1 - exp(-S^2 / (2 c^2)))

restricted to pixels where l2 has the sign that matches the polarity (negative
for bright structures on a dark background).  The response is maximised over
scales and normalised to [0, 1], so the panel threshold behaves like the learned
methods.

Gaussian smoothing is done in the frequency domain, which keeps the cost low
without pulling in SciPy.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SIGMA_MIN = 1.0
DEFAULT_SIGMA_MAX = 3.0
DEFAULT_SCALES = 4
DEFAULT_BETA = 0.5
EPS = 1e-9


def describe() -> str:
    return "Frangi vesselness"


def _gaussian_smooth(image: np.ndarray, sigma: float) -> np.ndarray:
    height, width = image.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.fftfreq(width)[None, :]
    transfer = np.exp(-2.0 * (np.pi ** 2) * (sigma ** 2) * (fx ** 2 + fy ** 2))
    spectrum = np.fft.fft2(image)
    return np.real(np.fft.ifft2(spectrum * transfer))


def _eigenvalues(hessian_xx, hessian_yy, hessian_xy):
    trace = hessian_xx + hessian_yy
    determinant = hessian_xx * hessian_yy - hessian_xy * hessian_xy
    root = np.sqrt(np.clip(trace * trace / 4.0 - determinant, 0.0, None))
    first = trace / 2.0 - root
    second = trace / 2.0 + root
    swap = np.abs(first) > np.abs(second)
    smaller = np.where(swap, second, first)
    larger = np.where(swap, first, second)
    return smaller, larger


def segment(image: np.ndarray, sigma_min: float = DEFAULT_SIGMA_MIN,
            sigma_max: float = DEFAULT_SIGMA_MAX, scales: int = DEFAULT_SCALES,
            beta: float = DEFAULT_BETA, polarity: str = "bright",
            threshold: float = 0.5) -> np.ndarray:
    """Return the normalised vesselness response in [0, 1]."""
    img = np.asarray(image, dtype=np.float64)
    peak = float(np.max(img)) if img.size else 0.0
    if peak > 1.5:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    count = max(1, int(scales))
    minimum = max(0.5, float(sigma_min))
    maximum = max(minimum, float(sigma_max))
    sigmas = np.linspace(minimum, maximum, count)

    best = np.zeros_like(img)
    for sigma in sigmas:
        smooth = _gaussian_smooth(img, float(sigma))
        gradient_y, gradient_x = np.gradient(smooth)
        hessian_yy, hessian_xy = np.gradient(gradient_y)
        _, hessian_xx = np.gradient(gradient_x)
        hessian_xy = 0.5 * (hessian_xy + np.gradient(gradient_x, axis=0)[1])

        smaller, larger = _eigenvalues(hessian_xx, hessian_yy, hessian_xy)
        blobness = np.abs(smaller) / (np.abs(larger) + EPS)
        strength = np.sqrt(smaller ** 2 + larger ** 2)
        reference = 0.5 * float(np.max(strength)) if np.max(strength) > 0 else 1.0
        vesselness = (np.exp(-(blobness ** 2) / (2.0 * float(beta) ** 2))
                      * (1.0 - np.exp(-(strength ** 2) / (2.0 * reference ** 2))))
        if str(polarity).lower().startswith("dark"):
            vesselness = np.where(larger > 0, vesselness, 0.0)
        else:
            vesselness = np.where(larger < 0, vesselness, 0.0)
        best = np.maximum(best, vesselness)

    top = float(np.max(best)) if best.size else 0.0
    if top <= 0:
        return np.zeros_like(best, dtype=np.float32)
    return (best / top).astype(np.float32)
