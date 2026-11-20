"""Wavelet filter bank and a step-by-step 2-D DWT.

Two things live here:

* ``FILTERS`` - exact analysis taps for the families this addon offers (Haar,
  Daubechies, Symlets, biorthogonal and reverse biorthogonal), generated from
  PyWavelets once and embedded, so the addon needs no dependency;
* ``dwt2_steps`` - the transform written in the order a reader should follow it:
  filter every row (low and high pass), then every column of each result, and
  keep the four sub-bands LL, LH, HL, HH.

Conventions: images are float arrays in [0, 1], shape (H, W); filters are applied
with symmetric extension and decimated by two, and the extension is chosen so an
even-sized image gives exactly half-sized sub-bands. This is the decomposition
the network uses (analysis only): the inverse transform is not needed to
segment, and the addon does not claim one.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

#: name -> analysis taps, orthogonal flag and family label
FILTERS: Dict[str, Dict] = {
    "haar": {
        "dec_lo": [
            0.7071067812,
            0.7071067812
        ],
        "dec_hi": [
            -0.7071067812,
            0.7071067812
        ],
        "rec_lo": [
            0.7071067812,
            0.7071067812
        ],
        "rec_hi": [
            0.7071067812,
            -0.7071067812
        ],
        "orthogonal": True,
        "family": "Haar"
    },
    "db2": {
        "dec_lo": [
            -0.1294095226,
            0.224143868,
            0.8365163037,
            0.4829629131
        ],
        "dec_hi": [
            -0.4829629131,
            0.8365163037,
            -0.224143868,
            -0.1294095226
        ],
        "rec_lo": [
            0.4829629131,
            0.8365163037,
            0.224143868,
            -0.1294095226
        ],
        "rec_hi": [
            -0.1294095226,
            -0.224143868,
            0.8365163037,
            -0.4829629131
        ],
        "orthogonal": True,
        "family": "Daubechies"
    },
    "db4": {
        "dec_lo": [
            -0.0105974018,
            0.0328830117,
            0.0308413818,
            -0.1870348117,
            -0.0279837694,
            0.6308807679,
            0.7148465706,
            0.2303778133
        ],
        "dec_hi": [
            -0.2303778133,
            0.7148465706,
            -0.6308807679,
            -0.0279837694,
            0.1870348117,
            0.0308413818,
            -0.0328830117,
            -0.0105974018
        ],
        "rec_lo": [
            0.2303778133,
            0.7148465706,
            0.6308807679,
            -0.0279837694,
            -0.1870348117,
            0.0308413818,
            0.0328830117,
            -0.0105974018
        ],
        "rec_hi": [
            -0.0105974018,
            -0.0328830117,
            0.0308413818,
            0.1870348117,
            -0.0279837694,
            -0.6308807679,
            0.7148465706,
            -0.2303778133
        ],
        "orthogonal": True,
        "family": "Daubechies"
    },
    "db6": {
        "dec_lo": [
            -0.0010773011,
            0.0047772575,
            0.0005538422,
            -0.0315820393,
            0.0275228655,
            0.0975016056,
            -0.1297668676,
            -0.226264694,
            0.3152503517,
            0.751133908,
            0.4946238904,
            0.1115407434
        ],
        "dec_hi": [
            -0.1115407434,
            0.4946238904,
            -0.751133908,
            0.3152503517,
            0.226264694,
            -0.1297668676,
            -0.0975016056,
            0.0275228655,
            0.0315820393,
            0.0005538422,
            -0.0047772575,
            -0.0010773011
        ],
        "rec_lo": [
            0.1115407434,
            0.4946238904,
            0.751133908,
            0.3152503517,
            -0.226264694,
            -0.1297668676,
            0.0975016056,
            0.0275228655,
            -0.0315820393,
            0.0005538422,
            0.0047772575,
            -0.0010773011
        ],
        "rec_hi": [
            -0.0010773011,
            -0.0047772575,
            0.0005538422,
            0.0315820393,
            0.0275228655,
            -0.0975016056,
            -0.1297668676,
            0.226264694,
            0.3152503517,
            -0.751133908,
            0.4946238904,
            -0.1115407434
        ],
        "orthogonal": True,
        "family": "Daubechies"
    },
    "db8": {
        "dec_lo": [
            -0.0001174768,
            0.0006754494,
            -0.0003917404,
            -0.004870353,
            0.008746094,
            0.0139810279,
            -0.0440882539,
            -0.017369301,
            0.1287474266,
            0.0004724846,
            -0.284015543,
            -0.0158291053,
            0.5853546837,
            0.6756307363,
            0.3128715909,
            0.0544158422
        ],
        "dec_hi": [
            -0.0544158422,
            0.3128715909,
            -0.6756307363,
            0.5853546837,
            0.0158291053,
            -0.284015543,
            -0.0004724846,
            0.1287474266,
            0.017369301,
            -0.0440882539,
            -0.0139810279,
            0.008746094,
            0.004870353,
            -0.0003917404,
            -0.0006754494,
            -0.0001174768
        ],
        "rec_lo": [
            0.0544158422,
            0.3128715909,
            0.6756307363,
            0.5853546837,
            -0.0158291053,
            -0.284015543,
            0.0004724846,
            0.1287474266,
            -0.017369301,
            -0.0440882539,
            0.0139810279,
            0.008746094,
            -0.004870353,
            -0.0003917404,
            0.0006754494,
            -0.0001174768
        ],
        "rec_hi": [
            -0.0001174768,
            -0.0006754494,
            -0.0003917404,
            0.004870353,
            0.008746094,
            -0.0139810279,
            -0.0440882539,
            0.017369301,
            0.1287474266,
            -0.0004724846,
            -0.284015543,
            0.0158291053,
            0.5853546837,
            -0.6756307363,
            0.3128715909,
            -0.0544158422
        ],
        "orthogonal": True,
        "family": "Daubechies"
    },
    "sym4": {
        "dec_lo": [
            -0.0757657148,
            -0.0296355276,
            0.4976186676,
            0.8037387518,
            0.2978577956,
            -0.0992195436,
            -0.0126039673,
            0.0322231006
        ],
        "dec_hi": [
            -0.0322231006,
            -0.0126039673,
            0.0992195436,
            0.2978577956,
            -0.8037387518,
            0.4976186676,
            0.0296355276,
            -0.0757657148
        ],
        "rec_lo": [
            0.0322231006,
            -0.0126039673,
            -0.0992195436,
            0.2978577956,
            0.8037387518,
            0.4976186676,
            -0.0296355276,
            -0.0757657148
        ],
        "rec_hi": [
            -0.0757657148,
            0.0296355276,
            0.4976186676,
            -0.8037387518,
            0.2978577956,
            0.0992195436,
            -0.0126039673,
            -0.0322231006
        ],
        "orthogonal": True,
        "family": "Symlets"
    },
    "sym6": {
        "dec_lo": [
            0.0154041093,
            0.0034907121,
            -0.1179901111,
            -0.0483117426,
            0.4910559419,
            0.787641141,
            0.3379294217,
            -0.0726375228,
            -0.0210602925,
            0.0447249018,
            0.0017677119,
            -0.0078007083
        ],
        "dec_hi": [
            0.0078007083,
            0.0017677119,
            -0.0447249018,
            -0.0210602925,
            0.0726375228,
            0.3379294217,
            -0.787641141,
            0.4910559419,
            0.0483117426,
            -0.1179901111,
            -0.0034907121,
            0.0154041093
        ],
        "rec_lo": [
            -0.0078007083,
            0.0017677119,
            0.0447249018,
            -0.0210602925,
            -0.0726375228,
            0.3379294217,
            0.787641141,
            0.4910559419,
            -0.0483117426,
            -0.1179901111,
            0.0034907121,
            0.0154041093
        ],
        "rec_hi": [
            0.0154041093,
            -0.0034907121,
            -0.1179901111,
            0.0483117426,
            0.4910559419,
            -0.787641141,
            0.3379294217,
            0.0726375228,
            -0.0210602925,
            -0.0447249018,
            0.0017677119,
            0.0078007083
        ],
        "orthogonal": True,
        "family": "Symlets"
    },
    "sym8": {
        "dec_lo": [
            -0.003382416,
            -0.0005421323,
            0.0316950878,
            0.0076074873,
            -0.1432942384,
            -0.0612733591,
            0.4813596513,
            0.7771857517,
            0.3644418948,
            -0.0519458381,
            -0.0272190299,
            0.0491371797,
            0.003808752,
            -0.0149522583,
            -0.0003029205,
            0.0018899503
        ],
        "dec_hi": [
            -0.0018899503,
            -0.0003029205,
            0.0149522583,
            0.003808752,
            -0.0491371797,
            -0.0272190299,
            0.0519458381,
            0.3644418948,
            -0.7771857517,
            0.4813596513,
            0.0612733591,
            -0.1432942384,
            -0.0076074873,
            0.0316950878,
            0.0005421323,
            -0.003382416
        ],
        "rec_lo": [
            0.0018899503,
            -0.0003029205,
            -0.0149522583,
            0.003808752,
            0.0491371797,
            -0.0272190299,
            -0.0519458381,
            0.3644418948,
            0.7771857517,
            0.4813596513,
            -0.0612733591,
            -0.1432942384,
            0.0076074873,
            0.0316950878,
            -0.0005421323,
            -0.003382416
        ],
        "rec_hi": [
            -0.003382416,
            0.0005421323,
            0.0316950878,
            -0.0076074873,
            -0.1432942384,
            0.0612733591,
            0.4813596513,
            -0.7771857517,
            0.3644418948,
            0.0519458381,
            -0.0272190299,
            -0.0491371797,
            0.003808752,
            0.0149522583,
            -0.0003029205,
            -0.0018899503
        ],
        "orthogonal": True,
        "family": "Symlets"
    },
    "bior2.2": {
        "dec_lo": [
            0.0,
            -0.1767766953,
            0.3535533906,
            1.0606601718,
            0.3535533906,
            -0.1767766953
        ],
        "dec_hi": [
            -0.0,
            0.3535533906,
            -0.7071067812,
            0.3535533906,
            -0.0,
            0.0
        ],
        "rec_lo": [
            0.0,
            0.3535533906,
            0.7071067812,
            0.3535533906,
            0.0,
            0.0
        ],
        "rec_hi": [
            0.0,
            0.1767766953,
            0.3535533906,
            -1.0606601718,
            0.3535533906,
            0.1767766953
        ],
        "orthogonal": False,
        "family": "Biorthogonal"
    },
    "bior4.4": {
        "dec_lo": [
            0.0,
            0.0378284555,
            -0.023849465,
            -0.1106244044,
            0.3774028556,
            0.852698679,
            0.3774028556,
            -0.1106244044,
            -0.023849465,
            0.0378284555
        ],
        "dec_hi": [
            -0.0,
            -0.0645388826,
            0.0406894176,
            0.4180922732,
            -0.7884856164,
            0.4180922732,
            0.0406894176,
            -0.0645388826,
            -0.0,
            0.0
        ],
        "rec_lo": [
            0.0,
            -0.0645388826,
            -0.0406894176,
            0.4180922732,
            0.7884856164,
            0.4180922732,
            -0.0406894176,
            -0.0645388826,
            0.0,
            0.0
        ],
        "rec_hi": [
            0.0,
            -0.0378284555,
            -0.023849465,
            0.1106244044,
            0.3774028556,
            -0.852698679,
            0.3774028556,
            0.1106244044,
            -0.023849465,
            -0.0378284555
        ],
        "orthogonal": False,
        "family": "Biorthogonal"
    },
    "rbio2.2": {
        "dec_lo": [
            0.0,
            0.0,
            0.3535533906,
            0.7071067812,
            0.3535533906,
            0.0
        ],
        "dec_hi": [
            0.1767766953,
            0.3535533906,
            -1.0606601718,
            0.3535533906,
            0.1767766953,
            0.0
        ],
        "rec_lo": [
            -0.1767766953,
            0.3535533906,
            1.0606601718,
            0.3535533906,
            -0.1767766953,
            0.0
        ],
        "rec_hi": [
            0.0,
            -0.0,
            0.3535533906,
            -0.7071067812,
            0.3535533906,
            -0.0
        ],
        "orthogonal": False,
        "family": "Reverse biorthogonal"
    },
    "rbio4.4": {
        "dec_lo": [
            0.0,
            0.0,
            -0.0645388826,
            -0.0406894176,
            0.4180922732,
            0.7884856164,
            0.4180922732,
            -0.0406894176,
            -0.0645388826,
            0.0
        ],
        "dec_hi": [
            -0.0378284555,
            -0.023849465,
            0.1106244044,
            0.3774028556,
            -0.852698679,
            0.3774028556,
            0.1106244044,
            -0.023849465,
            -0.0378284555,
            0.0
        ],
        "rec_lo": [
            0.0378284555,
            -0.023849465,
            -0.1106244044,
            0.3774028556,
            0.852698679,
            0.3774028556,
            -0.1106244044,
            -0.023849465,
            0.0378284555,
            0.0
        ],
        "rec_hi": [
            0.0,
            -0.0,
            -0.0645388826,
            0.0406894176,
            0.4180922732,
            -0.7884856164,
            0.4180922732,
            0.0406894176,
            -0.0645388826,
            -0.0
        ],
        "orthogonal": False,
        "family": "Reverse biorthogonal"
    }
}

ORTHOGONAL = tuple(name for name, spec in FILTERS.items() if spec["orthogonal"])
BIORTHOGONAL = tuple(name for name, spec in FILTERS.items() if not spec["orthogonal"])
ALL_WAVELETS = tuple(FILTERS)

#: The network uses the orthogonal families: their low-pass analysis filter is
#: all the DWT layer needs, and orthogonality keeps the energy interpretation
#: ("LL" really is a smoothed version of the image).
MODEL_WAVELETS = ORTHOGONAL


def _pad_for(taps: int) -> int:
    """Total symmetric padding giving exactly half the size after decimation."""
    return max(0, taps - 2)


def _filter_rows(image: np.ndarray, low: List[float], high: List[float]):
    """Step 1: one-dimensional filtering along the rows, decimated by two."""
    pad = _pad_for(len(low))
    before = pad // 2
    after = pad - before
    padded = np.pad(image, ((0, 0), (before, after)), mode="symmetric")
    rows_low = np.apply_along_axis(
        lambda row: np.convolve(row, low, mode="valid")[::2], 1, padded)
    rows_high = np.apply_along_axis(
        lambda row: np.convolve(row, high, mode="valid")[::2], 1, padded)
    return rows_low, rows_high


def _filter_cols(image: np.ndarray, low: List[float], high: List[float]):
    """Step 2: the same filtering along the columns of each row result."""
    pad = _pad_for(len(low))
    before = pad // 2
    after = pad - before
    padded = np.pad(image, ((before, after), (0, 0)), mode="symmetric")
    cols_low = np.apply_along_axis(
        lambda col: np.convolve(col, low, mode="valid")[::2], 0, padded)
    cols_high = np.apply_along_axis(
        lambda col: np.convolve(col, high, mode="valid")[::2], 0, padded)
    return cols_low, cols_high


def dwt2_steps(image: np.ndarray, wavelet: str = "haar") -> Dict[str, np.ndarray]:
    """Every intermediate result of one decomposition level.

    Keys: ``rows_low`` / ``rows_high`` (after the row pass, before the column
    pass) and the four sub-bands ``LL``, ``LH``, ``HL``, ``HH``.
    """
    spec = FILTERS[wavelet]
    image = np.asarray(image, dtype=np.float64)
    rows_low, rows_high = _filter_rows(image, spec["dec_lo"], spec["dec_hi"])
    ll, lh = _filter_cols(rows_low, spec["dec_lo"], spec["dec_hi"])
    hl, hh = _filter_cols(rows_high, spec["dec_lo"], spec["dec_hi"])
    return {"rows_low": rows_low, "rows_high": rows_high,
            "LL": ll, "LH": lh, "HL": hl, "HH": hh}


def dwt2(image: np.ndarray, wavelet: str = "haar"):
    """Sub-bands only: ``(LL, (LH, HL, HH))``."""
    steps = dwt2_steps(image, wavelet)
    return steps["LL"], (steps["LH"], steps["HL"], steps["HH"])


def cascade(image: np.ndarray, wavelet: str = "haar", levels: int = 2):
    """Apply the transform repeatedly to LL, which is what a deeper network does.

    Returns a list of ``(level, steps_dict)`` with level 1 first.
    """
    out = []
    current = np.asarray(image, dtype=np.float64)
    for level in range(1, max(1, levels) + 1):
        steps = dwt2_steps(current, wavelet)
        out.append((level, steps))
        current = steps["LL"]
    return out


def sub_band_stats(step: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Energy and dynamic range per band, for the explanatory text."""
    stats: Dict[str, Dict[str, float]] = {}
    for band in ("LL", "LH", "HL", "HH"):
        values = np.abs(step[band])
        stats[band] = {
            "energy": float(np.sum(values ** 2)),
            "mean_abs": float(np.mean(values)),
            "max_abs": float(np.max(values)) if values.size else 0.0,
        }
    return stats


def describe_band(band: str) -> str:
    return {
        "LL": "low pass in both directions: main structure and global shapes",
        "LH": "low pass along rows, high pass along columns: horizontal edges",
        "HL": "high pass along rows, low pass along columns: vertical edges",
        "HH": "high pass in both directions: diagonal edges, texture, noise",
    }[band]


def describe_wavelet(name: str) -> str:
    spec = FILTERS[name]
    family = spec["family"]
    length = len(spec["dec_lo"])
    note = {
        "Haar": "square step functions: the simplest basis, sharp but blocky",
        "Daubechies": "orthogonal, maximally compact support; smoothness grows with order",
        "Symlets": "quasi-symmetric Daubechies variants that preserve texture better",
        "Biorthogonal": "independent analysis/synthesis bases: linear phase, better edges",
        "Reverse biorthogonal": "the biorthogonal pair used in the opposite direction",
    }.get(family, "wavelet basis")
    return f"{family} ({length} taps) - {note}"
