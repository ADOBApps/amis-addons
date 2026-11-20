# AMIS addons

Public catalogue for the **AMIS addon marketplace**. The app reads
`index.json` from this repository; the packages themselves live in
[Releases](../../releases), not in Git, so this repo stays small.

```
index.json                       the catalogue (schema v1)
sources/<id>/                    source of every published package
.github/validate_catalog.py      validator, run on every PR
.github/workflows/validate.yml   CI wiring
README.md                        this file
CONTRIBUTING.md                  how to submit an addon
```

## Using it in AMIS

1. **Toolbar → Addon marketplace**.
2. Press *Refresh* (the default source is this repository's `index.json`).
3. Pick an addon, read the version, size and SHA-256, press *Install*.
4. The addon appears in the method list immediately; remove it the same way.

No account, token or subscription is needed to browse or install.

## What is in the catalogue

| id | name | licence | what it is | params |
|---|---|---|---|---|
| `threshold` | Otsu threshold | GPL-3.0-or-later | classical baseline, no PyTorch; no weights. Left the application core in 0.1.5 | 0 |
| `refwranet` | refWRANet | GPL-3.0-or-later | multilayer retina (14 classes) **+ 2-D DWT explorer**; bundled demo model (Dice 0.636); works on AMIS 0.1.4+ | 2.03 M |
| `wranet_layers` | WRANet Layers | GPL-3.0-or-later | multi-class retinal layers and fluid spaces (14 classes); needs AMIS 0.1.3 | 32.47 M (base 64) |
| `wranet` | WRANet | GPL-3.0-or-later | wavelet residual attention U-Net (Zhao et al., 2023); the project's reference method, shipped as an addon so it iterates without rebuilding the app | 32.47 M |
| `attention_unet` | Attention U-Net | GPL-3.0-or-later | Oktay et al. (2018) attention-gated U-Net; ablation baseline for WRANet | 0.09 M |
| `unet` | U-Net | GPL-3.0-or-later | Ronneberger et al. (2015) reference architecture | 22.45 M |
| `unetpp` | UNet++ | GPL-3.0-or-later | Zhou et al. (2018) nested dense decoder | 2.29 M |
| `resunet` | ResUNet | GPL-3.0-or-later | residual blocks in encoder and decoder | 2.03 M |
| `flatfield` | Flat-field threshold | GPL-3.0-or-later | illumination-corrected relative threshold (no training) | -- |
| `frangi` | Frangi vesselness | GPL-3.0-or-later | multiscale Hessian vesselness for vessels/aneurysms | -- |

The three learned networks share one panel (data, model, training, prediction,
ONNX export) and the same training loop, so their numbers are directly
comparable; the two classical methods need no training and give a reference on
the same data. Sources for every package live in `sources/<id>/`, which keeps
the GPL corresponding-source requirement satisfied without shipping ZIPs in
Git history.

## Rules

- Every version entry carries `url`, `sha256` and `size`; AMIS refuses to unpack
  anything whose digest does not match.
- `api` must equal the AMIS addon API level the package targets; `min_amis`
  keeps a package out of older builds.
- Packages are Python source. Everything published here is publicly
  downloadable: **never submit private, paid or NDA-covered code, clinical
  images, or weights derived from patient data without written permission.**
- Addon authors keep their copyright; each package declares its licence in its
  manifest and in the catalogue entry.

## Validating locally

```bash
python .github/validate_catalog.py
```

It checks the schema, the digest format, the URLs, and — when a package can be
downloaded — that the `manifest.json` inside matches the catalogue entry.

Maintainer: Acxel Orozco ([@ADOBApps](https://github.com/ADOBApps)).
