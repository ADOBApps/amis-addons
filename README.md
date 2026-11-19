# AMIS addons

Public catalogue for the **AMIS addon marketplace**. The app reads
`index.json` from this repository; the packages themselves live in
[Releases](../../releases), not in Git, so this repo stays small.

```
index.json                       the catalogue (schema v1)
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

| id | name | licence | what it is |
|---|---|---|---|
| `attention_unet` | Attention U-Net | GPL-3.0-or-later | Oktay et al. (2018) attention-gated U-Net; the ablation baseline for WRANet |

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
