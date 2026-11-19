# Flat-field threshold

AMIS addon `flatfield` (classical).

Illumination-robust threshold: estimates the background with a large box mean, divides it out, and keeps pixels a few percent brighter than that local level. Survives the gradient that defeats a global threshold, without the background leakage of Sauvola-style local methods.

## Install

```bash
python -m amis.market install flatfield
```

or copy this folder into the AMIS addons directory and press *Reload addons*.

- addon API: 1
- licence: GPL-3.0-or-later
