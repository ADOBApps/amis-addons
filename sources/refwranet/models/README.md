# Bundled model

`refwranet_layers.onnx` is the demo checkpoint that ships with this addon. The
panel uses it automatically when no `.pt` and no ONNX file is selected.

## Contract

| | |
|---|---|
| input | `float32`, NCHW, 1 channel, **160 × 160**, values in `[0, 1]` |
| output | **14 channels** of logits (apply softmax over the channel axis, then argmax) |
| opset | 17, dynamic batch |
| parameters | 2.03 M (base 16, depth 4, haar) |
| parity | max \|PyTorch − ONNX Runtime\| = 3.3e-06 |

In the *Layers* tab set **Image size = 160**; the panel resizes the slice, runs
the model, and maps the class map back to the native resolution.

## Training record

| | |
|---|---|
| data | OCTAVE (Kermany et al., 2025), 3 volumes → 57 b-scans, 14 classes (10 present) |
| split | 45 train / 12 validation |
| recipe | Adam 1e-3, batch 2, 160 px, CE + macro Dice, 25 epochs, CPU, 103 s |
| result | **mean class Dice 0.636** on validation |

Per class: ART 0.99, CHO 0.99, VIT 0.98, RET 0.93, RPE 0.69, FLU 0.31,
others absent from the validation split. Large structures are learned quickly;
**fluid (FLU) is the hard class**, which is exactly what the thesis is about —
it needs more volumes, higher resolution and probably the `keep_high` ablation.

This is a demonstration checkpoint on three volumes, not a clinical model.

## Licence

Code GPL-3.0-or-later; **these weights are CC BY-SA 4.0** because they derive
from OCTAVE annotations. Redistribution requires citing the dataset
(doi:10.5281/zenodo.14580071) and the paper (doi:10.1167/iovs.66.6.55), and the
same licence.
