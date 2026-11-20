# Bundled model

`wranet_octave_fluid.onnx` is a trained WRANet checkpoint, exported and verified
by `scripts/export_onnx.py` in the AMIS repository. The panel uses it
automatically when no `.pt` weights are selected.

## Contract

| | |
|---|---|
| input | `float32`, NCHW, 1 channel, **192 × 192**, values in `[0, 1]` |
| output | logits (apply sigmoid); one channel |
| opset | 17, dynamic batch |
| parameters | 2.03 M |
| parity | max \|PyTorch − ONNX Runtime\| = 5.7e-06 |
| sha256 | see `wranet_octave_fluid.json` / the release notes |

In the Predict tab set **Image size = 192** so the slice is resized to the
resolution the weights were fitted for; the panel then upsamples the
probability map back to the native size of the image.

## Training record

| | |
|---|---|
| data | OCTAVE (Kermany et al., 2025), 198 volumes → 3 762 b-scans, class 3 (FLU) |
| subset | all 569 fluid slices + 900 randomly sampled empty slices (seed 0) |
| split | 1 175 train / 294 validation slices |
| recipe | Adam, lr 1e-3, batch 4, 192 px, BCE + Dice, 10 epochs, CPU (~16 min) |
| metrics | **Dice on the 109 fluid validation slices: 0.450** (all slices 0.252, empty 0.135) |

This is a deliberately small run: a research checkpoint that proves the
pipeline, **not** a clinical model. Expect a much better number with 256 px,
more epochs, and the full data.

## Licence

* the **code** of this addon: GPL-3.0-or-later;
* the **weights**: CC BY-SA 4.0, because they were trained on OCTAVE-derived
  annotations. If you redistribute the model, cite the dataset
  (doi:10.5281/zenodo.14580071) and the paper (doi:10.1167/iovs.66.6.55) and keep
  the same licence.
