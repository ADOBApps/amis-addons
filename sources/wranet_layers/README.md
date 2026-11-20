# WRANet Layers

Multi-class retinal segmentation with the WRANet backbone: **one output channel
per class** instead of a single foreground logit.

    background | SRM | HRM | FLU | HTD | RPE | RET | CHO | VIT | HYA | RHS | ART | ERM | SES

Those are OCTAVE's 14 classes (Kermany et al., 2025); the addon reads them from
`classes.json` next to the dataset folders, so a different label scheme needs
only a new `classes.json`.

## Why wavelets help here

Layer boundaries are thin and low-contrast, so pooling tends to erase them.
WRANet replaces pooling with a fixed discrete wavelet transform and keeps only
the low-frequency sub-band (LL): the noisy high-frequency detail is discarded
while the boundary structure, which is low-frequency along the layer, survives.
The residual attention gates then weight each skip pixel at the resolution where
the boundary lives. This addon exposes the two knobs that matter for the
ablation: `wavelet` (haar, db2, db4) and `keep_high` (propagate LH/HL/HH instead
of discarding them).

## Data

```
dataset/
  images/   <stem>.png          8-bit grayscale b-scans
  masks/    <stem>.png          class-index map (0..N-1), NOT 0/255
  classes.json                  {"num_classes": 14, "names": [...]}
```

Build one from OCTAVE with:

```bash
python scripts/octave_to_amis.py \
    --src  nnUNet_raw/Dataset001_OCTAVE \
    --out  octave_layers --multiclass --keep-empty
```

Unlike the binary `wranet` addon, masks keep their class ids — use
`amis.ml.data.load_classes`, never `load_mask` (which binarises at 127).

## Training

CE + macro soft Dice, per-class Dice on the validation split, best checkpoint on
the **mean class Dice**. The log prints one line per epoch with the Dice of every
class that occurs, and the panel repeats it under *Predict → per class*.

## Prediction

`argmax` over the class axis, rendered with a fixed colour palette and saved as a
class-index PNG (values = class id, so it opens as labels in Slicer/napari, not as
a picture). Metrics: per-class Dice and the mean over the classes present in the
reference mask.

## Export

ONNX export writes N output channels (logits). Apply **softmax** over the channel
axis, then argmax; the sidecar records the class count.

## Licences

Code GPL-3.0-or-later. Models trained on OCTAVE-derived masks are CC BY-SA 4.0
and must carry the OCTAVE citations (doi:10.5281/zenodo.14580071,
doi:10.1167/iovs.66.6.55).
