# Models

This version ships **no weights**. Version 2.0.0 changed the topology (max-pool
encoder, channel + spatial RAM, transposed-conv decoder, 1x1 head), so the
`wranet` 1.1.x checkpoints and ONNX exports - including the trained 14-class
demo from 1.1.4 - are not compatible and are deliberately not copied here.

`wranet` 1.1.4 (trained 14-class demo, mean class Dice 0.638) remains published
in the catalogue if you need out-of-the-box predictions with the old graph.

The panel still looks for `models/*.onnx` and picks the first one when no `.pt`
file is selected, so dropping a trained export into this directory makes the
addon work without any clicks.

## Producing one

1. **Layers** tab: set the folders and the training controls, press
   *Start training*. `best.pt`, `last.pt` and `history.json` are written to the
   addon's data directory (`<data>/AMIS/addon-data/wranet`).
2. Press *Export ONNX*: writes `wranet.onnx` next to the checkpoints, traces at
   the *Image size* in the panel, verifies PyTorch/ONNX Runtime parity and
   writes the JSON sidecar.
3. Copy the `.onnx` and its `.json` sidecar here if the model should travel
   with the addon.

## Contract

| | |
|---|---|
| input | `float32`, NCHW, 1 channel, any size (internally padded to a multiple of `2**depth`) |
| output | **14 channels** of logits (apply softmax over the channel axis, then argmax) |
| downsampling | fixed at export time: `maxpool` or `dwt` + family, whichever the panel had selected |
| sizing | free: the network has no fixed resolution; exported at the traced size |
| parameters | see *Count parameters* in the panel (≈ 1.23 M at base 16, depth 4, repeats 2) |

## Licence

Code GPL-3.0-or-later. Any weights trained on OCTAVE inherit **CC BY-SA 4.0**
and must cite the dataset (doi:10.5281/zenodo.14580071) and the paper
(doi:10.1167/iovs.66.6.55).
