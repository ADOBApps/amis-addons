# WRANet 2.0 (pooling + RAM)

Third-generation `wranet` package. Same data contract and panel workflow as
`refwranet` (class-index masks + `classes.json`, multiclass training, ONNX
export, per-class Dice, 2-D DWT explorer), different backbone - so the two
addons are a direct ablation pair on the same data and recipe.

> **Upgrade note.** 2.0.0 is a new topology. Checkpoints and ONNX files from
> `wranet` 1.1.x (wavelet U-Net, binary or 14-class) do not load into it, and
> this version ships **no weights**. Version 1.1.4 (trained 14-class demo,
> Dice 0.638) stays published if you need out-of-the-box predictions.

## Backbone

Encoder-decoder with attention skip connections:

| stage | operation | tensor |
|---|---|---|
| input | — | `[1, 256, 256]` |
| enc 1 | Conv3x3+BN+ReLU | `[16, 256, 256]` |
| enc 2 | downsample + Conv3x3+BN+ReLU | `[32, 128, 128]` |
| enc 3 | downsample + Conv3x3+BN+ReLU | `[64, 64, 64]` |
| enc 4 | downsample + Conv3x3+BN+ReLU | `[128, 32, 32]` |
| bottleneck | downsample + Conv3x3+BN+ReLU | `[256, 16, 16]` |
| dec 1 | ConvTranspose2d + conv, concat RAM(skip) | `[256, 32, 32]` |
| dec 2 | ConvTranspose2d + conv, concat RAM(skip) | `[128, 64, 64]` |
| dec 3 | ConvTranspose2d + conv, concat RAM(skip) | `[64, 128, 128]` |
| dec 4 | ConvTranspose2d + conv, concat RAM(skip) | `[32, 256, 256]` |
| head | Conv1x1 | `[14, 256, 256]` |

`downsample` is a panel control:

| value | operator | note |
|---|---|---|
| `maxpool` (default) | `MaxPool2d(2, 2)` | the architecture in the spec sheet |
| `dwt` | LL sub-band of a fixed 2-D analysis DWT | ablation; same H/2 x W/2 and the same channel count, so the ladder above does not change |

The DWT families offered by the model are the orthogonal ones (`haar`,
`db2/4/6/8`, `sym4/6/8`); biorthogonal families are rejected by the network and
remain available in the DWT explorer tab.

`RAM` (residual attention model) = channel attention (average + max
squeeze-and-excite) x spatial attention gate driven by the decoder feature,
then `repeats` x (Conv3x3+BN+ReLU, optional Dropout2d) and a residual add that
keeps the original skip. It is applied to **every** skip.

`channel_plan()` in `model.py` computes the ladder for any width, depth, class
count, input size or downsampling choice; the panel prints it and refreshes it
when a control changes, so the GUI and the code cannot disagree.

## Training-free analysis (2-D DWT tab)

The analysis group needs **no weights and no training**. It reads the image
through the same outline using fixed rules:

| Outline element | Training-free operator |
|---|---|
| encoder stages | forward DWT cascade, LL kept at each level |
| skip connections | LH / HL / HH bands of every level |
| RAM gate | `g = abs(c) / (abs(c) + T)`, `T = k * sigma * sqrt(2 ln n)`, `sigma = median(abs(c)) / 0.6745` |
| decoder | inverse DWT with the synthesis taps (exact to ~1e-10) |
| head | maps: reconstruction, detail energy, gate |

Controls: image, wavelet family, levels (1-3), gate strength `k`, threshold
mode (`soft`, `hard`, `none`). Outputs: denoised reconstruction, detail-energy
map, gate map, and a per-level/per-band table (sigma, threshold, kept %,
energy %) that can be saved as CSV; the reconstruction can be saved as PNG.

`none` is a sanity check: it keeps every coefficient, so the reconstruction is
the inverse transform itself (RMSE ~1e-10). The analysis is deterministic and
CPU-only (numpy FFT), and it is **not** a segmentation - class labels need a
trained model.

The analysis uses a *periodized* DWT (`dwt2_periodic` / `idwt2_periodic` in
`wavelib.py`) because that one is exactly invertible; the four band pictures in
the explorer keep the symmetric extension inherited from `refwranet`, so they
match its figures. Interior coefficients are the same; only the border differs.

## Differences from `refwranet`

* downsampling: `MaxPool2d(2, 2)` by default (the spec), with the DWT LL
  sub-band selectable as an ablation; `refwranet` always uses the DWT;
* skips: channel + spatial attention instead of spatial-only attention;
* decoder: `ConvTranspose2d` upsampling instead of bilinear;
* head: a single 1x1 convolution;
* no bundled ONNX weights - see `models/README.md`;
* adds the training-free analysis described above.

## Development

```bash
# architecture, DWT round trip and analysis checks (forward passes only)
python ../tests/check_refwranet2.py .

# the panel smoke through the real registry
QT_QPA_PLATFORM=offscreen PYTHONPATH=<app>/src python ../tests/smoke_refwranet2.py wranet

# the AI gate: manifest, panel, handlers, syntax, real registry load
PYTHONPATH=<app>/src python _devctx/ai-kit/tools/validate_addon.py . --load
```

Point AMIS at the folder that contains this addon:

```bash
AMIS_ADDON_PATH=/path/to/addons python -m amis
```
