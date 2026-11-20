# Otsu threshold

The classical baseline: one global intensity cut, chosen by maximising the
between-class variance of the histogram (Otsu, 1979). It is here as the control
for every learned method — if a network cannot beat this on the same ground
truth, the network is not doing anything useful.

**This addon used to be bundled inside the application** (`src/amis/addons/threshold`).
It now lives at addon level for two reasons: the core should ship no method at
all, and the addon could not be reviewed or fixed without a source build.

## Interface

`panel.json` is the whole GUI — image, invert flag, *Segment*, *Save mask*,
preview, metrics, log. There is no Qt layout code and no PyTorch dependency.

## Behaviour

| step | detail |
|---|---|
| read | `amis.ml.data.load_gray` (8/16-bit, float, stacked TIFF, colour → luminance) |
| threshold | Otsu over a 256-bin histogram of `[0, 1]` |
| mask | `image > threshold`, optionally inverted |
| output | `uint8` 0/1; *Save mask* writes PNG |
| metrics | Dice, IoU, precision, sensitivity against a reference mask, when one is found |

The metrics line always names the reference file and how it was interpreted, e.g.
`labels 1..13 collapsed to foreground` or `binary`, so a number can never be
mistaken for a per-class score.

## Known limitations (read before quoting a number)

1. **One global cut.** It has no notion of position, so on an OCT b-scan the
   bright choroid and bright fluid are the same object to it, and the thin
   layer boundaries are lost. Expected Dice against multi-layer ground truth is
   low; that is the method, not a bug.
2. **No multi-class output.** Comparison is foreground vs background, whatever
   the reference contains.
3. **Sensitive to the intensity range.** A wrongly scaled image (the 16-bit bug
   fixed in AMIS 0.1.4) produced an all-foreground mask; the addon now warns
   when the mask covers ≥ 99.5 % or ≤ 0.5 % of the frame.
4. **No denoising.** Speckle is segmented as signal; `flatfield` or `frangi`
   in front of it change the result substantially.

## Reference lookup

A mask is looked for in `masks/<stem>` beside the image, `../masks/<stem>`, and
`<stem>_mask` / `<stem>_label` in the same folder. A bare `<stem>.png` is
deliberately **not** accepted: that matches previews and same-named files, which
is how a meaningless Dice appeared in an earlier build.

## Install

```bash
cp -r threshold ~/.local/share/AMIS/addons/     # then: Reload addons
```

or point the app at the folder:

```bash
AMIS_ADDON_PATH="$PWD/.." python -m amis
```

## Licence

Code GPL-3.0-or-later. No trained weights, so no dataset licence applies.
