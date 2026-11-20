"""Otsu threshold - the classical baseline, now a normal addon.

This used to live in the core (``src/amis/addons/threshold``). It is here
because the core must ship no method at all: the boundary between "the
application" and "a method" then cannot drift.

There is **no Qt code** in this file. The interface is ``panel.json`` (open it in
the visual designer) and the behaviour is three handler methods whose names match
the ``on_click`` fields in that document.

## What it does and does not do

Otsu picks the intensity that maximises the between-class variance of the
histogram, i.e. a single global cut. That is a sensible control for a learned
model, and it is a poor segmenter on its own:

* it has no notion of *where* an object is, so bright choroid and bright fluid
  are the same thing to it;
* on OCT b-scans the layer boundaries are thin and low contrast, so a global cut
  keeps the thick bright bands and loses everything else;
* it cannot produce more than one class.

If a mask is available it is compared as **foreground**, i.e. any non-zero label.
That is stated in the metrics line, because comparing a binary map with a
14-class label image and reporting a single Dice without saying so is how
nonsense numbers get into a thesis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog

from amis.core.addon_api import AddonBase
from amis.ml import data as data_io
from amis.ml import metrics as metric_lib


def otsu_threshold(image: np.ndarray, bins: int = 256) -> float:
    """Otsu's method on a float image in [0, 1]; returns the threshold."""
    values = np.clip(np.asarray(image, dtype=np.float64), 0.0, 1.0).ravel()
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return 0.5

    centres = (edges[:-1] + edges[1:]) / 2.0
    weights = np.cumsum(histogram) / total
    means = np.cumsum(histogram * centres) / np.maximum(np.cumsum(histogram), 1e-12)
    global_mean = float(np.sum(histogram * centres) / total)

    numerator = (global_mean * weights - means) ** 2
    denominator = weights * (1.0 - weights)
    score = np.divide(numerator, denominator,
                      out=np.zeros_like(numerator), where=denominator > 1e-12)
    return float(centres[int(np.argmax(score))])


class ThresholdAddon(AddonBase):
    """Segments with Otsu; proves the declarative-panel route end to end."""

    def initialize(self) -> None:
        self._prediction: Optional[np.ndarray] = None
        self._image: Optional[np.ndarray] = None
        self._threshold: float = 0.5
        self.ui = None

    # -- handlers referenced by panel.json --------------------------------
    def on_run_clicked(self) -> None:
        path = self.ui.get("image_path")
        if not path:
            self.ui.set("log", "Pick an image first")
            return

        # One override, one restore: the previous version restored the cursor in
        # both the except branch and a finally block, which pops the stack twice
        # and can leave the whole application with a wait cursor.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            QApplication.processEvents()
            image = data_io.load_gray(path)
        except Exception as exc:
            self.ui.set("log", f"could not read image ({Path(path).suffix}): {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._image = image
        self._threshold = otsu_threshold(image)
        mask = image > self._threshold
        if self.ui.get("invert"):
            mask = ~mask
        self._prediction = mask.astype(np.uint8)

        preview = self.ui.values.get("preview")
        if preview is not None and hasattr(preview, "set_image"):
            preview.set_image(self._prediction.astype(np.float32))

        fraction = 100.0 * float(self._prediction.mean())
        self.ui.set("log",
                    f"Otsu threshold = {self._threshold:.4f} on {Path(path).name} "
                    f"({image.shape[1]}x{image.shape[0]}, "
                    f"{int(self._prediction.sum())} px = {fraction:.1f} % of the frame)")
        if fraction >= 99.5 or fraction <= 0.5:
            # the usual causes are a wrongly scaled image or a flat frame
            self.ui.set("log", self.ui.get("log")
                        + "\nwarning: the mask is (almost) the whole frame or empty; "
                          "check the image range and try Invert")
        self._report_metrics(path)

    def on_save_clicked(self) -> None:
        if self._prediction is None:
            self.ui.set("log", "Nothing to save yet")
            return
        path, _ = QFileDialog.getSaveFileName(
            self.context.qt_parent, "Save mask",
            str(Path.home() / "otsu_mask.png"), "PNG (*.png)")
        if path:
            data_io.save_mask(path, self._prediction)
            self.ui.set("log", f"saved {path}")

    # -- helpers ----------------------------------------------------------
    def _truth_for(self, image_path: str) -> Optional[Path]:
        """Find a reference mask next to the image.

        Only unambiguous layouts are accepted: ``masks/<stem>`` beside or above
        the image, or ``<stem>_mask`` / ``<stem>_label`` in the same folder.
        A bare ``<stem>.png`` is *not* taken, because that matches a preview or
        any same-named file and silently produces a meaningless Dice.
        """
        image = Path(image_path)
        suffixes = (".png", ".tif", ".tiff", ".bmp")
        patterns = ("{stem}", "{stem}_mask", "{stem}_label")
        for folder in (image.parent / "masks", image.parent.parent / "masks"):
            for pattern in patterns:
                for suffix in suffixes:
                    candidate = folder / (pattern.format(stem=image.stem) + suffix)
                    if candidate.is_file():
                        return candidate
        for pattern in patterns[1:]:
            for suffix in suffixes:
                candidate = image.parent / (pattern.format(stem=image.stem) + suffix)
                if candidate.is_file():
                    return candidate
        return None

    def _binary_truth(self, truth_path: Path) -> Tuple[np.ndarray, str]:
        """Read the reference, collapsing a label map to foreground.

        ``load_mask`` gives {0, 1} for a binary mask. For a multi-class label
        image every non-zero label becomes foreground; the note says so, and the
        metrics line repeats it, so the number cannot be mistaken for a
        per-class score.
        """
        if truth_path.suffix.lower() in (".tif", ".tiff"):
            try:
                import tifffile

                raw = tifffile.imread(str(truth_path))
                if raw.ndim == 3:
                    raw = raw[0]
                if raw.dtype.kind in "iu":
                    distinct = sorted(int(value) for value in np.unique(raw) if value)
                    if len(distinct) > 1:
                        note = f"labels {distinct[0]}..{distinct[-1]} collapsed to foreground"
                        return (raw > 0).astype(np.uint8), note
            except Exception as exc:               # fall back to the core loader
                self.context.logger.debug("tifffile on %s failed: %s", truth_path, exc)
        return data_io.load_mask(truth_path), "binary"

    def _report_metrics(self, image_path: str) -> None:
        truth_path = self._truth_for(image_path)
        if truth_path is None:
            self.ui.set("metrics", "no reference mask found next to the image - "
                                   "metrics unavailable")
            return
        try:
            truth, note = self._binary_truth(truth_path)
        except Exception as exc:
            self.ui.set("metrics", f"could not read the reference mask: {exc}")
            return
        truth = truth[: self._image.shape[0], : self._image.shape[1]]
        prediction = self._prediction[: truth.shape[0], : truth.shape[1]]
        if truth.size == 0:
            self.ui.set("metrics", "reference mask is empty")
            return
        values = metric_lib.all_metrics(prediction, truth)
        self.ui.set(
            "metrics",
            f"dice {values['dice']:.4f}   iou {values['iou']:.4f}   "
            f"precision {values['precision']:.4f}   "
            f"sensitivity {values['sensitivity']:.4f}\n"
            f"reference: {truth_path.name} ({note}, "
            f"{100.0 * float(truth.mean()):.1f} % foreground)",
        )


ADDON = ThresholdAddon
