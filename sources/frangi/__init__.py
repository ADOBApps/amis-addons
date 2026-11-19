"""Frangi vesselness.

Frangi et al. (1998) multiscale Hessian vesselness: enhances tubular structures such as vessels and aneurysms, with a polarity switch for dark or bright structures.

Classical method: no training, no PyTorch.  ``model.py`` implements the
algorithm as a pure function; this module only moves data between the panel and
that function.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtWidgets import QFileDialog

from amis.core.addon_api import AddonBase
from amis.ml import data as data_io
from amis.ml import metrics as metric_lib

from .model import describe, segment

log = logging.getLogger(__name__)


class FrangiAddon(AddonBase):
    def initialize(self) -> None:
        self._prediction: Optional[np.ndarray] = None
        self._image: Optional[np.ndarray] = None
        self.ui = None

    def build_panel(self):
        panel = super().build_panel()
        self._echo(describe())
        return panel

    def _echo(self, message: str) -> None:
        if self.ui is not None:
            self.ui.set("log", message)
        self.context.logger.info(message)

    def _params(self) -> dict:
        values = {}
        for key in ("sigma_min", "sigma_max", "scales", "beta", "polarity", "threshold"):
            values[key] = self.ui.get(key) if self.ui is not None else None
        return {k: v for k, v in values.items() if v is not None}

    def on_run_clicked(self) -> None:
        path = self.ui.get("image_path", "")
        if not path:
            self._echo("pick an image first")
            return
        try:
            image = data_io.load_gray(path)
            self._image = image
            result = segment(image, **self._params())
        except Exception as exc:
            self._echo(f"segmentation failed: {exc}")
            return

        self._prediction = result
        preview = self.ui.values.get("preview")
        if preview is not None and hasattr(preview, "set_image"):
            preview.set_image(result)
        self._echo(f"{describe()} -> {Path(path).name}")

        truth_path = self._truth_for(path)
        if truth_path is None:
            self.ui.set("metrics", "no ground-truth mask found next to the image")
            return
        truth = data_io.load_mask(truth_path, image.shape)
        values = metric_lib.all_metrics(result, truth)
        self.ui.set("metrics",
                    f"dice {values['dice']:.4f}  iou {values['iou']:.4f}  "
                    f"precision {values['precision']:.4f}  "
                    f"sensitivity {values['sensitivity']:.4f}")

    def _truth_for(self, image_path: str) -> Optional[Path]:
        image = Path(image_path)
        for suffix in (".png", ".tif", ".tiff", ".bmp"):
            for candidate in (image.parent / "masks" / f"{image.stem}{suffix}",
                              image.parent.parent / "masks" / f"{image.stem}{suffix}"):
                if candidate.is_file():
                    return candidate
        for suffix in (".png", ".tif", ".tiff", ".bmp"):
            candidate = image.with_suffix(suffix)
            if candidate.is_file() and candidate != image:
                return candidate
        return None

    def on_save_mask_clicked(self) -> None:
        if self._prediction is None:
            self._echo("nothing to save yet")
            return
        path, _ = QFileDialog.getSaveFileName(
            self.context.qt_parent, "Save mask", str(Path.home() / "mask.png"), "PNG (*.png)")
        if path:
            data_io.save_mask(path, self._prediction >= 0.5)
            self._echo(f"saved {path}")


ADDON = FrangiAddon
