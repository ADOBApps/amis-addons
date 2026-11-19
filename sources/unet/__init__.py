"""U-Net.

Ronneberger et al. (2015) U-Net: symmetric encoder-decoder with concatenated skip connections. The reference architecture every other method here is compared against.

Generated for the AMIS addon catalogue.  The interface is ``panel.json``; this
module only wires the shared plumbing (data, trainer, metrics, ONNX export) to
the method in ``model.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QFileDialog

from amis.core.addon_api import AddonBase
from amis.ml import data as data_io
from amis.ml import metrics as metric_lib

from .model import DEFAULT_CONFIG, build_model, count_parameters

log = logging.getLogger(__name__)


class _Bridge(QObject):
    log = Signal(str)
    epoch = Signal(dict)
    done = Signal(dict)
    error = Signal(str)


class UnetAddon(AddonBase):
    def initialize(self) -> None:
        self._bridge: Optional[_Bridge] = None
        self._trainer = None
        self._cases: list = []
        self._prediction: Optional[np.ndarray] = None
        self._segmenter = None
        self.ui = None

    def build_panel(self):
        panel = super().build_panel()
        self._bridge = _Bridge()
        self._bridge.log.connect(self._append_log)
        self._bridge.epoch.connect(self._on_epoch)
        self._bridge.done.connect(self._on_done)
        self._bridge.error.connect(self._on_error)
        self._restore()
        return panel

    def shutdown(self) -> None:
        if self._trainer is not None:
            self._trainer.stop()
            self._trainer = None

    # ------------------------------------------------------------- helpers
    def _echo(self, message: str) -> None:
        self._append_log(message)
        self.context.logger.info(message)

    def _append_log(self, message: str) -> None:
        box = self.ui.values.get("log") if self.ui is not None else None
        if box is not None:
            box.append(message)

    def model_config(self) -> dict:
        return {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": int(self.ui.get("base_channels", 32)) if self.ui else 32,
            "depth": int(self.ui.get("depth", 4)) if self.ui else 4,
            "dropout": float(self.ui.get("dropout", 0.0)) if self.ui else 0.0,
        }

    # ---------------------------------------------------------------- data
    def on_scan_clicked(self) -> None:
        images = self.ui.get("images_dir", "")
        masks = self.ui.get("masks_dir", "")
        if not images:
            self._echo("Pick an image folder first")
            return
        try:
            self._cases = data_io.discover_cases(images, masks or None)
        except Exception as exc:
            self._echo(f"scan failed: {exc}")
            return
        labelled = sum(1 for case in self._cases if case.labelled)
        self.ui.set("dataset_info",
                    f"{len(self._cases)} images, {labelled} with masks "
                    f"({Path(images).name})")
        self._persist()

    # --------------------------------------------------------------- model
    def on_inspect_clicked(self) -> None:
        try:
            config = self.model_config()
            model = build_model(config)
            self._echo(f"{type(model).__name__}: "
                       f"{count_parameters(model) / 1e6:.2f} M parameters "
                       f"(base={config['base_channels']}, depth={config['depth']})")
        except Exception as exc:
            self._echo(f"model error: {exc}")

    # ------------------------------------------------------------ training
    def on_start_clicked(self) -> None:
        from amis.ml.trainer import Trainer, TrainingConfig

        if self._trainer is not None and self._trainer.running:
            self._echo("training already running")
            return
        images, masks = self.ui.get("images_dir", ""), self.ui.get("masks_dir", "")
        if not images or not masks:
            self._echo("set image and mask folders first")
            return
        config = TrainingConfig(
            epochs=int(self.ui.get("epochs", 20)),
            batch_size=int(self.ui.get("batch", 2)),
            lr=float(self.ui.get("lr", 1e-3)),
            size=int(self.ui.get("img_size", 256)),
            val_fraction=float(self.ui.get("val_fraction", 0.2)),
            loss=str(self.ui.get("loss", "bce_dice")),
            device="" if str(self.ui.get("device", "auto")) == "auto"
            else str(self.ui.get("device")),
        )
        out_dir = Path(self.context.data_dir)
        self._trainer = Trainer(
            build_model=build_model,
            model_config=self.model_config(),
            config=config,
            image_dir=images,
            mask_dir=masks,
            out_dir=out_dir,
            on_log=self._bridge.log.emit,
            on_epoch=lambda report: self._bridge.epoch.emit(report.to_dict()),
            on_done=self._bridge.done.emit,
            on_error=self._bridge.error.emit,
        )
        self.ui.widgets["start"].setEnabled(False)
        self.ui.widgets["stop"].setEnabled(True)
        self.ui.set("progress", 0)
        self._echo(f"output: {out_dir}")
        self._trainer.start()
        self._persist()

    def on_stop_clicked(self) -> None:
        if self._trainer is not None:
            self._trainer.stop()
            self._echo("stop requested")

    def _on_epoch(self, report: dict) -> None:
        total = max(1, int(report.get("total", 1)))
        if self.ui is not None:
            self.ui.set("progress", int(100 * report.get("epoch", 0) / total))

    def _on_done(self, payload: dict) -> None:
        if self.ui is not None:
            self.ui.widgets["start"].setEnabled(True)
            self.ui.widgets["stop"].setEnabled(False)
            self.ui.set("progress", 100)
        self._echo(f"finished: best dice {payload.get('best_dice', 0):.4f} "
                   f"in {payload.get('epochs_done', 0)} epochs")

    def _on_error(self, message: str) -> None:
        if self.ui is not None:
            self.ui.widgets["start"].setEnabled(True)
            self.ui.widgets["stop"].setEnabled(False)
        self._echo(f"ERROR {message}")

    # ------------------------------------------------------------- predict
    def on_predict_clicked(self) -> None:
        path = self.ui.get("image_path", "")
        if not path:
            self._echo("pick an image first")
            return
        weights = self.ui.get("weights_path", "")
        if not weights:
            candidate = Path(self.context.data_dir) / "best.pt"
            if candidate.is_file():
                weights = str(candidate)
                self.ui.set("weights_path", weights)
                self._echo(f"using {weights}")
        try:
            from amis.ml.backend import make_segmenter

            if self._segmenter is None:
                self._segmenter = make_segmenter(
                    build_model=build_model,
                    config=self.model_config(),
                    weights=weights or None,
                    input_size=int(self.ui.get("img_size", 256)),
                )
            image = data_io.load_gray(path)
            self._prediction = self._segmenter.predict(image)
            preview = self.ui.values.get("preview")
            if preview is not None and hasattr(preview, "set_image"):
                preview.set_image(self._prediction)
            truth_path = self._truth_for(path)
            if truth_path is not None:
                truth = data_io.load_mask(truth_path, image.shape)
                values = metric_lib.all_metrics(
                    self._prediction, truth, float(self.ui.get("threshold", 0.5)))
                self.ui.set("metrics",
                            f"dice {values['dice']:.4f}  iou {values['iou']:.4f}  "
                            f"precision {values['precision']:.4f}  "
                            f"sensitivity {values['sensitivity']:.4f}")
            else:
                self.ui.set("metrics", "no ground-truth mask found next to the image")
        except Exception as exc:
            self._echo(f"prediction failed: {exc}")

    def _truth_for(self, image_path: str) -> Optional[Path]:
        stem = Path(image_path).stem
        masks_dir = self.ui.get("masks_dir", "") if self.ui else ""
        if masks_dir:
            for suffix in (".png", ".tif", ".tiff", ".bmp"):
                candidate = Path(masks_dir) / f"{stem}{suffix}"
                if candidate.is_file():
                    return candidate
        image = Path(image_path)
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
            data_io.save_mask(path, self._prediction >= float(self.ui.get("threshold", 0.5)))
            self._echo(f"saved {path}")

    # -------------------------------------------------------------- export
    def on_export_clicked(self) -> None:
        from amis.ml.onnx_export import export_onnx

        output = self.ui.get("onnx_path", "")
        if not output:
            output = str(Path(self.context.data_dir) / "unet.onnx")
            self.ui.set("onnx_path", output)
        try:
            metadata = export_onnx(
                build_model=build_model,
                model_config=self.model_config(),
                output_path=output,
                input_size=int(self.ui.get("img_size", 256)),
                weights=self.ui.get("weights_path", "") or None,
            )
            self._echo(f"exported {output} (verified={metadata.get('verified')}, "
                       f"max|diff|={metadata.get('parity_max_abs_diff')})")
        except Exception as exc:
            self._echo(f"export failed: {exc}")

    # ------------------------------------------------------------ settings
    def _restore(self) -> None:
        settings = self.context.settings
        if settings is None or self.ui is None:
            return
        for key in ("images_dir", "masks_dir", "image_path", "weights_path"):
            value = settings.get(key, "")
            if value:
                self.ui.set(key, value)

    def _persist(self) -> None:
        settings = self.context.settings
        if settings is None or self.ui is None:
            return
        settings.update({
            "images_dir": self.ui.get("images_dir", ""),
            "masks_dir": self.ui.get("masks_dir", ""),
            "image_path": self.ui.get("image_path", ""),
            "weights_path": self.ui.get("weights_path", ""),
        })
        settings._parent.save()  # type: ignore[attr-defined]


ADDON = UnetAddon
