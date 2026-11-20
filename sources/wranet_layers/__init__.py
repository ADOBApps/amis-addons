"""WRANet Layers - retinal layer and fluid segmentation as one multi-class task.

Entry point for the addon.  The interface is ``panel.json``; this module wires
the shared training loop, metrics and ONNX export to the multi-class model in
``model.py``.

Reference for the backbone: Zhao et al., *WRANet: wavelet integrated residual
attention U-Net network for medical image segmentation*, Complex & Intelligent
Systems 9:6971-6983 (2023), doi:10.1007/s40747-023-01119-y.
"""

from __future__ import annotations

import json
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

#: Fixed colour per class, in the order of OCTAVE's label map.  Chosen to stay
#: distinguishable on a grayscale background and stable across runs.
CLASS_COLOURS = np.array([
    (0, 0, 0),        # background
    (220, 60, 60),    # SRM
    (240, 150, 40),   # HRM
    (40, 200, 120),   # FLU
    (200, 200, 60),   # HTD
    (60, 120, 220),   # RPE
    (150, 90, 220),   # RET
    (220, 90, 180),   # CHO
    (90, 200, 220),   # VIT
    (160, 160, 160),  # HYA
    (120, 200, 90),   # RHS
    (200, 120, 60),   # ART
    (90, 90, 200),    # ERM
    (235, 235, 235),  # SES
], dtype=np.uint8)


class _Bridge(QObject):
    log = Signal(str)
    epoch = Signal(dict)
    done = Signal(dict)
    error = Signal(str)


class WranetLayersAddon(AddonBase):
    def initialize(self) -> None:
        self._bridge: Optional[_Bridge] = None
        self._trainer = None
        self._prediction: Optional[np.ndarray] = None
        self.ui = None

    def build_panel(self):
        panel = super().build_panel()          # renders panel.json
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

    def num_classes(self) -> int:
        """Class count from the dataset when known, otherwise from the panel."""
        masks = self.ui.get("masks_dir", "") if self.ui is not None else ""
        if masks:
            classes_file = Path(masks).parent / "classes.json"
            if classes_file.is_file():
                try:
                    reported = int(json.loads(classes_file.read_text())["num_classes"])
                    if reported > 1:
                        return reported
                except Exception:
                    pass
        return int(self.ui.get("num_classes", 14)) if self.ui is not None else 14

    def class_names(self) -> list:
        masks = self.ui.get("masks_dir", "") if self.ui is not None else ""
        if masks:
            classes_file = Path(masks).parent / "classes.json"
            if classes_file.is_file():
                try:
                    names = json.loads(classes_file.read_text()).get("names") or []
                    if names:
                        return names
                except Exception:
                    pass
        return list(self.manifest.config.get("classes", []))

    def model_config(self) -> dict:
        return {
            "in_channels": 1,
            "out_channels": self.num_classes(),
            "num_classes": self.num_classes(),
            "base_channels": int(self.ui.get("base_channels", 16)) if self.ui else 16,
            "depth": int(self.ui.get("depth", 4)) if self.ui else 4,
            "wavelet": str(self.ui.get("wavelet", "haar")) if self.ui else "haar",
            "keep_high": bool(self.ui.get("keep_high", False)) if self.ui else False,
            "dropout": float(self.ui.get("dropout", 0.0)) if self.ui else 0.0,
        }

    # ---------------------------------------------------------------- data
    def on_scan_clicked(self) -> None:
        images = self.ui.get("images_dir", "")
        masks = self.ui.get("masks_dir", "")
        if not images or not masks:
            self._echo("set the image and mask folders first")
            return
        try:
            cases = data_io.discover_cases(images, masks)
        except Exception as exc:
            self._echo(f"scan failed: {exc}")
            return
        names = self.class_names()
        self.ui.set("dataset_info",
                    f"{len(cases)} slices | {self.num_classes()} classes "
                    f"({', '.join(names[:5])}{'…' if len(names) > 5 else ''})")
        self._echo(f"dataset ok: {len(cases)} image/mask pairs")
        self._persist()

    # --------------------------------------------------------------- model
    def on_inspect_clicked(self) -> None:
        try:
            config = self.model_config()
            model = build_model(config)
            self._echo(
                f"{type(model).__name__}: {count_parameters(model)/1e6:.2f} M parameters, "
                f"{config['out_channels']} classes, base {config['base_channels']}, "
                f"depth {config['depth']}")
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
            self._echo("set both folders first")
            return

        config = TrainingConfig(
            epochs=int(self.ui.get("epochs", 20)),
            batch_size=int(self.ui.get("batch", 2)),
            lr=float(self.ui.get("lr", 1e-3)),
            size=int(self.ui.get("img_size", 256)),
            val_fraction=float(self.ui.get("val_fraction", 0.2)),
            device="" if str(self.ui.get("device", "auto")) == "auto"
            else str(self.ui.get("device")),
            num_classes=self.num_classes(),
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
            per_class = report.get("val_dice_classes") or {}
            detail = "  ".join(
                f"{self._class_label(key)} {value:.3f}"
                for key, value in sorted(per_class.items())
                if key.startswith("class_") and value > 0)
            if detail:
                self.ui.set("per_class", detail)

    def _class_label(self, key: str) -> str:
        try:
            index = int(key.split("_")[1])
        except (IndexError, ValueError):
            return key
        names = self.class_names()
        return names[index] if 0 <= index < len(names) else key

    def _on_done(self, payload: dict) -> None:
        if self.ui is not None:
            self.ui.widgets["start"].setEnabled(True)
            self.ui.widgets["stop"].setEnabled(False)
            self.ui.set("progress", 100)
        self._echo(f"finished: best mean class Dice "
                   f"{payload.get('best_dice', 0):.4f} after "
                   f"{payload.get('epochs_done', 0)} epochs")

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
        try:
            import torch
            from amis.ml.backend import make_segmenter

            weights = self.ui.get("weights_path", "")
            if not weights:
                candidate = Path(self.context.data_dir) / "best.pt"
                if candidate.is_file():
                    weights = str(candidate)
                    self.ui.set("weights_path", weights)
            segmenter = make_segmenter(
                build_model=build_model,
                config=self.model_config(),
                weights=weights or None,
                input_size=int(self.ui.get("img_size", 256)),
            )
            # segmentation must run at the training size, then be mapped back
            size = int(self.ui.get("img_size", 256))
            image = data_io.load_gray(path)
            working = data_io.load_gray(path, size)

            model = segmenter.model
            padded_input = torch.from_numpy(working)[None, None]
            with torch.no_grad():
                logits = model(padded_input.to(segmenter.device))
                classes = torch.argmax(logits, dim=1)[0].cpu().numpy()

            if classes.shape != image.shape:
                from PIL import Image

                classes = np.asarray(
                    Image.fromarray(classes.astype(np.uint8)).resize(
                        (image.shape[1], image.shape[0]), Image.NEAREST),
                    dtype=np.int64)

            self._prediction = classes
            preview = self.ui.values.get("preview")
            if preview is not None and hasattr(preview, "set_image"):
                rgb = CLASS_COLOURS[np.clip(classes, 0, len(CLASS_COLOURS) - 1)]
                preview.set_image(rgb)

            truth_path = self._truth_for(path)
            if truth_path is None:
                self.ui.set("metrics", "no ground-truth mask for this image")
                return
            truth = data_io.load_classes(truth_path, image.shape)
            scores = metric_lib.dice_per_class(classes, truth, self.num_classes())
            detail = "  ".join(
                f"{self._class_label(key)} {value:.3f}"
                for key, value in sorted(scores.items())
                if key.startswith("class_"))
            self.ui.set("per_class", detail)
            self.ui.set("metrics",
                        f"mean class Dice {scores['mean']:.4f} over "
                        f"{int(scores['classes_present'])} classes")
        except Exception as exc:
            self._echo(f"prediction failed: {exc}")

    def _truth_for(self, image_path: str) -> Optional[Path]:
        image = Path(image_path)
        masks = self.ui.get("masks_dir", "")
        if masks:
            for suffix in (".png", ".tif", ".tiff"):
                candidate = Path(masks) / f"{image.stem}{suffix}"
                if candidate.is_file():
                    return candidate
        return None

    def on_save_mask_clicked(self) -> None:
        if self._prediction is None:
            self._echo("nothing to save yet")
            return
        path, _ = QFileDialog.getSaveFileName(
            self.context.qt_parent, "Save class map",
            str(Path.home() / "layers.png"), "PNG (*.png)")
        if path:
            from PIL import Image

            Image.fromarray(self._prediction.astype(np.uint8)).save(path)
            self._echo(f"saved class map (values = class index) to {path}")

    # -------------------------------------------------------------- export
    def on_export_clicked(self) -> None:
        from amis.ml.onnx_export import export_onnx

        output = self.ui.get("onnx_path", "") or str(
            Path(self.context.data_dir) / "wranet_layers.onnx")
        self.ui.set("onnx_path", output)
        try:
            metadata = export_onnx(
                build_model=build_model,
                model_config=self.model_config(),
                output_path=output,
                input_size=int(self.ui.get("img_size", 256)),
                weights=self.ui.get("weights_path", "") or None,
            )
            self._echo(
                f"exported {output} with {metadata.get('model_config', {}).get('out_channels')} "
                f"channels (verified={metadata.get('verified')}). "
                f"Apply softmax over the channel axis, then argmax.")
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


ADDON = WranetLayersAddon
