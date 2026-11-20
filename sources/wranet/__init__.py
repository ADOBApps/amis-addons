"""WRANet addon: model + panel, dropped in as a normal AMIS addon.

This module is the template for every future method: subclass
:class:`~amis.core.addon_api.AddonBase`, declare the model factory, build a
panel.  Nothing here is imported by the core, so the method can be deleted,
replaced or duplicated without touching the framework.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QProgressBar,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from amis.core.addon_api import AddonBase
from amis.ml import data as data_io
from amis.ml import metrics as metric_lib
from amis.ui.designer.render import DirField, FileField, ImageView, ScrollPage
from amis.ui.locale import c_locale

from .model import DEFAULT_CONFIG, build_model, count_parameters

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- panel
class WranetPanel(QWidget):
    """Five tabs: data, model, train, predict, export."""

    sig_log = Signal(str)
    sig_epoch = Signal(dict)
    sig_done = Signal(dict)
    sig_error = Signal(str)

    def __init__(self, addon: "WranetAddon") -> None:
        super().__init__()
        self.addon = addon
        self.context = addon.context
        self.trainer = None
        self._cases: list = []
        self._segmenter = None
        self._last_prediction: Optional[np.ndarray] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "Data")
        self.tabs.addTab(self._build_model_tab(), "Model")
        self.tabs.addTab(self._build_train_tab(), "Train")
        self.tabs.addTab(self._build_predict_tab(), "Predict")
        self.tabs.addTab(self._build_export_tab(), "Export")
        layout.addWidget(self.tabs)

        self.sig_log.connect(self._append_log)
        self.sig_epoch.connect(self._on_epoch)
        self.sig_done.connect(self._on_done)
        self.sig_error.connect(self._on_error)
        self._restore_settings()

    # ---------------------------------------------------------------- tabs
    def _build_data_tab(self) -> QWidget:
        page = ScrollPage()
        layout = page.body

        self.images_dir = DirField("Images", "")
        self.masks_dir = DirField("Masks", "")
        scan = QPushButton("Scan dataset")
        scan.clicked.connect(self.on_scan_clicked)

        self.dataset_label = QLabel("No dataset scanned")
        self.dataset_label.setWordWrap(True)
        self.case_list = QListWidget()

        layout.addWidget(self.images_dir)
        layout.addWidget(self.masks_dir)
        layout.addWidget(scan)
        layout.addWidget(self.dataset_label)
        layout.addWidget(self.case_list, 1)
        return page

    def _build_model_tab(self) -> QWidget:
        page = ScrollPage()
        layout = page.body
        box = QGroupBox("WRANet configuration")
        form = QFormLayout(box)

        self.base_spin = QSpinBox()
        self.base_spin.setRange(8, 128)
        self.base_spin.setSingleStep(8)
        self.base_spin.setValue(int(DEFAULT_CONFIG["base_channels"]))

        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(2, 5)
        self.depth_spin.setValue(int(DEFAULT_CONFIG["depth"]))

        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(["haar", "db2", "db4"])

        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 3)
        self.repeats_spin.setValue(int(DEFAULT_CONFIG.get("repeats", 2)))

        self.keep_high_check = QCheckBox("Propagate high-frequency sub-bands (ablation)")
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setLocale(c_locale())
        self.dropout_spin.setRange(0.0, 0.8)
        self.dropout_spin.setSingleStep(0.05)

        form.addRow("Base channels", self.base_spin)
        form.addRow("Depth (down/up stages)", self.depth_spin)
        form.addRow("Wavelet", self.wavelet_combo)
        form.addRow("Convs per block", self.repeats_spin)
        form.addRow("", self.keep_high_check)
        form.addRow("Dropout", self.dropout_spin)

        build = QPushButton("Instantiate and count parameters")
        build.clicked.connect(self.on_build_clicked)
        self.model_summary = QTextEdit()
        self.model_summary.setReadOnly(True)
        self.model_summary.setMinimumHeight(130)

        layout.addWidget(box)
        layout.addWidget(build)
        layout.addWidget(self.model_summary)
        layout.addStretch(1)
        return page

    def _build_train_tab(self) -> QWidget:
        page = ScrollPage()
        layout = page.body
        box = QGroupBox("Training")
        form = QFormLayout(box)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(20)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(2)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setLocale(c_locale())
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setSingleStep(1e-4)
        self.lr_spin.setValue(1e-3)

        self.size_spin = QSpinBox()
        self.size_spin.setRange(64, 1024)
        self.size_spin.setSingleStep(32)
        self.size_spin.setValue(256)

        self.val_spin = QDoubleSpinBox()
        self.val_spin.setLocale(c_locale())
        self.val_spin.setRange(0.0, 0.6)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.2)

        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])

        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["bce_dice", "dice"])

        for label, widget in (
            ("Epochs", self.epochs_spin),
            ("Batch size", self.batch_spin),
            ("Learning rate", self.lr_spin),
            ("Image size", self.size_spin),
            ("Validation fraction", self.val_spin),
            ("Device", self.device_combo),
            ("Loss", self.loss_combo),
        ):
            form.addRow(label, widget)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start training")
        self.start_button.clicked.connect(self.on_start_training)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.on_stop_training)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)

        self.train_log = QTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMinimumHeight(200)

        layout.addWidget(box)
        layout.addLayout(buttons)
        layout.addWidget(self.train_log, 1)
        return page

    def _build_predict_tab(self) -> QWidget:
        page = ScrollPage()
        layout = page.body

        self.predict_file = FileField("Image", "open", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        self.weights_field = FileField("Weights", "open", "PyTorch (*.pt *.pth)")
        # An exported ONNX model carries the trained network and runs without
        # torch weights; keep both paths so a shipped .onnx is usable as-is.
        self.onnx_field = FileField("ONNX model", "open", "ONNX (*.onnx)")
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setLocale(c_locale())
        self.threshold_spin.setRange(0.0, 1.0)
        self.threshold_spin.setSingleStep(0.05)
        self.threshold_spin.setValue(0.5)

        # A fully convolutional net *runs* at any size, but the model was
        # trained at ``img_size``.  Resizing the input to that size and
        # upsampling the probability back keeps inference in the regime the
        # weights were fitted for; without it the numbers silently degrade.
        self.resize_check = QCheckBox("Predict at the training size")
        self.resize_check.setChecked(True)
        self.resize_check.setToolTip(
            "Resize the slice to the image size used for training, predict, then "
            "upsample the probability map back to the original resolution.")

        row = QHBoxLayout()
        run = QPushButton("Segment")
        run.clicked.connect(self.on_predict_clicked)
        save = QPushButton("Save mask...")
        save.clicked.connect(self.on_save_mask_clicked)
        row.addWidget(run)
        row.addWidget(save)
        row.addWidget(QLabel("threshold"))
        row.addWidget(self.threshold_spin)
        row.addStretch(1)

        self.predict_progress = QProgressBar()
        self.predict_progress.setVisible(False)
        self.predict_status = QLabel("")
        self.predict_status.setWordWrap(True)

        self.preview = ImageView(300)
        self.metrics_label = QLabel("")
        self.metrics_label.setWordWrap(True)

        layout.addWidget(self.predict_file)
        layout.addWidget(self.weights_field)
        layout.addWidget(self.onnx_field)
        layout.addWidget(self.resize_check)
        layout.addLayout(row)
        layout.addWidget(self.predict_progress)
        layout.addWidget(self.predict_status)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.metrics_label)
        return page

    def _build_export_tab(self) -> QWidget:
        page = ScrollPage()
        layout = page.body

        self.onnx_output = FileField("ONNX output", "save", "ONNX (*.onnx)")
        export = QPushButton("Export ONNX")
        export.clicked.connect(self.on_export_clicked)
        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        self.export_log.setMinimumHeight(200)

        note = QLabel(
            "The exported file expects NCHW float32 input in [0, 1] and returns "
            "logits; apply sigmoid for probabilities. A JSON sidecar records the "
            "input contract and the export parity check."
        )
        note.setWordWrap(True)

        layout.addWidget(note)
        layout.addWidget(self.onnx_output)
        layout.addWidget(export)
        layout.addWidget(self.export_log, 1)
        return page

    # ------------------------------------------------------------- settings
    def _restore_settings(self) -> None:
        settings = self.context.settings
        if settings is None:
            return
        self.images_dir.set_path(settings.get("images_dir", "") or "")
        self.masks_dir.set_path(settings.get("masks_dir", "") or "")
        self.epochs_spin.setValue(int(settings.get("epochs", 20)))
        self.size_spin.setValue(int(settings.get("size", 256)))
        self.batch_spin.setValue(int(settings.get("batch", 2)))
        self.base_spin.setValue(int(settings.get("base_channels", 64)))

    def _persist_settings(self) -> None:
        settings = self.context.settings
        if settings is None:
            return
        settings.update(
            {
                "images_dir": self.images_dir.path(),
                "masks_dir": self.masks_dir.path(),
                "epochs": self.epochs_spin.value(),
                "size": self.size_spin.value(),
                "batch": self.batch_spin.value(),
                "base_channels": self.base_spin.value(),
            }
        )
        settings._parent.save()  # type: ignore[attr-defined]

    # ------------------------------------------------------------ model cfg
    def model_config(self) -> Dict[str, Any]:
        return {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": self.base_spin.value(),
            "depth": self.depth_spin.value(),
            "wavelet": self.wavelet_combo.currentText(),
            "keep_high": self.keep_high_check.isChecked(),
            "dropout": self.dropout_spin.value(),
            "repeats": self.repeats_spin.value(),
        }

    # ---------------------------------------------------------------- slots
    def on_scan_clicked(self) -> None:
        images = self.images_dir.path()
        masks = self.masks_dir.path()
        if not images:
            self.sig_log.emit("Pick an image folder first")
            return
        try:
            self._cases = data_io.discover_cases(images, masks or None)
        except Exception as exc:
            self.sig_log.emit(f"scan failed: {exc}")
            return
        self.case_list.clear()
        labelled = sum(1 for case in self._cases if case.labelled)
        self.dataset_label.setText(
            f"{len(self._cases)} images, {labelled} with masks "
            f"({Path(images).name})"
        )
        for case in self._cases[:400]:
            suffix = "" if case.labelled else "   (no mask)"
            self.case_list.addItem(f"{case.name}{suffix}")
        self._persist_settings()

    def on_build_clicked(self) -> None:
        try:
            model = build_model(self.model_config())
            params = count_parameters(model) / 1e6
            summary = (
                f"class      : {type(model).__name__}\n"
                f"parameters : {params:.2f} M\n"
                f"channels   : {[self.base_spin.value() * 2 ** i for i in range(self.depth_spin.value() + 1)]}\n"
                f"wavelet    : {self.wavelet_combo.currentText()} (fixed filters, 0 params)\n"
                f"keep_high  : {self.keep_high_check.isChecked()}\n"
                f"downsample : {2 ** self.depth_spin.value()}x\n"
                f"paper      : 34.6 M / 1.1 GFLOPs (WRANet, Zhao et al. 2023)"
            )
            self.model_summary.setPlainText(summary)
        except Exception as exc:
            self.model_summary.setPlainText(f"error: {exc}")

    def on_start_training(self) -> None:
        from amis.ml.trainer import Trainer, TrainingConfig

        if not self.images_dir.path() or not self.masks_dir.path():
            self.sig_log.emit("Set image and mask folders first")
            return
        config = TrainingConfig(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            lr=self.lr_spin.value(),
            size=self.size_spin.value(),
            val_fraction=self.val_spin.value(),
            loss=self.loss_combo.currentText(),
            device="" if self.device_combo.currentText() == "auto" else self.device_combo.currentText(),
        )
        out_dir = Path(self.context.data_dir) / "wranet"
        self.trainer = Trainer(
            build_model=build_model,
            model_config=self.model_config(),
            config=config,
            image_dir=self.images_dir.path(),
            mask_dir=self.masks_dir.path(),
            out_dir=out_dir,
            on_log=self.sig_log.emit,
            on_epoch=lambda report: self.sig_epoch.emit(report.to_dict()),
            on_done=self.sig_done.emit,
            on_error=self.sig_error.emit,
        )
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.sig_log.emit(f"output directory: {out_dir}")
        self.trainer.start()
        self._persist_settings()

    def on_stop_training(self) -> None:
        if self.trainer is not None:
            self.trainer.stop()
            self.sig_log.emit("stop requested")

    def on_predict_clicked(self) -> None:
        import torch  # noqa: F401 - fail early with a clear message

        image_path = self.predict_file.path()
        if not image_path:
            return

        self.predict_progress.setVisible(True)
        self.predict_progress.setRange(0, 0)      # indeterminate: one image
        self.predict_status.setText("segmenting…")
        self.tabs.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            weights = self.weights_field.path()
            if not weights:
                candidate = Path(self.context.data_dir) / "wranet" / "best.pt"
                if candidate.is_file():
                    weights = str(candidate)
                    self.weights_field.set_path(weights)
                    self.sig_log.emit(f"using {weights}")
            from amis.ml.backend import make_segmenter

            onnx_path = self.onnx_field.path()
            if not onnx_path:
                bundled = sorted((Path(self.context.addon_dir) / "models").glob("*.onnx"))
                if bundled and not weights:
                    onnx_path = str(bundled[0])
                    self.onnx_field.set_path(onnx_path)
                    self.sig_log.emit(f"using bundled model {onnx_path}")

            if onnx_path:
                segmenter = make_segmenter(
                    onnx_model=onnx_path, input_size=self.size_spin.value())
                self.sig_log.emit(f"inference backend: ONNX ({Path(onnx_path).name})")
            else:
                segmenter = make_segmenter(
                    build_model=build_model,
                    config=self.model_config(),
                    weights=weights or None,
                    input_size=self.size_spin.value(),
                )
            # An exported ONNX model may bake in its input size; that size wins
            # over the "Image size" spin, otherwise ONNX Runtime rejects the
            # tensor with "Got: 256 Expected: 192".
            model_size = 0
            if hasattr(segmenter, "recommended_size"):
                model_size = segmenter.recommended_size() or 0

            image = data_io.load_gray(image_path)
            native = image.shape
            training_size = int(self.size_spin.value())
            target = model_size or (training_size if self.resize_check.isChecked() else 0)
            if model_size and training_size and model_size != training_size:
                self.sig_log.emit(
                    f"note: this model expects {model_size} px, ignoring Image size "
                    f"{training_size}"
                )
            working = data_io.load_gray(image_path, target) if target > 0 else image

            probability = segmenter.predict(working)
            if working.shape != native:
                probability = self._to_shape(probability, native)

            self._last_prediction = probability
            self.preview.set_image(probability)

            mask_path = self._sibling_mask(image_path)
            if mask_path is not None:
                truth = data_io.load_mask(mask_path, native)
                values = metric_lib.all_metrics(self._last_prediction, truth)
                self.metrics_label.setText(
                    f"dice {values['dice']:.4f}  iou {values['iou']:.4f}  "
                    f"precision {values['precision']:.4f}  "
                    f"sensitivity {values['sensitivity']:.4f}"
                )
            else:
                self.metrics_label.setText("no ground-truth mask found next to the image")
        except Exception as exc:
            self.predict_status.setText(f"failed: {exc}")
            self.sig_log.emit(f"prediction failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
            self.tabs.setEnabled(True)
            self.predict_progress.setVisible(False)
            self.predict_progress.setRange(0, 100)
            if not self.predict_status.text().startswith("failed"):
                self.predict_status.setText(
                    f"{native[1]}x{native[0]} px | {self._backend_label(segmenter)}")

    @staticmethod
    def _backend_label(segmenter) -> str:
        name = type(segmenter).__name__
        if name == "OnnxSegmenter":
            return "ONNX Runtime"
        return "PyTorch"

    def on_save_mask_clicked(self) -> None:
        if self._last_prediction is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save mask", str(Path.home() / "mask.png"), "PNG (*.png)"
        )
        if path:
            data_io.save_mask(path, self._last_prediction >= self.threshold_spin.value())

    def on_export_clicked(self) -> None:
        output = self.onnx_output.path()
        if not output:
            output = str(Path(self.context.data_dir) / "wranet" / "wranet.onnx")
            self.onnx_output.set_path(output)
        weights = self.weights_field.path()
        if not weights:
            candidate = Path(self.context.data_dir) / "wranet" / "best.pt"
            if candidate.is_file():
                weights = str(candidate)
        try:
            from amis.ml.onnx_export import export_onnx

            metadata = export_onnx(
                build_model=build_model,
                model_config=self.model_config(),
                output_path=output,
                input_size=self.size_spin.value(),
                weights=weights or None,
            )
            self.export_log.setPlainText(
                "\n".join(f"{key:18s}: {value}" for key, value in metadata.items())
            )
        except Exception as exc:
            self.export_log.setPlainText(f"export failed: {exc}")

    # ------------------------------------------------------------ callbacks
    def _append_log(self, message: str) -> None:
        self.train_log.append(message)
        self.export_log.append(message)

    def _on_epoch(self, report: dict) -> None:
        # The trainer already emits a one-line summary through _append_log;
        # keep the hook so future versions can drive a progress bar.
        pass

    def _on_done(self, payload: dict) -> None:
        self.sig_log.emit(
            f"best dice {payload.get('best_dice', 0):.4f} "
            f"({payload.get('epochs_done', 0)} epochs, "
            f"{payload.get('seconds', 0):.0f}s)"
        )
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def _on_error(self, message: str) -> None:
        self.sig_log.emit(f"ERROR {message}")
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _to_shape(probability, shape):
        """Bilinear resize of a float probability map (never nearest)."""
        from PIL import Image

        scaled = np.clip(np.asarray(probability, dtype=np.float32), 0.0, 1.0) * 65535.0
        image = Image.fromarray(scaled.astype(np.uint16))
        image = image.resize((shape[1], shape[0]), Image.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 65535.0

    def _sibling_mask(self, image_path: str) -> Optional[Path]:
        """Ground truth: the Masks folder first, then a sibling with another suffix."""
        image = Path(image_path)
        masks_dir = self.masks_dir.path()
        if masks_dir:
            for suffix in (".png", ".tif", ".tiff", ".bmp"):
                candidate = Path(masks_dir) / f"{image.stem}{suffix}"
                if candidate.is_file():
                    return candidate
        for suffix in (".png", ".tif", ".tiff", ".bmp"):
            candidate = image.with_suffix(suffix)
            if candidate.is_file() and candidate != image:
                return candidate
        return None


# --------------------------------------------------------------------- addon
class WranetAddon(AddonBase):
    """Addon entry point."""

    def initialize(self) -> None:
        self.ui = None

    def build_panel(self) -> QWidget:
        return WranetPanel(self)

    def on_activated(self) -> None:
        self.context.logger.debug("WRANet panel activated")


ADDON = WranetAddon
