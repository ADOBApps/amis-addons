"""refWRANet addon: multi-class retinal layer segmentation + a 2-D DWT explorer.

Two tabs:

* **Layers** - the method: data, model (wavelet family, width, depth), training
  with per-class Dice, prediction with a class colour map, ONNX export;
* **2-D DWT** - the teaching tab: pick an image and a wavelet family, press
  *Decompose*, and see the four sub-bands (LL, LH, HL, HH) with what each one
  contains and how much energy it carries, plus a one-line description of the
  family (Haar, Daubechies, Symlets, biorthogonal).

The explorer is deliberately independent of the network: it is the same
transform, applied to whatever image you load, so a figure for the paper can be
produced from the same tool that trains the model.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
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
from amis.ml.trainer import Trainer, TrainingConfig
from amis.ui.designer.render import DirField, FileField, ImageView

try:                                  # AMIS >= 0.1.5 ships ScrollPage
    from amis.ui.designer.render import ScrollPage
except ImportError:                   # 0.1.4 and older: same behaviour, local
    from PySide6.QtWidgets import QFrame, QScrollArea

    class ScrollPage(QWidget):        # noqa: F811 - deliberate shadowing
        """A page whose body scrolls (fallback for cores without ScrollPage)."""

        def __init__(self, parent=None) -> None:
            super().__init__(parent)
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            self.scroll = QScrollArea(self)
            self.scroll.setWidgetResizable(True)
            self.scroll.setFrameShape(QFrame.Shape.NoFrame)
            self.scroll.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            content = QWidget()
            self.body = QVBoxLayout(content)
            self.body.setContentsMargins(6, 6, 6, 6)
            self.body.setSpacing(6)
            self.scroll.setWidget(content)
            outer.addWidget(self.scroll)

from amis.ui.locale import c_locale

from . import wavelib
from .model import DEFAULT_CONFIG, build_model, count_parameters

log = logging.getLogger(__name__)

PALETTE = np.array([
    [0, 0, 0], [220, 60, 60], [240, 150, 40], [40, 200, 120], [200, 200, 60],
    [60, 120, 220], [150, 90, 220], [220, 90, 180], [90, 200, 220],
    [160, 160, 160], [120, 200, 90], [200, 120, 60], [90, 90, 200],
    [235, 235, 235],
], dtype=np.uint8)


class _Bridge(QObject):
    log = Signal(str)
    epoch = Signal(dict)
    done = Signal(dict)
    error = Signal(str)


class RefWranetPanel(QWidget):
    """The addon's window: Layer segmentation on one tab, DWT on the other."""

    sig_log = Signal(str)
    sig_epoch = Signal(dict)
    sig_done = Signal(dict)
    sig_error = Signal(str)

    def __init__(self, addon: "RefWranetAddon") -> None:
        super().__init__()
        self.addon = addon
        self.context = addon.context
        self._trainer = None
        self._prediction: Optional[np.ndarray] = None
        self._dwt_steps = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_layers_tab(), "Layers")
        self.tabs.addTab(self._build_dwt_tab(), "2-D DWT")
        layout.addWidget(self.tabs)

        self.sig_log.connect(self._append_log)
        self.sig_epoch.connect(self._on_epoch)
        self.sig_done.connect(self._on_done)
        self.sig_error.connect(self._on_error)

    # ==================================================== layers tab ====
    def _build_layers_tab(self) -> QWidget:
        # a scroll page: the tab has ~1000 px of controls, the window may not
        page = ScrollPage()
        outer = page.body

        intro = QLabel(
            "Multi-class retinal segmentation with the WRANet backbone: one output "
            "channel per class, wavelet downsampling so thin layer boundaries survive. "
            "Masks are class-index PNGs plus classes.json (see the other tab for the "
            "transform itself).")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        data_box = QGroupBox("Data")
        data_layout = QVBoxLayout(data_box)
        self.images_dir = DirField("Images", "")
        self.masks_dir = DirField("Masks (class index)", "")
        scan = QPushButton("Scan dataset")
        scan.clicked.connect(self.on_scan_clicked)
        self.dataset_info = QLabel("No dataset scanned")
        self.dataset_info.setWordWrap(True)
        data_layout.addWidget(self.images_dir)
        data_layout.addWidget(self.masks_dir)
        data_layout.addWidget(scan)
        data_layout.addWidget(self.dataset_info)
        outer.addWidget(data_box)

        model_box = QGroupBox("Model")
        model_form = QFormLayout(model_box)
        self.classes_spin = QSpinBox()
        self.classes_spin.setRange(2, 32)
        self.classes_spin.setValue(int(DEFAULT_CONFIG["num_classes"]))
        self.base_spin = QSpinBox()
        self.base_spin.setRange(8, 64)
        self.base_spin.setSingleStep(8)
        self.base_spin.setValue(int(DEFAULT_CONFIG["base_channels"]))
        self.depth_spin = QSpinBox()
        self.depth_spin.setRange(2, 5)
        self.depth_spin.setValue(int(DEFAULT_CONFIG["depth"]))
        self.wavelet_combo = QComboBox()
        self.wavelet_combo.addItems(list(wavelib.MODEL_WAVELETS))
        inspect = QPushButton("Count parameters")
        inspect.clicked.connect(self.on_inspect_clicked)
        model_form.addRow("Classes", self.classes_spin)
        model_form.addRow("Base channels", self.base_spin)
        model_form.addRow("Depth", self.depth_spin)
        model_form.addRow("Wavelet", self.wavelet_combo)
        model_form.addRow(inspect)
        outer.addWidget(model_box)

        train_box = QGroupBox("Training")
        train_form = QFormLayout(train_box)
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
        self.lr_spin.setValue(1e-3)
        self.size_spin = QSpinBox()
        self.size_spin.setRange(64, 1024)
        self.size_spin.setSingleStep(32)
        self.size_spin.setValue(int(DEFAULT_CONFIG.get("img_size", 192)))
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setLocale(c_locale())
        self.val_spin.setRange(0.0, 0.6)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.2)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cpu", "cuda", "mps"])
        train_form.addRow("Epochs", self.epochs_spin)
        train_form.addRow("Batch size", self.batch_spin)
        train_form.addRow("Learning rate", self.lr_spin)
        train_form.addRow("Image size", self.size_spin)
        train_form.addRow("Validation fraction", self.val_spin)
        train_form.addRow("Device", self.device_combo)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start training")
        self.start_button.clicked.connect(self.on_start_clicked)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        outer.addWidget(train_box)
        outer.addLayout(buttons)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        outer.addWidget(self.progress)
        self.per_class = QLabel("")
        self.per_class.setWordWrap(True)
        outer.addWidget(self.per_class)

        predict_box = QGroupBox("Predict")
        predict_layout = QVBoxLayout(predict_box)
        self.image_path = FileField("Image", "open",
                                    "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        self.weights_path = FileField("Weights", "open", "PyTorch (*.pt *.pth)")
        self.onnx_path = FileField("ONNX model", "open", "ONNX (*.onnx)")
        row = QHBoxLayout()
        run = QPushButton("Segment layers")
        run.clicked.connect(self.on_predict_clicked)
        save = QPushButton("Save class map...")
        save.clicked.connect(self.on_save_clicked)
        row.addWidget(run)
        row.addWidget(save)
        row.addStretch(1)
        self.preview = ImageView(240)
        self.metrics = QLabel("")
        self.metrics.setWordWrap(True)
        predict_layout.addWidget(self.image_path)
        predict_layout.addWidget(self.weights_path)
        predict_layout.addWidget(self.onnx_path)
        predict_layout.addLayout(row)
        predict_layout.addWidget(self.preview)
        predict_layout.addWidget(self.metrics)
        outer.addWidget(predict_box)

        export_box = QGroupBox("Export")
        export_layout = QHBoxLayout(export_box)
        export = QPushButton("Export ONNX")
        export.clicked.connect(self.on_export_clicked)
        export_layout.addWidget(export)
        export_layout.addStretch(1)
        outer.addWidget(export_box)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(110)
        outer.addWidget(self.log)
        return page

    # ======================================================= DWT tab ====
    def _build_dwt_tab(self) -> QWidget:
        page = ScrollPage()
        outer = page.body

        intro = QLabel(
            "A 2-D discrete wavelet transform replaces pooling. The wavelet functions "
            "become digital filters: a low-pass filter l smooths and keeps the main "
            "structure (LL), a high-pass filter h keeps the fast variations, giving "
            "horizontal (LH), vertical (HL) and diagonal (HH) details. Pick a family "
            "below to see how it changes what survives.")
        intro.setWordWrap(True)
        outer.addWidget(intro)

        controls = QGroupBox("Decomposition")
        form = QFormLayout(controls)
        self.dwt_image = FileField("Image", "open",
                                   "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        self.dwt_wavelet = QComboBox()
        self.dwt_wavelet.addItems(list(wavelib.ALL_WAVELETS))
        self.dwt_level = QSpinBox()
        self.dwt_level.setRange(1, 3)
        self.dwt_level.setValue(1)
        run = QPushButton("Decompose step by step")
        run.clicked.connect(self.on_decompose_clicked)
        form.addRow(self.dwt_image)
        form.addRow("Wavelet family", self.dwt_wavelet)
        form.addRow("Levels (LL again)", self.dwt_level)
        form.addRow(run)
        outer.addWidget(controls)

        grid = QGridLayout()
        self.band_views = {}
        for position, band in enumerate(("LL", "LH", "HL", "HH")):
            box = QGroupBox(band)
            box_layout = QVBoxLayout(box)
            caption = QLabel(wavelib.describe_band(band))
            caption.setWordWrap(True)
            view = ImageView(140)
            box_layout.addWidget(view)
            box_layout.addWidget(caption)
            grid.addWidget(box, position // 2, position % 2)
            self.band_views[band] = view
        outer.addLayout(grid)

        self.dwt_text = QTextEdit()
        self.dwt_text.setReadOnly(True)
        self.dwt_text.setMinimumHeight(110)
        outer.addWidget(self.dwt_text)
        return page

    # =========================================================== helpers
    def _append_log(self, message: str) -> None:
        self.log.append(message)
        self.context.logger.info(message)

    def _echo(self, message: str) -> None:
        self.sig_log.emit(message)

    def num_classes(self) -> int:
        masks = self.masks_dir.path()
        if masks:
            import json

            classes_file = Path(masks).parent / "classes.json"
            if classes_file.is_file():
                try:
                    return max(2, int(json.loads(classes_file.read_text())["num_classes"]))
                except Exception:
                    pass
        return int(self.classes_spin.value())

    def class_names(self) -> list:
        masks = self.masks_dir.path()
        if masks:
            import json

            classes_file = Path(masks).parent / "classes.json"
            if classes_file.is_file():
                try:
                    names = json.loads(classes_file.read_text()).get("names") or []
                    if names:
                        return names
                except Exception:
                    pass
        return list(self.addon.manifest.config.get("classes", []))

    def model_config(self) -> dict:
        return {
            "in_channels": 1,
            "out_channels": self.num_classes(),
            "num_classes": self.num_classes(),
            "base_channels": self.base_spin.value(),
            "depth": self.depth_spin.value(),
            "wavelet": self.wavelet_combo.currentText(),
        }

    # ============================================================== data
    def on_scan_clicked(self) -> None:
        images, masks = self.images_dir.path(), self.masks_dir.path()
        if not images or not masks:
            self._echo("set both folders first")
            return
        try:
            cases = data_io.discover_cases(images, masks)
        except Exception as exc:
            self._echo(f"scan failed: {exc}")
            return
        names = self.class_names()
        self.dataset_info.setText(
            f"{len(cases)} slices | {self.num_classes()} classes "
            f"({', '.join(names[:4])}{'…' if len(names) > 4 else ''})")
        self._echo(f"dataset: {len(cases)} pairs, {self.num_classes()} classes")

    def on_inspect_clicked(self) -> None:
        try:
            model = build_model(self.model_config())
            self._echo(f"{type(model).__name__}: {count_parameters(model)/1e6:.2f} M "
                       f"parameters, {self.num_classes()} classes, "
                       f"wavelet {self.wavelet_combo.currentText()}")
        except Exception as exc:
            self._echo(f"model error: {exc}")

    # ========================================================== training
    def on_start_clicked(self) -> None:
        if self._trainer is not None and self._trainer.running:
            self._echo("training already running")
            return
        if not self.images_dir.path() or not self.masks_dir.path():
            self._echo("set both folders first")
            return
        config = TrainingConfig(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            lr=self.lr_spin.value(),
            size=self.size_spin.value(),
            val_fraction=self.val_spin.value(),
            device="" if self.device_combo.currentText() == "auto"
            else self.device_combo.currentText(),
            num_classes=self.num_classes(),
        )
        self._trainer = Trainer(
            build_model=build_model,
            model_config=self.model_config(),
            config=config,
            image_dir=self.images_dir.path(),
            mask_dir=self.masks_dir.path(),
            out_dir=Path(self.context.data_dir),
            on_log=self.sig_log.emit,
            on_epoch=lambda report: self.sig_epoch.emit(report.to_dict()),
            on_done=self.sig_done.emit,
            on_error=self.sig_error.emit,
        )
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress.setValue(0)
        self._echo(f"output: {self.context.data_dir}")
        self._trainer.start()

    def on_stop_clicked(self) -> None:
        if self._trainer is not None:
            self._trainer.stop()
            self._echo("stop requested")

    def _on_epoch(self, report: dict) -> None:
        total = max(1, int(report.get("total", 1)))
        self.progress.setValue(int(100 * report.get("epoch", 0) / total))
        per_class = report.get("val_dice_classes") or {}
        names = self.class_names()
        parts = []
        for key, value in sorted(per_class.items()):
            if not key.startswith("class_"):
                continue
            try:
                index = int(key.split("_")[1])
            except ValueError:
                continue
            label = names[index] if 0 <= index < len(names) else key
            if value > 0:
                parts.append(f"{label} {value:.3f}")
        if parts:
            self.per_class.setText("  ".join(parts))

    def _on_done(self, payload: dict) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress.setValue(100)
        self._echo(f"finished: best mean class Dice {payload.get('best_dice', 0):.4f} "
                   f"in {payload.get('epochs_done', 0)} epochs")

    def _on_error(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self._echo(f"ERROR {message}")

    # =========================================================== predict
    def on_predict_clicked(self) -> None:
        path = self.image_path.path()
        if not path:
            self._echo("pick an image first")
            return
        self.progress.setRange(0, 0)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            import torch
            from amis.ml.backend import OnnxSegmenter, make_segmenter

            onnx = self.onnx_path.path()
            weights = self.weights_path.path()
            if not onnx:
                bundled = sorted((Path(self.context.addon_dir) / "models").glob("*.onnx"))
                if bundled and not weights:
                    onnx = str(bundled[0])
                    self.onnx_path.set_path(onnx)

            image = data_io.load_gray(path)
            size = self.size_spin.value()

            if onnx:
                session = OnnxSegmenter(onnx)
                # An exported model declares the size it was traced at; the panel
                # cannot override it. Feeding the spin value gave
                # "Got 192 Expected 160" from ONNX Runtime.
                model_size = session.recommended_size()
                if model_size and model_size != size:
                    self._echo(f"ONNX model is fixed at {model_size} px; "
                               f"using it instead of {size} (the panel is updated)")
                    self.size_spin.setValue(model_size)
                    size = model_size
                working = data_io.load_gray(path, size)
                logits = session.session.run(
                    None, {session.input_name: working[None, None].astype(np.float32)})[0]
                classes = np.argmax(logits[0], axis=0).astype(np.int64)
                backend = f"ONNX Runtime [{size} px]"
            else:
                working = data_io.load_gray(path, size)
                segmenter = make_segmenter(build_model=build_model,
                                           config=self.model_config(),
                                           weights=weights or None)
                with torch.no_grad():
                    logits = segmenter.model(torch.from_numpy(working)[None, None])
                    classes = torch.argmax(logits, dim=1)[0].cpu().numpy()
                backend = f"PyTorch [{size} px]"

            if classes.shape != image.shape:
                from PIL import Image

                classes = np.asarray(
                    Image.fromarray(classes.astype(np.uint8)).resize(
                        (image.shape[1], image.shape[0]), Image.NEAREST), dtype=np.int64)

            self._prediction = classes
            self.preview.set_image(PALETTE[np.clip(classes, 0, len(PALETTE) - 1)])

            truth_file = None
            masks = self.masks_dir.path()
            if masks:
                for suffix in (".png", ".tif", ".tiff"):
                    candidate = Path(masks) / f"{Path(path).stem}{suffix}"
                    if candidate.is_file():
                        truth_file = candidate
                        break
            if truth_file is None:
                self.metrics.setText(f"{backend} | {image.shape[1]}x{image.shape[0]} px | "
                                     f"no reference mask for this image")
                return
            truth = data_io.load_classes(truth_file, image.shape)
            scores = metric_lib.dice_per_class(classes, truth, self.num_classes())
            names = self.class_names()
            detail = []
            for key, value in sorted(scores.items()):
                if not key.startswith("class_"):
                    continue
                index = int(key.split("_")[1])
                label = names[index] if 0 <= index < len(names) else key
                detail.append(f"{label} {value:.3f}")
            self.metrics.setText(
                f"{backend} | mean class Dice {scores['mean']:.4f} over "
                f"{int(scores['classes_present'])} classes | " + "  ".join(detail))
        except Exception as exc:
            self._echo(f"prediction failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()
            self.progress.setRange(0, 100)

    def on_save_clicked(self) -> None:
        if self._prediction is None:
            self._echo("nothing to save yet")
            return
        path, _ = QFileDialog.getSaveFileName(
            self.context.qt_parent, "Save class map",
            str(Path.home() / "refwranet_layers.png"), "PNG (*.png)")
        if path:
            from PIL import Image

            Image.fromarray(self._prediction.astype(np.uint8)).save(path)
            self._echo(f"saved class-index map to {path}")

    # ============================================================ export
    def on_export_clicked(self) -> None:
        from amis.ml.onnx_export import export_onnx

        output = str(Path(self.context.data_dir) / "refwranet.onnx")
        try:
            metadata = export_onnx(
                build_model=build_model,
                model_config=self.model_config(),
                output_path=output,
                input_size=self.size_spin.value(),
                weights=self.weights_path.path() or None,
            )
            self._echo(f"exported {output} with "
                       f"{metadata.get('model_config', {}).get('num_classes')} channels "
                       f"(verified={metadata.get('verified')}). Apply softmax, then argmax.")
        except Exception as exc:
            self._echo(f"export failed: {exc}")

    # =============================================================== DWT
    def on_decompose_clicked(self) -> None:
        path = self.dwt_image.path()
        if not path:
            self._echo("pick an image for the decomposition")
            return
        try:
            image = data_io.load_gray(path)
            wavelet = self.dwt_wavelet.currentText()
            levels = int(self.dwt_level.value())

            steps = wavelib.dwt2_steps(image, wavelet)
            self._dwt_steps = steps
            stats = wavelib.sub_band_stats(steps)
            total_energy = sum(item["energy"] for item in stats.values()) or 1.0

            for band, view in self.band_views.items():
                band_image = steps[band]
                # each band gets its own scale, otherwise LL dwarfs the details
                peak = float(np.max(np.abs(band_image))) or 1.0
                view.set_image(np.abs(band_image) / peak)

            lines = [f"image {image.shape[1]}x{image.shape[0]} -> "
                     f"{steps['LL'].shape[1]}x{steps['LL'].shape[0]}", "",
                     wavelib.describe_wavelet(wavelet), ""]
            for band in ("LL", "LH", "HL", "HH"):
                share = 100.0 * stats[band]["energy"] / total_energy
                lines.append(f"{band}: {wavelib.describe_band(band)}")
                lines.append(f"     {share:5.1f}% of the detail energy, "
                             f"max |coefficient| {stats[band]['max_abs']:.3f}")
            if levels > 1:
                chain = wavelib.cascade(image, wavelet, levels)
                shapes = " -> ".join(str(item[1]["LL"].shape[::-1]) for item in chain)
                lines += ["", f"cascade: LL decomposed again {levels} time(s): {shapes}"]
            lines += ["", "Display note: each sub-band is shown on its own scale "
                          "(normalised by its maximum) so the details are visible; "
                          "the percentages above are the real energies."]
            self.dwt_text.setPlainText("\n".join(lines))
            self._echo(f"DWT: {wavelet}, level {levels}, image {image.shape}")
        except Exception as exc:
            self._echo(f"decomposition failed: {exc}")


class RefWranetAddon(AddonBase):
    """Addon entry point."""

    def initialize(self) -> None:
        self.ui = None

    def build_panel(self) -> QWidget:
        self.panel_widget = RefWranetPanel(self)
        return self.panel_widget

    def on_activated(self) -> None:
        self.context.logger.debug("refWRANet panel activated")


ADDON = RefWranetAddon
