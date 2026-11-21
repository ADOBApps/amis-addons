"""WRANet addon: pooling encoder-decoder with RAM skip gating + a DWT explorer.

Two tabs:

* **Layers** - the method: data, model (width, depth, RAM repeats/dropout),
  training with per-class Dice, prediction with a class colour map, ONNX export.
  The architecture is the one in the addon's spec sheet: ``Conv+BN+ReLU``
  encoder stages separated by ``MaxPool2d(2, 2)``, a bottleneck at
  ``base * 2**depth``, a residual attention module (spatial + channel) on every
  skip connection, ``ConvTranspose2d`` upsampling and a 1x1 output head.
* **2-D DWT** - the teaching tab inherited from ``refwranet``: pick an image and
  a wavelet family, press *Decompose*, and see LL/LH/HL/HH with what each band
  contains and how much energy it carries. It is deliberately independent of
  the network (pooling replaced the wavelet downsampling, the explorer stays
  because it is still useful for figures).

The panel prints the tensor ladder (channels and spatial size per stage) and
recomputes it whenever a control changes, so what is built is exactly what the
spec says.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from PySide6.QtCore import Qt, Signal
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

from . import analysis as analysis_lib
from . import wavelib
from .model import DEFAULT_CONFIG, build_model, channel_plan, count_parameters

log = logging.getLogger(__name__)

PALETTE = np.array([
    [0, 0, 0], [220, 60, 60], [240, 150, 40], [40, 200, 120], [200, 200, 60],
    [60, 120, 220], [150, 90, 220], [220, 90, 180], [90, 200, 220],
    [160, 160, 160], [120, 200, 90], [200, 120, 60], [90, 90, 200],
    [235, 235, 235],
], dtype=np.uint8)


class WranetPanel(QWidget):
    """The addon's window: layer segmentation on one tab, DWT on the other."""

    sig_log = Signal(str)
    sig_epoch = Signal(dict)
    sig_done = Signal(dict)
    sig_error = Signal(str)

    def __init__(self, addon: "WranetAddon") -> None:
        super().__init__()
        self.addon = addon
        self.context = addon.context
        self._trainer = None
        self._prediction: Optional[np.ndarray] = None
        self._dwt_steps = None
        self._analysis = None

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
        self._update_architecture()

    # ==================================================== layers tab ====
    def _build_layers_tab(self) -> QWidget:
        # a scroll page: the tab has ~1000 px of controls, the window may not
        page = ScrollPage()
        outer = page.body

        intro = QLabel(
            "Multi-class retinal segmentation, encoder-decoder with attention skip "
            "connections: Conv+BN+ReLU stages separated by MaxPool2d (or by the LL "
            "sub-band of a DWT, as an ablation), a residual attention module "
            "(channel + spatial gate) on every skip, transposed-conv decoding and a "
            "1x1 class head. Masks are class-index PNGs plus classes.json.")
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

        # created before the model box: the architecture label reads its value
        self.size_spin = QSpinBox()
        self.size_spin.setRange(64, 1024)
        self.size_spin.setSingleStep(32)
        self.size_spin.setValue(int(DEFAULT_CONFIG.get("img_size", 256)))

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
        self.downsample_combo = QComboBox()
        self.downsample_combo.addItems(["maxpool", "dwt"])
        self.downsample_combo.setCurrentText(str(DEFAULT_CONFIG["downsample"]))
        self.downsample_combo.setToolTip(
            "maxpool = the architecture in the spec; dwt = LL sub-band ablation. "
            "Both halve the resolution and keep the channels.")
        self.model_wavelet = QComboBox()
        self.model_wavelet.addItems(list(wavelib.MODEL_WAVELETS))
        self.model_wavelet.setCurrentText(str(DEFAULT_CONFIG["wavelet"]))
        self.model_wavelet.setToolTip("Used only when downsampling is 'dwt'")
        self.repeats_spin = QSpinBox()
        self.repeats_spin.setRange(1, 4)
        self.repeats_spin.setValue(int(DEFAULT_CONFIG["repeats"]))
        self.repeats_spin.setToolTip(
            "Conv+BN+ReLU units inside each residual attention block")
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setLocale(c_locale())
        self.dropout_spin.setRange(0.0, 0.9)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setDecimals(2)
        self.dropout_spin.setValue(float(DEFAULT_CONFIG["dropout"]))
        self.architecture = QLabel("")
        self.architecture.setWordWrap(True)
        self.architecture.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        inspect = QPushButton("Count parameters")
        inspect.clicked.connect(self.on_inspect_clicked)
        model_form.addRow("Classes", self.classes_spin)
        model_form.addRow("Base channels", self.base_spin)
        model_form.addRow("Depth", self.depth_spin)
        model_form.addRow("Downsample", self.downsample_combo)
        model_form.addRow("DWT family", self.model_wavelet)
        model_form.addRow("RAM repeats", self.repeats_spin)
        model_form.addRow("RAM dropout", self.dropout_spin)
        model_form.addRow("Tensor ladder", self.architecture)
        model_form.addRow(inspect)
        outer.addWidget(model_box)

        train_box = QGroupBox("Training")
        train_form = QFormLayout(train_box)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(20)
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 32)
        self.batch_spin.setValue(1)
        self.batch_spin.setToolTip(
            "Batch 1 at 256 px keeps the memory footprint small; raise it if the "
            "machine has headroom")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setLocale(c_locale())
        self.lr_spin.setDecimals(6)
        self.lr_spin.setRange(1e-6, 1.0)
        self.lr_spin.setValue(1e-3)
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

        for spin in (self.classes_spin, self.base_spin, self.depth_spin,
                     self.size_spin):
            spin.valueChanged.connect(self._update_architecture)
        self.downsample_combo.currentTextChanged.connect(self._on_downsample_changed)
        self.model_wavelet.currentTextChanged.connect(self._update_architecture)
        self._sync_downsample_controls()
        return page

    # ======================================================= DWT tab ====
    def _build_dwt_tab(self) -> QWidget:
        page = ScrollPage()
        outer = page.body

        intro = QLabel(
            "A 2-D discrete wavelet transform can replace pooling: the wavelet "
            "functions become digital filters - a low-pass filter l smooths and keeps "
            "the main structure (LL), a high-pass filter h keeps the fast variations, "
            "giving horizontal (LH), vertical (HL) and diagonal (HH) details. The "
            "network uses MaxPool2d by default; set *Downsample* to 'dwt' in the "
            "Layers tab to put this LL sub-band inside the encoder instead. The "
            "*Analysis* group below applies the outline's stages as fixed wavelet "
            "rules (no weights, no training).")
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

        # ------------------------------------------------ training-free maps
        analysis_box = QGroupBox("Analysis (training-free)")
        analysis_form = QFormLayout(analysis_box)
        self.gate_k = QDoubleSpinBox()
        self.gate_k.setLocale(c_locale())
        self.gate_k.setRange(0.0, 10.0)
        self.gate_k.setSingleStep(0.5)
        self.gate_k.setDecimals(2)
        self.gate_k.setValue(3.0)
        self.gate_k.setToolTip(
            "Gate strength: T = k * sigma * sqrt(2 ln n), sigma from the median "
            "absolute deviation of each band")
        self.gate_mode = QComboBox()
        self.gate_mode.addItems(["soft", "hard", "none"])
        analyse = QPushButton("Analyse (no trained weights)")
        analyse.clicked.connect(self.on_analyse_clicked)
        analysis_form.addRow("Gate strength k", self.gate_k)
        analysis_form.addRow("Threshold", self.gate_mode)
        analysis_form.addRow(analyse)
        outer.addWidget(analysis_box)

        maps = QGridLayout()
        self.map_views = {}
        for position, (name, caption) in enumerate((
                ("reconstruction", "denoised reconstruction (inverse DWT)"),
                ("detail", "detail energy, finest level (normalised)"),
                ("gate", "gate |c| / (|c| + T): bright = kept"))):
            box = QGroupBox(name.capitalize())
            box_layout = QVBoxLayout(box)
            caption_label = QLabel(caption)
            caption_label.setWordWrap(True)
            view = ImageView(150)
            box_layout.addWidget(view)
            box_layout.addWidget(caption_label)
            maps.addWidget(box, position // 2, position % 2)
            self.map_views[name] = view
        outer.addLayout(maps)

        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setMinimumHeight(120)
        outer.addWidget(self.analysis_text)

        save_row = QHBoxLayout()
        save_recon = QPushButton("Save reconstruction...")
        save_recon.clicked.connect(self.on_save_recon_clicked)
        save_csv = QPushButton("Save report (.csv)...")
        save_csv.clicked.connect(self.on_save_analysis_csv_clicked)
        save_row.addWidget(save_recon)
        save_row.addWidget(save_csv)
        save_row.addStretch(1)
        outer.addLayout(save_row)
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
            "channel_multiplier": int(DEFAULT_CONFIG["channel_multiplier"]),
            "repeats": self.repeats_spin.value(),
            "dropout": float(self.dropout_spin.value()),
            "upsample": str(DEFAULT_CONFIG["upsample"]),
            "downsample": self.downsample_combo.currentText(),
            "wavelet": self.model_wavelet.currentText(),
        }

    def _plan_lines(self) -> list:
        plan = channel_plan(self.model_config(), self.size_spin.value())
        depth = plan["input"]
        lines = [f"input [{depth[0]}, {depth[1]}x{depth[2]}]"]
        padded = plan["padded_input"]
        if padded[1] != depth[1] or padded[2] != depth[2]:
            lines.append(f"padded to [{padded[0]}, {padded[1]}x{padded[2]}] "
                         f"for depth {self.depth_spin.value()}")
        if plan["downsample"] == "dwt":
            lines.append(f"downsampling: dwt (LL sub-band, {plan['wavelet']})")
        else:
            lines.append("downsampling: maxpool (spec)")
        for stage in plan["encoder"]:
            height, width = stage["size"]
            lines.append(f"enc {stage['stage']}: {stage['channels']} ch @ {height}x{width}")
        bottleneck = plan["bottleneck"]
        bottom_h, bottom_w = bottleneck["size"]
        lines.append(f"bottleneck: {bottleneck['channels']} ch @ {bottom_h}x{bottom_w}")
        for stage in plan["decoder"]:
            height, width = stage["size"]
            lines.append(
                f"dec {stage['stage']}: {stage['in_channels']} -> up {stage['upsampled_channels']}"
                f" + RAM skip {stage['skip_channels']} = {stage['out_channels']} ch"
                f" @ {height}x{width}")
        head = plan["head"]
        lines.append(f"head 1x1: {head['in_channels']} -> {head['out_channels']} classes")
        return lines

    def _sync_downsample_controls(self) -> None:
        self.model_wavelet.setEnabled(
            self.downsample_combo.currentText() == "dwt")

    def _on_downsample_changed(self, _text: str = "") -> None:
        self._sync_downsample_controls()
        self._update_architecture()

    def _update_architecture(self) -> None:
        try:
            self.architecture.setText("\n".join(self._plan_lines()))
        except Exception as exc:
            self.architecture.setText(f"architecture: {exc}")

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
            downsample = self.downsample_combo.currentText()
            if downsample == "dwt":
                downsample = f"dwt/{self.model_wavelet.currentText()}"
            self._echo(f"{type(model).__name__}: {count_parameters(model)/1e6:.2f} M "
                       f"parameters, {self.num_classes()} classes, "
                       f"downsample {downsample}, "
                       f"RAM repeats {self.repeats_spin.value()}, "
                       f"dropout {self.dropout_spin.value():.2f}")
            for line in self._plan_lines():
                self._echo("  " + line)
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
                if not weights:
                    self._echo("no weights selected and no bundled ONNX model: "
                               "the untrained network will output noise")
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
            str(Path.home() / "wranet_layers.png"), "PNG (*.png)")
        if path:
            from PIL import Image

            Image.fromarray(self._prediction.astype(np.uint8)).save(path)
            self._echo(f"saved class-index map to {path}")

    # ============================================================ export
    def on_export_clicked(self) -> None:
        from amis.ml.onnx_export import export_onnx

        output = str(Path(self.context.data_dir) / "wranet.onnx")
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

    # ============================================================ analysis
    def on_analyse_clicked(self) -> None:
        path = self.dwt_image.path()
        if not path:
            self._echo("pick an image for the analysis")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            image = data_io.load_gray(path)
            result = analysis_lib.run_analysis(
                image,
                wavelet=self.dwt_wavelet.currentText(),
                levels=int(self.dwt_level.value()),
                k=float(self.gate_k.value()),
                mode=self.gate_mode.currentText(),
            )
            self._analysis = result
            self.map_views["reconstruction"].set_image(result.recon)
            self.map_views["detail"].set_image(result.detail)
            self.map_views["gate"].set_image(result.gate)
            self.analysis_text.setPlainText(result.report)
            self._echo(f"analysis (no weights): {result.wavelet}, "
                       f"{result.levels} level(s), k {result.k:.2f}, "
                       f"RMSE {result.rmse:.6f}, PSNR {result.psnr:.2f} dB")
        except Exception as exc:
            self._echo(f"analysis failed: {exc}")
        finally:
            QApplication.restoreOverrideCursor()

    def on_save_recon_clicked(self) -> None:
        if self._analysis is None:
            self._echo("run the analysis first")
            return
        path, _ = QFileDialog.getSaveFileName(
            self.context.qt_parent, "Save reconstruction",
            str(Path.home() / "wranet_reconstruction.png"), "PNG (*.png)")
        if path:
            from PIL import Image

            data = (np.clip(self._analysis.recon, 0.0, 1.0) * 255).astype(np.uint8)
            Image.fromarray(data).save(path)
            self._echo(f"saved reconstruction to {path}")

    def on_save_analysis_csv_clicked(self) -> None:
        if self._analysis is None:
            self._echo("run the analysis first")
            return
        path, _ = QFileDialog.getSaveFileName(
            self.context.qt_parent, "Save analysis report",
            str(Path.home() / "wranet_analysis.csv"), "CSV (*.csv)")
        if not path:
            return
        import csv

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(analysis_lib.AnalysisResult.csv_header())
            writer.writerows(self._analysis.csv_rows())
        self._echo(f"saved analysis report to {path}")


class WranetAddon(AddonBase):
    """Addon entry point."""

    def initialize(self) -> None:
        self.ui = None

    def build_panel(self) -> QWidget:
        self.panel_widget = WranetPanel(self)
        return self.panel_widget

    def on_activated(self) -> None:
        self.context.logger.debug("WRANet panel activated")


ADDON = WranetAddon
