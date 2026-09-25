"""Main dock panel for the Aerial LiDAR Classifier plugin.

The panel is a QDockWidget intended to live on the right side of the
QGIS main window (like Processing or the Layer Styling dock). It uses
native QGIS widgets where possible (QgsFileWidget, QgsCollapsibleGroupBox,
QgsMessageBar) and stays out of the way when unused.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

from qgis.PyQt.QtCore import QSize, Qt, pyqtSignal
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import (
    QAbstractItemView,
    QAction,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QTextEdit,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from qgis.core import (
    Qgis,
    QgsApplication,
    QgsMapLayerProxyModel,
    QgsProject,
    QgsSettings,
)
from qgis.gui import (
    QgsCollapsibleGroupBox,
    QgsFileWidget,
    QgsMapLayerComboBox,
    QgsMessageBar,
)

from ..config import (
    PLUGIN_NAME,
    SETTINGS_PREFIX,
    TILE_DEFAULT_BUFFER_M,
)
from ..core.registry import MODELS, default_model_for_device, get_model
from ..utils.helpers import get_gpu_info, truncate_name
from ..utils.las_units import UNIT_OVERRIDES
from ..utils.las_utils import find_las_files
from ..utils.logger import LOG_TAG, log_error, log_info, log_warning
from ..utils.model_manager import ModelManager
from ..widgets.drag_drop_list import DragDropList
from ..workers.file_loader import FileInfoLoader


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _std_icon(style_enum) -> QIcon:
    """Return a standard Qt icon (used as a sensible fallback)."""
    return QgsApplication.style().standardIcon(style_enum)


def _qgis_icon(theme_path: str, fallback) -> QIcon:
    """Return a QGIS-themed icon by path with a Qt-standard fallback."""
    icon = QIcon(f":/images/themes/default/{theme_path}")
    if icon.isNull():
        return _std_icon(fallback)
    return icon


# ---------------------------------------------------------------------------
# Dock widget
# ---------------------------------------------------------------------------

class ClassifierDockWidget(QDockWidget):
    """Right-side dock panel hosting the entire plugin UI."""

    closed = pyqtSignal()

    def __init__(self, iface, parent=None):
        super().__init__(PLUGIN_NAME, parent)
        self.iface = iface
        self.setObjectName("AerialLidarClassifierDock")
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )

        # Runtime state
        self.files: list[Path] = []
        self.gpu_info: dict = {"available": False}
        self.task = None
        self._loaders: list[FileInfoLoader] = []
        self._auto_output_set = False
        # Weights of the selected model present on disk, and the model
        # able to run on the current compute device.
        self.model_ready = False
        self.model_gate_ok = True
        self.model_gate_message = ""

        # Settings (load BEFORE building UI so defaults populate widgets)
        self._load_settings()

        # UI
        self._build_ui()
        self._check_gpu()
        self._select_initial_model()
        self._check_model()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.gpu_check.toggled.connect(self._on_gpu_toggled)

        # Listen for plugin log messages
        QgsApplication.messageLog().messageReceived.connect(self._on_log_message)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def sizeHint(self):  # noqa: N802 - Qt API
        """Suggest a generous default size so QGIS docks the panel
        wide enough to show all collapsible group titles + the log +
        the footer without scrolling."""
        return QSize(420, 780)

    def closeEvent(self, event):  # noqa: N802 - Qt API
        self._save_settings()
        try:
            QgsApplication.messageLog().messageReceived.disconnect(
                self._on_log_message
            )
        except Exception:
            pass

        # Stop and reap any in-flight FileInfoLoader threads so we do
        # not leak QThread handles if the user closes the dock while
        # large file scans are still queued.
        for loader in list(self._loaders):
            try:
                if loader.isRunning():
                    loader.requestInterruption()
                    loader.quit()
                    loader.wait(2000)
            except Exception:
                pass
        self._loaders.clear()

        # Cancel any running classification task and disconnect its
        # signals so completion callbacks do not reach into a deleted
        # dock object after we are gone.
        if self.task is not None:
            try:
                if not self.task.isCanceled():
                    self.task.cancel()
            except Exception:
                pass
            for signal_name in (
                "progressChanged",
                "taskCompleted",
                    "taskTerminated"):
                try:
                    getattr(self.task, signal_name).disconnect()
                except Exception:
                    pass
            self.task = None

        self.closed.emit()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def _load_settings(self):
        s = QgsSettings()

        # One-time migration: v1.0.1 and earlier saved `use_gpu=False`
        # whenever the GPU probe failed (e.g. fresh install before the
        # venv had torch, or a CUDA wheel that didn't match the driver).
        # That False then stuck across sessions and left users silently
        # on CPU even after a successful CUDA install or reinstall.
        # On first launch of >=1.0.2, clear that stale False back to
        # True so the next _check_gpu() default is "use GPU if present".
        # The user can still uncheck it; the new _save_settings() only
        # writes the choice when the checkbox is actually enabled.
        settings_version = s.value(
            f"{SETTINGS_PREFIX}/settings_version", 0, type=int
        )
        if settings_version < 2:
            s.setValue(f"{SETTINGS_PREFIX}/use_gpu", True)
            s.setValue(f"{SETTINGS_PREFIX}/settings_version", 2)

        self._saved_suffix = s.value(
            f"{SETTINGS_PREFIX}/suffix", "_classified")
        self._saved_field = s.value(
            f"{SETTINGS_PREFIX}/field_name", "classification"
        )
        self._saved_load_result = s.value(
            f"{SETTINGS_PREFIX}/load_result", True, type=bool
        )
        self._saved_use_gpu = s.value(
            f"{SETTINGS_PREFIX}/use_gpu", True, type=bool
        )
        self._saved_output_dir = s.value(f"{SETTINGS_PREFIX}/output_dir", "")
        self._saved_tile_enabled = s.value(
            f"{SETTINGS_PREFIX}/tile_enabled", False, type=bool
        )
        self._saved_tile_auto = s.value(
            f"{SETTINGS_PREFIX}/tile_auto", True, type=bool
        )
        self._saved_tile_size_m = s.value(
            f"{SETTINGS_PREFIX}/tile_size_m", 500.0, type=float
        )
        self._saved_tile_buffer_m = s.value(
            f"{SETTINGS_PREFIX}/tile_buffer_m",
            TILE_DEFAULT_BUFFER_M,
            type=float)
        self._saved_tile_streaming = s.value(
            f"{SETTINGS_PREFIX}/tile_streaming", False, type=bool
        )
        self._saved_units = str(
            s.value(f"{SETTINGS_PREFIX}/input_units", "auto") or "auto"
        )
        self._saved_model_id = str(s.value(f"{SETTINGS_PREFIX}/model_id", "") or "")

    def _save_settings(self):
        s = QgsSettings()
        s.setValue(f"{SETTINGS_PREFIX}/suffix", self.suffix_edit.text())
        s.setValue(
            f"{SETTINGS_PREFIX}/field_name",
            self.field_edit.text().strip() or "classification",
        )
        s.setValue(
            f"{SETTINGS_PREFIX}/load_result",
            self.load_result_check.isChecked())
        # Only persist `use_gpu` when the checkbox was a real user choice.
        # When the GPU probe fails (e.g. during a fresh install before
        # torch is importable, or with a CUDA wheel that doesn't match
        # the driver), _check_gpu() force-disables and unchecks the box.
        # Saving that False here would stick across sessions and leave
        # the next launch with GPU silently off even after a successful
        # CUDA install. Preserve the previously stored preference instead.
        if self.gpu_check.isEnabled():
            s.setValue(f"{SETTINGS_PREFIX}/use_gpu", self.gpu_check.isChecked())
        if self.output_widget.filePath():
            s.setValue(
                f"{SETTINGS_PREFIX}/output_dir", self.output_widget.filePath()
            )
        s.setValue(
            f"{SETTINGS_PREFIX}/tile_enabled", self.tile_check.isChecked()
        )
        s.setValue(
            f"{SETTINGS_PREFIX}/tile_auto", self.tile_auto_check.isChecked()
        )
        s.setValue(
            f"{SETTINGS_PREFIX}/tile_size_m", self.tile_size_spin.value()
        )
        s.setValue(
            f"{SETTINGS_PREFIX}/tile_buffer_m", self.tile_buffer_spin.value()
        )
        s.setValue(
            f"{SETTINGS_PREFIX}/tile_streaming",
            self.tile_streaming_check.isChecked(),
        )
        s.setValue(
            f"{SETTINGS_PREFIX}/input_units",
            self.units_combo.currentData() or "auto",
        )
        s.setValue(
            f"{SETTINGS_PREFIX}/model_id", self.model_combo.currentData() or "",
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ---- Status strip --------------------------------------------------
        root_layout.addWidget(self._build_status_strip())

        # ---- Local message bar --------------------------------------------
        self.message_bar = QgsMessageBar()
        self.message_bar.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        root_layout.addWidget(self.message_bar)

        # ---- Vertical splitter: parameters on top, log always visible
        # below. The user can drag the splitter handle to give more or
        # less room to the running log.
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)

        # Top: scrollable parameters. Minimum height is sized to fit:
        # Input files (expanded ~250) + Output (expanded ~170) +
        # Advanced collapsed (~30) + spacing - so every group title is
        # visible without scrolling at the default split.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(350)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(8, 8, 8, 8)
        body_layout.setSpacing(6)
        body_layout.addWidget(self._build_input_group())
        body_layout.addWidget(self._build_output_group())
        body_layout.addWidget(self._build_advanced_group())
        body_layout.addStretch()
        scroll.setWidget(body)
        splitter.addWidget(scroll)

        # Bottom: always-visible log panel
        log_panel = self._build_log_panel()
        log_panel.setMinimumHeight(110)
        splitter.addWidget(log_panel)

        # Default split: parameters get ~75% of the splitter, log ~25%.
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([520, 140])
        root_layout.addWidget(splitter, 1)

        # ---- Sticky footer (progress + run/cancel) -------------------------
        root_layout.addWidget(self._build_footer())

        self.setWidget(root)
        # Default size: tall enough that every collapsible group title
        # (Input files / Output / Advanced parameters) is visible
        # without the user having to scroll inside the parameters
        # area. The splitter can still be dragged afterwards.
        self.setMinimumWidth(380)

    # ---- Status strip ----------------------------------------------------
    def _build_status_strip(self) -> QWidget:
        strip = QFrame()
        strip.setFrameShape(QFrame.Shape.StyledPanel)
        strip.setObjectName("AerialLidarStatusStrip")
        lay = QHBoxLayout(strip)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(12)

        self.device_label = QLabel("Device: detecting...")

        # Model selector: short names in the combo (the strip is narrow),
        # the full name and description in its tooltip.
        self.model_combo = QComboBox()
        for spec in MODELS:
            self.model_combo.addItem(spec.short_name, spec.id)
        self.model_label = QLabel("")
        self.model_label.setStyleSheet("color: palette(mid);")

        lay.addWidget(self.device_label)
        lay.addWidget(self._vline())
        lay.addWidget(QLabel("Model:"))
        lay.addWidget(self.model_combo)
        lay.addWidget(self.model_label)
        lay.addStretch()

        # Right-side icon buttons (download / import model weights)
        self.download_btn = QToolButton()
        self.download_btn.setIcon(
            _qgis_icon(
                "mActionFileSaveAs.svg",
                QStyle.StandardPixmap.SP_ArrowDown))
        self.download_btn.setToolTip("Download the selected model's weights")
        self.download_btn.setAutoRaise(True)
        self.download_btn.clicked.connect(self._download_model)
        lay.addWidget(self.download_btn)

        self.import_btn = QToolButton()
        self.import_btn.setIcon(
            _qgis_icon(
                "mActionFileOpen.svg",
                QStyle.StandardPixmap.SP_DialogOpenButton))
        self.import_btn.setToolTip(
            "Import a weights file you already have for the selected model "
            "(offline machines). The file is copied and its SHA-256 verified."
        )
        self.import_btn.setAutoRaise(True)
        self.import_btn.clicked.connect(self._import_model_file)
        lay.addWidget(self.import_btn)

        return strip

    def _vline(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    def _hline(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        return line

    # ---- Input -----------------------------------------------------------
    def _build_input_group(self) -> QgsCollapsibleGroupBox:
        group = QgsCollapsibleGroupBox("Input files")
        group.setCollapsed(False)
        lay = QVBoxLayout(group)
        lay.setSpacing(6)

        # --- Visible "use a layer already loaded in QGIS" row -----------
        layer_row = QHBoxLayout()
        layer_row.setSpacing(4)
        layer_row.addWidget(QLabel("From loaded layer:"))

        self.layer_combo = QgsMapLayerComboBox()
        # The point-cloud filter moved to Qgis.LayerFilter (3.34+, the
        # only spelling QGIS 4 keeps); older spellings as fallbacks.
        for candidate in (
            lambda: Qgis.LayerFilter.PointCloudLayer,
            lambda: QgsMapLayerProxyModel.Filter.PointCloudLayer,
            lambda: QgsMapLayerProxyModel.PointCloudLayer,  # type: ignore
        ):
            try:
                self.layer_combo.setFilters(candidate())
                break
            except Exception:
                continue
        self.layer_combo.setAllowEmptyLayer(
            True, "(select a point-cloud layer)")
        self.layer_combo.setShowCrs(False)
        layer_row.addWidget(self.layer_combo, 1)

        add_layer_btn = QPushButton("Add")
        add_layer_btn.setToolTip(
            "Add the selected layer's source file to the input list"
        )
        add_layer_btn.clicked.connect(self._add_current_layer)
        layer_row.addWidget(add_layer_btn)
        lay.addLayout(layer_row)

        lay.addWidget(self._hline())

        # File list (drag/drop)
        self.file_list = DragDropList()
        self.file_list.filesAdded.connect(self._on_files_dropped)
        self.file_list.setMinimumHeight(120)
        self.file_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        lay.addWidget(self.file_list)

        # Toolbar (compact icon buttons)
        toolbar = QToolBar()
        toolbar.setIconSize(QgsApplication.style().standardIcon(
            QStyle.StandardPixmap.SP_FileIcon).availableSizes()[0]
            if QgsApplication.style().standardIcon(
                QStyle.StandardPixmap.SP_FileIcon).availableSizes()
            else toolbar.iconSize()
        )

        add_files_action = QAction(
            _qgis_icon(
                "mActionAdd.svg",
                QStyle.StandardPixmap.SP_FileDialogStart),
            "Add files...",
            self,
        )
        add_files_action.triggered.connect(self._add_files)
        toolbar.addAction(add_files_action)

        add_folder_action = QAction(
            _qgis_icon("mActionAddRasterLayer.svg",
                       QStyle.StandardPixmap.SP_DirOpenIcon),
            "Add folder...",
            self,
        )
        add_folder_action.triggered.connect(self._add_folder)
        toolbar.addAction(add_folder_action)

        layers_action = QAction(
            _qgis_icon("mActionAddPointCloudLayer.svg",
                       QStyle.StandardPixmap.SP_FileDialogContentsView),
            "Pick from loaded layers...",
            self,
        )
        layers_action.setToolTip(
            "Choose one or more point-cloud layers already loaded in the "
            "QGIS project."
        )
        layers_action.triggered.connect(self._add_from_layers)
        toolbar.addAction(layers_action)

        toolbar.addSeparator()

        self.remove_action = QAction(
            _qgis_icon("mActionRemove.svg",
                       QStyle.StandardPixmap.SP_TrashIcon),
            "Remove selected",
            self,
        )
        self.remove_action.triggered.connect(self._remove_files)
        toolbar.addAction(self.remove_action)

        clear_action = QAction(
            _qgis_icon("mActionDeleteSelected.svg",
                       QStyle.StandardPixmap.SP_DialogResetButton),
            "Clear all",
            self,
        )
        clear_action.triggered.connect(self._clear_files)
        toolbar.addAction(clear_action)

        lay.addWidget(toolbar)

        self.file_stats = QLabel("No files")
        self.file_stats.setStyleSheet("color: palette(mid);")
        lay.addWidget(self.file_stats)

        return group

    # ---- Output ----------------------------------------------------------
    def _build_output_group(self) -> QGroupBox:
        """Output is intentionally a plain non-collapsible group box.

        Every parameter here is essential to every run, so the section
        is always visible (no expand/collapse caret). The output
        extension is automatic - LAS in / LAS out, LAZ in (or COPC) /
        LAZ out - so no format combo to pick from.
        """
        group = QGroupBox("Output")
        form = QFormLayout(group)
        form.setContentsMargins(6, 10, 6, 6)
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.output_widget = QgsFileWidget()
        self.output_widget.setStorageMode(
            QgsFileWidget.StorageMode.GetDirectory)
        self.output_widget.setDialogTitle("Select output folder")
        if self._saved_output_dir:
            self.output_widget.setFilePath(self._saved_output_dir)
        self.output_widget.fileChanged.connect(self._update_run_button)
        form.addRow("Folder:", self.output_widget)

        self.suffix_edit = QLineEdit(self._saved_suffix)
        self.suffix_edit.setPlaceholderText("_classified")
        form.addRow("Suffix:", self.suffix_edit)

        # Classification field. Default 'classification' is the ASPRS
        # standard LAS dimension (auto-upgrades to LAS 1.4 / PF 6 if a
        # code > 31 is written). Any other name creates an extra-byte
        # field instead.
        self.field_edit = QLineEdit(self._saved_field or "classification")
        self.field_edit.setPlaceholderText("classification")
        self.field_edit.setToolTip(
            "LAS dimension to write the classification into.\n"
            "- 'classification' (default): the ASPRS-standard dimension. "
            "The file is auto-upgraded to LAS 1.4 / point format 6 when "
            "an assigned code exceeds the legacy 5-bit limit.\n"
            "- Any other name: an extra-byte field is added to the "
            "output (non-standard, but useful when you must preserve "
            "the input's existing 'classification')."
        )
        form.addRow("Field:", self.field_edit)

        self.load_result_check = QCheckBox(
            "Load classified files in QGIS"
        )
        self.load_result_check.setChecked(self._saved_load_result)
        self.load_result_check.setToolTip(
            "Add every output file to the current QGIS project as a "
            "point-cloud layer once classification completes."
        )
        form.addRow("", self.load_result_check)

        return group

    # ---- Advanced parameters --------------------------------------------
    def _build_advanced_group(self) -> QgsCollapsibleGroupBox:
        """Flat 'Advanced parameters' group - no nested collapsibles.

        Two sub-sections built as flat blocks: Compute device and
        Performance / Tiling. The header of each block is a simple
        bold label rather than another collapsible inside a
        collapsible.
        """
        group = QgsCollapsibleGroupBox("Advanced parameters")
        group.setCollapsed(True)
        lay = QVBoxLayout(group)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(8)

        # ---- Compute device ----------------------------------------------
        lay.addWidget(self._section_header("Compute device"))

        self.gpu_check = QCheckBox("Use GPU (CUDA)")
        self.gpu_check.setEnabled(False)
        lay.addWidget(self.gpu_check)

        self.gpu_detail = QLabel("Detecting hardware...")
        self.gpu_detail.setWordWrap(True)
        self.gpu_detail.setStyleSheet("color: palette(mid);")
        lay.addWidget(self.gpu_detail)

        clear_gpu_btn = QPushButton("Clear GPU memory")
        clear_gpu_btn.clicked.connect(self._clear_gpu_memory)
        lay.addWidget(clear_gpu_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        lay.addWidget(self._hline())

        # ---- Performance / Tiling ----------------------------------------
        lay.addWidget(self._section_header("Performance / Tiling"))

        self.tile_check = QCheckBox(
            "Tile large files (process in spatial chunks)"
        )
        self.tile_check.setChecked(self._saved_tile_enabled)
        self.tile_check.setToolTip(
            "Splits the input into N x N spatial tiles with a buffer "
            "overlap, runs inference per tile and merges the predictions "
            "back. Recommended for files with many millions of points."
        )
        self.tile_check.toggled.connect(self._update_tile_controls_enabled)
        lay.addWidget(self.tile_check)

        perf_form = QFormLayout()
        perf_form.setSpacing(6)
        perf_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.tile_auto_check = QCheckBox("Auto-size (~10 M points per tile)")
        self.tile_auto_check.setChecked(self._saved_tile_auto)
        self.tile_auto_check.toggled.connect(
            self._update_tile_controls_enabled)
        perf_form.addRow("", self.tile_auto_check)

        self.tile_size_spin = QDoubleSpinBox()
        self.tile_size_spin.setRange(50.0, 50_000.0)
        self.tile_size_spin.setDecimals(0)
        self.tile_size_spin.setSingleStep(50.0)
        self.tile_size_spin.setSuffix(" m")
        self.tile_size_spin.setValue(self._saved_tile_size_m)
        self.tile_size_spin.setToolTip(
            "Side length of each square tile, in CRS units. Used only "
            "when 'Auto-size' is off."
        )
        perf_form.addRow("Tile size:", self.tile_size_spin)

        self.tile_buffer_spin = QDoubleSpinBox()
        self.tile_buffer_spin.setRange(0.0, 5_000.0)
        self.tile_buffer_spin.setDecimals(0)
        self.tile_buffer_spin.setSingleStep(5.0)
        self.tile_buffer_spin.setSuffix(" m")
        self.tile_buffer_spin.setValue(self._saved_tile_buffer_m)
        self.tile_buffer_spin.setToolTip(
            "Spatial buffer around each tile providing context for "
            "points near the tile edges. Predictions in the buffer "
            "are discarded."
        )
        perf_form.addRow("Buffer:", self.tile_buffer_spin)

        lay.addLayout(perf_form)

        self.tile_streaming_check = QCheckBox(
            "Streaming I/O (requires tiling - handles files larger than RAM)"
        )
        # Restore only if tiling was also enabled - streaming is
        # meaningless on its own.
        self.tile_streaming_check.setChecked(
            self._saved_tile_streaming and self._saved_tile_enabled
        )
        self.tile_streaming_check.setToolTip(
            "Stream the input chunk-by-chunk via laspy. Each tile is "
            "written to disk-backed sidecars instead of being held in "
            "memory, then merged back through a streaming writer. "
            "Required for files that do not fit in RAM.\n\n"
            "This option requires 'Tile large files' - turning it on "
            "automatically enables tiling; turning tiling off "
            "automatically disables streaming."
        )
        self.tile_streaming_check.toggled.connect(self._on_streaming_toggled)
        lay.addWidget(self.tile_streaming_check)

        lay.addWidget(self._hline())

        # ---- Input units -------------------------------------------------
        lay.addWidget(self._section_header("Input units"))

        units_form = QFormLayout()
        units_form.setSpacing(6)
        units_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.units_combo = QComboBox()
        for key, label in UNIT_OVERRIDES:
            self.units_combo.addItem(label, key)
        saved_idx = self.units_combo.findData(self._saved_units)
        self.units_combo.setCurrentIndex(saved_idx if saved_idx >= 0 else 0)
        self.units_combo.setToolTip(
            "The model works in metres. By default the unit is read from "
            "each file's CRS (WKT or GeoTIFF keys) and feet are converted "
            "before inference; the log says what was found. Force metres "
            "or feet when the header is missing or wrong."
        )
        units_form.addRow("Units:", self.units_combo)
        lay.addLayout(units_form)

        self._update_tile_controls_enabled()
        return group

    def _on_streaming_toggled(self, checked: bool) -> None:
        """When the user enables streaming, also enable tiling.

        Streaming is implemented as a per-tile pipeline; running it
        without tiling makes no sense. Coupling the two checkboxes
        explicitly avoids the surprising 'streaming checked but
        ignored' state the user could otherwise reach.
        """
        if checked and not self.tile_check.isChecked():
            # Block signals so we don't bounce back through
            # _update_tile_controls_enabled and undo the streaming
            # checkbox we just enabled.
            self.tile_check.blockSignals(True)
            self.tile_check.setChecked(True)
            self.tile_check.blockSignals(False)
            self._update_tile_controls_enabled()

    def _section_header(self, text: str) -> QLabel:
        """Small bold label used as a flat in-group section divider."""
        label = QLabel(text)
        f = label.font()
        f.setBold(True)
        label.setFont(f)
        return label

    def _update_tile_controls_enabled(self):
        on = self.tile_check.isChecked()
        self.tile_auto_check.setEnabled(on)
        self.tile_buffer_spin.setEnabled(on)
        self.tile_size_spin.setEnabled(
            on and not self.tile_auto_check.isChecked())
        if hasattr(self, "tile_streaming_check"):
            self.tile_streaming_check.setEnabled(on)
            # When tiling is turned off, also force-uncheck streaming
            # so the visual state matches the runtime gate (which
            # requires both flags). Block the toggled signal so this
            # doesn't reactivate tiling via _on_streaming_toggled.
            if not on and self.tile_streaming_check.isChecked():
                self.tile_streaming_check.blockSignals(True)
                self.tile_streaming_check.setChecked(False)
                self.tile_streaming_check.blockSignals(False)

    # ---- Log (always visible) -------------------------------------------
    def _build_log_panel(self) -> QWidget:
        """Build the always-visible log panel that sits below the
        parameters splitter pane."""
        panel = QFrame()
        panel.setFrameShape(QFrame.Shape.StyledPanel)

        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("Log")
        title_font = title.font()
        title_font.setBold(True)
        title.setFont(title_font)
        header.addWidget(title)
        header.addStretch()
        clear_btn = QToolButton()
        clear_btn.setText("Clear")
        clear_btn.setAutoRaise(True)
        clear_btn.clicked.connect(lambda: self.log_text.clear())
        header.addWidget(clear_btn)
        lay.addLayout(header)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setPlaceholderText(
            "Processing messages will appear here..."
        )
        self.log_text.setMinimumHeight(80)
        lay.addWidget(self.log_text, 1)
        return panel

    # ---- Footer ----------------------------------------------------------
    def _build_footer(self) -> QWidget:
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(4)

        self.current_file_label = QLabel("Ready")
        self.current_file_label.setStyleSheet("color: palette(mid);")
        lay.addWidget(self.current_file_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        lay.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel)
        btn_row.addWidget(self.cancel_btn)

        btn_row.addStretch()

        self.run_btn = QPushButton(
            _qgis_icon("mActionStart.svg", QStyle.StandardPixmap.SP_MediaPlay),
            "Run classification",
        )
        self.run_btn.setDefault(True)
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self._run)
        btn_row.addWidget(self.run_btn)

        lay.addLayout(btn_row)
        from ..widgets.cohort_card import CohortCard
        self.cohort_card = CohortCard(parent=bar)
        lay.addWidget(self.cohort_card)
        return bar

    # ------------------------------------------------------------------
    # GPU / Model status
    # ------------------------------------------------------------------

    def _check_gpu(self):
        try:
            self.gpu_info = get_gpu_info()
        except Exception:
            self.gpu_info = {"available": False}

        if self.gpu_info.get("available"):
            name = self.gpu_info.get("name", "GPU")
            self.gpu_check.setEnabled(True)
            self.gpu_check.setChecked(self._saved_use_gpu)
            self.gpu_check.setText(f"Use GPU ({truncate_name(name, 30)})")
            self.gpu_detail.setText(
                f"Backend: {self.gpu_info.get('backend', 'cuda').upper()} - "
                f"{self.gpu_info.get('mem', 0):.1f} GB"
            )
            self.device_label.setText(f"Device: {truncate_name(name, 24)}")
        else:
            self.gpu_check.setEnabled(False)
            self.gpu_check.setChecked(False)
            self.gpu_check.setText("Use GPU (not available)")
            reason = self.gpu_info.get("reason", "no_gpu")
            self.gpu_detail.setText(
                "No CUDA / MPS device detected - inference will run on CPU."
                if reason in ("no_gpu", "no_cuda")
                else f"GPU probe failed: {reason}"
            )
            self.device_label.setText("Device: CPU")

    def _current_device(self) -> str:
        """"cuda", "mps" or "cpu" as the run would use it right now."""
        if (
            self.gpu_check.isEnabled() and self.gpu_check.isChecked()
            and self.gpu_info.get("available")
        ):
            return str(self.gpu_info.get("backend", "cuda"))
        return "cpu"

    def _current_spec(self):
        try:
            return get_model(self.model_combo.currentData())
        except KeyError:
            return MODELS[-1]

    def _select_initial_model(self):
        """Saved choice when it still fits the device, else the default."""
        device = self._current_device()
        chosen = None
        if self._saved_model_id:
            try:
                chosen = get_model(self._saved_model_id)
            except KeyError:
                chosen = None
        if chosen is None or not chosen.supports_device(device):
            chosen = default_model_for_device(device)
        index = self.model_combo.findData(chosen.id)
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(max(index, 0))
        self.model_combo.blockSignals(False)

    def _on_model_changed(self, _index=None):
        self._check_model()

    def _on_gpu_toggled(self, _checked=None):
        # The device gate of the selected model depends on the checkbox.
        self._check_model()

    def _check_model(self):
        spec = self._current_spec()
        manager = ModelManager(spec)
        device = self._current_device()
        self.model_combo.setToolTip(
            f"{spec.display_name}\n{spec.description}\n"
            f"{spec.device_requirement_text()}."
        )

        if manager.is_model_available():
            size_mb = manager.get_model_size_mb()
            self.model_ready = True
            status = f"ready ({size_mb:.0f} MB)"
            self.download_btn.setEnabled(False)
            self.download_btn.setToolTip("Weights already downloaded")
        else:
            self.model_ready = False
            status = "not downloaded"
            self.download_btn.setEnabled(True)
            self.download_btn.setToolTip(
                f"Download the {spec.display_name} weights "
                f"(~{spec.weights_size_mb:.0f} MB)"
            )

        self.model_gate_ok = spec.supports_device(device)
        dependency_message = ""
        if self.model_gate_ok and spec.family == "litept":
            from ..utils.backend_readiness import litept_dependency_status
            self.model_gate_ok, dependency_message = litept_dependency_status()
        if self.model_gate_ok:
            self.model_gate_message = ""
            self.model_label.setStyleSheet("color: palette(mid);")
        else:
            status += " - dependencies unavailable" if dependency_message else " - needs an NVIDIA GPU"
            self.model_gate_message = dependency_message or (
                f"{spec.display_name} runs on NVIDIA CUDA GPUs only. Tick "
                "'Use GPU' in Advanced parameters, or choose SegFormer 3D "
                "which runs on CPU."
            )
            self.model_label.setStyleSheet("color: #b00020;")
        self.model_label.setText(status)
        self.model_label.setToolTip(self.model_gate_message)
        self._update_run_button()

    def _download_model(self):
        from ..dialogs.model_download_dialog import ModelDownloadDialog
        dlg = ModelDownloadDialog(self._current_spec(), self)
        if dlg.exec():
            self._check_model()

    def _import_model_file(self):
        spec = self._current_spec()
        path, _ = QFileDialog.getOpenFileName(
            self, f"Import {spec.display_name} weights", "",
            "PyTorch weights (*.pth *.pt);;All files (*)",
        )
        if not path:
            return
        ok, msg = ModelManager(spec).import_file(path)
        if ok:
            self.message_bar.pushMessage(
                f"{spec.display_name}: weights imported and verified.",
                Qgis.MessageLevel.Success, duration=5,
            )
        else:
            self.message_bar.pushMessage(msg, Qgis.MessageLevel.Critical, duration=0)
        self._check_model()

    def _clear_gpu_memory(self):
        try:
            from ..utils.venv_manager import ensure_venv_packages_available
            ensure_venv_packages_available()
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.message_bar.pushMessage(
                    "GPU memory cache cleared.", Qgis.MessageLevel.Success, duration=4
                )
            else:
                self.message_bar.pushMessage(
                    "No CUDA GPU detected.", Qgis.MessageLevel.Info, duration=4
                )
        except Exception as exc:
            self.message_bar.pushMessage(
                f"Could not clear GPU memory: {exc}", Qgis.MessageLevel.Warning, duration=6
            )

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _on_files_dropped(self, files):
        for f in files:
            if f in self.files:
                continue
            self.files.append(f)
            item = QListWidgetItem(f"{truncate_name(f.name)}  -  loading...")
            item.setToolTip(str(f))
            item.setData(Qt.ItemDataRole.UserRole, f)
            self.file_list.addItem(item)

            loader = FileInfoLoader(f)
            loader.info_ready.connect(self._on_file_info_loaded)
            loader.error.connect(
                lambda fp, err: log_warning(f"Error loading {fp}: {err}")
            )
            # When the worker thread is done, drop it from our keep-alive
            # list so QThread handles do not leak for the lifetime of the
            # dock. `finished` fires on both normal completion and error.
            loader.finished.connect(lambda _l=loader: self._reap_loader(_l))
            loader.start()
            self._loaders.append(loader)

        if not self._auto_output_set and self.files and not self.output_widget.filePath():
            self.output_widget.setFilePath(str(self.files[0].parent))
            self._auto_output_set = True

        self._update_stats()
        self._update_run_button()

    def _reap_loader(self, loader):
        """Drop a finished FileInfoLoader from the keep-alive list."""
        try:
            self._loaders.remove(loader)
        except ValueError:
            pass
        loader.deleteLater()

    def _on_file_info_loaded(self, filepath, point_count, file_size_mb):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == filepath:
                item.setText(
                    f"{truncate_name(filepath.name)}  -  "
                    f"{point_count:,} pts  -  {file_size_mb:.1f} MB"
                )
                break

    def _add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select LAS / LAZ files", "",
            "Point clouds (*.las *.laz);;All files (*)"
        )
        if paths:
            self._on_files_dropped([Path(p) for p in paths])

    def _add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select folder of LAS/LAZ files")
        if folder:
            found = find_las_files(folder)
            if not found:
                self.message_bar.pushMessage(
                    "No LAS / LAZ files found in this folder.",
                    Qgis.MessageLevel.Warning, duration=4,
                )
            else:
                self._on_files_dropped(found)

    def _add_current_layer(self):
        """Add the layer currently selected in the QgsMapLayerComboBox."""
        layer = self.layer_combo.currentLayer()
        if layer is None:
            self.message_bar.pushMessage(
                "Pick a point-cloud layer first.", Qgis.MessageLevel.Info, duration=4
            )
            return
        path = self._layer_source_path(layer)
        if path is None:
            self.message_bar.pushMessage(
                f"Layer '{layer.name()}' has no readable LAS / LAZ source on "
                "disk.", Qgis.MessageLevel.Warning, duration=5,
            )
            return
        self._on_files_dropped([path])

    def _add_from_layers(self):
        """Show a multi-select dialog of all point-cloud layers in the project."""
        candidates: list[tuple[str, Path]] = []
        for layer in QgsProject.instance().mapLayers().values():
            path = self._layer_source_path(layer)
            if path is not None:
                candidates.append((layer.name(), path))

        if not candidates:
            self.message_bar.pushMessage(
                "No LAS / LAZ point-cloud layers in the project.",
                Qgis.MessageLevel.Info, duration=4,
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Pick from loaded layers")
        dlg.setMinimumSize(440, 360)
        dlg_layout = QVBoxLayout(dlg)

        dlg_layout.addWidget(QLabel("Tick the layers to add:"))

        layer_list = QListWidget()
        layer_list.setSelectionMode(
            QAbstractItemView.SelectionMode.NoSelection)
        for name, path in candidates:
            item = QListWidgetItem(f"{name}    —   {path.name}")
            item.setToolTip(str(path))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            item.setData(Qt.ItemDataRole.UserRole, path)
            layer_list.addItem(item)
        dlg_layout.addWidget(layer_list, 1)

        btn_row = QHBoxLayout()
        sel_all = QPushButton("Select all")
        sel_none = QPushButton("Clear")
        sel_all.clicked.connect(lambda: [
            layer_list.item(i).setCheckState(Qt.CheckState.Checked)
            for i in range(layer_list.count())
        ])
        sel_none.clicked.connect(lambda: [
            layer_list.item(i).setCheckState(Qt.CheckState.Unchecked)
            for i in range(layer_list.count())
        ])
        btn_row.addWidget(sel_all)
        btn_row.addWidget(sel_none)
        btn_row.addStretch()

        ok_btn = QPushButton("Add selected")
        ok_btn.setDefault(True)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        dlg_layout.addLayout(btn_row)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        chosen = []
        for i in range(layer_list.count()):
            item = layer_list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                chosen.append(item.data(Qt.ItemDataRole.UserRole))
        if chosen:
            self._on_files_dropped(chosen)

    @staticmethod
    def _layer_source_path(layer) -> Optional[Path]:
        """Return the on-disk LAS/LAZ path backing ``layer``, or None."""
        src = layer.source() if layer is not None else None
        if not src:
            return None
        # PDAL/COPC sources can be plain paths or have provider URI options;
        # strip after a '|' separator if present.
        path_part = src.split("|", 1)[0]
        path = Path(path_part)
        if path.suffix.lower() in (".las", ".laz") and path.exists():
            return path
        return None

    def _remove_files(self):
        selected_paths = [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.file_list.selectedItems()
            if item.data(Qt.ItemDataRole.UserRole) is not None
        ]
        for path in selected_paths:
            for row in range(self.file_list.count() - 1, -1, -1):
                if self.file_list.item(row).data(
                        Qt.ItemDataRole.UserRole) == path:
                    self.file_list.takeItem(row)
                    break
        self.files = [f for f in self.files if f not in selected_paths]
        self._update_stats()
        self._update_run_button()

    def _clear_files(self):
        self.files.clear()
        self.file_list.clear()
        self._auto_output_set = False
        self._update_stats()
        self._update_run_button()

    def _update_stats(self):
        if not self.files:
            self.file_stats.setText("No files")
            return
        total_size = 0
        for f in self.files:
            if isinstance(f, Path) and f.exists():
                total_size += f.stat().st_size
        self.file_stats.setText(
            f"{len(self.files)} file(s) - {total_size / (1024**2):.1f} MB"
        )

    def _update_run_button(self):
        enabled = (
            bool(self.files) and self.model_ready and self.model_gate_ok
            and bool(self.output_widget.filePath())
        )
        self.run_btn.setEnabled(enabled)
        self.run_btn.setToolTip(
            self.model_gate_message if not self.model_gate_ok else ""
        )

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------

    def _run(self):
        out_dir = self.output_widget.filePath()
        if not out_dir:
            self.message_bar.pushMessage(
                "Choose an output folder first.", Qgis.MessageLevel.Warning, duration=4
            )
            return

        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.current_file_label.setText("Starting classification...")
        self._save_settings()

        # "cuda", "mps" or "cpu": the backend the GPU probe reported, but
        # only while the box is checked. v1.0.2 passed a boolean down and
        # the core called model.cuda() for it, which crashed every run on
        # Apple Silicon (where the probe reports MPS).
        if self.gpu_check.isChecked() and self.gpu_info.get("available"):
            device = str(self.gpu_info.get("backend", "cuda"))
        else:
            device = "cpu"
        spec = self._current_spec()
        if not spec.supports_device(device):
            self.message_bar.pushMessage(
                self.model_gate_message, Qgis.MessageLevel.Warning, duration=8,
            )
            self.run_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            return
        field = self.field_edit.text().strip() or "classification"
        is_asprs = field.lower() == "classification"
        self._log(
            f"Starting {spec.display_name} on {device.upper()} with "
            f"{len(self.files)} file(s)"
        )
        self._log(
            f"Output: {out_dir} - "
            f"suffix: {self.suffix_edit.text()} - "
            f"field: {field} "
            f"({'ASPRS-standard' if is_asprs else 'custom extra-byte'})"
        )

        from ..workers.classifier_task import ClassificationTask
        self.task = ClassificationTask(
            self.files.copy(),
            out_dir,
            self.suffix_edit.text(),
            spec.id,
            device,
            field,
            tile_enabled=self.tile_check.isChecked(),
            tile_auto=self.tile_auto_check.isChecked(),
            tile_size_m=self.tile_size_spin.value(),
            tile_buffer_m=self.tile_buffer_spin.value(),
            tile_streaming=self.tile_streaming_check.isChecked(),
            units_override=self.units_combo.currentData() or "auto",
        )
        self.task.progressChanged.connect(self._on_progress)
        self.task.taskCompleted.connect(self._on_task_completed)
        self.task.taskTerminated.connect(self._on_task_terminated)

        QgsApplication.taskManager().addTask(self.task)

    def _on_progress(self, p):
        self.progress_bar.setValue(int(p))
        if self.task and self.task.current_file_name:
            self.current_file_label.setText(
                f"Processing: {self.task.current_file_name} "
                f"({self.task.files_processed + 1}/{len(self.task.files)})"
            )

    def _on_task_completed(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setValue(100)
        count = len(self.task.output_files) if self.task else 0
        self.current_file_label.setText(
            f"Complete - {count} file(s) processed")
        self._log(f"Classification complete: {count} file(s) processed")

        if self.load_result_check.isChecked() and self.task:
            self._load_output_layers(self.task.output_files)

        self.iface.messageBar().pushMessage(
            PLUGIN_NAME,
            f"Classification complete: {count} file(s) processed",
            level=Qgis.MessageLevel.Success, duration=8,
        )

    def _on_task_terminated(self):
        self.run_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        if self.task and self.task.error_message:
            self.current_file_label.setText("Error - see Log")
            self._log(f"ERROR: {self.task.error_message}")
            self.iface.messageBar().pushMessage(
                PLUGIN_NAME,
                f"Classification failed: {self.task.error_message[:200]}",
                level=Qgis.MessageLevel.Critical, duration=0,
            )
        else:
            self.current_file_label.setText("Cancelled")
            self._log("Classification cancelled by user.")

    def _cancel(self):
        if self.task:
            self.task.cancel()
            self.current_file_label.setText("Cancelling...")

    # ------------------------------------------------------------------
    # Output layer loading
    # ------------------------------------------------------------------

    def _load_output_layers(self, output_files):
        try:
            from qgis.core import QgsPointCloudLayer
        except ImportError:
            log_warning(
                "QgsPointCloudLayer not available in this QGIS version")
            self.message_bar.pushMessage(
                "Cannot auto-load: QgsPointCloudLayer not available in "
                "this QGIS build. Drag the file from the file manager.",
                Qgis.MessageLevel.Warning, duration=8,
            )
            return

        # Flush any pending UI / filesystem events so the just-written
        # LAZ file is fully visible to QGIS providers.
        QgsApplication.processEvents()

        failures: list[tuple[Path, str]] = []
        loaded = 0
        for output_path in output_files:
            if not output_path.exists():
                failures.append((output_path, "file not found on disk"))
                log_error(f"Output file missing after write: {output_path}")
                continue

            layer = QgsPointCloudLayer(
                str(output_path), output_path.stem, "pdal"
            )
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                # Attach a 3D renderer so 3D Map Views show the points
                # correctly instead of as a flat 2D sprite.
                from ..utils.helpers import enable_point_cloud_3d_rendering
                enable_point_cloud_3d_rendering(layer)
                log_info(f"Loaded layer: {output_path.stem}")
                loaded += 1
                continue

            # Try to capture the actual provider error
            err_summary = "unknown (PDAL provider could not open the file)"
            try:
                err = layer.error()
                if hasattr(err, "summary") and err.summary():
                    err_summary = err.summary()
            except Exception:
                pass
            failures.append((output_path, err_summary))
            log_warning(
                f"Failed to load '{output_path.name}' as point cloud "
                f"layer: {err_summary}"
            )

        if failures:
            head = failures[0]
            extra = (
                f" (+{len(failures) - 1} more)" if len(failures) > 1 else ""
            )
            self.message_bar.pushMessage(
                f"Output ready but auto-load failed for "
                f"'{head[0].name}'{extra}: {head[1]}. "
                "You can drag the file into QGIS manually.",
                Qgis.MessageLevel.Warning, duration=0,
            )
        elif loaded:
            self.message_bar.pushMessage(
                f"Loaded {loaded} classified layer(s) into the project.",
                Qgis.MessageLevel.Success, duration=5,
            )

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------

    def _log(self, message: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{ts}] {message}")

    def _on_log_message(self, message, tag, level):
        if tag == LOG_TAG:
            self._log(message)
