"""Main dock panel for the Aerial LiDAR Classifier plugin.

The panel is a QDockWidget intended to live on the right side of the
QGIS main window (like Processing or the Layer Styling dock). It uses
native QGIS widgets where possible (QgsFileWidget, QgsCollapsibleGroupBox,
QgsMessageBar) and stays out of the way when unused.
"""

import html
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from qgis.PyQt.QtCore import QSize, Qt, QUrl, pyqtSignal
from qgis.PyQt.QtGui import QDesktopServices, QIcon, QPalette
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
from ..core.cuda_allocator import (
    MODE_AUTO as ALLOCATOR_AUTO,
    MODE_LABELS as ALLOCATOR_LABELS,
    MODES as ALLOCATOR_MODES,
    SETTING_NAME as ALLOCATOR_SETTING,
)
from ..core.registry import (
    CLASSIFIERS, DEFAULT_CLASSIFIER_ID, MODELS, classifier_for_model, get_classifier,
    get_model, select_engine,
)
from ..utils.class_mapping import (
    asprs_code_name, output_codes_json, output_model_spec, supports_output_codes,
)
from ..utils.helpers import get_gpu_info, truncate_name
from ..utils.las_units import UNIT_OVERRIDES
from ..utils.las_utils import find_las_files
from ..utils.logger import LOG_TAG, log_error, log_info, log_warning, message_log_signal
from ..utils.model_manager import ModelManager
from ..widgets.drag_drop_list import DragDropList
from ..workers.file_loader import FileInfoLoader
import logging

_LOG = logging.getLogger(__name__)


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


def _secondary_text(widget) -> str:
    """Muted text colour derived from the theme's own text colour.

    palette(mid) is nearly invisible on dark QGIS themes; the window text
    colour at 65 % opacity reads as secondary on light and dark alike.
    """
    colour = widget.palette().color(QPalette.ColorRole.WindowText)
    return (f"color: rgba({colour.red()}, {colour.green()}, "
            f"{colour.blue()}, 0.65);")


# Status colours readable on light and dark QGIS themes.
_OK = "#2e7d32"
_WARN = "#b26a00"
_ERROR = "#c62828"


def _format_elapsed(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds} s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} min {seconds:02d} s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


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
        self._running = False
        # Text -> copies waiting for their twin (see _on_log_message).
        self._log_twins = {}
        self._run_started = 0.0
        self._loaders: list[FileInfoLoader] = []
        self._auto_output_set = False
        # Weights of the selected model present on disk, and the model
        # able to run on the current compute device.
        self.model_ready = False
        self.model_gate_ok = True
        self.model_gate_message = ""
        # Set when the selected classifier runs a fallback engine here.
        self.engine_note = ""

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
        self._connect_log()

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
            message_log_signal().disconnect(self._on_log_message)
        except Exception:
            _LOG.debug("Ignored non-fatal error", exc_info=True)
        self._log_connected = False

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
                _LOG.debug("Ignored non-fatal error", exc_info=True)
        self._loaders.clear()

        # Cancel any running classification task and disconnect its
        # signals so completion callbacks do not reach into a deleted
        # dock object after we are gone, and its last messages (it may
        # still be unloading the model) do not enter the next run's log.
        # A run started after the dock reopens waits for this one to end
        # (core.run_lock).
        if self.task is not None:
            try:
                if not self.task.isCanceled():
                    self.task.cancel()
            except Exception:
                _LOG.debug("Ignored non-fatal error", exc_info=True)
            for signal_name in (
                "progressChanged",
                "taskCompleted",
                "taskTerminated",
                    "log_message"):
                try:
                    getattr(self.task, signal_name).disconnect()
                except Exception:
                    _LOG.debug("Ignored non-fatal error", exc_info=True)
            self.task = None
            # The dock is only hidden: when it is shown again it must be
            # able to start a new run.
            self._running = False
            self.cancel_btn.setEnabled(False)
            self.current_file_label.setText("Cancelled when the panel was closed.")
            self._update_run_button()

        self.closed.emit()
        super().closeEvent(event)

    def showEvent(self, event):  # noqa: N802 - Qt API
        # closeEvent disconnects the log; reconnect when the dock reopens.
        self._connect_log()
        super().showEvent(event)

    def _connect_log(self):
        if not getattr(self, "_log_connected", False):
            message_log_signal().connect(self._on_log_message)
            self._log_connected = True

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
            settings_version = 2
        # 1.3.3: the buffer is automatic (0, the model's own context). A
        # saved 50 m is the old default, not a choice: it becomes automatic;
        # any other saved value is kept (and raised to the model's need).
        if settings_version < 3:
            if s.value(f"{SETTINGS_PREFIX}/tile_buffer_m", None, type=float) == 50.0:
                s.setValue(f"{SETTINGS_PREFIX}/tile_buffer_m", TILE_DEFAULT_BUFFER_M)
            s.setValue(f"{SETTINGS_PREFIX}/settings_version", 3)

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
        # Not restored: a forced unit is meant for the files at hand, and a
        # remembered one would silently apply to later files in other units.
        self._saved_units = "auto"
        self._saved_allocator = str(
            s.value(f"{SETTINGS_PREFIX}/{ALLOCATOR_SETTING}", ALLOCATOR_AUTO) or ALLOCATOR_AUTO
        )
        if self._saved_allocator not in ALLOCATOR_MODES:
            self._saved_allocator = ALLOCATOR_AUTO
        self._saved_classifier_id = str(s.value(f"{SETTINGS_PREFIX}/classifier_id", "") or "")
        # 1.1 and 1.2 saved the model (engine) id; used once to pick its classifier.
        self._saved_model_id = str(s.value(f"{SETTINGS_PREFIX}/model_id", "") or "")
        self._airborne_field_name = str(s.value(
            f"{SETTINGS_PREFIX}/airborne_field_name", self._saved_field or "classification"
        ) or "classification")
        self._output_field_model_id = None
        self._output_codes_by_model = {}
        for spec in MODELS:
            if not supports_output_codes(spec):
                continue
            saved = s.value(f"{SETTINGS_PREFIX}/output_codes/{spec.id}", "")
            try:
                selected = output_model_spec(spec, saved)
            except ValueError as exc:
                log_warning(f"Saved {spec.display_name} output codes were invalid; using defaults. {exc}")
                selected = output_model_spec(spec)
            self._output_codes_by_model[spec.id] = {
                key: info.asprs_code for key, info in selected.class_mapping.items()
            }

    def _save_settings(self):
        s = QgsSettings()
        s.setValue(f"{SETTINGS_PREFIX}/suffix", self.suffix_edit.text())
        s.setValue(
            f"{SETTINGS_PREFIX}/field_name",
            self.field_edit.text().strip() or "classification",
        )
        s.setValue(f"{SETTINGS_PREFIX}/airborne_field_name", self._airborne_field_name)
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
        s.setValue(f"{SETTINGS_PREFIX}/{ALLOCATOR_SETTING}",
                   str(self.allocator_combo.currentData() or ALLOCATOR_AUTO))
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
            f"{SETTINGS_PREFIX}/classifier_id", self.model_combo.currentData() or "",
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
        splitter.addWidget(log_panel)

        # Default split: parameters get ~75% of the splitter, log ~25%.
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([640, 90])
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
        """Two rows: the classifier selector with its weights action, then a
        one-line status (weights, device gate, engine used here)."""
        strip = QFrame()
        strip.setFrameShape(QFrame.Shape.StyledPanel)
        strip.setObjectName("AerialLidarStatusStrip")
        outer = QVBoxLayout(strip)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(4)

        # Row 1: classifier selector, one per acquisition type (the combo
        # takes the width).
        row = QHBoxLayout()
        row.setSpacing(6)
        row.addWidget(QLabel("Classifier:"))
        self.model_combo = QComboBox()
        for classifier in CLASSIFIERS:
            self.model_combo.addItem(classifier.display_name, classifier.id)
        self.model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_combo.setMinimumContentsLength(10)
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        row.addWidget(self.model_combo, 1)

        # Weights download by themselves (Setup, then the first run of a
        # model). Importing a file is only for machines without internet.
        self.import_btn = QToolButton()
        self.import_btn.setIcon(
            _qgis_icon(
                "mActionFileOpen.svg",
                QStyle.StandardPixmap.SP_DialogOpenButton))
        self.import_btn.setToolTip(
            "Offline machines only: import a weights file you already have "
            "for the selected model. The file is copied and its SHA-256 "
            "verified. With internet access the weights download by "
            "themselves."
        )
        self.import_btn.setAutoRaise(True)
        self.import_btn.clicked.connect(self._import_model_file)
        row.addWidget(self.import_btn)
        outer.addLayout(row)

        # Row 2: status, device.
        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        self.model_label = QLabel("")
        self.model_label.setTextFormat(Qt.TextFormat.RichText)
        self.model_label.setWordWrap(True)
        status_row.addWidget(self.model_label, 1)
        status_row.addWidget(self._vline())
        self.device_label = QLabel("Device: detecting...")
        self.device_label.setStyleSheet(_secondary_text(self.device_label))
        status_row.addWidget(self.device_label)
        outer.addLayout(status_row)

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
        # spelling QGIS 4 keeps); the scoped class enum as a fallback.
        for candidate in (
            lambda: Qgis.LayerFilter.PointCloudLayer,
            lambda: QgsMapLayerProxyModel.Filter.PointCloudLayer,
        ):
            try:
                self.layer_combo.setFilters(candidate())
                break
            except Exception:
                _LOG.debug("Ignored non-fatal error", exc_info=True)
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
        self.file_list.setMinimumHeight(96)
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

        # Toolbar and the file count share one row (saves vertical room).
        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        tool_row.addWidget(toolbar)
        tool_row.addStretch()
        self.file_stats = QLabel("No files")
        self.file_stats.setStyleSheet(_secondary_text(self.file_stats))
        tool_row.addWidget(self.file_stats)
        lay.addLayout(tool_row)

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
            "The file is upgraded to LAS 1.4 only when a class code needs "
            "it.\n"
            "- Any other name: an extra-byte field holding the raw model "
            "class IDs (cars, trucks and fences stay distinct). The "
            "input's own 'classification' is kept unchanged. Standard LAS "
            "dimension names (X, intensity, red...) are refused."
        )
        self._airborne_field_tooltip = self.field_edit.toolTip()
        form.addRow("Field:", self.field_edit)

        self.mms_codes_row = QWidget()
        codes_layout = QHBoxLayout(self.mms_codes_row)
        codes_layout.setContentsMargins(0, 0, 0, 0)
        codes_layout.addWidget(QLabel("Class codes:"))
        self.mms_codes_btn = QPushButton("Edit output codes…")
        self.mms_codes_btn.clicked.connect(self._edit_output_codes)
        codes_layout.addWidget(self.mms_codes_btn, 1)
        form.addRow(self.mms_codes_row)
        self.mms_codes_row.hide()
        self.field_edit.textChanged.connect(self._update_output_codes_button)

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
        self.gpu_detail.setStyleSheet(_secondary_text(self.gpu_detail))
        lay.addWidget(self.gpu_detail)

        clear_gpu_btn = QPushButton("Clear GPU memory")
        clear_gpu_btn.clicked.connect(self._clear_gpu_memory)
        lay.addWidget(clear_gpu_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        allocator_form = QFormLayout()
        allocator_form.setSpacing(6)
        allocator_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.allocator_combo = QComboBox()
        for key in ALLOCATOR_MODES:
            self.allocator_combo.addItem(ALLOCATOR_LABELS[key], key)
        saved_allocator = self.allocator_combo.findData(self._saved_allocator)
        self.allocator_combo.setCurrentIndex(saved_allocator if saved_allocator >= 0 else 0)
        self.allocator_combo.setToolTip(
            "How PyTorch reserves GPU memory for the LitePT-L models.\n\n"
            "Automatic: PyTorch's standard allocator on Windows (some GPU "
            "drivers refuse expandable segments there) and on shared vGPU "
            "profiles; expandable segments elsewhere, with a switch to the "
            "standard allocator if the driver refuses them during a run. "
            "Both give the same labels; expandable segments reserve less GPU "
            "memory. The run log says which one was used ('CUDA memory:').\n\n"
            "Expandable segments forces them, also on Windows: if a run then "
            "fails with 'CUDA driver error', go back to Automatic. Applies to "
            "the dock and to Processing."
        )
        self.allocator_combo.currentIndexChanged.connect(self._on_allocator_changed)
        allocator_form.addRow("GPU memory allocator:", self.allocator_combo)
        lay.addLayout(allocator_form)

        lay.addWidget(self._hline())

        # ---- Performance / Tiling ----------------------------------------
        lay.addWidget(self._section_header("Performance / Tiling"))

        self.tile_check = QCheckBox(
            "Tile large files (process in spatial chunks)"
        )
        self.tile_check.setChecked(self._saved_tile_enabled)
        self.tile_check.setToolTip(
            "Splits the input into spatial tiles with a buffer overlap, runs "
            "inference per tile and merges the predictions back. Tiles denser "
            "than the free memory allows are split further.\n\n"
            "Files that do not fit in RAM are streamed from disk automatically "
            "(tiles kept in temporary files in the output folder, removed at "
            "the end). The log says when that happens.\n\n"
            "Unticked, each file is classified in one piece; a file too large "
            "for the free memory is refused with a message asking to tick "
            "this option."
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
            "Side length of each square tile, in metres (files in feet are "
            "converted). Used only when 'Auto-size' is off."
        )
        perf_form.addRow("Tile size:", self.tile_size_spin)

        self.tile_buffer_spin = QDoubleSpinBox()
        self.tile_buffer_spin.setRange(0.0, 5_000.0)
        self.tile_buffer_spin.setDecimals(0)
        self.tile_buffer_spin.setSingleStep(5.0)
        self.tile_buffer_spin.setSuffix(" m")
        # 0 shows as "Automatic": the context the model needs.
        self.tile_buffer_spin.setSpecialValueText("Automatic")
        self.tile_buffer_spin.setValue(self._saved_tile_buffer_m)
        self.tile_buffer_spin.setToolTip(
            "Context around each tile; predictions in the buffer are "
            "discarded.\n\n"
            "Automatic (recommended) uses the context the model needs: "
            "91.5 m for Airborne LitePT-L, 35 m for Mobile "
            "mapping and Railway (less on small GPUs), 34 m for SegFormer "
            "3D. A larger value is used as given; a smaller one is raised "
            "to the model's need."
        )
        perf_form.addRow("Buffer:", self.tile_buffer_spin)

        lay.addLayout(perf_form)

        streaming_note = QLabel(
            "With tiling, files larger than the free memory are streamed from "
            "disk automatically.")
        streaming_note.setWordWrap(True)
        streaming_note.setStyleSheet(_secondary_text(streaming_note))
        lay.addWidget(streaming_note)

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
            "or feet when the header is missing or wrong.\n\n"
            "A forced unit applies to every file of the run (the log warns "
            "about each file whose CRS says otherwise) and returns to "
            "auto-detect when QGIS restarts."
        )
        self.units_combo.currentIndexChanged.connect(self._update_units_note)
        units_form.addRow("Units:", self.units_combo)
        lay.addLayout(units_form)

        self._update_tile_controls_enabled()
        return group

    def _on_allocator_changed(self, _index: int) -> None:
        """Saved at once: Processing reads the same setting at its next run."""
        mode = str(self.allocator_combo.currentData() or ALLOCATOR_AUTO)
        QgsSettings().setValue(f"{SETTINGS_PREFIX}/{ALLOCATOR_SETTING}", mode)
        log_info(f"GPU memory allocator set to '{mode}' (applies at the next run).")

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
        self.log_text.setMinimumHeight(50)
        lay.addWidget(self.log_text, 1)
        return panel

    # ---- Footer ----------------------------------------------------------
    def _build_footer(self) -> QWidget:
        bar = QFrame()
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        lay = QVBoxLayout(bar)
        lay.setContentsMargins(8, 6, 8, 8)
        lay.setSpacing(4)

        # Idle: what is still missing before Run can start. Running: the
        # file being processed.
        self.current_file_label = QLabel("Ready")
        self.current_file_label.setWordWrap(True)
        self.current_file_label.setStyleSheet(_secondary_text(self.current_file_label))
        lay.addWidget(self.current_file_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        # Hidden until the first run: an idle "0 %" bar only takes room.
        self.progress_bar.hide()
        lay.addWidget(self.progress_bar)

        # Visible next to Run while a unit is forced (Advanced parameters
        # may be collapsed).
        self.units_note = QLabel()
        self.units_note.setWordWrap(True)
        self.units_note.setStyleSheet(f"color: {_WARN};")
        self.units_note.hide()
        lay.addWidget(self.units_note)

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
            short = name.replace("NVIDIA ", "").replace("GeForce ", "")
            self.device_label.setText(f"GPU: {truncate_name(short, 22)}")
            self.device_label.setToolTip(name)
        else:
            self.gpu_check.setEnabled(False)
            self.gpu_check.setChecked(False)
            self.gpu_check.setText("Use GPU (not available)")
            reason = self.gpu_info.get("reason", "no_gpu")
            if reason == "no_cuda":
                self.gpu_detail.setText(
                    "PyTorch CPU build: inference runs on the CPU. On a computer with an "
                    "NVIDIA GPU, Repair dependencies installs the GPU build.")
            elif reason == "no_gpu":
                self.gpu_detail.setText(
                    "PyTorch has CUDA support but sees no GPU: update the NVIDIA driver, "
                    "then restart QGIS. Inference runs on the CPU meanwhile.")
            elif reason == "torch_import_failed":
                self.gpu_detail.setText(
                    "PyTorch could not be loaded (see the Log). Use Repair dependencies, "
                    "then restart QGIS.")
            else:
                self.gpu_detail.setText(f"GPU probe failed: {reason}. See the plugin log for details.")
            self.device_label.setText("Device: CPU")

    def _current_device(self) -> str:
        """"cuda", "mps" or "cpu" as the run would use it right now."""
        if (
            self.gpu_check.isEnabled() and self.gpu_check.isChecked()
            and self.gpu_info.get("available")
        ):
            return str(self.gpu_info.get("backend", "cuda"))
        return "cpu"

    def _current_classifier(self):
        try:
            return get_classifier(self.model_combo.currentData())
        except KeyError:
            return get_classifier(DEFAULT_CLASSIFIER_ID)

    def _current_spec(self):
        """The engine the selected classifier uses on this computer now."""
        return select_engine(self._current_classifier(), self._current_device(),
                             self._model_runs_here)

    def _select_initial_model(self):
        """Restore the saved classifier (1.1 and 1.2 saved a model id).

        Mobile Mapping and Railway stay selected on a computer that cannot
        run them: the gate explains what is missing instead of silently
        switching to another acquisition type.
        """
        chosen = None
        for saved, lookup in ((self._saved_classifier_id, get_classifier),
                              (self._saved_model_id, classifier_for_model)):
            if saved and chosen is None:
                try:
                    chosen = lookup(saved)
                except KeyError:
                    chosen = None
        chosen = chosen or get_classifier(DEFAULT_CLASSIFIER_ID)
        index = self.model_combo.findData(chosen.id)
        self.model_combo.blockSignals(True)
        self.model_combo.setCurrentIndex(max(index, 0))
        self.model_combo.blockSignals(False)

    def _on_model_changed(self, _index=None):
        self._check_model()

    def _update_output_codes_button(self, _text=None):
        spec = self._current_spec()
        editable = supports_output_codes(spec)
        previous_id = self._output_field_model_id
        previous_editable = previous_id is not None and supports_output_codes(get_model(previous_id))
        self._output_field_model_id = spec.id
        self.field_edit.blockSignals(True)
        if editable:
            # Remember the airborne field only when leaving an airborne
            # model; between two editable-code models it reads
            # 'classification', which must not replace the user's name.
            if previous_id is not None and previous_id != spec.id and not previous_editable:
                self._airborne_field_name = self.field_edit.text().strip() or "classification"
            self.field_edit.setText("classification")
            self.field_edit.setToolTip(
                f"{spec.short_name} writes only to the existing LAS classification field. "
                "Use Edit output codes to choose the code of each class."
            )
        else:
            if previous_editable:
                self.field_edit.setText(self._airborne_field_name)
            self._airborne_field_name = self.field_edit.text().strip() or "classification"
            self.field_edit.setToolTip(self._airborne_field_tooltip)
        self.field_edit.blockSignals(False)
        self.field_edit.setEnabled(not editable)
        self.mms_codes_row.setVisible(editable)
        self.mms_codes_btn.setEnabled(True)
        if editable:
            selected = output_model_spec(spec, self._output_codes_by_model.get(spec.id))
            self.mms_codes_btn.setToolTip("\n".join(
                f"{info.name}: {info.asprs_code}" for info in selected.class_mapping.values()
            ))

    def _edit_output_codes(self):
        spec = self._current_spec()
        if not supports_output_codes(spec):
            return
        from ..dialogs.class_mapping_dialog import ClassMappingDialog
        dialog = ClassMappingDialog(spec, self._output_codes_by_model.get(spec.id), self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected = output_model_spec(spec, dialog.output_codes())
            self._output_codes_by_model[spec.id] = {
                key: info.asprs_code for key, info in selected.class_mapping.items()
            }
            QgsSettings().setValue(
                f"{SETTINGS_PREFIX}/output_codes/{spec.id}", output_codes_json(selected)
            )
            self._update_output_codes_button()
            self._check_model()

    def _on_gpu_toggled(self, _checked=None):
        # The device gate of the selected model depends on the checkbox.
        self._check_model()

    def _check_model(self):
        classifier = self._current_classifier()
        device = self._current_device()
        spec = select_engine(classifier, device, self._model_runs_here)
        self._update_output_codes_button()
        manager = ModelManager(spec)
        self.import_btn.setToolTip(
            f"Import a weights file for {spec.display_name} (offline computers); "
            "its SHA-256 is verified. Weights also download automatically on first use."
        )

        if manager.is_model_available():
            self.model_ready = True
            colour, status = _OK, "Ready"
        else:
            # Not an error: the run downloads the weights first.
            self.model_ready = False
            colour = _WARN
            status = (f"Weights download on first run "
                      f"({spec.weights_size_mb:.0f} MB)")

        self.model_gate_ok, gate_colour, gate_status, gate_message = self._gate(spec, device)
        self.engine_note = ""
        if not self.model_gate_ok:
            colour, status = gate_colour, gate_status
            self.model_gate_message = gate_message
        else:
            self.model_gate_message = ""
            if spec is not classifier.primary:
                # Airborne on a computer where LitePT-L cannot run: say which
                # engine runs instead, and why.
                _ok, _colour, why, reason = self._gate(classifier.primary, device)
                self.engine_note = (
                    f"{classifier.primary.display_name} cannot run here ({why}). "
                    f"{reason} The {classifier.display_name} uses the "
                    f"{spec.display_name} engine meanwhile ({len(spec.class_mapping)} classes)."
                )
                status = f"{status} · {spec.short_name} engine ({spec.resolution})"
                if self.gpu_info.get("available") and self.gpu_info.get("backend", "cuda") == "cuda":
                    # An NVIDIA GPU is there: the faster engine is one step away.
                    colour = _WARN

        self.model_label.setText(
            f"<span style='color:{colour};'>●</span> {status}")
        message = self.model_gate_message or self.engine_note
        classes = self._refresh_classifier_tooltips(device)
        self.model_label.setToolTip(
            (f"{html.escape(message, quote=False)}<hr>" if message else "") + classes)
        if message and message != getattr(self, "_last_model_gate_message", ""):
            # The log is selectable/copyable, unlike a transient tooltip.
            self._log(f"{classifier.display_name}: {message}")
        self._last_model_gate_message = message
        self._update_run_button()

    def _refresh_classifier_tooltips(self, device):
        """Class list of every classifier in the selector's item tooltips.

        Returns the selected classifier's tooltip (also the selector's own).
        Each list follows the engine that runs on this computer and the
        output codes currently saved for the editable models.
        """
        current = ""
        for index, classifier in enumerate(CLASSIFIERS):
            text = self._classifier_tooltip(classifier, device)
            self.model_combo.setItemData(index, text, Qt.ItemDataRole.ToolTipRole)
            if classifier.id == self.model_combo.currentData():
                current = text
        self.model_combo.setToolTip(current)
        return current

    def _classifier_tooltip(self, classifier, device):
        spec = select_engine(classifier, device, self._model_runs_here)
        selected = (output_model_spec(spec, self._output_codes_by_model.get(spec.id))
                    if supports_output_codes(spec) else spec)
        rows = "".join(
            f"<tr><td><span style='color:{info.color};'>&#9632;</span>&nbsp;"
            f"{html.escape(info.name)}</td><td align='right'>&nbsp;&nbsp;{info.asprs_code}</td>"
            f"<td>&nbsp;&nbsp;{html.escape(asprs_code_name(info.asprs_code))}</td></tr>"
            for info in selected.class_mapping.values()
        )
        note = ("Codes are yours to change: Output &gt; Class codes."
                if supports_output_codes(spec)
                else "Written to the LAS classification field (ASPRS codes).")
        return (
            f"<b>{html.escape(classifier.display_name)}</b><br>"
            f"{html.escape(classifier.description)}<br>"
            f"On this computer: {html.escape(spec.display_name)}, "
            f"{len(spec.class_mapping)} classes.<br>"
            "<table cellspacing='0' cellpadding='1'>"
            "<tr><th align='left'>Class</th><th align='right'>&nbsp;&nbsp;Code</th>"
            "<th align='left'>&nbsp;&nbsp;ASPRS meaning</th></tr>"
            f"{rows}</table>{note}"
        )

    def _gate(self, spec, device):
        """(runs, colour, status, explanation) for ``spec`` on ``device``."""
        ok = spec.supports_device(device)
        dependency_message = ""
        if ok and spec.family == "litept":
            from ..utils.backend_readiness import litept_dependency_status
            ok, dependency_message = litept_dependency_status()
        if ok:
            return True, _OK, "", ""
        if dependency_message:
            return False, _ERROR, "LitePT dependencies missing", dependency_message
        if (self.gpu_info.get("available")
                and self.gpu_info.get("backend", "cuda") == "cuda"
                and not self.gpu_check.isChecked()):
            return False, _WARN, "GPU use is switched off", (
                f"{self.gpu_info.get('name', 'An NVIDIA CUDA GPU')} is detected. "
                f"Enable 'Use GPU' in Advanced parameters to run {spec.display_name}."
            )
        if sys.platform == "darwin" or self.gpu_info.get("backend") == "mps":
            return False, _ERROR, "Needs an NVIDIA GPU", (
                f"{spec.display_name} runs on NVIDIA CUDA GPUs only, which macOS "
                "does not support."
            )
        if self.gpu_info.get("reason") == "no_cuda":
            return False, _ERROR, "PyTorch is CPU-only", (
                f"{spec.display_name} runs on NVIDIA CUDA GPUs only, and the "
                "installed PyTorch build has no CUDA support. On a computer with "
                f"an NVIDIA GPU, use Plugins > {PLUGIN_NAME} > Repair dependencies "
                "(GPU installation), then restart QGIS."
            )
        if self.gpu_info.get("reason") == "no_gpu":
            return False, _ERROR, "No GPU visible", (
                f"{spec.display_name} runs on NVIDIA CUDA GPUs only. PyTorch has "
                "CUDA support but sees no GPU: update the NVIDIA driver, then "
                "restart QGIS."
            )
        if self.gpu_info.get("reason") == "torch_import_failed":
            return False, _ERROR, "PyTorch not loaded", (
                "PyTorch could not be loaded in QGIS (see the Log). Use "
                f"Plugins > {PLUGIN_NAME} > Repair dependencies, then restart QGIS."
            )
        return False, _ERROR, "Needs an NVIDIA GPU", (
            f"{spec.display_name} runs on NVIDIA CUDA GPUs only."
        )

    @staticmethod
    def _model_runs_here(spec, device) -> bool:
        """Device supported and, for LitePT-L, its native libraries import."""
        if not spec.supports_device(device):
            return False
        if spec.family == "litept":
            from ..utils.backend_readiness import litept_dependency_status
            return litept_dependency_status()[0]
        return True

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
            _LOG.debug("Ignored non-fatal error", exc_info=True)
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
            item = QListWidgetItem(f"{name}  -  {path.name}")
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

    def _missing_before_run(self) -> str:
        """First thing the user still has to do, or "" when Run can start."""
        if not self.files:
            return "Add LAS / LAZ / COPC files to the list above."
        if not self.output_widget.filePath():
            return "Choose an output folder."
        if not self.model_gate_ok:
            return self.model_gate_message
        return ""

    def _update_run_button(self):
        missing = self._missing_before_run()
        self.run_btn.setEnabled(not missing and not self._running)
        self.run_btn.setToolTip(missing)
        if self._running:
            return
        if missing:
            self.current_file_label.setText(missing)
            return
        classifier = self._current_classifier()
        spec = self._current_spec()
        device = self._current_device()
        where = "CPU" if device == "cpu" else f"GPU ({device.upper()})"
        engine = "" if spec is classifier.primary else f", {spec.short_name} engine"
        text = (f"Ready: {len(self.files)} file(s), {classifier.display_name}{engine} "
                f"on {where}.")
        if not self.model_ready:
            text += (f" Its weights ({spec.weights_size_mb:.0f} MB) download "
                     "automatically first.")
        self.current_file_label.setText(text)

    # ------------------------------------------------------------------
    # Run / cancel
    # ------------------------------------------------------------------

    def _run(self):
        if self._running:
            return
        out_dir = self.output_widget.filePath()
        if not out_dir:
            self.message_bar.pushMessage(
                "Choose an output folder first.", Qgis.MessageLevel.Warning, duration=4
            )
            return

        self._running = True
        self._run_started = time.monotonic()
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
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
        classifier = self._current_classifier()
        spec = select_engine(classifier, device, self._model_runs_here)
        if not spec.supports_device(device):
            self.message_bar.pushMessage(
                self.model_gate_message, Qgis.MessageLevel.Warning, duration=8,
            )
            self._running = False
            self.cancel_btn.setEnabled(False)
            self._update_run_button()
            return
        field = self.field_edit.text().strip() or "classification"
        is_asprs = field.lower() == "classification"
        self._log(
            f"Starting the {classifier.display_name} ({spec.display_name}) on "
            f"{device.upper()} with {len(self.files)} file(s)"
        )
        if spec is not classifier.primary and self.engine_note:
            self._log(self.engine_note)
        self._log(
            f"Output: {out_dir} - "
            f"suffix: {self.suffix_edit.text()} - "
            f"field: {field} "
            f"({'ASPRS-standard' if is_asprs else 'custom extra-byte'})"
        )
        if supports_output_codes(spec):
            selected = output_model_spec(spec, self._output_codes_by_model.get(spec.id))
            self._log(f"{spec.short_name} output codes: {output_codes_json(selected)}")

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
            units_override=self.units_combo.currentData() or "auto",
            output_codes=self._output_codes_by_model.get(spec.id),
            prepare_qgis_view=self.load_result_check.isChecked(),
        )
        self.task.progressChanged.connect(self._on_progress)
        self.task.taskCompleted.connect(self._on_task_completed)
        self.task.taskTerminated.connect(self._on_task_terminated)
        self._log_twins.clear()
        self.task.log_message.connect(self._on_task_log_message)

        QgsApplication.taskManager().addTask(self.task)

    def _on_progress(self, p):
        self.progress_bar.setValue(int(p))
        if self.task and getattr(self.task, "status_text", ""):
            self.current_file_label.setText(self.task.status_text)
        elif self.task and self.task.current_file_name:
            self.current_file_label.setText(
                f"Processing: {self.task.current_file_name} "
                f"({self.task.files_processed + 1}/{len(self.task.files)})"
            )

    def _on_task_completed(self):
        self._running = False
        self.cancel_btn.setEnabled(False)
        # Refresh the model status (the run may have fetched the weights).
        self._check_model()
        self.progress_bar.setValue(100)
        outputs = list(self.task.output_files) if self.task else []
        count = len(outputs)
        elapsed = _format_elapsed(time.monotonic() - self._run_started)
        summary = f"Classification complete: {count} file(s) in {elapsed}"
        self.current_file_label.setText(summary + ".")
        self._log(summary)

        if self.load_result_check.isChecked() and outputs:
            views = getattr(self.task, "view_files", {}) if self.task else {}
            # A viewing copy is loaded under the classified file's name.
            self._load_output_layers([views.get(path, path) for path in outputs],
                                     names=[path.stem for path in outputs])
            self._remove_stale_views(outputs, views)

        self._push_completion_message(summary, outputs)
        self._push_unit_warnings()

    def _update_units_note(self, *_):
        """Show, next to Run, that the input units are forced."""
        if not hasattr(self, "units_note"):
            return
        key = self.units_combo.currentData() or "auto"
        if key == "auto":
            self.units_note.hide()
            return
        self.units_note.setText(
            f"Input units forced to {self.units_combo.currentText()} for every file "
            "(Advanced parameters > Input units). Files whose CRS says otherwise are "
            "reported in the log. Back to auto-detect when QGIS restarts.")
        self.units_note.show()

    def _push_unit_warnings(self):
        """After a run: the files whose CRS contradicts the forced units."""
        warnings = list(getattr(self.task, "unit_warnings", []) or []) if self.task else []
        if not warnings:
            return
        names = ", ".join(w.split(":", 1)[0] for w in warnings[:3])
        more = f" and {len(warnings) - 3} more" if len(warnings) > 3 else ""
        self.iface.messageBar().pushMessage(
            PLUGIN_NAME,
            f"Check the units: the CRS of {names}{more} says another unit than the forced "
            f"input units ({self.units_combo.currentText()}). See the log.",
            level=Qgis.MessageLevel.Warning, duration=0)

    def _push_completion_message(self, summary, outputs):
        """Success message with 'Open folder' and, unless hidden, the course."""
        bar = self.iface.messageBar()
        folder = outputs[0].parent if outputs else None
        try:
            from ..widgets.cohort_card import cohort_hidden, open_cohort
            item = bar.createMessage(PLUGIN_NAME, summary)
            if folder is not None:
                open_btn = QPushButton("Open folder")
                open_btn.clicked.connect(
                    lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))))
                item.layout().addWidget(open_btn)
            if not cohort_hidden():
                course_btn = QPushButton("Go further: live course ↗")
                course_btn.setToolTip(
                    "Optional paid cohort taught by the plugin author. "
                    "Opens Maven in your browser.")
                course_btn.clicked.connect(lambda: open_cohort("success"))
                item.layout().addWidget(course_btn)
            bar.pushWidget(item, Qgis.MessageLevel.Success, 15)
        except Exception:
            bar.pushMessage(PLUGIN_NAME, summary,
                            level=Qgis.MessageLevel.Success, duration=8)

    def _on_task_terminated(self):
        self._running = False
        self.cancel_btn.setEnabled(False)
        self._check_model()
        self._push_unit_warnings()

        if self.task and self.task.error_message:
            self.current_file_label.setText("Error - see the Log below.")
            self._log(f"ERROR: {self.task.error_message}")
            self.iface.messageBar().pushMessage(
                PLUGIN_NAME,
                f"Classification failed: {self.task.error_message[:200]}",
                level=Qgis.MessageLevel.Critical, duration=0,
            )
        elif self.task and self.task.output_files:
            done = list(self.task.output_files)
            self.current_file_label.setText(
                f"Cancelled after {len(done)} file(s) were classified; they are kept.")
            self._log("Classification cancelled by user. Classified files kept: "
                      + ", ".join(str(path) for path in done))
        else:
            self.current_file_label.setText(
                "Cancelled. Existing output files were left unchanged.")
            self._log("Classification cancelled by user.")

    def _cancel(self):
        if self.task:
            self.task.cancel()
            self.current_file_label.setText("Cancelling...")

    # ------------------------------------------------------------------
    # Output layer loading
    # ------------------------------------------------------------------

    def _remove_stale_views(self, outputs, views):
        """Delete viewing copies of earlier runs that no layer still uses."""
        from ..utils.pointcloud_view import remove_stale_views
        in_use = [layer.source() for layer in QgsProject.instance().mapLayers().values()]
        for output in outputs:
            try:
                remove_stale_views(output, keep=views.get(output), in_use=in_use)
            except Exception:
                _LOG.debug("Ignored non-fatal error", exc_info=True)

    def _load_output_layers(self, output_files, names=None):
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
        task = self.task  # snapshot before processing events can start another run
        QgsApplication.processEvents()

        failures: list[tuple[Path, str]] = []
        loaded = 0
        names = list(names) if names is not None else [p.stem for p in output_files]
        for output_path, layer_name in zip(output_files, names):
            if not output_path.exists():
                failures.append((output_path, "file not found on disk"))
                log_error(f"Output file missing after write: {output_path}")
                continue

            layer = QgsPointCloudLayer(
                str(output_path), layer_name,
                "copc" if output_path.name.lower().endswith(".copc.laz") else "pdal"
            )
            if layer.isValid():
                QgsProject.instance().addMapLayer(layer)
                # Attach a 3D renderer so 3D Map Views show the points
                # correctly instead of as a flat 2D sprite.
                from ..utils.helpers import enable_point_cloud_3d_rendering
                standard_field = task is not None and task.field_name.strip().lower() == "classification"
                enable_point_cloud_3d_rendering(
                    layer, class_mapping=task.spec.class_mapping if standard_field else None,
                    classify_2d=bool(standard_field and supports_output_codes(task.spec)),
                )
                log_info(f"Loaded layer: {layer_name} ({output_path.name})")
                loaded += 1
                continue

            # Try to capture the actual provider error
            err_summary = "unknown (PDAL provider could not open the file)"
            try:
                err = layer.error()
                if hasattr(err, "summary") and err.summary():
                    err_summary = err.summary()
            except Exception:
                _LOG.debug("Ignored non-fatal error", exc_info=True)
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

    def _on_log_message(self, message, tag, level, *_format):
        if tag != LOG_TAG:
            return
        if not self._running:
            self._log(message)
            return
        # During a run the task's messages arrive twice: through its own
        # signal and through QgsMessageLog. Show each once, whichever copy
        # comes first (the message log can also miss them entirely).
        pending = self._log_twins.get(message, 0)
        if pending > 0:
            self._log_twins[message] = pending - 1
            return
        self._log_twins[message] = pending - 1
        self._log(message)

    def _on_task_log_message(self, message):
        pending = self._log_twins.get(message, 0)
        if pending < 0:
            self._log_twins[message] = pending + 1
            return
        self._log_twins[message] = pending + 1
        self._log(message)
