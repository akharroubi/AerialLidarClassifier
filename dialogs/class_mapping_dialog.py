"""Edit the LAS classification output codes for the Mobile Mapping model."""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHeaderView, QLabel,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..utils.class_mapping import output_model_spec


class ClassMappingDialog(QDialog):
    def __init__(self, spec, output_codes=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mobile Mapping (MMS) output class codes")
        self.resize(620, 590)
        self._defaults = spec
        selected = output_model_spec(spec, output_codes)
        layout = QVBoxLayout(self)
        label = QLabel(
            f"{spec.display_name}\n\n"
            "Choose each class's output code for the LAS classification field (0–255). "
            "The values are saved for this model. Repeated codes merge classes.\n"
            "Codes above 31 automatically upgrade legacy LAS files to LAS 1.4. "
            "Use codes 64–255 for your own class definitions.\n\n"
            "Mobile Mapping writes these codes to the existing LAS classification field. "
            "Model IDs 1–9 identify the classes in this table; no extra label field is added."
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        self.table = QTableWidget(len(spec.class_mapping), 4)
        self.table.setHorizontalHeaderLabels(["Model ID", "Mobile Mapping class", "Default", "Output code"])
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self._spins = {}
        for row, (class_id, info) in enumerate(spec.class_mapping.items()):
            for column, value in enumerate((class_id, info.name, info.asprs_code)):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(row, column, item)
            spin = QSpinBox()
            spin.setRange(0, 255)
            spin.setValue(selected.class_mapping[class_id].asprs_code)
            spin.setToolTip(f"Output classification code for {info.name}")
            self.table.setCellWidget(row, 3, spin)
            self._spins[class_id] = spin
        layout.addWidget(self.table, 1)
        reset = QPushButton("Reset to model defaults")
        reset.clicked.connect(self._reset_defaults)
        layout.addWidget(reset)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _reset_defaults(self):
        for class_id, spin in self._spins.items():
            spin.setValue(self._defaults.class_mapping[class_id].asprs_code)

    def output_codes(self):
        return {class_id: spin.value() for class_id, spin in self._spins.items()}
