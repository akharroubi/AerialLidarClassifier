"""Optional course invitation. Opens Maven only after a user's click."""
from urllib.parse import urlencode

from qgis.core import QgsSettings
from qgis.PyQt.QtCore import QUrl, Qt
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout

from ..config import SETTINGS_PREFIX

COHORT_URL = "https://maven.com/geomatics/qgis3d"
COHORT_TITLE = "LiDAR Point Clouds Processing in QGIS"
_HIDE_KEY = f"{SETTINGS_PREFIX}/hide_cohort_card"


def cohort_url(placement):
    # Constant campaign tags only: no identifiers, filenames or usage telemetry.
    return COHORT_URL + "?" + urlencode({
        "utm_source": "aerial_lidar_classifier", "utm_medium": "plugin",
        "utm_campaign": "qgis3d", "utm_content": placement,
    })


def open_cohort(placement="menu"):
    return QDesktopServices.openUrl(QUrl(cohort_url(placement)))


class CohortCard(QFrame):
    def __init__(self, placement="panel", dismissible=True, parent=None):
        super().__init__(parent)
        self.setObjectName("cohortCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 9)
        layout.setSpacing(5)
        row = QHBoxLayout()
        compact = placement == "panel"
        eyebrow = QLabel("LIVE COHORT · WITH THE PLUGIN AUTHOR")
        eyebrow.setWordWrap(True)
        eyebrow.setStyleSheet("font-size: 10px; color: palette(text);")
        row.addWidget(eyebrow, 1)
        if dismissible:
            close = QToolButton()
            close.setText("×")
            close.setAutoRaise(True)
            close.setToolTip("Hide this course invitation. It remains available in the plugin menu and About.")
            close.setAccessibleName("Hide course invitation")
            close.clicked.connect(self._dismiss)
            row.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(row)
        title = QLabel("Go further with LiDAR in QGIS" if compact else COHORT_TITLE)
        title.setToolTip(COHORT_TITLE)
        title.setWordWrap(True)
        font = title.font(); font.setBold(True); title.setFont(font)
        layout.addWidget(title)
        description = QLabel("Learn classification, terrain models, COPC and 3D editing live with Abderrazzaq Kharroubi.")
        description.setWordWrap(True)
        if not compact:
            layout.addWidget(description)
        else:
            description.deleteLater()
        button = QPushButton("View syllabus && next cohort ↗")
        button.setStyleSheet("font-weight: 600; padding: 4px;")
        button.setAutoDefault(False)
        button.setToolTip("Optional paid training on Maven. Opens in your browser.")
        button.clicked.connect(lambda: open_cohort(placement))
        layout.addWidget(button)
        note = QLabel("Optional paid course · The plugin remains free")
        note.setWordWrap(True)
        note.setStyleSheet("font-size: 10px; color: palette(text);")
        layout.addWidget(note)
        if dismissible and QgsSettings().value(_HIDE_KEY, False, type=bool):
            self.hide()

    def _dismiss(self):
        QgsSettings().setValue(_HIDE_KEY, True)
        self.hide()
