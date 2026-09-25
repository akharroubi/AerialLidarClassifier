"""Optional course invitation. Opens Maven only after a user's click.

One setting controls every in-app placement: hiding the card in the
panel also hides the course button of the "classification complete"
message. The plugin menu entry and the About dialog stay available.
"""
from urllib.parse import urlencode

from qgis.core import QgsSettings
from qgis.PyQt.QtCore import QUrl, Qt
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout,
)

from ..config import SETTINGS_PREFIX

COHORT_URL = "https://maven.com/geomatics/qgis3d"
COHORT_TITLE = "LiDAR Point Clouds Processing in QGIS"
COHORT_PITCH = (
    "Learn classification, terrain models, COPC and 3D editing live "
    "with Abderrazzaq Kharroubi, the author of this plugin."
)
_HIDE_KEY = f"{SETTINGS_PREFIX}/hide_cohort_card"
_ACCENT = "#228B22"  # vegetation green of the plugin icon


def cohort_url(placement):
    # Constant campaign tags only: no identifiers, filenames or usage telemetry.
    return COHORT_URL + "?" + urlencode({
        "utm_source": "aerial_lidar_classifier", "utm_medium": "plugin",
        "utm_campaign": "qgis3d", "utm_content": placement,
    })


def open_cohort(placement="menu"):
    return QDesktopServices.openUrl(QUrl(cohort_url(placement)))


def cohort_hidden():
    """True once the user has hidden the invitation."""
    try:
        return bool(QgsSettings().value(_HIDE_KEY, False, type=bool))
    except Exception:
        return False


class CohortCard(QFrame):
    """Course invitation.

    ``placement="panel"`` is the compact block under the Run button
    (two short text lines beside the button); any other placement shows
    the full title and pitch.
    """

    def __init__(self, placement="panel", dismissible=True, parent=None):
        super().__init__(parent)
        self.setObjectName("cohortCard")
        self.setStyleSheet(
            "QFrame#cohortCard {"
            " border: 1px solid palette(mid);"
            f" border-left: 3px solid {_ACCENT};"
            " border-radius: 4px; }"
        )
        compact = placement == "panel"
        layout = QHBoxLayout(self) if compact else QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.setSpacing(8 if compact else 6)

        title = QLabel("Go further with LiDAR in QGIS" if compact else COHORT_TITLE)
        title.setToolTip(COHORT_TITLE + "\n" + COHORT_PITCH)
        title.setWordWrap(True)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        note = QLabel("Live cohort by the plugin author. Optional, paid."
                      if compact else
                      "Live cohort taught by the plugin author. Optional "
                      "paid course; the plugin stays free.")
        note.setToolTip("The course is optional and paid. The plugin is and "
                        "stays free.")
        note.setWordWrap(True)
        small = note.font()
        small.setPointSizeF(max(7.0, small.pointSizeF() * 0.9))
        note.setFont(small)
        button = QPushButton("See the syllabus ↗" if compact
                             else "View syllabus && next cohort ↗")
        button.setAutoDefault(False)
        button.setToolTip("Opens the course page on Maven in your browser.")
        button.clicked.connect(lambda: open_cohort(placement))
        close = None
        if dismissible:
            close = QToolButton()
            close.setText("×")
            close.setAutoRaise(True)
            close.setToolTip(
                "Hide this course invitation. The course stays listed in "
                "the plugin menu and in About."
            )
            close.setAccessibleName("Hide course invitation")
            close.clicked.connect(self._dismiss)

        if compact:
            # One block: the two text lines, then the button and the x.
            text = QVBoxLayout()
            text.setSpacing(1)
            text.addWidget(title)
            text.addWidget(note)
            layout.addLayout(text, 1)
            layout.addWidget(button, 0, Qt.AlignmentFlag.AlignVCenter)
            if close is not None:
                layout.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
        else:
            top = QHBoxLayout()
            top.addWidget(title, 1)
            if close is not None:
                top.addWidget(close, 0, Qt.AlignmentFlag.AlignTop)
            layout.addLayout(top)
            pitch = QLabel(COHORT_PITCH)
            pitch.setWordWrap(True)
            layout.addWidget(pitch)
            bottom = QHBoxLayout()
            bottom.addWidget(note, 1)
            bottom.addWidget(button, 0, Qt.AlignmentFlag.AlignBottom)
            layout.addLayout(bottom)

        if dismissible and cohort_hidden():
            self.hide()

    def _dismiss(self):
        QgsSettings().setValue(_HIDE_KEY, True)
        self.hide()
