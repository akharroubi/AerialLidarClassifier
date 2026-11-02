"""Cohort invitation: "Like the plugin? Help it grow". Opens the page only after a click.

One setting controls every in-app placement: hiding the card in the
panel also hides the Cohort button of the "classification complete"
message. The plugin menu entry and the About dialog stay available.
"""
from urllib.parse import urlencode

from qgis.core import QgsSettings
from qgis.PyQt.QtCore import QUrl, Qt
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton, QVBoxLayout,
)

from ..config import COURSE_TITLE, COURSE_URL, PLUGIN_VERSION, SETTINGS_PREFIX

COHORT_URL = COURSE_URL
COHORT_TITLE = COURSE_TITLE
# The one message of every placement (dock card, About, success bar, menu).
COHORT_HOOK = "Like the plugin? Help it grow"
COHORT_LINE = "The live Cohort funds the next models and releases."
COHORT_BUTTON = "See the Cohort ↗"
COHORT_PITCH = (
    "Classification, terrain models, COPC and 3D editing, live with "
    "Abderrazzaq Kharroubi, the author of this plugin."
)
COHORT_TOOLTIP = f"{COHORT_TITLE}: {COHORT_LINE} Optional and paid; the plugin stays free."
_HIDE_KEY = f"{SETTINGS_PREFIX}/hide_cohort_card"
_ACCENT = "34, 139, 34"  # vegetation green of the plugin icon, as RGB for rgba()


def soft_card_style(object_name, accent_rgb):
    """Minimal, soft look shared by the invitation and the access notice:
    a faint tint, a hairline of the accent, rounded corners, pill buttons.
    Palette-independent, so it reads the same on light and dark themes."""
    return (
        f"QFrame#{object_name} {{"
        f" background: rgba({accent_rgb}, 0.07);"
        f" border: 1px solid rgba({accent_rgb}, 0.28);"
        " border-radius: 8px; }"
        f"QFrame#{object_name} QPushButton {{"
        " background: transparent;"
        f" border: 1px solid rgba({accent_rgb}, 0.6);"
        " border-radius: 11px; padding: 3px 12px; }"
        f"QFrame#{object_name} QPushButton:hover {{"
        f" background: rgba({accent_rgb}, 0.16); }}"
        f"QFrame#{object_name} QPushButton:pressed {{"
        f" background: rgba({accent_rgb}, 0.28); }}"
        f"QFrame#{object_name} QToolButton {{ border: none; background: transparent; }}"
    )


def cohort_url(placement):
    # Constant campaign tags only: no identifiers, filenames or usage telemetry.
    return COHORT_URL + "?" + urlencode({
        "utm_source": "aerial_lidar_classifier", "utm_medium": "plugin",
        "utm_campaign": "qgis3d", "utm_content": placement,
    })


def open_cohort(placement="menu"):
    return QDesktopServices.openUrl(QUrl(cohort_url(placement)))


def cohort_hidden():
    """True when the user hid the invitation in this plugin version.

    The x stores the version it was clicked in, so every new release shows
    the card once more (new models get announced) and the x hides it again
    until the next one. The boolean of 1.1 to 1.3.3 counts as an older
    version: the card comes back with 1.3.4.
    """
    try:
        return str(QgsSettings().value(_HIDE_KEY, "") or "") == PLUGIN_VERSION
    except Exception:
        return False


class CohortCard(QFrame):
    """Cohort invitation.

    ``placement="panel"`` is the compact block under the Run button
    (two short text lines beside the button); any other placement shows
    the title of the Cohort and its pitch as well.
    """

    def __init__(self, placement="panel", dismissible=True, parent=None):
        super().__init__(parent)
        self.setObjectName("cohortCard")
        self.setStyleSheet(soft_card_style("cohortCard", _ACCENT))
        compact = placement == "panel"
        layout = QHBoxLayout(self) if compact else QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(10 if compact else 6)

        title = QLabel(COHORT_HOOK)
        title.setToolTip(COHORT_TOOLTIP)
        title.setWordWrap(True)
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        note = QLabel(COHORT_LINE)
        note.setToolTip(COHORT_TOOLTIP)
        note.setWordWrap(True)
        small = note.font()
        small.setPointSizeF(max(7.0, small.pointSizeF() * 0.9))
        note.setFont(small)
        button = QPushButton(COHORT_BUTTON)
        button.setAutoDefault(False)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setToolTip(f"Opens the Cohort page ({COHORT_URL}) in your browser.")
        button.clicked.connect(lambda: open_cohort(placement))
        close = None
        if dismissible:
            close = QToolButton()
            close.setText("×")
            close.setAutoRaise(True)
            close.setToolTip(
                "Hide this invitation until the next plugin version. The Cohort "
                "stays listed in the plugin menu and in About."
            )
            close.setAccessibleName("Hide Cohort invitation")
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
            pitch = QLabel(f"<b>{COHORT_TITLE}</b>. {COHORT_PITCH}")
            pitch.setWordWrap(True)
            layout.addWidget(pitch)
            bottom = QHBoxLayout()
            bottom.addWidget(note, 1)
            bottom.addWidget(button, 0, Qt.AlignmentFlag.AlignBottom)
            layout.addLayout(bottom)

        if dismissible and cohort_hidden():
            self.hide()

    def _dismiss(self):
        QgsSettings().setValue(_HIDE_KEY, PLUGIN_VERSION)
        self.hide()
