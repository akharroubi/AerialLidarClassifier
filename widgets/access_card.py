"""Access notice for a classifier the free plugin does not run.

Shown in the dock's classifier strip when the selected classifier has an
``access`` text (core/registry.py): one sentence saying who the model is
available to, and one button that opens the Cohort page in the browser.
Nothing opens by itself and the card cannot be hidden: it is the run gate
of that classifier, not an invitation, so the "hide Cohort invitation"
setting of the cohort card does not apply to it.
"""
from urllib.parse import urlencode

from qgis.PyQt.QtCore import QUrl, Qt
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

from .cohort_card import soft_card_style

_ACCENT = "217, 130, 43"  # a soft amber: the wire colour of the class tables, toned down


def access_url(classifier):
    """The classifier's access page with constant campaign tags (no identifiers)."""
    return classifier.access_url + "?" + urlencode({
        "utm_source": "aerial_lidar_classifier", "utm_medium": "plugin",
        "utm_campaign": "qgis3d", "utm_content": f"access_{classifier.id}",
    })


def open_access_page(classifier):
    return QDesktopServices.openUrl(QUrl(access_url(classifier)))


class AccessCard(QFrame):
    """One classifier's access notice; ``set_classifier`` swaps the text."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("accessCard")
        self.setStyleSheet(soft_card_style("accessCard", _ACCENT))
        self._classifier = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 8, 8)
        layout.setSpacing(6)

        self.text = QLabel("")
        self.text.setWordWrap(True)
        layout.addWidget(self.text)

        row = QHBoxLayout()
        row.addStretch(1)
        self.button = QPushButton("")
        self.button.setAutoDefault(False)
        self.button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.button.clicked.connect(self._open)
        row.addWidget(self.button)
        layout.addLayout(row)
        self.hide()

    def set_classifier(self, classifier):
        """Show the notice of ``classifier``; hide the card when it is not locked."""
        self._classifier = classifier
        if classifier is None or not classifier.locked:
            self.hide()
            return
        self.text.setText(classifier.access or classifier.access_message())
        self.button.setText(classifier.access_button + " ↗")
        self.button.setToolTip(f"Opens {classifier.access_url} in your browser.")
        self.show()

    def _open(self):
        if self._classifier is not None:
            open_access_page(self._classifier)
