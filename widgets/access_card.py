"""Access notice for a classifier the free plugin does not run.

Shown in the dock's classifier strip when the selected classifier has an
``access`` text (core/registry.py): a title, the sentence saying where
access is granted, and one button that opens that page (Maven) in the
browser. Nothing opens by itself and the card cannot be hidden: it is the
run gate of that classifier, not an advertisement, so the "hide course
invitation" setting of the cohort card does not apply to it.
"""
from urllib.parse import urlencode

from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtGui import QDesktopServices
from qgis.PyQt.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

_ACCENT = "#FFA500"  # the wire / power line colour of the class tables


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
        self.setStyleSheet(
            "QFrame#accessCard {"
            " border: 1px solid palette(mid);"
            f" border-left: 3px solid {_ACCENT};"
            " border-radius: 4px; }"
        )
        self._classifier = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 6, 6)
        layout.setSpacing(4)

        self.title = QLabel("")
        self.title.setWordWrap(True)
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        layout.addWidget(self.title)

        self.text = QLabel("")
        self.text.setWordWrap(True)
        layout.addWidget(self.text)

        row = QHBoxLayout()
        row.addStretch(1)
        self.button = QPushButton("")
        self.button.setAutoDefault(False)
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
        self.title.setText(f"{classifier.display_name}: access through the Cohort")
        self.text.setText(classifier.access or classifier.access_message())
        self.button.setText(classifier.access_button + " ↗")
        self.button.setToolTip(f"Opens {classifier.access_url} in your browser.")
        self.show()

    def _open(self):
        if self._classifier is not None:
            open_access_page(self._classifier)
