"""About dialog for the Aerial LiDAR Classifier plugin."""

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import (
    QDialog, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from ..config import PLUGIN_NAME, PLUGIN_VERSION


REPO_URL = "https://github.com/akharroubi/AerialLidarClassifier"
TREEAIBOX_URL = "https://github.com/NRCan/TreeAIBox"
LICENSE_URL = "https://creativecommons.org/licenses/by-nc/4.0/"


class AboutDialog(QDialog):
    """About / credits dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"About {PLUGIN_NAME}")
        self.setMinimumSize(520, 520)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, 1)

        body = QWidget()
        scroll.setWidget(body)

        layout = QVBoxLayout(body)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel(PLUGIN_NAME)
        font = title.font()
        font.setPointSize(16)
        font.setBold(True)
        title.setFont(font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        version = QLabel(f"Version {PLUGIN_VERSION}")
        version.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(version)

        layout.addSpacing(8)

        # Description
        desc = QLabel(
            "Semantic segmentation of aerial LiDAR point clouds "
            "(LAS / LAZ / COPC) with deep-learning models, written to the "
            "standard ASPRS classification codes.\n\n"
            "Two models are available: LitePT-L (default on NVIDIA GPUs) "
            "and the 3D SegFormer (runs on GPU or CPU). Classes without an "
            "ASPRS code (cars, trucks, fences) are written as 1, "
            "Unclassified."
        )
        desc.setWordWrap(True)
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc)

        from ..widgets.cohort_card import CohortCard
        layout.addWidget(CohortCard(placement="about", dismissible=False, parent=body))

        layout.addSpacing(12)

        # Model credit blocks, from the registry so they never go stale.
        from ..core.registry import MODELS
        for spec in MODELS:
            model_block = QLabel(
                "<div style='text-align:center;'>"
                f"<b>{spec.display_name}</b><br>"
                f"{spec.attribution}<br>"
                f"Training data: {spec.training_data}<br>"
                f"<small>{spec.licence}</small><br>"
                f'<a href="{spec.homepage}">{spec.homepage}</a>'
                "</div>"
            )
            model_block.setWordWrap(True)
            model_block.setOpenExternalLinks(True)
            model_block.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextBrowserInteraction
            )
            layout.addWidget(model_block)
            layout.addSpacing(6)

        # License
        license_lbl = QLabel(
            f'<small>Model weights are licensed under <a href="{LICENSE_URL}">'
            "Creative Commons Attribution-NonCommercial 4.0 International "
            "(CC BY-NC 4.0)</a>; the 3D SegFormer weights are the unchanged "
            f'<a href="{TREEAIBOX_URL}">TreeAIBox</a> release.</small>'
        )
        license_lbl.setWordWrap(True)
        license_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        license_lbl.setOpenExternalLinks(True)
        layout.addWidget(license_lbl)

        layout.addSpacing(8)

        # Plugin credit
        plugin_block = QLabel(
            "<div style='text-align:center;'>"
            "<b>QGIS plugin</b><br>"
            "Abderrazzaq Kharroubi<br>"
            "GeoScITY Lab - University of Liege<br>"
            f'<a href="{REPO_URL}">{REPO_URL}</a><br>'
            "<small>Plugin code: GPL-3.0</small>"
            "</div>"
        )
        plugin_block.setWordWrap(True)
        plugin_block.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plugin_block.setOpenExternalLinks(True)
        layout.addWidget(plugin_block)

        # Non-commercial notice for the model
        commercial = QLabel(
            "<div style='text-align:center; color:palette(mid);'>"
            "<small>Both downloaded weight sets are designated CC BY-NC 4.0 "
            "(non-commercial use only). Classifying LiDAR data for "
            "commercial purposes requires a separate licence from the "
            "model authors or a different model.</small>"
            "</div>"
        )
        commercial.setWordWrap(True)
        commercial.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(commercial)

        layout.addStretch()

        # Close button (lives outside the scroll area)
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        outer.addWidget(close)
