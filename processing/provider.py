"""Processing provider for Aerial LiDAR Classifier.

Registers the plugin's algorithms with the QGIS Processing framework so
they are available from the Toolbox, the Graphical Modeler and the
qgis_process command-line interface.
"""

import os

from qgis.core import QgsProcessingProvider
from qgis.PyQt.QtGui import QIcon

from ..config import PLUGIN_NAME, PLUGIN_VERSION
from .classify_algorithm import ClassifyLidarAlgorithm


class AerialLidarProvider(QgsProcessingProvider):
    """QGIS Processing provider for the plugin."""

    ID = "aeriallidar"

    def __init__(self):
        super().__init__()

    def loadAlgorithms(self):  # noqa: N802 - QGIS API
        self.addAlgorithm(ClassifyLidarAlgorithm())

    def id(self) -> str:
        return self.ID

    def name(self) -> str:
        return PLUGIN_NAME

    def longName(self) -> str:  # noqa: N802
        return f"{PLUGIN_NAME} (v{PLUGIN_VERSION})"

    def icon(self) -> QIcon:
        icon_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "icon.png"
        )
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        return super().icon()
