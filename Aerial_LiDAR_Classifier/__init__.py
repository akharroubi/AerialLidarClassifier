"""Aerial LiDAR Classifier - QGIS Plugin.

Deep-learning semantic segmentation of aerial LiDAR point clouds using a
3D SegFormer model from the TreeAIBox project (NRCan, Crown Copyright,
Government of Canada, CC BY-NC 4.0).
"""

import os

from qgis.PyQt.QtCore import QCoreApplication, QTranslator, QLocale, QSettings


PLUGIN_DIR = os.path.dirname(__file__)
_translator = None


def _install_translator() -> None:
    """Load the .qm translation matching the QGIS locale, if available."""
    global _translator

    locale = QSettings().value("locale/userLocale", QLocale().name()) or "en"
    locale = str(locale)[:2]

    qm_path = os.path.join(
        PLUGIN_DIR, "i18n", f"aerial_lidar_classifier_{locale}.qm"
    )
    if not os.path.exists(qm_path):
        return

    translator = QTranslator()
    if translator.load(qm_path):
        QCoreApplication.installTranslator(translator)
        _translator = translator


def classFactory(iface):  # noqa: N802 - required by QGIS API
    _install_translator()
    from .plugin import AerialLidarClassifierPlugin
    return AerialLidarClassifierPlugin(iface)


def tr(message: str, context: str = "AerialLidarClassifier") -> str:
    """Translate a string via Qt's translation system."""
    return QCoreApplication.translate(context, message)
