"""Centralized logging for the Aerial LiDAR Classifier plugin.

All plugin code should use these helpers instead of calling
QgsMessageLog directly. This keeps the log tag consistent and
makes it easy to redirect logging in tests.
"""

from qgis.core import QgsMessageLog, Qgis

LOG_TAG = "Aerial LiDAR Classifier"


def log_info(message: str) -> None:
    QgsMessageLog.logMessage(str(message), LOG_TAG, Qgis.MessageLevel.Info)


def log_warning(message: str) -> None:
    QgsMessageLog.logMessage(str(message), LOG_TAG, Qgis.MessageLevel.Warning)


def log_error(message: str) -> None:
    QgsMessageLog.logMessage(str(message), LOG_TAG, Qgis.MessageLevel.Critical)


def log_success(message: str) -> None:
    QgsMessageLog.logMessage(str(message), LOG_TAG, Qgis.MessageLevel.Success)
