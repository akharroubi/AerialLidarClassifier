"""Centralized logging for the Aerial LiDAR Classifier plugin.

All plugin code should use these helpers instead of calling
QgsMessageLog directly. This keeps the log tag consistent and
makes it easy to redirect logging in tests.
"""

import logging
import threading

from qgis.core import QgsMessageLog, Qgis
from ..config import PLUGIN_NAME

LOG_TAG = PLUGIN_NAME
_LOG = logging.getLogger(__name__)

# A callback per thread that receives every message logged by that thread,
# in addition to the QGIS message log. The classification task installs one
# for its worker thread so that its messages reach the dock's log through a
# Qt signal of its own, independent of how QgsMessageLog delivers them (GitHub
# issue 8: on QGIS 4 the dock showed none of the task's lines, because it
# listened to a signal QGIS 4 no longer emits; see message_log_signal).
_sink = threading.local()


def message_log_signal():
    """The QgsMessageLog signal that carries (message, tag, level).

    QGIS 4 emits messageReceivedWithFormat(message, tag, level, format) and
    no longer emits the three-argument messageReceived, which is deprecated
    there: a slot connected to it receives nothing. QGIS 3 has only
    messageReceived. Slots must accept an optional fourth argument.
    """
    from qgis.core import QgsApplication
    log = QgsApplication.messageLog()
    return getattr(log, "messageReceivedWithFormat", None) or log.messageReceived


def set_thread_sink(callback) -> None:
    """Forward this thread's messages to ``callback(message)``; None stops."""
    _sink.callback = callback


def _forward(message: str) -> None:
    callback = getattr(_sink, "callback", None)
    if callback is not None:
        try:
            callback(message)
        except Exception:
            _LOG.debug("Log sink failed", exc_info=True)


def log_info(message: str) -> None:
    QgsMessageLog.logMessage(str(message), LOG_TAG, Qgis.MessageLevel.Info)
    _forward(str(message))


def log_warning(message: str) -> None:
    QgsMessageLog.logMessage(str(message), LOG_TAG, Qgis.MessageLevel.Warning)
    _forward(str(message))


def log_error(message: str) -> None:
    QgsMessageLog.logMessage(str(message), LOG_TAG, Qgis.MessageLevel.Critical)
    _forward(str(message))


def log_success(message: str) -> None:
    QgsMessageLog.logMessage(str(message), LOG_TAG, Qgis.MessageLevel.Success)
    _forward(str(message))
