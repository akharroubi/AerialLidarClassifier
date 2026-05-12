"""
Background file information loader thread.
"""

from pathlib import Path
from qgis.PyQt.QtCore import QThread, pyqtSignal


class FileInfoLoader(QThread):
    """Load file information in background to avoid blocking UI."""
    info_ready = pyqtSignal(Path, int, float)  # filepath, point_count, size_mb
    error = pyqtSignal(Path, str)

    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            from ..utils.las_utils import get_las_info
            point_count, file_size = get_las_info(self.filepath)
            self.info_ready.emit(self.filepath, point_count, file_size)
        except Exception as e:
            self.error.emit(self.filepath, str(e))
