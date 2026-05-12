"""Drag-and-drop list widget for LAS / LAZ files."""

from qgis.PyQt.QtCore import pyqtSignal
from qgis.PyQt.QtWidgets import QAbstractItemView, QListWidget

from ..utils.las_utils import find_las_files


class DragDropList(QListWidget):
    """A QListWidget that accepts dropped LAS / LAZ files and folders."""

    filesAdded = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setUniformItemSizes(True)

    def dragEnterEvent(self, e):  # noqa: N802 - Qt API
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):  # noqa: N802 - Qt API
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802 - Qt API
        files = []
        for url in e.mimeData().urls():
            files.extend(find_las_files(url.toLocalFile()))
        if files:
            self.filesAdded.emit(files)
        e.acceptProposedAction()
