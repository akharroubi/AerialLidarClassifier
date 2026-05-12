"""
Model download dialog for Aerial LiDAR Classifier.

Downloads the pre-trained model weights from a configurable URL
with progress tracking.
"""

from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QProgressBar
)
from qgis.PyQt.QtCore import QThread, pyqtSignal


class DownloadWorker(QThread):
    """Background thread for model download."""
    progress = pyqtSignal(int, int)  # bytes_downloaded, total_bytes
    finished = pyqtSignal(bool, str)

    def __init__(self, url):
        super().__init__()
        self.url = url

    def run(self):
        from ..utils.model_manager import ModelManager
        success, error = ModelManager.download_model(
            self.url,
            progress_callback=lambda dl, total: self.progress.emit(dl, total)
        )
        self.finished.emit(success, error)


class ModelDownloadDialog(QDialog):
    """Dialog for downloading model weights."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Model")
        self.setMinimumWidth(500)
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        header = QLabel("Download Model Weights")
        font = header.font()
        font.setPointSize(14)
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        info = QLabel(
            "The deep learning model weights need to be downloaded on first use. "
            "The file is approximately 18 MB."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # URL field
        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URL:"))
        self.url_edit = QLineEdit()
        from ..utils.model_manager import ModelManager
        self.url_edit.setText(ModelManager.get_model_url())
        url_layout.addWidget(self.url_edit)
        layout.addLayout(url_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(self.cancel_btn)

        self.download_btn = QPushButton("Download")
        self.download_btn.clicked.connect(self._start_download)
        btn_layout.addWidget(self.download_btn)

        layout.addLayout(btn_layout)

    def _start_download(self):
        url = self.url_edit.text().strip()
        if not url:
            self.status_label.setText("Please enter a URL")
            return

        # Save URL to settings
        from ..utils.model_manager import ModelManager
        ModelManager.set_model_url(url)

        self.download_btn.setEnabled(False)
        self.url_edit.setEnabled(False)
        self.status_label.setText("Downloading...")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        self.worker = DownloadWorker(url)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, downloaded, total):
        if total > 0:
            pct = int(downloaded * 100 / total)
            self.progress_bar.setValue(pct)
            mb_dl = downloaded / (1024 * 1024)
            mb_total = total / (1024 * 1024)
            self.status_label.setText(f"Downloading... {mb_dl:.1f} / {mb_total:.1f} MB")
        else:
            mb_dl = downloaded / (1024 * 1024)
            self.status_label.setText(f"Downloading... {mb_dl:.1f} MB")

    def _on_finished(self, success, error):
        if success:
            self.status_label.setText("Download complete!")
            self.progress_bar.setValue(100)
            self.download_btn.setText("Done")
            self.download_btn.setEnabled(True)
            self.download_btn.clicked.disconnect()
            self.download_btn.clicked.connect(self.accept)
        else:
            self.status_label.setText(f"Failed: {error}")
            self.download_btn.setText("Retry")
            self.download_btn.setEnabled(True)
            self.url_edit.setEnabled(True)
