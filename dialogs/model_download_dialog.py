"""Model download dialog: fetches one model's weights with progress."""

from qgis.PyQt.QtCore import QThread, pyqtSignal
from qgis.PyQt.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton,
    QVBoxLayout,
)


class DownloadWorker(QThread):
    """Background thread for one model download."""
    progress = pyqtSignal(int, int)  # bytes_downloaded, total_bytes
    finished = pyqtSignal(bool, str)

    def __init__(self, spec, url, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.url = url

    def run(self):
        from ..utils.model_manager import ModelManager
        success, error = ModelManager(self.spec).download_model(
            self.url,
            progress_callback=lambda dl, total: self.progress.emit(dl, total),
        )
        self.finished.emit(success, error)


class ModelDownloadDialog(QDialog):
    """Dialog for downloading the weights of ``spec``."""

    def __init__(self, spec, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.setWindowTitle(f"Download {spec.display_name}")
        self.setMinimumWidth(520)
        self.worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        header = QLabel(f"Download {self.spec.display_name}")
        font = header.font()
        font.setPointSize(14)
        font.setBold(True)
        header.setFont(font)
        layout.addWidget(header)

        info = QLabel(
            f"{self.spec.description}\n\n"
            f"File: {self.spec.weights_filename} "
            f"(about {self.spec.weights_size_mb:.0f} MB), verified by SHA-256 "
            f"after download. Licence: {self.spec.licence}."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("URL:"))
        self.url_edit = QLineEdit()
        from ..utils.model_manager import ModelManager
        self.url_edit.setText(ModelManager(self.spec).get_model_url())
        url_layout.addWidget(self.url_edit)
        layout.addLayout(url_layout)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

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

        from ..utils.model_manager import ModelManager
        ModelManager(self.spec).set_model_url(url)

        self.download_btn.setEnabled(False)
        self.url_edit.setEnabled(False)
        # The blocking QGIS request cannot be interrupted midway; a
        # dialog closed during the download would otherwise destroy a
        # running thread (a Qt fatal error in v1.0.2).
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Downloading...")
        self.progress_bar.setRange(0, 0)

        self.worker = DownloadWorker(self.spec, url, parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, downloaded, total):
        self.progress_bar.setRange(0, 100)
        if total > 0:
            self.progress_bar.setValue(int(downloaded * 100 / total))
            self.status_label.setText(
                f"Downloaded {downloaded / (1024 * 1024):.1f} / "
                f"{total / (1024 * 1024):.1f} MB, verifying..."
            )
        else:
            self.status_label.setText(
                f"Downloaded {downloaded / (1024 * 1024):.1f} MB, verifying..."
            )

    def _on_finished(self, success, error):
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setRange(0, 100)
        if success:
            self.status_label.setText("Download complete, SHA-256 verified.")
            self.progress_bar.setValue(100)
            self.download_btn.setText("Done")
            self.download_btn.setEnabled(True)
            self.download_btn.clicked.disconnect()
            self.download_btn.clicked.connect(self.accept)
        else:
            self.status_label.setText(f"Failed: {error}")
            self.progress_bar.setValue(0)
            self.download_btn.setText("Retry")
            self.download_btn.setEnabled(True)
            self.url_edit.setEnabled(True)

    def closeEvent(self, event):  # noqa: N802 - Qt API
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait()
        super().closeEvent(event)

    def reject(self):
        if self.worker is not None and self.worker.isRunning():
            return
        super().reject()
