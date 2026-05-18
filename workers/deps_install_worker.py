"""Background worker thread for installing plugin dependencies.

Runs the full installation pipeline (download Python, create venv,
install packages) in a separate thread to keep the QGIS UI responsive.
"""

import traceback

from qgis.PyQt.QtCore import QThread, pyqtSignal


class DepsInstallWorker(QThread):
    """Worker thread that installs dependencies in the background.

    Signals:
        progress(int, str): Emitted with (percent, message) during install.
        finished(bool, str): Emitted with (success, message) when done.
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, cuda_enabled: bool = False, parent=None):
        """Initialize the worker.

        Args:
            cuda_enabled: Whether to install CUDA-enabled PyTorch.
            parent: Optional parent QObject.
        """
        super().__init__(parent)
        self._cancelled = False
        self._cuda_enabled = cuda_enabled

    def cancel(self):
        """Request cancellation of the installation."""
        self._cancelled = True

    def run(self):
        """Run the installation pipeline.

        Wipes any pre-existing cache directory first so a "Reinstall
        Dependencies" click always produces a clean install rather than
        layering on top of a venv left behind by a buggy older plugin
        version (the case that motivated the install-marker check).
        """
        try:
            import shutil
            import os

            from ..utils.venv_manager import CACHE_DIR, create_venv_and_install

            if os.path.exists(CACHE_DIR):
                self.progress.emit(
                    1,
                    "Removing previous installation...",
                )
                try:
                    shutil.rmtree(CACHE_DIR)
                except Exception as exc:
                    # Non-fatal: create_venv_and_install will also try
                    # to clean up partial state. Just log to the
                    # progress signal so the user sees what happened.
                    self.progress.emit(
                        1,
                        f"Could not fully remove old cache: {exc}",
                    )

            success, message = create_venv_and_install(
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled,
                cuda_enabled=self._cuda_enabled,
            )
            self.finished.emit(success, message)
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.finished.emit(False, error_msg)
