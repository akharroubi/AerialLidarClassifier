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
    completed = pyqtSignal(bool, str)

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

        Wipes only the venv subdirectory before reinstalling, NOT the
        entire cache directory. We deliberately preserve:

          - ``python_standalone/`` (40 MB) - the portable interpreter
            we already downloaded
          - ``uv/`` (15 MB) - the uv binary we already downloaded
          - ``models/`` and any other user-placed files - some users
            drop the model `.pth` here manually as a proxy / offline
            workaround documented in the README

        Re-downloading those on every "Reinstall Dependencies" click
        would waste bandwidth, time, and (in the models case) the
        user's manual setup. The bug we're recovering from is a stale
        venv, so only the venv needs to go.
        """
        try:
            import shutil
            import os
            import sys

            from ..utils.venv_manager import (
                CACHE_DIR, VENV_DIR, create_venv_and_install,
            )

            if self._cancelled:
                self.completed.emit(False, "Installation cancelled")
                return
            if _plugin_torch_loaded(CACHE_DIR):
                self.completed.emit(False, (
                    "The classifier's AI libraries are already loaded in "
                    "this QGIS session and cannot be replaced while it runs. "
                    "Restart QGIS, then choose Plugins > Aerial LiDAR "
                    "Classifier > Repair dependencies before opening the "
                    "classifier."
                ))
                return

            if os.path.exists(VENV_DIR):
                self.progress.emit(
                    1,
                    "Removing previous virtual environment...",
                )
                try:
                    shutil.rmtree(VENV_DIR)
                except PermissionError as exc:
                    # Almost always Windows holding torch's c10.dll
                    # (or similar) open because QGIS imported torch
                    # from the old venv earlier this session. Layering
                    # a new install on top of a half-wiped venv
                    # produces cryptic "Failed to read metadata" errors
                    # from uv. Fail fast with the right instruction
                    # instead.
                    if sys.platform == "win32":
                        msg = (
                            "Could not remove the previous virtual "
                            "environment because QGIS has loaded its "
                            "DLLs (Windows keeps loaded DLLs locked "
                            "until the process exits). Please close "
                            "QGIS completely, reopen it, then click "
                            "Reinstall Dependencies again. After the "
                            "restart no DLLs are pinned, so the old "
                            "venv can be rebuilt cleanly. Underlying "
                            "error: {}".format(exc)
                        )
                        self.completed.emit(False, msg)
                        return
                    # Non-Windows: continue and let the install layer
                    # surface a clear error if anything goes wrong.
                    self.progress.emit(
                        1,
                        f"Partial cleanup ({exc}); continuing...",
                    )
                except Exception as exc:
                    # Other failures: log and let create_venv_and_install
                    # attempt its own cleanup.
                    self.progress.emit(
                        1,
                        f"Could not fully remove old venv: {exc}",
                    )

            success, message = create_venv_and_install(
                progress_callback=lambda p, m: self.progress.emit(p, m),
                cancel_check=lambda: self._cancelled,
                cuda_enabled=self._cuda_enabled,
            )
            if success and not self._cancelled:
                message = self._download_weights(message)
            self.completed.emit(success, message)
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.completed.emit(False, error_msg)

    def _download_weights(self, message: str) -> str:
        """Fetch the weights of every model that can run here.

        Part of the one-time setup so the first classification starts
        straight away. A failure here does not fail the setup: the first
        run of a model downloads whatever is still missing.
        """
        from ..core.registry import MODELS
        from ..utils.model_manager import ModelManager
        from ..utils.venv_manager import _read_install_marker

        marker = _read_install_marker() or {}
        device = "cuda" if self._cuda_enabled else "cpu"
        wanted = [
            spec for spec in MODELS
            if spec.supports_device(device)
            and (spec.family != "litept" or marker.get("spconv_version"))
        ]
        failed = []
        for spec in wanted:
            if self._cancelled:
                break
            manager = ModelManager(spec)
            if manager.is_model_available():
                continue
            label = f"Downloading the {spec.short_name} weights"
            self.progress.emit(100, f"{label} ({spec.weights_size_mb:.0f} MB)...")

            def progress(received, total, _label=label):
                if total > 0:
                    self.progress.emit(
                        100, f"{_label}: {received / 1048576:.0f} / "
                             f"{total / 1048576:.0f} MB")

            try:
                ok, _msg = manager.ensure_available(
                    progress, lambda: self._cancelled)
            except InterruptedError:
                break
            if not ok:
                failed.append(spec.short_name)
        if failed:
            return (
                f"{message}. The {', '.join(failed)} weights could not be "
                "downloaded now; they will be downloaded when you first run "
                "the classifier."
            )
        return message


def _plugin_torch_loaded(cache_dir: str) -> bool:
    """True when this session imported torch from the plugin's own venv.

    Only then are its DLLs locked and the environment impossible to
    replace; a torch loaded by another plugin from elsewhere is no
    obstacle.
    """
    import os
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return False
    location = getattr(torch, "__file__", None) or ""
    if not location:
        return True  # unknown origin: be safe
    try:
        location = os.path.normcase(os.path.realpath(location))
        cache = os.path.normcase(os.path.realpath(cache_dir))
        return location.startswith(cache + os.sep)
    except Exception:
        return True
