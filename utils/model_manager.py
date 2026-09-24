"""Model weight discovery, download, import and cache management.

One ``ModelManager`` per model (``core.registry.ModelSpec``). Weights live
under the QGIS user profile, ``AerialLidarClassifier/models/<model id>/``.
Downloads are atomic (write to a temporary file, then rename) so a failure
never leaves truncated weights on disk, every candidate URL is tried in
order, and the SHA-256 declared in the registry is verified before the
file is accepted. ``import_file`` does the same for a file the user
already has (an offline machine, or a release that is not published yet).
"""

import hashlib
import shutil
from pathlib import Path
from typing import Callable, Iterable, Optional

from qgis.core import (
    QgsApplication,
    QgsBlockingNetworkRequest,
    QgsSettings,
)
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from ..config import SETTINGS_PREFIX
from ..core.registry import ModelSpec, get_model
from .logger import log_error, log_info, log_warning


_CHUNK = 1024 * 256


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class ModelManager:
    """Manage the weights of one model."""

    PLUGIN_FOLDER_NAME = "AerialLidarClassifier"

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @classmethod
    def for_id(cls, model_id: str) -> "ModelManager":
        return cls(get_model(model_id))

    # ------------------------------------------------------------------
    # Paths and availability
    # ------------------------------------------------------------------

    @staticmethod
    def models_root() -> Path:
        profile_dir = Path(QgsApplication.qgisSettingsDirPath())
        root = profile_dir / ModelManager.PLUGIN_FOLDER_NAME / "models"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def model_dir(self) -> Path:
        directory = self.models_root() / self.spec.id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def get_model_path(self) -> Path:
        path = self.model_dir() / self.spec.weights_filename
        if not path.exists():
            # v1.0 kept the single model directly under models/; adopt it.
            legacy = self.models_root() / self.spec.weights_filename
            if legacy.exists():
                try:
                    legacy.replace(path)
                    log_info(f"Moved {legacy.name} into models/{self.spec.id}/")
                except OSError as exc:
                    log_warning(f"Could not move legacy model file: {exc}")
                    return legacy
        return path

    def get_config_path(self) -> Path:
        """The bundled model-configuration file."""
        return self.spec.config_path

    def is_model_available(self) -> bool:
        return self.get_model_path().exists() and self.get_config_path().exists()

    def get_model_size_mb(self) -> float:
        path = self.get_model_path()
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
        return 0.0

    # ------------------------------------------------------------------
    # URL handling
    # ------------------------------------------------------------------

    def _url_setting_key(self) -> str:
        return f"{SETTINGS_PREFIX}/model_url/{self.spec.id}"

    def get_model_url(self) -> str:
        """Primary URL: a user override (QgsSettings) or the registry's first."""
        s = QgsSettings()
        default = self.spec.weights_urls[0] if self.spec.weights_urls else ""
        return s.value(self._url_setting_key(), default)

    def set_model_url(self, url: str) -> None:
        QgsSettings().setValue(self._url_setting_key(), url)

    def get_candidate_urls(self, override: Optional[str] = None) -> list:
        """Ordered URLs to try: the override, else the setting then the registry."""
        seen: set = set()
        ordered: list = []
        candidates: Iterable[str] = (
            [override] if override else
            [self.get_model_url(), *self.spec.weights_urls]
        )
        for url in candidates:
            if not url:
                continue
            url = url.strip()
            if url and url not in seen:
                seen.add(url)
                ordered.append(url)
        return ordered

    # ------------------------------------------------------------------
    # Download / import
    # ------------------------------------------------------------------

    def _verify_and_promote(self, tmp_path: Path, origin: str):
        """SHA-256 check then atomic rename onto the final path."""
        expected = (self.spec.weights_sha256 or "").lower().strip()
        if expected:
            got = _sha256_of(tmp_path)
            if got != expected:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
                return False, (
                    f"SHA-256 mismatch for {origin} (expected "
                    f"{expected[:12]}..., got {got[:12]}...). The file is "
                    "not the released model; it was discarded."
                )
            log_info(f"{self.spec.short_name}: SHA-256 verified.")
        final_path = self.get_model_path()
        if final_path.exists():
            final_path.unlink()
        tmp_path.replace(final_path)
        size_mb = final_path.stat().st_size / (1024 * 1024)
        log_info(f"{self.spec.short_name}: weights ready ({size_mb:.1f} MB) from {origin}")
        return True, ""

    def download_model(
        self,
        url: Optional[str] = None,
        progress_callback: Callable[[int, int], None] = None,
    ):
        """Download the weights, trying every candidate URL in turn.

        Uses :class:`QgsBlockingNetworkRequest` so the request honours the
        user's QGIS proxy / authentication / certificate settings.
        Returns ``(success, error_message)``.
        """
        urls = self.get_candidate_urls(url)
        if not urls:
            return False, "No model URL configured."

        final_path = self.get_model_path()
        tmp_path = final_path.with_suffix(final_path.suffix + ".part")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError as exc:
                log_warning(f"Could not delete stale partial download: {exc}")

        last_error = ""
        for idx, current_url in enumerate(urls, start=1):
            log_info(
                f"Downloading {self.spec.short_name} weights, candidate "
                f"{idx}/{len(urls)}: {current_url}"
            )
            ok, msg = self._download_one(current_url, tmp_path, progress_callback)
            if not ok:
                last_error = msg
                log_warning(f"Candidate {idx} failed: {msg}")
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                continue
            ok, msg = self._verify_and_promote(tmp_path, current_url)
            if ok:
                return True, ""
            last_error = msg
            log_warning(f"Candidate {idx}: {msg}")

        return False, last_error or "All download URLs failed."

    @staticmethod
    def _download_one(
        url: str,
        tmp_path: Path,
        progress_callback: Callable[[int, int], None] = None,
    ):
        """Download a single URL to ``tmp_path`` via the QGIS network stack."""
        try:
            request = QgsBlockingNetworkRequest()
            err = request.get(QNetworkRequest(QUrl(url)))
            try:
                no_error = QgsBlockingNetworkRequest.ErrorCode.NoError
            except AttributeError:
                no_error = QgsBlockingNetworkRequest.NoError
            if err != no_error:
                return False, f"Network error: {request.errorMessage() or err}"

            reply = request.reply()
            content = reply.content()
            total = len(content)
            if total == 0:
                return False, "Downloaded file is empty"

            # QgsBlockingNetworkRequest returns the whole body at once.
            if progress_callback:
                progress_callback(total, total)

            with open(str(tmp_path), "wb") as fh:
                fh.write(content.data())

            if not tmp_path.exists() or tmp_path.stat().st_size == 0:
                return False, "Downloaded file is empty"
            return True, ""

        except Exception as exc:  # pragma: no cover - surfaced to user
            log_error(f"Unexpected error downloading {url}: {exc}")
            return False, f"Error: {exc}"

    def import_file(self, source):
        """Adopt a weights file the user already has (copied, then verified)."""
        source = Path(source)
        if not source.is_file():
            return False, f"File not found: {source}"
        final_path = self.get_model_path()
        tmp_path = final_path.with_suffix(final_path.suffix + ".part")
        try:
            shutil.copyfile(str(source), str(tmp_path))
        except OSError as exc:
            return False, f"Could not copy the file: {exc}"
        return self._verify_and_promote(tmp_path, source.name)

    def delete_model(self) -> None:
        path = self.get_model_path()
        if path.exists():
            path.unlink()
            log_info(f"{self.spec.short_name}: weights deleted from cache.")
