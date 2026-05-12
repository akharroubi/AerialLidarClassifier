"""Model download and cache management.

Downloads model weights on first use and caches them inside the QGIS
user profile directory. Downloads are atomic (write to a temporary file
then rename) so a failure never leaves truncated weights on disk.

If multiple URLs are configured (primary + fallbacks in ``config.py``)
each is tried in order until one succeeds. A SHA-256 hash, when
provided, is verified after every successful download and the file is
rejected on mismatch.
"""

import hashlib
from pathlib import Path
from typing import Callable, Iterable

from qgis.core import (
    QgsApplication,
    QgsBlockingNetworkRequest,
    QgsSettings,
)
from qgis.PyQt.QtCore import QUrl
from qgis.PyQt.QtNetwork import QNetworkRequest

from ..config import (
    DEFAULT_MODEL_URL,
    MODEL_CONFIG_FILENAME,
    MODEL_FALLBACK_URLS,
    MODEL_FILENAME,
    MODEL_SHA256,
    SETTINGS_PREFIX,
)
from .logger import log_error, log_info, log_warning


_CHUNK = 1024 * 256


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(str(path), "rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class ModelManager:
    """Manage model weight discovery, caching and download."""

    PLUGIN_FOLDER_NAME = "AerialLidarClassifier"

    # ------------------------------------------------------------------
    # Paths and availability
    # ------------------------------------------------------------------

    @staticmethod
    def get_model_dir() -> Path:
        profile_dir = Path(QgsApplication.qgisSettingsDirPath())
        model_dir = profile_dir / ModelManager.PLUGIN_FOLDER_NAME / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    @staticmethod
    def get_model_path() -> Path:
        return ModelManager.get_model_dir() / MODEL_FILENAME

    @staticmethod
    def get_config_path() -> Path:
        """Return the bundled model-configuration JSON path."""
        plugin_root = Path(__file__).resolve().parent.parent
        return plugin_root / "core" / MODEL_CONFIG_FILENAME

    @staticmethod
    def is_model_available() -> bool:
        return (
            ModelManager.get_model_path().exists()
            and ModelManager.get_config_path().exists()
        )

    @staticmethod
    def get_model_size_mb() -> float:
        path = ModelManager.get_model_path()
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
        return 0.0

    # ------------------------------------------------------------------
    # URL handling
    # ------------------------------------------------------------------

    @staticmethod
    def get_model_url() -> str:
        """Primary URL: a user override (QgsSettings) or the default."""
        s = QgsSettings()
        return s.value(f"{SETTINGS_PREFIX}/model_url", DEFAULT_MODEL_URL)

    @staticmethod
    def set_model_url(url: str) -> None:
        s = QgsSettings()
        s.setValue(f"{SETTINGS_PREFIX}/model_url", url)

    @staticmethod
    def get_candidate_urls(override: str = None) -> list:
        """Build the ordered list of URLs to try.

        ``override`` (when truthy) takes precedence; otherwise we start
        with the configured primary URL followed by every non-empty
        entry in ``MODEL_FALLBACK_URLS``. Duplicates are removed while
        preserving order.
        """
        seen: set = set()
        ordered: list = []

        candidates: Iterable[str] = (
            [override] if override else
            [ModelManager.get_model_url(), *MODEL_FALLBACK_URLS]
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
    # Download
    # ------------------------------------------------------------------

    @staticmethod
    def download_model(
        url: str = None,
        progress_callback: Callable[[int, int], None] = None,
    ):
        """Download the model weights, trying every configured URL in turn.

        Uses :class:`QgsBlockingNetworkRequest` so the request honours
        the user's QGIS proxy / authentication / certificate settings
        (as recommended in the QGIS plugin guidelines).

        Returns ``(success, error_message)``. On success the file at
        ``get_model_path()`` is guaranteed to match ``MODEL_SHA256`` if
        that constant is set; otherwise the only guarantee is non-zero
        length.
        """
        urls = ModelManager.get_candidate_urls(url)
        if not urls:
            return False, "No model URL configured."

        final_path = ModelManager.get_model_path()
        tmp_path = final_path.with_suffix(final_path.suffix + ".part")

        # Wipe any stale .part residue from a previous failed download.
        if tmp_path.exists():
            try:
                stale_mb = tmp_path.stat().st_size / (1024 * 1024)
            except OSError:
                stale_mb = 0.0
            log_info(
                f"Removing stale partial download "
                f"({stale_mb:.1f} MB): {tmp_path.name}"
            )
            try:
                tmp_path.unlink()
            except OSError as exc:
                log_warning(f"Could not delete stale partial download: {exc}")

        last_error = ""
        for idx, current_url in enumerate(urls, start=1):
            log_info(
                f"Downloading model from candidate {idx}/{len(urls)}: {current_url}"
            )
            ok, msg = ModelManager._download_one(
                current_url, tmp_path, progress_callback
            )
            if not ok:
                last_error = msg
                log_warning(f"Candidate {idx} failed: {msg}")
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                continue

            # Integrity check
            if MODEL_SHA256:
                got = _sha256_of(tmp_path)
                expected = MODEL_SHA256.lower().strip()
                if got != expected:
                    last_error = (
                        f"SHA-256 mismatch (expected {expected[:12]}..., "
                        f"got {got[:12]}...)"
                    )
                    log_warning(
                        f"Candidate {idx}: {last_error}. Discarding."
                    )
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                    continue
                log_info("Model SHA-256 verified.")

            # Promote to the final path atomically.
            if final_path.exists():
                final_path.unlink()
            tmp_path.replace(final_path)
            size_mb = final_path.stat().st_size / (1024 * 1024)
            log_info(f"Model downloaded ({size_mb:.1f} MB) from {current_url}")
            return True, ""

        return False, last_error or "All download URLs failed."

    @staticmethod
    def _download_one(
        url: str,
        tmp_path: Path,
        progress_callback: Callable[[int, int], None] = None,
    ):
        """Download a single URL to ``tmp_path`` via the QGIS network stack.

        QgsBlockingNetworkRequest downloads the body into memory and
        respects the QGIS proxy + authentication + certificate settings.
        For the 18 MB model file the memory cost is negligible.
        """
        try:
            request = QgsBlockingNetworkRequest()
            err = request.get(QNetworkRequest(QUrl(url)))
            if err != QgsBlockingNetworkRequest.NoError:
                return False, f"Network error: {request.errorMessage() or err}"

            reply = request.reply()
            content = reply.content()
            total = len(content)
            if total == 0:
                return False, "Downloaded file is empty"

            # Surface a single progress update (Qgs blocking request does
            # not stream chunks; the read happens fully before we get here).
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

    @staticmethod
    def delete_model() -> None:
        path = ModelManager.get_model_path()
        if path.exists():
            path.unlink()
            log_info("Model weights deleted from cache.")
