"""Build an optional QGIS COPC viewing companion without changing the LAS.

Some QGIS PDAL providers reject LAS 1.4 Extra Bytes during their repeated
header preview. QGIS's bundled untwine reads the same file correctly and
the native COPC provider can display its indexed output. The classified
LAS/LAZ remains the authoritative result; this optional derivative may
reorder points and normalize metadata. Run this helper in a worker.
"""

import logging
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time
import uuid

from .logger import log_info, log_warning

_LOG = logging.getLogger(__name__)
VIEW_SUFFIX = ".qgis-view.copc.laz"
# untwine indexes roughly 1 M points per second; allow a generous margin so
# only a hung conversion times out (cancel always works meanwhile).
_MIN_TIMEOUT_S = 1800
_POINTS_PER_TIMEOUT_SECOND = 100_000


def is_view_file(path) -> bool:
    """True for a viewing copy written by prepare_qgis_view."""
    return Path(path).name.lower().endswith(VIEW_SUFFIX)


def _view_pattern(primary):
    return re.compile(re.escape(Path(primary).stem) + r"\.[0-9a-f]{12}" + re.escape(VIEW_SUFFIX) + "$",
                      re.IGNORECASE)


def remove_stale_views(primary, keep=None, in_use=()):
    """Delete older viewing copies of ``primary`` that no layer uses.

    Each run writes a new uniquely named copy (an open copy cannot be
    replaced on Windows). Called on the main thread once the new copy is
    loaded: copies still used by a project layer, or locked, stay.
    Returns the removed paths.
    """
    primary = Path(primary)
    pattern = _view_pattern(primary)
    protected = set()
    for path in [keep, *in_use]:
        if path:
            try:
                protected.add(os.path.normcase(str(Path(path).resolve())))
            except OSError:
                _LOG.debug("Ignored non-fatal error", exc_info=True)
    removed = []
    try:
        candidates = list(primary.parent.iterdir())
    except OSError:
        return removed
    for candidate in candidates:
        if not pattern.match(candidate.name):
            continue
        try:
            if os.path.normcase(str(candidate.resolve())) in protected:
                continue
            candidate.unlink()
            removed.append(candidate)
        except OSError:
            _LOG.debug("Viewing copy kept (in use or locked): %s", candidate, exc_info=True)
    if removed:
        log_info(f"Removed {len(removed)} older viewing cop{'y' if len(removed) == 1 else 'ies'} "
                 f"of {primary.name}.")
    return removed


def _untwine_path():
    from qgis.core import QgsApplication
    name = "untwine.exe" if sys.platform == "win32" else "untwine"
    return Path(QgsApplication.libexecPath()) / name


def _error_tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 4096))
        return stream.read(4096).decode("utf-8", errors="replace").strip()


def _class_counts(path, laspy_module, cancel_callback):
    """Verify class populations without retaining or reordering the cloud."""
    import numpy as np
    counts = np.zeros(256, dtype=np.int64)
    total = 0
    with laspy_module.open(str(path)) as reader:
        declared = int(reader.header.point_count)
        for chunk in reader.chunk_iterator(250_000):
            if cancel_callback():
                raise InterruptedError("Compatible QGIS view preparation was cancelled")
            counts += np.bincount(np.asarray(chunk.classification), minlength=256)
            total += len(chunk)
    if total != declared:
        raise ValueError("Compatible QGIS view verification found a truncated point cloud")
    return counts


def prepare_qgis_view(source, *, laspy_module=None, cancel_callback=None,
                      info_callback=None, warning_callback=None,
                      timeout_seconds=None):
    """Return a viewing path, falling back to the untouched primary output.

    Only LAS 1.4+ outputs containing Extra Bytes need the workaround. Each
    companion has a unique name and is published atomically after count,
    extra-dimension and classification checks. Cancellation or conversion
    failure removes temporary files and leaves the primary result intact.
    This must be called only when the user requested loading the result.
    ``timeout_seconds`` defaults to 30 minutes, more for very large files.
    """
    source = Path(source)
    info = info_callback or log_info
    warning = warning_callback or log_warning
    cancelled = cancel_callback or (lambda: False)
    process = None
    try:
        if cancelled():
            return source
        if laspy_module is None:
            import laspy as laspy_module
        with laspy_module.open(str(source)) as reader:
            header = reader.header
            extra_names = {dim.name for dim in header.point_format.extra_dimensions}
            if (header.version.major, header.version.minor) < (1, 4) or not extra_names:
                return source
            if any(getattr(v, "user_id", "") == "copc" and v.record_id == 1 for v in header.vlrs):
                return source
            count = int(header.point_count)
        executable = _untwine_path()
        if not executable.is_file():
            raise FileNotFoundError(
                f"QGIS's bundled untwine executable was not found at {executable}. "
                "Repair the QGIS installation to enable compatible COPC views."
            )
        if timeout_seconds is None:
            timeout_seconds = max(_MIN_TIMEOUT_S, count / _POINTS_PER_TIMEOUT_SECOND)
        if timeout_seconds <= 0:
            raise ValueError("COPC view timeout must be positive")
        # Unique names cannot collide with another input or an old QGIS index.
        view = source.with_name(f"{source.stem}.{uuid.uuid4().hex[:12]}{VIEW_SUFFIX}")
        info(
            f"Preparing compatible COPC view for {source.name}. The primary classified "
            "LAS/LAZ remains unchanged; the view may reorder points and normalize metadata."
        )
        from . import proc
        # Removed in the finally clause below, after the converter stopped; a
        # file still held by an antivirus scanner must not turn a published
        # view into a failure (TemporaryDirectory would raise on cleanup).
        work = Path(tempfile.mkdtemp(prefix=".qgis-view-", dir=str(source.parent)))
        try:
            temporary = work / "view.copc.laz"
            stderr = work / "stderr.log"
            command = [str(executable), "--files", str(source.resolve()),
                       "--output_dir", str(temporary), "--temp_dir", str(work / "index"), "--stats"]
            try:
                process = proc.Background(command, str(work / "stdout.log"), str(stderr))
                deadline = time.monotonic() + timeout_seconds
                while not process.wait(0.2):
                    if cancelled():
                        raise InterruptedError("Compatible QGIS view preparation was cancelled")
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"COPC view conversion exceeded {timeout_seconds:g} seconds")
                if cancelled():
                    raise InterruptedError("Compatible QGIS view preparation was cancelled")
                if process.returncode != 0:
                    raise RuntimeError(_error_tail(stderr) or f"untwine exited with code {process.returncode}")
            finally:
                if process is not None and process.running():
                    process.stop()
            with laspy_module.open(str(temporary)) as reader:
                view_header = reader.header
                if int(view_header.point_count) != count:
                    raise ValueError("COPC view point count differs from the primary classified output")
                if not extra_names.issubset({d.name for d in view_header.point_format.extra_dimensions}):
                    raise ValueError("COPC view is missing input extra dimensions")
                if not any(getattr(v, "user_id", "") == "copc" and v.record_id == 1
                           for v in view_header.vlrs):
                    raise ValueError("untwine did not produce a COPC index")
            import numpy as np
            tolerance = np.maximum(np.abs(header.scales), np.abs(view_header.scales))
            if (not np.isfinite(view_header.mins).all() or not np.isfinite(view_header.maxs).all()
                    or not np.allclose(header.mins, view_header.mins, rtol=0, atol=tolerance)
                    or not np.allclose(header.maxs, view_header.maxs, rtol=0, atol=tolerance)):
                raise ValueError("COPC view coordinate bounds differ from the primary output")
            if not np.array_equal(_class_counts(source, laspy_module, cancelled),
                                  _class_counts(temporary, laspy_module, cancelled)):
                raise ValueError("COPC view classification counts differ from the primary output")
            if cancelled():
                raise InterruptedError("Compatible QGIS view preparation was cancelled")
            if view.exists():
                raise FileExistsError(f"Refusing to replace an existing viewing file: {view}")
            os.replace(str(temporary), str(view))
        finally:
            if process is not None and process.running():
                process.stop()
            shutil.rmtree(str(work), ignore_errors=True)
        info(f"Compatible QGIS view ready: {view.name}. Primary classified output: {source.name}.")
        return view
    except InterruptedError as exc:
        warning(f"{exc}. The classified output is ready and unchanged: {source}")
    except Exception as exc:
        warning(
            f"Compatible COPC view could not be prepared: {exc}. The classified output "
            f"is ready and unchanged at {source}. View it in a LAS/LAZ viewer or convert "
            "a copy to COPC using QGIS point-cloud tools."
        )
    return source
