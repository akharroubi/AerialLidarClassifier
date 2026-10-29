"""Check native LitePT dependencies before offering Run."""
from functools import lru_cache
import importlib
import sys

from ..config import PLUGIN_NAME


@lru_cache(maxsize=1)
def _check_imports():
    for name in ("spconv.pytorch", "scipy.spatial"):
        try:
            importlib.import_module(name)
        except Exception as exc:
            return False, (
                f"LitePT cannot import {name} in the plugin environment. "
                f"Use Plugins > {PLUGIN_NAME} > Repair dependencies "
                "and restart QGIS. This import failure alone does not show that "
                f"the GPU is unsupported. Details: {type(exc).__name__}: {exc}"
            )
    return True, ""


def _prepare_environment():
    from .venv_manager import ensure_venv_packages_available
    return ensure_venv_packages_available()


def _torch_details():
    """Report the already loaded runtime, without triggering another import."""
    torch = sys.modules.get("torch")
    if torch is None:
        return "PyTorch has not been loaded in QGIS."
    cuda = getattr(getattr(torch, "version", None), "cuda", None)
    details = (f"PyTorch {getattr(torch, '__version__', 'unknown')}; "
               f"CUDA build {cuda or 'CPU-only'}; "
               f"module {getattr(torch, '__file__', 'unknown')}")
    try:
        if torch.cuda.is_available():
            details += f"; GPU {torch.cuda.get_device_name(0)}"
    except Exception as exc:
        details += f"; CUDA probe: {exc}"
    return details


def _numpy_mismatch_hint():
    """Explain an environment built for NumPy 2 inside a QGIS with NumPy 1.x."""
    try:
        from importlib import metadata
        from .venv_manager import (
            _host_numpy_version, _read_install_marker, get_venv_site_packages)
        host = _host_numpy_version()
        if not host or int(host.split(".", 1)[0]) >= 2:
            return ""
        # Packages chosen for this NumPy: a NumPy 2 in the environment is
        # expected where NumPy 1.x has no wheel for the plugin's Python.
        if (_read_install_marker() or {}).get("host_numpy") == host.split("+", 1)[0]:
            return ""
        ours = next((d.version for d in metadata.distributions(
            name="numpy", path=[get_venv_site_packages()])), None)
        if not ours or int(ours.split(".", 1)[0]) < 2:
            return ""
    except Exception:
        return ""
    return (
        f"QGIS provides NumPy {host}, but the plugin environment was installed "
        f"for NumPy {ours}. Restart QGIS, then use Plugins > {PLUGIN_NAME} > "
        "Repair dependencies before opening the classifier: the repair installs "
        "packages that match QGIS's NumPy."
    )


def _spconv_hint():
    """Why spconv is missing on this computer, from the install marker."""
    try:
        from .venv_manager import _read_install_marker, _spconv_is_fallback
        marker = _read_install_marker() or {}
    except Exception:
        return ""
    if marker.get("cuda_mode") != "cuda":
        return ""
    error = marker.get("spconv_error")
    if error:
        build = marker.get("spconv_build") or "cuXXX"
        return (
            f"Setup could not install spconv-{build} on this computer ({error}). "
            f"Use Plugins > {PLUGIN_NAME} > Repair dependencies to try again: it only "
            "adds spconv to the existing environment (PyTorch is kept). If it fails "
            "again, report the Setup log on the issue tracker."
        )
    if "spconv_build" not in marker and _spconv_is_fallback(marker.get("cuda_index")):
        return (
            "spconv has no build for the CUDA 12.8 PyTorch this GPU generation needs; "
            "the plugin adds spconv's CUDA 12.6 build to the existing environment. "
            f"Use Plugins > {PLUGIN_NAME} > Repair dependencies (PyTorch is kept)."
        )
    return ""


def litept_dependency_status():
    # Setup/Repair can make imports available after a first failed check.
    # Cache only successful imports, never a failure for the entire session.
    try:
        prepared = _prepare_environment()
    except Exception as exc:
        prepared = False
        detail = f" {type(exc).__name__}: {exc}"
    else:
        detail = ""
    if not prepared:
        _check_imports.cache_clear()
        return False, (
            "LitePT's plugin environment is unavailable. Use Plugins > "
            f"{PLUGIN_NAME} > Repair dependencies and restart QGIS."
            + detail
        )
    ready, reason = _check_imports()
    if not ready:
        _check_imports.cache_clear()
        hint = _numpy_mismatch_hint() or (_spconv_hint() if "spconv" in reason else "")
        if hint:
            reason = hint + "\n" + reason
        reason += "\n" + _torch_details()
    return ready, reason


# Keep the explicit invalidation hook for repairs and existing diagnostics.
litept_dependency_status.cache_clear = _check_imports.cache_clear
