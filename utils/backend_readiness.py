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
        reason += "\n" + _torch_details()
    return ready, reason


# Keep the explicit invalidation hook for repairs and existing diagnostics.
litept_dependency_status.cache_clear = _check_imports.cache_clear
