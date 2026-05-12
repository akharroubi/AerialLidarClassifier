"""Helper utilities for the plugin."""

from .logger import log_info, log_warning


def enable_point_cloud_3d_rendering(layer) -> bool:
    """Attach a classification-coloured 3D renderer to a point-cloud layer.

    When QGIS adds a ``QgsPointCloudLayer`` via Python without an
    explicit 3D renderer, opening a 3D Map View shows the layer as a
    flat 2D sprite (or invisibly). This helper wires up a 3D symbol
    keyed on the standard ASPRS ``Classification`` attribute so the
    layer renders correctly as a 3D point cloud immediately after
    auto-load - no manual ``Layer Properties -> 3D View`` step.

    Symbol preference: classification-categorised first (proper ASPRS
    palette), then colour-ramp-by-attribute, then single colour. Each
    is wrapped in feature-detection ``try/except`` so we degrade
    gracefully on older / leaner QGIS builds.

    Args:
        layer: A ``QgsPointCloudLayer`` already added to the project.

    Returns:
        True if a 3D renderer was attached, False otherwise (in which
        case the user can still open Layer Properties manually).
    """
    try:
        from qgis.core import QgsPointCloudLayer3DRenderer
    except ImportError:
        log_warning(
            "3D point-cloud rendering classes not available in this "
            "QGIS build; skipping auto-3D setup."
        )
        return False

    symbol = None
    # Preferred: classification-categorised symbol (matches user
    # intent for an ASPRS-classified file).
    try:
        from qgis.core import QgsClassificationPointCloud3DSymbol
        symbol = QgsClassificationPointCloud3DSymbol()
        symbol.setAttribute("Classification")
    except (ImportError, AttributeError):
        pass

    # Fallback 1: colour ramp keyed on Classification.
    if symbol is None:
        try:
            from qgis.core import QgsColorRampPointCloud3DSymbol
            symbol = QgsColorRampPointCloud3DSymbol()
            symbol.setAttribute("Classification")
        except (ImportError, AttributeError):
            pass

    # Fallback 2: single colour - at least the points show up in 3D.
    if symbol is None:
        try:
            from qgis.core import QgsSingleColorPointCloud3DSymbol
            symbol = QgsSingleColorPointCloud3DSymbol()
        except (ImportError, AttributeError):
            log_warning(
                "No usable point-cloud 3D symbol class in this QGIS "
                "build; 3D rendering remains off for the layer."
            )
            return False

    # Reasonable default point size; user can change in Layer Properties.
    try:
        symbol.setPointSize(2.0)
    except (AttributeError, TypeError):
        pass

    renderer3d = QgsPointCloudLayer3DRenderer()
    renderer3d.setSymbol(symbol)
    layer.setRenderer3D(renderer3d)
    log_info(f"3D rendering enabled for layer '{layer.name()}'")
    return True


# Lazy torch import to avoid loading CUDA DLLs at plugin start.
_torch = None


def _get_torch():
    global _torch
    if _torch is None:
        from .venv_manager import ensure_venv_packages_available
        ensure_venv_packages_available()
        import torch
        _torch = torch
    return _torch


def get_gpu_info() -> dict:
    """Probe PyTorch / CUDA / MPS and return GPU info.

    Returns a dict with at least the key ``available`` (bool). When
    ``available`` is True, ``name`` and ``mem`` (GB) are also set.
    """
    try:
        t = _get_torch()
    except Exception as exc:
        log_warning(f"Torch import failed during GPU detection: {exc}")
        return {"available": False, "reason": "torch_import_failed"}

    # NVIDIA CUDA
    try:
        if t.cuda.is_available():
            props = t.cuda.get_device_properties(0)
            log_info(
                f"GPU (CUDA): {props.name}, "
                f"{props.total_memory / (1024 ** 3):.1f} GB"
            )
            return {
                "available": True,
                "backend": "cuda",
                "name": props.name,
                "mem": props.total_memory / (1024 ** 3),
            }
    except Exception as exc:
        log_warning(f"CUDA probe error: {exc}")

    # Apple Silicon MPS
    try:
        mps_attr = getattr(t.backends, "mps", None)
        if mps_attr is not None and mps_attr.is_available():
            log_info("GPU (Apple MPS) available.")
            return {
                "available": True,
                "backend": "mps",
                "name": "Apple Silicon (MPS)",
                "mem": 0.0,
            }
    except Exception:
        pass

    cuda_built = hasattr(t.version, "cuda") and t.version.cuda is not None
    return {
        "available": False,
        "reason": "no_cuda" if not cuda_built else "no_gpu",
    }


def truncate_name(name: str, max_len: int = 35) -> str:
    """Truncate a filename with an ellipsis, preserving the extension."""
    if len(name) <= max_len:
        return name
    ext_start = name.rfind(".")
    if ext_start > 0:
        ext = name[ext_start:]
        return name[: max_len - len(ext) - 3] + "..." + ext
    return name[: max_len - 3] + "..."
