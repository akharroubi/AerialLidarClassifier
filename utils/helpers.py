"""Helper utilities for the plugin."""

from .logger import log_info, log_warning
import logging

_LOG = logging.getLogger(__name__)


def _point_cloud_categories(class_mapping):
    """One enabled legend category per actual output code, including merges."""
    from qgis.core import QgsPointCloudCategory
    from qgis.PyQt.QtGui import QColor
    grouped = {}
    for _class_id, info in sorted(class_mapping.items()):
        code = int(info.asprs_code)
        if code not in grouped:
            grouped[code] = (info.color, [])
        names = grouped[code][1]
        if info.name not in names:
            names.append(info.name)
    return [QgsPointCloudCategory(code, QColor(color), " / ".join(names), True)
            for code, (color, names) in sorted(grouped.items())]


def enable_point_cloud_3d_rendering(layer, class_mapping=None, classify_2d=False) -> bool:
    """Style actual classification codes, or mirror existing styling for raw fields.

    Pass the completed run's mapping only when it wrote the standard LAS
    classification field. Mobile Mapping also uses these categories in 2D.
    No point data is changed. Missing optional 3D support leaves 2D usable.
    """
    source_2d = None
    categories = _point_cloud_categories(class_mapping) if class_mapping else None
    if categories:
        from qgis.core import QgsPointCloudClassifiedRenderer
        source_2d = QgsPointCloudClassifiedRenderer("Classification", categories)
        if classify_2d:
            layer.setRenderer(source_2d.clone())
            layer.triggerRepaint()
    else:
        # Raw extra fields must not relabel the input's existing classification.
        # Match its existing RGB/elevation style, instead of an empty category list.
        source_2d = layer.renderer()

    try:
        from qgis import _3d as qgis3d
    except ImportError:
        from qgis import core as qgis3d
    renderer_class = getattr(qgis3d, "QgsPointCloudLayer3DRenderer", None)
    if renderer_class is None or source_2d is None:
        log_warning("3D point-cloud rendering is unavailable in this QGIS build.")
        return False

    # The 3D symbol is built by QGIS from a 2D renderer. QGIS 4 exposes the
    # 3D symbol classes (QgsClassificationPointCloud3DSymbol...) as abstract
    # in Python, so they cannot be instantiated directly there.
    try:
        renderer3d = renderer_class()
        renderer3d.setLayer(layer)
        if not renderer3d.convertFrom2DRenderer(source_2d):
            log_warning(f"No 3D style could be derived for layer '{layer.name()}'; "
                        "the 2D view is unaffected.")
            return False
        symbol = renderer3d.symbol()
        if symbol is not None:
            symbol.setPointSize(2.0)
        layer.setRenderer3D(renderer3d)
    except Exception as exc:
        log_warning(f"3D rendering could not be enabled for layer '{layer.name()}': {exc}. "
                    "The 2D view is unaffected.")
        return False
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
        _LOG.debug("Ignored non-fatal error", exc_info=True)

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
