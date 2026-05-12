"""Helper utilities for the plugin."""

from .logger import log_info, log_warning

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
