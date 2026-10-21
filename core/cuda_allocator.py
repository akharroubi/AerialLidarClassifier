"""Choose PyTorch's CUDA caching-allocator mode for the GPU at hand.

LitePT-L's sparse convolutions and attention create many short-lived
temporaries, so torch's default allocator reserves far more memory than it
uses (16 GB reserved for 4.5 GB allocated on the validated 70,000-point
crops). Expandable segments keep "reserved" close to "allocated", which is
what smaller cards need.

Expandable segments map memory through CUDA's virtual memory management
API (cuMemCreate, cuMemMap). Shared vGPU profiles (an NVIDIA A10-8Q on a
cloud desktop, for example) do not provide that API, and torch then fails
the very first allocation with "CUDA driver error: operation not supported".
So the mode is decided per computer: the driver is asked first, a 1 MB probe
allocation confirms the answer, and the standard allocator is kept where
expandable segments cannot work. Nothing here touches the model.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Callable, Optional

_LOG = logging.getLogger(__name__)

# cuda.h: CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
_CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
_EXPANDABLE = "expandable_segments:True"
_STANDARD = "expandable_segments:False"


def driver_supports_virtual_memory(device_index: int = 0) -> Optional[bool]:
    """Ask the CUDA driver whether the device supports virtual memory
    management. ``None`` when the driver cannot be asked (no driver library,
    macOS, a call that fails): the caller then relies on the probe.
    """
    try:
        if sys.platform == "win32":
            lib = ctypes.WinDLL("nvcuda.dll")
        elif sys.platform == "darwin":
            return None
        else:
            lib = ctypes.CDLL("libcuda.so.1")
        # cuInit is idempotent: torch has usually initialised the driver already.
        if lib.cuInit(0) != 0:
            return None
        device = ctypes.c_int()
        if lib.cuDeviceGet(ctypes.byref(device), int(device_index)) != 0:
            return None
        flag = ctypes.c_int(-1)
        status = lib.cuDeviceGetAttribute(
            ctypes.byref(flag), _CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device)
        if status != 0 or flag.value < 0:
            return None
        return bool(flag.value)
    except Exception:
        _LOG.debug("CUDA driver attribute query failed", exc_info=True)
        return None


def _set_mode(torch, setting: str) -> bool:
    """Apply an allocator setting; False when torch offers no way to."""
    setter = getattr(getattr(torch, "_C", None), "_accelerator_setAllocatorSettings", None)
    if setter is None:
        memory = getattr(getattr(torch, "cuda", None), "memory", None)
        setter = getattr(memory, "_set_allocator_settings", None)
    if setter is None:
        return False
    setter(setting)
    return True


def _looks_unsupported(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "not supported" in text or "driver error" in text


def configure_cuda_allocator(torch, log: Optional[Callable[[str], None]] = None,
                             device_index: int = 0,
                             supports: Callable[[int], Optional[bool]] = driver_supports_virtual_memory,
                             ) -> str:
    """Enable expandable segments where the driver supports them.

    Returns ``"expandable"``, ``"standard"`` (fallback, explained through
    ``log``) or ``"unchanged"`` (the user set PYTORCH_CUDA_ALLOC_CONF, or
    torch offers no setter; torch's own defaults then apply).
    """
    log = log if log is not None else (lambda message: None)
    if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        _LOG.info("PYTORCH_CUDA_ALLOC_CONF is set by the user; allocator left as is")
        return "unchanged"

    supported = supports(device_index)
    if supported is False:
        _set_mode(torch, _STANDARD)
        log("CUDA memory: this GPU driver has no virtual memory management "
            "(usual on shared vGPU profiles), so PyTorch's standard allocator "
            "is used.")
        return "standard"

    try:
        if not _set_mode(torch, _EXPANDABLE):
            return "unchanged"
        probe = torch.empty(1 << 20, dtype=torch.uint8, device=f"cuda:{int(device_index)}")
        del probe
        return "expandable"
    except RuntimeError as exc:
        if not _looks_unsupported(exc):
            raise
        _LOG.info("Expandable segments refused by the driver: %s", exc)
        _set_mode(torch, _STANDARD)
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        log("CUDA memory: expandable segments are not supported by this GPU "
            f"driver ({reason}), so PyTorch's standard allocator is used.")
        return "standard"
