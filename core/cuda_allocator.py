"""Choose PyTorch's CUDA caching-allocator mode for the GPU at hand.

LitePT-L's sparse convolutions and attention create many short-lived
temporaries, so torch's default allocator reserves far more memory than it
uses (16 GB reserved for 4.5 GB allocated on the validated 70,000-point
crops). Expandable segments keep "reserved" close to "allocated", which is
what smaller cards need.

Expandable segments map memory through CUDA's virtual memory management
API (cuMemCreate, cuMemMap, cuMemSetAccess). Shared vGPU profiles (an NVIDIA
A10-8Q on a cloud desktop, for example) do not provide that API, and torch
then fails the very first allocation with "CUDA driver error: operation not
supported". Some Windows drivers refuse the same calls on cards that do
support the API ("CUDA driver error: unknown error", then an allocator
assertion on the next run). So the mode is decided per computer: the driver
is asked first, a 1 MB probe allocation confirms the answer, and the
standard allocator is kept where expandable segments cannot work.

The user can also fix the mode: the "GPU memory allocator" choice in the
dock's Advanced parameters (QgsSettings) or, for scripts and qgis_process,
the AERIAL_LIDAR_CLASSIFIER_CUDA_ALLOCATOR environment variable. Nothing
here touches the model.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import warnings
from typing import Callable, Optional

_LOG = logging.getLogger(__name__)

# cuda.h: CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED
_CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102
_EXPANDABLE = "expandable_segments:True"
_STANDARD = "expandable_segments:False"

MODE_AUTO = "auto"
MODE_EXPANDABLE = "expandable"
MODE_STANDARD = "standard"
MODES = (MODE_AUTO, MODE_EXPANDABLE, MODE_STANDARD)
MODE_LABELS = {
    MODE_AUTO: "Automatic (recommended)",
    MODE_EXPANDABLE: "Expandable segments",
    MODE_STANDARD: "Standard allocator",
}
# QgsSettings key (under the plugin prefix) and the environment override.
SETTING_NAME = "cuda_allocator"
ENV_OVERRIDE = "AERIAL_LIDAR_CLASSIFIER_CUDA_ALLOCATOR"
SETTING_ADVICE = ("The mode can be fixed in the dock, Advanced parameters > "
                  "GPU memory allocator.")


def preferred_mode() -> str:
    """The user's choice: environment override, then the plugin setting,
    then automatic. Unknown values mean automatic."""
    value = os.environ.get(ENV_OVERRIDE, "").strip().lower()
    if not value:
        try:
            from qgis.core import QgsSettings
            from ..config import SETTINGS_PREFIX
            value = str(QgsSettings().value(
                f"{SETTINGS_PREFIX}/{SETTING_NAME}", MODE_AUTO) or "").strip().lower()
        except Exception:
            _LOG.debug("Allocator setting not readable; automatic mode", exc_info=True)
            value = MODE_AUTO
    return value if value in MODES else MODE_AUTO


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


# Some torch builds have no expandable segments on this platform at all
# (torch 2.11 on Windows, for example): the setting is accepted, a warning
# "expandable_segments not supported on this platform" is printed once per
# process, and the standard allocator serves silently. Remembered here so
# that the log tells the truth for every model loaded in the session.
_PLATFORM_UNSUPPORTED = False
_UNSUPPORTED_WARNING = "expandable_segments not supported"


def _probe(torch, device_index: int) -> None:
    """One small allocation under the current setting; records torch's
    "not supported on this platform" warning when it appears."""
    global _PLATFORM_UNSUPPORTED
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe = torch.empty(1 << 20, dtype=torch.uint8, device=f"cuda:{int(device_index)}")
        del probe
    if any(_UNSUPPORTED_WARNING in str(item.message) for item in caught):
        _PLATFORM_UNSUPPORTED = True
    for item in caught:
        if _UNSUPPORTED_WARNING not in str(item.message):
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)


def _platform_message() -> str:
    return ("CUDA memory: this PyTorch build has no expandable segments on this "
            "platform, so PyTorch's standard allocator is used.")


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else type(exc).__name__


def configure_cuda_allocator(torch, log: Optional[Callable[[str], None]] = None,
                             device_index: int = 0,
                             supports: Callable[[int], Optional[bool]] = driver_supports_virtual_memory,
                             mode: Optional[str] = None,
                             ) -> str:
    """Apply the allocator mode before the first GPU allocation of a run.

    ``mode`` is one of ``MODES`` (``None``: the user's preference). Returns
    ``"expandable"``, ``"standard"`` or ``"unchanged"`` (automatic mode with
    the user's own PYTORCH_CUDA_ALLOC_CONF, or a torch without the setter;
    torch's own defaults then apply). Every decision is reported through
    ``log`` so the run log says which allocator served it.
    """
    log = log if log is not None else (lambda message: None)
    mode = mode if mode in MODES else preferred_mode()

    if mode == MODE_STANDARD:
        if not _set_mode(torch, _STANDARD):
            return "unchanged"
        log("CUDA memory: PyTorch's standard allocator (plugin setting).")
        return "standard"

    if mode == MODE_EXPANDABLE:
        if not _set_mode(torch, _EXPANDABLE):
            return "unchanged"
        try:
            _probe(torch, device_index)
        except RuntimeError as exc:
            if not _looks_unsupported(exc):
                raise
            _set_mode(torch, _STANDARD)
            raise RuntimeError(
                "Expandable segments are selected in the plugin settings, but this "
                f"GPU driver refuses them ({_first_line(exc)}). Choose Automatic or "
                "Standard allocator under Advanced parameters > GPU memory "
                "allocator and run again."
            ) from exc
        if _PLATFORM_UNSUPPORTED:
            log(_platform_message())
            return "standard"
        log("CUDA memory: expandable segments (plugin setting).")
        return "expandable"

    if "expandable_segments" in os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""):
        _LOG.info("PYTORCH_CUDA_ALLOC_CONF is set by the user; allocator left as is")
        log("CUDA memory: PYTORCH_CUDA_ALLOC_CONF is set in the environment; "
            "the allocator is left as configured there.")
        return "unchanged"

    supported = supports(device_index)
    if supported is False:
        _set_mode(torch, _STANDARD)
        log("CUDA memory: this GPU driver has no virtual memory management "
            "(usual on shared vGPU profiles), so PyTorch's standard allocator "
            "is used. " + SETTING_ADVICE)
        return "standard"

    try:
        if not _set_mode(torch, _EXPANDABLE):
            return "unchanged"
        _probe(torch, device_index)
    except RuntimeError as exc:
        if not _looks_unsupported(exc):
            raise
        _LOG.info("Expandable segments refused by the driver: %s", exc)
        _set_mode(torch, _STANDARD)
        log("CUDA memory: expandable segments are not supported by this GPU "
            f"driver ({_first_line(exc)}), so PyTorch's standard allocator is "
            "used. " + SETTING_ADVICE)
        return "standard"
    if _PLATFORM_UNSUPPORTED:
        log(_platform_message())
        return "standard"
    log("CUDA memory: expandable segments (the driver supports virtual memory "
        "management). " + SETTING_ADVICE)
    return "expandable"
