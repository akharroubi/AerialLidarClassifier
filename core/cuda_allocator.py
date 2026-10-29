"""Choose PyTorch's CUDA caching-allocator mode for the GPU at hand.

LitePT-L's sparse convolutions and attention create many short-lived
temporaries, so torch's default allocator reserves more memory than it
uses. Expandable segments keep "reserved" close to "allocated".

Expandable segments map memory through CUDA's virtual memory management
API (cuMemCreate, cuMemMap, cuMemSetAccess), and that is where they fail:

- Shared vGPU profiles (an NVIDIA A10-8Q on a cloud desktop) do not provide
  the API: "CUDA driver error: operation not supported" at the first
  allocation. The driver says so (device attribute 102).
- Windows drivers (WDDM) can refuse the mapping calls intermittently on
  cards that report the API as supported: "CUDA driver error: unknown
  error" after anything from a 4 MB to a 32 MB allocation, varying from run
  to run, then "!handles_.at(i) INTERNAL ASSERT FAILED" on the next one
  (GitHub issue 8, RTX A6000). No probe allocation detects this reliably.
- Some torch builds have no expandable segments on Windows at all (2.11).

So the automatic mode uses the standard allocator on Windows, and
expandable segments elsewhere when the driver reports the API (with a
small probe allocation). If a run still fails inside the allocator with
expandable segments, the run switches to the standard allocator, retries,
and the computer is remembered so later runs start with the standard one.

Measured on the LitePT-L models (1.3.3 validation): the standard allocator
gives identical labels, with no out-of-memory retry on 6, 8 and 12 GB.

The user can fix the mode: the "GPU memory allocator" choice in the dock's
Advanced parameters (QgsSettings) or, for scripts and qgis_process, the
AERIAL_LIDAR_CLASSIFIER_CUDA_ALLOCATOR environment variable. Every decision
is reported in the run log with a line starting with "CUDA memory:".
Nothing here touches the model.
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
# QgsSettings keys (under the plugin prefix) and the environment override.
SETTING_NAME = "cuda_allocator"
FAILED_SETTING_NAME = "cuda_allocator_failed"
ENV_OVERRIDE = "AERIAL_LIDAR_CLASSIFIER_CUDA_ALLOCATOR"
SETTING_ADVICE = ("The mode can be fixed in the dock, Advanced parameters > "
                  "GPU memory allocator.")

# What configure_cuda_allocator() applied in this process: "expandable",
# "standard" or "unchanged", and whether the user forced it ("setting") or
# the automatic mode chose it ("auto"). Read by the in-run recovery.
_active = {"mode": None, "chosen_by": None, "signature": None}


def _settings_key(name: str) -> Optional[str]:
    try:
        from ..config import SETTINGS_PREFIX
    except Exception:
        return None
    return f"{SETTINGS_PREFIX}/{name}"


def preferred_mode() -> str:
    """The user's choice: environment override, then the plugin setting,
    then automatic. Unknown values mean automatic."""
    value = os.environ.get(ENV_OVERRIDE, "").strip().lower()
    if not value:
        try:
            from qgis.core import QgsSettings
            value = str(QgsSettings().value(
                _settings_key(SETTING_NAME), MODE_AUTO) or "").strip().lower()
        except Exception:
            _LOG.debug("Allocator setting not readable; automatic mode", exc_info=True)
            value = MODE_AUTO
    return value if value in MODES else MODE_AUTO


def _signature(torch, device_index: int) -> str:
    """GPU and torch build a failure is remembered for: a driver or torch
    update is a different computer as far as the allocator is concerned."""
    try:
        name = torch.cuda.get_device_name(device_index)
    except Exception:
        name = "unknown GPU"
    version = getattr(torch, "__version__", "unknown")
    return f"{name}|{version}|{driver_version() or 'unknown driver'}"


def driver_version() -> Optional[str]:
    """The NVIDIA driver version ("596.72") from NVML, or the CUDA driver
    API version ("CUDA 12090") when NVML is not available; None otherwise."""
    if sys.platform == "win32":
        candidates = ("nvml.dll", os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "NVIDIA Corporation", "NVSMI", "nvml.dll"))
    else:
        candidates = ("libnvidia-ml.so.1", "libnvidia-ml.so")
    for name in candidates:
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        initialised = False
        try:
            init = getattr(lib, "nvmlInit_v2", None) or getattr(lib, "nvmlInit")
            if init() != 0:
                continue
            initialised = True
            text = ctypes.create_string_buffer(96)
            if lib.nvmlSystemGetDriverVersion(text, ctypes.c_uint(96)) == 0:
                return text.value.decode("ascii", "replace")
        except Exception:
            _LOG.debug("NVML driver version query failed", exc_info=True)
        finally:
            if initialised:
                try:
                    lib.nvmlShutdown()
                except Exception:
                    _LOG.debug("NVML shutdown failed", exc_info=True)
    try:
        lib = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        value = ctypes.c_int(0)
        if lib.cuDriverGetVersion(ctypes.byref(value)) == 0 and value.value:
            return f"CUDA {value.value}"
    except Exception:
        _LOG.debug("CUDA driver version query failed", exc_info=True)
    return None


def _remembered_failures() -> list:
    try:
        from qgis.core import QgsSettings
        value = QgsSettings().value(_settings_key(FAILED_SETTING_NAME), [])
    except Exception:
        return []
    if isinstance(value, str):
        value = [value] if value else []
    return [str(item) for item in (value or [])]


def _remember_failure(signature: str) -> None:
    try:
        from qgis.core import QgsSettings
        known = _remembered_failures()
        if signature not in known:
            QgsSettings().setValue(_settings_key(FAILED_SETTING_NAME), known + [signature])
    except Exception:
        _LOG.debug("Could not remember the allocator failure", exc_info=True)


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


# PyTorch's allocator variables: PYTORCH_ALLOC_CONF (current name) and
# PYTORCH_CUDA_ALLOC_CONF (older name, still read).
_TORCH_ENV_NAMES = ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")


def _environment_setting():
    """``(name, True | False | None)`` when one of PyTorch's allocator
    variables sets expandable_segments (None: a value not understood),
    None when neither does."""
    import re
    for name in _TORCH_ENV_NAMES:
        value = os.environ.get(name, "")
        if "expandable_segments" not in value:
            continue
        found = re.search(r"expandable_segments\s*:\s*(true|false)", value, re.IGNORECASE)
        return name, None if found is None else found.group(1).lower() == "true"
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


def looks_like_allocator_failure(exc: BaseException) -> bool:
    """The driver refused expandable segments during a run: the driver
    error itself, or the allocator assertion that follows it."""
    text = str(exc)
    lowered = text.lower()
    return ("driver error" in lowered or "handles_" in text
            or "expandable_segment" in lowered
            or ("cudacachingallocator" in lowered and "internal assert" in lowered))


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


def _done(mode: str, chosen_by: str, signature: Optional[str] = None) -> str:
    _active.update(mode=mode, chosen_by=chosen_by, signature=signature)
    return mode


_NO_SETTER = ("CUDA memory: this PyTorch build offers no allocator setting; its "
              "default (standard) allocator is used.")


def configure_cuda_allocator(torch, log: Optional[Callable[[str], None]] = None,
                             device_index: int = 0,
                             supports: Callable[[int], Optional[bool]] = driver_supports_virtual_memory,
                             mode: Optional[str] = None,
                             platform: Optional[str] = None,
                             ) -> str:
    """Apply the allocator mode before the first GPU allocation of a run.

    ``mode`` is one of ``MODES`` (``None``: the user's preference);
    ``platform`` defaults to ``sys.platform``. Returns ``"expandable"``,
    ``"standard"`` or ``"unchanged"`` (automatic mode with a
    PYTORCH_ALLOC_CONF / PYTORCH_CUDA_ALLOC_CONF value it does not
    understand, or a torch without the setter). A value it understands is
    applied as given. Every decision
    is reported through ``log`` with a line starting with "CUDA memory:".
    """
    log = log if log is not None else (lambda message: None)
    mode = mode if mode in MODES else preferred_mode()
    platform = platform or sys.platform

    if mode == MODE_STANDARD:
        if not _set_mode(torch, _STANDARD):
            log(_NO_SETTER)
            return _done("unchanged", "setting")
        log("CUDA memory: PyTorch's standard allocator (plugin setting).")
        return _done("standard", "setting")

    if mode == MODE_EXPANDABLE:
        if not _set_mode(torch, _EXPANDABLE):
            log(_NO_SETTER)
            return _done("unchanged", "setting")
        try:
            _probe(torch, device_index)
        except RuntimeError as exc:
            if not _looks_unsupported(exc):
                raise
            _set_mode(torch, _STANDARD)
            _done("standard", "setting")
            raise RuntimeError(
                "Expandable segments are selected in the plugin settings, but this "
                f"GPU driver refuses them ({_first_line(exc)}). Choose Automatic or "
                "Standard allocator under Advanced parameters > GPU memory "
                "allocator and run again."
            ) from exc
        if _PLATFORM_UNSUPPORTED:
            log(_platform_message())
            return _done("standard", "setting")
        log("CUDA memory: expandable segments (plugin setting).")
        return _done("expandable", "setting")

    environment = _environment_setting()
    if environment is not None:
        # The user's own PyTorch setting. It is applied again explicitly: an
        # earlier run of this session may have forced the other mode.
        name, expandable = environment
        _LOG.info("%s is set by the user", name)
        if expandable is None:
            log(f"CUDA memory: {name} is set in the environment; the allocator is "
                "left as configured there.")
            return _done("unchanged", "environment")
        if not _set_mode(torch, _EXPANDABLE if expandable else _STANDARD):
            log(_NO_SETTER)
            return _done("unchanged", "environment")
        if not expandable:
            log(f"CUDA memory: PyTorch's standard allocator ({name} in the environment).")
            return _done("standard", "environment")
        _probe(torch, device_index)
        if _PLATFORM_UNSUPPORTED:
            log(_platform_message())
            return _done("standard", "environment")
        log(f"CUDA memory: expandable segments ({name} in the environment).")
        return _done("expandable", "environment")

    if platform == "win32":
        if not _set_mode(torch, _STANDARD):
            log(_NO_SETTER)
            return _done("unchanged", "auto")
        log("CUDA memory: PyTorch's standard allocator (automatic choice on "
            "Windows, where expandable segments fail with some GPU drivers). "
            + SETTING_ADVICE)
        return _done("standard", "auto")

    signature = _signature(torch, device_index)
    if signature in _remembered_failures():
        _set_mode(torch, _STANDARD)
        log("CUDA memory: PyTorch's standard allocator (expandable segments "
            "failed in an earlier run on this GPU and driver). " + SETTING_ADVICE)
        return _done("standard", "auto", signature)

    supported = supports(device_index)
    if supported is False:
        _set_mode(torch, _STANDARD)
        log("CUDA memory: this GPU driver has no virtual memory management "
            "(usual on shared vGPU profiles), so PyTorch's standard allocator "
            "is used. " + SETTING_ADVICE)
        return _done("standard", "auto", signature)

    try:
        if not _set_mode(torch, _EXPANDABLE):
            log(_NO_SETTER)
            return _done("unchanged", "auto", signature)
        _probe(torch, device_index)
    except RuntimeError as exc:
        if not _looks_unsupported(exc):
            raise
        _LOG.info("Expandable segments refused by the driver: %s", exc)
        _set_mode(torch, _STANDARD)
        _remember_failure(signature)
        log("CUDA memory: expandable segments are not supported by this GPU "
            f"driver ({_first_line(exc)}), so PyTorch's standard allocator is "
            "used. " + SETTING_ADVICE)
        return _done("standard", "auto", signature)
    if _PLATFORM_UNSUPPORTED:
        log(_platform_message())
        return _done("standard", "auto", signature)
    log("CUDA memory: expandable segments (the driver supports virtual memory "
        "management). " + SETTING_ADVICE)
    return _done("expandable", "auto", signature)


def recover_from_allocator_failure(torch, exc: BaseException,
                                   log: Optional[Callable[[str], None]] = None) -> bool:
    """Called when a GPU operation of a run raised ``exc``.

    When the automatic mode had chosen expandable segments and ``exc`` is
    the driver refusing them, switch this process to the standard
    allocator, remember the GPU and driver for later runs, and return True:
    the caller retries the operation once. With expandable segments forced
    in the settings, raise an error that names the setting to change.
    Return False for any other error (the caller re-raises it).
    """
    log = log if log is not None else (lambda message: None)
    if _active["mode"] != "expandable" or not looks_like_allocator_failure(exc):
        return False
    if _active["chosen_by"] == "environment":
        names = " / ".join(_TORCH_ENV_NAMES)
        raise RuntimeError(
            "Expandable segments are set in the environment (" + names + "), but this "
            f"GPU driver failed with them during the run ({_first_line(exc)}). Remove "
            "expandable_segments from that variable, restart QGIS and run again."
        ) from exc
    if _active["chosen_by"] == "setting":
        raise RuntimeError(
            "Expandable segments are selected in the plugin settings, but this GPU "
            f"driver failed with them during the run ({_first_line(exc)}). Choose "
            "Automatic or Standard allocator under Advanced parameters > GPU "
            "memory allocator, restart QGIS and run again."
        ) from exc
    _set_mode(torch, _STANDARD)
    try:
        torch.cuda.empty_cache()
    except Exception:
        _LOG.debug("empty_cache after the allocator failure", exc_info=True)
    if _active["signature"]:
        _remember_failure(_active["signature"])
    _done("standard", "auto", _active["signature"])
    log("CUDA memory: the GPU driver refused expandable segments during the run "
        f"({_first_line(exc)}); switched to PyTorch's standard allocator and "
        "retrying. Later runs on this GPU start with the standard allocator.")
    return True


def restart_advice(exc: BaseException) -> RuntimeError:
    """The error raised when the retry after a recovery fails as well."""
    return RuntimeError(
        "The GPU driver refused PyTorch's expandable segments and the retry with "
        f"the standard allocator failed too ({_first_line(exc)}). Restart QGIS and "
        "run again: the plugin now uses the standard allocator on this computer."
    )
