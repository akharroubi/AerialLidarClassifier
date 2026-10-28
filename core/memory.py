"""Host memory available to the plugin, and the tile size it allows.

No QGIS, torch or numpy imports (unit-testable anywhere).
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from typing import Optional

_LOG = logging.getLogger(__name__)

# Peak host memory of one tile, per point of its buffered window: the tile's
# coordinates and indices as read back (32 B) plus LitePT-L's voxelisation,
# crops and projection. Measured in 1.3.3 with tracemalloc (street tile,
# 1.08 M points): 131 B per point for Mobile Mapping at 5 cm, where nearly
# every point is its own voxel, 103 B for Airborne at 10 cm, 55 B on data 8
# times denser (88 B before the 1.3.3 reductions). 170 B covers the worst
# case with the tile's own arrays.
BYTES_PER_TILE_POINT = 170
# Share of the available memory one tile may use (QGIS, the model and the
# rest of the run need the remainder).
TILE_MEMORY_FRACTION = 0.5
MIN_TILE_POINTS = 2_000_000
# Scripts and tests can fix the budget (points per buffered tile window).
ENV_TILE_POINTS = "AERIAL_LIDAR_CLASSIFIER_TILE_POINTS"


def available_memory_bytes() -> Optional[int]:
    """Physical memory available to new allocations, or None if unknown."""
    try:
        if sys.platform == "win32":
            class _Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            status = _Status()
            status.dwLength = ctypes.sizeof(_Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullAvailPhys)
            return None
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo", encoding="ascii") as stream:
                for line in stream:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        # macOS and others: half the physical memory as a cautious estimate.
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return int(pages * page_size // 2)
    except Exception:
        _LOG.debug("Available memory not readable", exc_info=True)
    return None


def tile_point_budget(available: Optional[int] = None) -> int:
    """Largest number of points one buffered tile window may hold."""
    override = os.environ.get(ENV_TILE_POINTS, "").strip()
    if override:
        try:
            return max(int(float(override)), 1)
        except ValueError:
            _LOG.warning("Ignoring %s=%r (not a number)", ENV_TILE_POINTS, override)
    if available is None:
        available = available_memory_bytes()
    if not available:
        # Unknown: assume 8 GB free, a common laptop.
        available = 8 * 1024 ** 3
    return max(MIN_TILE_POINTS, int(available * TILE_MEMORY_FRACTION // BYTES_PER_TILE_POINT))


# The in-memory path holds the whole file: laspy's point records (the
# header's point record length) plus the metre coordinates, predictions,
# labels and output conversion. Measured in 1.3.3 on a 26 M-point file with
# 30-byte records (model step excluded): 73 B per point in one piece, so
# 43 B beyond the record; 48 B leaves a margin. Tiles add their own copy,
# counted in BYTES_PER_TILE_POINT.
IN_MEMORY_BYTES_PER_POINT = 48
# Share of the available memory a run may plan to use.
FIT_FRACTION = 0.8


def _gb(value: float) -> str:
    return f"{value / 1024 ** 3:.1f} GB"


def plan_processing(n_points: int, record_length: int, tiling: bool,
                    force_streaming: bool = False,
                    available: Optional[int] = None) -> tuple:
    """How to process a file: ``("untiled" | "tiled" | "streaming", reason)``.

    Tiling and streaming are one choice for the user: ``tiling`` is the
    "Tile large files" option. A file is streamed from disk when it would
    not fit in RAM in memory, and tiled when it was meant to run in one
    piece but that piece would not fit. ``reason`` is a log line for those
    automatic switches, empty otherwise.
    """
    if force_streaming:
        return "streaming", ""
    if available is None:
        available = available_memory_bytes() or 8 * 1024 ** 3
    usable = available * FIT_FRACTION
    budget = tile_point_budget(available)
    whole_file = n_points * (int(record_length) + IN_MEMORY_BYTES_PER_POINT)
    tiled_need = whole_file + min(n_points, budget) * BYTES_PER_TILE_POINT
    untiled_need = whole_file + n_points * BYTES_PER_TILE_POINT
    if not tiling and untiled_need <= usable:
        return "untiled", ""
    if tiled_need <= usable:
        if tiling:
            return "tiled", ""
        return "tiled", (
            f"{n_points:,} points need about {_gb(untiled_need)} of memory in one piece "
            f"and {_gb(available)} is available: tiling automatically.")
    return "streaming", (
        f"{n_points:,} points need about {_gb(tiled_need)} of memory to process in "
        f"memory and {_gb(available)} is available: streaming the file from disk "
        "automatically (temporary tile files in the output folder).")
