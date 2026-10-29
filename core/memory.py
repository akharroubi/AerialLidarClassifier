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
# coordinates and indices as read back (32 B) plus the engine's own work.
# LitePT-L (voxelisation, crops, projection), measured in 1.3.3 with
# tracemalloc on a 1.08 M-point street tile: 131 B per point for Mobile
# Mapping at 5 cm, where nearly every point is its own voxel, 103 B for
# Airborne at 10 cm, 55 B on data 8 times denser; 170 B covers the worst
# case with the tile's own arrays.
BYTES_PER_TILE_POINT = 170
# SegFormer 3D (block grouping and per-block voxel indices): 192 B per point
# measured on the same tile and on a 4.3 M-point copy (512 B before the
# grouping was computed per chunk); 230 B with the tile's own arrays.
SEGFORMER_BYTES_PER_TILE_POINT = 230
# Share of the available memory one tile may use (QGIS, the model and the
# rest of the run need the remainder).
TILE_MEMORY_FRACTION = 0.5
MIN_TILE_POINTS = 2_000_000
# Scripts and tests can fix the budget (points per buffered tile window).
ENV_TILE_POINTS = "AERIAL_LIDAR_CLASSIFIER_TILE_POINTS"


def bytes_per_tile_point(family: Optional[str]) -> int:
    """Host memory per point of a tile window for a model family
    (``ModelSpec.family``); the LitePT-L figure for unknown families."""
    if family == "segformer3d":
        return SEGFORMER_BYTES_PER_TILE_POINT
    return BYTES_PER_TILE_POINT


def available_memory_bytes() -> Optional[int]:
    """Memory available to new allocations, or None if unknown.

    On Windows the smaller of the free physical memory and the free commit
    charge: an allocation fails when the commit limit (RAM plus page file)
    is reached, even with physical memory free."""
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
                # ullAvailPageFile is the commit charge still available.
                return int(min(status.ullAvailPhys, status.ullAvailPageFile))
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


def tile_point_budget(available: Optional[int] = None,
                      bytes_per_point: int = BYTES_PER_TILE_POINT) -> int:
    """Largest number of points one buffered tile window may hold
    (``bytes_per_point``: see bytes_per_tile_point)."""
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
    per_point = max(int(bytes_per_point), 1)
    return max(MIN_TILE_POINTS, int(available * TILE_MEMORY_FRACTION // per_point))


# The in-memory path holds the whole file: laspy's point records (the
# header's point record length) plus the metre coordinates, predictions,
# labels and output conversion. Measured in 1.3.3 on a 26 M-point file with
# 30-byte records (model step excluded): 73 B per point in one piece, so
# 43 B beyond the record; 48 B leaves a margin. Tiles add their own copy,
# counted in BYTES_PER_TILE_POINT. When the output needs new point records
# (a LAS 1.4 format upgrade for codes above 31, or a new extra field), the
# old and the new records exist together: the caller passes the new
# record length as ``output_record_length``.
IN_MEMORY_BYTES_PER_POINT = 48
# Share of the available memory a run may plan to use.
FIT_FRACTION = 0.8


def _gb(value: float) -> str:
    return f"{value / 1024 ** 3:.1f} GB"


def plan_processing(n_points: int, record_length: int, tiling: bool,
                    force_streaming: bool = False,
                    available: Optional[int] = None,
                    bytes_per_point: int = BYTES_PER_TILE_POINT,
                    output_record_length: int = 0) -> tuple:
    """How to process a file: ``("untiled" | "tiled" | "streaming" |
    "too_large", reason)``.

    ``tiling`` is the "Tile large files" option. Ticked, a file is tiled in
    memory, or streamed from disk when it would not fit in RAM. Unticked, the
    file runs in one piece as asked, and one that would not fit in memory in
    one piece is "too_large": the caller refuses it and asks for tiling (the
    plugin never changes how an unticked run is processed, so its labels are
    always those of one piece). ``reason`` is a log line for the automatic
    streaming and the measure for "too_large", empty otherwise.
    ``bytes_per_point`` is the model engine's tile cost
    (bytes_per_tile_point); ``output_record_length`` the length of new output
    records, 0 when the labels are written in place.
    """
    if force_streaming:
        return "streaming", ""
    if available is None:
        available = available_memory_bytes() or 8 * 1024 ** 3
    usable = available * FIT_FRACTION
    budget = tile_point_budget(available, bytes_per_point)
    whole_file = n_points * (int(record_length) + int(output_record_length)
                             + IN_MEMORY_BYTES_PER_POINT)
    tiled_need = whole_file + min(n_points, budget) * int(bytes_per_point)
    untiled_need = whole_file + n_points * int(bytes_per_point)
    if not tiling:
        if untiled_need <= usable:
            return "untiled", ""
        return "too_large", (
            f"{n_points:,} points need about {_gb(untiled_need)} of memory in one piece "
            f"and {_gb(available)} is available.")
    if tiled_need <= usable:
        return "tiled", ""
    return "streaming", (
        f"{n_points:,} points need about {_gb(tiled_need)} of memory to process in "
        f"memory and {_gb(available)} is available: streaming the file from disk "
        "automatically (temporary tile files in the output folder).")
