"""LAS / LAZ file utilities for the plugin."""

from pathlib import Path
from typing import List, Tuple

from .logger import log_warning


def strip_copc_vlrs(las_or_header) -> None:
    """Drop COPC-specific VLR / EVLR records in place.

    Works on either a ``laspy.LasData`` object or directly on a
    ``laspy.LasHeader``. We have to mutate the existing ``VLRList`` in
    place (``clear()`` + ``extend()``) rather than replace it with a
    plain list - the writer later calls ``evlrs.write_to(...)``, which
    is a method on ``laspy.vlrs.vlrlist.VLRList`` and absent on
    ``list``.
    """
    def _is_copc(v) -> bool:
        return hasattr(v, "user_id") and v.user_id.lower() == "copc"

    def _filter_in_place(container) -> None:
        if container is None:
            return
        kept = [v for v in container if not _is_copc(v)]
        try:
            container.clear()
        except AttributeError:
            return
        container.extend(kept)

    if hasattr(las_or_header, "vlrs"):
        _filter_in_place(las_or_header.vlrs)
    if hasattr(las_or_header, "evlrs"):
        _filter_in_place(las_or_header.evlrs)


def find_las_files(path):
    """Find all LAS/LAZ files in a path. Avoids duplicates on case-insensitive filesystems."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() in ('.las', '.laz'):
        return [p]
    if p.is_dir():
        files = set()
        for ext in ('*.las', '*.laz'):
            files.update(p.glob(ext))
        return sorted(files)
    return []


def get_las_info(filepath: Path) -> Tuple[int, float]:
    """Get point count and file size from LAS/LAZ header without loading all points.

    Returns ``(0, file_size_mb)`` if the file cannot be opened (corrupt
    header, locked by another process, etc.) so the GUI keeps working
    even on a bad file. The underlying error is logged so users can
    diagnose why a file shows zero points.
    """
    try:
        import laspy
        with laspy.open(str(filepath)) as f:
            point_count = f.header.point_count
        file_size = filepath.stat().st_size / (1024 * 1024)
        return point_count, file_size
    except Exception as exc:
        try:
            file_size = filepath.stat().st_size / (1024 * 1024)
        except OSError:
            file_size = 0.0
        log_warning(
            f"Could not read LAS/LAZ header from {filepath.name}: {exc}"
        )
        return 0, file_size


def get_folder_info(folder_path: Path) -> Tuple[List[Path], int, float]:
    """Get tile list, total point count and size for a folder of LAS/LAZ files."""
    tiles = find_las_files(folder_path)
    total_points = 0
    total_size = 0.0
    for tile in tiles:
        pts, size = get_las_info(tile)
        total_points += pts
        total_size += size
    return tiles, total_points, total_size
