"""Spatial tile grid shared by the in-memory and the streaming tiled paths.

No QGIS or torch imports, so it can be unit-tested outside QGIS (see
tests/test_tile_grid.py).

``compute_tile_grid`` sizes a regular grid from the *average* density.
Mobile mapping and corridor data are anything but uniform: in GitHub issue
10 one 592 m tile of a 717 M-point MLS file held 211 M points and the run
ran out of RAM after two hours. ``PointCountGrid`` records where the points
actually are (built while the points are read anyway), and
``split_dense_tiles`` splits every tile whose buffered window holds more
points than the RAM budget allows, until it fits or until splitting no
longer helps (the buffer halo then dominates the window).
"""

from __future__ import annotations

import math

from ..config import TILE_AUTO_TARGET_POINTS, TILE_FALLBACK_BUFFER_M

Tile = tuple[float, float, float, float]


def compute_tile_grid(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    n_points: int,
    auto: bool,
    manual_tile_size_m: float | None,
    target_points: int = TILE_AUTO_TARGET_POINTS,
) -> tuple[list[Tile], float]:
    """Build a non-overlapping XY tile grid covering the extent.

    Returns ``(tiles, tile_size_m)`` where each tile is ``(x0, y0, x1,
    y1)``. Tiles cover the full extent without overlap; the buffer halo
    is added separately by the callers when they select points.

    Auto mode sizes tiles from the extent's **area** so that, at uniform
    density, each tile carries about ``target_points`` points. v1.0.2
    divided the longer side by ``ceil(sqrt(n_tiles))`` instead, which on
    a corridor-shaped extent (10 km x 0.5 km) produced a 4 x 1 grid with
    2.5 times the target per tile, defeating the memory bound that the
    streaming mode exists to guarantee.
    """
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)

    if auto:
        target = max(int(target_points), 1)
        n_tiles_needed = max(1, math.ceil(n_points / target))
        if n_tiles_needed == 1:
            tile_size_m = max(width, height)
        else:
            tile_size_m = math.sqrt(width * height / n_tiles_needed)
    else:
        tile_size_m = float(manual_tile_size_m or 500.0)

    # The last row / column absorbs the far edge: its upper bound is
    # nudged by a relative epsilon so the half-open core [lo, hi) still
    # covers points exactly at xmax / ymax. Looping on the nudged bound
    # instead (v1.0.2) produced an extra sliver tile whenever the tile
    # size divided the extent exactly, and that sliver's buffer halo
    # then cost a full inference pass for a handful of edge points.
    x_end = max(xmax, xmin + width)
    y_end = max(ymax, ymin + height)
    eps_x = max(1e-6, (x_end - xmin) * 1e-9)
    eps_y = max(1e-6, (y_end - ymin) * 1e-9)

    tiles: list[Tile] = []
    y = ymin
    while y < y_end:
        y1 = y + tile_size_m
        if y1 >= y_end:
            y1 = y_end + eps_y
        x = xmin
        while x < x_end:
            x1 = x + tile_size_m
            if x1 >= x_end:
                x1 = x_end + eps_x
            tiles.append((x, y, x1, y1))
            x += tile_size_m
        y += tile_size_m
    return tiles, tile_size_m


def resolve_buffer(requested_m, predict_fn) -> tuple:
    """Buffer a tiled run uses, and a log line saying how it was chosen.

    ``predict_fn.context_buffer`` (set by the callers) returns the loaded
    model's need: the buffer with which a tile's core gets exactly the
    labels of an untiled run (LitePT-L: twice the crop radius; SegFormer
    3D: one block). ``requested_m`` 0 or less means automatic (the need);
    a larger value is used as given, a smaller one is raised to the need.
    """
    requested = float(requested_m or 0.0)
    context = getattr(predict_fn, "context_buffer", None)
    need = context() if callable(context) else None
    # Rounded up to 0.1 m: the log shows it, and more context never hurts.
    need = None if need is None else math.ceil(float(need) * 10.0) / 10.0
    if need is None:
        buffer = requested if requested > 0 else TILE_FALLBACK_BUFFER_M
        return buffer, f"Tile buffer {buffer:g} m."
    if requested <= 0:
        return need, (f"Tile buffer {need:g} m (automatic: the context the model needs, "
                      "so tiles give the same labels as an untiled run).")
    if requested < need:
        return need, (f"Tile buffer raised from {requested:g} m to {need:g} m, the context "
                      "the model needs for tiles to give the same labels as an untiled run.")
    return requested, f"Tile buffer {requested:g} m (the model needs at least {need:g} m)."


# ---------------------------------------------------------------------------
# Point-count map and splitting of dense tiles
# ---------------------------------------------------------------------------

class PointCountGrid:
    """Point counts on a regular XY grid over an extent (metres).

    At most ``max_cells`` cells per side (the cell is at least ``min_cell_m``
    wide), so the map stays small (2048 x 2048 int64 = 32 MB) whatever the
    extent. ``count`` returns the number of points in the cells a rectangle
    touches: exact on cell boundaries, otherwise an upper bound, which is the
    safe side for a memory budget.
    """

    def __init__(self, xmin: float, ymin: float, xmax: float, ymax: float,
                 max_cells: int = 2048, min_cell_m: float = 0.5):
        import numpy as np

        width = max(float(xmax) - float(xmin), 0.0)
        height = max(float(ymax) - float(ymin), 0.0)
        self.cell = max(float(min_cell_m), max(width, height) / float(max_cells))
        self.x0, self.y0 = float(xmin), float(ymin)
        self.nx = int(width // self.cell) + 1
        self.ny = int(height // self.cell) + 1
        self.counts = np.zeros(self.nx * self.ny, dtype=np.int64)
        self._table = None

    def _cells(self, values, origin, size):
        import numpy as np

        index = np.floor((np.asarray(values, dtype=np.float64) - origin) / self.cell)
        return np.clip(index, 0, size - 1).astype(np.int64)

    def add(self, x, y) -> None:
        """Count points (metres); points outside the extent go to the edge cells."""
        import numpy as np

        if len(x) == 0:
            return
        flat = self._cells(y, self.y0, self.ny) * self.nx + self._cells(x, self.x0, self.nx)
        self.counts += np.bincount(flat, minlength=self.nx * self.ny)
        self._table = None

    @property
    def total(self) -> int:
        return int(self.counts.sum())

    def count(self, x0: float, y0: float, x1: float, y1: float) -> int:
        """Points in the cells touched by the closed rectangle [x0, x1] x [y0, y1]."""
        import numpy as np

        if self._table is None:
            table = np.zeros((self.ny + 1, self.nx + 1), dtype=np.int64)
            table[1:, 1:] = self.counts.reshape(self.ny, self.nx).cumsum(0).cumsum(1)
            self._table = table
        i0 = min(max(int(math.floor((x0 - self.x0) / self.cell)), 0), self.nx - 1)
        i1 = min(max(int(math.floor((x1 - self.x0) / self.cell)), 0), self.nx - 1)
        j0 = min(max(int(math.floor((y0 - self.y0) / self.cell)), 0), self.ny - 1)
        j1 = min(max(int(math.floor((y1 - self.y0) / self.cell)), 0), self.ny - 1)
        if x1 < x0 or y1 < y0:
            return 0
        t = self._table
        return int(t[j1 + 1, i1 + 1] - t[j0, i1 + 1] - t[j1 + 1, i0] + t[j0, i0])


def _halves(t: Tile, axis: int, c: float) -> tuple[Tile, Tile]:
    if axis == 0:
        return (t[0], t[1], c, t[3]), (c, t[1], t[2], t[3])
    return (t[0], t[1], t[2], c), (t[0], c, t[2], t[3])


def _balanced_cut(window, t: Tile, axis: int, min_core_m: float) -> float | None:
    """Cut along ``axis`` that minimises the larger of the two buffered
    windows: the left window only grows and the right one only shrinks as
    the cut moves right, so a binary search finds where they meet. Kept at
    least ``min_core_m`` from both edges; None if the tile is too narrow."""
    lo_edge, hi_edge = (t[0], t[2]) if axis == 0 else (t[1], t[3])
    lo, hi = lo_edge + min_core_m, hi_edge - min_core_m
    if hi < lo:
        return None
    for _ in range(40):
        mid = (lo + hi) / 2.0
        first, second = _halves(t, axis, mid)
        if window(first) < window(second):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def split_dense_tiles(
    tiles: list[Tile],
    grid: PointCountGrid,
    buffer_m: float,
    max_points: int,
    min_core_m: float | None = None,
    min_gain: float = 0.9,
) -> list[list[Tile]]:
    """Split every tile whose buffered window holds more than ``max_points``.

    Returns one list of tiles per input tile, in the same order: the tile
    itself, or the pieces that replace it. A tile is cut in two,
    recursively, while its window is over budget. Along each axis the cut
    is placed where the two buffered windows hold the same number of points
    (where the points are, not at the tile's middle: a street filling one
    half of a tile is cut, not the empty half), and the axis whose cut
    leaves the smaller larger half wins, so a corridor is cut across, not
    along.

    Splitting stops when the best cut no longer reduces the window by at
    least ``1 - min_gain``: the buffer halo then holds most of the points,
    and more pieces would mostly repeat work. No core gets narrower than
    ``min_core_m`` (default: a quarter of the buffer, at least 2 m). A
    budget that cannot be met is left to the caller to report.

    The pieces partition the tile's core exactly (shared cut coordinates,
    half-open cores), and each piece's buffered window lies inside the
    tile's, so the points read for the tile contain every piece's points.
    """
    b = float(buffer_m)
    limit = max(int(max_points), 1)
    if min_core_m is None:
        min_core_m = max(2.0, b / 4.0)

    def window(t: Tile) -> int:
        return grid.count(t[0] - b, t[1] - b, t[2] + b, t[3] + b)

    result: list[list[Tile]] = []
    for tile in tiles:
        leaves: list[Tile] = []
        stack = [tile]
        while stack:
            t = stack.pop()
            n = window(t)
            if n <= limit:
                leaves.append(t)
                continue
            best = None
            for axis in (0, 1):
                c = _balanced_cut(window, t, axis, min_core_m)
                if c is None:
                    continue
                first, second = _halves(t, axis, c)
                largest = max(window(first), window(second))
                if best is None or largest < best[0]:
                    best = (largest, first, second)
            if best is None or best[0] > min_gain * n:
                leaves.append(t)
                continue
            stack.append(best[2])
            stack.append(best[1])
        result.append(leaves)
    return result
