"""Spatial tile grid shared by the in-memory and the streaming tiled paths.

Pure Python and numpy-free on purpose: it has no QGIS or torch imports,
so it can be unit-tested outside QGIS (see tests/test_tile_grid.py).
"""

from __future__ import annotations

import math

from ..config import TILE_AUTO_TARGET_POINTS

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
