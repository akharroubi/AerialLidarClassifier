"""Tests for core/tiling.py (auto tile sizing used by tiled and streaming modes).

Run with any Python that has numpy::

    python tests/test_tile_grid.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_plugin_module  # noqa: E402

tiling = load_plugin_module("core.tiling")

TARGET = 200_000


def _points_per_tile(pts, tiles):
    counts = []
    for (x0, y0, x1, y1) in tiles:
        inside = (
            (pts[:, 0] >= x0) & (pts[:, 0] < x1)
            & (pts[:, 1] >= y0) & (pts[:, 1] < y1)
        )
        counts.append(int(inside.sum()))
    return counts


def _uniform(n, w, h, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, w, n), rng.uniform(0, h, n)])


def test_corridor_respects_point_target():
    """10 km x 0.5 km corridor: v1.0.2 made a 4 x 1 grid of 2500 m tiles
    with 2.5 x the target per tile; the area-based size must not."""
    n = 2_000_000
    pts = _uniform(n, 10_000, 500)
    tiles, size = tiling.compute_tile_grid(
        0, 0, 10_000, 500, n, True, None, target_points=TARGET
    )
    counts = _points_per_tile(pts, tiles)
    assert sum(counts) == n, "every point must fall in exactly one core"
    assert max(counts) <= 1.3 * TARGET, (
        f"largest tile holds {max(counts):,} points for a target of "
        f"{TARGET:,} (tile size {size:.0f} m, {len(tiles)} tiles)"
    )


def test_square_extent_respects_point_target():
    n = 2_000_000
    pts = _uniform(n, 1_000, 1_000)
    tiles, _ = tiling.compute_tile_grid(
        0, 0, 1_000, 1_000, n, True, None, target_points=TARGET
    )
    counts = _points_per_tile(pts, tiles)
    assert sum(counts) == n
    assert max(counts) <= 1.3 * TARGET


def test_small_cloud_is_a_single_tile():
    tiles, size = tiling.compute_tile_grid(
        0, 0, 3_000, 200, 50_000, True, None, target_points=TARGET
    )
    assert len(tiles) == 1
    assert size == 3_000


def test_manual_size_is_honoured():
    tiles, size = tiling.compute_tile_grid(0, 0, 1_000, 1_000, 10, False, 250.0)
    assert size == 250.0
    assert len(tiles) == 16


def test_points_on_the_max_edges_are_inside_a_core():
    pts = np.array([[1_000.0, 1_000.0], [0.0, 0.0], [1_000.0, 0.0], [0.0, 1_000.0]])
    tiles, _ = tiling.compute_tile_grid(0, 0, 1_000, 1_000, 4, False, 300.0)
    assert sum(_points_per_tile(pts, tiles)) == 4


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
