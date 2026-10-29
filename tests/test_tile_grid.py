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


def _window_counts(pts, tiles, buffer_m):
    return _points_per_tile(pts, [(x0 - buffer_m, y0 - buffer_m, x1 + buffer_m + 1e-9,
                                   y1 + buffer_m + 1e-9) for (x0, y0, x1, y1) in tiles])


def _corridor(n_line, n_background, seed=1):
    """An MLS-like extent: a dense 20 m wide street crossing a 2 km x 2 km
    area with a sparse background (GitHub issue 10)."""
    rng = np.random.default_rng(seed)
    street = np.column_stack([rng.uniform(0, 2_000, n_line), rng.uniform(990, 1_010, n_line)])
    return np.vstack([street, _uniform(n_background, 2_000, 2_000, seed)])


def test_point_count_grid_counts_and_bounds():
    pts = _corridor(300_000, 100_000)
    grid = tiling.PointCountGrid(0, 0, 2_000, 2_000)
    grid.add(pts[:200_000, 0], pts[:200_000, 1])
    grid.add(pts[200_000:, 0], pts[200_000:, 1])
    assert grid.total == len(pts)
    assert grid.count(-1e9, -1e9, 1e9, 1e9) == len(pts)
    rng = np.random.default_rng(5)
    for _ in range(200):
        x0, y0 = rng.uniform(-100, 2_000, 2)
        x1, y1 = x0 + rng.uniform(0, 900), y0 + rng.uniform(0, 900)
        exact = _points_per_tile(pts, [(x0, y0, x1 + 1e-9, y1 + 1e-9)])[0]
        # Cells touched by the rectangle: never fewer than the exact count.
        assert grid.count(x0, y0, x1, y1) >= exact
    # On cell boundaries the count is exact.
    c = grid.cell
    exact = _points_per_tile(pts, [(10 * c, 20 * c, 300 * c, 400 * c)])[0]
    assert grid.count(10 * c, 20 * c, 300 * c - 1e-9, 400 * c - 1e-9) == exact
    assert grid.count(5, 5, 4, 4) == 0


def test_dense_corridor_tiles_are_split_to_the_budget():
    """The average-density grid puts the whole street in a few tiles; after
    splitting, every buffered window fits the budget and the pieces still
    partition the extent exactly."""
    pts = _corridor(1_500_000, 100_000)
    n = len(pts)
    tiles, _ = tiling.compute_tile_grid(0, 0, 2_000, 2_000, n, True, None, target_points=400_000)
    buffer_m = 20.0
    budget = 150_000
    assert max(_window_counts(pts, tiles, buffer_m)) > 3 * budget
    grid = tiling.PointCountGrid(0, 0, 2_000, 2_000)
    grid.add(pts[:, 0], pts[:, 1])
    plans = tiling.split_dense_tiles(tiles, grid, buffer_m, budget)
    assert len(plans) == len(tiles)
    pieces = [piece for plan in plans for piece in plan]
    assert len(pieces) > len(tiles)
    assert sum(_points_per_tile(pts, pieces)) == n, "every point in exactly one core"
    assert max(_window_counts(pts, pieces, buffer_m)) <= budget
    for tile, plan in zip(tiles, plans):
        for x0, y0, x1, y1 in plan:
            assert tile[0] <= x0 < x1 <= tile[2] and tile[1] <= y0 < y1 <= tile[3]
    # The street runs along x: it is cut across (pieces narrower in x), not along.
    widths = [p[2] - p[0] for plan in plans if len(plan) > 1 for p in plan]
    heights = [p[3] - p[1] for plan in plans if len(plan) > 1 for p in plan]
    assert np.median(widths) < np.median(heights)


def test_street_in_one_half_of_a_tile_is_still_split():
    """1.3.3 scale test: the street filled one half of a 1.7 km tile, so a cut
    at the tile's middle left nearly every point on one side; the cut must
    follow the points."""
    rng = np.random.default_rng(4)
    n = 600_000
    street = np.column_stack([rng.uniform(900, 1_700, n), rng.uniform(800, 860, n)])
    pts = np.vstack([street, _uniform(500, 1_730, 1_730, 2)])
    tiles = [(0.0, 0.0, 1_730.0, 1_730.0)]
    grid = tiling.PointCountGrid(0, 0, 1_730, 1_730)
    grid.add(pts[:, 0], pts[:, 1])
    budget = 100_000
    pieces = tiling.split_dense_tiles(tiles, grid, 50.0, budget)[0]
    assert len(pieces) > 1
    assert max(_window_counts(pts, pieces, 50.0)) <= budget
    assert sum(_points_per_tile(pts, pieces)) == len(pts)


def test_tiles_within_budget_are_unchanged():
    pts = _uniform(500_000, 1_000, 1_000)
    tiles, _ = tiling.compute_tile_grid(0, 0, 1_000, 1_000, len(pts), True, None, target_points=TARGET)
    grid = tiling.PointCountGrid(0, 0, 1_000, 1_000)
    grid.add(pts[:, 0], pts[:, 1])
    plans = tiling.split_dense_tiles(tiles, grid, 50.0, 10 * TARGET)
    assert plans == [[t] for t in tiles]


def test_splitting_stops_when_the_buffer_dominates():
    """A budget smaller than the points within one buffer distance cannot be
    met; splitting must stop instead of producing slivers."""
    pts = _corridor(1_000_000, 0)
    tiles = [(0.0, 0.0, 2_000.0, 2_000.0 + 1e-6)]
    grid = tiling.PointCountGrid(0, 0, 2_000, 2_000)
    grid.add(pts[:, 0], pts[:, 1])
    buffer_m = 100.0
    plans = tiling.split_dense_tiles(tiles, grid, buffer_m, max_points=1_000)
    pieces = plans[0]
    # No core narrower than a quarter of the buffer: 2 km / 25 m at most,
    # and the 20 m wide street is cut across only, never along.
    assert 1 < len(pieces) <= 80
    assert min(p[2] - p[0] for p in pieces) >= buffer_m / 4 - 1e-6
    assert all(p[3] - p[1] > 1_000 for p in pieces)
    assert sum(_points_per_tile(pts, pieces)) == len(pts)
    # Each point is read a bounded number of times despite the impossible budget.
    assert sum(_window_counts(pts, pieces, buffer_m)) < 12 * len(pts)


def test_memory_budget():
    import os
    from unittest import mock
    memory = load_plugin_module("core.memory")
    with mock.patch.dict(os.environ, {memory.ENV_TILE_POINTS: ""}):
        assert memory.tile_point_budget(available=16 * 1024 ** 3) == int(
            16 * 1024 ** 3 * memory.TILE_MEMORY_FRACTION // memory.BYTES_PER_TILE_POINT)
        assert memory.tile_point_budget(available=10 ** 6) == memory.MIN_TILE_POINTS
    with mock.patch.dict(os.environ, {memory.ENV_TILE_POINTS: "1234567"}):
        assert memory.tile_point_budget() == 1_234_567
    available = memory.available_memory_bytes()
    assert available is None or available > 0


def test_processing_plan_streams_only_what_does_not_fit():
    import os
    from unittest import mock
    memory = load_plugin_module("core.memory")
    gb = 1024 ** 3
    with mock.patch.dict(os.environ, {memory.ENV_TILE_POINTS: ""}):
        plan = memory.plan_processing
        # A small file: as asked, nothing to report.
        assert plan(5_000_000, 30, tiling=False, available=32 * gb) == ("untiled", "")
        assert plan(5_000_000, 30, tiling=True, available=32 * gb) == ("tiled", "")
        # Issue 10: 717 M points, 34-byte records, 32 GB free: streamed.
        mode, reason = plan(717_000_000, 34, tiling=True, available=32 * gb)
        assert mode == "streaming" and "streaming the file from disk automatically" in reason
        # Unticked: never tiled or streamed by the plugin; refused instead.
        mode, reason = plan(717_000_000, 34, tiling=False, available=32 * gb)
        assert mode == "too_large" and "in one piece" in reason
        mode, reason = plan(120_000_000, 30, tiling=False, available=32 * gb)
        assert mode == "too_large"
        assert plan(120_000_000, 30, tiling=True, available=32 * gb) == ("tiled", "")
        assert plan(100_000_000, 30, tiling=False, available=32 * gb) == ("untiled", "")
        # Forced streaming (Processing TILE_STREAMING, scripts) is honoured.
        assert plan(1_000, 30, tiling=True, force_streaming=True, available=32 * gb) == ("streaming", "")
        # More memory, same file: in memory.
        assert plan(717_000_000, 34, tiling=True, available=512 * gb)[0] == "tiled"


def test_buffer_is_the_models_context_unless_set_larger():
    import types
    need = types.SimpleNamespace()

    def predict_with(context):
        fn = types.SimpleNamespace()
        fn.context_buffer = lambda: context
        return fn
    assert tiling.resolve_buffer(0, predict_with(35.0))[0] == 35.0
    assert "automatic" in tiling.resolve_buffer(0, predict_with(35.0))[1]
    assert tiling.resolve_buffer(20, predict_with(35.0))[0] == 35.0
    assert "raised" in tiling.resolve_buffer(20, predict_with(35.0))[1]
    assert tiling.resolve_buffer(120, predict_with(91.49))[0] == 120.0
    # Rounded up to 0.1 m, never below the need.
    assert tiling.resolve_buffer(0, predict_with(91.4918))[0] == 91.5
    # A backend that does not state its need: the user's value, or 50 m.
    assert tiling.resolve_buffer(0, need)[0] == 50.0
    assert tiling.resolve_buffer(12, need)[0] == 12.0


def test_segformer_blocks_of_a_tile_match_the_untiled_lattice():
    """With the file's minimum as origin and a buffer of one block, every
    core point lies in the same blocks, holding the same points, as untiled."""
    core = load_plugin_module("core.classifier_core")
    rng = np.random.default_rng(8)
    pts = np.column_stack([rng.uniform(0, 300, 40_000), rng.uniform(0, 200, 40_000), rng.uniform(0, 30, 40_000)])
    block = np.array([33.6, 33.6, 51.2])
    origin = pts.min(axis=0)
    full_origins, full_groups = core.sliding_blocks_point_indices(pts, block, 0.1)
    full = {tuple(np.round(o, 6)): set(g.tolist()) for o, g in zip(full_origins, full_groups)}
    tile = (100.0, 50.0, 160.0, 110.0)
    b = 33.6
    window = np.flatnonzero((pts[:, 0] >= tile[0] - b) & (pts[:, 0] <= tile[2] + b)
                            & (pts[:, 1] >= tile[1] - b) & (pts[:, 1] <= tile[3] + b))
    origins, groups = core.sliding_blocks_point_indices(pts[window], block, 0.1, origin=origin)
    in_core = set(window[(pts[window, 0] >= tile[0]) & (pts[window, 0] < tile[2])
                         & (pts[window, 1] >= tile[1]) & (pts[window, 1] < tile[3])].tolist())
    checked = 0
    for o, g in zip(origins, groups):
        members = set(window[g].tolist())
        if members & in_core:
            assert full[tuple(np.round(o, 6))] == members
            checked += 1
    assert checked > 0
    # Without the file origin, the tile's lattice is shifted (the old behaviour).
    shifted, _ = core.sliding_blocks_point_indices(pts[window], block, 0.1)
    assert not {tuple(np.round(o, 6)) for o in shifted} <= set(full)


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
