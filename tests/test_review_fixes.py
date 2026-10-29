"""Regression tests for the fixes made after the independent review of 1.3.3.

Run with the plugin venv Python (numpy, numpy_indexed, torch)::

    ~/.qgis_aerial_lidar_classifier/venv_py3.12/Scripts/python.exe -m pytest tests/test_review_fixes.py

No QGIS, no GPU, no network.
"""

import math
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_plugin_module  # noqa: E402

core = load_plugin_module("core.classifier_core")
memory = load_plugin_module("core.memory")
tiling = load_plugin_module("core.tiling")
run_lock = load_plugin_module("core.run_lock")
units = load_plugin_module("utils.las_units")
safety = load_plugin_module("utils.output_safety")
hashes = load_plugin_module("utils.download_hashes")
litept = load_plugin_module("core.backends.litept")

BLOCK = [33.6, 33.6, 51.2]


# ---------------------------------------------------------------------------
# SegFormer 3D block grouping per chunk: the same groups as in one pass
# ---------------------------------------------------------------------------

def _groups(pts, dim, origin=None):
    o = None if origin is None else origin[:dim]
    return core.sliding_blocks_point_indices(pts[:, :dim], BLOCK[:dim], 0.1, origin=o)


@pytest.mark.parametrize("n", [1, 5, 999, 1000, 1001, 4321])
@pytest.mark.parametrize("dim", [2, 3])
def test_block_grouping_per_chunk_equals_one_pass(monkeypatch, n, dim):
    rng = np.random.default_rng(n)
    pts = rng.uniform(-40, 260, (n, 3))
    monkeypatch.setattr(core, "_BLOCK_CHUNK", 10 ** 9)
    whole = _groups(pts, dim, origin=pts.min(0) - 3.0)
    monkeypatch.setattr(core, "_BLOCK_CHUNK", 1000)
    chunked = _groups(pts, dim, origin=pts.min(0) - 3.0)
    np.testing.assert_array_equal(whole[0], chunked[0])
    assert len(whole[1]) == len(chunked[1])
    for a, b in zip(whole[1], chunked[1]):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Memory planning
# ---------------------------------------------------------------------------

def test_engine_memory_figures():
    assert memory.bytes_per_tile_point("litept") == memory.BYTES_PER_TILE_POINT
    assert memory.bytes_per_tile_point("segformer3d") == memory.SEGFORMER_BYTES_PER_TILE_POINT
    assert memory.bytes_per_tile_point(None) == memory.BYTES_PER_TILE_POINT
    gb = 1024 ** 3
    assert memory.tile_point_budget(16 * gb, 230) < memory.tile_point_budget(16 * gb, 170)


def test_plan_counts_the_engine_and_the_output_records():
    gb = 1024 ** 3
    n = 12_000_000
    # Fits with LitePT's figure, not with SegFormer's.
    available = int((n * (30 + memory.IN_MEMORY_BYTES_PER_POINT + 200)) / memory.FIT_FRACTION)
    assert memory.plan_processing(n, 30, False, available=available)[0] == "untiled"
    assert memory.plan_processing(n, 30, False, available=available,
                                  bytes_per_point=230)[0] != "untiled"
    # A second copy of the records (LAS 1.4 upgrade) is counted too.
    available = int((n * (34 + memory.IN_MEMORY_BYTES_PER_POINT + 170) + n * 20) / memory.FIT_FRACTION)
    assert memory.plan_processing(n, 34, False, available=available)[0] == "untiled"
    assert memory.plan_processing(n, 34, False, available=available,
                                  output_record_length=42)[0] != "untiled"
    assert gb


# ---------------------------------------------------------------------------
# Dense tiles: a coarse file-wide grid is refined for the tiles that need it
# ---------------------------------------------------------------------------

def test_coarse_grid_is_refined_locally():
    rng = np.random.default_rng(5)
    street = np.column_stack([rng.uniform(0, 1000, 100_000), rng.uniform(0, 10, 100_000)])
    xy = np.vstack([street, [[3_000_000.0, 0.0]]])  # one stray point 3,000 km away
    grid = tiling.PointCountGrid(xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    grid.add(xy[:, 0], xy[:, 1])
    assert grid.cell > 1000  # the whole street in one or two cells
    # The automatic grid makes one tile of the whole stretched extent.
    tile = (0.0, 0.0, 3_000_001.0, 3_000_001.0)
    street = street  # all of the street is in the tile's core
    budget, buffer_m = 60_000, 35.0
    plans = tiling.split_dense_tiles([tile], grid, buffer_m, budget)
    assert len(plans[0]) == 1  # the coarse grid sees no useful cut

    def local_grid(_index, window):
        inside = ((xy[:, 0] >= window[0]) & (xy[:, 0] <= window[2])
                  & (xy[:, 1] >= window[1]) & (xy[:, 1] <= window[3]))
        fine = tiling.PointCountGrid(*window)
        fine.add(xy[inside, 0], xy[inside, 1])
        return fine
    refined = tiling.refine_dense_plans([tile], plans, grid, buffer_m, budget, local_grid)[0]
    assert len(refined) >= 2
    # The pieces partition the core exactly and each window is within budget.
    core_pts = xy
    owners = np.zeros(len(core_pts), dtype=int)
    for piece in refined:
        owners += ((core_pts[:, 0] >= piece[0]) & (core_pts[:, 0] < piece[2])
                   & (core_pts[:, 1] >= piece[1]) & (core_pts[:, 1] < piece[3]))
        w = tiling.window_of(piece, buffer_m)
        n = int(((xy[:, 0] >= w[0]) & (xy[:, 0] <= w[2]) & (xy[:, 1] >= w[1]) & (xy[:, 1] <= w[3])).sum())
        assert n <= budget
    assert (owners == 1).all()


def test_fine_grid_is_not_rebuilt_when_the_file_grid_is_fine_enough():
    grid = tiling.PointCountGrid(0, 0, 100, 100)
    grid.add(np.full(50, 50.0), np.full(50, 50.0))
    calls = []
    plans = [[(40.0, 40.0, 60.0, 60.0)]]
    tiling.refine_dense_plans([(40.0, 40.0, 60.0, 60.0)], plans, grid, 5.0, 10,
                              lambda i, w: calls.append(w))
    assert calls == []


# ---------------------------------------------------------------------------
# One run at a time
# ---------------------------------------------------------------------------

def test_second_run_waits_and_can_be_cancelled():
    order, messages = [], []
    release = threading.Event()

    def first():
        with run_lock.exclusive_run():
            order.append("first in")
            release.wait(5)
            order.append("first out")
    worker = threading.Thread(target=first)
    worker.start()
    while not order:
        time.sleep(0.01)
    with pytest.raises(InterruptedError):
        with run_lock.exclusive_run(lambda: True, messages.append, poll_s=0.01):
            order.append("never")
    assert messages and "Another classification is running" in messages[0]
    release.set()
    with run_lock.exclusive_run(lambda: False, messages.append, poll_s=0.01):
        order.append("second")
    worker.join()
    assert order == ["first in", "first out", "second"]


# ---------------------------------------------------------------------------
# Units from WKT
# ---------------------------------------------------------------------------

def _projcs(unit):
    return ('PROJCS["Test",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
            'PROJECTION["Transverse_Mercator"],' + unit + ']')


@pytest.mark.parametrize("factor", ["0", "-1", "nan", "inf"])
def test_invalid_unit_factor_falls_back_to_the_name(factor):
    found = units.units_from_wkt(_projcs(f'UNIT["metre",{factor}]'))
    assert found is not None and found.xy_to_m == 1.0
    assert units.units_from_wkt(_projcs(f'UNIT["unknown thing",{factor}]')) is None


def test_kilometre_without_a_factor():
    found = units.units_from_wkt(_projcs('UNIT["kilometre"]'))
    assert found is not None and math.isclose(found.xy_to_m, 1000.0)
    found = units.units_from_wkt(_projcs('UNIT["metre"]'))
    assert found is not None and found.xy_to_m == 1.0


# ---------------------------------------------------------------------------
# LitePT-L voxel keys: overflow is refused, not wrapped
# ---------------------------------------------------------------------------

def test_voxel_keys_refuse_values_beyond_int64():
    local = np.array([[0, 0, 0], [1e18, 0, 0], [2e18, 0, 0]], dtype=np.float32)
    with pytest.raises(ValueError, match="too large"):
        litept._voxel_partition(local, 0.05)


def test_voxel_partition_unchanged_on_ordinary_data():
    rng = np.random.default_rng(3)
    local = rng.uniform(0, 50, (20_000, 3)).astype(np.float32)
    local[5000:6000] = local[:1000]  # duplicates
    keep, projection = litept._voxel_partition(local, 0.1)
    keys = np.floor(local.astype(np.float64) / 0.1).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    np.testing.assert_array_equal(keep, first[order])
    np.testing.assert_array_equal(projection, rank[inverse.ravel()])


# ---------------------------------------------------------------------------
# Bootstrap checksums
# ---------------------------------------------------------------------------

def test_every_bootstrap_archive_has_a_pinned_checksum():
    platforms = ("x86_64-pc-windows-msvc", "aarch64-apple-darwin", "x86_64-apple-darwin",
                 "aarch64-unknown-linux-gnu", "x86_64-unknown-linux-gnu")
    for version in ("3.9.21", "3.10.16", "3.11.11", "3.12.8", "3.13.1"):
        for platform in platforms:
            name = f"cpython-{version}+20241219-{platform}-install_only.tar.gz"
            assert len(hashes.PYTHON_STANDALONE_SHA256[name]) == 64
    for platform, ext in zip(platforms, (".zip", ".tar.gz", ".tar.gz", ".tar.gz", ".tar.gz")):
        assert len(hashes.UV_SHA256[f"uv-{platform}{ext}"]) == 64


def test_check_archive(tmp_path):
    path = tmp_path / "a.zip"
    path.write_bytes(b"abc")
    digest = hashes.sha256_of(str(path))
    assert hashes.check_archive(str(path), "a.zip", {"a.zip": digest}) == ""
    assert "mismatch" in hashes.check_archive(str(path), "a.zip", {"a.zip": "0" * 64})
    assert "no pinned" in hashes.check_archive(str(path), "b.zip", {"a.zip": digest})


# ---------------------------------------------------------------------------
# Publication: cancel and input changes keep the previous output
# ---------------------------------------------------------------------------

def test_before_publish_can_keep_the_previous_output(tmp_path):
    target = tmp_path / "out.las"
    target.write_bytes(b"previous")

    def refuse():
        raise InterruptedError("cancelled")
    with pytest.raises(InterruptedError):
        with safety.atomic_output_path(target, before_publish=refuse) as temporary:
            temporary.write_bytes(b"new")
    assert target.read_bytes() == b"previous"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.las"]
    with safety.atomic_output_path(target, before_publish=lambda: None) as temporary:
        temporary.write_bytes(b"new")
    assert target.read_bytes() == b"new"


def test_input_change_is_detected(tmp_path):
    source = tmp_path / "in.las"
    source.write_bytes(b"12345")
    signature = safety.file_signature(source)
    safety.require_unchanged(source, signature)
    os.utime(source, ns=(signature[1] + 10 ** 9, signature[1] + 10 ** 9))
    with pytest.raises(RuntimeError, match="changed during the run"):
        safety.require_unchanged(source, signature)


def test_output_record_length():
    class PF:
        def __init__(self, pid, size, dims=("x", "classification")):
            self.id, self.size, self.dimension_names = pid, size, dims

    class Info:
        def __init__(self, code):
            self.asprs_code = code
    low = {1: Info(2), 2: Info(6)}
    high = {1: Info(2), 2: Info(64)}
    assert safety.output_record_length(PF(3, 34), "classification", low) == 0
    assert safety.output_record_length(PF(3, 34), "classification", high) == 42
    assert safety.output_record_length(PF(6, 30), "classification", high) == 0
    assert safety.output_record_length(PF(6, 30), "ai_class", low) == 34
    assert safety.output_record_length(PF(6, 30, ("x", "ai_class")), "ai_class", low) == 0


@pytest.mark.parametrize("seed", range(12))
def test_refined_pieces_always_partition_the_tiles(seed):
    """Whatever the geometry (clusters, stray points, empty areas), every
    point of a tile core is in exactly one piece, and every piece window
    lies in the tile's window (so the tile's points contain the piece's)."""
    rng = np.random.default_rng(seed)
    parts = [rng.normal(rng.uniform(0, 5000, 2), rng.uniform(2, 300), (int(rng.integers(200, 6000)), 2))
             for _ in range(int(rng.integers(1, 5)))]
    parts.append(rng.uniform(-1e6, 1e6, (int(rng.integers(0, 4)), 2)))  # stray points
    xy = np.vstack(parts)
    x0, y0 = xy.min(0)
    x1, y1 = xy.max(0)
    tiles, _ = tiling.compute_tile_grid(x0, y0, x1, y1, len(xy), True, None, target_points=4000)
    grid = tiling.PointCountGrid(x0, y0, x1, y1, max_cells=int(rng.integers(16, 2048)))
    grid.add(xy[:, 0], xy[:, 1])
    buffer_m = float(rng.uniform(5, 60))
    budget = int(rng.integers(300, 3000))

    def local_grid(_index, window):
        inside = ((xy[:, 0] >= window[0]) & (xy[:, 0] <= window[2])
                  & (xy[:, 1] >= window[1]) & (xy[:, 1] <= window[3]))
        fine = tiling.PointCountGrid(*window)
        fine.add(xy[inside, 0], xy[inside, 1])
        return fine
    plans = tiling.split_dense_tiles(tiles, grid, buffer_m, budget)
    plans = tiling.refine_dense_plans(tiles, plans, grid, buffer_m, budget, local_grid)
    for tile, plan in zip(tiles, plans):
        inside = ((xy[:, 0] >= tile[0]) & (xy[:, 0] < tile[2])
                  & (xy[:, 1] >= tile[1]) & (xy[:, 1] < tile[3]))
        pts = xy[inside]
        owners = np.zeros(len(pts), dtype=int)
        tw = tiling.window_of(tile, buffer_m)
        for piece in plan:
            owners += ((pts[:, 0] >= piece[0]) & (pts[:, 0] < piece[2])
                       & (pts[:, 1] >= piece[1]) & (pts[:, 1] < piece[3]))
            w = tiling.window_of(piece, buffer_m)
            assert tw[0] <= w[0] + 1e-6 and tw[1] <= w[1] + 1e-6
            assert w[2] <= tw[2] + 1e-6 and w[3] <= tw[3] + 1e-6
        assert (owners == 1).all(), (seed, int((owners != 1).sum()))

