"""Regression tests for ``sliding_blocks_point_indices`` (core/classifier_core.py).

Run with the plugin venv Python (needs numpy, numpy_indexed and torch, all
present there), either through pytest or directly::

    ~/.qgis_aerial_lidar_classifier/venv_py3.12/Scripts/python.exe \\
        tests/test_block_partition.py

These tests do not import QGIS and do not run the model: they only check
that the block partition hands every point to at least one block whose
voxel grid can hold it. Before v1.1, a tile whose Z range was under 10 %
of the 51.2 m Z block collapsed into a single block and about 99.5 % of
its points were left at the fill value (class 7 = Building).
"""

import importlib.util
from pathlib import Path

import numpy as np

_CORE = Path(__file__).resolve().parent.parent / "core" / "classifier_core.py"
_spec = importlib.util.spec_from_file_location("classifier_core", _CORE)
classifier_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classifier_core)

# UrbanFiltering eSegFormer3D block geometry (core/model_config.json).
NBMAT_SZ = np.array([112, 112, 256])
MIN_RES = np.array([0.3, 0.3, 0.2])
BLOCK_SIZE = MIN_RES * NBMAT_SZ  # [33.6, 33.6, 51.2] m
OVERLAP = 0.1


def _coverage(pts: np.ndarray, cut_dim: int):
    """Return (n_blocks, fraction of points visited, fraction kept by nb_sel).

    Mirrors the voxel filter in ``filterPoints`` so the test exercises the
    same condition that decides whether a point gets a real prediction.
    """
    _, groups = classifier_core.sliding_blocks_point_indices(
        pts[:, :cut_dim], BLOCK_SIZE[:cut_dim], overlap_ratio=OVERLAP
    )
    visited = np.zeros(len(pts), dtype=bool)
    kept = np.zeros(len(pts), dtype=bool)
    for idx in groups:
        visited[idx] = True
        block = pts[idx]
        ijk = np.floor((block[:, :3] - block[:, :3].min(axis=0)) / MIN_RES)
        sel = np.all((ijk < NBMAT_SZ) & (ijk >= 0), axis=1)
        kept[idx[sel]] = True
    return len(groups), visited.mean(), kept.mean()


def _tile(z_range: float, xy_range: float = 500.0, n: int = 200_000, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0.0, xy_range, n),
        rng.uniform(0.0, xy_range, n),
        rng.uniform(0.0, z_range, n),
    ])


def test_flat_tile_3d_blocks_is_fully_covered():
    """Z range far below 10 % of the Z block (the v1.0.2 failure case)."""
    n_blocks, visited, kept = _coverage(_tile(z_range=3.0), cut_dim=3)
    assert n_blocks > 1, "flat tile collapsed into a single block"
    assert visited == 1.0
    assert kept == 1.0, f"only {kept:.2%} of points would be classified"


def test_z_range_just_under_threshold():
    n_blocks, visited, kept = _coverage(_tile(z_range=5.0), cut_dim=3)
    assert n_blocks > 1
    assert visited == 1.0
    assert kept == 1.0


def test_normal_tile_unchanged():
    n_blocks, visited, kept = _coverage(_tile(z_range=40.0), cut_dim=3)
    assert n_blocks > 1
    assert visited == 1.0
    assert kept == 1.0


def test_tiny_footprint_all_axes():
    """A cloud smaller than a block on every axis (tiny tile + buffer)."""
    n_blocks, visited, kept = _coverage(
        _tile(z_range=1.0, xy_range=2.0, n=5_000), cut_dim=3
    )
    assert n_blocks == 1
    assert visited == 1.0
    assert kept == 1.0


def test_degenerate_single_point():
    pts = np.array([[10.0, 20.0, 30.0]])
    n_blocks, visited, kept = _coverage(pts, cut_dim=3)
    assert n_blocks == 1
    assert visited == 1.0
    assert kept == 1.0


def test_flat_tile_2d_blocks_unchanged():
    """Bottom-only mode never blocked on Z, so it must keep working."""
    n_blocks, visited, kept = _coverage(_tile(z_range=3.0), cut_dim=2)
    assert n_blocks > 1
    assert visited == 1.0
    assert kept == 1.0


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
