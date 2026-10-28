"""Smoke test of the LitePT-L backend (needs CUDA, spconv and the weights).

Skips cleanly when any of them is missing. Weights are looked up in
``$ALC_LITEPT_WEIGHTS``, the QGIS default profile, and the repository's
``v1.1/model_release`` folder. Run with the plugin venv Python.
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import PLUGIN_ROOT, load_plugin_module  # noqa: E402

WEIGHTS_NAME = "litept_l_dales_10cm_ema_fp16.pth"


def _find_weights():
    candidates = [
        os.environ.get("ALC_LITEPT_WEIGHTS", ""),
        Path(os.environ.get("APPDATA", "")) / "QGIS" / "QGIS3" / "profiles" / "default"
        / "AerialLidarClassifier" / "models" / "litept_l_dales_10cm" / WEIGHTS_NAME,
        PLUGIN_ROOT.parent / "model_release" / WEIGHTS_NAME,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _ready():
    try:
        import torch
        import spconv.pytorch  # noqa: F401
    except ImportError:
        return False, "torch or spconv not importable"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    if _find_weights() is None:
        return False, "weights not found"
    return True, ""


def test_shims_match_torch_scatter_semantics():
    import torch
    shims = load_plugin_module("core.litept._shims")
    src = torch.tensor([[1.0], [3.0], [2.0], [10.0]])
    indptr = torch.tensor([0, 2, 3, 4])
    assert torch.equal(shims.segment_csr(src, indptr, "sum"), torch.tensor([[4.0], [2.0], [10.0]]))
    assert torch.equal(shims.segment_csr(src, indptr, "mean"), torch.tensor([[2.0], [2.0], [10.0]]))
    assert torch.equal(shims.segment_csr(src, indptr, "max"), torch.tensor([[3.0], [2.0], [10.0]]))
    assert torch.equal(shims.segment_csr(src, indptr, "min"), torch.tensor([[1.0], [2.0], [10.0]]))
    # Packed attention: equal patches (batched path) and unequal (loop path)
    # must give the same numbers as plain SDPA per patch.
    torch.manual_seed(0)
    qkv = torch.randn(8, 3, 2, 4)
    out_equal = shims.flash_attn_varlen_qkvpacked_func(qkv, torch.tensor([0, 4, 8]), 4)
    out_loop = shims.flash_attn_varlen_qkvpacked_func(qkv, torch.tensor([0, 3, 8]), 8)
    assert out_equal.shape == (8, 2, 4) and out_loop.shape == (8, 2, 4)
    q, k, v = qkv[:4].unbind(1)
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
    ).squeeze(0).transpose(0, 1)
    assert torch.allclose(out_equal[:4], ref, atol=1e-6)


def test_backend_refuses_cpu():
    litept = load_plugin_module("core.backends.litept")
    weights = _find_weights() or "missing.pth"
    backend = litept.LitePTBackend(PLUGIN_ROOT / "core" / "litept_l_dales_10cm.json", weights)
    try:
        backend.load("cpu")
        raise AssertionError("expected a RuntimeError on cpu")
    except RuntimeError as exc:
        assert "CUDA" in str(exc)


def test_backend_predicts_synthetic_cloud():
    ok, why = _ready()
    if not ok:
        print(f"SKIP ({why})")
        return
    litept = load_plugin_module("core.backends.litept")
    backend = litept.LitePTBackend(
        PLUGIN_ROOT / "core" / "litept_l_dales_10cm.json", _find_weights(), log=print,
    )
    backend.load("cuda")
    rng = np.random.default_rng(5)
    n = 30_000
    ground = np.column_stack([rng.uniform(0, 60, n), rng.uniform(0, 60, n), rng.normal(0, 0.05, n)])
    box = np.column_stack([rng.uniform(20, 35, 6000), rng.uniform(20, 35, 6000), rng.uniform(0, 8, 6000)])
    pts = np.vstack([ground, box])
    progress = []
    ids = backend.predict(pts, progress_callback=progress.append)
    backend.unload()
    assert ids.shape == (len(pts),) and ids.min() >= 1 and ids.max() <= 8
    assert progress and progress[-1] == 100.0
    # A flat plane must be overwhelmingly ground.
    assert (ids[:n] == 1).mean() > 0.9, f"ground share {(ids[:n] == 1).mean():.2%}"


def test_spconv_compilation_is_decided_by_the_gpu_generation():
    import types
    litept = load_plugin_module("core.backends.litept")

    def torch_with(capability):
        return types.SimpleNamespace(cuda=types.SimpleNamespace(
            get_device_capability=lambda index=0: capability))

    # Hopper and older: native kernels in every spconv build.
    assert litept.spconv_build_needing_compilation(torch_with((8, 6))) is None
    assert litept.spconv_build_needing_compilation(torch_with((9, 0))) is None
    # Blackwell: the installed build (if any) is compiled by the driver.
    build = litept.spconv_build_needing_compilation(torch_with((12, 0)))
    assert build == litept.installed_spconv_build()
    assert build is None or build.startswith("cu")
    broken = types.SimpleNamespace(cuda=types.SimpleNamespace(
        get_device_capability=lambda index=0: (_ for _ in ()).throw(RuntimeError("no GPU"))))
    assert litept.spconv_build_needing_compilation(broken) is None
    assert litept._cuda_label("cu126") == "12.6" and litept._cuda_label("cu118") == "11.8"
    crop = litept._synthetic_crop(0.1, 30.0)
    assert crop.dtype == np.float32 and crop.shape == (4000, 3) and np.isfinite(crop).all()
    assert np.array_equal(crop, litept._synthetic_crop(0.1, 30.0))


def test_spconv_install_advice_names_the_rtx50_case():
    import types
    litept = load_plugin_module("core.backends.litept")
    cu128 = types.SimpleNamespace(version=types.SimpleNamespace(cuda="12.8"))
    cu126 = types.SimpleNamespace(version=types.SimpleNamespace(cuda="12.6"))
    assert "CUDA 12.6 build" in litept._spconv_install_advice(cu128)
    assert "Repair dependencies" in litept._spconv_install_advice(cu128)
    assert "CUDA 12.6 build" not in litept._spconv_install_advice(cu126)
    assert "Repair dependencies" in litept._spconv_install_advice(cu126)


def test_compile_spconv_kernels_runs_the_inference_path():
    """The warm-up uses the real crop pipeline; forced here on a GPU with
    native kernels, where it must simply run and report its duration."""
    ok, why = _ready()
    if not ok:
        print(f"SKIP ({why})")
        return
    import torch
    litept = load_plugin_module("core.backends.litept")
    messages = []
    backend = litept.LitePTBackend(
        PLUGIN_ROOT / "core" / "litept_l_dales_10cm.json", _find_weights(), log=messages.append,
    )
    backend.load("cuda")
    try:
        seconds = backend._compile_spconv_kernels(torch, "cu126")
        assert 0.0 <= seconds < 120.0
        assert any("compiles them now" in m for m in messages)
        assert any("spconv kernels ready" in m for m in messages)
        assert any(m.startswith("CUDA memory:") for m in messages), messages
    finally:
        backend.unload()


def _reference_voxel_partition(local, grid_size):
    """The 1.3.2 implementation, kept as the reference for the lean one."""
    grid = np.floor(local.astype(np.float64) / float(grid_size)).astype(np.int64)
    grid -= grid.min(axis=0)
    span = grid.max(axis=0) + 1
    keys = (grid[:, 0] * span[1] + grid[:, 1]) * span[2] + grid[:, 2]
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    return first[order], rank[np.asarray(inverse).reshape(-1)]


def test_lean_voxel_partition_is_identical_to_the_reference():
    """GitHub issue 10: the partition now builds keys chunk by chunk and
    replaces np.unique by a stable sort; the result must not change."""
    litept = load_plugin_module("core.backends.litept")
    rng = np.random.default_rng(10)
    clouds = [
        rng.uniform(-3, 40, (50_000, 3)).astype(np.float32),
        # Many points per voxel, negative and large coordinates, duplicates.
        np.repeat(rng.uniform(-500, 900, (20_000, 3)).astype(np.float32), 7, axis=0),
        np.round(rng.uniform(0, 5, (30_000, 3)), 1).astype(np.float32),
        np.zeros((1, 3), dtype=np.float32),
    ]
    old_chunk = litept._CHUNK
    try:
        for chunk in (old_chunk, 7_919):  # several chunks, uneven last one
            litept._CHUNK = chunk
            for cloud in clouds:
                for grid in (0.05, 0.1, 0.3):
                    keep, projection = litept._voxel_partition(cloud, grid)
                    ref_keep, ref_projection = _reference_voxel_partition(cloud, grid)
                    assert np.array_equal(keep, ref_keep)
                    assert np.array_equal(projection, ref_projection)
                    assert projection.dtype == np.int32
    finally:
        litept._CHUNK = old_chunk


def test_lean_float32_conversion_is_identical():
    litept = load_plugin_module("core.backends.litept")
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0, 1, (25_000, 3)) * [3e3, 5e3, 80] + [6.5e5, 5.7e6, 120.0]
    origin = np.floor(xyz.min(axis=0))
    reference = (xyz - origin).astype(np.float32)
    lean = np.empty(xyz.shape, dtype=np.float32)
    old_chunk = litept._CHUNK
    try:
        litept._CHUNK = 6_007
        for start in range(0, len(xyz), litept._CHUNK):
            lean[start:start + litept._CHUNK] = xyz[start:start + litept._CHUNK] - origin
    finally:
        litept._CHUNK = old_chunk
    assert np.array_equal(lean, reference)


def _fake_backend(fail_times, error_text):
    """A LitePT backend whose first ``fail_times`` forward passes raise
    ``error_text`` (the driver refusing expandable segments)."""
    import torch
    import types
    litept = load_plugin_module("core.backends.litept")
    backend = litept.LitePTBackend(PLUGIN_ROOT / "core/litept_l_dales_10cm.json", "unused.pth",
                                   log=[].append)
    messages = []
    backend.log = messages.append
    state = {"left": fail_times, "calls": 0}

    class Backbone:
        def __call__(self, batch):
            state["calls"] += 1
            if state["left"] > 0:
                state["left"] -= 1
                raise RuntimeError(error_text)
            n = len(batch["coord"])
            return types.SimpleNamespace(feat=torch.ones(n, 8))
    backend.model = (Backbone(), lambda x: x)
    backend.device = "cpu"
    backend.amp = False
    return backend, messages, state


def test_driver_failure_mid_run_switches_allocator_and_retries_the_crop():
    from unittest import mock
    alloc = load_plugin_module("core.cuda_allocator")
    remembered = []
    with mock.patch.object(alloc, "_remember_failure", remembered.append), \
            mock.patch.dict(alloc._active, mode="expandable", chosen_by="auto", signature="sig"), \
            mock.patch.object(alloc, "_set_mode", return_value=True) as set_mode:
        backend, messages, state = _fake_backend(1, "CUDA driver error: unknown error")
        cloud = np.random.default_rng(7).uniform(0, 12, (20000, 3))
        ids = backend.predict(cloud)
        assert len(ids) == len(cloud) and ids.min() >= 1
        assert remembered == ["sig"]
        set_mode.assert_called_with(mock.ANY, "expandable_segments:False")
        assert any("switched to PyTorch's standard allocator" in m for m in messages), messages
        assert alloc._active["mode"] == "standard"


def test_repeated_driver_failure_mid_run_asks_for_a_restart():
    from unittest import mock
    alloc = load_plugin_module("core.cuda_allocator")
    with mock.patch.object(alloc, "_remember_failure", lambda s: None), \
            mock.patch.dict(alloc._active, mode="expandable", chosen_by="auto", signature="sig"), \
            mock.patch.object(alloc, "_set_mode", return_value=True):
        backend, messages, state = _fake_backend(2, "CUDA driver error: unknown error")
        cloud = np.random.default_rng(7).uniform(0, 12, (20000, 3))
        try:
            backend.predict(cloud)
            raise AssertionError("expected the restart advice")
        except RuntimeError as exc:
            assert "Restart QGIS" in str(exc), exc
        assert state["calls"] == 2


def test_other_errors_mid_run_are_raised_unchanged():
    from unittest import mock
    alloc = load_plugin_module("core.cuda_allocator")
    with mock.patch.dict(alloc._active, mode="expandable", chosen_by="auto", signature="sig"):
        backend, messages, state = _fake_backend(1, "CUDA error: an illegal memory access was encountered")
        try:
            backend.predict(np.random.default_rng(7).uniform(0, 12, (2000, 3)))
            raise AssertionError("expected the original error")
        except RuntimeError as exc:
            assert "illegal memory access" in str(exc)
        assert not any("switched" in m for m in messages)


def test_context_buffer_and_reference_origin():
    litept = load_plugin_module("core.backends.litept")
    for card, radius, spacing in (("litept_l_dales_10cm.json", 30.0, None), ("litept_l_mls_5cm.json", 12.0, 11.0)):
        backend = litept.LitePTBackend(PLUGIN_ROOT / "core" / card, "unused.pth")
        assert backend.max_radius == radius
        expected = 2 * radius + (spacing if spacing is not None else 2 * radius / (np.sqrt(3) * 1.1))
        assert abs(backend.context_buffer_m - expected) < 1e-9
        backend._set_crop_budget(50_000, 6.0, 5.5)  # small-GPU profile
        assert backend.context_buffer_m == 2 * 6.0 + 5.5
    backend.set_reference_origin([651234.7, 5712345.2, 101.9])
    assert backend.reference_origin.tolist() == [651234.0, 5712345.0, 101.0]
    backend.set_reference_origin(None)
    assert backend.reference_origin is None


def test_bounded_neighbour_search_selects_the_same_crop_points():
    """1.3.3: the crop search is bounded by the crop radius; the points kept
    (k nearest, then those within the radius) must not change, in dense
    areas where k binds and in sparse areas with few neighbours."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(12)
    dense = rng.normal(0, 3, (60_000, 3))
    sparse = rng.uniform(-200, 200, (3_000, 3))
    grid_like = np.round(rng.uniform(0, 20, (20_000, 3)), 2)  # many equal distances
    cloud = np.vstack([dense, sparse, grid_like]).astype(np.float32)
    tree = cKDTree(cloud)
    radius = 6.0
    bound = float(np.nextafter(radius, np.inf))
    for k in (500, 20_000, len(cloud)):
        for i in rng.choice(len(cloud), 80, replace=False):
            d, idx = tree.query(cloud[i], k=k)
            db, idxb = tree.query(cloud[i], k=k, distance_upper_bound=bound)
            old = np.sort(np.atleast_1d(idx)[np.atleast_1d(d) <= radius])
            new = np.sort(np.atleast_1d(idxb)[np.atleast_1d(db) <= radius])
            if len(old) < k:
                assert np.array_equal(old, new), (k, i)
            else:
                # k binds: the same number of points and the same farthest
                # distance (a tie at that distance may pick another point).
                assert len(new) == len(old)
                assert np.isclose(np.atleast_1d(d)[np.atleast_1d(d) <= radius].max(),
                                  np.atleast_1d(db)[np.atleast_1d(db) <= radius].max())


def test_oom_retry_reaches_5000_point_floor():
    import torch
    import types
    litept = load_plugin_module("core.backends.litept")
    backend = litept.LitePTBackend(PLUGIN_ROOT / "core/litept_l_dales_10cm.json", "unused.pth")
    calls = []
    class Backbone:
        def __call__(self, batch):
            n = len(batch['coord'])
            calls.append(n)
            if n > 5000:
                raise torch.cuda.OutOfMemoryError('injected OOM')
            return types.SimpleNamespace(feat=torch.ones(n, 8))
    backend.model = (Backbone(), lambda x: x)
    backend.device = 'cpu'
    backend.amp = False
    cloud = np.random.default_rng(143).uniform(0, 12, (72000, 3))
    ids = backend.predict(cloud)
    assert len(ids) == len(cloud)
    assert backend.points_per_crop == 5000
    assert any(n > 5000 for n in calls) and any(n <= 5000 for n in calls)
    backend.unload()


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
