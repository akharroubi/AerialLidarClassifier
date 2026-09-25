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
