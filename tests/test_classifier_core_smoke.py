"""End-to-end smoke test of ``filterPoints`` on synthetic clouds.

Needs the plugin venv Python (torch, numpy_indexed, timm) and the model
weights; the weight-dependent tests skip when the file is not found.
Weights are looked up in ``$ALC_MODEL_PATH``, the QGIS default profile
cache, and ``<repo>/model/``. Run with::

    ~/.qgis_aerial_lidar_classifier/venv_py3.12/Scripts/python.exe \\
        tests/test_classifier_core_smoke.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import PLUGIN_ROOT, load_plugin_module  # noqa: E402

core = load_plugin_module("core.classifier_core")

CONFIG = PLUGIN_ROOT / "core" / "model_config.json"
WEIGHTS_NAME = "urbanfiltering_als_esegformer3D_112_30cm_GPU3GB.pth"


def _find_weights():
    candidates = [
        os.environ.get("ALC_MODEL_PATH", ""),
        Path(os.environ.get("APPDATA", "")) / "QGIS" / "QGIS3" / "profiles"
        / "default" / "AerialLidarClassifier" / "models" / WEIGHTS_NAME,
        PLUGIN_ROOT.parent.parent / "model" / WEIGHTS_NAME,
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _cloud(n, xy_range, z_range, seed):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        rng.uniform(0, xy_range, n),
        rng.uniform(0, xy_range, n),
        rng.uniform(0, z_range, n),
    ])


def test_resolve_device_accepts_legacy_booleans():
    assert core.resolve_device(False) == "cpu"
    assert core.resolve_device(None) == "cpu"
    assert core.resolve_device("CPU") == "cpu"


def test_resolve_device_rejects_unavailable_backends():
    import torch
    if not torch.cuda.is_available():
        try:
            core.resolve_device("cuda")
            raise AssertionError("expected a RuntimeError for cuda")
        except RuntimeError as exc:
            assert "CUDA" in str(exc)
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        try:
            core.resolve_device("mps")
            raise AssertionError("expected a RuntimeError for mps")
        except RuntimeError as exc:
            assert "MPS" in str(exc)
    try:
        core.resolve_device("tpu")
        raise AssertionError("expected a RuntimeError for an unknown device")
    except RuntimeError:
        pass


def test_missing_config_raises_instead_of_returning_none():
    try:
        core.filterPoints(
            "/nonexistent/model_config.json", np.zeros((10, 3)),
            "/nonexistent/model.pth", if_bottom_only=False, device="cpu",
        )
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "configuration" in str(exc)


def test_flat_cloud_gets_real_predictions_on_cpu():
    """Z range 2 m: v1.0.2 left 99.5 % of these points at fill value 7."""
    weights = _find_weights()
    if weights is None:
        print("SKIP (weights not found)")
        return
    pts = _cloud(n=40_000, xy_range=60.0, z_range=2.0, seed=1)
    preds = core.filterPoints(
        str(CONFIG), pts, weights, if_bottom_only=False, device="cpu",
    )
    assert preds.shape == (len(pts),)
    assert preds.dtype == np.int32
    assert preds.min() >= 0 and preds.max() <= 7
    predicted = float((preds > 0).mean())
    assert predicted > 0.99, f"only {predicted:.2%} of points got a prediction"


def test_cuda_and_cpu_agree_when_cuda_is_available():
    import torch
    weights = _find_weights()
    if weights is None or not torch.cuda.is_available():
        print("SKIP (no weights or no CUDA)")
        return
    pts = _cloud(n=40_000, xy_range=60.0, z_range=25.0, seed=2)
    p_cpu = core.filterPoints(
        str(CONFIG), pts, weights, if_bottom_only=False, device="cpu")
    p_gpu = core.filterPoints(
        str(CONFIG), pts, weights, if_bottom_only=False, device="cuda")
    agreement = float((p_cpu == p_gpu).mean())
    # CPU and GPU kernels differ at float precision; a few borderline
    # voxels may flip, whole classes must not.
    assert agreement > 0.98, f"CPU/CUDA agreement only {agreement:.2%}"


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
