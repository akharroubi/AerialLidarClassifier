"""Railway backend contracts and optional real CUDA inference.

Set ALC_RAIL_WEIGHTS to the released weights file to require the real model
smoke test. Without it, the weights downloaded into the QGIS profile (or
the maintainer's model_release folder) are used when present, and CUDA
inference skips if this machine lacks CUDA/spconv or the weights.
"""

import os
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import PLUGIN_ROOT, load_plugin_module


CARD = PLUGIN_ROOT / "core" / "litept_l_rail_5cm.json"
WEIGHTS_NAME = "litept_l_rail_5cm_fp16.pth"
CLASSES = ["ground", "rail", "pole", "wire", "vegetation", "fence", "other"]


def _backend():
    return load_plugin_module("core.backends.litept").LitePTBackend(CARD, "unused.pt")


def test_rail_card_preserves_inference_settings():
    backend = _backend()
    assert backend.num_classes == 7
    assert backend.in_channels == 2
    assert backend.grid_size == 0.05
    assert (backend.points_per_crop, backend.max_radius, backend.center_spacing) == (240000, 12.0, 11.0)
    assert backend.amp is False
    assert backend.card["weights_file"] == WEIGHTS_NAME
    assert backend.card["class_names"] == CLASSES
    assert backend.card["features"] == ["constant 1", "height above crop minimum after mean centering"]
    # Same network as the Mobile Mapping model except the 7-class head.
    mls = load_plugin_module("core.backends.litept").LitePTBackend(
        PLUGIN_ROOT / "core" / "litept_l_mls_5cm.json", "unused.pt")
    assert backend.architecture == mls.architecture
    assert backend.card["eval"] == mls.card["eval"]


def test_rail_reduced_memory_never_expands_crop_radius():
    for memory_gb, expected_points in [(6, 50000), (7.99, 50000), (11.99, 120000),
                                       (15.99, 240000), (24, 240000), (0, 240000)]:
        backend = _backend()
        backend._configure_crop_budget(memory_gb)
        assert backend.points_per_crop == expected_points
        assert backend.max_radius <= 12.0
        assert backend.center_spacing <= 11.0


def test_rail_every_raw_point_gets_a_class_from_1_to_7():
    import torch
    backend = _backend()

    class GeometryBackbone:
        def __call__(self, batch):
            return types.SimpleNamespace(feat=batch["feat"])

    def head(features):
        # 1.5 m height bands above the crop floor stand in for the seven classes.
        logits = torch.full((len(features), 7), -10.0)
        labels = torch.clamp(((features[:, 1] + 0.75) / 1.5).long(), 0, 6)
        logits[torch.arange(len(features)), labels] = 10.0
        return logits

    backend.model = (GeometryBackbone(), head)
    backend.device = "cpu"
    # One crop holds one group; the second group needs a fill crop.
    backend.points_per_crop = 7
    backend.center_spacing = 100.0
    heights = np.arange(7) * 1.5 + 0.2
    xyz = np.array([[x, 0.0, z] for x in (0.0, 60.0) for z in heights], dtype=np.float64)
    xyz = np.repeat(xyz, 2, axis=0) + [190000.0, 460000.0, -3.0]
    progress = []
    ids = backend.predict(xyz, progress_callback=progress.append)
    assert ids.shape == (len(xyz),)
    assert ids.min() >= 1 and ids.max() <= 7
    assert set(np.unique(ids)) == set(range(1, 8))
    assert np.array_equal(ids[0::2], ids[1::2]), "duplicate points must share a label"
    assert progress[-1] == 100.0
    try:
        backend.predict(xyz, cancel_callback=lambda: True)
    except InterruptedError:
        pass
    else:
        raise AssertionError("Cancellation did not stop Railway inference")


def _released_weights():
    explicit = os.environ.get("ALC_RAIL_WEIGHTS")
    appdata = Path(os.environ.get("APPDATA", Path.home()))
    candidates = [Path(explicit)] if explicit else [
        appdata / "QGIS" / "QGIS3" / "profiles" / "default"
        / "AerialLidarClassifier" / "models" / "litept_l_rail_5cm" / WEIGHTS_NAME,
        PLUGIN_ROOT.parent / "model_release" / WEIGHTS_NAME,
    ]
    return next((p for p in candidates if p.is_file()), None)


def _cuda_ready(label):
    import torch
    explicit = os.environ.get("ALC_RAIL_WEIGHTS")
    weights = _released_weights()
    if explicit and weights is None:
        raise AssertionError("ALC_RAIL_WEIGHTS does not exist")
    try:
        import spconv.pytorch  # noqa: F401
    except ImportError:
        if explicit:
            raise
        print(f"SKIP {label}: spconv unavailable")
        return None
    if not torch.cuda.is_available() or weights is None:
        if explicit:
            raise AssertionError(f"CUDA unavailable for requested {label}")
        print(f"SKIP {label}: CUDA or the released weights unavailable")
        return None
    return weights


def test_rail_real_cuda_model():
    weights = _cuda_ready("Railway CUDA smoke")
    if weights is None:
        return
    backend = load_plugin_module("core.backends.litept").LitePTBackend(CARD, weights, log=print)
    backend.load("cuda")
    try:
        rng = np.random.default_rng(1435)
        n = 16000
        ground = np.column_stack((rng.uniform(0, 18, n), rng.uniform(0, 18, n),
                                  rng.normal(0, 0.012, n)))
        xyz = np.vstack((ground, ground[:20])) + [190000, 460000, 5]
        ids = backend.predict(xyz)
        assert ids.shape == (len(xyz),)
        assert ids.min() >= 1 and ids.max() <= 7
        assert np.array_equal(ids[:20], ids[-20:])
        assert np.array_equal(backend.predict(xyz), ids), "Inference is not deterministic"
        print(f"Railway CUDA smoke: {len(ids):,} points, ground {float((ids[:n] == 1).mean()):.2%}")
    finally:
        backend.unload()


def test_rail_matches_reference_predictions():
    """Released weights through the plugin backend vs the reference pipeline.

    Reference: 267,004 points of a railway scan (31 duplicates at the end),
    classified by the model's reference implementation. The data is not
    redistributed, so the check runs where the maintainer's validation folder exists.
    """
    validation = PLUGIN_ROOT.parent / "validation"
    xyz_path = validation / "rail_reference_input.npy"
    ref_path = validation / "rail_reference_predictions.npy"
    if not (xyz_path.is_file() and ref_path.is_file()):
        print("SKIP railway reference: validation data not on this machine")
        return
    weights = _cuda_ready("railway reference")
    if weights is None:
        return
    xyz = np.load(xyz_path)
    reference = np.load(ref_path)
    backend = load_plugin_module("core.backends.litept").LitePTBackend(CARD, weights, log=print)
    backend.load("cuda")
    try:
        ids = backend.predict(xyz)
    finally:
        backend.unload()
    assert ids.shape == reference.shape
    assert np.array_equal(ids[:31], ids[-31:]), "duplicate points must share a label"
    agreement = float((ids == reference).mean())
    print(f"Railway reference agreement: {agreement:.4%} ({int((ids != reference).sum())} labels differ)")
    # Whole tiles: 99.996 to 99.998 % with the released float16 weights,
    # 99.999 % with float32 (CUDA kernels are not bit-reproducible).
    assert agreement >= 0.9995, f"agreement {agreement:.4%} with the reference predictions"
    assert set(np.unique(reference)) <= set(np.unique(ids)) | {7}


if __name__ == "__main__":
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
