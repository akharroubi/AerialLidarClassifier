"""Mobile Mapping backend contracts and optional real CUDA inference.

Set ALC_MLS_WEIGHTS to the released weights file to require the real model
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


CARD = PLUGIN_ROOT / "core" / "litept_l_mls_5cm.json"
WEIGHTS_NAME = "litept_l_mls_5cm_fp16.pth"


def _backend():
    return load_plugin_module("core.backends.litept").LitePTBackend(CARD, "unused.pt")


def test_mls_card_preserves_inference_settings():
    backend = _backend()
    assert backend.num_classes == 9
    assert backend.grid_size == 0.05
    assert (backend.points_per_crop, backend.max_radius, backend.center_spacing) == (240000, 12.0, 11.0)
    assert backend.amp is False
    # The released file is a bare fp16 state dict, like the airborne model.
    assert "weights_state_key" not in backend.card
    assert backend.card["weights_file"] == WEIGHTS_NAME
    assert backend.card["class_names"] == [
        "ground", "low_vegetation", "high_vegetation", "building", "pole_like",
        "vehicle", "fence_barrier", "wire", "unknown",
    ]


def test_mls_reduced_memory_never_expands_crop_radius():
    # GiB as PyTorch reports them: 8 GB cards ~7.99, 12 GB ~11.99, 16 GB ~15.99.
    for memory_gb, expected_points in [(6, 50000), (7.99, 50000), (11.99, 120000),
                                       (15.99, 240000), (24, 240000), (0, 240000)]:
        backend = _backend()
        backend._configure_crop_budget(memory_gb)
        assert backend.points_per_crop == expected_points
        assert backend.max_radius <= 12.0
        assert backend.center_spacing <= 11.0
    airborne = load_plugin_module("core.backends.litept").LitePTBackend(
        PLUGIN_ROOT / "core" / "litept_l_dales_10cm.json", "unused.pth"
    )
    airborne._configure_crop_budget(6)
    assert (airborne.points_per_crop, airborne.max_radius) == (35000, 20.0)


def test_mls_ground_and_unknown_are_real_classes_for_every_raw_point():
    import torch
    backend = _backend()

    class GeometryBackbone:
        def __call__(self, batch):
            return types.SimpleNamespace(feat=batch["feat"])

    def head(features):
        logits = torch.full((len(features), 9), -10.0)
        labels = (features[:, 1] > 1.0).long() * 8
        logits[torch.arange(len(features)), labels] = 10.0
        return logits

    # Two disconnected groups require fill crops when the deliberately tiny
    # nearest-neighbour cap leaves representatives outside the regular crops.
    backend.model = (GeometryBackbone(), head)
    backend.device = "cpu"
    backend.points_per_crop = 4
    backend.center_spacing = 100.0
    xyz = np.array([[x, y, z] for x in (0.0, 50.0) for y in (0.0, 0.4)
                    for z in (0.0, 3.0)], dtype=np.float64)
    xyz = np.repeat(xyz, 3, axis=0) + [500000.0, 5500000.0, 700.0]
    progress = []
    ids = backend.predict(xyz, progress_callback=progress.append)
    assert np.array_equal(ids, np.tile(np.repeat([1, 9], 3), 4))
    assert ids.shape == (len(xyz),)
    assert progress[-1] == 100.0
    assert np.array_equal(backend.predict(xyz), ids)
    try:
        backend.predict(xyz, cancel_callback=lambda: True)
    except InterruptedError:
        pass
    else:
        raise AssertionError("Cancellation did not stop Mobile Mapping inference")


def test_mls_real_cuda_model():
    import torch
    explicit = os.environ.get("ALC_MLS_WEIGHTS")
    weights = _released_weights()
    if explicit and weights is None:
        raise AssertionError("ALC_MLS_WEIGHTS does not exist")
    try:
        import spconv.pytorch  # noqa: F401
    except ImportError:
        if explicit:
            raise
        print("SKIP Mobile Mapping CUDA smoke: spconv unavailable")
        return
    if not torch.cuda.is_available() or weights is None:
        if explicit:
            raise AssertionError("CUDA unavailable for requested Mobile Mapping test")
        print("SKIP Mobile Mapping CUDA smoke: CUDA or the released weights unavailable")
        return
    backend = load_plugin_module("core.backends.litept").LitePTBackend(CARD, weights, log=print)
    backend.load("cuda")
    try:
        rng = np.random.default_rng(812)
        n = 16000
        ground = np.column_stack((rng.uniform(0, 18, n), rng.uniform(0, 18, n),
                                  rng.normal(0, 0.012, n)))
        xyz = np.vstack((ground, ground[:20])) + [500000, 5500000, 100]
        ids = backend.predict(xyz)
        assert ids.shape == (len(xyz),)
        assert ids.min() >= 1 and ids.max() <= 9
        assert np.array_equal(ids[:20], ids[-20:])
        assert np.array_equal(backend.predict(xyz), ids), "Inference is not deterministic"
        # Synthetic random points are outside the scanner's sampling domain;
        # do not invent an accuracy threshold for them. The optional
        # reference audit checks real MLS geometry against fixed predictions.
        share = float((ids[:n] == 1).mean())
        print(f"Mobile Mapping CUDA smoke: {len(ids):,} points, ground {share:.2%}")
    finally:
        backend.unload()


def _released_weights():
    explicit = os.environ.get("ALC_MLS_WEIGHTS")
    appdata = Path(os.environ.get("APPDATA", Path.home()))
    candidates = [Path(explicit)] if explicit else [
        appdata / "QGIS" / "QGIS3" / "profiles" / "default"
        / "AerialLidarClassifier" / "models" / "litept_l_mls_5cm" / WEIGHTS_NAME,
        PLUGIN_ROOT.parent / "model_release" / WEIGHTS_NAME,
    ]
    return next((p for p in candidates if p.is_file()), None)


def test_mls_matches_reference_predictions():
    """Released weights through the plugin backend vs reference predictions.

    Reference: 80,031 points including 31 duplicates. Data is not redistributed, so
    the check runs where the maintainer's validation folder exists.
    """
    import torch
    validation = PLUGIN_ROOT.parent / "validation"
    xyz_path = validation / "mls_reference_input.npy"
    ref_path = validation / "mls_reference_predictions.npy"
    weights = _released_weights()
    if not (xyz_path.is_file() and ref_path.is_file()) or weights is None:
        print("SKIP held-out reference: validation data or weights not on this machine")
        return
    try:
        import spconv.pytorch  # noqa: F401
    except ImportError:
        print("SKIP held-out reference: spconv unavailable")
        return
    if not torch.cuda.is_available():
        print("SKIP held-out reference: CUDA unavailable")
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
    print(f"Held-out reference agreement: {agreement:.4%} ({int((ids != reference).sum())} labels differ)")
    # float32 weights: 3 labels differ; released float16 weights: 10 to 14
    # (CUDA kernels are not bit-reproducible across processes).
    assert agreement >= 0.9995, f"agreement {agreement:.4%} with the reference predictions"
    # Every class the reference predicts is still predicted.
    assert set(np.unique(reference)) == set(np.unique(ids))


if __name__ == "__main__":
    for name, function in sorted(list(globals().items())):
        if name.startswith("test_") and callable(function):
            function()
            print(f"PASS {name}")
