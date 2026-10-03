"""Consistency checks for core/registry.py (no QGIS, no torch needed)."""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_plugin_module  # noqa: E402

registry = load_plugin_module("core.registry")


def test_every_model_has_its_bundled_config():
    for spec in registry.MODELS:
        assert spec.config_path.is_file(), f"{spec.id}: {spec.config_path} missing"


def test_hashes_and_urls_look_right():
    for spec in registry.MODELS:
        assert re.fullmatch(r"[0-9a-f]{64}", spec.weights_sha256), spec.id
        # Every model downloads on first use: nothing is bundled in the ZIP.
        assert spec.weights_urls and all(u.startswith("https://") for u in spec.weights_urls)
        assert spec.weights_filename == spec.weights_urls[0].rsplit("/", 1)[-1]
        assert 0 < spec.weights_size_mb < 400


def test_class_mappings_are_asprs_codes():
    for spec in registry.MODELS:
        ids = sorted(spec.class_mapping)
        assert ids == list(range(1, len(ids) + 1)), f"{spec.id}: ids must be 1..n"
        for info in spec.class_mapping.values():
            assert 0 <= info.asprs_code <= 255
            assert info.model_id in spec.class_mapping


def test_ids_and_families_are_unique_and_known():
    ids = [spec.id for spec in registry.MODELS]
    assert len(set(ids)) == len(ids)
    assert {spec.family for spec in registry.MODELS} <= {"litept", "segformer3d"}
    assert registry.get_model(registry.DEFAULT_MODEL_ID).family == "litept"


def test_default_model_per_device():
    assert registry.default_model_for_device("cuda").id == registry.DEFAULT_MODEL_ID
    assert registry.default_model_for_device("cpu").id == registry.FALLBACK_MODEL_ID
    assert registry.default_model_for_device("mps").id == registry.FALLBACK_MODEL_ID
    assert not registry.get_model(registry.DEFAULT_MODEL_ID).supports_device("cpu")
    assert registry.get_model(registry.FALLBACK_MODEL_ID).supports_device("cpu")


def test_litept_card_matches_spec():
    import json
    spec = registry.get_model("litept_l_dales_10cm")
    card = json.loads(spec.config_path.read_text(encoding="utf-8"))
    # The bundled card carries no hash (the plugin repository's secret
    # scanner flags hex strings); the registry is the source of truth.
    assert "weights_sha256" not in card
    assert card["weights_file"] == spec.weights_filename
    assert card["num_classes"] == len(spec.class_mapping)
    assert [c.lower() for c in card["class_names"]] == [
        info.name.lower().replace(" ", "_") for info in spec.class_mapping.values()
    ]


def test_mobile_mapping_card_and_enum_compatibility():
    import json
    assert [s.id for s in registry.MODELS[:2]] == [
        "litept_l_dales_10cm", "segformer3d_urbanfiltering"]
    spec = registry.LITEPT_L_MLS
    assert registry.MODELS[2] is spec
    assert "Mobile Mapping" in spec.display_name
    assert "Airborne" in registry.LITEPT_L_DALES.display_name
    card = json.loads(spec.config_path.read_text(encoding="utf-8"))
    assert card["num_classes"] == len(spec.class_mapping) == 9
    assert card["grid_size"] == .05
    assert card["weights_file"] == spec.weights_filename
    assert round(card["weights_bytes"] / 1e6, 1) == spec.weights_size_mb
    assert spec.weights_urls == (
        "https://github.com/akharroubi/AerialLidarClassifier/releases/download/v1.2/"
        + spec.weights_filename,)
    # Default output codes: ASPRS where a code exists, user-definable 64+ otherwise.
    assert {i: c.asprs_code for i, c in spec.class_mapping.items()} == {
        1: 2, 2: 3, 3: 5, 4: 6, 5: 64, 6: 65, 7: 66, 8: 14, 9: 1}
    # No model name carries an em dash (user-facing labels).
    assert all("—" not in s.display_name for s in registry.MODELS)
    assert card["class_names"] == [
        c.name.lower().replace(" ", "_") for c in spec.class_mapping.values()]


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
