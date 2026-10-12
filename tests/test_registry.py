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



def test_railway_card_codes_and_enum_compatibility():
    import json
    spec = registry.LITEPT_L_RAIL
    # Append-only: the 1.1 / 1.2 MODEL indexes 0-2 keep their models.
    assert [s.id for s in registry.MODELS] == [
        "litept_l_dales_10cm", "segformer3d_urbanfiltering", "litept_l_mls_5cm", "litept_l_rail_5cm"]
    card = json.loads(spec.config_path.read_text(encoding="utf-8"))
    assert card["id"] == spec.id
    assert card["num_classes"] == len(spec.class_mapping) == 7
    assert card["grid_size"] == .05
    assert card["weights_file"] == spec.weights_filename
    assert round(card["weights_bytes"] / 1e6, 1) == spec.weights_size_mb
    assert "weights_sha256" not in card
    assert spec.weights_urls == (
        "https://github.com/akharroubi/AerialLidarClassifier/releases/download/v1.3/"
        + spec.weights_filename,)
    assert card["class_names"] == [c.name.lower() for c in spec.class_mapping.values()]
    # ASPRS where a class exists (Ground 2, Rail 10, Wire 14, Tower 15, High
    # vegetation 5, Unclassified 1); fences take the first user-definable code.
    assert {i: c.asprs_code for i, c in spec.class_mapping.items()} == {
        1: 2, 2: 10, 3: 15, 4: 14, 5: 5, 6: 64, 7: 1}
    assert spec.editable_codes and registry.LITEPT_L_MLS.editable_codes
    assert not registry.LITEPT_L_DALES.editable_codes
    assert not registry.SEGFORMER3D_URBANFILTERING.editable_codes
    assert spec.supported_devices == ("cuda",)
    assert spec.extra_packages == ("spconv", "scipy")
    assert "CC BY-NC 4.0" in spec.licence


def test_user_defined_default_codes_start_at_64_without_gaps():
    for spec in registry.MODELS:
        user = sorted({c.asprs_code for c in spec.class_mapping.values() if c.asprs_code >= 64})
        assert user == list(range(64, 64 + len(user))), (spec.id, user)


def test_classifiers_cover_every_model_once():
    assert [c.id for c in registry.CLASSIFIERS] == ["airborne", "mobile_mapping", "railway"]
    assert registry.DEFAULT_CLASSIFIER_ID == "airborne"
    engines = [spec.id for c in registry.CLASSIFIERS for spec in c.engines]
    assert sorted(engines) == sorted(spec.id for spec in registry.MODELS)
    assert registry.AIRBORNE.primary is registry.LITEPT_L_DALES
    for spec in registry.MODELS:
        classifier = registry.classifier_for_model(spec.id)
        assert spec in classifier.engines
    assert registry.get_classifier("railway") is registry.RAILWAY
    for bad in (None, "", "unknown"):
        for lookup in (registry.get_classifier, registry.classifier_for_model):
            try:
                lookup(bad)
            except KeyError:
                pass
            else:
                raise AssertionError(f"{lookup.__name__}({bad!r}) did not raise")


def test_select_engine_prefers_litept_and_falls_back_only_for_airborne():
    def runs(ready):
        return lambda spec, device: spec.supports_device(device) and (spec.family != "litept" or ready)

    airborne, mls, rail = registry.CLASSIFIERS
    assert registry.select_engine(airborne, "cuda", runs(True)) is registry.LITEPT_L_DALES
    for device, ready in (("cuda", False), ("cpu", True), ("mps", True), ("cpu", False)):
        assert registry.select_engine(airborne, device, runs(ready)) is registry.SEGFORMER3D_URBANFILTERING
    # No airborne model ever replaces the mobile mapping or railway model:
    # the primary engine comes back so its gate can explain what is missing.
    for classifier, spec in ((mls, registry.LITEPT_L_MLS), (rail, registry.LITEPT_L_RAIL)):
        for device, ready in (("cuda", True), ("cuda", False), ("cpu", True), ("mps", False)):
            assert registry.select_engine(classifier, device, runs(ready)) is spec

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
