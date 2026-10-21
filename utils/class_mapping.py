"""Validate and snapshot per-model output codes without QGIS dependencies."""

from dataclasses import replace
import json


MOBILE_MAPPING_MODEL_ID = "litept_l_mls_5cm"


# ASPRS LAS 1.4 R15 standard classes (Table 17).
ASPRS_CLASS_NAMES = {
    0: "Created, never classified",
    1: "Unclassified",
    2: "Ground",
    3: "Low vegetation",
    4: "Medium vegetation",
    5: "High vegetation",
    6: "Building",
    7: "Low point (noise)",
    8: "Reserved",
    9: "Water",
    10: "Rail",
    11: "Road surface",
    12: "Reserved",
    13: "Wire guard (shield)",
    14: "Wire conductor (phase)",
    15: "Transmission tower",
    16: "Wire-structure connector (insulator)",
    17: "Bridge deck",
    18: "High noise",
    19: "Overhead structure",
    20: "Ignored ground",
    21: "Snow",
    22: "Temporal exclusion",
}


def asprs_code_name(code):
    """ASPRS meaning of a LAS 1.4 classification code (0..255)."""
    code = int(code)
    if code in ASPRS_CLASS_NAMES:
        return ASPRS_CLASS_NAMES[code]
    if 23 <= code <= 63:
        return "Reserved"
    if 64 <= code <= 255:
        return "User definable"
    raise ValueError(f"{code} is not a LAS classification code (0-255)")


def supports_output_codes(spec):
    """True for models whose output codes the user can edit (registry flag)."""
    return bool(getattr(spec, "editable_codes", False))


def editable_model_names():
    from ..core.registry import MODELS
    return ", ".join(spec.short_name for spec in MODELS if supports_output_codes(spec))


def validate_model_output_field(spec, field_name):
    """Models with editable codes write only the existing classification field."""
    field = (field_name or "classification").strip() or "classification"
    if field.lower() == "classification":
        return "classification"
    if supports_output_codes(spec):
        raise ValueError(
            f"{spec.short_name} writes to the existing LAS 'classification' field only. "
            "Set Classification field to 'classification' and customize its output codes."
        )
    return field


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate model class ID {key!r} in output codes.")
        result[key] = value
    return result


def output_model_spec(spec, output_codes=None):
    """Copy a spec with validated output codes; omitted IDs keep defaults.

    JSON uses raw model IDs as keys, for example ``{"5": 64, "6": 65}``.
    Codes may repeat intentionally (merging classes); integers 0..255 only.
    A copy of every ClassInfo prevents UI edits from changing a running task
    or the registry. Raw IDs identify mapping rows; output stores the codes.
    """
    if output_codes is None:
        output_codes = {}
    if isinstance(output_codes, str):
        if not output_codes.strip():
            output_codes = {}
        else:
            try:
                output_codes = json.loads(output_codes, object_pairs_hook=_unique_object)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid output codes JSON: {exc.msg}.") from exc
    if not isinstance(output_codes, dict):
        raise ValueError('Output codes must be a JSON object, for example {"5": 64}.')
    if output_codes and not supports_output_codes(spec):
        raise ValueError(
            "Custom output codes are available only for the models with editable codes "
            f"({editable_model_names()}).")

    mapping = {key: replace(info) for key, info in spec.class_mapping.items()}
    seen = set()
    for raw_id, code in output_codes.items():
        if type(raw_id) is int:
            class_id = raw_id
        elif isinstance(raw_id, str) and raw_id.isascii() and raw_id.isdecimal():
            class_id = int(raw_id)
            if str(class_id) != raw_id:
                raise ValueError(f"Invalid model class ID {raw_id!r}; use plain integer IDs.")
        else:
            raise ValueError(f"Invalid model class ID {raw_id!r}; use integer IDs.")
        if class_id in seen:
            raise ValueError(f"Duplicate model class ID {class_id} in output codes.")
        seen.add(class_id)
        if class_id not in mapping:
            raise ValueError(f"Unknown model class ID {class_id}; valid IDs: {sorted(mapping)}.")
        if type(code) is not int or not 0 <= code <= 255:
            raise ValueError(f"Output code for class {class_id} must be an integer from 0 to 255.")
        mapping[class_id] = replace(mapping[class_id], asprs_code=code)
    return replace(spec, class_mapping=mapping)


def output_codes_json(spec):
    """A reproducible complete mapping, suitable for settings or Processing."""
    return json.dumps({str(key): info.asprs_code for key, info in spec.class_mapping.items()},
                      sort_keys=True)
