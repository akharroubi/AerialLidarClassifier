"""MMS overrides must stay lossless, explicit and isolated from defaults."""

import json
import unittest

from _load import load_plugin_module

mapping = load_plugin_module("utils.class_mapping")
registry = load_plugin_module("core.registry")


class MobileMappingOutputCodes(unittest.TestCase):
    def setUp(self):
        self.spec = registry.get_model(mapping.MOBILE_MAPPING_MODEL_ID)

    def test_partial_overrides_preserve_defaults_and_snapshot_values(self):
        codes = {5: 255, "6": 0}
        selected = mapping.output_model_spec(self.spec, codes)
        codes[5] = 12
        self.assertEqual(selected.class_mapping[5].asprs_code, 255)
        self.assertEqual(selected.class_mapping[6].asprs_code, 0)
        self.assertEqual(self.spec.class_mapping[5].asprs_code, 64)
        self.assertEqual(selected.class_mapping[1].asprs_code, 2)
        selected.class_mapping[1].asprs_code = 99
        self.assertEqual(self.spec.class_mapping[1].asprs_code, 2)

    def test_blank_defaults_and_json_roundtrip(self):
        for blank in (None, "", "  ", {}):
            self.assertEqual(mapping.output_model_spec(self.spec, blank), self.spec)
        selected = mapping.output_model_spec(self.spec, '{"5": 200, "6": 200}')
        saved = mapping.output_codes_json(selected)
        self.assertEqual(len(json.loads(saved)), 9)
        self.assertEqual(mapping.output_model_spec(self.spec, saved), selected)

    def test_invalid_ids_codes_shapes_and_duplicate_ids_are_rejected(self):
        invalid = [
            '[]', 'null', '"text"', '{broken', '{"5": 1, "5": 2}',
            {0: 2}, {10: 2}, {"05": 2}, {True: 2}, {1.0: 2}, {"1.0": 2},
            {5: True}, {5: 2.0}, {5: "2"}, {5: -1}, {5: 256}, {5: None},
            {5: 2, "5": 3}, [2, 3],
        ]
        for codes in invalid:
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                mapping.output_model_spec(self.spec, codes)

    def test_airborne_mapping_is_unchanged_and_rejects_overrides(self):
        for spec in (registry.LITEPT_L_DALES, registry.SEGFORMER3D_URBANFILTERING):
            self.assertFalse(mapping.supports_output_codes(spec))
            self.assertEqual(mapping.output_model_spec(spec), spec)
            with self.assertRaisesRegex(ValueError, "only.*Mobile Mapping"):
                mapping.output_model_spec(spec, {1: 255})

    def test_mobile_mapping_uses_existing_classification_field_only(self):
        for field in (None, "", "classification", " Classification "):
            self.assertEqual(mapping.validate_model_output_field(self.spec, field), "classification")
        with self.assertRaisesRegex(ValueError, "existing LAS 'classification' field only"):
            mapping.validate_model_output_field(self.spec, "mms_ids")
        self.assertEqual(mapping.validate_model_output_field(registry.LITEPT_L_DALES, "raw_ids"), "raw_ids")


if __name__ == "__main__":
    unittest.main()
