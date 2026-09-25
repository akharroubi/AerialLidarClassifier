"""Release regressions runnable with the plugin Python, without QGIS."""
import hashlib
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import laspy
import numpy as np
from _load import load_plugin_module

safety = load_plugin_module('utils.output_safety')
weights = load_plugin_module('utils.weights')
registry = load_plugin_module('core.registry')


class ReleaseSafety(unittest.TestCase):
    def test_batch_alias_and_duplicate_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b, link = [Path(folder) / n for n in ('a.las', 'a_out.las', 'link.las')]
            a.write_bytes(b'a'); b.write_bytes(b'b')
            with self.assertRaises(ValueError):
                safety.validate_output_paths([a, b], [b, Path(folder) / 'b_out.las'])
            os.link(a, link)
            with self.assertRaises(ValueError):
                safety.validate_output_paths([a], [link])
            with self.assertRaises(ValueError):
                safety.validate_output_paths([a], [b, b])
            self.assertEqual(a.read_bytes(), b'a')
            self.assertEqual(b.read_bytes(), b'b')

    def test_atomic_failure_retains_existing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'result.laz'
            target.write_bytes(b'good')
            with self.assertRaises(InterruptedError):
                with safety.atomic_output_path(target) as temporary:
                    self.assertEqual(temporary.suffix, '.laz')
                    temporary.write_bytes(b'partial')
                    raise InterruptedError()
            self.assertEqual(target.read_bytes(), b'good')
            self.assertEqual(list(Path(folder).iterdir()), [target])

    def test_standard_dimensions_cannot_be_label_fields(self):
        for name in ('X', 'x', 'red', 'intensity', 'gps_time', 'scan_angle_rank'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                safety.validate_label_field(laspy.PointFormat(6), name, laspy)

    def test_incompatible_extra_fields_rejected(self):
        for dtype, options in [('float64', {}), ('3int16', {}), ('uint64', {'scales': [.01], 'offsets': [0]})]:
            h = laspy.LasHeader(point_format=6, version='1.4')
            h.add_extra_dim(laspy.ExtraBytesParams(name='labels', type=dtype, **options))
            with self.assertRaises(ValueError):
                safety.validate_label_field(h.point_format, 'labels', laspy)

    def test_raw_classes_and_unmapped_rejection(self):
        ids = np.array([3, 4, 6], dtype=np.int32)
        mapping = registry.LITEPT_L_DALES.class_mapping
        np.testing.assert_array_equal(safety.label_values(ids, mapping, True), ids)
        np.testing.assert_array_equal(safety.label_values(ids, mapping), [1, 1, 1])
        with self.assertRaises(ValueError):
            safety.label_values(np.array([0]), mapping)
        with self.assertRaises(ValueError):
            safety.checked_predictions(np.array([1]), 2)

    def test_added_dimension_preserves_scaled_uint64_bytes(self):
        h = laspy.LasHeader(point_format=6, version='1.4')
        h.add_extra_dim(laspy.ExtraBytesParams(name='scaled', type='uint64', scales=[.001], offsets=[100]))
        d = laspy.LasData(h)
        d.points = laspy.ScaleAwarePointRecord.zeros(9, header=h)
        original = np.uint64(2**63) + np.arange(9, dtype=np.uint64)
        d.points.array['scaled'] = original
        safety.add_extra_dim_preserving_raw(d, laspy.ExtraBytesParams(name='labels', type='int32'))
        np.testing.assert_array_equal(d.points.array['scaled'], original)

    def test_upgrade_preserves_rgb_and_legacy_angle(self):
        for source, target in [(1, 6), (3, 7), (5, 10)]:
            h = laspy.LasHeader(point_format=source, version='1.3')
            d = laspy.LasData(h)
            d.points = laspy.ScaleAwarePointRecord.zeros(3, header=h)
            d.scan_angle_rank = [-89, 1, 89]
            if source in (3, 5):
                d.red = [123, 456, 789]
            new = safety.upgrade_header_preserving_fields(h, laspy)
            converted = safety.convert_points_preserving_fields(d.points, new.point_format, new.scales, new.offsets, laspy)
            self.assertEqual(new.point_format.id, target)
            np.testing.assert_array_equal(converted['scan_angle_rank'], d.scan_angle_rank)
            np.testing.assert_allclose(converted.scan_angle * .006, d.scan_angle_rank, atol=.0031)
            if source in (3, 5):
                np.testing.assert_array_equal(converted.red, d.red)

    def test_corrupted_cached_weights_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'weights.pth'
            p.write_bytes(b'approved')
            digest = hashlib.sha256(b'approved').hexdigest()
            with patch.object(registry, 'get_model', return_value=types.SimpleNamespace(weights_sha256=digest)):
                self.assertTrue(weights.is_verified(p, digest))
                with weights.verified_weights(p, 'test') as stream:
                    self.assertEqual(stream.read(), b'approved')
                p.write_bytes(b'tampered')
                self.assertFalse(weights.is_verified(p, digest))
                with self.assertRaises(ValueError):
                    with weights.verified_weights(p, 'test'):
                        self.fail('tampered weights opened')


if __name__ == '__main__':
    unittest.main(verbosity=2)
