"""Real QGIS/untwine viewing checks plus failure and cancellation regressions."""
import os
import sys
from pathlib import Path
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from qgis.core import QgsApplication, QgsPointCloudLayer
app = QgsApplication([], False)
app.initQgis()
try:
    from qgis.PyQt import sip
except ImportError:
    import sip

import hashlib
import builtins
import importlib
import shutil
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.append(str(Path.home() / '.qgis_aerial_lidar_classifier/venv_py3.12/Lib/site-packages'))
import laspy
import numpy as np

view = importlib.import_module(ROOT.name + '.utils.pointcloud_view')
proc = importlib.import_module(ROOT.name + '.utils.proc')


def make_source(path, extras=True, legacy=False):
    header = laspy.LasHeader(point_format=3 if legacy else 7, version='1.2' if legacy else '1.4')
    header.scales = [.001] * 3
    header.offsets = [500000., 5600000., 100.]
    if extras:
        header.add_extra_dim(laspy.ExtraBytesParams(name='source_index', type='uint32'))
    cloud = laspy.LasData(header)
    cloud.points = laspy.ScaleAwarePointRecord.zeros(50, header=header)
    i = np.arange(50)
    cloud.X = i % 10 * 1000
    cloud.Y = i // 10 * 1000
    cloud.Z = i % 3 * 500
    cloud.classification = np.asarray(([2, 3, 5, 6, 1] if legacy else [2, 64, 65, 66, 255]) * 10)
    cloud.intensity = i + 200
    cloud.red = i + 1000
    cloud.green = i + 2000
    cloud.blue = i + 3000
    if extras:
        cloud.source_index = i
    cloud.write(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PointCloudViews(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_dir = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.fixture_dir.name)
        cls.primary = cls.fixture / 'classified.las'
        make_source(cls.primary)
        cls.primary_digest = digest(cls.primary)
        warnings = []
        cls.companion = view.prepare_qgis_view(cls.primary, laspy_module=laspy,
            warning_callback=warnings.append, info_callback=lambda message: None, timeout_seconds=60)
        if cls.companion == cls.primary:
            raise AssertionError('Actual untwine conversion failed: ' + '; '.join(warnings))

    @classmethod
    def tearDownClass(cls):
        cls.fixture_dir.cleanup()

    def test_actual_untwine_companion_reads_with_native_qgis_provider(self):
        self.assertEqual(digest(self.primary), self.primary_digest)
        self.assertNotEqual(self.companion, self.primary)
        self.assertTrue(self.companion.name.endswith('.qgis-view.copc.laz'))
        self.assertEqual(list(self.fixture.glob('.qgis-view-*')), [])
        source = laspy.read(self.primary)
        result = laspy.read(self.companion)
        self.assertEqual(len(source.points), len(result.points))
        order = np.argsort(result.source_index)
        for name in ('classification', 'intensity', 'red', 'green', 'blue', 'source_index'):
            np.testing.assert_array_equal(source[name], np.asarray(result[name])[order])
        for name in ('x', 'y', 'z'):
            np.testing.assert_allclose(source[name], np.asarray(result[name])[order], atol=.001, rtol=0)
        options = QgsPointCloudLayer.LayerOptions()
        options.skipIndexGeneration = True
        options.skipStatisticsCalculation = True
        options.skipCrsValidation = True
        layer = QgsPointCloudLayer(str(self.companion), 'Classified view', 'copc', options)
        self.assertTrue(layer.isValid(), layer.error().message())
        self.assertEqual(layer.dataProvider().pointCount(), 50)
        self.assertIn('source_index', [a.name() for a in layer.dataProvider().attributes().attributes()])

    def test_no_converter_for_standard_only_legacy_or_existing_copc(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(proc, 'Background') as start:
            folder = Path(folder)
            for label, extras, legacy in [('standard', False, False), ('legacy', True, True)]:
                path = folder / (label + '.las')
                make_source(path, extras=extras, legacy=legacy)
                self.assertEqual(view.prepare_qgis_view(path, laspy_module=laspy), path)
            self.assertEqual(view.prepare_qgis_view(self.companion, laspy_module=laspy), self.companion)
            self.assertEqual(view.prepare_qgis_view(self.primary, cancel_callback=lambda: True), self.primary)
            start.assert_not_called()

    def fake_process(self, mode, state):
        fixture = self.companion

        class Process:
            returncode = 7 if mode == 'failure' else 0

            def __init__(self, command, stdout, stderr):
                state['command'] = command
                state['work'] = Path(stderr).parent
                self.active = mode in ('cancel', 'timeout')
                target = Path(command[command.index('--output_dir') + 1])
                shutil.copyfile(fixture, target)
                if mode == 'count':
                    with target.open('r+b') as stream:
                        stream.seek(247)
                        stream.write((49).to_bytes(8, 'little'))
                if mode == 'bounds':
                    with target.open('r+b') as stream:
                        stream.seek(179)
                        stream.write(struct.pack('<d', 999999.))
                Path(stderr).write_text('bad conversion' if mode == 'failure' else '', encoding='utf8')

            def wait(self, timeout):
                if mode in ('cancel', 'done_cancel'):
                    state['cancelled'] = True
                return not self.active

            def running(self):
                return self.active

            def stop(self):
                state['stopped_before_cleanup'] = state['work'].is_dir()
                self.active = False

        return Process

    def test_failure_cancellation_timeout_and_count_mismatch_keep_primary(self):
        for mode in ('failure', 'cancel', 'done_cancel', 'timeout', 'count', 'bounds'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                primary = folder / 'classified.las'
                shutil.copyfile(self.primary, primary)
                state, warnings = {}, []
                with patch.object(proc, 'Background', self.fake_process(mode, state)), \
                     patch.object(view, '_untwine_path', return_value=Path(__file__)), \
                     patch.object(view.time, 'monotonic', side_effect=[0., 10.]):
                    result = view.prepare_qgis_view(primary, laspy_module=laspy,
                        cancel_callback=lambda: state.get('cancelled', False),
                        info_callback=lambda message: None, warning_callback=warnings.append,
                        timeout_seconds=1)
                self.assertEqual(result, primary)
                self.assertEqual(digest(primary), self.primary_digest)
                self.assertEqual(list(folder.iterdir()), [primary])
                self.assertTrue(warnings)
                if mode in ('cancel', 'timeout'):
                    self.assertTrue(state['stopped_before_cleanup'])

    def test_published_view_survives_a_temp_cleanup_failure_and_large_files_get_more_time(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            primary = folder / 'classified.las'
            shutil.copyfile(self.primary, primary)
            state = {}
            real_rmtree = shutil.rmtree

            def locked_rmtree(path, ignore_errors=False, **kwargs):
                # A scanner still holds a file: the folder stays, nothing raises.
                if not ignore_errors:
                    raise PermissionError('locked by an antivirus scanner')
            with patch.object(proc, 'Background', self.fake_process('success', state)), \
                 patch.object(view, '_untwine_path', return_value=Path(__file__)), \
                 patch.object(view.shutil, 'rmtree', side_effect=locked_rmtree):
                result = view.prepare_qgis_view(primary, laspy_module=laspy,
                                                info_callback=lambda message: None)
            self.assertNotEqual(result, primary)
            self.assertTrue(result.is_file())
            real_rmtree(state['work'])
        # 50 points: the 30-minute floor; 1e9 points: scaled up.
        self.assertEqual(max(view._MIN_TIMEOUT_S, 50 / view._POINTS_PER_TIMEOUT_SECOND), 1800)
        self.assertGreater(1e9 / view._POINTS_PER_TIMEOUT_SECOND, view._MIN_TIMEOUT_S)

    def test_stale_views_are_removed_unless_used_and_folder_scans_skip_them(self):
        las_utils = importlib.import_module(ROOT.name + '.utils.las_utils')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            primary = folder / 'street_classified.laz'
            primary.write_bytes(b'primary')
            other = folder / 'other_classified.0123456789ab.qgis-view.copc.laz'
            other.write_bytes(b'view of another output')
            old, loaded, new = (folder / f'street_classified.{tag}.qgis-view.copc.laz'
                                for tag in ('aaaaaaaaaaaa', 'bbbbbbbbbbbb', 'cccccccccccc'))
            for path in (old, loaded, new):
                path.write_bytes(b'view')
            unrelated = folder / 'street_classified.notes.txt'
            unrelated.write_bytes(b'keep')
            self.assertEqual(las_utils.find_las_files(folder), [primary])
            # A view dropped explicitly as a file is still accepted.
            self.assertEqual(las_utils.find_las_files(new), [new])
            removed = view.remove_stale_views(primary, keep=new, in_use=[str(loaded), 'memory?'])
            self.assertEqual(removed, [old])
            self.assertEqual(sorted(p.name for p in folder.iterdir()),
                             sorted(p.name for p in (primary, other, loaded, new, unrelated)))
            self.assertEqual(view.remove_stale_views(folder / 'missing_folder' / 'x.las'), [])

    def test_missing_converter_and_failed_publication_keep_primary(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            primary = folder / 'classified.las'
            shutil.copyfile(self.primary, primary)
            warnings = []
            with patch.object(view, '_untwine_path', return_value=folder / 'missing.exe'), \
                 patch.object(proc, 'Background') as start:
                self.assertEqual(view.prepare_qgis_view(primary, warning_callback=warnings.append), primary)
                start.assert_not_called()
            state = {}
            with patch.object(proc, 'Background', self.fake_process('success', state)), \
                 patch.object(view, '_untwine_path', return_value=Path(__file__)), \
                 patch.object(view.os, 'replace', side_effect=PermissionError('locked')):
                result = view.prepare_qgis_view(primary, warning_callback=warnings.append,
                                               info_callback=lambda message: None)
            self.assertEqual(result, primary)
            self.assertEqual(digest(primary), self.primary_digest)
            self.assertEqual(list(folder.iterdir()), [primary])
            self.assertTrue(any('locked' in message for message in warnings))

    def test_collision_and_classification_mismatch_never_publish_or_overwrite(self):
        for mode in ('collision', 'classes'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                folder = Path(folder)
                primary = folder / 'classified.las'
                shutil.copyfile(self.primary, primary)
                existing = folder / 'classified.0123456789ab.qgis-view.copc.laz'
                if mode == 'collision':
                    existing.write_bytes(b'existing unrelated file')
                state, warnings = {}, []
                with patch.object(proc, 'Background', self.fake_process('success', state)), \
                     patch.object(view, '_untwine_path', return_value=Path(__file__)), \
                     patch.object(view.uuid, 'uuid4', return_value=types.SimpleNamespace(hex='0123456789abcdef')):
                    if mode == 'classes':
                        with patch.object(view, '_class_counts', side_effect=[np.array([1]), np.array([2])]):
                            result = view.prepare_qgis_view(primary, laspy_module=laspy,
                                info_callback=lambda message: None, warning_callback=warnings.append)
                    else:
                        result = view.prepare_qgis_view(primary, laspy_module=laspy,
                            info_callback=lambda message: None, warning_callback=warnings.append)
                self.assertEqual(result, primary)
                self.assertEqual(digest(primary), self.primary_digest)
                self.assertFalse(state['work'].exists())
                self.assertTrue(warnings)
                if mode == 'collision':
                    self.assertEqual(existing.read_bytes(), b'existing unrelated file')
                else:
                    self.assertFalse(existing.exists())

    def test_success_never_reuses_stale_companion(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            primary = folder / 'classified.las'
            shutil.copyfile(self.primary, primary)
            stale = folder / 'classified.copc.laz'
            stale.write_bytes(b'stale index from an earlier run')
            state = {}
            with patch.object(proc, 'Background', self.fake_process('success', state)), \
                 patch.object(view, '_untwine_path', return_value=Path(__file__)):
                result = view.prepare_qgis_view(primary, laspy_module=laspy, info_callback=lambda message: None)
            self.assertNotEqual(result, primary)
            self.assertNotEqual(result, stale)
            self.assertTrue(result.is_file())
            self.assertEqual(stale.read_bytes(), b'stale index from an earlier run')
            self.assertEqual(digest(primary), self.primary_digest)
            self.assertFalse(state['work'].exists())

    def rendering_layer(self):
        options = QgsPointCloudLayer.LayerOptions()
        options.skipIndexGeneration = True
        options.skipStatisticsCalculation = True
        options.skipCrsValidation = True
        return QgsPointCloudLayer(str(self.companion), 'MMS rendering test', 'copc', options)

    def test_mms_renderers_enable_default_and_custom_merged_codes(self):
        helpers = importlib.import_module(ROOT.name + '.utils.helpers')
        registry = importlib.import_module(ROOT.name + '.core.registry')
        mapping = importlib.import_module(ROOT.name + '.utils.class_mapping')
        processing = importlib.import_module(ROOT.name + '.processing.classify_algorithm')
        from qgis.core import QgsProcessingContext, QgsProcessingFeedback
        for overrides in ({}, {5: 255, 6: 255}):
            with self.subTest(overrides=overrides):
                spec = mapping.output_model_spec(registry.LITEPT_L_MLS, overrides)
                layer = self.rendering_layer()
                post = processing._PointCloud3DPostProcessor(spec.class_mapping, classify_2d=True)
                post.postProcessLayer(layer, QgsProcessingContext(), QgsProcessingFeedback())
                expected = {info.asprs_code for info in spec.class_mapping.values()}
                renderer = layer.renderer()
                self.assertEqual(renderer.type(), 'classified')
                self.assertEqual(renderer.attribute(), 'Classification')
                self.assertEqual({c.value() for c in renderer.categories()}, expected)
                for code in expected:
                    self.assertTrue(renderer.willRenderPoint({'Classification': code}), code)
                renderer3d = layer.renderer3D()
                self.assertIsNotNone(renderer3d)
                self.assertEqual(renderer3d.layer(), layer)
                symbol3d = renderer3d.symbol()
                self.assertEqual(symbol3d.symbolType(), 'classification')
                self.assertEqual(symbol3d.pointSize(), 2.0)
                # QGIS returns the base symbol type; cast to read the categories.
                from qgis._3d import QgsClassificationPointCloud3DSymbol
                symbol3d = sip.cast(symbol3d, QgsClassificationPointCloud3DSymbol)
                self.assertEqual(symbol3d.attribute(), 'Classification')
                categories3d = symbol3d.categoriesList()
                self.assertEqual({c.value() for c in categories3d}, expected)
                self.assertTrue(all(c.renderState() for c in categories3d))
                self.assertEqual(len(categories3d), len(expected))
                if overrides:
                    merged = next(c for c in categories3d if c.value() == 255)
                    self.assertEqual(merged.label(), 'Pole like / Vehicle')
                    self.assertEqual(merged.color().name().lower(), spec.class_mapping[5].color.lower())

    def test_raw_output_styling_and_missing_3d_support_keep_2d_usable(self):
        helpers = importlib.import_module(ROOT.name + '.utils.helpers')
        registry = importlib.import_module(ROOT.name + '.core.registry')
        layer = self.rendering_layer()
        original_type = layer.renderer().type()
        self.assertEqual(original_type, 'rgb')
        self.assertTrue(helpers.enable_point_cloud_3d_rendering(layer))
        self.assertEqual(layer.renderer().type(), original_type)
        self.assertEqual(layer.renderer3D().symbol().symbolType(), 'rgb')
        real_import = builtins.__import__

        def no_3d(name, globals=None, locals=None, fromlist=(), level=0):
            if name == 'qgis' and '_3d' in fromlist:
                raise ImportError('3D module unavailable')
            return real_import(name, globals, locals, fromlist, level)

        fallback = self.rendering_layer()
        with patch.object(builtins, '__import__', no_3d):
            self.assertFalse(helpers.enable_point_cloud_3d_rendering(
                fallback, registry.LITEPT_L_MLS.class_mapping, classify_2d=True))
        self.assertEqual(fallback.renderer().type(), 'classified')
        for code in (64, 65, 66):
            self.assertTrue(fallback.renderer().willRenderPoint({'Classification': code}))


if __name__ == '__main__':
    unittest.main(verbosity=2)
