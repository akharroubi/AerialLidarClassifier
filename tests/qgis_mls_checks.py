"""v1.2 integration checks with real QGIS/Qt and LAS I/O, stubbed inference.

Run with QGIS's python-qgis-ltr.bat (QGIS 3) or python-qgis.bat (QGIS 4).
No network, GPU, installed plugin, or changes to the user's settings required.
Actual MLS model inference is covered separately by test_litept_mls_backend.py.
"""
import os
import sys
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# QGIS must load its SSL/Qt DLLs before unittest.mock imports Python's SSL.
from qgis.core import QgsApplication, QgsProcessingContext, QgsProcessingFeedback, QgsProcessingException, QgsPointCloudLayer
app = QgsApplication([], False)
app.initQgis()

import hashlib
import importlib
import json
import tempfile
import types
import unittest
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

from qgis.PyQt.QtCore import QTimer

# Windows Qt's offscreen platform has an empty font database. Give optional
# visual evidence the same readable font the normal Windows application uses.
if os.environ.get('ALC_UI_CAPTURE_DIR') and Path('C:/Windows/Fonts/segoeui.ttf').is_file():
    from qgis.PyQt.QtGui import QFont, QFontDatabase
    QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
    app.setFont(QFont('Segoe UI', 9))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.append(str(Path.home() / '.qgis_aerial_lidar_classifier/venv_py3.12/Lib/site-packages'))
import laspy
import numpy as np


def module(name):
    return importlib.import_module(ROOT.name + '.' + name)


registry = module('core.registry')
MLS = registry.get_model('litept_l_mls_5cm')
tasks = module('workers.classifier_task')
manager_module = module('utils.model_manager')
mapping = module('utils.class_mapping')
processing = module('processing.classify_algorithm')
venv = module('utils.venv_manager')
backends = module('core.backends')
config = module('config')


def make_source(path, extras=True):
    """Legacy RGB LAS with distinct attributes and scaled 64-bit extra data."""
    header = laspy.LasHeader(point_format=3, version='1.2')
    header.scales = [.01, .01, .01]
    header.offsets = [500000., 5600000., 0.]
    if extras:
        header.add_extra_dim(laspy.ExtraBytesParams(
            name='precise', type='uint64', scales=[.001], offsets=[100.]))
    header.vlrs.append(laspy.VLR(user_id='MLS-test', record_id=42, record_data=b'keep-metadata'))
    cloud = laspy.LasData(header)
    cloud.points = laspy.ScaleAwarePointRecord.zeros(27, header=header)
    index = np.arange(27)
    cloud.X = index * 100
    cloud.Y = index % 3 * 100
    cloud.Z = index % 7 * 25
    cloud.intensity = index + 1000
    cloud.classification = np.full(27, 9, dtype=np.uint8)
    cloud.return_number = index % 3 + 1
    cloud.number_of_returns = np.full(27, 3, dtype=np.uint8)
    cloud.synthetic = index % 2
    cloud.key_point = index % 3 == 0
    cloud.withheld = index % 5 == 0
    cloud.scan_direction_flag = index % 2
    cloud.edge_of_flight_line = index % 7 == 0
    cloud.scan_angle_rank = index - 13
    cloud.user_data = index + 20
    cloud.point_source_id = index + 300
    cloud.gps_time = index + 100000.25
    cloud.red = index + 100
    cloud.green = index + 200
    cloud.blue = index + 300
    if extras:
        cloud.points.array['precise'] = np.uint64(2**63) + index.astype(np.uint64)
    cloud.write(path)
    return laspy.read(path)


def predictions(xyz, progress=None, cancel=None):
    if progress is not None:
        progress(80.)
    return ((np.rint(xyz[:, 0]).astype(np.int64) - 500000) % 9 + 1).astype(np.int32)


class StubBackend:
    def __init__(self):
        self.loaded = []
        self.unloaded = 0

    def load(self, device):
        self.loaded.append(device)

    def predict(self, xyz, progress=None, cancel=None):
        return predictions(xyz, progress, cancel)

    def unload(self):
        self.unloaded += 1


class MlsChecks(unittest.TestCase):
    def assert_preserved(self, before, after, classification=False):
        for name in ('X', 'Y', 'Z', 'intensity', 'return_number', 'number_of_returns',
                     'synthetic', 'key_point', 'withheld', 'scan_direction_flag',
                     'edge_of_flight_line', 'scan_angle_rank', 'user_data',
                     'point_source_id', 'gps_time', 'red', 'green', 'blue'):
            if name == 'scan_angle_rank' and name not in after.point_format.dimension_names:
                np.testing.assert_allclose(before[name], np.asarray(after.scan_angle) * .006, atol=.003, rtol=0)
            else:
                np.testing.assert_array_equal(before[name], after[name], err_msg=name)
        if 'precise' in before.point_format.dimension_names:
            np.testing.assert_array_equal(before.points.array['precise'], after.points.array['precise'])
        np.testing.assert_array_equal(before.header.scales, after.header.scales)
        np.testing.assert_array_equal(before.header.offsets, after.header.offsets)
        if classification:
            np.testing.assert_array_equal(before.classification, after.classification)
        self.assertTrue(any(v.user_id == 'MLS-test' and v.record_data_bytes() == b'keep-metadata'
                            for v in after.vlrs))

    def assert_metadata(self, cloud, field, codes):
        records = [v for v in cloud.vlrs if v.user_id == 'AerialLiDAR' and v.record_id == 1]
        self.assertEqual(len(records), 1)
        data = json.loads(records[0].record_data_bytes().decode('utf8'))
        self.assertEqual(data['model'], MLS.id)
        self.assertEqual(data['field'], field)
        self.assertEqual(data['encoding'], 'ASPRS' if field == 'classification' else 'raw_model_ids')
        self.assertEqual(data['classes']['5']['asprs'], codes[5])
        self.assertEqual(data['classes']['6']['asprs'], codes[6])

    def test_model_order_labels_and_processing_parameter(self):
        self.assertEqual([s.id for s in registry.MODELS[:3]],
                         ['litept_l_dales_10cm', 'segformer3d_urbanfiltering', MLS.id])
        self.assertIn('Airborne', registry.MODELS[0].display_name)
        self.assertIn('Mobile Mapping', MLS.display_name)
        self.assertEqual(list(MLS.class_mapping), list(range(1, 10)))
        algorithm = processing.ClassifyLidarAlgorithm()
        algorithm.initAlgorithm()
        self.assertEqual(algorithm.parameterDefinition('MODEL').options()[2], MLS.display_name)
        self.assertIsNotNone(algorithm.parameterDefinition('OUTPUT_CODES_JSON'))

    def test_extra_dimension_descriptions_fit_utf8_byte_limit(self):
        safety = module('utils.output_safety')
        for text in ('AI classification (' + MLS.display_name + ')',
                     'a' * 31 + '\u2014 tail', '\u00e9' * 20, '\U0001f30d' * 9):
            description = safety.extra_dim_description(text)
            self.assertLessEqual(len(description.encode('utf8')), 32)
            self.assertTrue(text.startswith(description))
            header = laspy.LasHeader(point_format=6, version='1.4')
            header.add_extra_dim(laspy.ExtraBytesParams(name='mms_ids', type='int32', description=description))

    def test_task_modes_map_custom_codes_upgrade_and_preserve_attributes(self):
        codes = {5: 255, 6: 17}
        expected = np.array([2, 3, 5, 6, 255, 17, 66, 14, 1] * 3, dtype=np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            original = make_source(source)
            source_bytes = source.read_bytes()
            for mode in ('memory', 'tiled', 'streaming'):
                with self.subTest(mode=mode):
                    output = folder / (mode + '.laz')
                    task = tasks.ClassificationTask([source], folder, '_out', MLS.id, 'cuda',
                        'classification', tile_enabled=mode != 'memory', tile_auto=False,
                        tile_size_m=10, tile_buffer_m=1, tile_streaming=mode == 'streaming',
                        units_override='metre', output_codes=codes)
                    task._process_one(source, output, predictions, laspy, 0, 100)
                    result = laspy.read(output)
                    self.assertEqual(result.point_format.id, 7)
                    self.assertEqual(str(result.header.version), '1.4')
                    np.testing.assert_array_equal(result.classification, expected)
                    self.assert_preserved(original, result)
                    self.assert_metadata(result, 'classification', codes)
                    self.assertEqual(source.read_bytes(), source_bytes)
        self.assertEqual(MLS.class_mapping[5].asprs_code, 64)
        self.assertEqual(MLS.class_mapping[6].asprs_code, 65)

    def test_mms_refuses_extra_class_field_without_writing_output(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            make_source(source)
            original_bytes = source.read_bytes()
            for stream in (False, True):
                with self.assertRaisesRegex(ValueError, 'classification'):
                    tasks.ClassificationTask([source], folder, '_raw', MLS.id, 'cuda', 'mms_ids',
                        tile_enabled=stream, tile_auto=False, tile_size_m=10, tile_buffer_m=0,
                        tile_streaming=stream, units_override='metre')
            self.assertEqual(source.read_bytes(), original_bytes)
            self.assertEqual(list(folder.iterdir()), [source])

    def test_mms_standard_only_outputs_read_in_qgis_without_new_dimensions(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / 'input.las'
            original = make_source(source, extras=False)
            for stream in (False, True):
                for extension in ('.las', '.laz'):
                    with self.subTest(stream=stream, extension=extension):
                        output = folder / (('stream' if stream else 'memory') + extension)
                        task = tasks.ClassificationTask([source], folder, '_out', MLS.id, 'cuda',
                            'classification', tile_enabled=stream, tile_auto=False, tile_size_m=10,
                            tile_buffer_m=0, tile_streaming=stream, units_override='metre')
                        task._process_one(source, output, predictions, laspy, 0, 100)
                        written = laspy.read(output)
                        self.assertEqual(list(written.point_format.extra_dimension_names), [])
                        self.assert_preserved(original, written)
                        options = QgsPointCloudLayer.LayerOptions()
                        options.skipIndexGeneration = True
                        options.skipStatisticsCalculation = True
                        options.skipCrsValidation = True
                        layer = QgsPointCloudLayer(str(output), output.stem, 'pdal', options)
                        self.assertTrue(layer.isValid(), layer.error().message())
                        self.assertEqual(layer.dataProvider().pointCount(), 27)

    def processing_params(self, algorithm, source, folder, stream=False):
        return {
            algorithm.INPUT: str(source), algorithm.MODEL: 2,
            algorithm.OUTPUT_FOLDER: str(folder), algorithm.SUFFIX: '_processed',
            algorithm.DEVICE: 1, algorithm.FIELD_NAME: 'classification',
            algorithm.LOAD_AS_LAYER: False, algorithm.TILE_ENABLED: stream,
            algorithm.TILE_STREAMING: stream, algorithm.TILE_SIZE_M: 10.,
            algorithm.TILE_BUFFER_M: 0., algorithm.UNITS: 1,
            algorithm.OUTPUT_CODES_JSON: '{"5":255,"6":17}',
        }

    def processing_patches(self, stack, backend, cuda=True):
        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: cuda),
            backends=types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False)))
        stack.enter_context(patch.dict(sys.modules, {'torch': fake_torch}))
        stack.enter_context(patch.object(venv, 'get_venv_status', return_value=(True, 'ready')))
        stack.enter_context(patch.object(venv, 'ensure_venv_packages_available'))
        stack.enter_context(patch.object(manager_module.ModelManager, 'is_model_available', return_value=True))
        stack.enter_context(patch.object(manager_module.ModelManager, 'get_model_path', return_value=Path('unused.pt')))
        return stack.enter_context(patch.object(backends, 'create_backend', return_value=backend))

    def test_processing_real_parameter_parsing_and_mms_output(self):
        algorithm = processing.ClassifyLidarAlgorithm()
        algorithm.initAlgorithm()
        for stream, field in ((False, 'classification'), (True, 'classification')):
            with self.subTest(stream=stream, field=field), tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
                folder = Path(folder)
                source = folder / 'input.las'
                original = make_source(source)
                backend = StubBackend()
                factory = self.processing_patches(stack, backend)
                parameters = self.processing_params(algorithm, source, folder, stream)
                parameters[algorithm.FIELD_NAME] = field
                result = algorithm.processAlgorithm(parameters,
                    QgsProcessingContext(), QgsProcessingFeedback())
                written = laspy.read(result[algorithm.OUTPUT_FILE])
                self.assertEqual(factory.call_args.args[0].id, MLS.id)
                self.assertEqual(backend.loaded, ['cuda'])
                self.assertEqual(backend.unloaded, 1)
                expected = [2,3,5,6,255,17,66,14,1] * 3 if field == 'classification' else list(range(1, 10)) * 3
                np.testing.assert_array_equal(written[field], expected)
                self.assert_preserved(original, written, classification=field != 'classification')
                self.assert_metadata(written, field, {5: 255, 6: 17})

    def test_processing_and_task_refuse_cpu_without_switching_models(self):
        algorithm = processing.ClassifyLidarAlgorithm()
        algorithm.initAlgorithm()
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            source = Path(folder) / 'input.las'
            make_source(source)
            factory = self.processing_patches(stack, StubBackend(), cuda=False)
            parameters = self.processing_params(algorithm, source, folder)
            parameters[algorithm.DEVICE] = 0
            with self.assertRaisesRegex(QgsProcessingException, 'Mobile Mapping'):
                algorithm.processAlgorithm(parameters, QgsProcessingContext(), QgsProcessingFeedback())
            factory.assert_not_called()
            task = tasks.ClassificationTask([source], folder, '_cpu', MLS.id, 'cpu', 'classification')
            with patch.object(tasks, '_get_torch'), patch.object(tasks, 'create_backend') as task_factory:
                self.assertFalse(task.run())
                task_factory.assert_not_called()
                self.assertIn('Mobile Mapping', task.error_message)
            self.assertFalse((Path(folder) / 'input_cpu.las').exists())

    def test_processing_rejects_bad_code_before_inference(self):
        algorithm = processing.ClassifyLidarAlgorithm()
        algorithm.initAlgorithm()
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            source = Path(folder) / 'input.las'
            make_source(source)
            factory = self.processing_patches(stack, StubBackend())
            parameters = self.processing_params(algorithm, source, folder)
            for invalid in ('{"5":256}', '{"0":3}', '{"5":true}', '[2,3]', '{broken}'):
                with self.subTest(value=invalid), self.assertRaises(QgsProcessingException):
                    parameters[algorithm.OUTPUT_CODES_JSON] = invalid
                    algorithm.processAlgorithm(parameters, QgsProcessingContext(), QgsProcessingFeedback())
            factory.assert_not_called()
            self.assertFalse((Path(folder) / 'input_processed.las').exists())
            parameters[algorithm.OUTPUT_CODES_JSON] = ''
            parameters[algorithm.FIELD_NAME] = 'mms_extra'
            with self.assertRaisesRegex(QgsProcessingException, 'classification'):
                algorithm.processAlgorithm(parameters, QgsProcessingContext(), QgsProcessingFeedback())

    def test_download_ignores_queued_progress_after_feedback_destruction(self):
        from qgis.PyQt import sip
        from qgis.PyQt.QtCore import QByteArray

        class Signal:
            def connect(self, callback):
                self.queued_callback = callback
                self.disconnected = False

            def disconnect(self, callback):
                self.disconnected = callback is self.queued_callback

        class Request:
            ErrorCode = types.SimpleNamespace(NoError=0)

            def __init__(self, error):
                self.downloadProgress = Signal()
                self.error = error

            def get(self, *args):
                self.downloadProgress.queued_callback(1, 2)
                return self.error

            def errorMessage(self):
                return 'test network failure'

            def reply(self):
                return types.SimpleNamespace(content=lambda: QByteArray(b'weights'))

        for outcome in ('success', 'network_error', 'cancelled'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as folder:
                caller_feedback = QgsProcessingFeedback()
                if outcome == 'cancelled':
                    caller_feedback.cancel()
                request = Request(1 if outcome == 'network_error' else 0)
                factory = types.SimpleNamespace(
                    ErrorCode=Request.ErrorCode,
                )
                # Keep a queued callback even after disconnect, as Qt can do.
                class RequestFactory:
                    ErrorCode = factory.ErrorCode

                    def __new__(cls):
                        return request

                with patch.object(manager_module, 'QgsBlockingNetworkRequest', RequestFactory):
                    ok, message = manager_module.ModelManager._download_one(
                        'http://localhost/weights', Path(folder) / 'weights',
                        lambda r, t: caller_feedback.setProgress(100 * r / t),
                        caller_feedback.isCanceled,
                    )
                self.assertEqual(ok, outcome == 'success', message)
                self.assertTrue(request.downloadProgress.disconnected)
                sip.delete(caller_feedback)
                # A stale slot must never call the deleted Processing object.
                request.downloadProgress.queued_callback(2, 2)

    def test_mls_weights_download_verify_import_and_refuse_bad_files(self):
        """Real QGIS network download from a local server, as on first use."""
        import http.server
        import threading
        payload = b'released mobile mapping weights'
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            served = folder / 'served'
            served.mkdir()
            (served / MLS.weights_filename).write_bytes(payload)
            (served / 'tampered.pth').write_bytes(b'tampered weights')

            class Quiet(http.server.SimpleHTTPRequestHandler):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, directory=str(served), **kwargs)

                def log_message(self, *args):
                    pass

            server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Quiet)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f'http://127.0.0.1:{server.server_address[1]}'
            try:
                spec = replace(MLS, weights_sha256=hashlib.sha256(payload).hexdigest(),
                               weights_urls=(f'{base}/missing.pth', f'{base}/tampered.pth',
                                             f'{base}/{MLS.weights_filename}'))
                manager = manager_module.ModelManager(spec)
                cache = folder / 'cache' / spec.weights_filename
                cache.parent.mkdir()
                progress = []
                with patch.object(manager, '_cache_path', return_value=cache), \
                     patch.object(manager, 'models_root', return_value=folder / 'cache'), \
                     patch.object(manager, 'get_model_url', return_value=''):
                    self.assertFalse(manager.is_model_available())
                    # A 404 and a wrong-SHA file are skipped; the third URL wins.
                    ok, message = manager.ensure_available(lambda r, t: progress.append((r, t)))
                    self.assertTrue(ok, message)
                    self.assertEqual(cache.read_bytes(), payload)
                    self.assertTrue(manager.is_model_available())
                    self.assertTrue(progress and progress[-1][0] == len(payload))
                    self.assertEqual(sorted(p.name for p in cache.parent.iterdir()), [cache.name])
                    # Already present: no second download.
                    with patch.object(manager, 'download_model') as download:
                        self.assertEqual(manager.ensure_available(), (True, ''))
                        download.assert_not_called()
                    # Damaged cache is not trusted, then repaired by a download.
                    cache.write_bytes(b'damaged cached weights')
                    self.assertFalse(manager.is_model_available())
                    self.assertTrue(manager.ensure_available()[0])
                    self.assertEqual(cache.read_bytes(), payload)
                    # Only tampered or missing files: refused, nothing promoted.
                    manager.delete_model()
                    bad = replace(spec, weights_urls=(f'{base}/missing.pth', f'{base}/tampered.pth'))
                    manager.spec = bad
                    ok, message = manager.ensure_available()
                    self.assertFalse(ok)
                    self.assertIn('SHA-256 mismatch', message)
                    self.assertFalse(cache.exists())
                    self.assertEqual(list(cache.parent.iterdir()), [])
                    # Offline import: verified copy, wrong file refused.
                    ok, message = manager.import_file(str(served / 'tampered.pth'))
                    self.assertFalse(ok)
                    self.assertFalse(cache.exists())
                    ok, message = manager.import_file(str(served / MLS.weights_filename))
                    self.assertTrue(ok, message)
                    self.assertEqual(cache.read_bytes(), payload)
                    # No URL at all: an import instruction, never a silent failure.
                    manager.delete_model()
                    manager.spec = replace(spec, weights_urls=())
                    with patch.object(manager, 'download_model') as download:
                        ok, message = manager.ensure_available()
                        download.assert_not_called()
                    self.assertFalse(ok)
                    self.assertIn('Import', message)
            finally:
                server.shutdown()
                server.server_close()

    def test_mls_weights_are_published_not_bundled(self):
        self.assertEqual(len(MLS.weights_urls), 1)
        self.assertTrue(MLS.weights_urls[0].startswith(
            'https://github.com/akharroubi/AerialLidarClassifier/releases/download/'))
        self.assertTrue(MLS.weights_urls[0].endswith('/' + MLS.weights_filename))
        card = json.loads(MLS.config_path.read_text(encoding='utf-8'))
        self.assertEqual(card['weights_file'], MLS.weights_filename)
        self.assertAlmostEqual(card['weights_bytes'] / 1e6, MLS.weights_size_mb, delta=0.1)
        self.assertFalse((ROOT / 'models').exists() and any((ROOT / 'models').iterdir()),
                         'weights must not be shipped inside the plugin folder')

    def test_real_qt_editor_accept_reset_persistence_and_device_gate(self):
        panel_module = module('gui.main_panel')
        dialog_module = module('dialogs.class_mapping_dialog')
        settings = {f'{config.SETTINGS_PREFIX}/settings_version': 2,
                    f'{config.SETTINGS_PREFIX}/model_id': MLS.id}

        class Settings:
            def value(self, key, default=None, **kwargs):
                return settings.get(key, default)

            def setValue(self, key, value):
                settings[key] = value

        with patch.object(panel_module, 'QgsSettings', Settings), \
             patch.object(panel_module, 'get_gpu_info', return_value={'available': False}), \
             patch.object(manager_module.ModelManager, 'is_model_available', return_value=True):
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(panel.model_combo.currentData(), MLS.id)
                self.assertFalse(panel.model_gate_ok)
                self.assertIsNone(panel._fix_action)
                self.assertFalse(panel.mms_codes_row.isHidden())
                self.assertEqual(panel.field_edit.text(), 'classification')
                self.assertFalse(panel.field_edit.isEnabled())
                editor_error = []

                def edit_and_accept():
                    dialog = app.activeModalWidget()
                    try:
                        self.assertIsInstance(dialog, dialog_module.ClassMappingDialog)
                        self.assertEqual(len(dialog._spins), 9)
                        self.assertEqual(dialog._spins[5].maximum(), 255)
                        dialog._spins[5].setValue(255)
                        dialog._spins[6].setValue(17)
                        dialog.accept()
                    except BaseException as error:
                        editor_error.append(error)
                        if dialog is not None:
                            dialog.reject()

                QTimer.singleShot(0, edit_and_accept)
                panel._edit_output_codes()
                if editor_error:
                    raise editor_error[0]
                self.assertEqual(panel._output_codes_by_model[MLS.id][5], 255)
                panel._save_settings()
                saved = json.loads(settings[f'{config.SETTINGS_PREFIX}/output_codes/{MLS.id}'])
                self.assertEqual(saved['5'], 255)
                self.assertEqual(saved['6'], 17)
                panel.model_combo.setCurrentIndex(0)
                self.assertTrue(panel.mms_codes_row.isHidden())
                panel.model_combo.setCurrentIndex(2)
                self.assertEqual(panel._output_codes_by_model[MLS.id][5], 255)
            finally:
                panel.close()
            restored = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(restored.model_combo.currentData(), MLS.id)
                self.assertEqual(restored._output_codes_by_model[MLS.id][5], 255)
                dialog = dialog_module.ClassMappingDialog(MLS, restored._output_codes_by_model[MLS.id])
                capture_dir = os.environ.get('ALC_UI_CAPTURE_DIR')
                if capture_dir:
                    capture_dir = Path(capture_dir)
                    capture_dir.mkdir(parents=True, exist_ok=True)
                    dialog.show()
                    app.processEvents()
                    self.assertTrue(dialog.grab().save(str(capture_dir / 'mms_output_codes.png')))
                dialog._reset_defaults()
                self.assertEqual(dialog.output_codes()[5], 64)
                self.assertEqual(dialog.output_codes()[6], 65)
                dialog.close()
            finally:
                restored.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
