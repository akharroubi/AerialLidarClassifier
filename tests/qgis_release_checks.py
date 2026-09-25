"""Run with QGIS's Python (python-qgis-ltr.bat on QGIS 3, python-qgis.bat on
QGIS 4); no network, live settings, or GPU required."""
import os
import sys
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
# QGIS first: on QGIS 4 (OSGeo4W) a module that imports ssl before
# qgis._core (unittest.mock does) loads Python's own OpenSSL, and Qt's
# network stack then fails to load. Inside QGIS this cannot happen.
from qgis.core import QgsApplication  # noqa: E402
app = QgsApplication([], False)
app.initQgis()

import importlib  # noqa: E402
import io  # noqa: E402
import subprocess  # noqa: E402,F401
import tarfile  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
from contextlib import ExitStack  # noqa: E402
from unittest.mock import patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.append(str(Path.home() / '.qgis_aerial_lidar_classifier/venv_py3.12/Lib/site-packages'))
import laspy
import numpy as np

def module(name):
    return importlib.import_module(ROOT.name + '.' + name)

v = module('utils.venv_manager')
pm = module('utils.python_manager')
uv = module('utils.uv_manager')
units = module('utils.las_units')
streaming = module('workers.streaming_classifier')
tasks = module('workers.classifier_task')
spec = module('core.registry').LITEPT_L_DALES


class ReleaseChecks(unittest.TestCase):
    def test_batch_preflight_protects_later_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = Path(folder) / 'a.las', Path(folder) / 'a_classified.las'
            a.write_bytes(b'first'); b.write_bytes(b'second')
            task = tasks.ClassificationTask([a, b], Path(folder), '_classified', spec.id, 'cuda', 'classification')
            with patch.object(tasks, '_get_torch'), patch.object(tasks, 'create_backend') as backend:
                self.assertFalse(task.run())
                backend.assert_not_called()
            self.assertEqual(b.read_bytes(), b'second')
            self.assertIn('overwrite an input', task.error_message)

    def test_cohort_link_and_dismissal(self):
        card_module = module('widgets.cohort_card')
        from qgis.PyQt.QtWidgets import QPushButton
        settings = {}
        class Settings:
            def value(self, key, default=None, **kw): return settings.get(key, default)
            def setValue(self, key, value): settings[key] = value
        with patch.object(card_module, 'QgsSettings', Settings), \
             patch.object(card_module.QDesktopServices, 'openUrl', return_value=True) as browser:
            card = card_module.CohortCard()
            browser.assert_not_called()
            card.findChild(QPushButton).click()
            url = browser.call_args.args[0].toString()
            self.assertTrue(url.startswith('https://maven.com/geomatics/qgis3d?'))
            self.assertIn('utm_content=panel', url)
            card._dismiss()
            self.assertTrue(card_module.CohortCard().isHidden())
            self.assertTrue(settings[card_module._HIDE_KEY])

    def test_waveform_payloads_refused_instead_of_lost(self):
        safety = module('utils.output_safety')
        h = laspy.LasHeader(point_format=5, version='1.3')
        h.global_encoding.waveform_data_packets_internal = True
        with self.assertRaisesRegex(ValueError, 'Waveform'):
            safety.validate_waveform_storage(h)

    def test_tls_failure_never_retries_insecurely(self):
        for use_uv in (True, False):
            calls = []
            failure = v._PipResult(1, '', 'certificate verify failed')
            result = v._retry_if_tls_error(failure, ['install'], use_uv, calls.append)
            self.assertIs(result, failure)
            self.assertEqual(calls, [])
            self.assertFalse(v._tls_insecure_fallback_active())

    def test_failed_weight_promotion_retains_good_weights(self):
        import hashlib
        manager_module = module('utils.model_manager')
        with tempfile.TemporaryDirectory() as folder:
            final, temporary = Path(folder) / 'model.pth', Path(folder) / 'new.part'
            final.write_bytes(b'old good model'); temporary.write_bytes(b'new good model')
            manager = object.__new__(manager_module.ModelManager)
            manager.spec = types.SimpleNamespace(weights_sha256=hashlib.sha256(b'new good model').hexdigest(), short_name='test')
            with patch.object(manager, 'get_model_path', return_value=final), patch.object(Path, 'replace', side_effect=PermissionError('locked')):
                ok, reason = manager._verify_and_promote(temporary, 'test')
            self.assertFalse(ok); self.assertIn('retained', reason)
            self.assertEqual(final.read_bytes(), b'old good model')
            self.assertFalse(temporary.exists())

    def test_missing_weights_download_automatically_before_the_run(self):
        seg = module('core.registry').SEGFORMER3D_URBANFILTERING
        manager_module = module('utils.model_manager')
        calls = []

        class Backend:
            def load(self, device): calls.append(('load', device))
            def predict(self, xyz, cb=None, cancel=None):
                return np.ones(len(xyz), dtype=np.int32)
            def unload(self): pass

        def ensure(self, progress=None, cancel=None):
            calls.append('download')
            progress(50, 100)
            return True, ''

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'in.las'
            h = laspy.LasHeader(point_format=6, version='1.4')
            d = laspy.LasData(h); d.points = laspy.ScaleAwarePointRecord.zeros(20, header=h)
            d.X = np.arange(20) * 100; d.write(source)
            task = tasks.ClassificationTask([source], Path(folder) / 'out', '_c', seg.id, 'cpu', 'classification')
            with patch.object(tasks, '_get_torch'), \
                 patch.object(tasks, 'create_backend', return_value=Backend()), \
                 patch.object(manager_module.ModelManager, 'is_model_available', return_value=False), \
                 patch.object(manager_module.ModelManager, 'ensure_available', ensure):
                self.assertTrue(task.run(), task.error_message)
            self.assertEqual(calls[:2], ['download', ('load', 'cpu')])
            self.assertEqual(len(task.output_files), 1)
            self.assertEqual(task.status_text, '')

    def test_weights_download_cancel_and_offline_message(self):
        seg = module('core.registry').SEGFORMER3D_URBANFILTERING
        manager_module = module('utils.model_manager')
        manager = manager_module.ModelManager(seg)
        with patch.object(manager_module.ModelManager, 'is_model_available', return_value=False), \
             patch.object(manager_module.ModelManager, 'download_model', return_value=(False, 'Download cancelled.')):
            with self.assertRaises(InterruptedError):
                manager.ensure_available(None, lambda: True)
        with patch.object(manager_module.ModelManager, 'is_model_available', return_value=False), \
             patch.object(manager_module.ModelManager, 'download_model', return_value=(False, 'Network error: offline')):
            ok, msg = manager.ensure_available(None, lambda: False)
        self.assertFalse(ok)
        self.assertIn('Network error: offline', msg)
        self.assertIn('import the weights file', msg)
        # Downloaded fine but could not be moved into place: folder advice.
        with patch.object(manager_module.ModelManager, 'is_model_available', return_value=False), \
             patch.object(manager_module.ModelManager, 'download_model',
                          return_value=(False, 'Cannot replace the model file; the previous weights were retained: x')):
            ok, msg = manager.ensure_available(None, lambda: False)
        self.assertFalse(ok)
        self.assertIn('writable', msg)
        self.assertNotIn('internet', msg)
        with patch.object(manager_module.ModelManager, 'is_model_available', return_value=True), \
             patch.object(manager_module.ModelManager, 'download_model') as download:
            self.assertEqual(manager.ensure_available(), (True, ''))
            download.assert_not_called()

    def test_installer_subprocesses_do_not_inherit_qgis_python_identity(self):
        # QGIS 4's launcher sets PYTHONEXECUTABLE; leaked into `python -m
        # venv` and uv it sent every package into the portable Python.
        leaked = {'PYTHONEXECUTABLE': r'C:\QGIS\bin\python3.exe', 'PYTHONHOME': r'C:\QGIS\apps\Python312',
                  'PYTHONPATH': r'C:\QGIS\apps\qgis\python', '__PYVENV_LAUNCHER__': 'x', 'PYTHONSTARTUP': 'y'}
        with patch.dict(os.environ, leaked):
            for env in (v._get_clean_env_for_venv(), pm._get_clean_env()):
                for name in leaked:
                    self.assertNotIn(name, env)

    def test_setup_guard_only_blocks_the_plugins_own_torch(self):
        worker = module('workers.deps_install_worker')
        with tempfile.TemporaryDirectory() as cache:
            own = types.SimpleNamespace(__file__=str(Path(cache) / 'venv' / 'torch' / '__init__.py'))
            other = types.SimpleNamespace(__file__=str(Path(cache).parent / 'elsewhere' / 'torch' / '__init__.py'))
            with patch.dict(sys.modules, {'torch': own}):
                self.assertTrue(worker._plugin_torch_loaded(cache))
            with patch.dict(sys.modules, {'torch': other}):
                self.assertFalse(worker._plugin_torch_loaded(cache))

    def test_installer_commands_require_wheels_and_fail_without_fallback(self):
        commands = []
        def run(**kw):
            commands.append(kw['cmd'])
            return v._PipResult(1, '', 'certificate verify failed')
        for gpu in (False, True):
            commands.clear()
            with patch.object(v, 'venv_exists', return_value=True), \
                 patch.object(uv, 'uv_exists', return_value=True), \
                 patch.object(v, '_get_required_packages', return_value=[('torch', '>=2')]), \
                 patch.object(v, '_is_cpu_torch_installed', return_value=False), \
                 patch.object(v, 'detect_nvidia_gpu', return_value=(True, {})), \
                 patch.object(v, '_select_cuda_index', return_value='cu126'), \
                 patch.object(v, '_run_pip_install', side_effect=run):
                self.assertFalse(v.install_dependencies(cuda_enabled=gpu)[0])
            self.assertTrue(commands)
            for cmd in commands:
                self.assertIn('--only-binary=:all:', cmd)
                self.assertIn('https://download.pytorch.org/whl/' + ('cu126' if gpu else 'cpu'), cmd)
                self.assertNotIn('--allow-insecure-host', cmd)
                self.assertNotIn('--trusted-host', cmd)

    def test_cancel_during_verification_never_writes_ready_marker(self):
        cancelled = False
        def verify(**kw):
            nonlocal cancelled
            cancelled = True
            return True, 'ok'
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            for obj, name, value in [(v, 'CACHE_DIR', folder), (v, 'VENV_DIR', folder)]:
                stack.enter_context(patch.object(obj, name, value))
            for obj, name, value in [(pm, 'standalone_python_exists', True), (uv, 'uv_exists', True),
                                     (v, 'venv_exists', True), (v, 'cleanup_old_venv_directories', []),
                                     (v, 'install_dependencies', (True, 'ok'))]:
                stack.enter_context(patch.object(obj, name, return_value=value))
            stack.enter_context(patch.object(v, 'verify_venv', side_effect=verify))
            marker = stack.enter_context(patch.object(v, '_write_install_marker'))
            result = v.create_venv_and_install(cancel_check=lambda: cancelled)
            self.assertFalse(result[0]); marker.assert_not_called()

    def test_spconv_failure_is_required_and_gate_explains_repair(self):
        self.assertFalse(v._is_optional_verify_package('spconv-cu126'))
        readiness = module('utils.backend_readiness')
        readiness.litept_dependency_status.cache_clear()
        with patch.dict(sys.modules, {'spconv': None, 'spconv.pytorch': None}):
            ready, reason = readiness.litept_dependency_status()
        readiness.litept_dependency_status.cache_clear()
        self.assertFalse(ready); self.assertIn('Repair', reason)

    def test_tar_internal_links_and_escape_rejection(self):
        # Capture extraction to test pre-3.12 fallback on Windows without needing
        # OS symlink privileges. Actual symlink extraction is covered on Linux.
        for unsafe in (False, True):
            data = io.BytesIO()
            with tarfile.open(fileobj=data, mode='w') as tar:
                item = tarfile.TarInfo('python/bin/2to3')
                item.type = tarfile.SYMTYPE
                item.linkname = '../../../escape' if unsafe else '2to3-3.10'
                tar.addfile(item)
            data.seek(0)
            with tempfile.TemporaryDirectory() as folder, tarfile.open(fileobj=data) as tar:
                with patch.object(tar, 'extract') as extract:
                    if unsafe:
                        with self.assertRaises(ValueError): pm._safe_extract_tar(tar, folder)
                        extract.assert_not_called()
                    else:
                        pm._safe_extract_tar(tar, folder)
                        self.assertEqual(extract.call_count, 1)

    def test_partial_python_executable_is_not_ready(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'python.exe'; p.write_bytes(b'partial')
            with patch.object(pm, 'get_standalone_python_path', return_value=str(p)):
                self.assertFalse(pm.standalone_python_exists())

    def test_vertical_epsg_and_geographic_override(self):
        result = units.units_from_geokeys({1024: 1, 3076: 9001, 4096: 6360})
        self.assertAlmostEqual(result.z_to_m, 1200 / 3937)
        from laspy.vlrs.known import WktCoordinateSystemVlr
        h = laspy.LasHeader(point_format=6, version='1.4')
        h.vlrs.append(WktCoordinateSystemVlr('GEOGCS["WGS 84",UNIT["degree",0.0174532925199433]]'))
        self.assertTrue(units.resolve_units(h, 'metre').angular)

    def test_provider_headless_registration_is_idempotent(self):
        plugin = module('plugin').AerialLidarClassifierPlugin(None)
        plugin.initProcessing(); first = plugin.provider
        plugin.initProcessing()
        self.assertIs(plugin.provider, first)
        self.assertIsNotNone(QgsApplication.processingRegistry().algorithmById('aeriallidar:classify_lidar'))
        plugin.unload()

    def test_streaming_and_memory_preserve_raw_attributes_and_evlr(self):
        from laspy.vlrs.vlrlist import VLRList
        h = laspy.LasHeader(point_format=8, version='1.4')
        h.add_extra_dim(laspy.ExtraBytesParams(name='scaled', type='uint64', scales=[.001], offsets=[100]))
        h.evlrs = VLRList([laspy.VLR(user_id='test', record_id=42, record_data=b'preserve-evlr')])
        d = laspy.LasData(h); d.points = laspy.ScaleAwarePointRecord.zeros(43, header=h)
        d.X = np.arange(43) * 100; d.Y = np.arange(43) % 7 * 100
        d.red = np.arange(43) + 100
        d.classification = np.full(43, 9, dtype=np.uint8)
        d.points.array['scaled'] = np.uint64(2**63) + np.arange(43, dtype=np.uint64)
        def predict(xyz, cb): return np.full(len(xyz), 3, dtype=np.int32)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'in.las'; d.write(source)
            original = laspy.read(source)
            for mode in ('stream', 'memory'):
                output = Path(folder) / (mode + '.laz')
                if mode == 'stream':
                    streaming.streaming_tiled_classify(source, output, predict, 'cpu', spec.class_mapping,
                        laspy, lambda p: None, lambda: False, field_name='labels', tile_auto=False,
                        tile_size_m=1, buffer_m=0, chunk_size=7, model_spec=spec)
                else:
                    task = tasks.ClassificationTask([source], Path(folder), '_out', spec.id, 'cuda',
                        'labels', tile_enabled=True, tile_auto=False, tile_size_m=1, tile_buffer_m=0)
                    task._process_one(source, output, predict, laspy, 0, 100)
                result = laspy.read(output)
                for name in original.points.array.dtype.names:
                    np.testing.assert_array_equal(result.points.array[name], original.points.array[name], err_msg=name)
                np.testing.assert_array_equal(result.labels, np.full(43, 3))
                self.assertEqual(result.evlrs[0].record_data_bytes(), b'preserve-evlr')
                self.assertTrue(any(v.user_id == 'AerialLiDAR' for v in result.vlrs))

    def test_truncation_and_cancellation_preserve_previous_output(self):
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'in.las', Path(folder) / 'out.las'
            h = laspy.LasHeader(point_format=6, version='1.4')
            d = laspy.LasData(h); d.points = laspy.ScaleAwarePointRecord.zeros(10, header=h); d.write(source)
            original = source.read_bytes()
            output.write_bytes(b'previous output')
            source.write_bytes(original[:-1])
            with self.assertRaises(ValueError):
                streaming.streaming_tiled_classify(source, output, lambda xyz, cb: np.ones(len(xyz), dtype=np.int32),
                    'cpu', spec.class_mapping, laspy, lambda p: None, lambda: False, chunk_size=3)
            self.assertEqual(output.read_bytes(), b'previous output')
            source.write_bytes(original)
            cancelled = False
            def progress(p):
                nonlocal cancelled
                if p > 90: cancelled = True
            with self.assertRaises(InterruptedError):
                streaming._pass4_write(input_path=source, output_path=output,
                    predictions=np.ones(10, dtype=np.uint8), field_name='classification',
                    is_asprs_field=True, laspy_module=laspy, progress_callback=progress,
                    cancel_callback=lambda: cancelled, chunk_size=3, input_pf_id=6)
            self.assertEqual(output.read_bytes(), b'previous output')


if __name__ == '__main__':
    unittest.main(verbosity=2)
