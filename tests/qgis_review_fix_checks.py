"""Checks for the fixes made after the independent review of 1.3.3.

Run with QGIS's Python (python-qgis-ltr.bat on QGIS 3, python-qgis.bat on
QGIS 4); no network, live settings, or GPU required.
"""
import os
import sys
from pathlib import Path

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
from qgis.core import QgsApplication  # noqa: E402
app = QgsApplication([], False)
app.initQgis()

import gc  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402
import tempfile  # noqa: E402
import types  # noqa: E402
import unittest  # noqa: E402
import weakref  # noqa: E402
from unittest.mock import patch  # noqa: E402

from qgis.PyQt.QtCore import QByteArray, QObject, pyqtSignal  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.append(str(Path.home() / '.qgis_aerial_lidar_classifier/venv_py3.12/Lib/site-packages'))
import laspy  # noqa: E402
import numpy as np  # noqa: E402


def module(name):
    return importlib.import_module(ROOT.name + '.' + name)


v = module('utils.venv_manager')
pm = module('utils.python_manager')
uv = module('utils.uv_manager')
memory = module('core.memory')
streaming = module('workers.streaming_classifier')
tasks = module('workers.classifier_task')
compat = module('utils.compat')
spec = module('core.registry').LITEPT_L_DALES


def _labels(xyz):
    """A label that depends only on the point, not on the tile it came in."""
    return (np.floor(xyz[:, 0] / 37.0).astype(np.int64) % 7 + 1).astype(np.int32)


def _write(path, xyz, point_format=6, scale=0.001):
    h = laspy.LasHeader(point_format=point_format, version='1.4' if point_format >= 6 else '1.2')
    h.scales = [scale, scale, scale]
    h.offsets = [0, 0, 0]
    d = laspy.LasData(h)
    d.x, d.y, d.z = xyz.T
    d.write(path)
    return laspy.read(path)


def _expected(las):
    ids = _labels(np.column_stack([las.x, las.y, las.z]))
    return np.array([spec.class_mapping[int(i)].asprs_code for i in ids])


class _FakeRequest:
    """QgsBlockingNetworkRequest that returns bytes which are not the archive."""
    NoError = 0
    ErrorCode = types.SimpleNamespace(NoError=0)

    def get(self, request):
        return 0

    def reply(self):
        return types.SimpleNamespace(content=lambda: QByteArray(b'not the published archive'))

    def errorMessage(self):
        return ''


class ReviewFixChecks(unittest.TestCase):
    # ---- streaming -------------------------------------------------------
    def test_header_bounds_without_all_points_are_recovered(self):
        rng = np.random.default_rng(1)
        xyz = np.column_stack([rng.uniform(0, 200, 5000), rng.uniform(0, 50, 5000), rng.uniform(0, 5, 5000)])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'bounds.las'
            original = _write(source, xyz)
            raw = bytearray(source.read_bytes())
            struct.pack_into('<d', raw, 179, 100.0)  # Max X: half of the true extent
            source.write_bytes(bytes(raw))
            warnings = []
            output = Path(folder) / 'out' / 'bounds.las'
            streaming.streaming_tiled_classify(
                source, output, lambda p, cb: _labels(p), 'cpu', spec.class_mapping, laspy,
                lambda p: None, lambda: False, tile_auto=False, tile_size_m=40, buffer_m=5,
                chunk_size=1000, warning_callback=warnings.append, model_spec=spec)
            self.assertTrue(any('do not contain all its points' in w for w in warnings), warnings)
            result = laspy.read(output)
            np.testing.assert_array_equal(np.asarray(result.classification), _expected(original))

    def test_input_changed_during_the_run_is_refused(self):
        rng = np.random.default_rng(2)
        xyz = rng.uniform(0, 100, (3000, 3))
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'in.las'
            _write(source, xyz)
            output = Path(folder) / 'out.las'
            output.write_bytes(b'previous output')

            def predict(p, cb):
                stat = source.stat()
                os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5 * 10 ** 9))
                return _labels(p)
            with self.assertRaisesRegex(RuntimeError, 'changed during the run'):
                streaming.streaming_tiled_classify(
                    source, output, predict, 'cpu', spec.class_mapping, laspy, lambda p: None,
                    lambda: False, buffer_m=5, chunk_size=1000, model_spec=spec)
            self.assertEqual(output.read_bytes(), b'previous output')
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()), ['in.las', 'out.las'])

    def test_unsplittable_tile_is_reported_with_the_memory_it_needs(self):
        rng = np.random.default_rng(3)
        xyz = rng.uniform(0, 1, (100, 3))
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(os.environ, {memory.ENV_TILE_POINTS: '10'}):
            source = Path(folder) / 'tiny.las'
            original = _write(source, xyz)
            warnings = []
            output = Path(folder) / 'out.las'
            streaming.streaming_tiled_classify(
                source, output, lambda p, cb: _labels(p), 'cpu', spec.class_mapping, laspy,
                lambda p: None, lambda: False, buffer_m=35, warning_callback=warnings.append,
                model_spec=spec)
            self.assertTrue(any('still hold more than the memory budget' in w and 'GB' in w
                                for w in warnings), warnings)
            np.testing.assert_array_equal(np.asarray(laspy.read(output).classification),
                                          _expected(original))
            logged = []
            pcd = np.column_stack([original.x, original.y, original.z])
            with patch.object(tasks, 'log_warning', logged.append):
                tasks._classify_tiled(pcd, lambda p, cb: _labels(p), lambda p: None, lambda: False,
                                      auto=True, buffer_m=35)
            self.assertTrue(any('more than the memory budget' in m for m in logged), logged)

    def test_stray_point_far_away_does_not_defeat_the_split(self):
        """A dense street and one point 3,000 km away: the file-wide count grid
        has 1.5 km cells, the street is split on a grid of its own."""
        rng = np.random.default_rng(4)
        street = np.column_stack([rng.uniform(0, 2000, 60_000), rng.uniform(995, 1005, 60_000),
                                  rng.uniform(0, 10, 60_000)])
        xyz = np.vstack([street, [[3_000_000.0, 1000.0, 5.0]]])
        budget = 20_000
        for mode in ('stream', 'memory'):
            with tempfile.TemporaryDirectory() as folder, \
                    patch.dict(os.environ, {memory.ENV_TILE_POINTS: str(budget)}):
                source = Path(folder) / 'stray.las'
                original = _write(source, xyz, scale=0.01)
                calls = []

                def predict(p, cb):
                    calls.append(len(p))
                    return _labels(p)
                output = Path(folder) / 'out.las'
                if mode == 'stream':
                    streaming.streaming_tiled_classify(
                        source, output, predict, 'cpu', spec.class_mapping, laspy, lambda p: None,
                        lambda: False, buffer_m=10, chunk_size=20_000, model_spec=spec)
                    labels = np.asarray(laspy.read(output).classification)
                else:
                    pcd = np.column_stack([original.x, original.y, original.z])
                    ids = tasks._classify_tiled(pcd, predict, lambda p: None, lambda: False,
                                                auto=True, buffer_m=10)
                    labels = np.array([spec.class_mapping[int(i)].asprs_code for i in ids])
                self.assertLessEqual(max(calls), budget, mode)
                np.testing.assert_array_equal(labels, _expected(original), err_msg=mode)

    def test_previous_tile_is_released_before_the_next_one(self):
        rng = np.random.default_rng(5)
        xyz = np.column_stack([rng.uniform(0, 400, 40_000), rng.uniform(0, 50, 40_000),
                               rng.uniform(0, 5, 40_000)])
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'two.las'
            original = _write(source, xyz)
            seen = []
            alive_at_next = []

            def predict(p, cb):
                gc.collect()
                alive_at_next.append([ref() is not None for ref in seen])
                seen.append(weakref.ref(p))
                return _labels(p)
            output = Path(folder) / 'out.las'
            streaming.streaming_tiled_classify(
                source, output, predict, 'cpu', spec.class_mapping, laspy, lambda p: None,
                lambda: False, tile_auto=False, tile_size_m=100, buffer_m=5, model_spec=spec)
            self.assertGreaterEqual(len(seen), 3)
            for alive in alive_at_next:
                self.assertFalse(any(alive), alive_at_next)
            np.testing.assert_array_equal(np.asarray(laspy.read(output).classification),
                                          _expected(original))

    # ---- in-memory run, planning, Processing ---------------------------
    def test_cancel_while_writing_keeps_the_previous_output(self):
        rng = np.random.default_rng(6)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'in.las'
            _write(source, rng.uniform(0, 50, (2000, 3)))
            output = Path(folder) / 'in_classified.las'
            output.write_bytes(b'previous output')
            task = tasks.ClassificationTask([source], Path(folder), '_classified', spec.id, 'cpu',
                                            'classification')

            def predict(p, cb):
                task.cancel()  # pressed while the model finishes
                return _labels(p)
            with self.assertRaises(InterruptedError):
                task._process_one(source, output, predict, laspy, 0, 100)
            self.assertEqual(output.read_bytes(), b'previous output')
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()),
                             ['in.las', 'in_classified.las'])

    def test_tile_streaming_streams_without_tiling(self):
        rng = np.random.default_rng(7)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'in.las'
            _write(source, rng.uniform(0, 50, (500, 3)))
            task = tasks.ClassificationTask([source], Path(folder), '_c', spec.id, 'cpu',
                                            'classification', tile_enabled=False, tile_streaming=True)
            self.assertEqual(task._plan(source, laspy), 'streaming')
            task = tasks.ClassificationTask([source], Path(folder), '_c', spec.id, 'cpu',
                                            'classification', tile_enabled=False)
            self.assertEqual(task._plan(source, laspy), 'untiled')

    def test_unticked_tiling_is_never_switched_on_by_the_plugin(self):
        """Tiling unticked and a file too large for the free memory: refused
        before the file is read, with the advice to tick Tile large files.
        Ticked, the same file is streamed."""
        rng = np.random.default_rng(8)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'big.las'
            original = _write(source, rng.uniform(0, 50, (5000, 3)))
            output = Path(folder) / 'big_classified.las'
            calls = []

            def predict(p, cb):
                calls.append(len(p))
                return _labels(p)
            with patch.object(memory, 'available_memory_bytes', return_value=10 ** 5):
                task = tasks.ClassificationTask([source], Path(folder), '_classified', spec.id,
                                                'cpu', 'classification', tile_enabled=False)
                with self.assertRaisesRegex(RuntimeError, "Tick 'Tile large files'"):
                    task._process_one(source, output, predict, laspy, 0, 100)
                self.assertEqual(calls, [])
                self.assertFalse(output.exists())
                algorithm = module('processing.classify_algorithm')
                self.assertEqual(algorithm.plan_processing(5000, 30, False)[0], 'too_large')
                task = tasks.ClassificationTask([source], Path(folder), '_classified', spec.id,
                                                'cpu', 'classification', tile_enabled=True)
                self.assertEqual(task._plan(source, laspy), 'streaming')
                task._process_one(source, output, predict, laspy, 0, 100)
            np.testing.assert_array_equal(np.asarray(laspy.read(output).classification),
                                          _expected(original))

    def test_plan_counts_the_upgraded_records(self):
        # Railway codes above 31 on a point format 3 file: the output needs
        # LAS 1.4 records next to the old ones.
        rail = module('core.registry').LITEPT_L_RAIL
        safety = module('utils.output_safety')
        pf3 = laspy.PointFormat(3)
        self.assertGreater(safety.output_record_length(pf3, 'classification', rail.class_mapping), 0)
        self.assertEqual(safety.output_record_length(laspy.PointFormat(7), 'classification',
                                                     rail.class_mapping), 0)

    def test_dock_close_disconnects_the_task_log(self):
        panel_module = module('gui.main_panel')

        class FakeTask(QObject):
            progressChanged = pyqtSignal(float)
            taskCompleted = pyqtSignal()
            taskTerminated = pyqtSignal()
            log_message = pyqtSignal(str)

            def isCanceled(self):
                return True

            def cancel(self):
                pass
        settings = {}

        class Settings:  # never the user's real QGIS settings
            def value(self, key, default=None, **kwargs):
                return settings.get(key, default)

            def setValue(self, key, value):
                settings[key] = value
        with patch.object(panel_module, 'QgsSettings', Settings), \
                patch.object(panel_module, 'get_gpu_info', return_value={'available': False}), \
                patch.object(panel_module.ModelManager, 'is_model_available', return_value=True):
            panel = panel_module.ClassifierDockWidget(None)
        task = FakeTask()
        task.log_message.connect(panel._on_task_log_message)
        panel.task = task
        panel._running = True
        shown = []
        panel._log = shown.append
        panel.close()
        task.log_message.emit('late message of the cancelled run')
        self.assertEqual(shown, [])
        self.assertFalse(panel._running)

    # ---- installer --------------------------------------------------------
    def test_blackwell_datacenter_gpus_get_cu128(self):
        for name, cap in (('NVIDIA B200', 10.0), ('NVIDIA GB300', 10.3),
                          ('NVIDIA GeForce RTX 5060', 12.0)):
            info = {'name': name, 'compute_cap': cap, 'driver_version': '580.82'}
            self.assertEqual(v._select_cuda_index(info), 'cu128', name)
        info = {'name': 'NVIDIA RTX A6000', 'compute_cap': 8.6, 'driver_version': '596.72'}
        self.assertNotEqual(v._select_cuda_index(info), 'cu128')

    def test_gpu_check_reports_missing_kernels_instead_of_the_driver(self):
        out = 'torch=2.14.0+cu126\ncuda_built=12.6\ndevice=NVIDIA B200\narch=sm_100 built=sm_90\narch_ok=False\n'
        result = types.SimpleNamespace(returncode=1, stdout=out, stderr='AssertionError: NO_KERNELS_FOR_GPU')
        with patch.object(v.proc, 'run', return_value=result) as run:
            self.assertFalse(v._verify_cuda_in_venv(tempfile.gettempdir()))
        self.assertEqual(run.call_count, 1)  # deterministic: no retry
        self.assertEqual(v._last_cuda_check.get('no_kernels_for'), 'sm_100')

    def test_other_python_environments_are_kept(self):
        with tempfile.TemporaryDirectory() as cache, patch.object(v, 'CACHE_DIR', cache):
            names = {'venv_py3.10': None, 'venv_py3.11': v._INSTALL_SCHEMA_VERSION,
                     'venv_py3.13': 'old-schema'}
            for name, schema in names.items():
                os.makedirs(os.path.join(cache, name))
                if schema is not None:
                    with open(os.path.join(cache, name, v._INSTALL_MARKER_FILENAME), 'w') as f:
                        json.dump({'install_schema_version': schema}, f)
            removed = sorted(Path(p).name for p in v.cleanup_old_venv_directories())
            self.assertEqual(removed, ['venv_py3.10', 'venv_py3.13'])
            self.assertTrue(os.path.isdir(os.path.join(cache, 'venv_py3.11')))
        # Never before the new environment is ready: a failed install keeps them.
        with tempfile.TemporaryDirectory() as cache, patch.object(v, 'CACHE_DIR', cache), \
                patch.object(v, 'cleanup_old_venv_directories') as cleanup, \
                patch.object(pm, 'standalone_python_exists', return_value=False), \
                patch.object(pm, 'download_python_standalone', return_value=(False, 'offline')), \
                patch.object(v, '_get_qgis_python', return_value=None):
            self.assertFalse(v.create_venv_and_install()[0])
            cleanup.assert_not_called()

    def test_failed_spconv_is_retried_alone(self):
        marker = {'install_schema_version': v._INSTALL_SCHEMA_VERSION, 'cuda_mode': 'cuda',
                  'cuda_index': 'cu128', 'packages': ['torch'], 'spconv_build': 'cu126',
                  'spconv_error': 'temporary network error'}
        with patch.object(v, '_read_install_marker', return_value=marker):
            self.assertEqual(v.missing_venv_packages(marker), [])  # still ready
            self.assertEqual(v.spconv_retry_pending(marker), ['spconv-cu126'])
            self.assertTrue(v.venv_completion_pending())
            self.assertIn('earlier attempt', v.completion_status_message())
        installed = dict(marker, packages=['torch', 'spconv-cu126'], spconv_error=None)
        self.assertEqual(v.spconv_retry_pending(installed), [])

    def test_access_denied_is_not_called_antivirus(self):
        self.assertFalse(v._is_antivirus_error('PermissionError: [WinError 5] Access is denied'))
        self.assertTrue(v._is_access_denied_error('PermissionError: [WinError 5] Access is denied'))
        self.assertIn('not writable', v._format_access_denied_help('torch'))
        self.assertTrue(v._is_antivirus_error(
            'Operation did not complete successfully because the file contains a virus'))

    def test_bootstrap_downloads_are_checked_before_use(self):
        with tempfile.TemporaryDirectory() as cache, \
                patch.object(uv, 'UV_DIR', os.path.join(cache, 'uv')), \
                patch.object(uv, 'QgsBlockingNetworkRequest', _FakeRequest), \
                patch.object(uv, 'verify_uv', return_value=(False, 'damaged')):
            os.makedirs(os.path.join(cache, 'uv'))
            Path(uv.get_uv_path()).write_bytes(b'half a binary')
            ok, message = uv.download_uv()
            self.assertFalse(ok)
            self.assertIn('integrity check', message)
            self.assertFalse(os.path.exists(uv.get_uv_path()))
        with tempfile.TemporaryDirectory() as cache, \
                patch.object(pm, 'CACHE_DIR', cache), \
                patch.object(pm, 'STANDALONE_DIR', os.path.join(cache, 'python_standalone')), \
                patch.object(pm, 'QgsBlockingNetworkRequest', _FakeRequest):
            ok, message = pm.download_python_standalone()
            self.assertFalse(ok)
            self.assertIn('integrity check', message)
            self.assertFalse(os.path.exists(pm.get_standalone_dir()))

    def test_portable_python_folder_per_version(self):
        major, minor = pm.get_qgis_python_version()

        def fake_python(folder, version_minor):
            python = Path(folder) / 'python'
            if sys.platform == 'win32':
                python.mkdir(parents=True)
                (python / f'python3{version_minor}.dll').write_bytes(b'')
                (python / 'python.exe').write_bytes(b'')
            else:
                (python / 'lib' / f'python3.{version_minor}').mkdir(parents=True)
                (python / 'bin').mkdir()
                (python / 'bin' / 'python3').write_bytes(b'')
        for legacy_minor, expected in ((minor, 'python_standalone'),
                                       (minor + 1, f'python_standalone_{major}.{minor}')):
            with tempfile.TemporaryDirectory() as cache, \
                    patch.object(pm, 'CACHE_DIR', cache), \
                    patch.object(pm, 'STANDALONE_DIR', os.path.join(cache, 'python_standalone')):
                fake_python(pm.STANDALONE_DIR, legacy_minor)
                self.assertEqual(Path(pm.get_standalone_dir()).name, expected)
                # Files only: no interpreter is started on the main thread.
                with patch.object(pm.proc, 'run', side_effect=AssertionError('started a process')):
                    self.assertEqual(pm.standalone_python_exists(), legacy_minor == minor)

    def test_partial_python_is_downloaded_again(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'python.exe'
            p.write_bytes(b'partial')
            with patch.object(pm, 'CACHE_DIR', folder), \
                    patch.object(pm, 'standalone_python_exists', return_value=True), \
                    patch.object(pm, 'get_standalone_python_path', return_value=str(p)):
                self.assertFalse(pm.verify_standalone_python(str(p))[0])
                ok, message = pm.download_python_standalone(cancel_check=lambda: True)
            self.assertFalse(ok)
            self.assertIn('cancelled', message)  # it went on to download


if __name__ == '__main__':
    unittest.main(verbosity=2)
