"""Regression checks with simulated GPU/driver responses; no GPU is required."""

import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from qgis.core import QgsApplication
app = QgsApplication([], False)
app.initQgis()

import importlib
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.append(str(Path.home() / '.qgis_aerial_lidar_classifier/venv_py3.12/Lib/site-packages'))


def module(name):
    return importlib.import_module(ROOT.name + "." + name)


venv = module("utils.venv_manager")
pm = module("utils.python_manager")
uv = module("utils.uv_manager")
registry = module("core.registry")
config = module("config")


class GpuReadiness(unittest.TestCase):
    def test_ada_3500_uses_cuda_index_with_spconv_for_supported_drivers(self):
        for driver, expected in (("580.97", "cu126"), ("560.94", "cu126"),
                                 ("555.99", "cu124"), ("536.67", "cu121")):
            with self.subTest(driver=driver):
                index = venv._select_cuda_index({"name": "NVIDIA RTX 3500 Ada Generation Laptop GPU",
                                                "compute_cap": 8.9, "driver_version": driver})
                self.assertEqual(index, expected)
                self.assertEqual(venv._spconv_extra_for(index), ["spconv-" + index])
        self.assertNotIn("cu128", venv._cuda_candidates(False))

    def test_workstation_ada_name_does_not_match_geforce_rtx50(self):
        for name in ("NVIDIA RTX 3500 Ada Generation Laptop GPU", "NVIDIA RTX 5000 Ada Generation",
                     "NVIDIA RTX 5000"):
            with self.subTest(name=name):
                self.assertEqual(venv._select_cuda_index({"name": name, "driver_version": "580.97"}), "cu126")
        for name in ("NVIDIA GeForce RTX 5090", "NVIDIA RTX PRO 5000 Blackwell"):
            self.assertEqual(venv._select_cuda_index({"name": name, "driver_version": "580.97"}), "cu128")

    def test_nvidia_smi_ada_detection_is_not_excluded(self):
        result = types.SimpleNamespace(returncode=0, stderr="",
            stdout="NVIDIA RTX 3500 Ada Generation Laptop GPU, 8.9, 580.97, 12288\n")
        with patch.object(venv, "_gpu_detect_cache", None), \
             patch.object(venv, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(venv.proc, "run", return_value=result):
            detected, info = venv.detect_nvidia_gpu()
        self.assertTrue(detected)
        self.assertEqual(info["compute_cap"], 8.9)
        self.assertEqual(venv._select_cuda_index(info), "cu126")

    RTX5080 = {"name": "NVIDIA GeForce RTX 5080 Laptop GPU", "compute_cap": 12.0,
               "memory_mb": 16303, "driver_version": "595.99.02"}

    def test_flatpak_without_nvidia_smi_finds_the_gpu_through_nvml(self):
        """Flathub QGIS: the sandbox has the driver libraries but no nvidia-smi."""
        with patch.object(venv, "_gpu_detect_cache", None), \
             patch.object(venv, "_find_nvidia_smi", return_value=None), \
             patch.object(venv, "_detect_with_nvml", return_value=[dict(self.RTX5080)]) as nvml:
            self.assertEqual(venv.detect_nvidia_gpu(), (True, self.RTX5080))
            self.assertEqual(venv.detect_nvidia_gpu(), (True, self.RTX5080))   # cached
        self.assertEqual(nvml.call_count, 1)
        index = venv._select_cuda_index(self.RTX5080)
        self.assertEqual(index, "cu128")
        # No spconv-cu128 wheel: the cu126 build, compiled by the driver.
        self.assertEqual(venv._spconv_build_for(index), "cu126")
        self.assertTrue(venv._spconv_is_fallback(index))
        self.assertEqual(venv._spconv_extra_for(index), ["spconv-cu126"])
        for native in ("cu118", "cu121", "cu124", "cu126"):
            self.assertEqual(venv._spconv_build_for(native), native)
            self.assertFalse(venv._spconv_is_fallback(native))
        self.assertIsNone(venv._spconv_build_for(None))
        self.assertEqual(venv._spconv_extra_for("cu130"), [])

    def test_nvidia_smi_failure_falls_back_to_nvml(self):
        failed = types.SimpleNamespace(returncode=9, stdout="", stderr="NVML: Unknown Error")
        for run in ({"return_value": failed},
                    {"side_effect": venv.proc.TimeoutExpired("nvidia-smi", 20)}):
            with self.subTest(run=run), patch.object(venv, "_gpu_detect_cache", None), \
                 patch.object(venv, "_find_nvidia_smi", return_value="nvidia-smi"), \
                 patch.object(venv.proc, "run", **run), \
                 patch.object(venv, "_detect_with_nvml", return_value=[dict(self.RTX5080)]):
                self.assertEqual(venv.detect_nvidia_gpu(), (True, self.RTX5080))

    def test_timeout_without_nvml_is_not_cached(self):
        with patch.object(venv, "_gpu_detect_cache", None), \
             patch.object(venv, "_find_nvidia_smi", return_value="nvidia-smi"), \
             patch.object(venv.proc, "run", side_effect=venv.proc.TimeoutExpired("nvidia-smi", 20)), \
             patch.object(venv, "_detect_with_nvml", return_value=[]):
            self.assertEqual(venv.detect_nvidia_gpu(), (False, {}))
            self.assertIsNone(venv._gpu_detect_cache)
        with patch.object(venv, "_gpu_detect_cache", None), \
             patch.object(venv, "_find_nvidia_smi", return_value=None), \
             patch.object(venv, "_detect_with_nvml", return_value=[]):
            self.assertEqual(venv.detect_nvidia_gpu(), (False, {}))
            self.assertEqual(venv._gpu_detect_cache, (False, {}))

    def test_nvml_matches_nvidia_smi_on_this_computer(self):
        smi, _ = venv._detect_with_nvidia_smi()
        if not smi:
            self.skipTest("no NVIDIA GPU on this computer")
        self.assertEqual(venv._detect_with_nvml(), smi)

    def test_setup_text_for_rtx50_explains_the_compiled_spconv_build(self):
        dialog_module = module("dialogs.deps_install_dialog")
        texts = {}
        for name, info in (("rtx50", self.RTX5080),
                           ("ada", {"name": "NVIDIA RTX 3500 Ada Generation Laptop GPU",
                                    "compute_cap": 8.9, "driver_version": "580.97"})):
            with patch.object(venv, "detect_nvidia_gpu", return_value=(True, info)):
                dock = dialog_module.DepsInstallDockWidget()
                texts[name] = dock.gpu_label.text()
                dock.deleteLater()
        self.assertIn("RTX 5080 Laptop GPU (16303 MB)", texts["rtx50"])
        self.assertIn("will both be available", texts["rtx50"])
        self.assertIn("CUDA 12.6 build", texts["rtx50"])
        self.assertIn("compiles it for the GPU", texts["rtx50"])
        self.assertIn("will both be available", texts["ada"])
        self.assertNotIn("CUDA 12.6 build", texts["ada"])

    CU128_MARKER_131 = {"install_schema_version": venv._INSTALL_SCHEMA_VERSION,
                        "plugin_version": "1.3.1", "cuda_mode": "cuda", "cuda_index": "cu128",
                        "torch_version": "2.11.0+cu128",
                        "packages": ["torch", "torchvision", "laspy", "lazrs", "timm",
                                     "numpy_indexed", "scipy"]}

    def test_missing_packages_only_for_rtx50_environments_of_earlier_releases(self):
        self.assertEqual(venv.missing_venv_packages(dict(self.CU128_MARKER_131)), ["spconv-cu126"])
        # 1.3.2 recorded its attempt (success or failure): never asked again.
        done = dict(self.CU128_MARKER_131, spconv_build="cu126", spconv_error="No solution")
        self.assertEqual(venv.missing_venv_packages(done), [])
        native = dict(self.CU128_MARKER_131, cuda_index="cu126",
                      packages=self.CU128_MARKER_131["packages"] + ["spconv-cu126"])
        self.assertEqual(venv.missing_venv_packages(native), [])
        self.assertEqual(venv.missing_venv_packages({"cuda_mode": "cpu", "packages": ["torch"]}), [])
        self.assertEqual(venv.missing_venv_packages(None if False else {}), [])

    def test_venv_status_asks_to_complete_the_rtx50_environment_not_to_rebuild(self):
        with patch.object(pm, "standalone_python_exists", return_value=True), \
             patch.object(venv, "venv_exists", return_value=True), \
             patch.object(venv, "_read_install_marker", return_value=dict(self.CU128_MARKER_131)), \
             patch.object(venv, "_quick_check_packages", return_value=(True, "")):
            ready, message = venv.get_venv_status()
            self.assertFalse(ready)
            self.assertIn("spconv-cu126", message)
            self.assertIn("existing environment", message)
            self.assertNotIn("Reinstall", message)
            self.assertTrue(venv.venv_completion_pending())
            with patch.object(venv, "_read_install_marker", return_value=dict(
                    self.CU128_MARKER_131, install_schema_version="1")):
                # An outdated pipeline is rebuilt, never completed.
                self.assertFalse(venv.venv_completion_pending())

    def _completion(self, install_result, import_rc=0):
        written = []
        run = types.SimpleNamespace(returncode=import_rc, stdout="2.3.8", stderr="DLL load failed")
        with patch.object(venv, "_read_install_marker", return_value=dict(self.CU128_MARKER_131)), \
             patch.object(venv, "_install_spconv", return_value=install_result) as install, \
             patch.object(venv.proc, "run", return_value=run), \
             patch.object(venv, "_get_installed_versions", return_value={"spconv-cu126": "2.3.8"}), \
             patch.object(venv, "_write_install_marker",
                          side_effect=lambda venv_dir, extra: written.append(extra)), \
             patch.object(uv, "uv_exists", return_value=True), \
             patch.object(uv, "get_uv_path", return_value="uv"):
            progress = []
            ok, message = venv.complete_venv_packages(progress_callback=lambda p, m: progress.append(m))
        self.assertEqual(install.call_args.args[3], "cu126")
        self.assertEqual(len(written), 1)
        return ok, message, written[0], progress

    def test_completion_adds_spconv_in_place_and_records_it(self):
        ok, message, marker, progress = self._completion((True, ""))
        self.assertTrue(ok, message)
        self.assertIn("spconv-cu126", marker["packages"])
        self.assertEqual(marker["packages"][:2], ["torch", "torchvision"])
        self.assertEqual(marker["spconv_build"], "cu126")
        self.assertTrue(marker["spconv_fallback"])
        self.assertIsNone(marker["spconv_error"])
        self.assertEqual(marker["spconv_version"], "2.3.8")
        self.assertEqual(marker["cuda_index"], "cu128")
        self.assertNotIn("install_schema_version", marker)
        self.assertTrue(any("existing environment" in m for m in progress))
        self.assertEqual(venv.missing_venv_packages(dict(self.CU128_MARKER_131, **marker)), [])

    def test_failed_completion_is_recorded_and_explained(self):
        ok, message, marker, _ = self._completion((False, "No solution found"))
        self.assertFalse(ok)
        self.assertIn("No solution found", message)
        self.assertIn("SegFormer 3D", message)
        self.assertNotIn("spconv-cu126", marker["packages"])
        self.assertEqual(marker["spconv_error"], "No solution found")
        self.assertEqual(marker["spconv_build"], "cu126")
        ok, message, marker, _ = self._completion((True, ""), import_rc=1)
        self.assertFalse(ok)
        self.assertIn("DLL load failed", marker["spconv_error"])
        self.assertNotIn("spconv-cu126", marker["packages"])

    def test_install_worker_completes_instead_of_rebuilding(self):
        import shutil
        worker_module = module("workers.deps_install_worker")
        results = []
        worker = worker_module.DepsInstallWorker(cuda_enabled=True)
        worker.completed.connect(lambda ok, message: results.append((ok, message)))
        with patch.object(venv, "venv_completion_pending", return_value=True), \
             patch.object(venv, "complete_venv_packages", return_value=(True, "Virtual environment ready")), \
             patch.object(worker, "_download_weights", side_effect=lambda message: message), \
             patch.object(shutil, "rmtree") as rmtree:
            worker.run()
        rmtree.assert_not_called()
        self.assertEqual(results, [(True, "Virtual environment ready")])
        # The CPU-only choice is a rebuild, never a completion.
        results.clear()
        cpu_worker = worker_module.DepsInstallWorker(cuda_enabled=False)
        cpu_worker.completed.connect(lambda ok, message: results.append((ok, message)))
        with patch.object(venv, "venv_completion_pending", return_value=True) as pending, \
             patch.object(worker_module, "_plugin_torch_loaded", return_value=True):
            cpu_worker.run()
        pending.assert_not_called()
        self.assertEqual(len(results), 1)
        self.assertIn("Restart QGIS", results[0][1])

    def test_setup_dock_completion_ui_offers_to_add_packages(self):
        dialog_module = module("dialogs.deps_install_dialog")
        with patch.object(venv, "detect_nvidia_gpu", return_value=(True, self.RTX5080)):
            dock = dialog_module.DepsInstallDockWidget()
        try:
            dock.show_completion_ui(venv.completion_message(["spconv-cu126"]))
            self.assertEqual(dock.install_button.text(), "Add missing packages")
            self.assertTrue(dock.cpu_only.isHidden())
            self.assertIn("existing environment", dock.status_label.text())
        finally:
            dock.deleteLater()

    def test_allocator_setting_round_trips_through_the_dock_and_the_backend(self):
        panel_module = module("gui.main_panel")
        allocator = module("core.cuda_allocator")
        settings = {f"{config.SETTINGS_PREFIX}/settings_version": 2}

        class Settings:
            def value(self, key, default=None, **kwargs): return settings.get(key, default)
            def setValue(self, key, value): settings[key] = value
        gpu = {"available": True, "backend": "cuda", "name": "NVIDIA RTX 3500 Ada", "mem": 12.}
        with patch.object(panel_module, "QgsSettings", Settings), \
             patch.object(panel_module, "get_gpu_info", return_value=gpu), \
             patch.object(panel_module.ModelManager, "is_model_available", return_value=True):
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(panel.allocator_combo.currentData(), "auto")
                self.assertEqual(panel.allocator_combo.count(), 3)
                panel.allocator_combo.setCurrentIndex(panel.allocator_combo.findData("standard"))
                self.assertEqual(settings[f"{config.SETTINGS_PREFIX}/cuda_allocator"], "standard")
            finally:
                panel.close()
            restored = panel_module.ClassifierDockWidget(None)
            try:
                self.assertEqual(restored.allocator_combo.currentData(), "standard")
            finally:
                restored.close()
        # The backend reads the real QgsSettings; Processing shares them.
        from qgis.core import QgsSettings
        key = f"{config.SETTINGS_PREFIX}/{allocator.SETTING_NAME}"
        previous = QgsSettings().value(key, None)
        try:
            QgsSettings().setValue(key, "expandable")
            self.assertEqual(allocator.preferred_mode(), "expandable")
            QgsSettings().setValue(key, "bogus")
            self.assertEqual(allocator.preferred_mode(), "auto")
            with patch.dict(os.environ, {allocator.ENV_OVERRIDE: "standard"}):
                self.assertEqual(allocator.preferred_mode(), "standard")
        finally:
            if previous is None:
                QgsSettings().remove(key)
            else:
                QgsSettings().setValue(key, previous)

    def test_detected_but_unchecked_gpu_runs_airborne_on_segformer_and_says_why(self):
        panel_module = module("gui.main_panel")
        readiness = module("utils.backend_readiness")
        settings = {f"{config.SETTINGS_PREFIX}/settings_version": 2,
                    f"{config.SETTINGS_PREFIX}/model_id": registry.LITEPT_L_DALES.id,
                    f"{config.SETTINGS_PREFIX}/use_gpu": False}
        class Settings:
            def value(self, key, default=None, **kwargs): return settings.get(key, default)
            def setValue(self, key, value): settings[key] = value
        gpu = {"available": True, "backend": "cuda", "name": "NVIDIA RTX 3500 Ada", "mem": 12.}
        with patch.object(panel_module, "QgsSettings", Settings), \
             patch.object(panel_module, "get_gpu_info", return_value=gpu), \
             patch.object(panel_module.ModelManager, "is_model_available", return_value=True), \
             patch.object(readiness, "litept_dependency_status", return_value=(True, "")):
            panel = panel_module.ClassifierDockWidget(None)
            try:
                # GPU off: the Airborne classifier runs its CPU engine and
                # names the one-click way back to LitePT-L.
                self.assertEqual(panel.model_combo.currentData(), "airborne")
                self.assertEqual(panel._current_spec(), registry.SEGFORMER3D_URBANFILTERING)
                self.assertTrue(panel.model_gate_ok)
                self.assertIn("SegFormer 3D engine (30 cm)", panel.model_label.text())
                self.assertIn("GPU use is switched off", panel.engine_note)
                self.assertIn("RTX 3500 Ada", panel.engine_note)
                self.assertIn("Enable 'Use GPU'", panel.engine_note)
                tooltip = panel.model_label.toolTip()
                self.assertTrue(tooltip.startswith(panel.engine_note.replace("&", "&amp;")), tooltip)
                # The class list follows the engine that runs: SegFormer 3D's seven classes.
                self.assertIn("On this computer: SegFormer 3D Airborne (30 cm), 7 classes", tooltip)
                self.assertIn("Transmission tower", tooltip)
                panel.gpu_check.setChecked(True)
                self.assertTrue(panel.model_gate_ok)
                self.assertEqual(panel._current_spec(), registry.LITEPT_L_DALES)
                self.assertEqual(panel.engine_note, "")
                self.assertNotIn("engine", panel.model_label.text())
                self.assertEqual(panel.model_combo.currentData(), "airborne")
            finally:
                panel.close()

    def test_cpu_only_torch_runs_airborne_on_segformer_and_is_not_an_unsupported_card(self):
        panel_module = module("gui.main_panel")
        settings = {f"{config.SETTINGS_PREFIX}/settings_version": 2,
                    f"{config.SETTINGS_PREFIX}/model_id": registry.LITEPT_L_DALES.id}
        class Settings:
            def value(self, key, default=None, **kwargs): return settings.get(key, default)
            def setValue(self, key, value): settings[key] = value
        with patch.object(panel_module, "QgsSettings", Settings), \
             patch.object(panel_module, "get_gpu_info", return_value={"available": False, "reason": "no_cuda"}), \
             patch.object(panel_module.ModelManager, "is_model_available", return_value=True):
            panel = panel_module.ClassifierDockWidget(None)
            try:
                self.assertTrue(panel.model_gate_ok)
                self.assertEqual(panel._current_spec(), registry.SEGFORMER3D_URBANFILTERING)
                self.assertIn("PyTorch is CPU-only", panel.engine_note)
                self.assertIn("GPU installation", panel.engine_note)
                self.assertIn("SegFormer 3D engine", panel.model_label.text())
                self.assertEqual(panel.model_combo.currentData(), "airborne")
                self.assertTrue(panel.run_btn.isEnabled() or "Add LAS" in panel.run_btn.toolTip())
            finally:
                panel.close()

    def _panel(self, gpu, use_gpu=True, model_id=None):
        panel_module = module("gui.main_panel")
        settings = {f"{config.SETTINGS_PREFIX}/settings_version": 2,
                    f"{config.SETTINGS_PREFIX}/model_id": model_id or registry.LITEPT_L_MLS.id,
                    f"{config.SETTINGS_PREFIX}/use_gpu": use_gpu}

        class Settings:
            def value(self, key, default=None, **kwargs): return settings.get(key, default)
            def setValue(self, key, value): settings[key] = value
        patches = [patch.object(panel_module, "QgsSettings", Settings),
                   patch.object(panel_module, "get_gpu_info", return_value=gpu),
                   patch.object(panel_module.ModelManager, "is_model_available", return_value=True)]
        for item in patches:
            item.start()
        self.addCleanup(lambda: [item.stop() for item in patches])
        panel = panel_module.ClassifierDockWidget(None)
        self.addCleanup(panel.close)
        return panel

    def test_dock_log_shows_each_task_message_once_with_or_without_the_message_log(self):
        """GitHub issue 8: the dock showed none of the task's lines (no "Model:",
        no "CUDA memory:"): on QGIS 4 it listened to a signal that is no longer
        emitted. The task now also hands its messages to the dock through its
        own signal; each line must appear exactly once whether QgsMessageLog
        delivers its copy or not."""
        import time
        from qgis.PyQt.QtCore import QThread
        tasks = module("workers.classifier_task")
        logger = module("utils.logger")
        panel = self._panel({"available": False})
        panel._connect_log()
        task = tasks.ClassificationTask([], Path("."), "_x", registry.LITEPT_L_DALES.id,
                                        "cpu", "classification")

        class Worker(QThread):
            def __init__(self, lines):
                super().__init__()
                self.lines = lines

            def run(self):
                logger.set_thread_sink(task.log_message.emit)
                try:
                    for level, line in self.lines:
                        getattr(logger, level)(line)
                finally:
                    logger.set_thread_sink(None)

        def run_lines(lines):
            panel._running = True
            panel._log_twins.clear()
            worker = Worker(lines)
            worker.start()
            worker.wait(10000)
            for _ in range(30):
                app.processEvents()
                time.sleep(0.01)
            panel._running = False
            return panel.log_text.toPlainText()

        task.log_message.connect(panel._on_task_log_message)
        lines = [("log_info", "Model: LitePT-L Airborne (10 cm) on CUDA"),
                 ("log_warning", "CUDA memory: test line 1"),
                 ("log_warning", "LitePT-L: GPU out of memory; retrying"),
                 ("log_warning", "LitePT-L: GPU out of memory; retrying")]
        text = run_lines(lines)
        self.assertEqual(text.count("Model: LitePT-L Airborne (10 cm) on CUDA"), 1)
        self.assertEqual(text.count("CUDA memory: test line 1"), 1)
        # A message the task logs twice is shown twice, not merged.
        self.assertEqual(text.count("LitePT-L: GPU out of memory; retrying"), 2)

        # QgsMessageLog's copy never arrives (the dock is not connected to it).
        logger.message_log_signal().disconnect(panel._on_log_message)
        panel._log_connected = False
        text = run_lines([("log_warning", "CUDA memory: test line 2"),
                          ("log_error", "Classification error: test")])
        self.assertEqual(text.count("CUDA memory: test line 2"), 1)
        self.assertEqual(text.count("Classification error: test"), 1)

        # Outside a run, plugin messages from the main thread still appear.
        panel._connect_log()
        logger.log_info("Setup message outside a run")
        for _ in range(100):
            app.processEvents()
            if "Setup message outside a run" in panel.log_text.toPlainText():
                break
            time.sleep(0.01)
        self.assertEqual(panel.log_text.toPlainText().count("Setup message outside a run"), 1)

    def test_dock_mirrors_the_message_log_on_this_qgis(self):
        """The dock's log follows the plugin's QgsMessageLog messages on QGIS 3
        (messageReceived) and QGIS 4 (messageReceivedWithFormat)."""
        import time
        logger = module("utils.logger")
        panel = self._panel({"available": False})
        panel._connect_log()
        logger.log_warning("Mirrored message check")
        for _ in range(100):
            app.processEvents()
            if "Mirrored message check" in panel.log_text.toPlainText():
                break
            time.sleep(0.01)
        self.assertEqual(panel.log_text.toPlainText().count("Mirrored message check"), 1)
        # closeEvent disconnects; reopening reconnects.
        panel.close()
        logger.log_warning("While closed")
        app.processEvents()
        panel.show()
        logger.log_warning("After reopening")
        for _ in range(100):
            app.processEvents()
            if "After reopening" in panel.log_text.toPlainText():
                break
            time.sleep(0.01)
        self.assertNotIn("While closed", panel.log_text.toPlainText())
        self.assertEqual(panel.log_text.toPlainText().count("After reopening"), 1)

    def test_buffer_setting_becomes_automatic_and_streaming_option_is_gone(self):
        """1.3.3: a saved 50 m buffer (the old default) becomes Automatic (0);
        a value the user chose is kept; the Streaming I/O checkbox is gone."""
        panel_module = module("gui.main_panel")
        for saved, expected in ((50.0, 0.0), (80.0, 80.0), (None, 0.0)):
            settings = {f"{config.SETTINGS_PREFIX}/settings_version": 2}
            if saved is not None:
                settings[f"{config.SETTINGS_PREFIX}/tile_buffer_m"] = saved

            class Settings:
                def value(self, key, default=None, **kwargs):
                    return settings.get(key, default)

                def setValue(self, key, value):
                    settings[key] = value
            with patch.object(panel_module, "QgsSettings", Settings), \
                 patch.object(panel_module, "get_gpu_info", return_value={"available": False}), \
                 patch.object(panel_module.ModelManager, "is_model_available", return_value=True):
                panel = panel_module.ClassifierDockWidget(None)
                try:
                    self.assertEqual(panel.tile_buffer_spin.value(), expected, saved)
                    self.assertEqual(settings[f"{config.SETTINGS_PREFIX}/settings_version"], 3)
                    if expected == 0.0:
                        self.assertEqual(panel.tile_buffer_spin.text(), "Automatic")
                    self.assertFalse(hasattr(panel, "tile_streaming_check"))
                    self.assertIn("streamed from disk automatically", panel.tile_check.toolTip())
                finally:
                    panel.close()

    def test_gate_messages_never_ask_for_an_impossible_action(self):
        cases = (
            ({"available": False, "reason": "no_gpu"}, "No GPU visible", "driver"),
            ({"available": False, "reason": "torch_import_failed"}, "PyTorch not loaded", "Repair"),
            ({"available": True, "backend": "mps", "name": "Apple Silicon (MPS)", "mem": 0.},
             "Needs an NVIDIA GPU", "macOS"),
        )
        for gpu, status, advice in cases:
            with self.subTest(gpu=gpu):
                panel = self._panel(gpu)
                self.assertFalse(panel.model_gate_ok)
                self.assertIn(status, panel.model_label.text())
                self.assertIn(advice, panel.model_gate_message)
                self.assertNotIn("Tick 'Use GPU'", panel.model_gate_message)
                # The mobile mapping classifier is kept, never swapped for airborne.
                self.assertEqual(panel.model_combo.currentData(), "mobile_mapping")
                self.assertFalse(panel.run_btn.isEnabled())

    def test_closing_the_dock_during_a_run_lets_the_reopened_dock_run(self):
        from unittest.mock import MagicMock
        panel = self._panel({"available": True, "backend": "cuda", "name": "NVIDIA RTX 3090", "mem": 24.})
        task = MagicMock()
        task.isCanceled.return_value = False
        panel.task, panel._running = task, True
        panel.cancel_btn.setEnabled(True)
        panel.close()
        task.cancel.assert_called_once()
        self.assertIsNone(panel.task)
        self.assertFalse(panel._running)
        self.assertFalse(panel.cancel_btn.isEnabled())
        self.assertFalse(panel._log_connected)
        panel.show()
        self.assertTrue(panel._log_connected)
        panel.show()  # a second show must not connect the log twice
        self.assertTrue(panel._log_connected)

    def test_cancel_after_written_outputs_says_they_are_kept(self):
        panel = self._panel({"available": True, "backend": "cuda", "name": "NVIDIA RTX 3090", "mem": 24.})
        panel.task = types.SimpleNamespace(error_message=None, output_files=[Path("street_classified.laz")])
        panel._on_task_terminated()
        self.assertIn("Cancelled after 1 file(s) were classified", panel.current_file_label.text())
        panel.task = types.SimpleNamespace(error_message=None, output_files=[])
        panel._on_task_terminated()
        self.assertIn("left unchanged", panel.current_file_label.text())

    @unittest.skipUnless(sys.platform == "win32", "Windows DLL search paths")
    def test_dll_directories_are_registered_once_per_session(self):
        import tempfile
        with tempfile.TemporaryDirectory() as site:
            (Path(site) / "torch" / "lib").mkdir(parents=True)
            calls = []
            with patch.object(venv, "_DLL_DIRECTORY_HANDLES", {}), \
                 patch.object(venv.os, "add_dll_directory", side_effect=lambda d: calls.append(d) or d), \
                 patch.dict(venv.os.environ, {"PATH": ""}):
                for _ in range(3):
                    venv._add_windows_dll_directories(site)
                self.assertEqual(calls, [str(Path(site) / "torch" / "lib")])
                self.assertEqual(venv.os.environ["PATH"].split(os.pathsep).count(calls[0]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
