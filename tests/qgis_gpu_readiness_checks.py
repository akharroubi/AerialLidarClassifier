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
        self.assertEqual(venv._spconv_extra_for(index), [])

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

    def test_setup_text_for_rtx50_does_not_promise_litept(self):
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
        self.assertIn("SegFormer 3D on the GPU", texts["rtx50"])
        self.assertNotIn("will both be available", texts["rtx50"])
        self.assertIn("will both be available", texts["ada"])

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
