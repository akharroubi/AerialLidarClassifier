"""A transient dependency error must not condemn a supported GPU."""

import sys
import types
import unittest
from unittest.mock import patch

from _load import load_plugin_module

readiness = load_plugin_module("utils.backend_readiness")


class BackendReadiness(unittest.TestCase):
    def setUp(self):
        readiness.litept_dependency_status.cache_clear()

    def tearDown(self):
        readiness.litept_dependency_status.cache_clear()

    def test_import_failure_recovers_without_restarting_probe(self):
        with patch.object(readiness, "_prepare_environment", return_value=True), \
             patch.object(readiness.importlib, "import_module",
                          side_effect=[ImportError("missing DLL"), object(), object()]) as imports:
            ready, reason = readiness.litept_dependency_status()
            self.assertFalse(ready)
            self.assertIn("spconv.pytorch", reason)
            self.assertIn("ImportError: missing DLL", reason)
            self.assertNotIn("RTX 50", reason)
            self.assertTrue(readiness.litept_dependency_status()[0])
            self.assertEqual(imports.call_count, 3)

    def test_scipy_error_is_not_misreported_as_spconv_or_gpu_support(self):
        torch = types.SimpleNamespace(__version__="2.test+cu126", __file__="plugin/torch.py",
            version=types.SimpleNamespace(cuda="12.6"),
            cuda=types.SimpleNamespace(is_available=lambda: True,
                get_device_name=lambda index: "NVIDIA RTX 3500 Ada Generation Laptop GPU"))
        with patch.object(readiness, "_prepare_environment", return_value=True), \
             patch.object(readiness.importlib, "import_module",
                          side_effect=[object(), ImportError("scipy native DLL")]), \
             patch.dict(sys.modules, {"torch": torch}):
            ready, reason = readiness.litept_dependency_status()
        self.assertFalse(ready)
        self.assertIn("cannot import scipy.spatial", reason)
        self.assertIn("CUDA build 12.6", reason)
        self.assertIn("plugin/torch.py", reason)
        self.assertIn("RTX 3500 Ada", reason)

    def _with_marker(self, marker, fallback=True):
        """A stand-in venv_manager: the real one imports qgis."""
        fake = types.SimpleNamespace(
            _read_install_marker=lambda: marker,
            _spconv_is_fallback=lambda index: fallback,
        )
        name = readiness.__name__.rsplit(".", 1)[0] + ".venv_manager"
        return patch.dict(sys.modules, {name: fake})

    def test_rtx50_environment_without_spconv_explains_the_completion(self):
        marker = {"cuda_mode": "cuda", "cuda_index": "cu128", "packages": ["torch"]}
        with patch.object(readiness, "_prepare_environment", return_value=True), \
             patch.object(readiness.importlib, "import_module",
                          side_effect=[ImportError("No module named 'spconv'")]), \
             self._with_marker(marker):
            ready, reason = readiness.litept_dependency_status()
        self.assertFalse(ready)
        self.assertIn("CUDA 12.6 build to the existing environment", reason)
        self.assertIn("Repair dependencies", reason)
        self.assertIn("No module named 'spconv'", reason)

    def test_recorded_spconv_failure_is_reported_instead_of_a_loop(self):
        marker = {"cuda_mode": "cuda", "cuda_index": "cu128", "packages": ["torch"],
                  "spconv_build": "cu126", "spconv_fallback": True,
                  "spconv_error": "No solution found"}
        with patch.object(readiness, "_prepare_environment", return_value=True), \
             patch.object(readiness.importlib, "import_module",
                          side_effect=[ImportError("No module named 'spconv'")]), \
             self._with_marker(marker):
            ready, reason = readiness.litept_dependency_status()
        self.assertFalse(ready)
        self.assertIn("could not install spconv-cu126", reason)
        self.assertIn("No solution found", reason)
        self.assertNotIn("existing environment", reason)

    def test_native_spconv_environment_gets_no_rtx50_hint(self):
        marker = {"cuda_mode": "cuda", "cuda_index": "cu126", "packages": ["torch", "spconv-cu126"]}
        with patch.object(readiness, "_prepare_environment", return_value=True), \
             patch.object(readiness.importlib, "import_module",
                          side_effect=[ImportError("DLL load failed")]), \
             self._with_marker(marker, fallback=False):
            ready, reason = readiness.litept_dependency_status()
        self.assertFalse(ready)
        self.assertNotIn("CUDA 12.6", reason)
        self.assertIn("DLL load failed", reason)

    def test_environment_failure_does_not_import_system_packages(self):
        with patch.object(readiness, "_prepare_environment", return_value=False), \
             patch.object(readiness.importlib, "import_module") as imports:
            ready, reason = readiness.litept_dependency_status()
        self.assertFalse(ready)
        self.assertIn("environment is unavailable", reason)
        imports.assert_not_called()


if __name__ == "__main__":
    unittest.main()
