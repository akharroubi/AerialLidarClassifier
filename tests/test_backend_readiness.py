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

    def test_environment_failure_does_not_import_system_packages(self):
        with patch.object(readiness, "_prepare_environment", return_value=False), \
             patch.object(readiness.importlib, "import_module") as imports:
            ready, reason = readiness.litept_dependency_status()
        self.assertFalse(ready)
        self.assertIn("environment is unavailable", reason)
        imports.assert_not_called()


if __name__ == "__main__":
    unittest.main()
