"""QGIS builds whose own NumPy is 1.x: the dependency install must match it.

Recent SciPy (1.17+) needs NumPy 2. In a QGIS that ships NumPy 1.x and
loads it before the plugin, the plugin environment's SciPy then fails with
"module 'numpy' has no attribute 'long'" and LitePT is unavailable. Setup
therefore pins NumPy to QGIS's version on those builds only; QGIS builds
with NumPy 2 must get exactly the install commands they got before.

Run with QGIS's python-qgis-ltr.bat (QGIS 3) or python-qgis.bat (QGIS 4).
No network, GPU or installed plugin required.
"""
import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from qgis.core import QgsApplication
app = QgsApplication([], False)
app.initQgis()

import importlib
import importlib.metadata  # noqa: F401 - loaded before tests replace sys.path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
vm = importlib.import_module(ROOT.name + ".utils.venv_manager")
readiness = importlib.import_module(ROOT.name + ".utils.backend_readiness")
uv_manager = importlib.import_module(ROOT.name + ".utils.uv_manager")

INSTALLED = {"torch": "2.14.0+cu126", "torchvision": "0.29.0+cu126"}


def fake_numpy(location, version):
    module = types.ModuleType("numpy")
    module.__file__ = str(Path(location) / "numpy" / "__init__.py")
    module.__version__ = version
    return module


def fake_dist_info(site, name, version):
    folder = Path(site) / f"{name}-{version}.dist-info"
    folder.mkdir(parents=True)
    (folder / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n",
                                     encoding="utf-8")


class HostNumpy(unittest.TestCase):
    def test_version_comes_from_qgis_not_from_the_plugin_environment(self):
        with tempfile.TemporaryDirectory() as root:
            venv = Path(root) / "venv"
            qgis_site = Path(root) / "qgis_site"
            fake_dist_info(qgis_site, "numpy", "1.26.4")
            fake_dist_info(venv / "Lib" / "site-packages", "numpy", "2.5.2")
            paths = [str(venv / "Lib" / "site-packages"), str(qgis_site)]
            # Loaded by QGIS first: that copy wins.
            with patch.dict(sys.modules, {"numpy": fake_numpy(qgis_site, "1.26.3")}):
                self.assertEqual(vm._host_numpy_version(str(venv)), "1.26.3")
            # Loaded from the plugin environment: ignored, QGIS's own copy is found.
            with patch.dict(sys.modules, {"numpy": fake_numpy(venv / "Lib" / "site-packages", "2.5.2")}), \
                 patch.object(vm.sys, "path", paths):
                self.assertEqual(vm._host_numpy_version(str(venv)), "1.26.4")
            # Not loaded yet: same lookup on QGIS's own path.
            saved = sys.modules.pop("numpy", None)
            try:
                with patch.object(vm.sys, "path", paths):
                    self.assertEqual(vm._host_numpy_version(str(venv)), "1.26.4")
                with patch.object(vm.sys, "path", [str(venv / "Lib" / "site-packages")]):
                    self.assertIsNone(vm._host_numpy_version(str(venv)))
            finally:
                if saved is not None:
                    sys.modules["numpy"] = saved

    def test_pin_only_for_numpy1(self):
        for host, expected in (("1.26.4", "numpy==1.26.4"), ("1.24.2+ds1", "numpy==1.24.2"),
                               ("1.21.5", "numpy==1.21.5"), ("2.0.0", None), ("2.4.4", None),
                               ("2.5.0rc1", None), (None, None), ("", None), ("unknown", None)):
            with self.subTest(host=host), patch.object(vm, "_host_numpy_version", return_value=host):
                self.assertEqual(vm._host_numpy_pin(), expected)

    def test_constraints_file(self):
        with tempfile.TemporaryDirectory() as venv:
            with patch.object(vm, "_host_numpy_version", return_value="1.26.4"):
                path = vm._write_torch_constraints(venv, INSTALLED)
                self.assertEqual(Path(path).read_text().split(),
                                 ["torch==2.14.0+cu126", "torchvision==0.29.0+cu126", "numpy==1.26.4"])
                self.assertEqual(Path(vm._write_torch_constraints(venv, {})).read_text().split(),
                                 ["numpy==1.26.4"])
            with patch.object(vm, "_host_numpy_version", return_value="2.4.4"):
                path = vm._write_torch_constraints(venv, INSTALLED)
                self.assertEqual(Path(path).read_text().split(),
                                 ["torch==2.14.0+cu126", "torchvision==0.29.0+cu126"])
                self.assertIsNone(vm._write_torch_constraints(venv, {}))

    def run_install(self, host_numpy):
        """install_dependencies with every external step replaced; returns the commands."""
        commands = []

        def fake_pip(cmd, **kwargs):
            cmd = list(cmd)
            if "--constraint" in cmd:
                at = cmd.index("--constraint") + 1
                text = Path(cmd[at]).read_text()
                cmd[at] = "<constraints file>"   # temp path differs between runs
                commands.append((cmd, text.split()))
            else:
                commands.append((cmd, None))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as venv, \
             patch.object(vm, "_host_numpy_version", return_value=host_numpy), \
             patch.object(vm, "venv_exists", return_value=True), \
             patch.object(uv_manager, "uv_exists", return_value=True), \
             patch.object(uv_manager, "get_uv_path", return_value="uv"), \
             patch.object(vm, "get_venv_python_path", return_value="python"), \
             patch.object(vm, "_is_cpu_torch_installed", return_value=False), \
             patch.object(vm, "detect_nvidia_gpu", return_value=(True, {"name": "NVIDIA A10-8Q"})), \
             patch.object(vm, "_select_cuda_index", return_value="cu126"), \
             patch.object(vm, "_get_installed_versions", return_value=dict(INSTALLED)), \
             patch.object(vm, "_run_pip_install", side_effect=fake_pip), \
             patch.object(vm.proc, "run", return_value=types.SimpleNamespace(returncode=0, stdout="", stderr="")):
            ok, message = vm.install_dependencies(venv_dir=venv, cuda_enabled=True)
        self.assertTrue(ok, message)
        return commands

    def test_numpy1_host_pins_every_install_step(self):
        commands = self.run_install("1.26.4")
        labels = [next(a for a in cmd if a.startswith(("torch", "spconv")) or a == "timm>=0.9.0")
                  for cmd, _ in commands]
        self.assertEqual(labels[:2], ["torch>=2.0.0,<2.15", "torchvision>=0.15.0"])
        self.assertEqual(labels[-1], "spconv-cu126")
        for cmd, pins in commands:
            self.assertIsNotNone(pins, cmd)
            self.assertIn("numpy==1.26.4", pins, cmd)
        # The requested NumPy range itself is unchanged; the constraint narrows it.
        self.assertTrue(any("numpy>=1.26,<3" in cmd for cmd, _ in commands))

    def test_numpy2_host_install_commands_are_unchanged(self):
        commands = self.run_install("2.4.4")
        first_cmd, first_pins = commands[0]
        self.assertIsNone(first_pins)                       # torch: no constraint, as before
        self.assertNotIn("--constraint", first_cmd)
        for cmd, pins in commands[1:]:
            self.assertEqual(pins, ["torch==2.14.0+cu126", "torchvision==0.29.0+cu126"], cmd)
        no_numpy = self.run_install(None)
        self.assertEqual([cmd for cmd, _ in no_numpy], [cmd for cmd, _ in commands])


class ReadinessMessage(unittest.TestCase):
    def check(self, host, ours):
        with tempfile.TemporaryDirectory() as site:
            fake_dist_info(site, "numpy", ours)
            with patch.object(vm, "_host_numpy_version", return_value=host), \
                 patch.object(vm, "get_venv_site_packages", return_value=site):
                return readiness._numpy_mismatch_hint()

    def test_hint_only_for_the_mismatch(self):
        hint = self.check("1.26.4", "2.5.2")
        self.assertIn("QGIS provides NumPy 1.26.4", hint)
        self.assertIn("installed for NumPy 2.5.2", hint)
        self.assertIn("Restart QGIS, then use Plugins >", hint)
        self.assertEqual(self.check("2.4.4", "2.5.2"), "")
        self.assertEqual(self.check("1.26.4", "1.26.4"), "")
        self.assertEqual(self.check(None, "2.5.2"), "")

    def test_status_puts_the_hint_first_and_keeps_details(self):
        failure = (False, "LitePT cannot import scipy.spatial in the plugin environment. "
                          "Details: AttributeError: module 'numpy' has no attribute 'long'")
        with patch.object(readiness, "_prepare_environment", return_value=True), \
             patch.object(readiness, "_check_imports", return_value=failure), \
             patch.object(readiness, "_numpy_mismatch_hint", return_value="QGIS provides NumPy 1.26.4 ..."):
            ready, reason = readiness.litept_dependency_status()
        self.assertFalse(ready)
        self.assertTrue(reason.startswith("QGIS provides NumPy 1.26.4"))
        self.assertIn("has no attribute 'long'", reason)
        with patch.object(readiness, "_prepare_environment", return_value=True), \
             patch.object(readiness, "_check_imports", return_value=(True, "")):
            self.assertEqual(readiness.litept_dependency_status(), (True, ""))


if __name__ == "__main__":
    unittest.main(verbosity=2)
