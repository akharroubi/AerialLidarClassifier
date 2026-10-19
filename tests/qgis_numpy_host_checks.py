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

    def run_install(self, host_numpy, proc_run=None, expect_ok=True):
        """install_dependencies with every external step replaced; returns the commands."""
        commands = []
        if proc_run is None:
            proc_run = lambda cmd, **kwargs: types.SimpleNamespace(returncode=0, stdout="", stderr="")

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
             patch.object(vm.proc, "run", side_effect=proc_run):
            ok, message = vm.install_dependencies(venv_dir=venv, cuda_enabled=True)
        if not expect_ok:
            self.assertFalse(ok)
            return message
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


COMPILED = "laspy==2.7.0\nlazrs==0.8.2\nnumpy==1.26.4\nnumpy-indexed==0.3.7\nscipy==1.17.1\n"
CHOSEN = ["laspy==2.7.0", "lazrs==0.8.2", "numpy-indexed==0.3.7", "scipy==1.17.1"]


def no_wheel_for_host_numpy(compile_result=None):
    """proc.run where numpy==1.26.4 has no wheel (QGIS on Python 3.13)."""
    calls = []

    def run(cmd, **kwargs):
        cmd = list(cmd)
        calls.append(cmd)
        if "--dry-run" in cmd and "numpy==1.26.4" in cmd:
            return types.SimpleNamespace(returncode=1, stdout="", stderr=(
                "No solution found: numpy==1.26.4 has no usable wheels"))
        if cmd[1:3] == ["pip", "compile"]:
            return compile_result or types.SimpleNamespace(returncode=0, stdout=COMPILED, stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    run.calls = calls
    return run


class HostNumpyWithoutWheel(unittest.TestCase):
    """QGIS's NumPy 1.x has no wheel for the plugin's Python (Python 3.13+)."""
    run_install = HostNumpy.run_install

    def test_packages_are_chosen_for_qgis_numpy(self):
        run = no_wheel_for_host_numpy()
        commands = self.run_install("1.26.4", proc_run=run)
        compile_cmd = next(c for c in run.calls if c[1:3] == ["pip", "compile"])
        self.assertIn("--only-binary=:all:", compile_cmd)
        self.assertIn("--find-links", compile_cmd)
        for cmd, pins in commands:
            self.assertIsNotNone(pins, cmd)
            self.assertNotIn("numpy==1.26.4", pins, cmd)    # no wheel: never asked for
            for pin in CHOSEN:
                self.assertIn(pin, pins, cmd)
        self.assertEqual(commands[0][1], CHOSEN)             # torch already constrained
        self.assertEqual(commands[1][1], list(INSTALLED_PINS) + CHOSEN)

    def test_stand_in_wheel_resolves_with_compile_flags(self):
        """The resolution input: the packages after torch, and a NumPy stand-in."""
        seen = {}

        def run(cmd, **kwargs):
            cmd = list(cmd)
            if cmd[1:3] == ["pip", "compile"]:
                folder = cmd[cmd.index("--find-links") + 1]
                seen["wheels"] = sorted(os.listdir(folder))
                seen["requirements"] = Path(cmd[3]).read_text().split()
                seen["constraints"] = Path(cmd[cmd.index("--constraint") + 1]).read_text().split()
                import zipfile
                with zipfile.ZipFile(os.path.join(folder, "numpy-1.26.4-py3-none-any.whl")) as z:
                    seen["metadata"] = z.read("numpy-1.26.4.dist-info/METADATA").decode()
            return types.SimpleNamespace(returncode=0, stdout=COMPILED, stderr="")
        with patch.object(vm.proc, "run", side_effect=run):
            pins, error = vm._resolve_against_host_numpy("1.26.4", "venv", "python", {}, {}, True, "uv")
        self.assertEqual((pins, error), (CHOSEN, ""))
        self.assertIn("numpy-1.26.4-py3-none-any.whl", seen["wheels"])
        self.assertIn("Version: 1.26.4", seen["metadata"])
        self.assertEqual(seen["constraints"], ["numpy==1.26.4"])
        self.assertFalse(any(r.startswith(("torch", "timm")) for r in seen["requirements"]))
        self.assertIn("scipy>=1.10", seen["requirements"])

    def test_pip_report_is_read(self):
        def run(cmd, **kwargs):
            cmd = list(cmd)
            if "--report" in cmd:
                import json
                Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps({"install": [
                    {"metadata": {"name": n, "version": v}}
                    for n, v in (line.split("==") for line in COMPILED.split())]}))
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch.object(vm.proc, "run", side_effect=run):
            pins, error = vm._resolve_against_host_numpy("1.26.4", "venv", "python", {}, {}, False, None)
        self.assertEqual((pins, error), (CHOSEN, ""))

    def test_unresolvable_stops_with_a_clear_message(self):
        failed = types.SimpleNamespace(returncode=1, stdout="", stderr="No solution found")
        message = self.run_install("1.26.4", proc_run=no_wheel_for_host_numpy(failed), expect_ok=False)
        self.assertIn("QGIS provides NumPy 1.26.4", message)
        self.assertIn("NumPy 2", message)

    def test_wheel_available_keeps_the_exact_pin(self):
        commands = self.run_install("1.26.4")
        for cmd, pins in commands:
            self.assertIn("numpy==1.26.4", pins, cmd)
            self.assertNotIn("scipy==1.17.1", pins, cmd)


INSTALLED_PINS = ("torch==2.14.0+cu126", "torchvision==0.29.0+cu126")


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

    def test_no_hint_when_packages_were_chosen_for_this_numpy(self):
        with patch.object(vm, "_read_install_marker", return_value={"host_numpy": "1.26.4"}):
            self.assertEqual(self.check("1.26.4", "2.5.2"), "")
        with patch.object(vm, "_read_install_marker", return_value={"host_numpy": "1.24.2"}):
            self.assertIn("QGIS provides NumPy 1.26.4", self.check("1.26.4", "2.5.2"))

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
