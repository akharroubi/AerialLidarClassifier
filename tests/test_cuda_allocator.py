"""The allocator mode: in automatic mode the standard allocator on Windows
(GitHub issue 8: intermittent driver failures no probe detects), expandable
segments elsewhere where the driver supports them, the standard allocator
(with an explanation) where it does not or where they failed before, no
change when the user configured PYTORCH_CUDA_ALLOC_CONF; the user's explicit
choice (plugin setting or environment override) wins; a failure during a
run switches to the standard allocator once and is remembered."""

import os
import unittest
from unittest import mock

from _load import load_plugin_module

alloc = load_plugin_module("core.cuda_allocator")


class _FakeTorch:
    """Records allocator settings; ``fail`` makes the probe allocation raise."""

    uint8 = "uint8"

    def __init__(self, fail=None, new_api=True, warn=None):
        self.calls = []
        self.fail = fail
        self.warn = warn
        if new_api:
            self._C = mock.Mock(_accelerator_setAllocatorSettings=self.calls.append)
        else:
            self.cuda = mock.Mock()
            self.cuda.memory._set_allocator_settings = self.calls.append

    def empty(self, n, dtype=None, device=None):
        assert n == 1 << 20 and dtype == "uint8" and device == "cuda:0"
        if self.warn is not None:
            import warnings
            warnings.warn(self.warn, UserWarning)
        if self.fail is not None:
            raise self.fail
        return object()


class AllocatorMode(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ.pop("PYTORCH_ALLOC_CONF", None)
        os.environ.pop(alloc.ENV_OVERRIDE, None)
        alloc._PLATFORM_UNSUPPORTED = False
        self.addCleanup(setattr, alloc, "_PLATFORM_UNSUPPORTED", False)
        alloc._active.update(mode=None, chosen_by=None, signature=None)
        self.addCleanup(alloc._active.update, mode=None, chosen_by=None, signature=None)
        self.remembered = []
        for name, value in (("_remembered_failures", lambda: list(self.remembered)),
                            ("_remember_failure", self.remembered.append),
                            ("driver_version", lambda: "596.72")):
            patcher = mock.patch.object(alloc, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.messages = []

    def configure(self, torch, supports, mode="auto", platform="linux"):
        return alloc.configure_cuda_allocator(torch, self.messages.append, supports=supports,
                                              mode=mode, platform=platform)

    def test_driver_without_virtual_memory_keeps_the_standard_allocator(self):
        torch = _FakeTorch()
        self.assertEqual(self.configure(torch, lambda i: False), "standard")
        self.assertEqual(torch.calls, ["expandable_segments:False"])
        self.assertIn("standard allocator", self.messages[0])
        self.assertIn("vGPU", self.messages[0])
        self.assertIn("GPU memory allocator", self.messages[0])

    def test_supported_driver_enables_expandable_segments_and_says_so(self):
        torch = _FakeTorch()
        self.assertEqual(self.configure(torch, lambda i: True), "expandable")
        self.assertEqual(torch.calls, ["expandable_segments:True"])
        self.assertEqual(len(self.messages), 1)
        self.assertIn("expandable segments", self.messages[0])

    def test_unknown_driver_relies_on_the_probe(self):
        torch = _FakeTorch(fail=RuntimeError("CUDA driver error: operation not supported"))
        self.assertEqual(self.configure(torch, lambda i: None), "standard")
        self.assertEqual(torch.calls, ["expandable_segments:True", "expandable_segments:False"])
        self.assertIn("operation not supported", self.messages[0])

    def test_driver_refusing_the_probe_falls_back_and_is_remembered(self):
        torch = _FakeTorch(fail=RuntimeError("CUDA driver error: unknown error"))
        self.assertEqual(self.configure(torch, lambda i: True), "standard")
        self.assertEqual(torch.calls[-1], "expandable_segments:False")
        self.assertIn("unknown error", self.messages[0])
        self.assertEqual(self.remembered, ["unknown GPU|unknown|596.72"])

    def test_windows_automatic_mode_uses_the_standard_allocator_without_probing(self):
        # Issue 8: dual RTX A6000, driver 596.72, Windows: the driver reports
        # virtual memory management, a 1 MB probe passes, and allocations of
        # 4 to 32 MB fail later, differently from run to run.
        torch = _FakeTorch(fail=AssertionError("no GPU allocation may happen here"))
        self.assertEqual(self.configure(torch, lambda i: True, platform="win32"), "standard")
        self.assertEqual(torch.calls, ["expandable_segments:False"])
        self.assertEqual(len(self.messages), 1)
        self.assertTrue(self.messages[0].startswith("CUDA memory: PyTorch's standard allocator"))
        self.assertIn("automatic choice on Windows", self.messages[0])
        self.assertIn("GPU memory allocator", self.messages[0])
        self.assertEqual(alloc._active["mode"], "standard")
        # The explicit setting still forces expandable segments on Windows.
        self.messages.clear()
        forced = _FakeTorch()
        self.assertEqual(self.configure(forced, lambda i: True, mode="expandable",
                                        platform="win32"), "expandable")
        self.assertEqual(forced.calls, ["expandable_segments:True"])

    def test_remembered_failure_starts_with_the_standard_allocator(self):
        self.remembered.append("unknown GPU|unknown|596.72")
        torch = _FakeTorch(fail=AssertionError("no GPU allocation may happen here"))
        self.assertEqual(self.configure(torch, lambda i: True), "standard")
        self.assertEqual(torch.calls, ["expandable_segments:False"])
        self.assertIn("failed in an earlier run", self.messages[0])
        # Another driver version is another computer for the allocator.
        with mock.patch.object(alloc, "driver_version", lambda: "610.88"):
            self.messages.clear()
            self.assertEqual(self.configure(_FakeTorch(), lambda i: True), "expandable")

    def test_failure_during_a_run_switches_to_standard_once_and_is_remembered(self):
        torch = _FakeTorch()
        self.assertEqual(self.configure(torch, lambda i: True), "expandable")
        self.messages.clear()
        for text in ("CUDA driver error: unknown error",
                     '!handles_.at(i) INTERNAL ASSERT FAILED at "C:\\actions-runner\\_work\\'
                     'pytorch\\pytorch\\c10\\cuda\\CUDACachingAllocator.cpp":487'):
            self.assertTrue(alloc.looks_like_allocator_failure(RuntimeError(text)), text)
        self.assertTrue(alloc.recover_from_allocator_failure(
            torch, RuntimeError("CUDA driver error: unknown error"), self.messages.append))
        self.assertEqual(torch.calls[-1], "expandable_segments:False")
        self.assertEqual(self.remembered, ["unknown GPU|unknown|596.72"])
        self.assertIn("switched to PyTorch's standard allocator", self.messages[0])
        self.assertTrue(self.messages[0].startswith("CUDA memory:"))
        # Already on the standard allocator: nothing more to recover.
        self.assertFalse(alloc.recover_from_allocator_failure(
            torch, RuntimeError("CUDA driver error: unknown error"), self.messages.append))
        self.assertIn("restart QGIS", str(alloc.restart_advice(RuntimeError("x"))).replace("Restart", "restart"))

    def test_other_errors_during_a_run_are_not_recovered(self):
        torch = _FakeTorch()
        self.configure(torch, lambda i: True)
        for text in ("CUDA error: an illegal memory access was encountered",
                     "Expected all tensors to be on the same device"):
            self.assertFalse(alloc.looks_like_allocator_failure(RuntimeError(text)), text)
            self.assertFalse(alloc.recover_from_allocator_failure(torch, RuntimeError(text)))
        self.assertEqual(torch.calls, ["expandable_segments:True"])

    def test_no_recovery_with_the_standard_allocator_or_a_forced_setting(self):
        torch = _FakeTorch()
        self.configure(torch, lambda i: True, mode="standard")
        self.assertFalse(alloc.recover_from_allocator_failure(
            torch, RuntimeError("CUDA driver error: unknown error")))
        forced = _FakeTorch()
        self.configure(forced, lambda i: True, mode="expandable")
        with self.assertRaisesRegex(RuntimeError, "Automatic or Standard"):
            alloc.recover_from_allocator_failure(
                forced, RuntimeError("CUDA driver error: unknown error"))
        self.assertEqual(self.remembered, [])

    def test_other_errors_are_not_swallowed(self):
        torch = _FakeTorch(fail=RuntimeError("CUDA out of memory"))
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            self.configure(torch, lambda i: True)

    def test_user_environment_setting_is_respected_in_automatic_mode(self):
        # Applied explicitly, so a mode forced earlier in the session is undone.
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        torch = _FakeTorch()
        self.assertEqual(self.configure(torch, lambda i: True, mode="expandable"), "expandable")
        self.assertEqual(self.configure(torch, lambda i: True, mode="auto"), "standard")
        self.assertEqual(torch.calls[-1], "expandable_segments:False")
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF", self.messages[-1])

    def test_new_environment_name_is_read_too(self):
        for platform, value, expected in (("win32", "True", "expandable"),
                                          ("linux", "False", "standard")):
            os.environ["PYTORCH_ALLOC_CONF"] = f"expandable_segments:{value}"
            try:
                torch = _FakeTorch()
                self.assertEqual(self.configure(torch, lambda i: True, mode="auto",
                                                platform=platform), expected)
                self.assertEqual(torch.calls[0], f"expandable_segments:{value}")
                self.assertIn("PYTORCH_ALLOC_CONF", self.messages[-1])
            finally:
                os.environ.pop("PYTORCH_ALLOC_CONF", None)

    def test_unreadable_environment_value_is_left_alone(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:maybe"
        torch = _FakeTorch()
        self.assertEqual(self.configure(torch, lambda i: True), "unchanged")
        self.assertEqual(torch.calls, [])

    def test_failure_with_environment_expandable_names_the_variable(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch = _FakeTorch()
        self.assertEqual(self.configure(torch, lambda i: True, platform="linux"), "expandable")
        with self.assertRaisesRegex(RuntimeError, "PYTORCH_CUDA_ALLOC_CONF"):
            alloc.recover_from_allocator_failure(
                torch, RuntimeError("CUDA driver error: unknown error"))

    def test_standard_setting_skips_the_probe(self):
        torch = _FakeTorch(fail=RuntimeError("must not be allocated"))
        self.assertEqual(self.configure(torch, lambda i: True, mode="standard"), "standard")
        self.assertEqual(torch.calls, ["expandable_segments:False"])
        self.assertIn("plugin setting", self.messages[0])

    def test_expandable_setting_forces_them_and_reports_a_refusal(self):
        torch = _FakeTorch()
        self.assertEqual(self.configure(torch, lambda i: False, mode="expandable"), "expandable")
        self.assertEqual(torch.calls, ["expandable_segments:True"])
        refused = _FakeTorch(fail=RuntimeError("CUDA driver error: unknown error"))
        with self.assertRaisesRegex(RuntimeError, "Automatic or Standard"):
            self.configure(refused, lambda i: True, mode="expandable")
        self.assertEqual(refused.calls[-1], "expandable_segments:False")

    def test_explicit_setting_wins_over_the_environment_variable(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        torch = _FakeTorch()
        self.assertEqual(self.configure(torch, lambda i: True, mode="standard"), "standard")
        self.assertEqual(torch.calls, ["expandable_segments:False"])

    def test_environment_override_selects_the_mode(self):
        os.environ[alloc.ENV_OVERRIDE] = "Standard"
        self.assertEqual(alloc.preferred_mode(), "standard")
        os.environ[alloc.ENV_OVERRIDE] = "nonsense"
        self.assertEqual(alloc.preferred_mode(), "auto")
        del os.environ[alloc.ENV_OVERRIDE]
        # No QGIS here: the setting cannot be read, automatic mode applies.
        self.assertEqual(alloc.preferred_mode(), "auto")
        torch = _FakeTorch()
        os.environ[alloc.ENV_OVERRIDE] = "standard"
        self.assertEqual(alloc.configure_cuda_allocator(torch, self.messages.append,
                                                        supports=lambda i: True), "standard")

    def test_torch_without_platform_support_reports_the_standard_allocator(self):
        # torch 2.11 on Windows: the setting is accepted, a warning is printed
        # once per process, the standard allocator serves. Seen in the 1.3.2
        # qgis_process run with torch 2.11.0+cu128.
        warning = ("expandable_segments not supported on this platform (Triggered "
                   "internally at c10/cuda/CUDAAllocatorConfig.h:39.)")
        torch = _FakeTorch(warn=warning)
        self.assertEqual(self.configure(torch, lambda i: True), "standard")
        self.assertIn("no expandable segments on this platform", self.messages[0])
        self.assertTrue(alloc._PLATFORM_UNSUPPORTED)
        # The warning is printed once per process: the second model of the
        # session must still be reported truthfully.
        self.messages.clear()
        self.assertEqual(self.configure(_FakeTorch(), lambda i: True), "standard")
        self.assertIn("no expandable segments on this platform", self.messages[0])
        # The explicit setting cannot force what the build lacks: reported, not raised.
        self.messages.clear()
        self.assertEqual(self.configure(_FakeTorch(), lambda i: True, mode="expandable"), "standard")
        self.assertIn("no expandable segments on this platform", self.messages[0])

    def test_other_warnings_from_the_probe_are_passed_on(self):
        torch = _FakeTorch(warn="something unrelated")
        with self.assertWarnsRegex(UserWarning, "something unrelated"):
            self.assertEqual(self.configure(torch, lambda i: True), "expandable")
        self.assertFalse(alloc._PLATFORM_UNSUPPORTED)

    def test_older_torch_setter_is_used(self):
        torch = _FakeTorch(new_api=False)
        self.assertEqual(self.configure(torch, lambda i: True), "expandable")
        self.assertEqual(torch.calls, ["expandable_segments:True"])

    def test_torch_without_setter_changes_nothing_and_says_so(self):
        torch = mock.Mock(spec=[])
        self.assertEqual(self.configure(torch, lambda i: True), "unchanged")
        self.assertEqual(self.configure(torch, lambda i: True, mode="standard"), "unchanged")
        self.assertEqual(self.configure(torch, lambda i: True, platform="win32"), "unchanged")
        self.assertEqual(len(self.messages), 3)
        self.assertTrue(all(m.startswith("CUDA memory:") for m in self.messages))

    def test_every_decision_writes_one_cuda_memory_line(self):
        cases = [dict(supports=lambda i: True), dict(supports=lambda i: False),
                 dict(supports=lambda i: True, platform="win32"),
                 dict(supports=lambda i: True, mode="standard"),
                 dict(supports=lambda i: True, mode="expandable")]
        for case in cases:
            with self.subTest(case=case):
                self.messages.clear()
                self.configure(_FakeTorch(), **case)
                self.assertEqual(len(self.messages), 1)
                self.assertTrue(self.messages[0].startswith("CUDA memory:"))

    def test_driver_query_never_raises(self):
        # Whatever the machine, the query answers True, False or None.
        self.assertIn(alloc.driver_supports_virtual_memory(0), (True, False, None))
        self.assertIsNone(alloc.driver_supports_virtual_memory(10 ** 6) or None)

    def test_labels_cover_every_mode(self):
        self.assertEqual(set(alloc.MODE_LABELS), set(alloc.MODES))
        self.assertEqual(alloc.MODES[0], alloc.MODE_AUTO)


if __name__ == "__main__":
    unittest.main()
