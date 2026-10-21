"""The allocator mode must follow the driver: expandable segments where they
work, the standard allocator (with an explanation) where the driver refuses
them, and no change when the user configured PYTORCH_CUDA_ALLOC_CONF."""

import os
import unittest
from unittest import mock

from _load import load_plugin_module

alloc = load_plugin_module("core.cuda_allocator")


class _FakeTorch:
    """Records allocator settings; ``fail`` makes the probe allocation raise."""

    uint8 = "uint8"

    def __init__(self, fail=None, new_api=True):
        self.calls = []
        self.fail = fail
        if new_api:
            self._C = mock.Mock(_accelerator_setAllocatorSettings=self.calls.append)
        else:
            self.cuda = mock.Mock()
            self.cuda.memory._set_allocator_settings = self.calls.append

    def empty(self, n, dtype=None, device=None):
        assert n == 1 << 20 and dtype == "uint8" and device == "cuda:0"
        if self.fail is not None:
            raise self.fail
        return object()


class AllocatorMode(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        self.messages = []

    def test_driver_without_virtual_memory_keeps_the_standard_allocator(self):
        torch = _FakeTorch()
        mode = alloc.configure_cuda_allocator(torch, self.messages.append, supports=lambda i: False)
        self.assertEqual(mode, "standard")
        self.assertEqual(torch.calls, ["expandable_segments:False"])
        self.assertIn("standard allocator", self.messages[0])
        self.assertIn("vGPU", self.messages[0])

    def test_supported_driver_enables_expandable_segments(self):
        torch = _FakeTorch()
        mode = alloc.configure_cuda_allocator(torch, self.messages.append, supports=lambda i: True)
        self.assertEqual(mode, "expandable")
        self.assertEqual(torch.calls, ["expandable_segments:True"])
        self.assertEqual(self.messages, [])

    def test_unknown_driver_relies_on_the_probe(self):
        torch = _FakeTorch(fail=RuntimeError("CUDA driver error: operation not supported"))
        mode = alloc.configure_cuda_allocator(torch, self.messages.append, supports=lambda i: None)
        self.assertEqual(mode, "standard")
        self.assertEqual(torch.calls, ["expandable_segments:True", "expandable_segments:False"])
        self.assertIn("operation not supported", self.messages[0])

    def test_other_errors_are_not_swallowed(self):
        torch = _FakeTorch(fail=RuntimeError("CUDA out of memory"))
        with self.assertRaisesRegex(RuntimeError, "out of memory"):
            alloc.configure_cuda_allocator(torch, self.messages.append, supports=lambda i: True)

    def test_user_setting_is_respected(self):
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"
        torch = _FakeTorch()
        mode = alloc.configure_cuda_allocator(torch, self.messages.append, supports=lambda i: True)
        self.assertEqual(mode, "unchanged")
        self.assertEqual(torch.calls, [])

    def test_older_torch_setter_is_used(self):
        torch = _FakeTorch(new_api=False)
        mode = alloc.configure_cuda_allocator(torch, self.messages.append, supports=lambda i: True)
        self.assertEqual(mode, "expandable")
        self.assertEqual(torch.calls, ["expandable_segments:True"])

    def test_driver_query_never_raises(self):
        # Whatever the machine, the query answers True, False or None.
        self.assertIn(alloc.driver_supports_virtual_memory(0), (True, False, None))
        self.assertIsNone(alloc.driver_supports_virtual_memory(10 ** 6) or None)


if __name__ == "__main__":
    unittest.main()
