"""One classification at a time in a QGIS session.

The dock and Processing (the toolbox, the Graphical Modeler, scripts) can
start runs at the same time. Two runs would share the GPU's memory and its
allocator setting (process-wide in PyTorch), and each run plans its RAM use
as if it were alone. So a run waits for the previous one to finish.

No QGIS or torch imports (unit-testable anywhere).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Optional

_RUN_LOCK = threading.Lock()


@contextmanager
def exclusive_run(cancel_callback: Optional[Callable[[], bool]] = None,
                  log: Optional[Callable[[str], None]] = None, poll_s: float = 0.5):
    """Hold the plugin's run lock for the duration of the block.

    Waits while another run holds it, checking ``cancel_callback`` every
    ``poll_s`` seconds (InterruptedError when it returns True); says once
    through ``log`` that the run is waiting."""
    if not _RUN_LOCK.acquire(blocking=False):
        if log is not None:
            log("Another classification is running in this QGIS session; this one "
                "starts when it finishes.")
        while not _RUN_LOCK.acquire(timeout=poll_s):
            if cancel_callback is not None and cancel_callback():
                raise InterruptedError()
    try:
        yield
    finally:
        _RUN_LOCK.release()
