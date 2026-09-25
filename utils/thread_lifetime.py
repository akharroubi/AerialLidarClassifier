"""Keep background QThreads alive even when their initiating widget closes."""
_running = set()


def retain_thread(worker):
    _running.add(worker)
    worker.finished.connect(lambda: _running.discard(worker))
    return worker
