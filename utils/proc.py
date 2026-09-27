"""Run external programs through Qt's QProcess.

The dependency installer starts the plugin's own portable Python, uv and
nvidia-smi. QProcess takes the program and an argument list (no shell is
ever involved) and, in a GUI application such as QGIS, starts console
programs without opening a console window.

``run`` mirrors the parts of ``subprocess.run`` the installer uses
(captured text output, timeout, environment, working directory), and
``Background`` covers the long pip/uv installs whose output goes to files
that the installer reads for progress.
"""
import locale
from typing import Dict, List, Optional, Sequence

from qgis.PyQt.QtCore import QProcess, QProcessEnvironment

from .compat import scoped_enum

_NOT_RUNNING = scoped_enum(QProcess, "ProcessState", "NotRunning")
_NORMAL_EXIT = scoped_enum(QProcess, "ExitStatus", "NormalExit")
_START_TIMEOUT_MS = 30000


class TimeoutExpired(Exception):
    """The program did not finish within its time limit (it was stopped)."""

    def __init__(self, cmd: Sequence[str], timeout: float):
        name = str(cmd[0]) if cmd else "program"
        super().__init__(f"{name} did not finish within {timeout:g} s")
        self.cmd = list(cmd)
        self.timeout = timeout


class CompletedProcess:
    """Result of ``run``: same attribute names as subprocess.CompletedProcess."""

    def __init__(self, args: List[str], returncode: int, stdout: str, stderr: str):
        self.args = args
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _decode(data) -> str:
    raw = bytes(data)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode(locale.getpreferredencoding(False) or "latin-1", errors="replace")


def _prepare(cmd: Sequence[str], env: Optional[Dict[str, str]], cwd: Optional[str]) -> QProcess:
    if not cmd:
        raise ValueError("empty command")
    process = QProcess()
    if env is not None:
        qenv = QProcessEnvironment()
        for key, value in env.items():
            qenv.insert(str(key), str(value))
        process.setProcessEnvironment(qenv)
    if cwd:
        process.setWorkingDirectory(str(cwd))
    process.setProgram(str(cmd[0]))
    process.setArguments([str(arg) for arg in cmd[1:]])
    return process


def _start(process: QProcess, cmd: Sequence[str]) -> None:
    process.start()
    if not process.waitForStarted(_START_TIMEOUT_MS):
        reason = process.errorString()
        process.kill()
        raise FileNotFoundError(f"Could not start {cmd[0]}: {reason}")


def _returncode(process: QProcess) -> int:
    return process.exitCode() if process.exitStatus() == _NORMAL_EXIT else -1


def _running(process: QProcess) -> bool:
    return process.state() != _NOT_RUNNING


def _stop(process: QProcess) -> None:
    """Ask the program to stop, then force it after 10 s."""
    if not _running(process):
        return
    process.terminate()
    if not process.waitForFinished(10000):
        process.kill()
        process.waitForFinished(5000)


def run(
    cmd: Sequence[str],
    timeout: Optional[float] = None,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[str] = None,
) -> CompletedProcess:
    """Run ``cmd`` to completion and return its exit code and text output.

    Raises FileNotFoundError when the program cannot be started and
    TimeoutExpired (after stopping it) when it runs longer than ``timeout``.
    """
    process = _prepare(cmd, env, cwd)
    _start(process, cmd)
    msecs = -1 if timeout is None else max(1, int(timeout * 1000))
    if not process.waitForFinished(msecs) and _running(process):
        _stop(process)
        raise TimeoutExpired(cmd, timeout or 0)
    return CompletedProcess(
        [str(arg) for arg in cmd],
        _returncode(process),
        _decode(process.readAllStandardOutput()),
        _decode(process.readAllStandardError()),
    )


class Background:
    """A long-running program whose output is written to two files."""

    def __init__(
        self,
        cmd: Sequence[str],
        stdout_path: str,
        stderr_path: str,
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        self.cmd = list(cmd)
        self._process = _prepare(cmd, env, cwd)
        self._process.setStandardOutputFile(stdout_path)
        self._process.setStandardErrorFile(stderr_path)
        _start(self._process, cmd)

    def wait(self, timeout: float) -> bool:
        """Wait up to ``timeout`` seconds; True once the program has finished."""
        finished = self._process.waitForFinished(max(1, int(timeout * 1000)))
        return finished or not _running(self._process)

    def running(self) -> bool:
        return _running(self._process)

    def stop(self) -> None:
        _stop(self._process)

    @property
    def returncode(self) -> Optional[int]:
        return None if _running(self._process) else _returncode(self._process)
