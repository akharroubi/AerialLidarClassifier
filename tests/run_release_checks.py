"""Run every automated release check and print one summary table.

    python tests/run_release_checks.py [--skip-gpu] [--qgis3 BAT] [--qgis4 BAT]

1. pytest suite (unit + backend tests, real CUDA inference when available)
   with the plugin's own environment (torch, spconv, laspy).
2. The QGIS integration scripts (qgis_*_checks.py) under QGIS 3 and QGIS 4,
   with real Qt, Processing, LAS/LAZ readers and Untwine.
3. Optionally the real GPU end-to-end check (qgis_mls_gpu.py) under each QGIS.

Defaults match the maintainer's Windows test machine; override the paths
with the options or the ALC_QGIS3_PYTHON / ALC_QGIS4_PYTHON / ALC_VENV_PYTHON
environment variables. A QGIS whose launcher is missing is reported as
"not found", never as passed. Exit status is non-zero if anything failed.
"""
import argparse
import os
import subprocess  # nosec B404 - developer tool, not shipped in the plugin ZIP
import sys
import time
from pathlib import Path

TESTS = Path(__file__).resolve().parent
QGIS_SCRIPTS = ("qgis_release_checks.py", "qgis_mls_checks.py",
                "qgis_gpu_readiness_checks.py", "qgis_pointcloud_view_checks.py")
DEFAULTS = {
    "venv": os.environ.get("ALC_VENV_PYTHON", str(
        Path.home() / ".qgis_aerial_lidar_classifier/venv_py3.12/Scripts/python.exe")),
    "qgis3": os.environ.get("ALC_QGIS3_PYTHON",
                            r"C:\Program Files\QGIS 3.44.10\bin\python-qgis-ltr.bat"),
    "qgis4": os.environ.get("ALC_QGIS4_PYTHON",
                            r"C:\Users\user\qgis4_test\app\bin\python-qgis.bat"),
}


def run(label, command, log_dir):
    log = log_dir / (label.replace(" ", "_").replace("/", "_") + ".log")
    start = time.monotonic()
    with log.open("w", encoding="utf-8", errors="replace") as stream:
        if command[0].lower().endswith(".bat"):
            # OSGeo4W launchers are batch files: run them through cmd.exe.
            command = ["cmd", "/c", *command]
        code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,  # nosec B603
                              cwd=str(TESTS.parent), check=False).returncode
    text = log.read_text(encoding="utf-8", errors="replace")
    summary = next((line for line in reversed(text.splitlines())
                    if line.startswith(("Ran ", "OK", "FAILED")) or " passed" in line
                    or "PASS real" in line), "")
    if code == 0 and "Ran " in text:
        summary = next(line for line in text.splitlines() if line.startswith("Ran ")) + ", OK"
    return code, f"{time.monotonic() - start:6.1f}s", summary.strip(), log


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--venv", default=DEFAULTS["venv"])
    parser.add_argument("--qgis3", default=DEFAULTS["qgis3"])
    parser.add_argument("--qgis4", default=DEFAULTS["qgis4"])
    parser.add_argument("--skip-gpu", action="store_true",
                        help="skip the real CUDA end-to-end QGIS run")
    parser.add_argument("--logs", default=str(TESTS.parent.parent / "test_logs"))
    args = parser.parse_args()
    log_dir = Path(args.logs)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    if Path(args.venv).is_file():
        rows.append(("pytest", *run("pytest", [args.venv, "-m", "pytest", str(TESTS), "-q",
                                                "--rootdir", str(TESTS), "--confcutdir", str(TESTS),
                                                "-p", "no:cacheprovider"], log_dir)))
    else:
        rows.append(("pytest", 1, "", f"plugin venv Python not found: {args.venv}", None))
    for name, launcher in (("QGIS 3", args.qgis3), ("QGIS 4", args.qgis4)):
        if not Path(launcher).is_file():
            rows.append((name, 1, "", f"launcher not found: {launcher}", None))
            continue
        scripts = list(QGIS_SCRIPTS) + ([] if args.skip_gpu else ["qgis_mls_gpu.py"])
        for script in scripts:
            label = f"{name} {script[:-3]}"
            rows.append((label, *run(label, [launcher, str(TESTS / script)], log_dir)))

    failed = 0
    print(f"\n{'check':45} {'result':6} {'time':>8}  summary")
    for label, code, elapsed, summary, _log in rows:
        failed += code != 0
        print(f"{label:45} {'PASS' if code == 0 else 'FAIL':6} {elapsed:>8}  {summary}")
    print(f"\nLogs: {log_dir}")
    print("ALL PASSED" if not failed else f"{failed} CHECK(S) FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
