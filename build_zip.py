"""Build the plugins.qgis.org ZIP from this checkout (flat repository layout).

    python build_zip.py            -> dist/lidar_ai_classifier_v<version>.zip

The archive holds one top-level folder, ``Aerial_LiDAR_Classifier``: the
package name of the published plugin (plugins.qgis.org/plugins/
Aerial_LiDAR_Classifier). Any other folder name would be a different
plugin for QGIS and for the repository, so it must never change.

Only an explicit list of runtime files is packed (no tests, reports,
caches, weights or stray files): model weights download on first use
from the GitHub releases listed in core/registry.py. Timestamps are fixed
and text files use LF, so the same source always gives the same bytes; a
SHA-256 sidecar is written next to the ZIP.

The build refuses to produce an archive that plugins.qgis.org would
reject or hold for manual review: larger than 25 MB, containing binaries
or weights, failing Bandit, Flake8's blocking checks, or the secrets scan.
"""
import argparse
import re
import sys
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PACKAGE = "Aerial_LiDAR_Classifier"
DIRS = ("core", "dialogs", "gui", "processing", "utils", "widgets",
        "workers", "assets", "i18n")
FILES = ("__init__.py", "plugin.py", "config.py", "metadata.txt", "icon.png",
         "LICENSE", "README.md", "CHANGELOG.md", "THIRD_PARTY_NOTICES.md")
SKIP_SUFFIXES = (".pyc", ".pyo", ".pth", ".pt", ".log", ".zip", ".bak")
TEXT_SUFFIXES = (".py", ".md", ".txt", ".json", ".svg", ".ts", ".ui", ".cfg")
FIXED_TIME = (2026, 9, 25, 0, 0, 0)
# plugins.qgis.org: "The size of the plugin package should not exceed 25MB"
# and "Don't include binaries".
MAX_ZIP_BYTES = 25 * 1000 * 1000
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".onnx", ".npy", ".npz", ".exe", ".dll",
                      ".so", ".dylib", ".pyd", ".sh", ".bat", ".cmd", ".las", ".laz")


def read_version() -> str:
    for line in (ROOT / "metadata.txt").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("version="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("metadata.txt has no version=")


def collect():
    files = [ROOT / name for name in FILES]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise SystemExit(f"Missing files: {missing}")
    for directory in DIRS:
        files.extend(
            p for p in (ROOT / directory).rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
            and p.suffix.lower() not in SKIP_SUFFIXES
        )
    return sorted(files)


_HEX_RUN = re.compile(rb"[0-9a-fA-F]{32,}")


def check_secrets_scanner(files) -> None:
    """Refuse to build when plugins.qgis.org's scanner would block the ZIP.

    Its secrets check flags any run of 32 or more hex characters (a
    SHA-256 checksum or a git commit id counts) unless the same line
    carries ``pragma: allowlist secret``. Version 1.1.0 was blocked on
    upload for checksums left in README.md and THIRD_PARTY_NOTICES.md.
    """
    problems = []
    for path in files:
        if path.suffix.lower() in (".png", ".jpg", ".ico"):
            continue
        for number, line in enumerate(path.read_bytes().splitlines(), 1):
            if _HEX_RUN.search(line) and b"pragma: allowlist secret" not in line:
                problems.append(f"{path.relative_to(ROOT)}:{number}")
    if problems:
        raise SystemExit(
            "Long hex strings would block the upload (shorten them or add "
            "'# pragma: allowlist secret' on the line):\n  " + "\n  ".join(problems))


def check_bandit() -> None:
    """Refuse to build unless Bandit reports nothing, with no config file.

    plugins.qgis.org runs Bandit on every upload. A clean scan without a
    developer-supplied config (.bandit) keeps the version "Validated"
    rather than "Validated (configured)", which needs an admin review.
    """
    import json
    import subprocess  # nosec B404 - build tool only, never shipped
    targets = [str(ROOT / name) for name in DIRS if (ROOT / name).is_dir()]
    targets += [str(ROOT / name) for name in FILES if name.endswith(".py")]
    try:
        result = subprocess.run(  # nosec B603 - fixed argument list
            [sys.executable, "-m", "bandit", "-q", "-f", "json", "-r", *targets],
            capture_output=True, text=True, check=False)
    except OSError as exc:
        raise SystemExit(f"Could not run Bandit: {exc}")
    if "No module named bandit" in result.stderr:
        raise SystemExit("Bandit is required to build: python -m pip install bandit")
    if result.returncode not in (0, 1):
        raise SystemExit(f"Bandit failed to run: {result.stderr}")
    scan = json.loads(result.stdout or "{}")
    if "results" not in scan or scan.get("errors"):
        raise SystemExit("Bandit did not complete a full source scan: " + result.stderr)
    findings = scan["results"]
    if findings:
        listed = [f"{f['test_id']} {Path(f['filename']).relative_to(ROOT)}:{f['line_number']}"
                  for f in findings]
        raise SystemExit("Bandit findings would send the upload to manual review:\n  "
                         + "\n  ".join(listed))
    if (ROOT / ".bandit").exists():
        raise SystemExit("Remove .bandit: the scan must pass without a config file.")


def check_no_binaries(files) -> None:
    """Weights, data and executables never go into the plugin ZIP."""
    found = [str(p.relative_to(ROOT)) for p in files if p.suffix.lower() in FORBIDDEN_SUFFIXES]
    if found:
        raise SystemExit("Binary or data files must not be packed:\n  " + "\n  ".join(found))


def check_flake8_blockers() -> None:
    """Refuse the Flake8 errors plugins.qgis.org treats as blocking.

    E9 (syntax / IO errors), F821 (undefined name), F823 (local variable
    referenced before assignment) and F831 (duplicate argument name).
    """
    import subprocess  # nosec B404 - build tool only, never shipped
    targets = [str(ROOT / name) for name in DIRS if (ROOT / name).is_dir()]
    targets += [str(ROOT / name) for name in FILES if name.endswith(".py")]
    result = subprocess.run(  # nosec B603 - fixed argument list
        [sys.executable, "-m", "flake8", "--isolated", "--select=E9,F821,F823,F831", *targets],
        capture_output=True, text=True, check=False)
    if "No module named flake8" in result.stderr:
        raise SystemExit("Flake8 is required to build: python -m pip install flake8")
    if result.returncode != 0:
        raise SystemExit("Flake8 blocking errors:\n" + (result.stdout or result.stderr))


def build(destination: Path) -> str:
    files = collect()
    check_no_binaries(files)
    check_secrets_scanner(files)
    check_bandit()
    check_flake8_blockers()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            name = f"{PACKAGE}/{path.relative_to(ROOT).as_posix()}"
            info = zipfile.ZipInfo(name, date_time=FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            data = path.read_bytes()
            if path.suffix.lower() in TEXT_SUFFIXES or path.name.startswith("LICENSE"):
                data = data.replace(b"\r\n", b"\n")
            archive.writestr(info, data)
    if destination.stat().st_size > MAX_ZIP_BYTES:
        size_mb = destination.stat().st_size / 1e6
        destination.unlink()
        raise SystemExit(f"The ZIP is {size_mb:.1f} MB; plugins.qgis.org accepts at most 25 MB.")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_name(destination.name + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="utf-8")
    size_kb = destination.stat().st_size / 1024
    print(f"{destination}\n{len(files)} files, {size_kb:.0f} KB\nSHA-256 {digest}")
    return digest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", nargs="?", default=None)
    args = parser.parse_args()
    default = ROOT / "dist" / f"lidar_ai_classifier_v{read_version()}.zip"
    build(Path(args.output) if args.output else default)
