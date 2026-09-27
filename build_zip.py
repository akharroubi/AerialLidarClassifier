"""Build the plugins.qgis.org ZIP from this checkout (flat repository layout).

    python build_zip.py            -> dist/aerial_lidar_classifier_v<version>.zip

The archive holds one top-level folder, ``Aerial_LiDAR_Classifier``: the
package name of the published plugin (plugins.qgis.org/plugins/
Aerial_LiDAR_Classifier). Any other folder name would be a different
plugin for QGIS and for the repository, so it must never change.

Only an explicit list of runtime files is packed (no tests, reports,
caches, weights or stray files). Timestamps are fixed and text files
use LF, so the same source always gives the same bytes; a SHA-256
sidecar is written next to the ZIP.
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
    findings = json.loads(result.stdout or "{}").get("results", [])
    if findings:
        listed = [f"{f['test_id']} {Path(f['filename']).relative_to(ROOT)}:{f['line_number']}"
                  for f in findings]
        raise SystemExit("Bandit findings would send the upload to manual review:\n  "
                         + "\n  ".join(listed))
    if (ROOT / ".bandit").exists():
        raise SystemExit("Remove .bandit: the scan must pass without a config file.")


def build(destination: Path) -> str:
    files = collect()
    check_secrets_scanner(files)
    check_bandit()
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
    default = ROOT / "dist" / f"aerial_lidar_classifier_v{read_version()}.zip"
    build(Path(args.output) if args.output else default)
