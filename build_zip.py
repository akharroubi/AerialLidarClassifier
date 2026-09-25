"""Build a deterministic, single-folder QGIS plugin ZIP from this checkout."""
import argparse
import hashlib
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
PACKAGE = 'AerialLidarClassifier'
DIRS = ('core', 'dialogs', 'gui', 'processing', 'utils', 'widgets', 'workers', 'assets', 'i18n')
FILES = ('__init__.py', 'plugin.py', 'config.py', 'metadata.txt', 'icon.png',
         'LICENSE', 'README.md', 'CHANGELOG.md', 'THIRD_PARTY_NOTICES.md')


def build(destination):
    files = [ROOT / name for name in FILES]
    for directory in DIRS:
        files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix not in ('.pyc', '.pyo', '.pth', '.pt'))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(files):
            name = f'{PACKAGE}/{path.relative_to(ROOT).as_posix()}'
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 25, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            data = path.read_bytes()
            if path.suffix.lower() in ('.py', '.md', '.txt', '.json', '.svg', '.ts', '.ui') or path.name.startswith('LICENSE'):
                data = data.replace(b'\r\n', b'\n')
            archive.writestr(info, data)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + '.sha256').write_text(
        f'{digest}  {destination.name}\n', encoding='utf-8')
    print(f'{destination.resolve()}\n{len(files)} files\nSHA-256 {digest}')
    return digest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', nargs='?', default=str(ROOT / 'dist/AerialLidarClassifier-1.1.0-rc1.zip'))
    build(parser.parse_args().output)
