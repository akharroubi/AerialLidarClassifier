"""Every plugin file must compile on the oldest Python a supported QGIS ships.

``metadata.txt`` allows QGIS 3.34, whose official macOS build runs Python
3.9 and whose Ubuntu 22.04 / Debian 12 packages run 3.10 / 3.11. v1.0.2
shipped single-quoted f-strings with a line break inside ``{...}``, which
is Python 3.12+ syntax (PEP 701), so those builds could not open the dock
or the Setup panel at all.

Run with any Python::

    python tests/test_syntax_compat.py

On Python < 3.12 the sources are simply compiled. On 3.12+ (where PEP 701
strings parse fine) each file is tokenized and any single-quoted f-string
spanning several lines is reported. ``ast.parse(feature_version=(3, 9))``
does not catch this construct, which is why the tokenizer is used.
"""

import io
import sys
import tokenize
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _plugin_sources():
    return sorted(p for p in PLUGIN_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _multiline_fstrings(source: str):
    """Return (start_line, end_line) of single-quoted f-strings that span lines."""
    if not hasattr(tokenize, "FSTRING_START"):
        return []  # < 3.12: such strings are already a SyntaxError
    found = []
    stack = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.FSTRING_START:
            stack.append(tok)
        elif tok.type == tokenize.FSTRING_END:
            start = stack.pop()
            triple = start.string.endswith(('"""', "'''"))
            if not triple and tok.end[0] > start.start[0]:
                found.append((start.start[0], tok.end[0]))
    return found


def test_every_file_compiles_on_this_python():
    failures = []
    for path in _plugin_sources():
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(PLUGIN_ROOT)}:{exc.lineno}: {exc.msg}")
    assert not failures, "\n".join(failures)


def test_no_python312_only_fstrings():
    offenders = []
    for path in _plugin_sources():
        for start, end in _multiline_fstrings(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(PLUGIN_ROOT)}:{start}-{end}")
    assert not offenders, (
        "single-quoted f-strings spanning lines (Python 3.12+ only):\n"
        + "\n".join(offenders)
    )


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}:\n{exc}")
    print(f"(checked with Python {sys.version.split()[0]})")
    raise SystemExit(1 if failures else 0)
