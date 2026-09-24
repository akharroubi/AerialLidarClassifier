"""Import plugin modules outside QGIS for unit tests.

The package ``__init__`` imports ``qgis.PyQt``, so the real package
cannot be imported in a plain Python. This helper registers lightweight
stand-in packages for ``Aerial_LiDAR_Classifier`` (and the sub-packages
on the way) without running their ``__init__`` files, then loads the
requested module under its real dotted name so its relative imports
(``from ..config import ...``) resolve to the real files.

Only modules that do not import qgis themselves can be loaded this way:
``config``, ``core.tiling``, ``core.classifier_core``.
"""

import importlib.util
import sys
import types
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
PACKAGE = PLUGIN_ROOT.name  # "Aerial_LiDAR_Classifier"


def _ensure_package(dotted: str, path: Path) -> None:
    if dotted in sys.modules:
        return
    module = types.ModuleType(dotted)
    module.__path__ = [str(path)]
    module.__package__ = dotted
    sys.modules[dotted] = module


def load_plugin_module(relative_name: str):
    """Return the plugin module ``relative_name`` (e.g. ``"core.tiling"``)."""
    parts = relative_name.split(".")
    _ensure_package(PACKAGE, PLUGIN_ROOT)
    dotted = PACKAGE
    path = PLUGIN_ROOT
    for part in parts[:-1]:
        dotted = f"{dotted}.{part}"
        path = path / part
        _ensure_package(dotted, path)

    full_name = f"{dotted}.{parts[-1]}"
    if full_name in sys.modules:
        return sys.modules[full_name]
    file = path / f"{parts[-1]}.py"
    spec = importlib.util.spec_from_file_location(full_name, file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module
