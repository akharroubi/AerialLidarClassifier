"""Linear units of a LAS file and the factors that bring XYZ to metres.

The model was trained on metric data (30 cm voxels, 33.6 m blocks). Most
US LiDAR is delivered in US survey feet; fed as-is, every distance is
3.28 times too small for the model, buildings come out as wires and
towers (GitHub issue #5) and there are about ten times more voxel
blocks to run (issue #2). The plugin therefore reads the units from the
LAS header (WKT CRS or GeoTIFF keys), converts the coordinates it hands
to the model, and says what it did in the log. When the header carries
no CRS at all it assumes metres and says that too; the user can force
the units from the dock or the Processing algorithm.

No laspy, QGIS or pyproj is required. QGIS is used only as an optional
EPSG lookup when the header gives a CRS code but no unit key, and only
when it is importable (the plugin always runs inside QGIS; the unit
tests do not).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

METRE = 1.0
INTERNATIONAL_FOOT = 0.3048
US_SURVEY_FOOT = 1200.0 / 3937.0  # 0.30480060960121924

# EPSG unit-of-measure codes as used by the GeoTIFF keys
# ProjLinearUnitsGeoKey (3076) and VerticalUnitsGeoKey (4099).
_EPSG_LINEAR_UNIT_FACTORS = {
    9001: METRE,
    9002: INTERNATIONAL_FOOT,
    9003: US_SURVEY_FOOT,
    9036: 1000.0,  # kilometre
}

# Override names accepted from the dock and the Processing algorithm,
# in display order. "auto" means: read the header.
UNIT_OVERRIDES = [
    ("auto", "Auto-detect from the file's CRS"),
    ("metre", "Metres"),
    ("foot", "International feet"),
    ("us_foot", "US survey feet"),
]
_OVERRIDE_FACTORS = {
    "metre": METRE,
    "foot": INTERNATIONAL_FOOT,
    "us_foot": US_SURVEY_FOOT,
}


def unit_label(factor: float) -> str:
    """Human-readable name of a unit given its length in metres."""
    if abs(factor - METRE) < 1e-9:
        return "metre"
    if abs(factor - US_SURVEY_FOOT) < 1e-9:
        return "US survey foot"
    if abs(factor - INTERNATIONAL_FOOT) < 1e-9:
        return "foot"
    if abs(factor - 1000.0) < 1e-9:
        return "kilometre"
    return f"{factor:g} m unit"


@dataclass
class LinearUnits:
    """Factors that turn the file's XY and Z into metres, and where they came from."""

    xy_to_m: float
    z_to_m: float
    source: str
    detected: bool = True   # False: the header said nothing, metres assumed
    angular: bool = False   # geographic CRS (degrees): no linear factor

    @property
    def is_metric(self) -> bool:
        return (
            not self.angular
            and abs(self.xy_to_m - 1.0) < 1e-9
            and abs(self.z_to_m - 1.0) < 1e-9
        )

    def describe(self) -> str:
        if self.angular:
            return f"geographic coordinates in degrees ({self.source})"
        if self.is_metric:
            return f"metres ({self.source})"
        return (
            f"XY in {unit_label(self.xy_to_m)}, Z in {unit_label(self.z_to_m)}"
            f" ({self.source}); converting to metres for the model"
        )

    def apply(self, pcd: np.ndarray) -> np.ndarray:
        """Return ``pcd`` (N, 3) scaled to metres (a copy when scaling)."""
        out = np.asarray(pcd, dtype=np.float64)
        if self.is_metric:
            return out
        out = out.copy()
        out[:, 0] *= self.xy_to_m
        out[:, 1] *= self.xy_to_m
        out[:, 2] *= self.z_to_m
        return out


# ---------------------------------------------------------------------------
# WKT (1 and 2)
# ---------------------------------------------------------------------------

class _Node:
    __slots__ = ("keyword", "args")

    def __init__(self, keyword: str, args: list):
        self.keyword = keyword
        self.args = args


def _parse_wkt(text: str) -> Optional[_Node]:
    """Parse a WKT string into a tree of ``KEYWORD[args...]`` nodes.

    Arguments are quoted strings (str), numbers (float), bare tokens
    such as ``east`` or ``Cartesian`` (str) or nested nodes. Both
    ``[]`` and ``()`` brackets are accepted. Returns None on malformed
    input instead of raising.
    """
    n = len(text)
    pos = 0

    def skip_ws() -> None:
        nonlocal pos
        while pos < n and text[pos] in " \t\r\n":
            pos += 1

    def parse_value():
        nonlocal pos
        skip_ws()
        if pos >= n:
            raise ValueError("unexpected end of WKT")
        if text[pos] == '"':
            pos += 1
            buf = []
            while pos < n:
                ch = text[pos]
                if ch == '"':
                    if pos + 1 < n and text[pos + 1] == '"':
                        buf.append('"')
                        pos += 2
                        continue
                    break
                buf.append(ch)
                pos += 1
            pos += 1  # closing quote
            return "".join(buf)
        start = pos
        while pos < n and text[pos] not in "[](),":
            pos += 1
        token = text[start:pos].strip()
        skip_ws()
        if pos < n and text[pos] in "[(":
            close = "]" if text[pos] == "[" else ")"
            pos += 1
            args = []
            while True:
                skip_ws()
                if pos < n and text[pos] == close:
                    pos += 1
                    break
                args.append(parse_value())
                skip_ws()
                if pos < n and text[pos] == ",":
                    pos += 1
                    continue
                if pos < n and text[pos] == close:
                    pos += 1
                    break
                raise ValueError("malformed WKT")
            return _Node(token.upper(), args)
        try:
            return float(token)
        except ValueError:
            return token

    try:
        node = parse_value()
    except (ValueError, IndexError, RecursionError):
        return None
    return node if isinstance(node, _Node) else None


_LINEAR_UNIT_KEYWORDS = {"UNIT", "LENGTHUNIT"}
# Subtrees whose UNIT / LENGTHUNIT nodes are not the coordinate units:
# the geographic base (degrees), the projection parameters (a false
# easting may be given in metres for a feet CRS), datums and metadata.
_SKIP_SUBTREES = {
    "GEOGCS", "GEOGCRS", "GEODCRS", "GEODETICCRS", "BASEGEOGCRS",
    "BASEGEODCRS", "CONVERSION", "PROJECTION", "PARAMETER", "DATUM",
    "VERT_DATUM", "VDATUM", "TOWGS84", "PRIMEM", "SPHEROID", "ELLIPSOID",
    "ID", "AUTHORITY", "REMARK", "USAGE", "BBOX", "AREA", "SCOPE",
    "EXTENSION", "DERIVINGCONVERSION",
}
_PROJECTED = {"PROJCS", "PROJCRS", "PROJECTEDCRS"}
_LOCAL = {"LOCAL_CS", "ENGCRS", "ENGINEERINGCRS"}
_VERTICAL = {"VERT_CS", "VERTCRS", "VERTICALCRS"}
_GEOGRAPHIC = {"GEOGCS", "GEOGCRS", "GEODCRS", "GEODETICCRS"}


def _factor_from_name(name: str) -> Optional[float]:
    n = name.lower()
    if "foot" in n or "feet" in n or n in ("ft", "ftus", "us-ft", "us_ft"):
        if "us" in n or "survey" in n:
            return US_SURVEY_FOOT
        return INTERNATIONAL_FOOT
    if "met" in n:
        return METRE
    if "kilomet" in n:
        return 1000.0
    return None


def _unit_node_factor(unit: _Node) -> Optional[float]:
    for arg in unit.args[1:]:
        if isinstance(arg, float):
            return arg
    if unit.args and isinstance(unit.args[0], str):
        return _factor_from_name(unit.args[0])
    return None


def _find_unit_factor(node: _Node) -> Optional[float]:
    """First linear unit under ``node``, skipping subtrees with other units."""
    for arg in node.args:
        if not isinstance(arg, _Node):
            continue
        if arg.keyword in _LINEAR_UNIT_KEYWORDS:
            factor = _unit_node_factor(arg)
            if factor is not None:
                return factor
        elif arg.keyword not in _SKIP_SUBTREES:
            factor = _find_unit_factor(arg)
            if factor is not None:
                return factor
    return None


def _find_node(node: _Node, keywords: set) -> Optional[_Node]:
    if node.keyword in keywords:
        return node
    for arg in node.args:
        if isinstance(arg, _Node):
            found = _find_node(arg, keywords)
            if found is not None:
                return found
    return None


def units_from_wkt(wkt: str) -> Optional[LinearUnits]:
    """Read the horizontal and vertical linear units from a WKT CRS string.

    Returns None when the string cannot be parsed or carries no linear
    unit; an ``angular`` result for a geographic (degrees) CRS.
    """
    root = _parse_wkt(wkt or "")
    if root is None:
        return None
    horizontal = _find_node(root, _PROJECTED) or _find_node(root, _LOCAL)
    if horizontal is None:
        if _find_node(root, _GEOGRAPHIC) is not None:
            return LinearUnits(1.0, 1.0, "WKT CRS is geographic", angular=True)
        return None
    xy = _find_unit_factor(horizontal)
    if xy is None:
        return None
    vertical = _find_node(root, _VERTICAL)
    z = _find_unit_factor(vertical) if vertical is not None else None
    if z is None:
        return LinearUnits(
            xy, xy,
            f"WKT CRS: {unit_label(xy)}; no vertical CRS, Z assumed in the "
            "same unit",
        )
    return LinearUnits(
        xy, z,
        f"WKT CRS: {unit_label(xy)} horizontal, {unit_label(z)} vertical",
    )


# ---------------------------------------------------------------------------
# GeoTIFF keys (LAS 1.0 to 1.3, and LAS 1.4 files without a WKT record)
# ---------------------------------------------------------------------------

def _epsg_crs_unit_factor(code: int) -> Optional[float]:
    """Length of the CRS unit in metres via QGIS, or None outside QGIS."""
    try:
        from qgis.core import Qgis, QgsCoordinateReferenceSystem, QgsUnitTypes
    except Exception:
        return None
    try:
        crs = QgsCoordinateReferenceSystem(f"EPSG:{int(code)}")
        if not crs.isValid() or crs.isGeographic():
            return None
        unit = crs.mapUnits()
        try:
            metres = Qgis.DistanceUnit.Meters
        except AttributeError:
            metres = QgsUnitTypes.DistanceMeters
        factor = float(QgsUnitTypes.fromUnitToUnitFactor(unit, metres))
        return factor if factor > 0 else None
    except Exception:
        return None


def units_from_geokeys(keys: dict) -> Optional[LinearUnits]:
    """Read the units from GeoTIFF SHORT keys (``{key id: value}``).

    Uses ProjLinearUnitsGeoKey (3076) and VerticalUnitsGeoKey (4099);
    falls back to the EPSG CRS codes (3072, 4096) through QGIS.
    """
    if not keys:
        return None
    model_type = keys.get(1024)  # 1 projected, 2 geographic, 3 geocentric
    projected_code = keys.get(3072)
    if model_type == 2 and projected_code in (None, 0, 32767):
        return LinearUnits(1.0, 1.0, "GeoTIFF keys: geographic CRS", angular=True)

    xy = None
    parts = []
    if keys.get(3076) in _EPSG_LINEAR_UNIT_FACTORS:
        xy = _EPSG_LINEAR_UNIT_FACTORS[keys[3076]]
        parts.append(f"ProjLinearUnitsGeoKey {keys[3076]} ({unit_label(xy)})")
    elif projected_code not in (None, 0, 32767):
        xy = _epsg_crs_unit_factor(projected_code)
        if xy is not None:
            parts.append(f"EPSG:{projected_code} ({unit_label(xy)})")
    if xy is None:
        return None

    z = None
    if keys.get(4099) in _EPSG_LINEAR_UNIT_FACTORS:
        z = _EPSG_LINEAR_UNIT_FACTORS[keys[4099]]
        parts.append(f"VerticalUnitsGeoKey {keys[4099]} ({unit_label(z)})")
    vertical_code = keys.get(4096)
    if z is None and vertical_code not in (None, 0, 32767):
        z = _epsg_crs_unit_factor(vertical_code)
        if z is None:
            raise ValueError(f"Cannot resolve units of vertical EPSG:{vertical_code}; select explicit input units.")
        parts.append(f"vertical EPSG:{vertical_code} ({unit_label(z)})")
    if z is None:
        z = xy
        parts.append("no vertical CRS/unit key, Z assumed in the same unit")
    return LinearUnits(xy, z, "GeoTIFF keys: " + ", ".join(parts))


# ---------------------------------------------------------------------------
# LAS header access (laspy objects, duck-typed so tests can stub them)
# ---------------------------------------------------------------------------

def _header_vlrs(header) -> list:
    vlrs = []
    for attr in ("vlrs", "evlrs"):
        try:
            vlrs.extend(list(getattr(header, attr, None) or []))
        except Exception:
            pass
    return vlrs


def _header_wkt(header) -> Optional[str]:
    for vlr in _header_vlrs(header):
        is_wkt = type(vlr).__name__ == "WktCoordinateSystemVlr" or (
            str(getattr(vlr, "user_id", "")).strip("\x00") == "LASF_Projection"
            and getattr(vlr, "record_id", None) == 2112
        )
        if not is_wkt:
            continue
        text = getattr(vlr, "string", None)
        if text:
            return str(text)
        raw = getattr(vlr, "record_data", None)
        if raw:
            try:
                return bytes(raw).decode("utf-8", errors="ignore").strip("\x00")
            except Exception:
                pass
    return None


def _header_geokeys(header) -> dict:
    for vlr in _header_vlrs(header):
        if type(vlr).__name__ != "GeoKeyDirectoryVlr":
            continue
        keys = {}
        for entry in getattr(vlr, "geo_keys", []) or []:
            try:
                if int(entry.tiff_tag_location) == 0:
                    keys[int(entry.id)] = int(entry.value_offset)
            except Exception:
                continue
        return keys
    return {}


def detect_linear_units(header) -> LinearUnits:
    """Units of a laspy header: WKT first, then GeoTIFF keys, else metres."""
    wkt = _header_wkt(header)
    if wkt:
        units = units_from_wkt(wkt)
        if units is not None:
            return units
    keys = _header_geokeys(header)
    if keys:
        units = units_from_geokeys(keys)
        if units is not None:
            return units
    if wkt or keys:
        return LinearUnits(
            1.0, 1.0,
            "the LAS header has a CRS but its unit could not be read; "
            "metres assumed",
            detected=False,
        )
    return LinearUnits(
        1.0, 1.0, "no CRS in the LAS header; metres assumed", detected=False,
    )


def resolve_units(header, override: Optional[str] = None) -> LinearUnits:
    """Units to use for ``header``: the user's override, else detection."""
    key = (override or "auto").strip().lower()
    try:
        detected = detect_linear_units(header)
    except ValueError:
        if key not in _OVERRIDE_FACTORS:
            raise
        # GeoTIFF angular checks precede vertical-unit resolution. An explicit
        # linear override can therefore repair unknown vertical units safely.
        detected = None
    if detected is not None and detected.angular:
        return detected
    if key in _OVERRIDE_FACTORS:
        factor = _OVERRIDE_FACTORS[key]
        return LinearUnits(
            factor, factor, f"user override: {unit_label(factor)}",
        )
    return detected
