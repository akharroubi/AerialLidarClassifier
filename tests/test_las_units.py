"""Tests for utils/las_units.py (CRS unit detection, GitHub issues #5 and #2).

Run with any Python that has numpy::

    python tests/test_las_units.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _load import load_plugin_module  # noqa: E402

units = load_plugin_module("utils.las_units")

FT_US = 0.30480060960121924

WKT1_COMPOUND_FTUS = (
    'COMPD_CS["NAD83 / California zone 5 (ftUS) + NAVD88 height (ftUS)",'
    'PROJCS["NAD83 / California zone 5 (ftUS)",GEOGCS["NAD83",'
    'DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,'
    '298.257222101,AUTHORITY["EPSG","7019"]],AUTHORITY["EPSG","6269"]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
    'AUTHORITY["EPSG","4269"]],PROJECTION["Lambert_Conformal_Conic_2SP"],'
    'PARAMETER["standard_parallel_1",35.46666666666667],'
    'PARAMETER["false_easting",6561666.667],'
    'UNIT["US survey foot",0.304800609601219,AUTHORITY["EPSG","9003"]],'
    'AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","2229"]],'
    'VERT_CS["NAVD88 height (ftUS)",VERT_DATUM["North American Vertical '
    'Datum 1988",2005,AUTHORITY["EPSG","5103"]],'
    'UNIT["US survey foot",0.304800609601219,AUTHORITY["EPSG","9003"]],'
    'AXIS["Up",UP],AUTHORITY["EPSG","6360"]]]'
)

WKT2_COMPOUND_FTUS = (
    'COMPOUNDCRS["NAD83(2011) / Wisconsin Dane (ftUS) + NAVD88 height (ftUS)",'
    'PROJCRS["NAD83(2011) / Wisconsin Dane (ftUS)",BASEGEOGCRS["NAD83(2011)",'
    'DATUM["NAD83 (National Spatial Reference System 2011)",'
    'ELLIPSOID["GRS 1980",6378137,298.257222101,LENGTHUNIT["metre",1]]],'
    'PRIMEM["Greenwich",0,ANGLEUNIT["degree",0.0174532925199433]],'
    'ID["EPSG",6318]],CONVERSION["Wisconsin CRS Dane County",'
    'METHOD["Lambert Conic Conformal (1SP)"],'
    'PARAMETER["Latitude of natural origin",43.0695160375,'
    'ANGLEUNIT["degree",0.0174532925199433]],'
    'PARAMETER["False easting",811000,LENGTHUNIT["metre",1]]],'
    'CS[Cartesian,2],AXIS["easting (X)",east,ORDER[1],'
    'LENGTHUNIT["US survey foot",0.304800609601219]],'
    'AXIS["northing (Y)",north,ORDER[2],'
    'LENGTHUNIT["US survey foot",0.304800609601219]],ID["EPSG",8193]],'
    'VERTCRS["NAVD88 height (ftUS)",VDATUM["North American Vertical Datum '
    '1988"],CS[vertical,1],AXIS["gravity-related height (H)",up,'
    'LENGTHUNIT["US survey foot",0.304800609601219]],ID["EPSG",6360]]]'
)

WKT2_METRIC = (
    'PROJCRS["RGF93 v1 / Lambert-93",BASEGEOGCRS["RGF93 v1",'
    'DATUM["Reseau Geodesique Francais 1993 v1",ELLIPSOID["GRS 1980",'
    '6378137,298.257222101,LENGTHUNIT["metre",1]]],PRIMEM["Greenwich",0,'
    'ANGLEUNIT["degree",0.0174532925199433]],ID["EPSG",4171]],'
    'CONVERSION["Lambert-93",METHOD["Lambert Conic Conformal (2SP)"],'
    'PARAMETER["False easting",700000,LENGTHUNIT["metre",1]]],'
    'CS[Cartesian,2],AXIS["easting (X)",east,ORDER[1]],'
    'AXIS["northing (Y)",north,ORDER[2]],LENGTHUNIT["metre",1],'
    'USAGE[SCOPE["Engineering survey"],AREA["France"],BBOX[41.15,-9.86,'
    '51.56,10.38]],ID["EPSG",2154]]'
)

WKT1_PROJCS_FEET_NO_VERTICAL = (
    'PROJCS["NAD83 / Texas South Central (ftUS)",GEOGCS["NAD83",'
    'DATUM["North_American_Datum_1983",SPHEROID["GRS 1980",6378137,'
    '298.257222101]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Lambert_Conformal_Conic_2SP"],PARAMETER["false_easting",'
    '1968500],UNIT["US survey foot",0.30480060960121924],AUTHORITY["EPSG","2278"]]'
)

WKT1_GEOGRAPHIC = (
    'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],'
    'AUTHORITY["EPSG","4326"]]'
)


def test_wkt1_compound_us_feet():
    u = units.units_from_wkt(WKT1_COMPOUND_FTUS)
    assert u is not None and not u.angular
    assert abs(u.xy_to_m - FT_US) < 1e-9 and abs(u.z_to_m - FT_US) < 1e-9
    assert not u.is_metric
    assert "US survey foot" in u.describe()


def test_wkt2_compound_us_feet_ignores_metric_projection_parameters():
    u = units.units_from_wkt(WKT2_COMPOUND_FTUS)
    assert u is not None
    assert abs(u.xy_to_m - FT_US) < 1e-9, u.describe()
    assert abs(u.z_to_m - FT_US) < 1e-9, u.describe()


def test_wkt2_metric():
    u = units.units_from_wkt(WKT2_METRIC)
    assert u is not None and u.is_metric, u.describe()


def test_wkt1_feet_without_vertical_crs_assumes_same_unit_for_z():
    u = units.units_from_wkt(WKT1_PROJCS_FEET_NO_VERTICAL)
    assert u is not None
    assert abs(u.xy_to_m - FT_US) < 1e-9 and abs(u.z_to_m - FT_US) < 1e-9
    assert "assumed" in u.source


def test_wkt_geographic_is_reported_as_angular():
    u = units.units_from_wkt(WKT1_GEOGRAPHIC)
    assert u is not None and u.angular


def test_wkt_garbage_returns_none():
    assert units.units_from_wkt('PROJCS["broken",UNIT["metre"') is None
    assert units.units_from_wkt("") is None


def test_geokeys_feet_xy_metres_z():
    u = units.units_from_geokeys({1024: 1, 3072: 2229, 3076: 9003, 4099: 9001})
    assert u is not None
    assert abs(u.xy_to_m - FT_US) < 1e-9 and u.z_to_m == 1.0


def test_geokeys_metric_and_missing_vertical():
    u = units.units_from_geokeys({1024: 1, 3072: 3812, 3076: 9001})
    assert u is not None and u.is_metric


def test_geokeys_geographic():
    u = units.units_from_geokeys({1024: 2, 2048: 4326})
    assert u is not None and u.angular


def test_header_without_crs_assumes_metres_and_says_so():
    class Header:
        vlrs = []
        evlrs = []
    u = units.detect_linear_units(Header())
    assert u.is_metric and not u.detected
    assert "assumed" in u.source


def test_header_with_wkt_vlr():
    class WktCoordinateSystemVlr:
        string = WKT1_COMPOUND_FTUS

    class Header:
        vlrs = [WktCoordinateSystemVlr()]
        evlrs = []
    u = units.detect_linear_units(Header())
    assert abs(u.xy_to_m - FT_US) < 1e-9 and u.detected


def test_header_with_geokey_vlr():
    class Entry:
        def __init__(self, id_, value):
            self.id, self.tiff_tag_location, self.count, self.value_offset = id_, 0, 1, value

    class GeoKeyDirectoryVlr:
        geo_keys = [Entry(1024, 1), Entry(3076, 9002), Entry(4099, 9001)]

    class Header:
        vlrs = [GeoKeyDirectoryVlr()]
    u = units.detect_linear_units(Header())
    assert abs(u.xy_to_m - 0.3048) < 1e-12 and u.z_to_m == 1.0


def test_override_wins_over_header():
    class WktCoordinateSystemVlr:
        string = WKT2_METRIC

    class Header:
        vlrs = [WktCoordinateSystemVlr()]
    u = units.resolve_units(Header(), "us_foot")
    assert abs(u.xy_to_m - FT_US) < 1e-9 and "override" in u.source
    assert units.resolve_units(Header(), "auto").is_metric
    assert units.resolve_units(Header(), None).is_metric


def test_apply_scales_a_copy():
    u = units.LinearUnits(FT_US, 1.0, "test")
    pts = np.array([[1000.0, 2000.0, 30.0]])
    out = u.apply(pts)
    assert out is not pts
    assert abs(out[0, 0] - 1000 * FT_US) < 1e-9
    assert out[0, 2] == 30.0
    metric = units.LinearUnits(1.0, 1.0, "test")
    assert np.array_equal(metric.apply(pts), pts)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL  {name}: {exc}")
    raise SystemExit(1 if failures else 0)
