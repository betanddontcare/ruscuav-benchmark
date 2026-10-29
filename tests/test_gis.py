"""Tests for ruscuav_benchmark.gis (worlds from OpenStreetMap) and ``ruscuav build-aoi``.

The builder runs on the synthetic extracts of tests/osm_fixtures.py. With
RUSCUAV_ARCHIVE set, it is compared with the notebook's builder, layer for layer
(scripts/migration/gis_parity.py does this for more areas); with
RUSCUAV_VIENNA_WORLD set, the archived Vienna world is checked against the builder's
rules.
"""

from pathlib import Path
import contextlib
import importlib.util
import io
import json
import math
import os
import re
import sys

import numpy as np
import pytest

pytest.importorskip("osmnx")
pytest.importorskip("rasterio")
pytest.importorskip("geopandas")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from osm_fixtures import _Osm, margin_osm_xml, synthetic_osm_xml  # noqa: E402
from ruscuav_benchmark.cli import main  # noqa: E402
from ruscuav_benchmark.gis import (EXTRACT_MARGIN_M, PAPER_AOIS, Aoi, aoi_geometry, build_world,  # noqa: E402
                                   check_builder_rules, extract_bbox, load_osm_layers, scan_osm_file)
from ruscuav_benchmark.gis import layers as L  # noqa: E402
from ruscuav_benchmark.gis.osm import utm_epsg  # noqa: E402
from ruscuav_benchmark.validate import validate_world_npz  # noqa: E402
from ruscuav_benchmark.world import REQUIRED_LAYER_KEYS, UrbanWorldLayers  # noqa: E402

VIENNA = Aoi(PAPER_AOIS["vienna"]["center_lat"], PAPER_AOIS["vienna"]["center_lon"])
# PyProj 3.7 with NumPy 2.0 warns when GeoPandas projects a layer of a single point
ONE_POINT_LAYERS = pytest.mark.filterwarnings("ignore:Conversion of an array with ndim > 0:DeprecationWarning")
VIENNA_CENTER = (VIENNA.center_lat, VIENNA.center_lon)
SOUTH = Aoi(-33.9249, 18.4241)


@pytest.fixture(scope="module")
def osm_file(tmp_path_factory):
    return synthetic_osm_xml(tmp_path_factory.mktemp("osm") / "town.osm", VIENNA.center_lat, VIENNA.center_lon)


@pytest.fixture(scope="module")
def built(osm_file):
    return {variant: build_world(VIENNA, "town", osm_file=osm_file, sidewalks=variant)
            for variant in ("footway-only", "proxy")}


def test_the_area_is_the_papers():
    geo = aoi_geometry(VIENNA)
    # as the archived metadata of the Vienna world records it (worlds/real_gis/metadata/); the parity script
    # compares them bit for bit, here within a micrometre so that other PROJ versions pass
    assert geo.utm_crs == "EPSG:32633" and geo.epsg == 32633
    assert geo.metric_bounds == pytest.approx(
        (603311.7866921963, 5340663.372400162, 604311.7866921963, 5341663.372400162), abs=1e-6)
    expected = {"north": 48.21977870428312, "south": 48.210620923915215, "east": 16.4043518449135,
                "west": 16.390649332876446}
    assert geo.wgs_bbox == pytest.approx(expected, abs=1e-11)
    assert aoi_geometry(SOUTH).epsg == 32734 and VIENNA.shape == (1000, 1000)
    assert Aoi(1.0, 2.0, size_m=500.0, resolution_m=2.0).shape == (250, 250)
    for kwargs in ({"size_m": 1000.0, "resolution_m": 3.0}, {"size_m": -1.0}, {"resolution_m": 0.0},
                   {"size_m": math.inf}, {"resolution_m": math.nan}):
        with pytest.raises(ValueError):
            Aoi(48.0, 16.0, **kwargs)
    with pytest.raises(ValueError, match="UTM"):
        Aoi(88.0, 16.0)
    assert utm_epsg(10.0, 180.0) == 32660 and utm_epsg(-10.0, -180.0) == 32701
    with pytest.raises(ValueError, match="antimeridian"):
        aoi_geometry(Aoi(10.0, 179.999))
    # the extract to cut for the area: its bounding box and a margin
    west, south, east, north = extract_bbox(VIENNA)
    box = geo.wgs_bbox
    assert west < box["west"] and south < box["south"] and east > box["east"] and north > box["north"]
    assert (box["north"] - box["south"]) * 1.4 < north - south < (box["north"] - box["south"]) * 1.6
    assert EXTRACT_MARGIN_M == 250.0


def test_layer_rules():
    assert L.parse_osm_numeric("12 m") == 12.0 and L.parse_osm_numeric("12,5") == 12.5
    assert L.parse_osm_numeric("7;9") == 7.0 and L.parse_osm_numeric(21.5) == 21.5
    assert all(math.isnan(L.parse_osm_numeric(v)) for v in (None, "", "tall", "nan", "None"))
    row = lambda **tags: tags  # noqa: E731 - a mapping with get() and `in`, as the rows are
    assert L.building_height(row(height="200")) == 150.0 and L.building_height(row(height="1")) == 3.0
    assert L.building_height(row(**{"building:levels": "2.5"})) == pytest.approx(8.0)
    assert L.building_height(row(height="0", **{"building:levels": "5"})) == pytest.approx(16.0)
    assert L.building_height(row(height="tall", building="office")) == 18.0
    assert [L.building_height(row(building=b)) for b in ("house", "warehouse", "church", "school", "yes")] == \
        [12.0, 10.0, 25.0, 16.0, 12.0]
    assert [L.canopy_height(row(**t)) for t in ({"landuse": "forest"}, {"natural": "scrub"}, {"leisure": "park"})] \
        == [18.0, 6.0, 12.0]
    assert L.is_canopy(row(leisure="garden")) and not L.is_canopy(row(landuse="grass"))
    terrain = L.fallback_terrain((1000, 1000))
    assert terrain.dtype == np.float32 and 15.5 <= terrain.min() and terrain.max() <= 20.5
    # the fallback terrain is a surface in metres: at 2 m the cells take every other value
    assert np.array_equal(L.fallback_terrain((500, 500), 2.0), terrain[::2, ::2])


def test_build_world_from_an_osm_file(built, osm_file, tmp_path):
    fo, proxy = built["footway-only"], built["proxy"]
    assert fo.world.aoi_id == "town_footway_only" and proxy.world.aoi_id == "town"
    assert fo.world.crs == "LOCAL_RUSCUAV_0_1000" and fo.world.bbox == (0.0, 0.0, 1000.0, 1000.0)
    for b in (fo, proxy):
        assert check_builder_rules(b.world, fallback_terrain=True)["passed"]
        assert min(b.meta["osm"]["features_in_area"].values()) > 0
        assert b.warnings == [] and b.meta["warnings"] == []
    # the variants differ in the sidewalk layer only: the footways, or the footways and the proxy
    for key in REQUIRED_LAYER_KEYS:
        same = np.array_equal(getattr(fo.world, key), getattr(proxy.world, key))
        assert same == (key != "sidewalk_mask"), key
    footways, everything = fo.world.sidewalk_mask > 0.5, proxy.world.sidewalk_mask > 0.5
    assert footways.sum() < everything.sum() and not (footways & ~everything).any()
    assert np.array_equal(footways, (fo.footway_mask > 0.5) & (fo.world.road_mask < 0.5) &
                          (fo.world.building_mask < 0.5))
    meta = fo.meta
    assert meta["osm"]["source"] == "file" and meta["osm"]["file"] == "town.osm" and len(meta["osm"]["sha256"]) == 64
    assert meta["osm"]["area_covered"] == 1.0 and meta["osm"]["incomplete_multipolygons"] == 0
    assert len(meta["osm"]["bounds"]) == 1 and meta["osm"]["nodes_extent"] is not None
    assert meta["terrain"]["source"] == "fallback" and "OpenStreetMap" in meta["attribution"]
    assert meta["aoi"]["utm_crs"] == "EPSG:32633" and meta["sidewalks"] == "footway-only"
    assert meta["coverage"]["building"] == pytest.approx(float(fo.world.building_mask.mean()))
    # the file: valid (overlapping classes are warnings), read back with its metadata, deterministic
    path = Path(fo.world.save_npz(tmp_path / "town.npz"))
    report = validate_world_npz(path)
    assert report.ok and any("road and green share" in w for w in report.warnings)
    loaded = UrbanWorldLayers.load_npz(path)
    assert loaded.layers_sha256() == fo.world.layers_sha256() and loaded.meta["osm"]["sha256"] == meta["osm"]["sha256"]
    again = build_world(VIENNA, "town", osm_file=osm_file)
    assert again.world.layers_sha256() == fo.world.layers_sha256()
    # NumPy adds .npz to a file name without it; save_npz says where the file is
    assert fo.world.save_npz(tmp_path / "plain") == str(tmp_path / "plain.npz")


def _raster(path, geo, values, transform, crs="utm", nodata=None):
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1,
                       dtype="float32", crs=geo.utm_crs if crs == "utm" else crs, transform=transform,
                       nodata=nodata) as dst:
        dst.write(values.astype(np.float32), 1)
    return path


def test_dem_and_dsm(osm_file, tmp_path):
    from rasterio.transform import from_origin

    geo = aoi_geometry(VIENNA)
    minx, miny, maxx, maxy = geo.metric_bounds
    n = 300
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    dem = (150.0 + 0.02 * xx + 0.01 * yy).astype(np.float32)
    dem[10:20, 10:20] = -9999.0  # no data outside the area
    over = from_origin(minx - 100.0, maxy + 100.0, 4.0, 4.0)
    _raster(tmp_path / "dem.tif", geo, dem, over, nodata=-9999.0)
    _raster(tmp_path / "dsm.tif", geo, dem + 40.0, over)
    absolute = build_world(VIENNA, "town", osm_file=osm_file, dem=tmp_path / "dem.tif", dsm=tmp_path / "dsm.tif",
                           terrain_datum="absolute")
    w = absolute.world
    terrain_meta = absolute.meta["terrain"]
    assert terrain_meta["source"] == "dem" and len(terrain_meta["dsm"]["sha256"]) == 64
    assert terrain_meta["dem_coverage"] == 1.0 and terrain_meta["dsm_coverage"] == 1.0 and absolute.warnings == []
    assert 150.0 < float(w.terrain_z.min()) and float(w.terrain_z.max()) < 170.0 and "notes" not in absolute.meta
    assert np.all(w.surface_z >= w.terrain_z + 39.0)  # the DSM is 40 m above the terrain
    assert check_builder_rules(w, dsm=True)["passed"] and not check_builder_rules(w)["passed"]
    # by default the heights are relative to the area's lowest whole metre, as the environment needs them
    relative = build_world(VIENNA, "town", osm_file=osm_file, dem=tmp_path / "dem.tif", dsm=tmp_path / "dsm.tif")
    offset = relative.meta["terrain"]["offset_m"]
    assert offset == np.floor(float(w.terrain_z.min())) and relative.meta["terrain"]["datum"] == "relative"
    assert 0.0 <= float(relative.world.terrain_z.min()) < 1.0
    assert np.array_equal(relative.world.terrain_z, (w.terrain_z - np.float32(offset)).astype(np.float32))
    assert check_builder_rules(relative.world, dsm=True)["passed"]
    # terrain the environment cannot fly over: the validator refuses it
    path = Path(absolute.world.save_npz(tmp_path / "absolute.npz"))
    report = validate_world_npz(path)
    assert not report.ok and any("lowest point" in e for e in report.errors)
    # a DEM with data for the western half of the area only: the rest gets its median, with a warning
    half = from_origin(minx - 100.0, maxy + 100.0, 2.0, 2.0)
    _raster(tmp_path / "dem_half.tif", geo, np.full((700, 350), 20.0, np.float32), half)
    partial = build_world(VIENNA, "town", osm_file=osm_file, dem=tmp_path / "dem_half.tif")
    share = partial.meta["terrain"]["dem_coverage"]
    assert 0.55 < share < 0.65 and any("DEM has data for" in m for m in partial.warnings)
    assert partial.meta["warnings"] == partial.warnings
    # a raster without a CRS is refused (it would be taken for the area's UTM zone)
    _raster(tmp_path / "dem_nocrs.tif", geo, dem, over, crs=None)
    with pytest.raises(ValueError, match="has no CRS"):
        build_world(VIENNA, "town", osm_file=osm_file, dem=tmp_path / "dem_nocrs.tif")
    with pytest.raises(FileNotFoundError):
        build_world(VIENNA, "town", osm_file=osm_file, dem=tmp_path / "missing.tif")
    with pytest.raises(ValueError, match="needs the DEM"):
        build_world(VIENNA, "town", osm_file=osm_file, dsm=tmp_path / "dsm.tif")
    with pytest.raises(ValueError, match="terrain_datum"):
        build_world(VIENNA, "town", osm_file=osm_file, dem=tmp_path / "dem.tif", terrain_datum="sea")
    elsewhere = Aoi(52.2297, 21.0122)  # the DEM does not cover Warsaw
    with pytest.raises(ValueError, match="no data in the area"):
        build_world(elsewhere, "town", osm_layers={k: None for k in ("buildings", "highways", "green", "sensitive")},
                    dem=tmp_path / "dem.tif")


def test_sources_and_downloads(osm_file, tmp_path, monkeypatch):
    import osmnx as ox

    for kwargs in ({}, {"osm_file": osm_file, "download": True}):
        with pytest.raises(ValueError, match="one source"):
            build_world(VIENNA, "town", **kwargs)
    with pytest.raises(ValueError, match="sidewalks"):
        build_world(VIENNA, "town", osm_file=osm_file, sidewalks="all")
    with pytest.raises(ValueError, match="date applies to downloads"):
        load_osm_layers(VIENNA, osm_file, date="2026-07-21T00:00:00Z")
    with pytest.raises(ValueError, match="cache_folder applies to downloads"):
        load_osm_layers(VIENNA, osm_file, cache_folder=tmp_path)
    for date, message in (("21.07.2026", "ISO 8601"), ("2026-13-45T25:61:61Z", "ISO 8601"),
                          ("2026-7-21T00:00:00Z", "ISO 8601"), ("2005-01-01T00:00:00Z", "history"),
                          ("2999-01-01T00:00:00Z", "future")):
        with pytest.raises(ValueError, match=message):
            load_osm_layers(VIENNA, download=True, date=date)
    calls = []
    responses = {"save": True}

    def from_overpass(bbox, tags):
        settings = ox.settings
        calls.append((bbox, settings.overpass_settings, settings.use_cache, settings.cache_folder))
        if settings.use_cache and responses["save"]:  # as OSMnx keeps a downloaded response
            Path(settings.cache_folder).mkdir(parents=True, exist_ok=True)
            (Path(settings.cache_folder) / f"{len(calls)}.json").write_text("{}", encoding="utf-8")
        return ox.features_from_xml(osm_file, polygon=ox.utils_geo.bbox_to_poly(bbox), tags=tags)

    saved = ox.settings.overpass_settings, ox.settings.use_cache, ox.settings.cache_folder
    monkeypatch.setattr(ox, "features_from_bbox", from_overpass)
    cache = tmp_path / "cache"
    downloaded = build_world(VIENNA, "town", download=True, date="2026-07-21T00:00:00Z", cache_folder=cache)
    assert len(calls) == 4 and calls[0][0] == aoi_geometry(VIENNA).bbox_west_south_east_north
    assert calls[0][1].endswith('[date:"2026-07-21T00:00:00Z"]') and calls[0][2:] == (True, str(cache))
    assert (ox.settings.overpass_settings, ox.settings.use_cache, ox.settings.cache_folder) == saved  # restored
    osm = downloaded.meta["osm"]
    assert osm["source"] == "overpass" and osm["date"] == "2026-07-21T00:00:00Z" and osm["downloaded_at"]
    assert osm["cache"] == {"folder": "cache", "responses_saved": 4, "from_cache": False}
    from_file = build_world(VIENNA, "town", osm_file=osm_file)
    assert downloaded.world.layers_sha256() == from_file.world.layers_sha256()
    # the same download again: every response comes from the cache
    responses["save"] = False
    again = build_world(VIENNA, "town", download=True, date="2026-07-21T00:00:00Z", cache_folder=cache)
    assert again.meta["osm"]["cache"]["from_cache"] and again.meta["osm"]["downloaded_at"] is None
    # without a cache folder OSMnx keeps and reuses nothing (its default would be ./cache)
    calls.clear()
    plain = build_world(VIENNA, "town", download=True)
    assert [c[2] for c in calls] == [False] * 4 and plain.meta["osm"]["cache"] is None
    assert (ox.settings.overpass_settings, ox.settings.use_cache, ox.settings.cache_folder) == saved


def _incomplete_multipolygon(path):
    """A park whose multipolygon lacks the member way outside the area, as an export of openstreetmap.org has it."""
    import pyproj

    geo = aoi_geometry(VIENNA)
    minx, miny, maxx, maxy = geo.metric_bounds
    osm = _Osm(pyproj.Transformer.from_crs(geo.utm_crs, "EPSG:4326", always_xy=True), minx, maxy)
    a = osm.way([(600.0, 300.0), (1100.0, 300.0)], {})
    b = osm.way([(1100.0, 300.0), (1400.0, 300.0), (1400.0, 700.0), (1100.0, 700.0)], {})
    c = osm.way([(1100.0, 700.0), (600.0, 700.0), (600.0, 300.0)], {})
    members = "".join(f'<member type="way" ref="{r}" role="outer"/>' for r in (a, b, c))
    osm.relations.append(f'<relation id="{osm._id()}" version="1">{members}<tag k="type" v="multipolygon"/>'
                         '<tag k="leisure" v="park"/></relation>')
    osm.way([(0.0, 500.0), (1000.0, 500.0)], {"highway": "residential"})
    xml = re.sub(rf'<way id="{b}" version="1">.*?</way>', "", osm.xml(geo.bbox_west_south_east_north))
    path.write_text(xml, encoding="utf-8")
    return path


@ONE_POINT_LAYERS
def test_osm_files_that_do_not_fit_the_area(osm_file, tmp_path):
    # a file of another place: refused
    with pytest.raises(ValueError, match="does not cover the area"):
        build_world(Aoi(52.2297, 21.0122), "x", osm_file=osm_file)
    # a file that covers part of the area: a warning
    shifted = Aoi(VIENNA.center_lat, VIENNA.center_lon + 0.0067)  # about 500 m east
    partly = build_world(shifted, "x", osm_file=osm_file)
    assert 0.5 < partly.meta["osm"]["area_covered"] < 0.65
    assert any(re.search(r"covers about 5\d(\.\d)?% of the area", w) for w in partly.warnings)
    # multipolygons without some of their member ways are left out by OSMnx: a warning
    export = build_world(VIENNA, "x", osm_file=_incomplete_multipolygon(tmp_path / "export.osm"))
    assert export.meta["osm"]["incomplete_multipolygons"] == 1 and export.world.green_mask.sum() == 0
    assert any(w.startswith("1 multipolygon of the layers lacks member ways") for w in export.warnings)
    # layers whose features lie only between the bounding box and the square are read and clipped away
    margin = build_world(VIENNA, "x", osm_file=margin_osm_xml(tmp_path / "margin.osm", *VIENNA_CENTER))
    assert margin.meta["osm"]["features_in_area"] == {"buildings": 1, "highways": 1, "green": 0, "sensitive": 0}
    assert margin.warnings == [] and scan_osm_file(tmp_path / "margin.osm")["nodes"] > 0
    # files that are not OSM XML
    (tmp_path / "area.osm.pbf").write_bytes(b"\x00\x00\x00\x0d\x0a\x09OSMHeader\x18\x7c" + bytes(200))
    (tmp_path / "404.osm").write_text("404: Not Found", encoding="utf-8")
    (tmp_path / "page.osm").write_text("<html><body>Too many requests</body></html>", encoding="utf-8")
    (tmp_path / "cut.osm").write_text('<?xml version="1.0"?>\n<osm version="0.6"><node id="1" lat="48', "utf-8")
    for name, message in (("area.osm.pbf", "PBF file"), ("404.osm", "not an OSM XML file"),
                          ("page.osm", "root element is <html>"), ("cut.osm", "not a well-formed")):
        with pytest.raises(ValueError, match=message):
            build_world(VIENNA, "x", osm_file=tmp_path / name)
        with pytest.raises(ValueError, match=message):
            load_osm_layers(VIENNA, tmp_path / name)



def test_an_area_without_some_layers(tmp_path):
    path = tmp_path / "empty.osm"
    path.write_text('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6"></osm>\n', encoding="utf-8")
    b = build_world(VIENNA, "nothing", osm_file=path)
    assert b.meta["osm"]["features_in_area"] == {"buildings": 0, "highways": 0, "green": 0, "sensitive": 0}
    assert b.world.building_mask.sum() == 0 and check_builder_rules(b.world, fallback_terrain=True)["passed"]
    assert b.meta["osm"]["area_covered"] is None and any("no OpenStreetMap features" in w for w in b.warnings)


def test_build_aoi_from_the_command_line(osm_file, tmp_path, capsys, monkeypatch):
    out = tmp_path / "worlds" / "town.npz"
    code = main(["build-aoi", "--center", f"{VIENNA.center_lat},{VIENNA.center_lon}", "--aoi-id", "town",
                 "--osm", str(osm_file), "--output", str(out)])
    assert code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["aoi_id"] == "town_footway_only" and printed["valid"] and printed["terrain"] == "fallback"
    assert printed["layers_sha256"] == UrbanWorldLayers.load_npz(out).layers_sha256()
    assert main(["build-aoi", "--preset", "vienna", "--bbox-only"]) == 0
    box = json.loads(capsys.readouterr().out)
    assert box["wgs_bbox"] == aoi_geometry(VIENNA).wgs_bbox and box["extract_margin_m"] == EXTRACT_MARGIN_M
    e = box["extract_bbox"]
    assert box["osmium_extract"] == (f"osmium extract -s smart -b {e['west']!r},{e['south']!r},{e['east']!r},"
                                     f"{e['north']!r} REGION.osm.pbf -o area.osm")
    # a southern latitude (argparse alone would take -33.9 for an option), and a file name without .npz
    south_osm = synthetic_osm_xml(tmp_path / "south.osm", SOUTH.center_lat, SOUTH.center_lon)
    assert main(["build-aoi", "--center", f"{SOUTH.center_lat},{SOUTH.center_lon}", "--aoi-id", "cape",
                 "--osm", str(south_osm), "--output", str(tmp_path / "worlds" / "cape")]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["file"] == str(tmp_path / "worlds" / "cape.npz") and printed["valid"]
    assert printed["utm_crs"] == "EPSG:32734"
    # the builder's warnings come first among the warnings
    shifted = f"{VIENNA.center_lat},{VIENNA.center_lon + 0.0067}"
    assert main(["build-aoi", "--center", shifted, "--aoi-id", "east", "--osm", str(osm_file),
                 "--output", str(tmp_path / "east.npz")]) == 0
    assert "covers about" in json.loads(capsys.readouterr().out)["warnings"][0]
    assert main(["build-aoi", "--preset", "vienna", "--osm", str(osm_file)]) == 2
    assert "--output" in capsys.readouterr().err
    (tmp_path / "area.osm.pbf").write_bytes(b"\x00\x00\x00\x0d\x0a\x09OSMHeader" + bytes(200))
    for argv, message in (
        (["--aoi-id", "x", "--osm", str(osm_file)], "--center"),
        (["--preset", "vienna"], "--osm FILE"),
        (["--preset", "vienna", "--osm", str(osm_file), "--download"], "--osm FILE"),
        (["--preset", "vienna", "--osm", str(osm_file), "--osm-date", "2026-07-01T00:00:00Z"], "apply to --download"),
        (["--center", "48.2,abc", "--aoi-id", "x", "--osm", str(osm_file)], "LAT,LON"),
        (["--preset", "vienna", "--osm", str(tmp_path / "missing.osm")], "not found"),
        (["--preset", "vienna", "--center", shifted, "--osm", str(osm_file)], "give --aoi-id"),
        (["--preset", "warsaw", "--osm", str(osm_file)], "does not cover the area"),
        (["--preset", "vienna", "--osm", str(tmp_path / "area.osm.pbf")], "osmium cat"),
        (["--preset", "vienna", "--size", "inf", "--aoi-id", "x", "--osm", str(osm_file)], "finite"),
        (["--preset", "vienna", "--osm", str(osm_file), "--protocol", str(tmp_path / "p.json"), "--episodes", "0"],
         "--episodes"),
    ):
        assert main(["build-aoi", *argv, "--output", str(tmp_path / "x.npz")]) == 2, argv
        assert message in capsys.readouterr().err, argv
    assert not (tmp_path / "x.npz").exists()  # every error came before the world was written
    # the v2.0.0 script runs build-aoi, also for southern latitudes
    spec = importlib.util.spec_from_file_location("build_real_gis_world", REPO / "scripts" / "build_real_gis_world.py")
    wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(wrapper)
    monkeypatch.setattr(sys, "argv", ["build_real_gis_world.py", "--aoi-id", "cape", "--center-lat",
                                      str(SOUTH.center_lat), "--center-lon", str(SOUTH.center_lon),
                                      "--osm", str(south_osm), "--out", str(tmp_path / "v2" / "cape.npz")])
    assert wrapper.main() == 0
    assert json.loads(capsys.readouterr().out)["aoi_id"] == "cape"  # the v2.0.0 default: the sidewalk proxy
    # a protocol needs the "env" extra: said before the world is built
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "gymnasium" else find_spec(name, *a))
    assert main(["build-aoi", "--preset", "vienna", "--osm", str(osm_file), "--output", str(tmp_path / "noenv.npz"),
                 "--protocol", str(tmp_path / "noenv.json")]) == 2
    assert '"env" extra' in capsys.readouterr().err and not (tmp_path / "noenv.npz").exists()
    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    pytest.importorskip("gymnasium")
    pytest.importorskip("scipy")
    protocol = tmp_path / "protocols" / "town.json"
    assert main(["build-aoi", "--preset", "vienna", "--osm", str(osm_file), "--output", str(out), "--protocol",
                 str(protocol), "--episodes", "2", "--seed0", "7000"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["protocol"]["missions"] == 2
    data = json.loads(protocol.read_text(encoding="utf-8"))
    assert data["world"]["file"] == "../worlds/town.npz" and data["missions"][0]["seed"] == 7000
    assert "© OpenStreetMap contributors" in data["description"] and "ODbL 1.0" in data["description"]
    # the protocol finds its world next to it
    assert main(["evaluate", "--protocol", str(protocol), "--planner", "STATIC_Astar", "--episodes", "1",
                 "--output-dir", str(tmp_path / "eval"), "--quiet"]) == 0
    assert json.loads(capsys.readouterr().out)["protocol"] == "vienna_prater_leopoldstadt_osm_1km_v1_footway_only"


@ONE_POINT_LAYERS
@pytest.mark.skipif(not os.environ.get("RUSCUAV_ARCHIVE"), reason="needs RUSCUAV_ARCHIVE (the extracted archive)")
def test_the_builder_is_the_notebooks(osm_file, tmp_path):
    sys.path.insert(0, str(REPO / "scripts" / "migration"))
    import gis_parity
    from ruscuav_benchmark.legacy.runtime import load_legacy_namespace

    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_legacy_namespace(os.environ["RUSCUAV_ARCHIVE"], tmp_path / "legacy", include_repair_cells=False)
    margin = margin_osm_xml(tmp_path / "margin.osm", *VIENNA_CENTER)
    for case, path in (("town", osm_file), ("margin", margin)):
        entry = gis_parity.run_case(ns, case, path, VIENNA.center_lat, VIENNA.center_lon, tmp_path)
        assert entry["geometry_equal"], case
        for variant in ("proxy", "footway-only"):
            assert entry[variant]["layers_differing"] == {} and entry[variant]["identifiers_differing"] == {}, case


@pytest.mark.skipif(not os.environ.get("RUSCUAV_VIENNA_WORLD"), reason="needs RUSCUAV_VIENNA_WORLD")
def test_the_archived_vienna_world_follows_the_builder_rules():
    world = UrbanWorldLayers.load_npz(os.environ["RUSCUAV_VIENNA_WORLD"])
    report = check_builder_rules(world, fallback_terrain=True)
    assert report["passed"], report["cells"]
