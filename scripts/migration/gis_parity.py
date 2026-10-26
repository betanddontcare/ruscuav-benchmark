#!/usr/bin/env python3
"""The package's world builder against the notebook's, on the same OpenStreetMap data.

For each case the notebook's ``build_real_world_for_aoi``, with its footway-only
variant (CELL 10 of the notebook's Vienna cells), and ``ruscuav_benchmark.gis.build_world``
build the world of one area from the same OSM XML file, and every layer, the
identifiers and the area's UTM bounds are compared bit for bit, for both sidewalk
variants (the notebook's world with the sidewalk proxy, the package's ``proxy``).

The notebook downloaded its features with OSMnx; here its download function reads
them from the file with the OSMnx function the package uses
(``features_from_xml``, filtered by the area's bounding box as the download was).
One more change: for a layer without features in the area (none in the bounding
box, or none left after clipping to the square) the notebook builds an empty
GeoDataFrame with the CRS name "LOCAL_RUSCUAV", which GeoPandas 1 refuses; the
harness gives it an empty frame without CRS there. Areas with every layer do not
use this.

Cases: the synthetic town of tests/osm_fixtures.py at the paper's Vienna and Warsaw
areas and in the southern hemisphere (one of them with a synthetic DEM and DSM), an
area whose green areas and sensitive facilities lie only in the margin between its
bounding box and its square (``margin_osm_xml``), and OSM XML files given with
``--osm LAT,LON=FILE``. With ``--vienna`` also the builder's rules
(``check_builder_rules``) on the archived Vienna world.

    python scripts/migration/gis_parity.py --archive /data/RUSCUAV-v2.0.0 --work runs/gis_parity \\
        --vienna /data/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from ruscuav_benchmark.gis import Aoi, aoi_geometry, build_world, check_builder_rules  # noqa: E402
from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402
from ruscuav_benchmark.world import REQUIRED_LAYER_KEYS, UrbanWorldLayers  # noqa: E402

SYNTHETIC_CASES = [
    # (case, centre latitude, centre longitude, seed, with a DEM and a DSM)
    ("synthetic_vienna", 48.2152, 16.3975, 0, False),
    ("synthetic_warsaw", 52.2297, 21.0122, 1, False),
    ("synthetic_south_dem_dsm", -33.9249, 18.4241, 2, True),
]


def _write_rasters(directory: Path, lat: float, lon: float) -> Tuple[Path, Path]:
    """A DEM in WGS 84 (with a hole of no data) and a DSM in the area's UTM zone."""
    import rasterio
    from rasterio.transform import from_origin

    geo = aoi_geometry(Aoi(lat, lon))
    west, south, east, north = geo.bbox_west_south_east_north
    step = 0.0001
    w, h = int((east - west) / step) + 20, int((north - south) / step) + 20
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    dem = (120.0 + 8.0 * np.sin(xx / 37.0) + 5.0 * np.cos(yy / 23.0)).astype(np.float32)
    dem[h // 3: h // 3 + 15, w // 4: w // 4 + 25] = -9999.0
    dem_path = directory / "dem_wgs84.tif"
    with rasterio.open(dem_path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(west - 10 * step, north + 10 * step, step, step),
                       nodata=-9999.0) as dst:
        dst.write(dem, 1)
    minx, miny, maxx, maxy = geo.metric_bounds
    n = 520
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    dsm = (130.0 + 25.0 * np.exp(-((xx - 260) ** 2 + (yy - 240) ** 2) / 4000.0)).astype(np.float32)
    dsm_path = directory / "dsm_utm.tif"
    with rasterio.open(dsm_path, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32",
                       crs=geo.utm_crs, transform=from_origin(minx - 20.0, maxy + 20.0, 2.0, 2.0)) as dst:
        dst.write(dsm, 1)
    return dem_path, dsm_path


def _notebook_worlds(ns: Any, osm_file: Path, lat: float, lon: float, aoi_id: str, work: Path,
                     dem: Optional[Path], dsm: Optional[Path]) -> Tuple[Dict[str, UrbanWorldLayers], Dict[str, Any]]:
    import geopandas as gpd
    import osmnx as ox
    from pyproj.exceptions import CRSError

    def features(north, south, east, west, tags):
        return ox.features_from_xml(osm_file, polygon=ox.utils_geo.bbox_to_poly((west, south, east, north)),
                                    tags=tags)

    original_features, project = ns.ox_features_from_bbox_compat, ns.project_clip_to_local

    def project_or_empty(gdf, aoi_geo, size_m):
        # the CRS name "LOCAL_RUSCUAV" of the notebook's two empty-layer branches is the
        # only CRS GeoPandas 1 refuses in it
        try:
            return project(gdf, aoi_geo, size_m)
        except CRSError:
            return gpd.GeoDataFrame(geometry=[], crs=None)

    ns.ox_features_from_bbox_compat = features
    ns.project_clip_to_local = project_or_empty
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            world, aoi_geo, _, masks = ns.build_real_world_for_aoi(
                aoi_id=aoi_id, center_lat=lat, center_lon=lon, size_m=1000.0, resolution_m=1.0,
                dem_tif_path=None if dem is None else str(dem), dsm_tif_path=None if dsm is None else str(dsm),
                verbose=False)
            # CELL 10: the footway-only variant
            explicit = np.asarray(masks["footway_mask"], dtype=np.float32).copy()
            explicit[world.road_mask > 0.5] = 0.0
            explicit[world.building_mask > 0.5] = 0.0
            footway_only = ns.clone_world_with_replaced_sidewalk(world, explicit, "footway_only")
            out = {}
            for variant, w in (("proxy", world), ("footway-only", footway_only)):
                path = work / f"notebook_{aoi_id}_{variant}.npz"
                ns.save_real_world_manual_npz_v2(w, path)
                out[variant] = UrbanWorldLayers.load_npz(path)
    finally:
        ns.ox_features_from_bbox_compat, ns.project_clip_to_local = original_features, project
    geometry = {"utm_crs": aoi_geo["utm_crs"], "epsg": aoi_geo["epsg"],
                "metric_bounds_utm": list(aoi_geo["metric_bounds"]), "wgs_bbox": dict(aoi_geo["wgs_bbox"])}
    return out, geometry


def compare_worlds(ours: UrbanWorldLayers, theirs: UrbanWorldLayers) -> Dict[str, Any]:
    differing = {}
    for key in REQUIRED_LAYER_KEYS:
        a, b = getattr(ours, key), getattr(theirs, key)
        if a.shape != b.shape or not np.array_equal(a, b):
            differing[key] = int(np.sum(a != b)) if a.shape == b.shape else "shape"
    ids = {k: (getattr(ours, k), getattr(theirs, k)) for k in ("aoi_id", "crs", "bbox", "resolution_m")
           if getattr(ours, k) != getattr(theirs, k)}
    return {"layers_differing": differing, "identifiers_differing": {k: [str(a), str(b)] for k, (a, b) in ids.items()},
            "layers_sha256_equal": ours.layers_sha256() == theirs.layers_sha256()}


def run_case(ns: Any, case: str, osm_file: Path, lat: float, lon: float, work: Path, dem: Optional[Path] = None,
             dsm: Optional[Path] = None) -> Dict[str, Any]:
    t0 = time.time()
    notebook, nb_geometry = _notebook_worlds(ns, osm_file, lat, lon, case, work, dem, dsm)
    entry: Dict[str, Any] = {"osm_file": osm_file.name, "center": [lat, lon], "dem": dem is not None,
                             "dsm": dsm is not None}
    aoi = Aoi(lat, lon)
    entry["geometry_equal"] = aoi_geometry(aoi).to_dict() == nb_geometry
    for variant in ("proxy", "footway-only"):
        # the notebook kept the DEM's heights: the absolute datum
        built = build_world(aoi, case, osm_file=osm_file, dem=dem, dsm=dsm, terrain_datum="absolute",
                            sidewalks=variant)
        entry[variant] = compare_worlds(built.world, notebook[variant])
        entry[variant]["features_in_area"] = built.meta["osm"]["features_in_area"]
        entry[variant]["builder_rules"] = check_builder_rules(built.world, fallback_terrain=dem is None,
                                                              dsm=dsm is not None)["passed"]
        # what the file covers of the area, and the builder's warnings about it
        entry["area_covered"], entry["warnings"] = built.meta["osm"]["area_covered"], built.warnings
    entry["elapsed_s"] = round(time.time() - t0, 1)
    return entry


def main() -> int:
    from osm_fixtures import margin_osm_xml, synthetic_osm_xml

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path, help="extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--work", type=Path, default=Path("runs/gis_parity"))
    p.add_argument("--osm", action="append", default=[], metavar="LAT,LON=FILE",
                   help="also an OSM XML file, with the centre of the area to build from it")
    p.add_argument("--vienna", type=Path, default=None,
                   help="the archived Vienna footway-only world, to check the builder's rules on it")
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_legacy_namespace(args.archive, args.work / "legacy", include_repair_cells=False)

    cases: List[Tuple[str, Path, float, float, Optional[Path], Optional[Path]]] = []
    for case, lat, lon, seed, rasters in SYNTHETIC_CASES:
        path = synthetic_osm_xml(args.work / f"{case}.osm", lat, lon, seed=seed)
        dem = dsm = None
        if rasters:
            dem, dsm = _write_rasters(args.work, lat, lon)
        cases.append((case, path, lat, lon, dem, dsm))
    lat, lon = SYNTHETIC_CASES[0][1:3]
    cases.append(("margin_only_layers", margin_osm_xml(args.work / "margin_only_layers.osm", lat, lon), lat, lon,
                  None, None))
    for item in args.osm:
        center, _, file = item.partition("=")
        lat, lon = (float(v) for v in center.split(","))
        name = Path(file).name.split(".osm")[0].replace(",", "_").replace(".", "_")
        cases.append((f"{name}_at_{lat:g}_{lon:g}", Path(file), lat, lon, None, None))

    report: Dict[str, Any] = {"cases": {}}
    for case, path, lat, lon, dem, dsm in cases:
        entry = run_case(ns, case, path, lat, lon, args.work, dem, dsm)
        report["cases"][case] = entry
        print(case, json.dumps(entry), flush=True)
    if args.vienna is not None:
        world = UrbanWorldLayers.load_npz(args.vienna)
        report["vienna_builder_rules"] = check_builder_rules(world, fallback_terrain=True)
        print("vienna", json.dumps(report["vienna_builder_rules"]), flush=True)
    out = args.work / "gis_parity.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    bad = [c for c, e in report["cases"].items() if not e["geometry_equal"] or any(
        e[v]["layers_differing"] or e[v]["identifiers_differing"] or not e[v]["builder_rules"]
        for v in ("proxy", "footway-only"))]
    if args.vienna is not None and not report["vienna_builder_rules"]["passed"]:
        bad.append("vienna")
    print(f"{len(report['cases'])} cases; report: {out}" + (f"; differences in {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
