"""Building a world NPZ from OpenStreetMap features.

:func:`build_world` runs the notebook's ``build_real_world_for_aoi`` (the pipeline of
the paper's Vienna world) and its footway-only variant:

1. the AOI's UTM bounds and the features of its four layers (:mod:`.osm`);
2. buildings and their heights, roads with a sidewalk proxy around them, footways,
   green areas and canopy, sensitive zones, rasterized on the grid (:mod:`.layers`);
3. the terrain from a DEM or the fallback surface, roofs and canopy tops, and the
   surface (with a DSM if given, at least the terrain, roofs and canopy tops);
4. exclusivity: no green area, canopy or sidewalk under buildings, no sidewalk on
   roads (roads may cross buildings and green areas, and footways green areas);
5. the sidewalk layer: the footways minus roads and buildings (``footway-only``, the
   worlds of the paper's real-GIS evaluations) or the footways and the sidewalk proxy
   (``proxy``).

The surface is computed before step 4, so where a building replaced canopy the
surface keeps the canopy top, as in the paper's worlds. The world records its
provenance in ``meta`` (``meta_json`` in the NPZ): the AOI, the OpenStreetMap source
(for a file: what it covers), the terrain, the libraries, the builder's warnings and
the attribution OpenStreetMap's licence requires.

On the same OpenStreetMap data, the notebook's 1 km grid at 1 m and the terrain the
notebook used (no DEM, or a DEM with ``terrain_datum="absolute"``) the world is the
notebook's, every layer bit for bit (``scripts/migration/gis_parity.py``).
"""

from __future__ import annotations

import hashlib
import math
import platform
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np

from ..world import UrbanWorldLayers
from . import layers as L
from .osm import (LAYERS, OSM_TAGS, Aoi, AoiGeometry, _load_osm_layers, aoi_geometry, file_coverage,
                  project_clip_to_local, scan_osm_file)

__all__ = ["SIDEWALK_VARIANTS", "TERRAIN_DATUMS", "ATTRIBUTION", "BuiltWorld", "rasterize_world", "build_world",
           "check_builder_rules"]

#: Sidewalk layers: the footways minus roads and buildings (the paper's evaluations),
#: or the footways and the sidewalk proxy around roads.
SIDEWALK_VARIANTS = ("footway-only", "proxy")
#: Heights of a DEM: relative to the area's lowest whole metre, or as the DEM has them.
TERRAIN_DATUMS = ("relative", "absolute")
#: Attribution required for worlds built from OpenStreetMap data.
ATTRIBUTION = ("Contains information from OpenStreetMap (© OpenStreetMap contributors), "
               "available under the Open Database License (ODbL 1.0)")


@dataclass
class BuiltWorld:
    """Result of :func:`build_world`."""

    world: UrbanWorldLayers
    geometry: AoiGeometry
    #: The intermediate masks: the footways and the sidewalk proxy, before exclusivity.
    footway_mask: np.ndarray
    sidewalk_proxy_mask: np.ndarray
    #: The features of each layer in the local frame.
    features: Dict[str, Any] = field(default_factory=dict)
    #: What to check in the world: an OSM file that covers only part of the area or
    #: lacks members of multipolygons, an area without features, a DEM or DSM with
    #: data for only part of it (also in ``meta["warnings"]``).
    warnings: List[str] = field(default_factory=list)

    @property
    def meta(self) -> Dict[str, Any]:
        return self.world.meta or {}


def rasterize_world(aoi: Aoi, features: Mapping[str, Any], geometry: Optional[AoiGeometry] = None, *,
                    dem: Union[str, Path, None] = None, dsm: Union[str, Path, None] = None,
                    terrain_datum: str = "relative") -> Dict[str, Any]:
    """The world's layers from the features of the four layers in the local frame
    (:func:`~ruscuav_benchmark.gis.osm.project_clip_to_local`): steps 2-4 above, with
    the sidewalk layer of the ``proxy`` variant; ``footway_mask`` and
    ``sidewalk_proxy_mask`` are the masks before exclusivity, ``terrain_source``
    ``"dem"`` or ``"fallback"``, ``dem_coverage`` and ``dsm_coverage`` the shares of
    the cells the rasters have data for (the others get the raster's median). With a
    DEM and ``terrain_datum="relative"`` the heights of the DEM and the DSM are
    lowered by the whole metres below the area's lowest point (``terrain_offset_m``),
    because the environment flies between 0 and 150 m; ``"absolute"`` keeps them (the
    notebook's arithmetic)."""
    if terrain_datum not in TERRAIN_DATUMS:
        raise ValueError(f"terrain_datum={terrain_datum!r}; expected one of {list(TERRAIN_DATUMS)}")
    if dsm is not None and dem is None:
        raise ValueError("a DSM needs the DEM of the same height reference (give dem as well)")
    geometry = aoi_geometry(aoi) if geometry is None else geometry
    shape, transform = aoi.shape, L.local_transform(aoi.resolution_m)

    buildings = L.prepare_buildings(features.get("buildings"))
    building_mask = L.rasterize_mask(buildings, shape, transform)
    building_height = (L.rasterize_values(buildings, "height_m", shape, transform) * building_mask).astype(np.float32)

    roads, proxy, footways = L.prepare_roads_and_sidewalks(features.get("highways"))
    road_mask = L.rasterize_mask(roads, shape, transform)
    sidewalk_proxy_mask = L.rasterize_mask(proxy, shape, transform)
    footway_mask = L.rasterize_mask(footways, shape, transform)
    sidewalk_mask = np.clip(sidewalk_proxy_mask + footway_mask, 0.0, 1.0).astype(np.float32)
    sidewalk_mask[road_mask > 0.5] = 0.0

    green = L.prepare_green(features.get("green"))
    green_mask = L.rasterize_mask(green, shape, transform)
    canopy = L.prepare_canopy(green)
    canopy_mask = L.rasterize_mask(canopy, shape, transform)
    canopy_height = (L.rasterize_values(canopy, "canopy_height_m", shape, transform) * canopy_mask).astype(np.float32)
    canopy_density = np.zeros(shape, dtype=np.float32)
    canopy_density[canopy_mask > 0.5] = L.CANOPY_DENSITY

    sensitive_mask = L.rasterize_mask(L.prepare_sensitive(features.get("sensitive")), shape, transform)

    terrain_source, offset, dsm_z = "fallback", 0.0, None
    dem_coverage: Optional[float] = None
    dsm_coverage: Optional[float] = None
    if dem is None:
        terrain = L.fallback_terrain(shape, aoi.resolution_m)
    else:
        terrain_source = "dem"
        terrain, dem_coverage = L.raster_to_grid(dem, geometry.metric_bounds, geometry.utm_crs, shape,
                                                 aoi.resolution_m)
        if terrain is None:
            raise ValueError(f"the DEM {dem} has no data in the area")
        if dsm is not None:
            dsm_z, dsm_coverage = L.raster_to_grid(dsm, geometry.metric_bounds, geometry.utm_crs, shape,
                                                   aoi.resolution_m)
            if dsm_z is None:
                raise ValueError(f"the DSM {dsm} has no data in the area")
        if terrain_datum == "relative":
            offset = float(np.floor(float(terrain.min())))
            terrain = (terrain - np.float32(offset)).astype(np.float32)
            if dsm_z is not None:
                dsm_z = (dsm_z - np.float32(offset)).astype(np.float32)
    roof = terrain + building_height
    canopy_top = terrain + canopy_height
    surface = np.maximum.reduce([terrain, roof, canopy_top]).astype(np.float32)
    if dsm_z is not None:
        surface = np.maximum(surface, dsm_z).astype(np.float32)

    # exclusivity (the notebook's step 7)
    green_mask = green_mask.astype(np.float32)
    green_mask[building_mask > 0.5] = 0.0
    canopy_mask = canopy_mask.astype(np.float32)
    canopy_mask[building_mask > 0.5] = 0.0
    canopy_height = canopy_height * canopy_mask
    canopy_density = canopy_density * canopy_mask
    sidewalk_mask[building_mask > 0.5] = 0.0

    # the world object and its compatibility patch: heights on their masks, derived layers
    terrain = terrain.astype(np.float32)
    surface = surface.astype(np.float32)
    building_height = (building_height.astype(np.float32) * building_mask).astype(np.float32)
    canopy_height = (canopy_height.astype(np.float32) * canopy_mask).astype(np.float32)
    canopy_density = (canopy_density.astype(np.float32) * canopy_mask).astype(np.float32)
    return {
        "terrain_z": terrain, "surface_z": surface,
        "height_above_ground": np.maximum(surface - terrain, 0.0).astype(np.float32),
        "building_mask": building_mask, "building_height": building_height,
        "building_roof_z": (terrain + building_height).astype(np.float32),
        "road_mask": road_mask, "sidewalk_mask": sidewalk_mask, "green_mask": green_mask,
        "canopy_mask": canopy_mask, "canopy_height": canopy_height,
        "canopy_top_z": (terrain + canopy_height).astype(np.float32), "canopy_density": canopy_density,
        "sensitive_mask": sensitive_mask,
        "footway_mask": footway_mask, "sidewalk_proxy_mask": sidewalk_proxy_mask, "terrain_source": terrain_source,
        "terrain_offset_m": offset, "dem_coverage": dem_coverage, "dsm_coverage": dsm_coverage,
    }


def _sha256(path: Union[str, Path]) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _libraries() -> Dict[str, Optional[str]]:
    """Versions of the libraries the layers depend on (buffers and rasterization can
    change in the last decimals between GEOS and GDAL versions)."""
    out: Dict[str, Optional[str]] = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("geopandas", "shapely", "rasterio", "pyproj", "osmnx"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", None)
        except ImportError:  # pragma: no cover - the gis extra has them all
            out[name] = None
    try:
        import shapely
        out["geos"] = shapely.geos_version_string
    except (ImportError, AttributeError):  # pragma: no cover
        out["geos"] = None
    try:
        import rasterio
        out["gdal"] = rasterio.__gdal_version__
    except (ImportError, AttributeError):  # pragma: no cover
        out["gdal"] = None
    try:
        import pyproj
        out["proj"] = pyproj.proj_version_str
    except (ImportError, AttributeError):  # pragma: no cover
        out["proj"] = None
    return out


def _raster_record(path: Union[str, Path, None]) -> Optional[Dict[str, str]]:
    return None if path is None else {"file": Path(path).name, "sha256": _sha256(path)}


def _percent(share: float) -> str:
    """A share as a percentage, rounded down (so that a share below 1 never shows 100%)."""
    return f"{math.floor(share * 1000) / 10:g}%"


def _box(boxes: Sequence[Sequence[float]]) -> str:
    west, south = min(b[0] for b in boxes), min(b[1] for b in boxes)
    east, north = max(b[2] for b in boxes), max(b[3] for b in boxes)
    return f"west {west:.5f}, south {south:.5f}, east {east:.5f}, north {north:.5f}"


def _file_checks(osm_file: Union[str, Path], geometry: AoiGeometry) -> Dict[str, Any]:
    """What an OSM file covers of the area, and the warnings about it; raises if it
    covers none of it."""
    scan = scan_osm_file(osm_file)
    covered = file_coverage(scan, geometry)
    boxes = scan["bounds"] or ([scan["nodes_extent"]] if scan["nodes_extent"] else [])
    where = "by its bounds" if scan["bounds"] else "by the extent of its nodes"
    name = Path(osm_file).name
    if covered == 0.0:
        lie = "its bounds are" if scan["bounds"] else "its nodes lie within"
        raise ValueError(f"the OSM file {name} does not cover the area: {lie} {_box(boxes)}, and the area's "
                         f"bounding box is {_box([geometry.bbox_west_south_east_north])}; check the centre, or cut "
                         "an extract for the area (ruscuav build-aoi --bbox-only prints the command)")
    warnings = []
    if covered is not None and covered < (0.99 if scan["bounds"] else 0.9):
        warnings.append(f"the OSM file {name} covers about {_percent(covered)} of the area ({where}): the features of "
                        "the rest of the area are missing")
    n = scan["incomplete_multipolygons"]
    if n:
        what = "1 multipolygon of the layers lacks" if n == 1 else f"{n} multipolygons of the layers lack"
        warnings.append(f"{what} member ways in the OSM file {name}, and OSMnx leaves {'it' if n == 1 else 'them'} "
                        "out: where such an area reaches into the world it is missing (an export from "
                        "openstreetmap.org has only the member ways with a node in its box; cut extracts with "
                        "osmium extract -s smart, or use --download)")
    return {"record": {"bounds": scan["bounds"], "nodes_extent": scan["nodes_extent"], "area_covered": covered,
                       "incomplete_multipolygons": scan["incomplete_multipolygons"]}, "warnings": warnings}


def build_world(aoi: Aoi, aoi_id: str, *, osm_file: Union[str, Path, None] = None, download: bool = False,
                date: Optional[str] = None, cache_folder: Union[str, Path, None] = None,
                osm_layers: Optional[Mapping[str, Any]] = None, dem: Union[str, Path, None] = None,
                dsm: Union[str, Path, None] = None, terrain_datum: str = "relative",
                sidewalks: str = "footway-only") -> BuiltWorld:
    """Build the world of ``aoi`` from OpenStreetMap: from an OSM XML file
    (``osm_file``), a download (``download``, with ``date`` and ``cache_folder`` as in
    :func:`~ruscuav_benchmark.gis.osm.load_osm_layers`) or features already loaded
    (``osm_layers``, as that function returns them). An OSM file that covers none of
    the area is refused; one that covers part of it, or lacks member ways of
    multipolygons, gives warnings (:attr:`BuiltWorld.warnings`). ``dem`` and ``dsm``
    are GeoTIFFs of the terrain and of the surface (``terrain_datum`` as in
    :func:`rasterize_world`); without a DEM the terrain is the fallback surface.
    ``sidewalks`` chooses the sidewalk layer (:data:`SIDEWALK_VARIANTS`); the
    footway-only world's ``aoi_id`` gets the suffix ``_footway_only``, as the paper's
    worlds have it. Needs the "gis" extra."""
    if sidewalks not in SIDEWALK_VARIANTS:
        raise ValueError(f"sidewalks={sidewalks!r}; expected one of {list(SIDEWALK_VARIANTS)}")
    if not str(aoi_id).strip():
        raise ValueError("aoi_id must not be empty")
    given = sum(x for x in (osm_file is not None, bool(download), osm_layers is not None))
    if given != 1:
        raise ValueError("give one source: an OSM XML file, download=True or osm_layers")
    for path in (dem, dsm):
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(f"raster not found: {path}")
    if terrain_datum not in TERRAIN_DATUMS:
        raise ValueError(f"terrain_datum={terrain_datum!r}; expected one of {list(TERRAIN_DATUMS)}")
    if dsm is not None and dem is None:
        raise ValueError("a DSM needs the DEM of the same height reference (give dem as well)")
    geometry = aoi_geometry(aoi)
    checks = _file_checks(osm_file, geometry) if osm_file is not None else {"record": {}, "warnings": []}
    warnings: List[str] = list(checks["warnings"])
    download_info: Dict[str, Any] = {}
    if osm_layers is not None:
        raw = dict(osm_layers)
    else:
        raw, download_info = _load_osm_layers(aoi, osm_file, download=download, date=date, cache_folder=cache_folder)
    missing = [name for name in LAYERS if name not in raw]
    if missing:
        raise ValueError(f"osm_layers has no layers {missing}")
    local = {name: project_clip_to_local(raw[name], geometry, aoi.size_m) for name in LAYERS}
    if not any(len(local[name]) for name in LAYERS):
        warnings.append("the area has no OpenStreetMap features of the four layers (buildings, highways, green areas, "
                        "sensitive facilities)")
    arrays = rasterize_world(aoi, local, geometry, dem=dem, dsm=dsm, terrain_datum=terrain_datum)
    footway_mask, proxy_mask = arrays.pop("footway_mask"), arrays.pop("sidewalk_proxy_mask")
    terrain_source, offset = arrays.pop("terrain_source"), arrays.pop("terrain_offset_m")
    dem_coverage, dsm_coverage = arrays.pop("dem_coverage"), arrays.pop("dsm_coverage")
    for what, share in (("DEM", dem_coverage), ("DSM", dsm_coverage)):
        if share is not None and share < 1.0:
            warnings.append(f"the {what} has data for {_percent(share)} of the area's cells; the others get its median")
    world_id = str(aoi_id)
    if sidewalks == "footway-only":
        explicit = footway_mask.copy()
        explicit[arrays["road_mask"] > 0.5] = 0.0
        explicit[arrays["building_mask"] > 0.5] = 0.0
        arrays["sidewalk_mask"] = explicit.astype(np.float32)
        world_id += "_footway_only"

    from .. import __version__

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if osm_file is not None:
        source: Dict[str, Any] = {"source": "file", "file": Path(osm_file).name, "sha256": _sha256(osm_file),
                                  **checks["record"]}
    elif download:
        source = {"source": "overpass", "date": date, "downloaded_at": now, "cache": None}
        if cache_folder is not None:
            saved = download_info.get("responses_saved", 0)
            source["cache"] = {"folder": Path(cache_folder).name, "responses_saved": saved, "from_cache": saved == 0}
            if saved == 0:  # every response was read from the cache: downloaded earlier
                source["downloaded_at"] = None
    else:
        source = {"source": "features"}
    source["tags"] = {k: dict(v) for k, v in OSM_TAGS.items()}
    source["features_in_area"] = {name: int(len(local[name])) for name in LAYERS}
    meta = {
        "builder": "ruscuav_benchmark.gis.build_world (the notebook's build_real_world_for_aoi)",
        "ruscuav_benchmark": __version__,
        "created_at": now,
        "aoi": {"aoi_id": str(aoi_id), "center_lat": float(aoi.center_lat), "center_lon": float(aoi.center_lon),
                "size_m": float(aoi.size_m), "resolution_m": float(aoi.resolution_m), **geometry.to_dict()},
        "sidewalks": sidewalks,
        "osm": source,
        "terrain": {"source": terrain_source, "dem": _raster_record(dem), "dsm": _raster_record(dsm),
                    "datum": None if dem is None else terrain_datum, "offset_m": offset,
                    "dem_coverage": dem_coverage, "dsm_coverage": dsm_coverage},
        "coverage": {name.replace("_mask", ""): float(arrays[name].mean()) for name in (
            "building_mask", "road_mask", "sidewalk_mask", "green_mask", "canopy_mask", "sensitive_mask")},
        "libraries": _libraries(),
        "warnings": warnings,
        "attribution": ATTRIBUTION,
    }
    if terrain_source == "fallback":
        meta["notes"] = ["terrain_z is the smooth fallback surface of the paper's real-GIS worlds, not real terrain"]
    size = float(aoi.size_m)
    world = UrbanWorldLayers(aoi_id=world_id, crs=f"LOCAL_RUSCUAV_0_{size:g}", bbox=(0.0, 0.0, size, size),
                             resolution_m=float(aoi.resolution_m), meta=meta, **arrays)
    return BuiltWorld(world, geometry, footway_mask, proxy_mask, local, warnings)


def check_builder_rules(world: UrbanWorldLayers, fallback_terrain: bool = False, dsm: bool = False) -> Dict[str, Any]:
    """How many cells of ``world`` break the rules worlds built by :func:`build_world`
    keep (0 each for such a world): no sidewalk on roads or buildings, no green area
    or canopy under buildings, canopy inside green areas, heights only on their masks
    (buildings 3-150 m, canopy 6, 10, 12 or 18 m, density 0.75), roofs, canopy tops
    and height above ground derived from their layers, and the surface the maximum of
    terrain, roofs and canopy tops outside buildings and at least that inside them
    (with ``dsm``: at least that everywhere); with ``fallback_terrain`` also the
    fallback terrain, bit for bit. Returns ``{"passed": bool, "cells": {rule: count}}``."""
    b, r, s = world.building_mask > 0.5, world.road_mask > 0.5, world.sidewalk_mask > 0.5
    g, c = world.green_mask > 0.5, world.canopy_mask > 0.5
    terrain, bh, ch = world.terrain_z, world.building_height, world.canopy_height
    top = np.maximum.reduce([terrain, world.building_roof_z, world.canopy_top_z])
    cells = {
        "sidewalk_on_road": int((s & r).sum()),
        "sidewalk_on_building": int((s & b).sum()),
        "green_under_building": int((g & b).sum()),
        "canopy_under_building": int((c & b).sum()),
        "canopy_outside_green": int((c & ~g).sum()),
        "building_height_outside_buildings": int((bh[~b] != 0).sum()),
        "building_height_outside_3_150_m": int(((bh[b] < 3.0) | (bh[b] > 150.0)).sum()),
        "canopy_height_outside_canopy": int((ch[~c] != 0).sum()),
        "canopy_height_not_6_10_12_18_m": int((~np.isin(ch[c], np.float32([6.0, 10.0, 12.0, 18.0]))).sum()),
        "canopy_density_not_0_75": int((world.canopy_density != np.where(c, np.float32(0.75), np.float32(0))).sum()),
        "roof_not_terrain_plus_height": int((world.building_roof_z != (terrain + bh)).sum()),
        "canopy_top_not_terrain_plus_height": int((world.canopy_top_z != (terrain + ch)).sum()),
        "height_above_ground_not_surface_minus_terrain": int(
            (world.height_above_ground != np.maximum(world.surface_z - terrain, 0.0)).sum()),
    }
    if dsm:
        cells["surface_below_top"] = int((world.surface_z < top).sum())
    else:
        cells["surface_not_top_outside_buildings"] = int((world.surface_z[~b] != top[~b]).sum())
        cells["surface_below_top_in_buildings"] = int((world.surface_z[b] < top[b]).sum())
    if fallback_terrain:
        cells["terrain_not_fallback"] = int((terrain != L.fallback_terrain(terrain.shape, world.resolution_m)).sum())
    return {"passed": not any(cells.values()), "cells": cells}
