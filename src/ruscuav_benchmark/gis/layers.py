"""From OpenStreetMap features to raster layers.

Each function is a step of the notebook's real-GIS pipeline, with its arithmetic,
so that a world built from the same features is the same, layer for layer:

- buildings: footprint polygons; height from ``height`` or ``building:height``
  (3 to 150 m), else ``building:levels`` or ``levels`` times 3.2 m, else by building
  type (:func:`building_height`);
- roads: drivable highways buffered to a width by class (:data:`ROAD_WIDTH_BY_HIGHWAY`),
  with a 4 m sidewalk proxy around them; footways (footway, path, pedestrian, steps,
  crossing, and ``footway=sidewalk`` or ``crossing``) buffered to 3 m;
- green areas: park, garden, grass, forest and similar polygons; canopy where they
  are forest, wood, scrub, park or garden (18, 6 or 12 m high, density 0.75);
- sensitive zones: schools, health care facilities and playgrounds, buffered by
  35-55 m (points and lines) or 10 m (areas);
- terrain: a DEM GeoTIFF reprojected bilinearly onto the grid, or, without one, the
  notebook's smooth fallback surface (18 m ± 2.5 m), which is not real terrain.

The features are clipped to the area before they are buffered, so features outside
it add nothing (a hospital just outside has no buffer inside). Rasterization marks
every cell a geometry touches (``all_touched``); when values overlap, the later
feature wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np

__all__ = [
    "ROAD_WIDTH_BY_HIGHWAY", "FOOTWAY_WIDTH_M", "SIDEWALK_PROXY_WIDTH_M", "CANOPY_DENSITY", "local_transform",
    "rasterize_mask", "rasterize_values", "parse_osm_numeric", "building_height", "prepare_buildings",
    "prepare_roads_and_sidewalks", "prepare_green", "is_canopy", "canopy_height", "prepare_canopy",
    "prepare_sensitive", "raster_to_grid", "fallback_terrain",
]

#: Road width (m) by ``highway`` class; other drivable classes are not roads.
ROAD_WIDTH_BY_HIGHWAY = {
    "motorway": 28.0, "trunk": 24.0, "primary": 18.0, "secondary": 15.0, "tertiary": 12.0, "unclassified": 9.0,
    "residential": 8.0, "living_street": 6.0, "service": 5.0, "road": 7.0, "track": 4.0,
}
#: Width (m) of a footway.
FOOTWAY_WIDTH_M = 3.0
#: Width (m) of the sidewalk proxy on each side of a road.
SIDEWALK_PROXY_WIDTH_M = 4.0
#: Density of the canopy cells.
CANOPY_DENSITY = 0.75
_NOT_ROADS = ("footway", "path", "pedestrian", "steps", "cycleway", "bridleway", "corridor", "platform")
_FOOTWAYS = ("footway", "path", "pedestrian", "steps", "crossing")


def local_transform(resolution_m: float) -> Any:
    """The raster transform of the local frame: column -> x, row -> y (southward)."""
    from affine import Affine

    return Affine(float(resolution_m), 0.0, 0.0, 0.0, float(resolution_m), 0.0)


def _valid_geoms(gdf: Any) -> List[Any]:
    if gdf is None or len(gdf) == 0:
        return []
    geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:  # noqa: BLE001 - as the notebook: keep the geometry
            pass
        if geom is not None and not geom.is_empty:
            geoms.append(geom)
    return geoms


def rasterize_mask(gdf: Any, shape: Tuple[int, int], transform: Any) -> np.ndarray:
    """1 on every cell a feature of ``gdf`` touches, else 0 (invalid geometries are
    repaired with ``buffer(0)``)."""
    from rasterio.features import rasterize

    geoms = _valid_geoms(gdf)
    if not geoms:
        return np.zeros(shape, dtype=np.float32)
    arr = rasterize([(g, 1.0) for g in geoms], out_shape=shape, fill=0.0, transform=transform, all_touched=True,
                    dtype="float32")
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def rasterize_values(gdf: Any, column: str, shape: Tuple[int, int], transform: Any,
                     default_value: float = 0.0) -> np.ndarray:
    """The value of ``column`` on every cell a feature touches (later features
    replace earlier ones), else 0; negative values become 0."""
    from rasterio.enums import MergeAlg
    from rasterio.features import rasterize

    if gdf is None or len(gdf) == 0 or column not in gdf.columns:
        return np.zeros(shape, dtype=np.float32)
    shapes = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:  # noqa: BLE001
            pass
        if geom is None or geom.is_empty:
            continue
        value = row.get(column, default_value)
        try:
            value = float(value)
        except Exception:  # noqa: BLE001
            value = default_value
        shapes.append((geom, value))
    if not shapes:
        return np.zeros(shape, dtype=np.float32)
    arr = rasterize(shapes, out_shape=shape, fill=0.0, transform=transform, all_touched=True, dtype="float32",
                    merge_alg=MergeAlg.replace)
    return np.maximum(arr, 0.0).astype(np.float32)


# ---------------------------------------------------------------------- buildings
def parse_osm_numeric(value: Any) -> float:
    """The number an OpenStreetMap tag value starts with ("12 m", "12,5", "3;4"), or NaN."""
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip().lower()
    if s in ("", "nan", "none", "null"):
        return np.nan
    s = s.replace("meters", "").replace("metres", "").replace("meter", "").replace("metre", "")
    s = s.replace("m", "")
    s = s.replace(",", ".")
    s = s.strip()
    for token in s.replace(";", " ").split():
        try:
            return float(token)
        except Exception:  # noqa: BLE001
            continue
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        return np.nan


def building_height(row: Any) -> float:
    """Height (m) of a building feature: its height tag, else its levels times 3.2 m
    (both clipped to 3-150 m), else by building type (12 m by default)."""
    for col in ("height", "building:height"):
        if col in row:
            h = parse_osm_numeric(row.get(col))
            if np.isfinite(h) and h > 0:
                return float(np.clip(h, 3.0, 150.0))
    for col in ("building:levels", "levels"):
        if col in row:
            levels = parse_osm_numeric(row.get(col))
            if np.isfinite(levels) and levels > 0:
                return float(np.clip(levels * 3.2, 3.0, 150.0))
    b = str(row.get("building", "")).lower()
    if b in ("apartments", "residential", "house", "detached", "terrace"):
        return 12.0
    if b in ("commercial", "retail", "office"):
        return 18.0
    if b in ("industrial", "warehouse"):
        return 10.0
    if b in ("church", "cathedral"):
        return 25.0
    if b in ("school", "hospital", "university"):
        return 16.0
    return 12.0


def prepare_buildings(buildings: Any) -> Any:
    """The building polygons with their heights (column ``height_m``)."""
    import geopandas as gpd

    if buildings is None or len(buildings) == 0:
        gdf = gpd.GeoDataFrame(geometry=[], crs=None)
        gdf["height_m"] = []
        return gdf
    gdf = buildings.copy()
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if len(gdf) == 0:
        gdf["height_m"] = []
        return gdf
    gdf["height_m"] = gdf.apply(building_height, axis=1).astype(float)
    return gdf


# ---------------------------------------------------------------------- roads and footways
def _highway(row: Any) -> str:
    h = row.get("highway", "")
    if isinstance(h, list):
        h = h[0] if h else ""
    return str(h).lower()


def _is_drive_road(row: Any) -> bool:
    h = _highway(row)
    if h in _NOT_ROADS:
        return False
    return h in ROAD_WIDTH_BY_HIGHWAY


def _is_pedestrian_way(row: Any) -> bool:
    if _highway(row) in _FOOTWAYS:
        return True
    return str(row.get("footway", "")).lower() in ("sidewalk", "crossing")


def _buffer(geom: Any, radius: float) -> Any:
    """A line or point buffered with flat ends and mitred joins; areas are kept."""
    if geom is None or geom.is_empty:
        return None
    try:
        if geom.geom_type in ("Polygon", "MultiPolygon"):
            return geom
        return geom.buffer(float(radius), cap_style=2, join_style=2)
    except Exception:  # noqa: BLE001
        try:
            return geom.buffer(float(radius))
        except Exception:  # noqa: BLE001
            return None


def prepare_roads_and_sidewalks(highways: Any) -> Tuple[Any, Any, Any]:
    """(road areas, sidewalk proxy rings around them, footway areas)."""
    import geopandas as gpd

    if highways is None or len(highways) == 0:
        return (gpd.GeoDataFrame(geometry=[], crs=None), gpd.GeoDataFrame(geometry=[], crs=None),
                gpd.GeoDataFrame(geometry=[], crs=None))
    gdf = highways.copy()
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    road_buffers, proxy_buffers, footway_buffers = [], [], []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if _is_drive_road(row):
            w = float(ROAD_WIDTH_BY_HIGHWAY.get(_highway(row), 7.0))
            road_poly = _buffer(geom, w / 2.0)
            outer_poly = _buffer(geom, w / 2.0 + SIDEWALK_PROXY_WIDTH_M)
            if road_poly is not None and not road_poly.is_empty:
                road_buffers.append(road_poly)
            if outer_poly is not None and road_poly is not None and not outer_poly.is_empty:
                try:
                    ring = outer_poly.difference(road_poly)
                except Exception:  # noqa: BLE001
                    ring = outer_poly
                if ring is not None and not ring.is_empty:
                    proxy_buffers.append(ring)
        if _is_pedestrian_way(row):
            foot_poly = _buffer(geom, FOOTWAY_WIDTH_M / 2.0)
            if foot_poly is not None and not foot_poly.is_empty:
                footway_buffers.append(foot_poly)
    return (gpd.GeoDataFrame(geometry=road_buffers, crs=None), gpd.GeoDataFrame(geometry=proxy_buffers, crs=None),
            gpd.GeoDataFrame(geometry=footway_buffers, crs=None))


# ---------------------------------------------------------------------- green, canopy, sensitive
def prepare_green(green: Any) -> Any:
    """The green-area polygons."""
    import geopandas as gpd

    if green is None or len(green) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    gdf = green.copy()
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    return gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()


def is_canopy(row: Any) -> bool:
    """Whether a green area has tree canopy: forest, wood, scrub, park or garden."""
    landuse = str(row.get("landuse", "")).lower()
    natural = str(row.get("natural", "")).lower()
    leisure = str(row.get("leisure", "")).lower()
    return landuse in ("forest",) or natural in ("wood", "scrub") or leisure in ("park", "garden")


def canopy_height(row: Any) -> float:
    """Canopy height (m): 18 for forest and wood, 6 for scrub, 12 for parks and gardens."""
    landuse = str(row.get("landuse", "")).lower()
    natural = str(row.get("natural", "")).lower()
    leisure = str(row.get("leisure", "")).lower()
    if landuse == "forest" or natural == "wood":
        return 18.0
    if natural == "scrub":
        return 6.0
    if leisure in ("park", "garden"):
        return 12.0
    return 10.0


def prepare_canopy(green: Any) -> Any:
    """The canopy polygons among the green areas, with ``canopy_height_m``."""
    import geopandas as gpd

    if green is None or len(green) == 0:
        gdf = gpd.GeoDataFrame(geometry=[], crs=None)
        gdf["canopy_height_m"] = []
        return gdf
    green = green.copy()
    green["is_canopy"] = green.apply(is_canopy, axis=1)
    gdf = green[green["is_canopy"] == True].copy()  # noqa: E712 - the notebook's selection
    if len(gdf) == 0:  # the notebook did not get this far without canopy
        gdf["canopy_height_m"] = []
        return gdf
    gdf["canopy_height_m"] = gdf.apply(canopy_height, axis=1)
    return gdf


def prepare_sensitive(sensitive: Any) -> Any:
    """Sensitive zones: health care facilities 55 m around, schools and universities
    45 m, playgrounds and others 35 m (for points and lines; areas 10 m)."""
    import geopandas as gpd

    if sensitive is None or len(sensitive) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=None)
    gdf = sensitive.copy()
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()
    buffers = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        amenity = str(row.get("amenity", "")).lower()
        healthcare = str(row.get("healthcare", "")).lower()
        leisure = str(row.get("leisure", "")).lower()
        if amenity in ("hospital", "clinic", "doctors") or healthcare not in ("", "nan", "none"):
            buf = 55.0
        elif amenity in ("school", "kindergarten", "university", "college"):
            buf = 45.0
        elif leisure in ("playground",):
            buf = 35.0
        else:
            buf = 35.0
        try:
            if geom.geom_type in ("Point", "MultiPoint", "LineString", "MultiLineString"):
                buffers.append(geom.buffer(buf))
            else:
                buffers.append(geom.buffer(10.0))
        except Exception:  # noqa: BLE001
            pass
    return gpd.GeoDataFrame(geometry=buffers, crs=None)


# ---------------------------------------------------------------------- terrain
def raster_to_grid(path: Union[str, Path], metric_bounds: Tuple[float, float, float, float], utm_crs: str,
                   shape: Tuple[int, int], resolution_m: float) -> Tuple[Optional[np.ndarray], float]:
    """A single-band GeoTIFF (a DEM or DSM, in any CRS it records) reprojected
    bilinearly onto the grid, missing cells filled with the median, and the share of
    the grid's cells it has data for; ``(None, 0.0)`` if it has no data on the grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"raster not found: {path}")
    minx, miny, maxx, maxy = metric_bounds
    dst_transform = from_origin(minx, maxy, float(resolution_m), float(resolution_m))
    dst = np.full(shape, np.nan, dtype=np.float32)
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f"the raster {path} has no CRS: record it in the file "
                             "(gdal_edit.py -a_srs EPSG:CODE FILE)")
        reproject(source=rasterio.band(src, 1), destination=dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=dst_transform, dst_crs=utm_crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    finite = np.isfinite(dst)
    if not finite.any():
        return None, 0.0
    median = float(np.nanmedian(dst))
    return np.where(finite, dst, median).astype(np.float32), float(finite.mean())


def fallback_terrain(shape: Tuple[int, int], resolution_m: float = 1.0) -> np.ndarray:
    """The notebook's terrain without a DEM, 18 m + 1.5 sin(x/180) + cos(y/220) with
    x, y in metres from the north-west corner (the notebook's grids are 1 m), in
    float32. It keeps the terrain smooth; it is not real terrain."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if float(resolution_m) != 1.0:
        yy, xx = yy * np.float32(resolution_m), xx * np.float32(resolution_m)
    terrain = 18.0 + 1.5 * np.sin(xx / 180.0) + 1.0 * np.cos(yy / 220.0)
    return terrain.astype(np.float32)
