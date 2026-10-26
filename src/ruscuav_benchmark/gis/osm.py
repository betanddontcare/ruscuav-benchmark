"""The area of interest and its OpenStreetMap features.

An area of interest (AOI) is a square of ``size_m`` metres around a centre given in
WGS 84, in the UTM zone of the centre. Its OpenStreetMap features are read from an
OSM XML file or downloaded from the Overpass API with OSMnx, in four layers
(buildings, highways, green areas, sensitive facilities) selected by the tags of
:data:`OSM_TAGS`, and then projected into the world's local frame: x eastward and
y southward from the AOI's north-west corner, in metres. These are the steps of the
notebook's ``build_real_world_for_aoi``, which built the paper's Vienna world.
:func:`scan_osm_file` and :func:`file_coverage` tell whether an OSM file covers the
AOI and holds its multipolygons whole.
"""

from __future__ import annotations

import bz2
import gzip
import math
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Set, Tuple, Union

__all__ = ["Aoi", "AoiGeometry", "OSM_TAGS", "LAYERS", "PAPER_AOIS", "OVERPASS_HISTORY_START", "EXTRACT_MARGIN_M",
           "utm_epsg", "aoi_geometry", "extract_bbox", "scan_osm_file", "file_coverage", "load_osm_layers",
           "project_clip_to_local"]

#: The OpenStreetMap tags of each layer: a feature belongs to a layer when it has one
#: of the keys with ``True`` or one of the listed values.
OSM_TAGS: Mapping[str, Mapping[str, Any]] = {
    "buildings": {"building": True},
    "highways": {"highway": True},
    "green": {
        "leisure": ["park", "garden", "recreation_ground", "playground"],
        "landuse": ["grass", "forest", "meadow", "recreation_ground", "village_green", "cemetery"],
        "natural": ["wood", "grassland", "scrub", "heath"],
    },
    "sensitive": {
        "amenity": ["school", "kindergarten", "hospital", "clinic", "doctors", "university", "college",
                    "social_facility", "nursing_home"],
        "healthcare": True,
        "leisure": ["playground"],
    },
}
#: The layers, in the order they are read.
LAYERS = ("buildings", "highways", "green", "sensitive")
#: Overpass keeps the history of OpenStreetMap from this time on; for an earlier
#: ``[date:...]`` it gives the data of this time.
OVERPASS_HISTORY_START = "2012-09-12T06:55:00Z"
#: The margin (m) of the extract that ``build-aoi --bbox-only`` proposes around the
#: AOI, so that ways crossing the AOI's edge between nodes outside it come with it.
EXTRACT_MARGIN_M = 250.0
#: The time format of Overpass's ``[date:...]``.
_OVERPASS_DATE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\Z")
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
#: The relations OSMnx builds areas from.
_AREA_RELATIONS = ("multipolygon", "boundary")

#: The areas of the paper's real-GIS worlds (``build-aoi --preset``). The Vienna world
#: was built with this pipeline from OpenStreetMap data of July 2026; the Warsaw world
#: was built step by step in the notebook with the same steps.
PAPER_AOIS: Mapping[str, Mapping[str, Any]] = {
    "vienna": {"aoi_id": "vienna_prater_leopoldstadt_osm_1km_v1", "center_lat": 48.2152, "center_lon": 16.3975,
               "size_m": 1000.0, "resolution_m": 1.0},
    "warsaw": {"aoi_id": "warsaw_centrum_osm_1km_v1", "center_lat": 52.2297, "center_lon": 21.0122,
               "size_m": 1000.0, "resolution_m": 1.0},
}


@dataclass(frozen=True)
class Aoi:
    """A square area of ``size_m`` metres around (``center_lat``, ``center_lon``),
    rasterized at ``resolution_m``."""

    center_lat: float
    center_lon: float
    size_m: float = 1000.0
    resolution_m: float = 1.0

    def __post_init__(self) -> None:
        try:
            values = [float(v) for v in (self.center_lat, self.center_lon, self.size_m, self.resolution_m)]
        except (TypeError, ValueError):
            raise ValueError("the centre, size_m and resolution_m of an area must be numbers") from None
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"the centre, size_m and resolution_m of an area must be finite; got {values}")
        if not (-80.0 <= float(self.center_lat) <= 84.0 and -180.0 <= float(self.center_lon) <= 180.0):
            raise ValueError(f"the centre ({self.center_lat}, {self.center_lon}) must lie in the UTM zones "
                             "(latitude -80 to 84, longitude -180 to 180)")
        if not (float(self.size_m) > 0 and float(self.resolution_m) > 0):
            raise ValueError("size_m and resolution_m must be positive")
        cells = float(self.size_m) / float(self.resolution_m)
        if abs(cells - round(cells)) > 1e-9 * max(1.0, cells):
            raise ValueError(f"size_m ({self.size_m}) must be a whole number of cells of {self.resolution_m} m")

    @property
    def shape(self) -> Tuple[int, int]:
        """The grid (rows, columns)."""
        n = int(round(float(self.size_m) / float(self.resolution_m)))
        return n, n


@dataclass(frozen=True)
class AoiGeometry:
    """The AOI in its UTM zone (``metric_bounds``: min x, min y, max x, max y) and as
    the WGS 84 bounding box of its corners (``wgs_bbox``: north, south, east, west)."""

    utm_crs: str
    epsg: int
    metric_bounds: Tuple[float, float, float, float]
    wgs_bbox: Dict[str, float]

    @property
    def bbox_west_south_east_north(self) -> Tuple[float, float, float, float]:
        b = self.wgs_bbox
        return b["west"], b["south"], b["east"], b["north"]

    def to_dict(self) -> Dict[str, Any]:
        return {"utm_crs": self.utm_crs, "epsg": self.epsg, "metric_bounds_utm": list(self.metric_bounds),
                "wgs_bbox": dict(self.wgs_bbox)}


def utm_epsg(lat: float, lon: float) -> int:
    """EPSG code of the WGS 84 / UTM zone of a point (longitude 180 is in zone 60)."""
    zone = min(int((lon + 180.0) // 6.0) + 1, 60)
    return 32600 + zone if lat >= 0 else 32700 + zone


def _square(aoi: Aoi, half: float) -> AoiGeometry:
    """The square of side ``2 * half`` around the AOI's centre (as the notebook computed it)."""
    import pyproj

    epsg = utm_epsg(aoi.center_lat, aoi.center_lon)
    utm_crs = f"EPSG:{epsg}"
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(aoi.center_lon, aoi.center_lat)
    minx, maxx, miny, maxy = cx - half, cx + half, cy - half, cy + half
    corners = [to_wgs.transform(x, y) for x, y in ((minx, miny), (minx, maxy), (maxx, miny), (maxx, maxy))]
    lons, lats = [p[0] for p in corners], [p[1] for p in corners]
    if max(lons) - min(lons) > 180.0:
        raise ValueError(f"the area around ({aoi.center_lat}, {aoi.center_lon}) crosses the antimeridian "
                         "(longitude 180), which the builder does not support")
    return AoiGeometry(utm_crs=utm_crs, epsg=epsg, metric_bounds=(float(minx), float(miny), float(maxx), float(maxy)),
                       wgs_bbox={"north": float(max(lats)), "south": float(min(lats)), "east": float(max(lons)),
                                 "west": float(min(lons))})


def aoi_geometry(aoi: Aoi) -> AoiGeometry:
    """The AOI's UTM bounds and WGS 84 bounding box (as the notebook computed them)."""
    return _square(aoi, float(aoi.size_m) / 2.0)


def extract_bbox(aoi: Aoi, margin_m: float = EXTRACT_MARGIN_M) -> Tuple[float, float, float, float]:
    """(west, south, east, north) in WGS 84 of the AOI widened by ``margin_m`` on each
    side: the box to cut an OSM extract for the AOI with, so that ways which cross
    the AOI's edge between nodes outside it are in the extract too."""
    return _square(aoi, float(aoi.size_m) / 2.0 + float(margin_m)).bbox_west_south_east_north


def _empty(crs: Any = None) -> Any:
    import geopandas as gpd

    return gpd.GeoDataFrame(geometry=[], crs=crs)


def _check_date(date: Any) -> None:
    text = str(date)
    try:
        if not _OVERPASS_DATE.match(text):
            raise ValueError
        when = datetime.strptime(text, _DATE_FORMAT).replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"date {date!r}: give an ISO 8601 time in UTC, e.g. 2026-07-21T00:00:00Z") from None
    if when < datetime.strptime(OVERPASS_HISTORY_START, _DATE_FORMAT).replace(tzinfo=timezone.utc):
        raise ValueError(f"date {text}: Overpass keeps the history of OpenStreetMap from {OVERPASS_HISTORY_START} "
                         "on, and gives the data of that time for earlier dates")
    if when > datetime.now(timezone.utc):
        raise ValueError(f"date {text} is in the future")


@contextmanager
def _overpass_settings(date: Optional[str], cache_folder: Optional[Path]) -> Iterator[None]:
    """OSMnx settings for one download: the Overpass ``[date:...]`` of a historical
    query, and OSMnx's cache of the raw responses in ``cache_folder`` or none (OSMnx
    would otherwise keep and reuse responses in ./cache)."""
    import osmnx as ox

    saved = (ox.settings.overpass_settings, ox.settings.use_cache, ox.settings.cache_folder)
    try:
        if date is not None:
            ox.settings.overpass_settings = saved[0] + f'[date:"{date}"]'
        ox.settings.use_cache = cache_folder is not None
        if cache_folder is not None:
            ox.settings.cache_folder = str(cache_folder)
        yield
    finally:
        ox.settings.overpass_settings, ox.settings.use_cache, ox.settings.cache_folder = saved


def _open_binary(path: Path) -> Any:
    """The file, decompressed as OSMnx does it (``.bz2`` and ``.gz``)."""
    if path.suffix == ".bz2":
        return bz2.open(path, "rb")
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _check_osm_xml(path: Path) -> None:
    """Fail with a clear message on files that are not OSM XML (PBF, HTML pages, ...)."""
    from xml.etree.ElementTree import ParseError, iterparse

    try:
        with _open_binary(path) as f:
            head = f.read(1024)
    except (OSError, EOFError) as exc:  # a .bz2 or .gz that is not compressed so
        raise ValueError(f"{path} cannot be read ({exc})") from None
    if b"OSMHeader" in head[:64]:
        raise ValueError(f"{path} is an OSM PBF file; the builder reads OSM XML: convert it with "
                         f"`osmium cat {path.name} -o area.osm`, or cut the area out of it with the osmium command "
                         "that `ruscuav build-aoi --bbox-only` prints")
    if not head.lstrip(b"\xef\xbb\xbf \t\r\n").startswith(b"<"):
        raise ValueError(f"{path} is not an OSM XML file")
    try:
        with _open_binary(path) as f:
            for _, elem in iterparse(f, events=("start",)):
                if elem.tag != "osm":
                    raise ValueError(f"{path} is XML but not OSM XML (its root element is <{elem.tag}>)")
                break
    except ParseError as exc:
        raise ValueError(f"{path} is not a well-formed OSM XML file ({exc})") from None


def _in_layers(tags: Mapping[str, Optional[str]]) -> bool:
    """Whether an element with ``tags`` belongs to one of the layers (OSMnx's rule)."""
    for layer in LAYERS:
        for key, wanted in OSM_TAGS[layer].items():
            if key in tags and (wanted is True or tags[key] in wanted):
                return True
    return False


def scan_osm_file(osm_file: Union[str, Path]) -> Dict[str, Any]:
    """What an OSM XML file (``.osm``, ``.osm.bz2``, ``.osm.gz``) holds: its
    ``bounds`` elements as (west, south, east, north) (exports from openstreetmap.org
    and extracts cut with osmium have one), the extent of its nodes, the numbers of
    nodes, ways and relations, and ``incomplete_multipolygons``: the multipolygon
    relations of the layers' tags that lack member ways in the file. OSMnx leaves
    those out, so their areas are missing from the world; an export from
    openstreetmap.org has only the member ways with a node in its box. Raises
    ValueError for files that are not OSM XML."""
    from xml.etree.ElementTree import ParseError, iterparse

    path = Path(osm_file)
    if not path.is_file():
        raise FileNotFoundError(f"OSM file not found: {path}")
    _check_osm_xml(path)
    bounds: List[List[float]] = []
    lat_lo, lat_hi, lon_lo, lon_hi = math.inf, -math.inf, math.inf, -math.inf
    nodes = relations = 0
    ways: Set[Optional[str]] = set()
    areas: List[List[Optional[str]]] = []
    try:
        with _open_binary(path) as f:
            context = iterparse(f, events=("start", "end"))
            _, root = next(context)  # <osm> (checked above)
            for event, elem in context:
                if event != "end":
                    continue
                if elem.tag == "node":
                    nodes += 1
                    lat, lon = elem.get("lat"), elem.get("lon")
                    if lat is not None and lon is not None:
                        lat_f, lon_f = float(lat), float(lon)
                        lat_lo, lat_hi = min(lat_lo, lat_f), max(lat_hi, lat_f)
                        lon_lo, lon_hi = min(lon_lo, lon_f), max(lon_hi, lon_f)
                elif elem.tag == "way":
                    ways.add(elem.get("id"))
                elif elem.tag == "relation":
                    relations += 1
                    tags = {c.get("k"): c.get("v") for c in elem if c.tag == "tag"}
                    if tags.get("type") in _AREA_RELATIONS and _in_layers(tags):
                        areas.append([c.get("ref") for c in elem if c.tag == "member" and c.get("type") == "way"])
                elif elem.tag == "bounds":
                    try:
                        bounds.append([float(elem.get(k)) for k in ("minlon", "minlat", "maxlon", "maxlat")])
                    except (TypeError, ValueError):
                        pass
                else:
                    continue
                root.clear()  # the element is read: free it
    except (ParseError, StopIteration) as exc:
        raise ValueError(f"{path} is not a well-formed OSM XML file ({str(exc) or 'no elements'})") from None
    extent = [lon_lo, lat_lo, lon_hi, lat_hi] if math.isfinite(lat_lo) else None
    return {"bounds": bounds or None, "nodes_extent": extent, "nodes": nodes, "ways": len(ways),
            "relations": relations, "incomplete_multipolygons": sum(any(r not in ways for r in refs) for refs in areas)}


def file_coverage(scan: Mapping[str, Any], geometry: AoiGeometry) -> Optional[float]:
    """The share of the AOI's WGS 84 bounding box inside the file's bounds, or, for a
    file without bounds, inside the extent of its nodes (from :func:`scan_osm_file`),
    estimated on 101 × 101 points; None for a file with neither."""
    import numpy as np

    boxes = scan.get("bounds") or ([scan["nodes_extent"]] if scan.get("nodes_extent") else [])
    if not boxes:
        return None
    west, south, east, north = geometry.bbox_west_south_east_north
    lon, lat = np.meshgrid(np.linspace(west, east, 101), np.linspace(south, north, 101))
    inside = np.zeros(lon.shape, dtype=bool)
    eps = 1e-6  # about 0.1 m: osmium writes bounds rounded to 1e-7 degrees
    for w, s, e, n in boxes:
        inside |= (lon >= w - eps) & (lon <= e + eps) & (lat >= s - eps) & (lat <= n + eps)
    return float(inside.mean())


def _files(folder: Optional[Path]) -> Set[str]:
    if folder is None or not folder.is_dir():
        return set()
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


def load_osm_layers(aoi: Aoi, osm_file: Union[str, Path, None] = None, *, download: bool = False,
                    date: Optional[str] = None, cache_folder: Union[str, Path, None] = None) -> Dict[str, Any]:
    """The features of the four layers (:data:`LAYERS`) that intersect the AOI's
    WGS 84 bounding box, as GeoDataFrames in EPSG:4326: from an OSM XML file
    (``osm_file``, e.g. exported from openstreetmap.org or extracted with osmium) or,
    with ``download``, from the Overpass API (``date``: the OpenStreetMap data as of
    an ISO 8601 time, e.g. ``2026-07-21T00:00:00Z``; ``cache_folder``: where OSMnx
    keeps the responses, so that the same download can be read again offline;
    without it OSMnx keeps nothing). A layer without features is empty. Needs the
    "gis" extra."""
    return _load_osm_layers(aoi, osm_file, download=download, date=date, cache_folder=cache_folder)[0]


def _load_osm_layers(aoi: Aoi, osm_file: Union[str, Path, None], *, download: bool, date: Optional[str],
                     cache_folder: Union[str, Path, None]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """:func:`load_osm_layers`, and for a download with a cache folder the number of
    responses it saved there (0: every response came from the cache)."""
    from xml.etree.ElementTree import ParseError
    from xml.sax import SAXException

    import osmnx as ox
    from osmnx._errors import InsufficientResponseError

    if (osm_file is None) == (not download):
        raise ValueError("give an OSM XML file or download=True, not both")
    if date is not None and not download:
        raise ValueError("date applies to downloads; an OSM file holds the data of its own date")
    if cache_folder is not None and not download:
        raise ValueError("cache_folder applies to downloads")
    if date is not None:
        _check_date(date)
    bbox = aoi_geometry(aoi).bbox_west_south_east_north
    tags_of = {name: {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in OSM_TAGS[name].items()}
               for name in LAYERS}
    layers: Dict[str, Any] = {}
    info: Dict[str, Any] = {}

    def keep(name: str, gdf: Any) -> None:
        layers[name] = gdf.set_crs("EPSG:4326") if gdf.crs is None else gdf

    if osm_file is not None:
        path = Path(osm_file)
        if not path.is_file():
            raise FileNotFoundError(f"OSM file not found: {path}")
        _check_osm_xml(path)
        for name in LAYERS:
            try:
                gdf = ox.features_from_xml(path, polygon=ox.utils_geo.bbox_to_poly(bbox), tags=tags_of[name])
            except InsufficientResponseError:  # no feature of the layer in the area
                gdf = _empty("EPSG:4326")
            except (ParseError, SAXException) as exc:
                raise ValueError(f"{path} is not a well-formed OSM XML file ({exc})") from None
            keep(name, gdf)
        return layers, info
    folder = None if cache_folder is None else Path(cache_folder)
    before = _files(folder)
    with _overpass_settings(date, folder):
        for name in LAYERS:
            try:
                gdf = ox.features_from_bbox(bbox=bbox, tags=tags_of[name])
            except InsufficientResponseError:
                gdf = _empty("EPSG:4326")
            keep(name, gdf)
    if folder is not None:
        info["responses_saved"] = len(_files(folder) - before)
    return layers, info


def project_clip_to_local(gdf: Any, geometry: AoiGeometry, size_m: float) -> Any:
    """Project features to the AOI's UTM zone, clip them to the AOI and move them into
    the local frame: x = utm_x - min x, y = max y - utm_y (the notebook's
    ``project_clip_to_local``). The result has no CRS."""
    import geopandas as gpd
    import shapely
    from shapely.geometry import box

    if gdf is None or len(gdf) == 0:
        return _empty()
    minx, miny, maxx, maxy = geometry.metric_bounds
    utm_crs = geometry.utm_crs
    try:
        gdf_utm = gdf.to_crs(utm_crs)
    except Exception:  # noqa: BLE001 - as the notebook: a frame without CRS is taken as WGS 84
        gdf_utm = gdf.copy()
        if gdf_utm.crs is None:
            gdf_utm = gdf_utm.set_crs("EPSG:4326")
        gdf_utm = gdf_utm.to_crs(utm_crs)
    aoi_poly_utm = box(minx, miny, maxx, maxy)
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_poly_utm], crs=utm_crs)
    try:
        clipped = gpd.clip(gdf_utm, aoi_gdf)
    except Exception:  # noqa: BLE001
        clipped = gdf_utm[gdf_utm.intersects(aoi_poly_utm)].copy()
        clipped["geometry"] = clipped.geometry.intersection(aoi_poly_utm)
    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notnull()].copy()
    if len(clipped) == 0:
        return _empty()

    def to_local(coords: Any) -> Any:
        # the notebook's shapely.ops.transform, coordinate by coordinate, gives the same numbers
        out = coords.copy()
        out[:, 0] = coords[:, 0] - minx
        out[:, 1] = maxy - coords[:, 1]
        return out

    local = clipped.copy()
    local["geometry"] = local.geometry.apply(lambda geom: shapely.transform(geom, to_local))
    local = local[~local.geometry.is_empty & local.geometry.notnull()].copy()
    local = local.set_crs(None, allow_override=True)
    local_box = box(0.0, 0.0, float(size_m), float(size_m))
    try:
        local["geometry"] = local.geometry.intersection(local_box)
        local = local[~local.geometry.is_empty & local.geometry.notnull()].copy()
    except Exception:  # noqa: BLE001
        pass
    return local
