"""A synthetic OpenStreetMap extract for the tests of ruscuav_benchmark.gis.

:func:`synthetic_osm_xml` writes an OSM XML file of an invented town in the area
of interest: a street grid of every highway class (roads, footways, sidewalks,
crossings, paths, cycleways), buildings with every kind of height tag (height with
units and lists, levels, building types, none), multipolygon buildings with
courtyards, invalid and unclosed footprints, green areas of every class with nested
canopy, and sensitive facilities as points, lines and areas, some across the edge of
the area. :func:`margin_osm_xml` writes an area whose green areas and sensitive
facilities lie only between its bounding box and its square, so that they are read
and clipped away. The data are made up, so they carry no OpenStreetMap licence.
Needs pyproj (and shapely for :func:`margin_osm_xml`).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Union
from xml.sax.saxutils import quoteattr

__all__ = ["synthetic_osm_xml", "margin_osm_xml"]

_ROAD_CLASSES = ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
                 "living_street", "service", "road", "track")
_OTHER_HIGHWAYS = ("cycleway", "bridleway", "corridor", "platform", "construction", "proposed", "bus_stop")
_BUILDING_TYPES = ("yes", "apartments", "residential", "house", "detached", "terrace", "commercial", "retail",
                   "office", "industrial", "warehouse", "church", "cathedral", "school", "hospital", "university",
                   "garage")
_HEIGHT_TAGS = (
    {"height": "15"}, {"height": "12 m"}, {"height": "12,5"}, {"height": "7;9"}, {"height": "tall"},
    {"height": "200"}, {"height": "1"}, {"building:height": "21.5 metres"}, {"building:levels": "4"},
    {"building:levels": "2.5"}, {"building:levels": "60"}, {"levels": "3"}, {"building:levels": "-1"},
    {"height": "0", "building:levels": "5"}, {},
)
_GREEN = ({"leisure": "park"}, {"leisure": "garden"}, {"leisure": "recreation_ground"}, {"landuse": "grass"},
          {"landuse": "forest"}, {"landuse": "meadow"}, {"landuse": "recreation_ground"},
          {"landuse": "village_green"}, {"landuse": "cemetery"}, {"natural": "wood"}, {"natural": "grassland"},
          {"natural": "scrub"}, {"natural": "heath"})
_SENSITIVE_POINTS = ({"amenity": "kindergarten"}, {"amenity": "clinic"}, {"amenity": "doctors"},
                     {"amenity": "nursing_home"}, {"amenity": "social_facility"}, {"healthcare": "pharmacy"},
                     {"amenity": "college"})
_SENSITIVE_AREAS = ({"amenity": "school"}, {"amenity": "hospital", "healthcare": "hospital"},
                    {"amenity": "university"}, {"leisure": "playground"})


class _Osm:
    def __init__(self, to_wgs, minx: float, maxy: float) -> None:
        self.to_wgs, self.minx, self.maxy = to_wgs, minx, maxy
        self.nodes: List[str] = []
        self.ways: List[str] = []
        self.relations: List[str] = []
        self.next_id = 1

    def _id(self) -> int:
        self.next_id += 1
        return self.next_id

    def node(self, x: float, y: float, tags: Dict[str, str] = None) -> int:
        lon, lat = self.to_wgs.transform(self.minx + x, self.maxy - y)
        nid = self._id()
        body = "".join(f"<tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in (tags or {}).items())
        self.nodes.append(f'<node id="{nid}" lat="{lat:.9f}" lon="{lon:.9f}" version="1">{body}</node>')
        return nid

    def way(self, points: Sequence[Tuple[float, float]], tags: Dict[str, str], closed: bool = False) -> int:
        refs = [self.node(x, y) for x, y in points]
        if closed:
            refs.append(refs[0])
        wid = self._id()
        body = "".join(f'<nd ref="{r}"/>' for r in refs)
        body += "".join(f"<tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items())
        self.ways.append(f'<way id="{wid}" version="1">{body}</way>')
        return wid

    def multipolygon(self, outer: Sequence[Tuple[float, float]], inners: Sequence[Sequence[Tuple[float, float]]],
                     tags: Dict[str, str]) -> int:
        members = [("outer", self.way(outer, {}, closed=True))]
        members += [("inner", self.way(ring, {}, closed=True)) for ring in inners]
        rid = self._id()
        body = "".join(f'<member type="way" ref="{ref}" role="{role}"/>' for role, ref in members)
        tags = {"type": "multipolygon", **tags}
        body += "".join(f"<tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in tags.items())
        self.relations.append(f'<relation id="{rid}" version="1">{body}</relation>')
        return rid

    def xml(self, bounds: Tuple[float, float, float, float]) -> str:
        west, south, east, north = bounds
        head = (f'<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="ruscuav-tests">\n'
                f'<bounds minlat="{south:.9f}" minlon="{west:.9f}" maxlat="{north:.9f}" maxlon="{east:.9f}"/>\n')
        return head + "\n".join(self.nodes + self.ways + self.relations) + "\n</osm>\n"


def _rect(x: float, y: float, w: float, h: float) -> List[Tuple[float, float]]:
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def synthetic_osm_xml(path: Union[str, Path], center_lat: float, center_lon: float, size_m: float = 1000.0,
                      seed: int = 0) -> Path:
    """Write the synthetic extract of the AOI centred on (``center_lat``,
    ``center_lon``) to ``path``; the features extend 60 m beyond the AOI."""
    import pyproj

    utm = _utm(center_lat, center_lon)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_wgs = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(center_lon, center_lat)
    half = size_m / 2.0
    osm = _Osm(to_wgs, cx - half, cy + half)
    rng = random.Random(seed)
    lo, hi = -60.0, size_m + 60.0

    # streets: a grid with every class; sidewalks and crossings along some of them
    step = size_m / 8.0
    k = 0
    for i in range(9):
        pos = i * step + rng.uniform(-8.0, 8.0)
        for horizontal in (True, False):
            klass = (_ROAD_CLASSES + _OTHER_HIGHWAYS)[k % (len(_ROAD_CLASSES) + len(_OTHER_HIGHWAYS))]
            k += 1
            line = [(lo, pos), (size_m * 0.5, pos + rng.uniform(-5, 5)), (hi, pos)] if horizontal else \
                [(pos, lo), (pos + rng.uniform(-5, 5), size_m * 0.5), (pos, hi)]
            osm.way(line, {"highway": klass, "name": f"street {k}"})
            if k % 3 == 0:
                off = 9.0
                side = [(x, y + off) for x, y in line] if horizontal else [(x + off, y) for x, y in line]
                osm.way(side, {"highway": "footway", "footway": "sidewalk"})
            if k % 4 == 0:
                c = size_m * (0.2 + 0.6 * rng.random())
                cross = [(c, pos - 12.0), (c, pos + 12.0)] if horizontal else [(pos - 12.0, c), (pos + 12.0, c)]
                osm.way(cross, {"highway": "footway", "footway": "crossing"})
    for tags in ({"highway": "pedestrian"}, {"highway": "steps"}, {"highway": "path"}, {"highway": "crossing"},
                 {"highway": "service", "footway": "sidewalk"}, {"highway": "footway", "area": "yes"}):
        x0, y0 = rng.uniform(50, size_m - 150), rng.uniform(50, size_m - 150)
        if tags.get("area") == "yes":
            osm.way(_rect(x0, y0, 40.0, 25.0), tags, closed=True)
        else:
            osm.way([(x0, y0), (x0 + 60.0, y0 + rng.uniform(-30, 30)), (x0 + 120.0, y0 + 10.0)], tags)
    osm.node(rng.uniform(100, 900), rng.uniform(100, 900), {"highway": "bus_stop"})

    # buildings in the blocks
    for bi in range(8):
        for bj in range(8):
            if rng.random() < 0.25:
                continue
            x, y = bi * step + 16.0, bj * step + 16.0
            for _ in range(rng.randint(1, 4)):
                w, h = rng.uniform(8.0, 45.0), rng.uniform(8.0, 40.0)
                bx, by = x + rng.uniform(0, max(1.0, step - 32 - w)), y + rng.uniform(0, max(1.0, step - 32 - h))
                tags = {"building": rng.choice(_BUILDING_TYPES), **rng.choice(_HEIGHT_TAGS)}
                osm.way(_rect(bx, by, w, h), tags, closed=True)
    osm.multipolygon(_rect(300.0, 300.0, 60.0, 50.0), [_rect(320.0, 315.0, 20.0, 20.0)],
                     {"building": "apartments", "building:levels": "6"})
    osm.multipolygon(_rect(610.0, 420.0, 40.0, 40.0), [], {"building": "yes", "height": "33"})
    osm.way([(700.0, 700.0), (740.0, 740.0), (740.0, 700.0), (700.0, 740.0)], {"building": "yes"}, closed=True)
    osm.way([(150.0, 820.0), (190.0, 820.0), (190.0, 850.0)], {"building": "yes"})  # not closed: a line
    osm.node(480.0, 260.0, {"building": "yes"})  # a point
    osm.way(_rect(size_m - 20.0, 450.0, 50.0, 30.0), {"building": "office", "height": "40"}, closed=True)  # edge
    osm.way(_rect(450.0, -25.0, 30.0, 40.0), {"building": "retail"}, closed=True)  # edge

    # green areas, with canopy inside some of them
    for i, tags in enumerate(_GREEN):
        x, y = rng.uniform(-30, size_m - 120), rng.uniform(-30, size_m - 120)
        w, h = rng.uniform(40.0, 160.0), rng.uniform(40.0, 160.0)
        osm.way(_rect(x, y, w, h), dict(tags), closed=True)
        if i % 4 == 0:
            osm.way(_rect(x + w * 0.25, y + h * 0.25, w * 0.4, h * 0.4), {"landuse": "forest"}, closed=True)
    osm.multipolygon(_rect(520.0, 560.0, 140.0, 120.0), [_rect(560.0, 590.0, 30.0, 30.0)], {"leisure": "park"})
    osm.way([(50.0, 50.0), (90.0, 60.0)], {"natural": "wood"})  # a line: not an area

    # sensitive facilities: points, areas and a line
    for tags in _SENSITIVE_POINTS:
        osm.node(rng.uniform(40, size_m - 40), rng.uniform(40, size_m - 40), dict(tags))
    for tags in _SENSITIVE_AREAS:
        x, y = rng.uniform(40, size_m - 140), rng.uniform(40, size_m - 140)
        osm.way(_rect(x, y, rng.uniform(30, 90), rng.uniform(30, 90)), dict(tags), closed=True)
    osm.way([(820.0, 120.0), (880.0, 160.0), (900.0, 220.0)], {"amenity": "school"})  # unclosed: a line
    osm.node(size_m + 20.0, 300.0, {"amenity": "hospital"})  # outside: clipped away before its buffer

    south, north, west, east = 90.0, -90.0, 180.0, -180.0
    for x, y in ((lo, lo), (lo, hi), (hi, lo), (hi, hi)):
        lon, lat = to_wgs.transform(osm.minx + x, osm.maxy - y)
        south, north, west, east = min(south, lat), max(north, lat), min(west, lon), max(east, lon)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(osm.xml((west, south, east, north)), encoding="utf-8")
    return path


def _utm(center_lat: float, center_lon: float) -> str:
    zone = min(int((center_lon + 180.0) // 6.0) + 1, 60)
    return f"EPSG:{32600 + zone if center_lat >= 0 else 32700 + zone}"


def margin_osm_xml(path: Union[str, Path], center_lat: float, center_lon: float, size_m: float = 1000.0) -> Path:
    """Write an area with a building and a road, and with a park and a clinic only in
    the margin between the area's WGS 84 bounding box and its square (the bounding
    box of a square in UTM is larger than it): the park and the clinic are read from
    the file and clipped away, so those layers are empty."""
    import pyproj
    from shapely.geometry import Polygon, box

    utm = _utm(center_lat, center_lon)
    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_wgs = pyproj.Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    cx, cy = to_utm.transform(center_lon, center_lat)
    half = size_m / 2.0
    minx, maxy = cx - half, cy + half
    corners = [to_wgs.transform(x, y) for x, y in ((cx - half, cy - half), (cx - half, cy + half),
                                                   (cx + half, cy - half), (cx + half, cy + half))]
    west, east = min(c[0] for c in corners), max(c[0] for c in corners)
    south, north = min(c[1] for c in corners), max(c[1] for c in corners)
    ring = []
    for lon, lat in ((west, south), (east, south), (east, north), (west, north)):
        x, y = to_utm.transform(lon, lat)
        ring.append((x - minx, maxy - y))
    point = Polygon(ring).difference(box(0.0, 0.0, size_m, size_m)).representative_point()
    osm = _Osm(to_wgs, minx, maxy)
    osm.way(_rect(200.0, 200.0, 30.0, 30.0), {"building": "yes"}, closed=True)
    osm.way([(0.0, size_m / 2.0), (size_m, size_m / 2.0)], {"highway": "primary"})
    osm.way(_rect(point.x - 1.0, point.y - 1.0, 2.0, 2.0), {"leisure": "park"}, closed=True)
    osm.node(point.x, point.y, {"amenity": "clinic"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(osm.xml((west - 0.01, south - 0.01, east + 0.01, north + 0.01)), encoding="utf-8")
    return path
