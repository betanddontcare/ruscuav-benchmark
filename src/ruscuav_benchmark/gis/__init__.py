"""Worlds from OpenStreetMap (extra "gis": OSMnx, GeoPandas, Shapely, Rasterio, PyProj).

``ruscuav build-aoi`` and :func:`build_world` turn the OpenStreetMap features of a
square area into a world NPZ with the pipeline of the paper's real-GIS worlds:
buildings with heights, roads, sidewalks (footway-only by default, as in the
paper's evaluations), green areas, canopy, sensitive zones and terrain
(docs/BUILDING_WORLDS.md). The features come from an OSM XML file or from the
Overpass API (optionally as of a date); a DEM and a DSM are optional.

>>> from ruscuav_benchmark.gis import Aoi, build_world      # doctest: +SKIP
>>> built = build_world(Aoi(48.2152, 16.3975), "my_area", osm_file="area.osm")  # doctest: +SKIP
>>> built.world.save_npz("my_area.npz")                       # doctest: +SKIP
"""

from .build import (ATTRIBUTION, SIDEWALK_VARIANTS, TERRAIN_DATUMS, BuiltWorld, build_world, check_builder_rules,
                    rasterize_world)
from .osm import (EXTRACT_MARGIN_M, LAYERS, OSM_TAGS, OVERPASS_HISTORY_START, PAPER_AOIS, Aoi, AoiGeometry,
                  aoi_geometry, extract_bbox, file_coverage, load_osm_layers, project_clip_to_local, scan_osm_file)

__all__ = [
    "Aoi", "AoiGeometry", "aoi_geometry", "extract_bbox", "EXTRACT_MARGIN_M", "OSM_TAGS", "LAYERS", "PAPER_AOIS",
    "OVERPASS_HISTORY_START", "load_osm_layers", "scan_osm_file", "file_coverage", "project_clip_to_local",
    "SIDEWALK_VARIANTS", "TERRAIN_DATUMS", "ATTRIBUTION", "BuiltWorld", "build_world", "rasterize_world",
    "check_builder_rules",
]
