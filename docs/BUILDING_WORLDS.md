# Building worlds from OpenStreetMap

`ruscuav build-aoi` builds a world NPZ (docs/WORLD_NPZ_SCHEMA.md) from the
OpenStreetMap features of a square area, with the pipeline of the paper's real-GIS
worlds: buildings with heights, roads, sidewalks, green areas, canopy, sensitive
zones and terrain. It validates the world and, if asked, records a protocol for it
(docs/PROTOCOLS.md), so that any city can be benchmarked as the paper's worlds were.
It needs the `gis` extra (and the `env` extra for the protocol):

```bash
pip install "ruscuav-benchmark[gis,env]"

# from an OpenStreetMap XML file of the area
ruscuav build-aoi --center 48.2152,16.3975 --aoi-id my_area --osm area.osm \
    --output worlds/my_area.npz --protocol protocols/my_area.json
ruscuav evaluate --protocol protocols/my_area.json --planner FULL_RH_Astar --output-dir runs/my_area

# downloaded from the Overpass API (needs internet access)
ruscuav build-aoi --center -33.9249,18.4241 --aoi-id cape_town --download --output worlds/cape_town.npz

# the area of the paper's Vienna world, with the OpenStreetMap data of a date
ruscuav build-aoi --preset vienna --download --osm-date 2026-07-15T00:00:00Z --cache osm_cache \
    --output worlds/vienna.npz
```

```python
from ruscuav_benchmark.gis import Aoi, build_world

built = build_world(Aoi(48.2152, 16.3975), "my_area", osm_file="area.osm")
print(built.warnings)
built.world.save_npz("worlds/my_area.npz")
```

Worlds built from OpenStreetMap are derived databases under the Open Database
License; see "Licence" below.

## The area

The area is a square of `--size` metres (default 1000) around `--center`
(latitude, longitude in WGS 84; a southern latitude is written as is,
`--center -33.9249,18.4241`), in the UTM zone of the centre, rasterized at
`--resolution` metres (default 1). The environment and the planners are validated on
1000 × 1000 grids at 1 m; other grids work but `ruscuav validate` warns. In the
world, x runs eastward and y southward from the north-west corner of the square.
Areas across the antimeridian (longitude 180) are not supported.

The features are those of the four layers that intersect the WGS 84 bounding box
of the square's corners, clipped to the square before they are buffered: features
outside the square add nothing (the 55 m buffer of a hospital just outside does not
reach in). `ruscuav build-aoi --center ... --bbox-only` prints the square's UTM
bounds and bounding box, and the osmium command that cuts an extract for it, with a
margin of 250 m so that ways which cross the square's edge between nodes outside it
are in the extract too.

## OpenStreetMap data

- `--osm FILE`: an OSM XML file (`.osm`, `.osm.bz2` or `.osm.gz`) that covers the
  area: cut from a regional extract (for example from Geofabrik) with
  [osmium](https://osmcode.org/osmium-tool/),
  `osmium extract -s smart -b WEST,SOUTH,EAST,NORTH region.osm.pbf -o area.osm`
  (the command `--bbox-only` prints), or exported from openstreetmap.org (Export,
  for small areas). The `smart` strategy keeps multipolygons whole; an export from
  openstreetmap.org has only the member ways that have a node in its box, and OSMnx
  leaves out a multipolygon with members missing, so large parks, forests and
  buildings with courtyards that cross the export's edge are lost. PBF files are not
  read: convert one with `osmium cat area.osm.pbf -o area.osm`. The file's SHA-256 is
  recorded in the world.
- `--download`: the features are downloaded from the Overpass API with OSMnx, as the
  notebook downloaded them. `--osm-date 2026-07-15T00:00:00Z` asks for the
  OpenStreetMap data as of that time (Overpass keeps the history from
  2012-09-12T06:55:00Z on), so a world can be built again from the same data later.
  `--cache DIR` keeps OSMnx's copies of the responses: the same command reads them
  again offline and gives the same world, and the world records whether its
  responses came from the cache. Without `--cache` nothing is kept and every
  download asks Overpass (OSMnx's own default would keep and reuse responses in
  `./cache`).

The builder checks an OSM file before it reads the features. It refuses a file that
covers none of the area (a file of another place, or a wrong centre) and warns when
the file covers only part of it (by the file's `bounds`, or, for a file without
them, by the extent of its nodes), when multipolygons of the layers lack member ways
in the file, and when the area has no features of the four layers. The warnings come
first in the output of `build-aoi` and are kept in the world's metadata.

## The layers

| Layer | From OpenStreetMap | Rule |
|---|---|---|
| buildings | `building=*` areas | height from `height` or `building:height` (3–150 m), else `building:levels` or `levels` × 3.2 m (3–150 m), else by type: 18 m (commercial, retail, office), 10 m (industrial, warehouse), 25 m (churches), 16 m (schools, hospitals, universities), 12 m (all others) |
| roads | `highway=` motorway … track | buffered to a width by class, 28 m (motorway) to 4 m (track) |
| sidewalks | `highway=` footway, path, pedestrian, steps, crossing; `footway=sidewalk` or `crossing` | footways buffered to 3 m; the `proxy` variant adds 4 m along each side of every road |
| green areas | `leisure=` park, garden, recreation ground, playground; `landuse=` grass, forest, meadow, recreation ground, village green, cemetery; `natural=` wood, grassland, scrub, heath | areas |
| canopy | green areas that are forest, wood, scrub, park or garden | 18 m (forest, wood), 6 m (scrub), 12 m (park, garden); density 0.75 |
| sensitive zones | `amenity=` school, kindergarten, hospital, clinic, doctors, university, college, social facility, nursing home; `healthcare=*`; `leisure=playground` | points and lines buffered by 55 m (health care), 45 m (education), 35 m (others); areas by 10 m |

Rasterization marks every cell a feature touches; where features overlap, the later
one gives the value. No green area, canopy or sidewalk lies under a building, and no
sidewalk on a road. Roads may cross buildings (passages) and green areas, and
footways green areas: such cells count as both classes (docs/WORLD_NPZ_SCHEMA.md).

`--sidewalks` chooses the sidewalk layer: `footway-only` (default), the footways
minus roads and buildings, as in the worlds of the paper's real-GIS evaluations (the
world's identifier gets the suffix `_footway_only`); or `proxy`, the footways and a
4 m proxy along roads, as the notebook's base worlds.

## Terrain and surface

Without a DEM the terrain is the notebook's smooth fallback surface, 18 m +
1.5 sin(x/180) + cos(y/220) with x and y in metres from the north-west corner
(15.5–20.5 m): it is not real terrain, and the world's metadata says so. The paper's
Vienna and Warsaw worlds were built with it (no DEM was given). `--dem FILE` takes the
terrain from a GeoTIFF in the CRS it records (a file without one is refused),
reprojected bilinearly onto the grid; cells without data get the median, and the
builder warns and records how much of the area the DEM covers. The environment flies
between 0 and 150 m, so by default the DEM's heights are lowered by the whole metres
below the area's lowest point (`--terrain-datum relative`; the offset is recorded);
`--terrain-datum absolute` keeps them, and `ruscuav validate` refuses a world whose
lowest terrain is above 148 m (the environment flies at least 2 m above the ground).
`--dsm FILE`, with a DEM, gives the surface (it is lowered by the same offset).

Roofs and canopy tops are the terrain plus the heights; the surface is the highest
of terrain, roofs, canopy tops and the DSM. It is computed before canopy under
buildings is removed, so where a building replaced canopy the surface keeps the
canopy top, as in the paper's worlds (`ruscuav validate` notes it).

## The world file

The NPZ has the schema `ruscuav-world/1.0` and its provenance in `meta_json`: the
area (centre, size, resolution, UTM zone and bounds, WGS 84 box), the OpenStreetMap
source (the file, its SHA-256, its bounds and how much of the area it covers; or the
download, its date and the cache), the number of features of each layer in the
area, the tags, the terrain (DEM and DSM files, SHA-256 and coverage, datum,
offset), the sidewalk variant, the layer coverages, the versions of the libraries,
the builder's warnings and the attribution. `build-aoi` writes the world to
`--output` (adding `.npz` if the name lacks it) and prints the world's layers
checksum (`layers_sha256`), its coverages, the builder's warnings and the report of
`ruscuav validate`; it exits with 0 for a valid world, 1 for an invalid one and 2 for
an error (the options and input files are checked before the world is written). With
`--protocol FILE` it records a protocol of `--episodes` missions from seed `--seed0`
(docs/PROTOCOLS.md), with the world file relative to the protocol file.

`ruscuav_benchmark.gis.check_builder_rules(world)` counts the cells of a world that
break the rules above (no sidewalk on roads or buildings, heights only on their
masks and from the rules' values, derived layers, surface); for a world built
without a DSM all counts are 0, and with `dsm=True` also for one built with a DSM.

## The paper's worlds

`--preset vienna` and `--preset warsaw` give the areas of the paper's worlds: the
Vienna Prater/Leopoldstadt and the Warsaw city-centre squares of 1 km at 1 m, with
their identifiers (with `--center`, `--size` or `--resolution` the area is another
one, so `--aoi-id` is needed).

The pipeline is the notebook's `build_real_world_for_aoi`, which built the Vienna
world, with the footway-only variant of its Vienna cells. On the same OpenStreetMap
data, on the notebook's grid of 1 km at 1 m and with its terrain (no DEM, or a DEM
with `--terrain-datum absolute`), the package and the notebook give the same world,
every layer bit for bit (`scripts/migration/gis_parity.py`, docs/REPRODUCIBILITY.md).
Where the notebook would fall back to its synthetic terrain, for a DEM that is
missing or has no data in the area and for a DSM without a DEM, the package refuses
the input. The archived Vienna world keeps every rule of the builder, its terrain is
the fallback surface bit for bit, and its UTM bounds and bounding box are those of
`--preset vienna`.

The Vienna world was built from the OpenStreetMap data of July 2026, downloaded at
the time; the copy the paper's primary Vienna evaluations used is a later build than
the one of 1–2 July (docs/REPRODUCIBILITY.md, "Real-GIS worlds") and has the layers
checksum `fa99c571…` recorded with its missions in `scenarios/real_gis/`. A build with
`--download --osm-date` at a time of that month, and with the library versions of
`requirements-reproducibility.txt` (other GEOS and GDAL versions could change buffers
and rasterization in the last decimals), reproduces it if OpenStreetMap did not change
in the area between that time and the download; its `layers_sha256` says whether it
does. Warsaw was built step by step in the notebook; its footway-only world was never
saved.

## Licence

OpenStreetMap data are © OpenStreetMap contributors and available under the Open
Database License (ODbL 1.0). A world built from them is a derived database: it
carries the attribution in its metadata, and sharing it means sharing it under the
ODbL with that attribution (docs/OSM_ATTRIBUTION.md). The repository does not
distribute the processed Vienna and Warsaw worlds.
