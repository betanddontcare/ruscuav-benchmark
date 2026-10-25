# World NPZ schema: `ruscuav-world/1.0`

A RUSCUAV world is a NumPy `.npz` archive of aligned rasters. This page is the
contract for world files, including worlds you build for your own city. Check a
file with:

```bash
ruscuav validate my_city.npz          # human-readable report, exit code 1 if invalid
ruscuav validate my_city.npz --json   # machine-readable report
```

Files are always opened with `allow_pickle=False`. A world file can never execute
code when loaded, and files containing pickled Python objects are rejected.

## Metadata

| Key | Type | Meaning |
|---|---|---|
| `schema_version` | string scalar | `"ruscuav-world/1.0"`. Files without it are read with the 1.0 rules and get a warning. |
| `aoi_id` | string scalar | World identifier, e.g. `AOI_SYNTH_000`. |
| `crs` | string scalar | Label of the coordinate frame of `bbox`. Synthetic worlds use `LOCAL_METRIC_SYNTHETIC`; local frames built from GIS data use labels such as `LOCAL_RUSCUAV_0_1000`. |
| `bbox` | float, shape (4,) | `(minx, miny, maxx, maxy)` in metres in that frame. |
| `resolution_m` | float scalar | Cell size in metres. |
| `meta_json` | string scalar, optional | JSON object with provenance (source, generator, notes). |

The extent must match the grid: `maxx - minx = width × resolution_m` and
`maxy - miny = height × resolution_m`. Row index is local y, column index is local x.

## Layers

All layers are 2-D, have the shape of `terrain_z`, and contain only finite numbers.
Heights are in metres.

| Key | Values | Meaning |
|---|---|---|
| `terrain_z` | float | Terrain elevation. |
| `surface_z` | float | Surface elevation (top of buildings and canopy). |
| `height_above_ground` | float | `surface_z - terrain_z`. |
| `building_mask` | 0/1 | Buildings (semantic class and physical obstacle). |
| `building_height` | float ≥ 0 | Building height; 0 outside buildings. |
| `building_roof_z` | float | `terrain_z + building_height` on building cells. |
| `road_mask` | 0/1 | Drivable roads (exposure class). |
| `sidewalk_mask` | 0/1 | Pedestrian ways (exposure class). |
| `green_mask` | 0/1 | Green areas. |
| `canopy_mask` | 0/1 | Tree canopy (physical obstacle), inside green areas. |
| `canopy_height` | float ≥ 0 | Canopy height; 0 outside canopy. |
| `canopy_top_z` | float | `terrain_z + canopy_height` on canopy cells. |
| `canopy_density` | [0, 1] | Canopy density; 0 outside canopy. |
| `sensitive_mask` | 0/1 | Sensitive zones; an overlay that may overlap other classes. |
| `water_mask`, `railway_mask`, `parking_mask` | 0/1, optional | Extra classes; unused by the current risk model. |

Masks should be binary. Fractional values are allowed but thresholded at 0.5 on load.

## Rules checked by `ruscuav validate`

Errors (the world cannot be used):

- unreadable or truncated archives, members that are not NumPy arrays, pickled
  objects, missing required keys, unsupported `schema_version`;
- metadata that are not 0-d arrays of the right kind (`aoi_id`, `crs`,
  `schema_version` and `meta_json` strings; `resolution_m` a number), a `bbox` that is
  not 4 numbers, or a `meta_json` that is not a JSON object;
- layers that are not 2-D, are empty, differ in shape, are non-numeric, or contain NaN/inf;
- masks outside [0, 1]; `bbox` or `resolution_m` invalid or inconsistent with the grid;
- derived layers that disagree with their base layers by more than 1 mm
  (`height_above_ground`, `building_roof_z`, `canopy_top_z`); `canopy_density` outside [0, 1];
- overlapping classes that must be disjoint: building, road, sidewalk and green are
  mutually exclusive, and canopy must lie inside green areas;
- height sanity: surface below terrain, negative heights, buildings or canopy
  without height.

Warnings (legal but worth checking):

- no `schema_version`; unknown extra keys;
- non-binary masks; `canopy_density` set outside canopy;
- `surface_z` different from max(terrain, roofs, canopy tops), which is expected only
  for DSM-based surfaces;
- a `bbox` starting at (0, 0) combined with an `EPSG:` CRS label;
- grids other than 1000 × 1000 at 1 m, on which the environment and planners are
  not yet validated.

## Bundled worlds

See `worlds/synthetic/README.md`. Processed Vienna and Warsaw worlds are not
distributed (OpenStreetMap ODbL); see `docs/OSM_ATTRIBUTION.md`.
