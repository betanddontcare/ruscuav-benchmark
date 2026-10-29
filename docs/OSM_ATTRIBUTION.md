# OpenStreetMap attribution and real-GIS release scope

Vienna and Warsaw semantic layers are generated from OpenStreetMap-derived vector features. Any regenerated maps or derived databases must retain the attribution:

> © OpenStreetMap contributors

OpenStreetMap data are licensed under the Open Data Commons Open Database License (ODbL). Consult the current OpenStreetMap copyright and ODbL guidance before redistributing regenerated databases.

## v2.0.0 scope

This release includes real-GIS scripts, query/tag logic, AOI metadata, benchmark configurations and evaluation results. It intentionally excludes the processed Vienna and Warsaw `UrbanWorldLayers` NPZ files. This choice avoids treating a transformed OSM-derived database as if it were solely an author-created CC BY asset.

## Since 2.1

`scenarios/real_gis/` contains the start and target points of the archived primary
Vienna missions. They were sampled in the processed world and checked against its
layers, so their JSON files carry the attribution above, and so does the description
of the package's `vienna-primary` protocol, which holds the same missions
(`src/ruscuav_benchmark/protocols/vienna-primary.json`). The worlds themselves remain
excluded from the repository; the Vienna world is published apart, as a dataset under
the ODbL with the attribution above, with the release (docs/RELEASING.md).

## Worlds you build

`ruscuav build-aoi` (docs/BUILDING_WORLDS.md) builds worlds from OpenStreetMap
data. Such a world is a database derived from OpenStreetMap: its metadata
(`meta_json`) carries the attribution above and records the data it was built from.
Keep the attribution when you publish results, figures or maps made from it, and
share the world itself only under the ODbL.
