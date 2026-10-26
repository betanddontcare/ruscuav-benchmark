# Real-GIS worlds

This directory contains AOI metadata and the frozen pipeline configuration for Vienna and Warsaw. Processed real-GIS NPZ worlds are intentionally absent. Regenerate them with `ruscuav build-aoi --preset vienna` or `--preset warsaw` (`docs/BUILDING_WORLDS.md`), the package's port of the real-GIS blocks in `source_code/top_uav_submission_pipeline.py` and the legacy/full Colab export, whose parameters `real_gis_pipeline_v0.yaml` and the archived metadata record.

Cross-AOI comparisons in the paper use the explicit-footway interpretation. The conservative road-buffer pedestrian proxy is retained only as a budget-calibration diagnostic.

The missions of the v2.0.0 primary Vienna evaluations (seeds 7000–7199) are recorded in `scenarios/real_gis/`. The JSON file next to the CSV gives the layers checksum of the world they were sampled in (`UrbanWorldLayers.layers_sha256()`, see `docs/WORLD_NPZ_SCHEMA.md`). OpenStreetMap data change over time, so a rebuilt world can have another checksum and sample other missions for the same seeds. The Warsaw footway-only world of the archived runs was never saved, so its missions are not recorded. Details: `docs/REPRODUCIBILITY.md`.
