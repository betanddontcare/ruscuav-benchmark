# Synthetic benchmark worlds

All files follow `ruscuav-world/1.0` (see `docs/WORLD_NPZ_SCHEMA.md`) and pass
`ruscuav validate`. Each is a 1 km × 1 km world at 1 m resolution in the local frame
`LOCAL_METRIC_SYNTHETIC`. The raster layers are bit-for-bit identical to the
RUSCUAV v2.0.0 Zenodo release; `meta_json` in each file records the SHA-256 of the
archived source file.

| World | Path | Role in the paper |
|---|---|---|
| `AOI_SYNTH_000` | `AOI_SYNTH_000/urban_world_layers.npz` | Canonical training and evaluation world (5.6% building cover). |
| `AOI_SYNTH_001` | `AOI_SYNTH_001_003/AOI_SYNTH_001/urban_world_layers.npz` | Morphology-shift world (16.0% building cover). |
| `AOI_SYNTH_002` | `AOI_SYNTH_001_003/AOI_SYNTH_002/urban_world_layers.npz` | Morphology-shift world (15.0% building cover). |
| `AOI_SYNTH_003` | `AOI_SYNTH_001_003/AOI_SYNTH_003/urban_world_layers.npz` | Morphology-shift world (15.9% building cover). |

`AOI_SYNTH_000` was called `AOI_FAKE_001` in v2.0.0 and earlier, where its CRS was
mislabelled `EPSG:2180`. The package resolves the old identifier
(`ruscuav_benchmark.canonical_aoi_id("AOI_FAKE_001") == "AOI_SYNTH_000"`), so
archived results and scenario files that use it still match.

The files were produced from the archive with `scripts/upgrade_world_npz.py`.
