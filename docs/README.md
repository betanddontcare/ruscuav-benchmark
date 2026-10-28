# Documentation

The benchmark is defined in `BENCHMARK_SPEC.md` (repository root). The guides:

## Worlds

| Guide | Contents |
|---|---|
| [WORLD_NPZ_SCHEMA.md](WORLD_NPZ_SCHEMA.md) | the world file format, `ruscuav validate` and its rules, the layers checksum |
| [BUILDING_WORLDS.md](BUILDING_WORLDS.md) | worlds of any area from OpenStreetMap: `ruscuav build-aoi`, the layers and their rules, DEM and DSM, the paper's Vienna and Warsaw areas |
| [OSM_ATTRIBUTION.md](OSM_ATTRIBUTION.md) | OpenStreetMap's licence for worlds built from its data |

## Methods and evaluation

| Guide | Contents |
|---|---|
| [ENVIRONMENT.md](ENVIRONMENT.md) | the Gymnasium environment: method presets, curriculum stages, regimes, configuration, observations, rewards and shields |
| [PLANNERS.md](PLANNERS.md) | the risk-cost A* planners of the paper and `ruscuav evaluate --planner` |
| [POLICIES.md](POLICIES.md) | the PPO policies: the archived checkpoints, evaluation, training |
| [PROTOCOLS.md](PROTOCOLS.md) | evaluation protocols: the paper's four, your own (`ruscuav protocol create`) |
| [TRAJECTORIES.md](TRAJECTORIES.md) | scoring the trajectories of any method (`ruscuav evaluate --trajectories`) |
| [SCORING.md](SCORING.md) | `ruscuav score`: run summaries, training-seed intervals, causes of non-acceptability, paired tests with Holm's correction |

## The paper and its archive

| Guide | Contents |
|---|---|
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | the archive of RUSCUAV v2.0.0, how the package reproduces it and with what results |
| [TABLE_FIGURE_ARTIFACT_MAP.md](TABLE_FIGURE_ARTIFACT_MAP.md) | the archive files behind the paper's tables and figures |
| [DATASET_CARD.md](DATASET_CARD.md) | the archive's intended uses and limitations |
| [ZENODO_UPLOAD.md](ZENODO_UPLOAD.md) | how the v2.0.0 archive was uploaded to Zenodo |
| [SOFTWAREX_CODE_METADATA.md](SOFTWAREX_CODE_METADATA.md) | the code metadata of the package, as SoftwareX asks for it |

`configs/README.md` describes the configurations of the archived experiments and
`scripts/README.md` the scripts; `CHANGELOG.md` lists the changes of each version.
