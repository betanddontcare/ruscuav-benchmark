# Changelog

## Unreleased (2.1.0.dev0)

- Legacy notebook code runs outside Colab: `ruscuav_benchmark.legacy.runtime` and
  `scripts/legacy_parity.py`, which reproduces archived primary episodes
  (see docs/REPRODUCIBILITY.md).
- World files are opened with `allow_pickle=False`; worlds from untrusted sources can
  no longer execute code when loaded.
- World NPZ schema `ruscuav-world/1.0` (docs/WORLD_NPZ_SCHEMA.md): `schema_version` and
  optional `meta_json` provenance; `ruscuav validate` checks files against it.
- New `ruscuav` command (`validate`, `evaluate`); `ruscuav-evaluate` is `ruscuav evaluate`.
- The canonical world `AOI_FAKE_001` is renamed `AOI_SYNTH_000` and its CRS label
  corrected from `EPSG:2180` to `LOCAL_METRIC_SYNTHETIC`; the old ID still resolves.
  The four synthetic worlds are now part of the repository (layers unchanged).
- Canonical-world missions for seeds 5000–5199 recorded in
  `scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv`.
- The quickstart example runs from a repository checkout.
- One package version, read from `ruscuav_benchmark.__version__`.
- New Gymnasium environment `ruscuav_benchmark.env.RuscuavEnv` (extra `env`), one
  configurable class in place of the six notebook environment classes: method
  presets of the paper (GEO-RL … DYN-OBS-RL), stages A/B/C, the primary and secondary
  regimes, and separate modules for dynamics, shields, cargo, LiDAR, reward,
  observations, missions and dynamic risk. It reproduces the notebook environments
  step for step (docs/ENVIRONMENT.md). `ruscuav_benchmark.env.legacy_factories`
  provides the notebook factory names on top of it.
- The legacy runtime also restores the notebook's reward-factor variants
  (`risk_only`, `green_only`) and shield-diagnostic variants of `M8_VARIANTS`,
  which the definitions-only extraction had dropped; the RISK-RL and GREEN-RL
  environments can be created again.
- `scripts/legacy_parity.py --env refactored` re-runs archived episodes in the new
  environment, and `--method REWARD_FACTOR:…` the archived reward-factor
  evaluations; `scripts/migration/env_parity.py` and `env_parity_paths.py` compare
  the two environments step by step (docs/REPRODUCIBILITY.md).
- `UrbanWorldLayers.layers_sha256()`: a checksum of a world's resolution and layers
  that does not depend on the file layout or metadata (docs/WORLD_NPZ_SCHEMA.md).
- Missions of the archived primary Vienna evaluations (footway-only world, seeds
  7000–7199) recorded in
  `scenarios/real_gis/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_seed7000_n200.csv`
  with the world's checksums; the world itself is still not distributed.
  `scripts/materialize_legacy_scenarios.py --world` records missions of a processed
  real-GIS world and also checks them in `ruscuav_benchmark.env`.
- `scripts/legacy_parity.py` gives real-GIS worlds to the notebook code under the
  footway-only names the archived runs used, needs only the world of the city being
  re-run, and records the world's checksums in its report.
- New `ruscuav_benchmark.planners`: the four planners of the paper
  (`STATIC_Astar`, `LOCAL_RH_Astar`, `FULL_RH_Astar`, `DYN_RH_Astar`), ported from the
  notebook's M9, M10 and M11 code with one A* implementation and flown in
  `ruscuav_benchmark.env`. `evaluate_planner` writes per-episode CSVs with the columns
  of the archived planner rollouts, and summaries (docs/PLANNERS.md). They reproduce
  the notebook planners step for step and all 200 episodes of each archived
  primary planner evaluation, Vienna included (docs/REPRODUCIBILITY.md).
- `ruscuav evaluate` (and `ruscuav-evaluate`, `scripts/run_static_astar_eval.py`)
  flies these planners: `--planner`, missions from `--scenarios` or from seeds,
  `--regime`, `--set FIELD=VALUE`, `--prefix`; it needs the `env` extra. Removed:
  the simplified static planner of v2.0.0 (`RiskCostAStarPlanner`,
  `RiskAStarConfig`, `run_static_astar_evaluation`: 10 m grid, λ = 0.05, metrics
  along the unflown path), which did not reproduce the paper, and the options
  `--plan-res-m`, `--agl-m`, `--risk-lambda` and `--max-episodes`. The quickstart
  flies `STATIC_Astar` on recorded canonical missions.
- `RuscuavEnv.reset(options={"start": ..., "target": ...})` flies a given mission;
  `ruscuav_benchmark.scenarios` reads missions from scenario CSVs (`read_scenarios`
  moved there and is still importable from `ruscuav_benchmark.evaluate`).
- The JSON of the canonical-world missions also records the world file, its layers
  checksum and the cross-check in `ruscuav_benchmark.env` (the CSV is unchanged).
- `scripts/legacy_parity.py` re-runs the archived dynamic planner evaluation
  (`DYN_RH_Astar`) and, with `--planners package`, flies the package planners;
  `scripts/migration/planner_parity.py` compares notebook and package planners step
  by step, and `make_planner_golden.py` records the test episodes.

## 2.0.0 — 2026-07-24

Major reproducibility and evaluation-protocol release.

- Changed the headline evaluation from strict soft-budget termination to primary no-terminal evaluation.
- Retained strict Stage C as a separately labelled operational-stopping stress test.
- Added five-seed primary no-terminal static and dynamic PPO evaluations.
- Added scenario-aligned primary no-terminal Vienna and Warsaw evaluations.
- Added zero-shot morphology-shift evaluation on AOI_SYNTH_001–003 as a secondary strict stress test.
- Added conditional/normalized risk, non-acceptability-cause decomposition and dynamic-budget sensitivity.
- Added complete fixed-checkpoint Holm families and provenance manifests.
- Added exact immutable Vienna short and extended fine-tuning checkpoints.
- Added final manuscript and supplementary PDFs and vector-quality final figures.
- Removed processed Vienna/Warsaw NPZ worlds from the public release; retained generation scripts and AOI metadata.
- Added v2 release verifier, full file/model/world manifests and SHA-256 sums.

## 1.0.0

Previous Zenodo release archived at DOI `10.5281/zenodo.21331403`.
