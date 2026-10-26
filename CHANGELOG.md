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
  (`DYN_RH_Astar`) and, with `--code package`, flies the package planners;
  `scripts/migration/planner_parity.py` compares notebook and package planners step
  by step, and `make_planner_golden.py` records the test episodes.
- The urban LiDAR evaluates all samples of a scan at once, and the cell lookups of
  `UrbanRiskModel` no longer go through NumPy scalars: environment steps are about
  50 times faster, with results identical bit for bit (docs/REPRODUCIBILITY.md).
  New: `UrbanRiskModel.cells_from_xy`, `UrbanHazardModel.unsafe_at_altitude`,
  `UrbanLidar.ray_hit_distances` and `UrbanLidar.march` (the sample-by-sample march).
- New `ruscuav_benchmark.policies` (extra `rl`: Stable-Baselines3): the 41 PPO
  checkpoints of the archive (`ARCHIVED_CHECKPOINTS`, `archived_checkpoint` checks
  their SHA-256) and its 31 primary policy evaluations (`ARCHIVED_EVALUATIONS`,
  `evaluate_archived`); `evaluate_policy` flies a checkpoint, a model with
  `predict` or a function of the observation and writes the per-episode rows and
  summaries of the archived evaluations, in their `posthoc`, `dynamic` or
  `transfer` format; `train_ppo` trains with the paper's PPO settings and
  curriculum; `TargetSeekingPolicy` and `ConstantPolicy` are simple reference
  policies (docs/POLICIES.md).
- New `ruscuav_benchmark.posthoc`: the post-hoc step of the archived primary
  evaluations (mission acceptability recomputed with the paper's budgets, derived
  rates and fractions, summaries over all, successful and acceptable episodes).
- `ruscuav evaluate --policy PATH --method METHOD` and `--checkpoint KEY --archive DIR`
  fly policies (`--format`, `--label`, `--stochastic`); `--evaluation NAME --archive DIR`
  re-runs an archived policy evaluation. `--world` is not needed for archived
  evaluations of the canonical world.
- `ruscuav_benchmark.tables` holds the CSV writer and the summary statistics shared
  by planners and policies; `missions_from_csv` moved to `ruscuav_benchmark.scenarios`
  (still importable from `ruscuav_benchmark.planners`). Planner summaries count
  successes as pandas compares cells (`numpy.bool_` and 1 count too).
- `requirements-reproducibility.txt` pins bottleneck 1.4.2, which pandas used for the
  standard deviations of the archived summaries, and records the library versions
  of the checkpoints correctly.
- `scripts/legacy_parity.py --method POLICY:<name>` re-runs an archived policy
  evaluation with the notebook's evaluator or, with `--code package` (formerly
  `--planners package`, which still works), with `ruscuav_benchmark.policies`;
  `scripts/migration/policy_parity.py` compares notebook and package policy rollouts
  step by step, and `make_policy_golden.py` records the test episodes.
- The dynamic-risk factories of `ruscuav_benchmark.env.legacy_factories` accept the
  notebook's own `M11DynamicRiskConfig`, so that `legacy_parity.py --env refactored`
  also runs the archived dynamic evaluators.

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
