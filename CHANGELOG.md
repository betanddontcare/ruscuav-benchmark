# Changelog

## 2.1.0 (unreleased)

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
- New `ruscuav_benchmark.protocols`: evaluation protocols (world checksum, missions,
  regime, dynamic risk, budgets) as JSON files; the paper's `canonical-primary`,
  `canonical-secondary`, `canonical-dynamic` and `vienna-primary` come with the
  package, `make_protocol` and `ruscuav protocol create` record one for any world
  (docs/PROTOCOLS.md). `evaluate_planner`, `evaluate_policy` and `ruscuav evaluate`
  take `protocol=` / `--protocol`: the protocol then fixes the environment's
  accounting (step length and limit, dynamics, cargo model, endings, budgets, regime,
  risk profile, dynamic risk; `Protocol.environment_config`, `check_environment`),
  and its budgets decide both the environment's and the post-hoc acceptability.
  `posthoc_row`, `run_policy_episode` and the policy summaries take `budgets`.
- New `ruscuav score` and `ruscuav_benchmark.scoring`: run summaries, training-seed
  intervals, causes of non-acceptability, the dynamic-budget sweep and exact paired
  tests with Holm's correction for any per-episode CSVs, with the archive's
  arithmetic (docs/SCORING.md); `ruscuav_benchmark.stats` has the statistics and
  `ruscuav_benchmark.paper_tables` rebuilds the paper's primary tables from the
  archive, character for character. With `--protocol` it refuses runs evaluated
  under another protocol, regime, world or dynamic risk.
- New `ruscuav_benchmark.trajectories` and `ruscuav evaluate --trajectories`: flown
  trajectories of any method (CSV, NPZ or `.npy` files) are scored on a protocol's
  missions with the environment's accounting (docs/TRAJECTORIES.md), and steps the
  environment's vehicle could not fly (speeds, workspace, motion into hazards) are
  reported.
- `ruscuav_benchmark.tables.read_rows_csv` reads CSVs as `pandas.read_csv` does
  (types, skipped lines and, by default, its float parser), without pandas.
- SciPy is a dependency of the package (the paired tests and intervals); `risk_profile`
  of `evaluate_planner` and `evaluate_policy` defaults to None (balanced), and
  `ruscuav evaluate --regime` and `--profile` default to unset.
- `scripts/migration/score_parity.py` and `trajectory_parity.py` check the tables, the
  CSV reader and the trajectory scoring against the archive, and `shield_steps.py`
  replays what the geometric shield did on given steps; `scripts/make_protocols.py`
  writes the protocol files.
- New `ruscuav_benchmark.gis` (extra `gis`: OSMnx, GeoPandas, Shapely, Rasterio, PyProj)
  and `ruscuav build-aoi`: worlds from OpenStreetMap with the pipeline of the paper's
  real-GIS worlds (the notebook's `build_real_world_for_aoi` and its footway-only
  variant), from an OSM XML file or the Overpass API (`--osm-date` for the data of a
  date, `--cache` to keep the responses), with an optional DEM and DSM, provenance in
  `meta_json`, validation and an optional protocol (docs/BUILDING_WORLDS.md). OSM
  files that cover none of the area are refused; files that cover part of it or lack
  members of multipolygons, areas without features and DEMs with gaps give warnings.
  On the same data, on the notebook's 1 km grid at 1 m and with its terrain it gives
  the notebook's worlds bit for bit (`scripts/migration/gis_parity.py`), and the
  archived Vienna world keeps all of its rules (`check_builder_rules`).
  `scripts/build_real_gis_world.py`, which could not run outside the notebook, now
  runs `ruscuav build-aoi`.
- `ruscuav validate` reports overlapping classes (building, road, sidewalk, green;
  canopy outside green areas) as warnings instead of errors: a cell's exposure counts
  towards each of its classes, and worlds built from OpenStreetMap, the paper's
  Vienna world among them, have such cells. Terrain whose lowest point is above
  148 m is an error, as `RuscuavEnv` refuses it, and terrain above 120 m, which leaves
  no room to cruise below the 150 m ceiling, a warning. `scripts/run_gate_checks.py`
  prints OVERLAP for worlds whose classes overlap (the notebook's gate of its
  synthetic worlds wants them exclusive).
- `UrbanWorldLayers.save_npz` returns the path of the file it wrote, with the `.npz`
  NumPy adds to a name without it.
- Packaging: version 2.1.0; extras `all` (`env`, `rl`, `gis`) and `test`; the licence
  as the SPDX expression `MIT AND CC-BY-4.0`, with the licence files in the wheel;
  classifiers for Python 3.10–3.13; PyYAML is no longer a dependency (no code used it).
  The source distribution holds the package (the tests need the bundled worlds and
  stay in the repository). The messages about missing extras say
  `pip install -e ".[...]"`: the package is installed from the repository.
- README rewritten as the benchmark's entry page (installation, quick start, the
  paper's results); new docs/README.md (index of the guides),
  docs/SOFTWAREX_CODE_METADATA.md, configs/README.md (which configurations the
  package holds and which are the early v0 profiles) and scripts/README.md.
  `CITATION.cff` and `codemeta.json` describe the software, version 2.1.0.
- Continuous integration (`.github/workflows/tests.yml`): ruff, the tests on Python
  3.10–3.13 with the base package and the `env` and `gis` extras, a check of the
  installed package (entry point, protocol files), and the tests with the lowest
  versions `pyproject.toml` allows. The `test` extra holds pandas below 3: the package
  reads CSVs as pandas 2 does, as the archive's post-processing did.
- Code style checked by ruff (`pyproject.toml`: pycodestyle errors and warnings,
  pyflakes, lines of up to 120 characters; the notebook's verbatim code excluded):
  unused imports removed, long lines wrapped, without changes of behaviour.
- `ruscuav validate` reports an empty file, or one that is neither a ZIP archive nor
  a NumPy file, as such (NumPy 1.23 called it pickled data).
- `scripts/verify_paper_ready_references.py` checks the tables of the v2.0.0 archive
  (`paper_ready_v2/`, before only `paper_ready_v1/`, so that it passed without
  checking anything) and fails when it finds no tables; docs/REPRODUCIBILITY.md gives
  its result and notes that the equal-information table holds operational-stopping
  evaluations, whose success rates are lower than the primary ones.
  `make_protocols.py` and `migration/extract_legacy_env.py` take `--help`.
- Release metadata: `.zenodo.json` describes the package (software, 2.1.0, MIT) for
  the archiving of GitHub releases on Zenodo; `LICENSE_DATA_CC_BY_4_0` is a licence
  notice (it was a note for before the release); the description of the
  `vienna-primary` protocol carries the OpenStreetMap attribution of its missions (its
  missions are unchanged, and the description is not part of the digest);
  docs/RELEASING.md lists the steps of a release.
- `Protocol.digest()` leaves out the location of the world file (`world.file`), as its
  docstring said it did; the world is identified by its layers checksum. A protocol
  recorded with its world at another path keeps its digest, so runs recorded under
  it are scored together. The digests of the packaged protocols change, and runs
  recorded with earlier development versions of 2.1.0 carry the old ones (`ruscuav
  score --protocol` refuses them and says why); so do the default file names of runs
  under a protocol whose settings checksum covers the protocol record (policies, and
  planners with `--set`). `ruscuav build-aoi --protocol` records the OpenStreetMap
  attribution in the protocol's description.
- Trajectory scoring also checks accelerations: `max_acc_z_mps2`,
  `within_acceleration_limits` (the norm of the vehicle's command box horizontally,
  `a_max_z` vertically; `acceleration_limits()`) and, in the summary,
  `episodes_outside_acceleration_limits`. The environment's own flights exceed these
  limits too, though only on steps that its geometric shield (or the workspace
  bounds) stopped or shortened; `ruscuav evaluate` therefore notes such trajectories
  apart from those the vehicle could not have flown (docs/TRAJECTORIES.md), and
  `scripts/migration/trajectory_parity.py` reports them.
- `ruscuav evaluate --policy` loads checkpoints of other Stable-Baselines3 algorithms
  with continuous actions: `--algorithm` A2C, SAC, TD3 or DDPG (PPO by default;
  `load_policy(..., algorithm=)`, `evaluate_policy(..., algorithm=)`,
  `ruscuav_benchmark.policies.ALGORITHMS`). The configuration file records the
  algorithm of any but PPO; PPO and A2C checkpoints share their policy class and load
  with either.
- `examples/softwarex/`: the examples of the SoftwareX paper (the external planner
  GEO_SHORTEST, the runs of its two examples, its Fig. 2 and the data of its Fig. 3),
  with their results.
- Tests for the v0 dynamic-risk proxy (`ruscuav_benchmark.dynamic`) and the world
  quicklook (`ruscuav_benchmark.visualize`).
- Continuous integration also flies two missions with the installed package outside
  the checkout, the world given with `--world`, and scores their paths again as
  trajectories.
- Trajectory scoring scores only what the environment's vehicle could have flown:
  the episode ends as `infeasible`, neither a success nor acceptable, before the first
  step faster than the vehicle's speed limits, outside its workspace, or that, from a
  position outside the hazards, ends in one or runs through one over at least the
  sampling spacing of the geometric shield (1 m in the protocols): steps the shield
  would have stopped. Before, such
  trajectories were only noted, and a trajectory flying from the start to the target
  in one step scored as a success. The rows give the reason (`infeasible`) and the
  longest stretch of a step through a hazard (`max_hazard_run_m`, `hazard_run_m()`),
  in place of `hazard_segment_steps`; the summary counts `episodes_infeasible`,
  `episodes_outside_speed_limits`, `episodes_outside_workspace` and
  `episodes_through_hazards` (in place of `episodes_with_hazard_segments`). A cargo
  health given with a trajectory is taken where it is at most the cargo model's
  estimate (within 1e-3), the estimate where it is higher (`cargo_health_source`
  `capped`, `cargo_health_capped`). None of the 1,800 episodes flown in the
  environment and scored again ends as infeasible (docs/TRAJECTORIES.md).
- `ruscuav score` refuses a CSV given on its command line without a column that
  mission acceptability reads (the archive's post-processing counted a missing
  column or cell as within its budget, so a misspelled column dropped a budget;
  `dynamic_cumulative_risk` is also required when other columns of the dynamic
  evaluations are there), with an `is_success` that is not true or false, with a
  successful episode that has no number in those columns or values out of range, or
  with trajectories scored by an earlier development version (`scoring.check_run`,
  `scoring.missing_columns`). The runs of a registry are scored as before, with a
  note of their missing columns and of trajectories scored by an earlier version.
  `score.json` records these and, for runs of scored trajectories, the infeasible
  episodes, those outside the acceleration limits and the sources of the cargo
  health (`scoring.trajectory_checks`), which are also noted.
- The rows of `DYN_RH_Astar` have `building_time`, and its summary
  `mean_building_time`. The archived rows of the dynamic planner had none, so the
  archive's post-hoc acceptability of that planner left out the building budget;
  its flights stay within it (at most 6.5 s over buildings), so no verdict changes.
  `scripts/legacy_parity.py --code package` and `scripts/migration/planner_parity.py`
  compare the dynamic planner's rows with the archive's and the notebook's without
  it.
- The acceptability budgets have one definition, `posthoc.DEFAULT_BUDGETS`, from
  which the defaults of `MissionConfig` and `metrics.ExposureBudgets` are taken.
- Continuous integration runs the tests of the Stable-Baselines3 policies (job `rl`,
  PyTorch for CPU).
- The title in `CITATION.cff` and `.zenodo.json` is the manuscript's;
  THIRD_PARTY_NOTICES.md names pyogrio, not Fiona; docs/ZENODO_UPLOAD.md says that
  `zenodo_metadata.json` is in the v2.0.0 archive. The processed Vienna world of the
  paper's evaluations is to be published apart, as a dataset under the ODbL, with the
  release (docs/RELEASING.md); README.md, docs/OSM_ATTRIBUTION.md, codemeta.json and
  examples/softwarex/README.md say so. The archive is cited by the DOI of all versions
  of its Zenodo record, 10.5281/zenodo.21331402 (`pyproject.toml`, `codemeta.json`,
  `.zenodo.json`, README.md, docs/SOFTWAREX_CODE_METADATA.md), in place of that of
  v1.0.0.
- The SoftwareX article's Example 2 is in Berlin, Alt-Treptow and Treptower Park
  (examples/softwarex/README.md; `figures.sh` draws its mission 9035). Its world is
  published apart with the Vienna world, as one dataset under the ODbL, with the
  OpenStreetMap extract it was built from (README.md, docs/OSM_ATTRIBUTION.md,
  docs/RELEASING.md, codemeta.json).

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
