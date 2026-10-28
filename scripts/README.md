# Scripts

Run them from the root of a checkout, in an environment where the package is
installed with the extras they use (`pip install -e ".[env]"`, README.md);
`--help` lists the options of each. The benchmark's commands are those of `ruscuav`;
these scripts build on them or check the package against the RUSCUAV v2.0.0 Zenodo
archive.

## Tools

| Script | What it does |
|---|---|
| `build_real_gis_world.py` | builds a real-GIS world from OpenStreetMap with the options of v2.0.0 (runs `ruscuav build-aoi`, docs/BUILDING_WORLDS.md) |
| `run_static_astar_eval.py` | flies the static risk-cost A* planner on a world (`ruscuav evaluate --planner STATIC_Astar`) |
| `make_protocols.py` | writes the protocol files of the package from `scenarios/` (docs/PROTOCOLS.md) |
| `upgrade_world_npz.py` | rewrites an archived world NPZ in the `ruscuav-world/1.0` layout (docs/WORLD_NPZ_SCHEMA.md) |
| `run_gate_checks.py` | the notebook's gate checks of its synthetic worlds (exclusive classes, sane heights) |
| `generate_worlds.py`, `generate_scenarios.py` | regenerate the AOI_SYNTH_001–003 worlds and their scenario CSV |

## Checks against the archive

They take the extracted archive (`--archive` or `--root`); docs/REPRODUCIBILITY.md
gives the results.

| Script | What it checks |
|---|---|
| `verify_v2_release.py` | the archive's required files, hashes, Vienna checkpoints, PDFs and primary tables |
| `verify_paper_ready_references.py` | that the files the paper-ready tables refer to are in the archive |
| `reproduce_tables_from_csv.py` | the archived per-episode CSVs against their summary JSON files |
| `legacy_parity.py` | re-runs archived evaluations with the notebook's code and compares them with the archive |
| `materialize_legacy_scenarios.py` | writes the missions of archived evaluations to a scenario CSV |
| `migration/*.py` | the package against the notebook: environment, planners, policies, tables and statistics, trajectory scoring, world builder |

## Scripts of the v1.0.0 release

Kept for the v1.0.0 release; they expect its layout (`paper_ready_v1/`,
`results/PER_EPISODE_CSV_REQUIRED.csv`) or run the notebook's code. The package does
the same with `ruscuav evaluate` and `ruscuav_benchmark.policies.train_ppo`.

| Script | What it does |
|---|---|
| `verify_reproduction_assets.py` | the integrity checks of the v1.0.0 artifact |
| `rebuild_core_tables_from_episode_csv.py` | summaries of the per-episode CSVs that v1.0.0 listed |
| `run_receding_planners.py`, `run_strict_eval_from_weights.py`, `run_multiseed_training.py` | the notebook's planner evaluations, policy evaluations and PPO training |
