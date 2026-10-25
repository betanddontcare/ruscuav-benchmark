# Planners

`ruscuav_benchmark.planners` holds the four planning baselines of the paper. Each
plans a path on a coarse risk-cost grid with A* and flies it in the RUSCUAV
environment with a waypoint controller.

| Planner | Kind | What it knows | Replanning |
|---|---|---|---|
| `STATIC_Astar` | static | full risk map | none: one plan at the start |
| `LOCAL_RH_Astar` | receding | risk within 75 m of where it replanned, the map mean elsewhere | every 5 steps |
| `FULL_RH_Astar` | receding | full risk map | every 5 steps |
| `DYN_RH_Astar` | dynamic | full static risk map; dynamic risk within 75 m of where it replanned, zero elsewhere | every 3 steps |

All four plan at 40 m above ground on a 5 m grid, with risk weight λ = 0.35, and
fly at 12 m/s. The real-GIS evaluations of the paper flew `FULL_RH_Astar` at 30 m.
These are the settings of the archived v2.0.0 runs (`PLANNER_PRESETS`), and the
planners reproduce those runs episode for episode (docs/REPRODUCIBILITY.md).

```bash
pip install -e ".[env]"        # the planners fly in the environment
```

## Running a planner

```python
from ruscuav_benchmark import UrbanWorldLayers
from ruscuav_benchmark.planners import evaluate_planner, missions_from_csv

world = UrbanWorldLayers.load_npz("worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz")

# Missions drawn by the environment from seeds 5000, 5001, ...
result = evaluate_planner(world, "FULL_RH_Astar", n_episodes=20, output_dir="runs/full_rh")
print(result.summary["success_rate"], result.summary["mission_acceptability_rate"])

# The recorded missions of the paper's canonical-world evaluation
missions = missions_from_csv("scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv", world)
result = evaluate_planner(world, "STATIC_Astar", missions=missions)

# Another world, another planning altitude, the operational-stopping regime
my_world = UrbanWorldLayers.load_npz("my_city.npz")
result = evaluate_planner(my_world, "FULL_RH_Astar", planner_agl_m=30.0,
                          regime="secondary_operational_stopping", n_episodes=50)
```

`evaluate_planner` flies one episode per seed or mission, each in a fresh
environment, and returns the episode rows, the flown paths and a summary. With
`output_dir` it writes `<prefix>_episodes.csv` (rewritten after every episode),
`<prefix>_summary.json` and `<prefix>_config.json`, which records the planner and
environment settings; with `save_paths=True` also the flown paths (and, for the
static planner, the planned paths) as `.npy` files in `<prefix>_paths/`. The
default prefix is `<planner>_<world>_<regime>`, followed by a checksum of the
settings when they differ from the preset, so runs with other settings do not
overwrite each other. Keyword arguments such as `planner_agl_m=30.0` change fields
of the planner configuration (and `dynamic_prior_mode` that of the dynamic
planner); the regime is `primary_no_terminal` (default) or
`secondary_operational_stopping`, or give `stage` and `mode`. `risk_profile`
(default `balanced`, as in every archived run) sets the risk profile of both the
planner and the environment.

The same from the command line:

```bash
ruscuav evaluate --world worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz \
    --planner FULL_RH_Astar --scenarios scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv \
    --output-dir runs/full_rh
ruscuav evaluate --world my_city.npz --planner STATIC_Astar --episodes 50 \
    --set planner_agl_m=30 --regime secondary_operational_stopping --output-dir runs/my_city
```

Without `--scenarios` the missions are drawn from seeds `--seed0`, `--seed0`+1, …
(default 5000). With `--scenarios`, the rows of the world are flown (matched by
`world_id`; a file without that column is taken to be of the world). The `seed`
column sets the episode seed, which also seeds the dynamic risk field: files
written with one generation seed for all rows, such as
`scenarios/legacy_v1/static_eval_seeds.csv`, give every episode the same field (a
warning says so). Rows without a seed get `--seed0` plus the row index.
`--set FIELD=VALUE` changes a configuration field of the next section (or
`dynamic_prior_mode`), `--prefix` the file names. `ruscuav-evaluate` is the same
command.

The building blocks can also be used directly: `RiskCostGrid` (the grid), `astar`
and `plan_path`, the planner classes `StaticPlanner`, `RecedingHorizonPlanner`
and `DynamicRecedingPlanner`, `waypoint_action` (the controller), and
`run_static_episode`, `run_receding_episode` and `run_dynamic_episode`, which fly
one episode in an environment you create (`ruscuav_benchmark.env.make_env`).

## How they work

**Planning grid.** The cell size is `plan_res_m` (5 m) rounded to a multiple of
the world resolution. The semantic risk of the risk profile at `planner_agl_m`
(40 m) is averaged over each cell, divided by its 95th percentile over the map and
clipped to [0, 10]. A cell costs 1 + `risk_lambda` (0.35) times this value. Cells
are blocked where buildings plus `building_clearance_m` (5 m), or tree canopy plus
`canopy_clearance_m` (3 m), reach the planning altitude, and sensitive zones are
blocked when the planning altitude is below `sensitive_min_agl_m` (80 m).

**A\*.** The search is 8-connected. A move costs its length times the mean cost
of the two cells, diagonal moves may not cut the corner of a blocked cell, and the
heuristic is the straight-line distance (`use_astar=False` gives Dijkstra's
search). A blocked start or goal moves to the nearest free cell within 20 cells.
The path runs from the start through the centres of the cells to the target.

**Static planner.** One plan from start to target on the full map.

**Receding planner.** At the start and before every replan, every
`replan_every_steps` steps, the planner senses the true risk within
`sensing_radius_m` (75 m) of the current position; sensed cells stay known. Elsewhere it uses the prior of
`prior_risk_mode`: `zero`, `mean` (the map mean; `LOCAL_RH_Astar`), `blurred` (the
true risk blurred with a 5-cell Gaussian) or `perfect` (the true risk:
`FULL_RH_Astar`). Blocked cells are known everywhere (`hard_blockers_known`) or are
sensed with the risk. It then replans from the current position. If the first plan
fails the episode ends (`planner_no_path`); a later failure keeps the last plan.

**Dynamic planner.** A receding planner that adds the episode's dynamic risk at
the planning altitude and the current step, divided by `dynamic_norm_denom` (10).
It knows the dynamic risk in the cells sensed so far, evaluated at the current
step, and assumes zero elsewhere (`dynamic_prior_mode="mean"`: the mean of the
current dynamic map). It flies in the `DYN-RISK-RL` environment with the reported
dynamic risk configuration and reads that environment's dynamic field.

**Controller.** Waypoints within `waypoint_radius_m` (8 m) are passed. The desired
velocity points at the current waypoint at `target_speed_mps` (12 m/s); closer to
the target than 1.25 times that many metres it drops to 0.8 m/s per metre of
remaining distance, but not below 2 m/s. The action is the horizontal acceleration
that reaches the desired velocity in one step, scaled by `a_max_xy` and clipped to
[-1, 1]; the environment's autopilot holds the altitude. An episode stops after
`max_steps` (1000) steps (`manual_max_steps` if the environment has not ended it).

The static and receding planners fly in the `RISK+OBS-RL` environment, as in the
archived runs. The planners do not use the observation or the reward; the
environment records the exposure, risk, cargo health and mission acceptability
(docs/ENVIRONMENT.md).

## Episode rows

The rows have the columns of the archived rollouts of each planner kind:

- all: `episode`, `seed`, `total_reward`, `steps`, `termination`, `is_success`,
  `mission_acceptable`, `dist_xy` (final distance to the target), `cargo_health`,
  `urban_mean_risk`, `urban_cumulative_risk`, the exposure times `road_time`,
  `sidewalk_time` (pedestrian ways), `green_time` and `sensitive_time` in seconds,
  and `urban_hard_violation_steps`;
- static and receding planners also: `urban_max_risk`, `building_time`,
  `canopy_time`, `building_collision_steps`, `canopy_collision_steps`,
  `sensitive_low_altitude_violation_steps`;
- static planner: `planner_success`, `planner_reason`, `planner_cost`,
  `planner_expanded`, `planned_path_length`;
- receding planners: `planner_type`, `prior_risk_mode`, `sensing_radius_m`,
  `replan_every_steps`, `risk_lambda`, replan counts, expanded cells, planning times
  and the known fraction of the map;
- dynamic planner: `method`, `dynamic_prior_mode`,
  `mission_acceptable_static_only`, `dynamic_risk_mean`, `dynamic_risk_max`,
  `dynamic_cumulative_risk`, `dynamic_budget_ok`, replan counts and the known
  fractions.

`termination` is the environment's termination (docs/ENVIRONMENT.md),
`planner_no_path`, or `manual_max_steps`. The summary has the success and
mission-acceptability rates, the acceptability rate among successes, means of the
episode metrics and the termination counts.

## Relation to the v2.0.0 notebook

The planners port the notebook's M9 (static), M10 (receding) and M11 (dynamic)
planners and rollouts operation for operation; the summaries compute the archived
summaries' statistics with pandas' semantics. The evidence that the planners
reproduce the archived runs, step by step and episode by episode, is in
docs/REPRODUCIBILITY.md. Differences that do not change any archived result:

- cell centres are clipped to the world's extent; the notebook clipped to a fixed
  1000 m, the extent of every archived world;
- the `blurred` prior needs SciPy; without it the notebook fell back to the `mean`
  prior;
- the dynamic planner reads the environment's dynamic field; the notebook built
  its own from the episode seed, which is the same field for seeded episodes;
- an episode resets the environment once; the notebook reset it twice with the
  same seed, to the same state;
- one grid serves all episodes of an evaluation; the notebook rebuilt it for every
  episode of the dynamic planner;
- `risk_profile` applies to the environment too; the notebook always built the
  environment with the `balanced` profile, the one of every archived run.

A notebook quirk is kept: a dynamic-planner episode whose first plan fails ends
after zero steps with `termination="manual_max_steps"`, while the receding planner
reports `planner_no_path`.

The archived primary dynamic evaluation recomputed mission acceptability post hoc
from the episode row and added derived metrics (risk per step and per metre,
exposure fractions); `scripts/legacy_parity.py` applies that step with the
notebook's functions. On its 200 episodes the recomputed acceptability equals the
environment's, which the rows carry. The two need not agree in general: the
dynamic rows have no `building_time`, so the post-hoc check skipped the building
budget. The post-hoc protocol is not part of this module yet.

The v2.0.0 package also had a simplified static planner (10 m grid, λ = 0.05,
metrics computed along the path without flying it), behind the old
`ruscuav evaluate`. It did not reproduce the paper and has been removed.
