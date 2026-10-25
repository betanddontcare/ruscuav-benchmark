# The RUSCUAV environment

`ruscuav_benchmark.env.RuscuavEnv` is the Gymnasium environment of the benchmark: a
cargo UAV flies a low-altitude mission between two points of a RUSCUAV world while
the environment logs its exposure to roads, pedestrian ways, buildings and
sensitive zones. One class covers every learned-policy method and evaluation regime
of the paper; a variant is a configuration.

```bash
pip install -e ".[env]"        # gymnasium, scipy and NumPy 2
```

```python
from ruscuav_benchmark import UrbanWorldLayers
from ruscuav_benchmark.env import make_env

world = UrbanWorldLayers.load_npz("worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz")
env = make_env(world, "RISK+GREEN-RL", regime="primary_no_terminal")
obs, info = env.reset(seed=5000)          # canonical mission of seed 5000
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
print(info["termination"], info["urban_road_time"], info["mission_acceptable"])
```

**Worlds.** The environment takes any world in the RUSCUAV schema (check it with
`ruscuav validate`), with two physical requirements. Terrain heights must be
relative to the area: the workspace ceiling is 150 m on the same height scale
(`dynamics.z_max_m`), so a DEM in metres above sea level needs shifting or a
higher ceiling. The world must be large enough for missions of at least 500 m
between endpoints 50 m from the edges (`scenario`). The environment refuses worlds
where no mission can be drawn. Its defaults, including the observation scales of
1000 m and 1500 m, were calibrated on 1000 m × 1000 m worlds at 1 m, the size of the
bundled worlds; other sizes run, with a warning.

## Choosing a variant

`make_env(world, method, regime=... | stage=..., mode=..., **overrides)` builds the
configuration with `make_config` and creates the environment. Any field of
`EnvConfig` can be overridden.

**Methods** (short names of the paper's method-naming table):

| Method | Observation | Urban reward | Dynamic risk |
|---|---|---|---|
| `GEO-RL` | geometry (80) | none | – |
| `OBS-RL` | semantic (100) | none | – |
| `RISK-RL` | geometry | risk | – |
| `GREEN-RL` | geometry | green | – |
| `RISK+GREEN-RL` | geometry | risk + green | – |
| `RISK+OBS-RL` | semantic | risk + green | – |
| `DYN-RISK-RL` | geometry | risk + green | reported field |
| `DYN-OBS-RL` | dynamic (86) | risk + green | reported field |

The static and receding planners of the paper ran in the `RISK+OBS-RL`
environment, the dynamic planner in `DYN-RISK-RL` (they ignore the observation;
docs/PLANNERS.md). `STATIC-RL-DYN` evaluates a static policy in the `DYN-RISK-RL` environment.

**Regimes, stages and modes:**

| Stage | Risk weight λ | Hard sensitive zones | Operational stopping |
|---|---|---|---|
| A | 0.05 | no | no |
| B | 0.10 | yes | no |
| C | 0.10 | yes | yes |

The primary no-terminal regime (`primary_no_terminal`) is stage B and the secondary
operational-stopping regime (`secondary_operational_stopping`) is stage C, both
in `strict_eval` mode. The training curriculum runs stages A, B and C in
`fast_train` mode. The two modes differ only in sampling resolution: LiDAR ray
marching every 5 m or 3 m, geometric-shield segment checks every 2 m or 1 m.

The urban reward kind sets the risk-weight multiplier and the green bonus:
`none` (no urban terms), `risk` (×1, no bonus), `green` (×0, bonus 0.03) and
`risk+green` (×1, bonus 0.03).

## Configuration

`EnvConfig` holds the high-level choices (`observation`, `urban_reward`, `stage`,
`mode`, `risk_profile`, `dynamic_risk`), optional overrides of the stage-derived
values (`urban_risk_lambda`, `green_bonus_lambda`, `hard_sensitive_zones`,
`terminate_on_urban_constraint`) and parameter groups whose defaults are the values
of the archived v2.0.0 environment:

| Group | Contents |
|---|---|
| `dynamics` | time step 0.5 s, speed, acceleration and jerk limits, filter weights, altitude band and autopilot gains, workspace ceiling 150 m, constant wind |
| `cargo` | soft overload limit 0.3 g and health decay |
| `lidar` | 64 rays over ±135° up to 250 m, urban hazard detection on/off |
| `shield` | action shield and geometric shield on/off, back-off 0.5 m, horizontal margin 1 m in the clearance score of shield candidates |
| `rewards` | weights of every reward term (below) |
| `episode` | step limit 1000, success radius 30 m, stuck patience 300 steps, shield-stuck patience 40 steps |
| `mission` | acceptability budgets (cargo ≥ 0.80, road ≤ 20 s, pedestrian ≤ 5 s, building ≤ 15 s, sensitive ≤ 0 s), operational-stopping budgets (30 / 10 / 20 / 0 s), safe-endpoint rules |
| `scenario` | mission sampling: spawn margin 50 m, minimum distance 500 m, feasibility check |

`EnvConfig.to_dict()` gives the configuration with all derived values, for logs.
Invalid combinations (an unknown stage, a dynamic observation without a dynamic
risk field, urban weights with `urban_reward="none"`, …) are rejected when the
configuration is built. The dynamic risk field is configured by
`DynamicRiskConfig` (or a dict of its fields); `REPORTED_DYNAMIC_CONFIG` is the field
of the reported dynamic experiments (16 Hann-pulse hotspots over roads and
sidewalks, budget 10, weight 0.20). This field, `ruscuav_benchmark.env.DynamicRiskField`,
is the one of the paper; `ruscuav_benchmark.dynamic` holds an older, simpler proxy.

## Action and dynamics

The action is a horizontal acceleration command in [-1, 1]², scaled by 6 m/s².
The action shield may replace it (below). The command is low-pass filtered and
jerk limited. A PD autopilot holds 30 m above terrain, or higher over buildings
and canopy (roof or canopy top plus the profile's minimum clearance), within
2–80 m above terrain. Speeds are limited to 18 m/s horizontally and 6 m/s
vertically. Cargo health drops quadratically with the effective acceleration
above 0.3 g.

**Safety stack.** The *urban LiDAR* reports the distance to the first unsafe point
along each ray at the current altitude. The *action shield* predicts the next
position. If that position is unsafe, it picks the best safe candidate among the
command, its two perpendiculars, a push along the clearance gradient and zero.
The *geometric shield* stops the position update short of the first unsafe point
on the motion segment. A point is unsafe when it is at most 2 m above the terrain,
below the minimum clearance over a building or canopy (5 m and 3 m), or, in stages
B and C, inside a sensitive zone below the overflight altitude. Hazards are judged
per world cell.

## Observation

| Values | Content |
|---|---|
| 0–2 | position / workspace size |
| 3–4 | target − position (x, y) / 1000 m |
| 5 | distance to target / 1500 m |
| 6–7 | unit direction to target |
| 8 | horizontal speed / 18 m/s |
| 9 | cosine between velocity and target direction |
| 10 | height above terrain / 100 m |
| 11–74 | LiDAR readings (1 = free up to 250 m); `lidar.n_rays` changes their number and shifts the later indices |
| 75–77 | velocity / speed limits |
| 78 | overload proxy, always 0 (see below) |
| 79 | cargo health |
| +20 semantic | risk and class flags under the UAV, building and canopy clearance, hard violation, risk and class shares sampled 20–220 m ahead, risk sampled to the left and right, exposure budgets used |
| +6 dynamic | dynamic risk here, ahead (mean, max), left, right, dynamic budget used |

## Reward

Terms in the order they are applied:

| Term | Value |
|---|---|
| progress | +12 × reduction of the distance to target |
| alignment | +0.8 × cosine between velocity and target direction |
| low speed | −0.5 below 2 m/s |
| time | −0.1 |
| corridor | −min(14, 0.002·(e−20)₊² + 0.03·(e−100)₊), e = distance from the start–target line |
| LiDAR proximity | −min(30, 150·(0.25 − closest reading)₊²) |
| terrain collision | −500, ends the episode, at 1 m or less above terrain |
| geometric shield | −5 when it changed the position, −10 more if the UAV moved < 0.1 m |
| shield stuck | −1500, ends the episode after 40 consecutive shielded steps without progress |
| altitude | −0.04 × \|height above terrain − 30\| / 30 |
| acceleration | −0.001 × \|effective acceleration\|² |
| overload | −80 × (overload above 0.3 g)² |
| boundary | −min(25, 0.015 × d²) within 35 m of the workspace edge |
| success | +3000 within 30 m of the target |
| near success / stuck | +1500 if 30–45 m away, else −200, after 300 steps without improving by 0.5 m |
| cargo failure | −200 when health reaches 0 |
| urban risk | −λ × semantic risk at the new position |
| green corridor | +bonus over green areas outside roads, sidewalks, buildings and sensitive zones |
| operational stop | −1000 (stage C) |
| acceptable success | +500 for a success within the acceptability budgets |
| dynamic risk | −λ_dyn × dynamic risk |

## Episode end and `info`

Terminations: `success`, `near_success`, `stuck`, `shield_stuck`,
`terrain_collision`, `cargo`; in stage C also `urban_terrain_constraint` (2 m or less
above terrain; at that height the step also counts as a hard violation in every stage),
`urban_building_collision`, `urban_canopy_collision`,
`urban_sensitive_low_altitude_violation` and
`urban_{sidewalk,road,building,sensitive}_budget_exceeded`. After 1000 steps the
episode is truncated with `timeout`.

Every step's `info` holds the episode's urban exposure (`urban_*_time`), risk
(`urban_risk`, `urban_mean_risk`, `urban_max_risk`, `urban_cumulative_risk` and the
cost components), violation counts (`urban_*_steps`), the class flags at the new
position, `mission_acceptable`, shield activity (`shielded`, `urban_action_shielded`,
`urban_geom_shielded`, `urban_shield_reason`, `urban_lidar_min_norm`,
`urban_signed_clearance_proxy_m`) and, with dynamic risk, `dynamic_*` values and
`mission_acceptable_static_only`.

## Missions and seeds

`reset(seed=s)` draws the mission from a generator seeded with `s`. If an endpoint
is unsafe (on a building, road, sidewalk or sensitive cell, in a hard violation,
or with risk above 1), it reseeds with `s + 9973·k` for attempt k, up to 300
attempts. `reset()` without a seed continues the generator, as training does. The
dynamic field of an episode is seeded with `s`, or drawn from the generator. The
missions of seeds 5000–5199 on the canonical world are listed in
`scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv`, those of the Vienna
evaluations in `scenarios/real_gis/`.

`reset(seed=s, options={"start": (x, y[, z]), "target": (x, y[, z])})` flies a
given mission instead, for example a row of a scenario file
(`ruscuav_benchmark.scenarios.mission_from_row`). Coordinates are in metres; a
missing z is the cruise altitude, 30 m above the ground. The mission is still
drawn first, so the random state, and the dynamic field, are those of `s`: a given
mission equal to the drawn one reproduces the plain reset exactly. Given endpoints
are not rejected when unsafe; `info` reports their safety and
`info["mission_source"]` is `"options"`.

## Relation to the v2.0.0 notebook environments

The archived results were produced by six notebook classes:
`RandomizedCurriculumEnvV14` → `RealisticUrbanCargoUAVEnv` → `…M4` → `…M11Dynamic`
→ `…M11DynamicV2` → `…M11CDynamicObs`, built by the notebook's factory functions.
`RuscuavEnv` reproduces the environments those factories build. (The final dynamic
factories build `…M11DynamicV2`; the reset of the superseded `…M11Dynamic`, which
used dynamic seed 0 for unseeded resets, is not reproduced.) The evidence, in
docs/REPRODUCIBILITY.md:

- `scripts/migration/env_parity.py` runs the notebook and the new environment in
  lockstep for every factory variant and compares observations, rewards, flags,
  `info` and state after every step, including unseeded resets;
- `scripts/migration/env_parity_paths.py` drives both into the rarely reached
  paths: every termination, stage C stops, reset retries, shield decisions near
  hazards;
- `scripts/legacy_parity.py --env refactored` re-runs archived planner and PPO
  episodes in the new environment and compares them with the archived results.

Identical bits need NumPy 2 (the `env` extra requires it): its type promotion
rules (NEP 50) are the ones the archived results were computed with.

`ruscuav_benchmark.env.legacy_factories` keeps the notebook factory names
(`make_static_eval_env`, `make_ruscuav_env_m4_mode`, …) on top of `RuscuavEnv`. The
evaluation code behind the reported results (planners, PPO evaluators) runs on it
unchanged; notebook diagnostics that call private methods of the notebook classes
do not.

Not carried over, because no RUSCUAV experiment used them:

- procedural cylinder and cone obstacles of the v14 base class (`n_obstacles` was 0
  in every experiment; the legacy factories reject other values);
- the optional shield features of the observation, LiDAR noise, and an unused
  hard-overload counter.

`info["min_clearance_safety_m"]`, the clearance to those obstacles, is kept at its
constant value 1e9 so episode logs keep their columns. Observation value 78, an
overload proxy, was never updated by the v2.0.0 environment and is therefore
always 0. It stays 0 because the archived policies were trained that way. Ground
heights are looked up with the world resolution. On the bundled 1 m worlds this
is the same cell as the v2.0.0 lookup, which assumed 1 m cells.
