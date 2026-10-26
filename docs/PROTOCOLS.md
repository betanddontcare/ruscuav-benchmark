# Protocols

A protocol fixes everything that results have to share to be compared: the world,
the missions, the evaluation regime, the dynamic risk, the mission-acceptability
budgets, the risk profile, the step length and the step limit. Two methods flown
under the same protocol fly the same missions, so their results can be compared
mission by mission (docs/SCORING.md).

## The paper's protocols

```bash
ruscuav protocol list
ruscuav protocol show canonical-dynamic
```

| Protocol | World | Missions | Regime | Acceptability |
|---|---|---|---|---|
| `canonical-primary` | AOI_SYNTH_000 (in `worlds/synthetic/`) | 200, seeds 5000–5199 | primary no-terminal | static |
| `canonical-secondary` | AOI_SYNTH_000 | the same | secondary operational stopping | static |
| `canonical-dynamic` | AOI_SYNTH_000 | the same, with the reported dynamic risk field | primary no-terminal | dynamic |
| `vienna-primary` | Vienna footway-only (not distributed) | 200, seeds 7000–7199 | primary no-terminal | static |

The regimes are those of BENCHMARK_SPEC.md: in the primary no-terminal regime
(stage B, strict sensing) hard physical and low-altitude sensitive-zone constraints
are active and soft-budget exceedances are logged without ending the episode; in the
secondary regime (stage C) collisions, low-altitude sensitive-zone violations and
exposure above the stopping budgets (30 s over roads, 10 s over pedestrian ways,
20 s over buildings, any time over sensitive zones) end it. Its success rate is
strict operational completion and must not be pooled with primary success rates.

Mission acceptability is computed after the episode with the paper's budgets
(`ruscuav_benchmark.posthoc.DEFAULT_BUDGETS`): success, a cargo health of at least
0.80, at most 20 s over roads, 5 s over pedestrian ways, 15 s over buildings and
0 s over sensitive zones, and no hard violation. Dynamic acceptability adds a
cumulative dynamic risk of at most 10. In the dynamic protocol the risk field of
each mission (16 time-varying hotspots over roads and sidewalks) is drawn from the
mission's seed, so every method meets the same hotspots.

The missions are the ones the archived evaluations flew, recorded in `scenarios/`
(docs/REPRODUCIBILITY.md); the protocol files carry them, with the checksum of the
CSV they come from (`scripts/make_protocols.py` writes them).

## Evaluating under a protocol

```bash
ruscuav evaluate --protocol canonical-primary --planner FULL_RH_Astar --output-dir runs/full
ruscuav evaluate --protocol canonical-dynamic --checkpoint DYN-OBS-RL/1001 --archive /data/RUSCUAV-v2.0.0 \
    --output-dir runs/dynobs
ruscuav evaluate --protocol vienna-primary --world /data/vienna_..._footway_only_urban_world_layers.npz \
    --planner FULL_RH_Astar --set planner_agl_m=30 --output-dir runs/vienna
ruscuav evaluate --protocol canonical-primary --trajectories my_flights/ --label MINE --output-dir runs/mine
```

```python
from ruscuav_benchmark.planners import evaluate_planner
from ruscuav_benchmark.protocols import load_protocol, protocol_world

protocol = load_protocol("canonical-primary")
world = protocol_world(protocol)                 # the protocol's world, checksum checked
result = evaluate_planner(world, "FULL_RH_Astar", protocol=protocol)
```

With a protocol, `evaluate_planner`, `evaluate_policy`, `evaluate_trajectories` and
`ruscuav evaluate` take the missions, the regime, the risk profile and the dynamic
risk from it; giving any of these as well is an error. The environment is the
protocol's (`Protocol.environment_config`, checked by `Protocol.check_environment`):
its step length and step limit, the vehicle dynamics, the cargo model, the endings
and the mission settings cannot be changed under a protocol, and the protocol's
budgets are those of the environment's own verdict (`mission_acceptable` of the
planners and of the policies' `dynamic` and `transfer` rows,
`mission_acceptable_logged` of the `posthoc` rows) and of the post-hoc one.
Only the method's own settings may change (`env_overrides` of `evaluate_policy`,
`ruscuav_benchmark.protocols.METHOD_FIELDS`): the observation, the reward, the
LiDAR and the shields. The semantic observations give the exposure relative to the
budgets, so a method with such an observation sees the protocol's budgets. A
planner's `max_steps` must be the protocol's step limit.

The world must have the protocol's layers checksum (`UrbanWorldLayers.layers_sha256()`,
docs/WORLD_NPZ_SCHEMA.md): the canonical world is found in a checkout of the
repository, other worlds are given with `--world` (`world=`) or found from the
protocol's `world.file` (below). A dynamic protocol needs a method that models
dynamic risk (the `DYN_RH_Astar` planner or a `DYN-` policy preset, with the
`dynamic` row format; the paper flew its static-trained RISK+GREEN-RL policies in
`DYN-RISK-RL`, which has their observation), and a static protocol refuses one.
`--episodes N` flies the first N missions only (1 to the number of missions); the
summary then records the run as incomplete (`"protocol": {"complete": false}`). The
summary and the configuration file record the protocol's name and digest, and
`ruscuav score --protocol` checks them (docs/SCORING.md).

Flying the paper's planners under their protocols gives the archived rows
(docs/REPRODUCIBILITY.md, "The tables and statistics").

## Your own world

```bash
ruscuav validate my_city.npz
ruscuav protocol create --world my_city.npz --name my-city --output my-city.json
ruscuav protocol create --world my_city.npz --name my-city-dynamic --dynamic --output my-city-dynamic.json
ruscuav evaluate --protocol my-city.json --world my_city.npz --planner FULL_RH_Astar --output-dir runs/my_city
```

`ruscuav protocol create` (`ruscuav_benchmark.protocols.make_protocol`) records the
missions that the environment samples for seeds `--seed0`, `--seed0 + 1`, ... (200
by default): start and target on safe, low-risk cells, as in the paper's worlds. The
file records the world's layers checksum, so results under it are tied to that exact
world; a world rebuilt from newer OpenStreetMap data has another checksum and needs
its own protocol. It records the world file relative to the protocol file, so that
the two can be moved together.

Other acceptability budgets are given to `make_protocol(..., budgets={"road_s": 10.0})`
(the budgets named replace the paper's), or written into the file. A protocol with
other budgets scores the same flights differently, so its results are compared only
with results under it.

## The file format

A protocol is a JSON file (`"format": "ruscuav-protocol-1"`):

```json
{
 "format": "ruscuav-protocol-1",
 "name": "canonical-primary",
 "description": "...",
 "world": {"id": "AOI_SYNTH_000", "layers_sha256": "be2bdac9...", "file": "worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz", "distributed": true},
 "regime": "primary_no_terminal",
 "dynamic_risk": null,
 "acceptability": "static",
 "budgets": {"cargo_min": 0.8, "road_s": 20.0, "ped_s": 5.0, "building_s": 15.0, "sensitive_s": 0.0, "dynamic": 10.0},
 "risk_profile": "balanced",
 "dt_s": 0.5,
 "max_steps": 1000,
 "missions_source": {"file": "scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv", "sha256": "4c19dc02..."},
 "missions": [
  {"seed": 5000, "start": [590.0, 773.0, 48.279541], "target": [180.0, 68.0, 50.910175]},
  ...
 ]
}
```

- `world.file` is the world's file: an absolute path, or a relative one, looked for
  from the folder of the protocol file, the current directory and the repository
  root, in this order (`find_world`); `distributed` says whether the repository has
  it.
- `regime` is `primary_no_terminal` or `secondary_operational_stopping`.
- `dynamic_risk` is null or the fields of `ruscuav_benchmark.env.DynamicRiskConfig`;
  `acceptability` is `static` or `dynamic` (dynamic needs dynamic risk).
- `budgets` has the six budgets of `DEFAULT_BUDGETS`, numbers of at least 0 (the
  cargo health at most 1). With dynamic risk, its `dynamic_cumulative_risk_budget`
  must equal `budgets.dynamic`: it is the budget of the environment's own verdict
  and of the dynamic observation.
- `dt_s` and `max_steps` are the step length and the step limit of the environment.
- Each mission has the episode seed (it also seeds the environment's random state
  and the dynamic risk field) and the start and target positions: x and y in metres
  in the world's grid frame, z the absolute altitude in the height frame of the
  world's terrain.
- `Protocol.digest()` is the SHA-256 of the content without the description.
