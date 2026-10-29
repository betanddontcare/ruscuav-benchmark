# Scoring trajectories

A method that does not run in `ruscuav_benchmark.env` (another simulator, a
planner with its own controller, recorded flights) is scored from its trajectories:
for each mission of a protocol (docs/PROTOCOLS.md), the positions it flew. The
scorer accounts them exactly as the environment accounts its own flights, so the
results can be compared with the paper's methods mission by mission
(docs/SCORING.md).

```bash
ruscuav evaluate --protocol canonical-primary --trajectories my_flights.csv --label MINE --output-dir runs/mine
ruscuav score --protocol canonical-primary MINE=runs/mine/MINE_canonical-primary_episodes.csv \
    FULL=runs/full/FULL_RH_Astar_canonical-primary_episodes.csv --compare MINE:FULL:MAR_static --output-dir scores/
```

```python
from ruscuav_benchmark.trajectories import evaluate_trajectories

result = evaluate_trajectories("my_flights.csv", "canonical-primary", label="MINE", output_dir="runs/mine")
```

## The format

A trajectory is the position at the start of the mission and after every step of
0.5 s (the protocol's `dt_s`): x and y in metres in the world's grid frame, z the
absolute altitude in metres in the height frame of the world's terrain (`terrain_z`
of the world NPZ), as float32 or float64. The first position must be the mission's
start (within 5 cm). Three layouts are read (`ruscuav_benchmark.trajectories.read_trajectories`):

- a CSV, one row per position:

  ```text
  seed,step,x,y,z,cargo_health,termination
  5000,0,590.0,773.0,48.279541,,
  5000,1,588.71,772.13,48.5,0.99998,
  ...
  ```

  `step` counts from 0 (the start); `cargo_health` (after each step, from 0 to 1,
  empty at step 0) and `termination` (the method's own ending, on any row) are
  optional, and trailing empty fields may be left out. The file is UTF-8, with or
  without a byte-order mark; blank lines are skipped;
- a folder of `.npy` files, one (n + 1, 3) array per episode with `seed_<seed>` in
  the file name, as `ruscuav evaluate --save-paths` writes them (files with
  `planned` in the name are skipped);
- an `.npz` file with one (n + 1, 3) array per key `seed_<seed>`.

Every mission of the protocol needs a trajectory (or the first N with `--episodes N`,
1 to the number of missions); trajectories of other seeds are refused.

## How a trajectory is scored

`score_trajectory` resets the environment on the mission (its seed draws the
dynamic risk field) and, for each position after the start, does what the
environment does at the end of a step, in its order:

1. the motion of the step: a step that the environment's vehicle could not have
   flown ends the episode before it, as `infeasible` (below);
2. the endings that depend on the position: terrain contact (the altitude at most
   1 m above the ground), success (closer than 30 m to the target horizontally),
   300 steps without getting 0.5 m closer to the target (near success within 45 m,
   otherwise stuck), a cargo health of zero, and the step limit (1,000);
3. the exposure: 0.5 s over each layer under the position, the semantic risk, the
   building and canopy collisions and clearance violations, the low-altitude
   sensitive-zone violations, the hard violations (terrain, building or canopy
   collision, or a low-altitude sensitive-zone violation) and, in the secondary
   regime, operational stopping;
4. with dynamic risk, the dynamic risk at the position and step.

The episode ends at the first ending; later positions are ignored
(`positions_ignored`). A trajectory that stops before any ending ends as `ended`.
The method's own ending (`termination` in the file) is kept as
`termination_reported` and decides nothing: an episode is a success only when its
positions reach the target region. The action shield of the environment has no
counterpart, so `shield_stuck` does not occur.

### What the vehicle could not have flown

Only what the environment's vehicle could have flown is scored. The episode ends
as `infeasible`, neither a success nor acceptable, before the first step
(`steps` counts the steps before it; `infeasible` says why)

- faster than the vehicle's speed limits, 18 m/s horizontally or 6 m/s vertically
  (`speed`);
- to a position outside its workspace, the world's extent from 0 to 150 m in
  altitude (`workspace`);
- from a position outside the hazards, to a position in one, or through one over at
  least the spacing at which the environment's geometric shield samples motion,
  1 m in the protocols (`hazard`).

The geometric shield samples the commanded motion of each step at most 1 m apart,
its end included, and stops the vehicle short of the first sample in a hazard (a
building, canopy or sensitive zone below its roof, top or overflight altitude, less
than the minimum clearance above a building or canopy, or 2 m or less above the
terrain), or keeps it where it was. From a position outside the hazards the
environment's vehicle therefore ends every step outside them, and a motion that
runs through a hazard over 1 m or more is one the shield would have stopped; a
thinner sliver can fall between its samples. The scorer finds the steps whose
motion has a sample in a hazard as the shield does, then measures the longest
stretch of the motion through one (`max_hazard_run_m`, `hazard_run_m()`: samples
every 5 cm, the ends of each stretch located to 0.1 mm). A step that ends in a
hazard, or crosses one over 1 m, is infeasible; a shorter stretch on the way is let
through, as the shield could have let it through. From a position already in a
hazard (a mission's start in one) the shield does not check the motion, and
neither does the scorer. So a trajectory that descends to the ground
ends as infeasible at the first step that ends 2 m or less above the terrain.

Accelerations above the limits of the vehicle's commands are reported, not
refused: the environment's own flights reach them where the geometric shield, or
the bounds of the workspace, stop or shorten a step at once (see Checks below), so
they mark a trajectory that turns or stops faster than the vehicle's commands can,
not one that the environment could not have produced. The acceleration limits
(`acceleration_limits()`) are horizontally the norm of the command box,
`sqrt(2) * a_max_xy` (8.49 m/s² with the default `a_max_xy` of 6 m/s²), vertically
`a_max_z` (3 m/s²). The accelerations are finite differences of the positions, at
rest at the mission's start, so a first step faster than 4.24 m/s (8.49 m/s² over
0.5 s) exceeds them. The low-pass filter and the jerk limit keep the commands
within these bounds, and the speed limits only shorten an acceleration. Jerk is not
checked separately: at the 0.5 s step, two accelerations within the bounds differ
by at most 34 m/s³, close to the 30 m/s³ jerk limit of the commands. A trajectory
that accelerates harder pays for it in cargo health.

### Cargo health

Cargo health is the environment's cargo model applied to the accelerations of the
positions (finite differences of the velocities, starting at rest): every step
whose acceleration exceeds the soft overload limit costs health. A cargo health
given with the trajectory is taken after each step where it is at most this
estimate (within 1e-3), and the estimate where it is higher (`cargo_health_source`:
`given`, `capped`, or `estimated` without one): a method can report a lower value
than the model, not a higher one. For episodes flown in the environment the
estimate differs from the recorded value by at most 1.4e-6, except along the edge
of the workspace, where the environment clips the position but keeps the velocity:
there the positions stop at once and the environment's velocity does not, so the
estimate counts decelerations that its cargo model does not, and comes out lower
(by up to 0.009 of cargo health in 2 of 1,800 episodes, docs/REPRODUCIBILITY.md).
Given with such a trajectory, the environment's own value is capped to the
estimate there.

### The rows

The row has the columns of the primary evaluations (the `posthoc` format of
docs/POLICIES.md): steps, ending, success, final distance, path lengths, cargo
health, risk, exposure times, collisions and hard violations, the mission
acceptability recomputed with the protocol's budgets (`mission_acceptable`; with
dynamic risk including the dynamic budget, and `mission_acceptable_static_only`
without it) and the environment's own verdict (`mission_acceptable_logged`); with
dynamic risk also `dynamic_risk_mean`, `dynamic_risk_max`, `dynamic_cumulative_risk`
and `dynamic_budget_ok`. The return (`total_reward`) is not computed. These columns
describe the scoring: `termination_reported`, `infeasible` (`speed`, `workspace`,
`hazard` or empty), `cargo_health_source`, `positions_ignored`, `max_speed_xy_mps`,
`max_speed_z_mps`, `max_acc_xy_mps2`, `max_acc_z_mps2` (over the steps flown and
the infeasible one), `within_speed_limits`, `within_acceleration_limits`,
`within_workspace` and `max_hazard_run_m`. The summary counts the episodes ended as
infeasible (`episodes_infeasible`) and, by reason, `episodes_outside_speed_limits`,
`episodes_outside_workspace` and `episodes_through_hazards`; the episodes outside
the acceleration limits (`episodes_outside_acceleration_limits`); and those with
estimated or capped cargo health (`cargo_health_estimated`, `cargo_health_capped`).
`ruscuav evaluate` notes them, and `ruscuav score` reports them for each run of
scored trajectories (docs/SCORING.md).

## Checks

Episodes flown in the environment and scored again from their saved paths give the
environment's rows: every column identical apart from the cargo health (estimated)
and the endings that the positions do not show (`shield_stuck`, a planner without a
path, `planner_no_path`, and a planner's own step limit, `manual_max_steps`), which
end as `ended`. None of them ends as infeasible: their speeds and positions are
within the vehicle's limits and workspace, and no step runs through a hazard over
1 m. In 6 of the 1,800 episodes of the paper's methods checked below (two of each
archived policy), a step that the geometric shield shortened crosses a sliver of a
sensitive zone below its overflight altitude, at most 0.73 m long, which the
shield's samples along the commanded motion skipped.
Their accelerations exceed the limits only on steps that the geometric shield (or
the bounds of the workspace) stopped or shortened, which it does at once. Under
`canonical-primary` this happens in 21 of FULL_RH_Astar's 200 missions and 20 of
STATIC_Astar's, and in 153 and 140 of the 200 missions of the archived checkpoints
RISK+GREEN-RL/4001 and GEO-RL/1001, every exceeding step at or next to a step of
the shield or the bounds. In the Vienna world, FULL_RH_Astar at 30 m exceeds them
in 1 of the 200 missions of `vienna-primary`, next to a step of the shield, and in none
of the missions of the SoftwareX example's protocol (seeds 9000–9199,
examples/softwarex/).
`tests/test_trajectories.py` checks the rows in the three canonical protocols and the
accelerations in `canonical-primary`, and the endings of synthetic trajectories that
the vehicle could not have flown; `scripts/migration/trajectory_parity.py` checks the
rows for the paper's planners and three archived policies on all 200 missions, that
none of them ends as infeasible, and reports the episodes outside the acceleration
limits and the longest stretch of a step through a hazard (docs/REPRODUCIBILITY.md).

`ruscuav_benchmark.metrics.evaluate_path`, kept from v2.0.0, is a lightweight helper
for waypoint paths: it samples the path at a fixed speed and altitude above the
ground and sums the exposure, without the endings, the dynamic risk or the cargo
model. Use `score_trajectory` for results to be compared with the paper's.
