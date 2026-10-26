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

1. the endings that depend on the position: terrain contact (the altitude at most
   1 m above the ground), success (closer than 30 m to the target horizontally),
   300 steps without getting 0.5 m closer to the target (near success within 45 m,
   otherwise stuck), a cargo health of zero, and the step limit (1,000);
2. the exposure: 0.5 s over each layer under the position, the semantic risk, the
   building and canopy collisions and clearance violations, the low-altitude
   sensitive-zone violations, the hard violations (terrain, building or canopy
   collision, or a low-altitude sensitive-zone violation) and, in the secondary
   regime, operational stopping;
3. with dynamic risk, the dynamic risk at the position and step.

The episode ends at the first ending; later positions are ignored
(`positions_ignored`). A trajectory that stops before any ending ends as `ended`.
The method's own ending (`termination` in the file) is kept as
`termination_reported` and decides nothing: an episode is a success only when its
positions reach the target region. The action shield of the environment has no
counterpart, so `shield_stuck` does not occur.

Cargo health comes from the trajectory when it has it; otherwise it is estimated
with the environment's cargo model from the accelerations of the positions (finite
differences of the velocities, starting at rest): every step whose acceleration
exceeds the soft overload limit costs health. For episodes flown in the environment
the estimate differs from the recorded value by at most 1.4e-6, except along the
edge of the workspace, where the environment clips the position but keeps the
velocity: there its cargo model sees accelerations the positions do not show (up to
0.009 of cargo health in 2 of 1,800 episodes, docs/REPRODUCIBILITY.md).

The scorer also reports what the environment's vehicle could not have flown: speeds
above its limits (18 m/s horizontally, 6 m/s vertically), positions outside its
workspace (the world's extent, from 0 to 150 m in altitude) and steps whose motion
enters a hazard between two positions (a building, canopy or sensitive zone below
its roof, top or overflight altitude, less than the minimum clearance above a
building or canopy, or 2 m or less above the terrain). The environment stops such a
step short of the hazard; the scorer can only count it. Such steps are not counted
as collisions, which are counted at the positions, as in the environment. The motion
is sampled every metre, as the environment's geometric shield samples it, so a
hazard thinner than the spacing can go unseen, by the shield as by the scorer. On a
step that the shield shortened, the samples of the flown segment lie elsewhere than
the shield's samples along the commanded one and can find a sliver the shield
skipped: in the paths flown in the environment for the checks below, this flags 7
steps in 6 of 1,800 episodes, each a sensitive zone below its overflight altitude.

The row has the columns of the primary evaluations (the `posthoc` format of
docs/POLICIES.md): steps, ending, success, final distance, path lengths, cargo
health, risk, exposure times, collisions and hard violations, the mission
acceptability recomputed with the protocol's budgets (`mission_acceptable`; with
dynamic risk including the dynamic budget, and `mission_acceptable_static_only`
without it) and the environment's own verdict (`mission_acceptable_logged`); with
dynamic risk also `dynamic_risk_mean`, `dynamic_risk_max`, `dynamic_cumulative_risk`
and `dynamic_budget_ok`. The return (`total_reward`) is not computed. These columns
describe the scoring: `termination_reported`, `cargo_health_source` (`given` or
`estimated`), `positions_ignored`, `max_speed_xy_mps`, `max_speed_z_mps`,
`max_acc_xy_mps2`, `within_speed_limits`, `within_workspace` and
`hazard_segment_steps`. The summary counts the episodes with estimated cargo health
(`cargo_health_estimated`), outside the speed limits or the workspace
(`episodes_outside_speed_limits`, `episodes_outside_workspace`) and with steps into
a hazard (`episodes_with_hazard_segments`), and `ruscuav evaluate` notes them: a
trajectory the vehicle could not fly is scored, but its results are not comparable
with those of the environment's methods.

## Checks

Episodes flown in the environment and scored again from their saved paths give the
environment's rows: every column identical apart from the cargo health (estimated)
and the endings that the positions do not show (`shield_stuck`, a planner without a
path, `planner_no_path`, and a planner's own step limit, `manual_max_steps`), which
end as `ended`. Their speeds and positions are within the vehicle's limits and
workspace, and apart from the shortened steps above no step moves into a hazard.
`tests/test_trajectories.py` checks this in the three canonical protocols; `scripts/migration/trajectory_parity.py` for the
paper's planners and three archived policies on all 200 missions
(docs/REPRODUCIBILITY.md).

`ruscuav_benchmark.metrics.evaluate_path`, kept from v2.0.0, is a lightweight helper
for waypoint paths: it samples the path at a fixed speed and altitude above the
ground and sums the exposure, without the endings, the dynamic risk or the cargo
model. Use `score_trajectory` for results to be compared with the paper's.
