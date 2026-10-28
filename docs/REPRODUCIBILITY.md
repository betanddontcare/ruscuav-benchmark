# Reproducibility guide

## Asset-level verification

```bash
python scripts/verify_v2_release.py --root .
sha256sum -c SHA256SUMS
```

The repository's `scripts/verify_paper_ready_references.py` checks that the files
the paper-ready tables refer to (their columns whose names end in `_path`) are in the
archive (the archive's own copy of the script reads only `paper_ready_v1/`, which
v2.0.0 does not have, and passes without checking anything):

```bash
python scripts/verify_paper_ready_references.py --root /path/to/RUSCUAV-v2.0.0
```

Of the 1,816 references of the v2.0.0 tables, 271 are not in the archive: those of secondary analyses whose runs the archive does not include (the λ
sweep of the risk-only reward, the Warsaw zero-shot PPO runs, and runs of the v1.0.0
release under `external/RUSCUAV-v1.0.0-zenodo/`), in six secondary tables (seed
levels of the λ sweep, of the Warsaw zero-shot runs and of the multi-world test,
cause combinations, top causes, green-time correlation). Every file the primary
tables refer to is present.

## Primary analysis provenance

The completed one-click repair run is recorded in:

```text
provenance/master_run/master_run_state.json
provenance/master_run/resolved_checkpoints.csv
provenance/submission_repair/claim_artifact_manifest.csv
provenance/submission_repair/reported_configuration_audit.csv
```

All stages—provenance, static RL, static planners, dynamic evaluation, Vienna, Warsaw, post-processing, Holm correction, Fig. 9 generation and release audit—are marked complete in the archived master-run state.

## Exact Vienna checkpoints

```text
models/vienna_finetuning/stage_M12C_vienna_reward_only_C_ckpt_completed_000075776_global_000405504.zip
models/vienna_finetuning/stage_M12C_vienna_reward_only_C_ckpt_completed_000251904_global_000581632.zip
```

The first is the short fine-tuning checkpoint used in the primary Vienna table; the second is the extended fine-tuning checkpoint.

## Primary results

The `experiments/paperA_submission_repair_v1/primary_no_terminal/` and `experiments/paperA_primary_no_terminal_*` trees contain source per-episode CSVs, protocol/configuration JSONs and summaries. The corresponding paper-facing tables are in `paper_ready_v2/tables/`.

## Secondary results

Operational-stopping reward-factor/shield diagnostics and the multi-world strict morphology-shift analysis are retained as secondary analyses. They are not pooled with primary no-terminal success rate.

`paper_table_equal_information_learning_vs_planning.csv`, which
docs/TABLE_FIGURE_ARTIFACT_MAP.md lists among the primary tables, holds evaluations
in the operational-stopping regime. Its PPO rows are the five-seed runs of
`paper_table_multiseed_rl_training_seed_ci_raw.csv` (`experiments/paperA_multiseed_rl_v1_*`),
whose missions stop when they exceed an operational budget, and one of its three
planner runs (`experiments/paperA_equal_information_learning_vs_planning/`) records
such a stop too.
Its success rates are therefore lower than those of the primary no-terminal table
(`paper_table_primary_no_terminal_static_training_seed_ci.csv`): GEO-RL 0.783
against 0.793, OBS-RL 0.813 against 0.825, RISK+GREEN-RL 0.835 against 0.844. Mission
by mission, over the 1,000 missions of each method (five training seeds, missions
5000–5199), the two evaluations differ in success only in the 10, 12 and 9 missions
that the operational-stopping runs stopped at a budget, and they agree on the
acceptability of every mission: an acceptable mission exceeds no budget, so both
regimes fly it alike. The mission acceptability rates (0.710, 0.721 and 0.758) are
therefore the same in both tables.

## Re-running archived evaluations locally (no Colab)

`ruscuav_benchmark.legacy.runtime` runs the archived notebook code on an ordinary
machine. It executes the legacy definitions and the submission-repair cells in one
namespace, as the Colab notebook did, maps the hard-coded Google Drive paths to a
local folder, and reads worlds and checkpoints from an extracted copy of the Zenodo
archive without writing to it. Notebook-cell statements that the definitions-only
extraction had dropped are restored in memory, where the notebook ran them:
`OUT_SHAPE`/`LOCAL_TRANSFORM`, `m13_planner_cfg`, and the reward-factor
(`risk_only`, `green_only`) and shield-diagnostic entries of `M8_VARIANTS`, without
which the RISK-RL and GREEN-RL environments cannot be created. The files on disk
are unchanged.

`scripts/legacy_parity.py` re-runs archived primary no-terminal episodes and compares
them, column by column, with the archived per-episode CSVs:

```bash
pip install -r requirements-reproducibility.txt
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method STATIC_Astar --episodes 5      # also LOCAL_RH_Astar, FULL_RH_Astar, DYN_RH_Astar
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method PPO:RISK+GREEN-RL:4001 --episodes 5
```

Check performed on 2026-10-25 (Python 3.12, NumPy 2.0.2, SciPy 1.16.3, Gymnasium
1.2.3, Stable-Baselines3 2.8.0 and PyTorch 2.11.0; the archived checkpoints record
PyTorch 2.11.0 and Stable-Baselines3 2.8.0 with Gymnasium 1.2.3, or 2.9.0 with
1.3.0):

| Method | Episodes (seeds) | Result |
|---|---|---|
| STATIC_Astar | 13 (5000–5004, 5100–5107) | Identical: the re-run CSV rows match the archived rows character for character, apart from the episode counter and planning wall-clock times. |
| LOCAL_RH_Astar | 11 (5000–5004, 5194–5199) | Identical, as above. |
| FULL_RH_Astar | 10 (5000–5004, 5195–5199) | Identical, as above. |
| DYN_RH_Astar (primary dynamic evaluation, with its post-hoc step) | 5 (5000–5004) | 4 rows identical, as above; in seed 5001 the cumulative risk differs by 7e-15 and other risk values by up to 1e-15. |
| PPO RISK+GREEN-RL (train seed 4001), GEO-RL (1001), OBS-RL (2001), RISK+OBS-RL (single checkpoint) | 5, 5, 3, 3 (from 5000) | Termination, success, acceptability, step counts and exposure times identical; cumulative risk equal to within 4e-15; positions within 0.2 mm; total reward within 0.003. |
| VIENNA_FULL_RH_Astar, footway-only world (see "Real-GIS worlds") | 20 (7000–7019) | 19 rows identical, as above; in seed 7004 the cumulative risk differs by 1.4e-14 and the mean risk by 4e-16. |

The PPO position differences come from float32 arithmetic in the policy network:
the archive was produced with the CPU-only PyTorch wheel, the check used the CUDA
wheel running on CPU. `legacy_parity.py` therefore compares numbers with a relative
tolerance of 1e-5 (`--rtol`), which these differences stay well within. Over full
evaluations they occasionally change a flight ("The policies" below).

The restored reward-factor variants were checked against the archived strict
(stage C) reward-factor evaluations, `experiments/paperA_reward_factor_*`, with the
notebook's own evaluation function: RISK-RL and GREEN-RL, training seeds 1001 and
3001, episodes 5000–5002. In 10 of the 12 episodes everything matches within the
tolerance. In GREEN-RL 3001, episode 5001, steps, termination and exposure match,
but the final distance differs by 1.6 cm and the return by 0.19 (2e-5 relative). In
RISK-RL 3001, episode 5001, the UAV gets pinned against the action shield near step
60: the re-run slips free and succeeds at step 117, the archived run stays pinned
until the stuck termination at step 358. The two trajectories coincide up to the
pinning (cumulative risk 42.497 and cargo health 0.95937 in both). The episode is
numerically chaotic there: scaling the policy output by 1 + N(0, 1e-4) noise
reproduces the archived outcome in 6 of 6 runs, and 1e-3 noise gives success,
stuck and shield-stuck outcomes. The difference therefore comes from float-level
sensitivity, not from the restored configuration. These episodes are re-run with
`--method REWARD_FACTOR:risk_only:1001` (or `green_only`, other training seeds).

`tests/test_legacy_runtime.py` re-runs one archived episode when `RUSCUAV_ARCHIVE`
points to the extracted archive.

## The refactored environment

`ruscuav_benchmark.env.RuscuavEnv` (docs/ENVIRONMENT.md) replaces the six notebook
environment classes. It was compared with the notebook code in three ways, with the
software versions above.

**Step by step, every factory variant.** `scripts/migration/env_parity.py` builds
each variant twice, with the notebook factory and with
`ruscuav_benchmark.env.legacy_factories`. It resets both with the same seed and
applies the same actions. After every step it compares the observation, reward,
termination flags, the whole `info` dict and the state, bit for bit.

The variants cover every stage, mode, method preset, safety-stack ablation, shield
diagnostic, risk profile, dynamic and real-world configuration: 39 in all. Each
was flown on missions 5003 and 5011 with a target-seeking and a random policy, two
episodes per run. The second episode starts with an unseeded reset, as in
training. That gives 156 runs and 36,588 steps, with no difference. Of the 312
episodes, 145 ended in success, 3 in shield-stuck, 12 in stage C stops (sensitive
zone, building collision), and the rest at the 150-step cap. (These are the numbers
of the repeat after the LiDAR change below.)

```bash
python scripts/migration/env_parity.py --archive /path/to/RUSCUAV-v2.0.0 --variants all \
    --seeds 5003 5011 --max-steps 150
```

**Rarely reached paths.** `scripts/migration/env_parity_paths.py` drives both
environments, in lockstep, into the paths those flights rarely reach. Where a
default limit would take too long, it shortens the limit identically in both
environments. The paths are:

- stuck after 300 steps of hovering (stages B and C);
- near-success, holding 38 m short of the target (421 and 458 steps);
- timeout, cargo failure, terrain collision, and shield-stuck (3 runs);
- the stage C stops: road and sidewalk budgets, the sensitive zone (2 runs) and
  building collisions (2 runs);
- a flight to the workspace boundary;
- 40 unseeded resets, including reset retries;
- 10 unseeded episodes with the dynamic observation;
- the altitude reference without the building and canopy raise.

It also compares the following function by function on 1,200 random states near
buildings, canopy and sensitive zones: both shields, the clearance proxy and its
gradient, the altitude reference, the endpoint check, the semantic observation, the
LiDAR and the ground lookup. On these states the action shield kept 736 commands,
replaced 44 and stopped 420. There was no difference anywhere.

```bash
python scripts/migration/env_parity_paths.py --archive /path/to/RUSCUAV-v2.0.0
```

**Archived episodes in the new environment.** With `--env refactored`,
`scripts/legacy_parity.py` runs the archived evaluation code on `RuscuavEnv`. It
re-ran the planner and PPO episodes checked at the top of this section:

- STATIC, LOCAL and FULL receding-horizon A* (5 episodes each, seeds 5000–5004):
  identical to the archive, bit for bit, as in the notebook environment.
- The four PPO checkpoints (16 episodes) and the RISK-RL and GREEN-RL
  reward-factor episodes (12): identical, bit for bit, to the re-runs in the
  notebook environment. They therefore match the archive exactly as described
  above: within the PyTorch float32 tolerance, apart from the one pinned episode.

```bash
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method FULL_RH_Astar --episodes 5 --env refactored
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method REWARD_FACTOR:risk_only:3001 --episodes 3 --env refactored
```

`tests/test_env.py` replays five trajectories recorded from the notebook
environment (`scripts/migration/make_env_golden.py`), so it needs no archive. The
trajectories are a success, a stage C sidewalk-budget stop, a stuck termination
after four reset attempts, and the semantic and dynamic observations. With
`RUSCUAV_EXACT_GOLDEN=1` the test requires identical bits; this holds on the
reference platform.

The canonical-world missions (seeds 5000–5199), which the archived runs sampled
inside `env.reset(seed)`, are recorded in
`scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv` by
`scripts/materialize_legacy_scenarios.py`.

**The vectorized LiDAR.** The urban LiDAR used to march every ray sample by sample,
with a hazard lookup per sample: 95 % of the time of a step. It now evaluates all
samples of a scan at once (`UrbanLidar.ray_hit_distances`, with
`UrbanHazardModel.unsafe_at_altitude` and `UrbanRiskModel.cells_from_xy`), with
the arithmetic of the sample-by-sample march, which is kept as `UrbanLidar.march`;
the scalar cell lookup no longer goes through NumPy scalars. A planner episode
takes 0.7 s instead of 38 s. The results are the same bit for bit: both checks
above were repeated with the change (the 156 runs of `env_parity.py` and every
path and function check of `env_parity_paths.py`, with no difference), and so were
the planner comparisons below (the 40 episodes and 20 function checks of the
step-by-step comparison were identical again, and the 1,000 re-run episodes of the
archived planner evaluations gave the same rows as before the change, bit for
bit). The policy comparisons were all made with the vectorized LiDAR. A comparison
of the two LiDAR codes on 950 random scans in the canonical world, AOI_SYNTH_001–003
and the Vienna world, with the presets' and odd ray counts, ranges and steps, also
found no difference; `tests/test_lidar_equivalence.py` repeats it on any machine,
with the fallback paths for NaN positions, integer directions and empty ranges.

## The planners

`ruscuav_benchmark.planners` (docs/PLANNERS.md) replaces the notebook's M9, M10 and
M11 planners. It was compared with the notebook code in three ways, with the
software versions above.

**Step by step.** `scripts/migration/planner_parity.py` flies every case twice on
the same mission: with the notebook's rollout function in the notebook
environment, and with the package rollout in `ruscuav_benchmark.env`. Both record
every plan (start, path, cost, expanded cells and outcome), every action and every
position. These, the reset info and the episode row are compared bit for bit; only
the planning wall-clock times are not. The cases are:

- the four planners of the paper on missions 5000, 5003 and 5011 in the primary
  regime, and `FULL_RH_Astar` and `DYN_RH_Astar` on missions 5003 and 5011 in the
  secondary regime: 16 episodes, all flown to the target;
- 18 configuration variants on mission 5003, flown for 60 steps (30 when
  replanning at every step): Dijkstra's search, the `zero` and `blurred` priors,
  sensed blockers, replanning at every step, a 30 m sensing radius, a slower
  controller, planning altitudes of 25 m (no path) and 90 m, no building and canopy
  blocking, risk weights of 0 and 3, grids of 3 and 10 m, and the dynamic planner
  with the `mean` prior and with other replanning and sensing settings;
- planning failures: the static, local and dynamic planners in a planner world in
  which the goal is buried in a building block or walled in (6 runs).

It also compares, function by function, the planning grids of 8 configurations
with 10 random plans on each, the beliefs of the four priors, with known and with
sensed blockers, after each of six sensing steps, each followed by a plan, and the
dynamic beliefs of both dynamic priors in two episodes at five steps. The 40
episodes (2,883 steps, 552 plans) and the 20 function checks were identical.

```bash
python scripts/migration/planner_parity.py --archive /path/to/RUSCUAV-v2.0.0
```

**The archived evaluations.** With `--code package`, `scripts/legacy_parity.py`
flies the package planners in `ruscuav_benchmark.env` and compares their rows with
the archived per-episode CSVs, as above. All 200 episodes of each primary planner
evaluation were re-run:

| Planner | Episodes (seeds) | Success / acceptable | Rows identical | Other rows: risk values differ by up to |
|---|---|---|---|---|
| STATIC_Astar | 200 (5000–5199) | 92.5 % / 91.5 % | 189 | 7e-15 (11 rows) |
| LOCAL_RH_Astar | 200 (5000–5199) | 98.5 % / 95.0 % | 189 | 7e-15 (11 rows) |
| FULL_RH_Astar | 200 (5000–5199) | 98.5 % / 97.0 % | 183 | 7e-15 (17 rows) |
| DYN_RH_Astar | 200 (5000–5199) | 99.5 % / 52.0 % | 164 | 1.4e-14 (36 rows) |
| VIENNA_FULL_RH_Astar (`FULL_RH_Astar` at 30 m, see "Real-GIS worlds") | 200 (7000–7199) | 97.5 % / 65.0 % | 174 | 2.8e-14 (26 rows) |

The success and acceptability rates are those of the archived runs. In the other
rows only the mean, maximum and cumulative risk (and, for the dynamic planner, the
dynamic risk and the risk rates derived post hoc) differ, in the last bits; steps,
terminations, final distances, exposure times and cargo health are identical. The
notebook code gives the same values as the package on this machine (checked for
STATIC_Astar 5008, FULL_RH_Astar 5106, DYN_RH_Astar 5001 and Vienna 7004), so
the differences come from the floating-point arithmetic of the machine that
produced the archive, not from the port. The 200 episodes of one planner take 3–5
minutes on one core (70–90 minutes before the vectorized LiDAR).

For the dynamic planner, `legacy_parity.py` applies the archived evaluation's
post-hoc step with the notebook's functions (docs/PLANNERS.md). From the archived
episode rows, `summarize_planner_rows` gives the values of the archived summaries
exactly: every statistic of the static and receding planners' summaries, and the
14 it computes of the dynamic planner's summary (which adds conditional
statistics of the post-hoc step). `tests/test_planner_outputs.py` checks this when
`RUSCUAV_ARCHIVE` points to the archive.

```bash
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method DYN_RH_Astar --episodes 200 --code package
```

`tests/test_planner_episodes.py` replays ten episodes recorded from the notebook
planners (`scripts/migration/make_planner_golden.py`), so it needs no archive: the
first 30 steps of each planner, and of the dynamic planner also in stage C, a
static flight to the target, and planning failures at the start and in mid-flight.
It compares the rows, the plans and the reset info; with `RUSCUAV_EXACT_GOLDEN=1`
bit for bit.

## The policies

`ruscuav_benchmark.policies` (docs/POLICIES.md) replaces the notebook's policy
evaluators. It was compared with the notebook code and the archive in three ways,
with the software versions above.

**Step by step.** `scripts/migration/policy_parity.py` flies an archived checkpoint
twice on the same mission: with the notebook's rollout function in the notebook
environment, and with `ruscuav_benchmark.policies.run_policy_episode` in
`ruscuav_benchmark.env`. It compares every observation the policy receives, every
action it returns, the flown path, the episode row and the derived post-hoc
columns, bit for bit. The 19 episodes cover the three row formats with checkpoints
of every method of the evaluations below (GEO-RL, OBS-RL, RISK+GREEN-RL and
RISK+OBS-RL; the static-trained RISK+GREEN-RL, DYN-OBS-RL and DYN-RISK-RL with
dynamic risk; the transfer format in the canonical world), both regimes, and the
four archived Vienna evaluations: 2,144 steps, all identical.

```bash
python scripts/migration/policy_parity.py --archive /path/to/RUSCUAV-v2.0.0 \
    --vienna /path/to/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz
```

**The archived evaluations.** `scripts/legacy_parity.py --method POLICY:<name>
--code package` re-runs an archived evaluation with `evaluate_archived` and
compares its rows with the archived CSV, as above: integers, flags and text
exactly, other numbers to within a relative 1e-5. All 31 evaluations were re-run,
200 episodes each. In the table, a "same flight" has the steps, ending, collision
and violation counts and verdicts of the archived episode, and its other values
within 0.1 %; an "other flight" differs in one of these.

| Evaluation | Archived: success / acceptable (%) | Re-run | Rows within tolerance | Same flight, values within 0.1 % | Other flight |
|---|---|---|---|---|---|
| `static/GEO-RL/1001` | 77.5 / 70.0 | 77.0 / 69.5 | 194 | 1 | 5 |
| `static/GEO-RL/2001` | 77.0 / 66.5 | 77.0 / 66.5 | 194 | 3 | 3 |
| `static/GEO-RL/3001` | 81.5 / 74.5 | 81.5 / 74.5 | 192 | 2 | 6 |
| `static/GEO-RL/4001` | 82.0 / 76.0 | 82.5 / 76.5 | 198 | 0 | 2 |
| `static/GEO-RL/5001` | 78.5 / 68.0 | 78.5 / 68.0 | 190 | 6 | 4 |
| `static/OBS-RL/1001` | 80.0 / 71.0 | 80.0 / 71.0 | 193 | 2 | 5 |
| `static/OBS-RL/2001` | 82.0 / 71.5 | 82.0 / 71.5 | 196 | 0 | 4 |
| `static/OBS-RL/3001` | 83.0 / 70.5 | 82.5 / 70.0 | 190 | 8 | 2 |
| `static/OBS-RL/4001` | 80.5 / 69.5 | 80.5 / 69.5 | 195 | 5 | 0 |
| `static/OBS-RL/5001` | 87.0 / 78.0 | 87.0 / 78.0 | 197 | 1 | 2 |
| `static/RISK+GREEN-RL/1001` | 83.5 / 74.5 | 83.5 / 74.5 | 195 | 1 | 4 |
| `static/RISK+GREEN-RL/2001` | 81.0 / 71.5 | 81.0 / 71.5 | 197 | 0 | 3 |
| `static/RISK+GREEN-RL/3001` | 88.0 / 82.5 | 87.5 / 82.0 | 195 | 2 | 3 |
| `static/RISK+GREEN-RL/4001` | 88.0 / 77.0 | 88.0 / 77.0 | 192 | 4 | 4 |
| `static/RISK+GREEN-RL/5001` | 81.5 / 73.5 | 81.5 / 73.5 | 188 | 6 | 6 |
| `static/RISK+OBS-RL/single` | 88.0 / 78.5 | 88.0 / 78.5 | 194 | 3 | 3 |
| `dynamic/RISK+GREEN-RL/1001` | 83.5 / 47.0 | 83.5 / 47.0 | 194 | 2 | 4 |
| `dynamic/RISK+GREEN-RL/2001` | 81.0 / 47.0 | 81.0 / 47.0 | 196 | 1 | 3 |
| `dynamic/RISK+GREEN-RL/3001` | 88.0 / 54.0 | 87.5 / 54.0 | 195 | 2 | 3 |
| `dynamic/RISK+GREEN-RL/4001` | 88.0 / 51.5 | 88.0 / 51.5 | 192 | 4 | 4 |
| `dynamic/RISK+GREEN-RL/5001` | 81.5 / 48.0 | 81.5 / 48.0 | 187 | 7 | 6 |
| `dynamic/DYN-OBS-RL/1001` | 80.0 / 47.0 | 80.0 / 47.0 | 190 | 5 | 5 |
| `dynamic/DYN-OBS-RL/2001` | 89.0 / 50.5 | 89.0 / 50.5 | 196 | 1 | 3 |
| `dynamic/DYN-OBS-RL/3001` | 73.0 / 29.5 | 73.5 / 29.5 | 187 | 3 | 10 |
| `dynamic/DYN-OBS-RL/4001` | 87.5 / 47.0 | 87.5 / 47.0 | 196 | 3 | 1 |
| `dynamic/DYN-OBS-RL/5001` | 83.0 / 52.5 | 82.5 / 52.5 | 192 | 1 | 7 |
| `dynamic/DYN-RISK-RL/single` | 84.5 / 49.0 | 84.5 / 49.5 | 193 | 4 | 3 |
| `vienna/GEO-RL/1001` | 84.5 / 50.5 | 84.5 / 50.5 | 195 | 2 | 3 |
| `vienna/RISK+GREEN-RL/extended_finetuned` | 87.0 / 51.5 | 87.0 / 51.5 | 193 | 3 | 4 |
| `vienna/RISK+GREEN-RL/short_finetuned` | 87.5 / 46.0 | 87.5 / 46.0 | 197 | 2 | 1 |
| `vienna/RISK+GREEN-RL/zero_shot` | 82.5 / 47.0 | 82.5 / 47.0 | 196 | 2 | 2 |
| All 31 (6,200 episodes) | | | 5,999 | 86 | 115 |

Of the 6,200 episodes, 5,999 (96.8 %) match the archived rows, 86 (1.4 %) are the
same flight with slightly different values, and 115 (1.9 %) are another flight. Of
these, 13 changed their success and 9 their acceptability, in both directions: the
success rates of 7 evaluations and the acceptability rates of 5 differ from the
archived ones by one episode (0.5 percentage points), and the means over the five
training seeds by at most 0.1 points (success of OBS-RL, for example, 82.5 %
archived and 82.4 % re-run). The other rates are the archived ones.

Unlike the planner rows, the policy rows are almost never identical: one row is
identical bit for bit, and 10 agree to within 1e-9. The differences come from the
float32 arithmetic of the policy network, which the closed loop amplifies, and not
from the package:

- On one machine, the package flies the notebook's flights bit for bit (above).
- The notebook's own evaluators, run on this machine, give the package's rows, not
  the archived ones. They gave, bit for bit, the package's 200 rows of
  `static/GEO-RL/1001`, `dynamic/RISK+GREEN-RL/1001`, `dynamic/DYN-OBS-RL/1001` and
  `vienna/RISK+GREEN-RL/extended_finetuned`, one evaluation for each of the four
  evaluators, in `ruscuav_benchmark.env` (`--code notebook --env refactored`). In
  the notebook environment they gave the package's rows of six other flights
  (`static/GEO-RL/1001` 5006 and 5015, `static/GEO-RL/2001` 5072,
  `static/RISK+GREEN-RL/1001` 5081 and 5091, `static/RISK+GREEN-RL/3001` 5048).
- The archive was produced in Colab with the CPU-only build of PyTorch 2.11.0; the
  re-runs used the CUDA build, on the CPU of another machine. The planners, which
  have no network, reproduce their archived rows with at most last-bit differences
  in risk values.
- The smallest change of the actions is enough. Moving each component of every
  action of the package's re-run by one float32 ulp, up or down at random, changes
  the rows as much as the other machine did: of the 600 episodes of
  `static/GEO-RL/1001`, `dynamic/DYN-OBS-RL/1001` and
  `vienna/RISK+GREEN-RL/extended_finetuned`, 582 match the unperturbed re-run
  within the tolerance, 5 are the same flight and 13 another one (577, 9 and 14
  against the archive), and none is identical.

A re-run on another machine can therefore be expected to reproduce the rates
closely, not every row. A re-run of 200 episodes takes 6–14 minutes here (three
runs sharing two cores).

```bash
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method POLICY:dynamic/DYN-OBS-RL/1001 --episodes 200 --code package
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method POLICY:vienna/RISK+GREEN-RL/short_finetuned --episodes 200 --code package \
    --real-world vienna=/path/to/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz
```

**Tables and golden episodes.** From the archived episode rows,
`ruscuav_benchmark.posthoc` and `summarize_policy_rows` give the archived derived
columns and summaries of all 31 evaluations exactly (and those of the primary
dynamic planner evaluation); `tests/test_policies.py` checks this when
`RUSCUAV_ARCHIVE` points to the archive, and checks the code against the notebook's
pandas functions (`tests/notebook_reference.py`) without it.
`tests/test_policy_episodes.py` replays twelve episodes recorded from the
notebook's rollout functions with the simple policies of
`ruscuav_benchmark.policies.simple` (`scripts/migration/make_policy_golden.py`), so
it needs neither PyTorch nor the archive: every row format, both regimes,
successes, stuck and shield-stuck endings, episodes cut short and episodes of zero
steps. With `RUSCUAV_EXACT_GOLDEN=1` it also compares digests of the observations,
actions and paths, bit for bit.

## The tables and statistics

`ruscuav_benchmark.scoring` (docs/SCORING.md) replaces the notebook's
post-processing, and `ruscuav_benchmark.protocols` and
`ruscuav_benchmark.trajectories` (docs/PROTOCOLS.md, docs/TRAJECTORIES.md) let any
method be evaluated and scored as the paper's methods were. They were compared with
the archive as follows, with the software versions above.

**The paper's tables.** `scripts/migration/score_parity.py` rebuilds the eight
primary tables of the post-processing (the one-click runner's `stage_postprocess`,
`stage_holm` and `_write_static_aggregate`) from the archived per-episode CSVs of the
52 primary runs listed in `paper_table_primary_no_terminal_csv_registry.csv`
(static, dynamic, Vienna and Warsaw) and compares the files line by line. All eight
are identical, character for character:

| Table (`paper_ready_v2/tables/`) | Rows |
|---|---|
| `paper_table_primary_no_terminal_all_run_summaries.csv` | 52 |
| `paper_table_conditional_normalized_risk_metrics.csv` | 52 |
| `paper_table_non_acceptability_cause_decomposition.csv` | 416 |
| `paper_table_dynamic_budget_sensitivity.csv` | 96 |
| `paper_table_primary_no_terminal_training_seed_ci_all_scopes.csv` | 24 |
| `paper_table_primary_no_terminal_static_training_seed_ci.csv` | 33 |
| `paper_table_holm_full_families.csv` | 15 |
| `paper_table_holm_claim_driving_subset.csv` | 10 |

Three details of the archive's arithmetic matter here. The post-processing read the
CSVs back with `pandas.read_csv`, whose default float parser gives another last bit
than exact parsing for about a fifth of the cells; read exactly, 16 of the 52 run
summaries would differ in their last digits. `ruscuav_benchmark.tables.read_rows_csv`
reads CSVs as pandas does, without pandas: on all 470 CSVs of the archive it gives
the values and types of `pandas.read_csv` (pandas 2.2.2) for every one of 6.7 million
cells, with the default and with the round-trip float parser. The static five-seed
table, in contrast, summarized the evaluator's own summaries, computed before the
rows were written; it is rebuilt from the exactly read rows (read as pandas reads
them, one of its 33 rows would differ). And the paired tests used SciPy's
`binomtest`: SciPy 1.16.3 and 1.18.1 give the 15 archived p-values bit for bit,
whereas the exact binomial tail, rounded once, differs from 7 of them in the last
bit, so the package uses SciPy too. With NumPy 2.5.4 and SciPy 1.18.1 instead of the
versions above, six of the eight tables are again identical; in the two
training-seed tables the bounds of the intervals differ in the last digits (by at
most 5.4e-15, relatively), because SciPy 1.18 computes the Student-t quantile more
accurately (with 4 degrees of freedom 2.7764451051977934 instead of
2.7764451051977987).

The episode bootstrap of the multi-seed tables (`stats.episode_bootstrap_ci`, the
secondary uncertainty), recomputed for the 20 runs of
`paper_table_multiseed_rl_episode_bootstrap_secondary.csv`, gives 245 of its 320 rows
identically and the others to within 3.2e-16, relatively.

```bash
python scripts/migration/score_parity.py --archive /path/to/RUSCUAV-v2.0.0 --pandas
```

`tests/test_scoring.py` checks the tables against the notebook's own pandas code
(`tests/notebook_reference.py`) on synthetic runs, and against the archived files
when `RUSCUAV_ARCHIVE` points to the archive.

**Protocols.** The protocol files carry the missions recorded in `scenarios/`
(`tests/test_scoring.py` checks them against the CSVs and their checksums).
`scripts/migration/trajectory_parity.py` flies the paper's planners and three
archived policies under their protocols, all 200 missions, and compares the rows with
the archived ones as `legacy_parity.py` does (integers, flags and text exactly, other
numbers to a relative 1e-5; planning times are not compared):

| Run | Protocol | Rows identical | Rows within tolerance |
|---|---|---|---|
| `STATIC_Astar` | `canonical-primary` | 189 | 200 |
| `LOCAL_RH_Astar` | `canonical-primary` | 189 | 200 |
| `FULL_RH_Astar` | `canonical-primary` | 183 | 200 |
| `DYN_RH_Astar` | `canonical-dynamic` | 164 | 200 |
| `FULL_RH_Astar` at 30 m | `vienna-primary` | 174 | 200 |
| `GEO-RL/1001` | `canonical-primary` | 0 | 194 |
| `DYN-OBS-RL/1001` | `canonical-dynamic` | 0 | 190 |
| `VIENNA/RISK+GREEN-RL/extended_ft` | `vienna-primary` | 1 | 193 |

These are the results of "The planners" and "The policies" above, row for row: a
protocol run is the archived evaluation.

**Trajectories.** The same script scores the flown paths of these runs, and of
`FULL_RH_Astar` in `canonical-secondary`, with `ruscuav_benchmark.trajectories` and
compares the rows with those of the flown episodes, every column the two have:

| Run | Protocol | Episodes | Identical but for the cargo health | Ended for another reason | Largest cargo-health difference |
|---|---|---|---|---|---|
| `STATIC_Astar` | `canonical-primary` | 200 | 196 | 4 shield-stuck | 1.1e-6 |
| `LOCAL_RH_Astar` | `canonical-primary` | 200 | 199 | 1 shield-stuck | 4.0e-7 |
| `FULL_RH_Astar` | `canonical-primary` | 200 | 197 | 3 shield-stuck | 3.7e-7 |
| `FULL_RH_Astar` | `canonical-secondary` | 200 | 197 | 3 shield-stuck | 3.7e-7 |
| `DYN_RH_Astar` | `canonical-dynamic` | 200 | 200 | | 3.2e-7 |
| `FULL_RH_Astar` at 30 m | `vienna-primary` | 200 | 195 | 5 without a path | 3.2e-7 |
| `GEO-RL/1001` | `canonical-primary` | 200 | 172 | 28 shield-stuck | 2.7e-3 (7.4e-7 in the other episodes) |
| `DYN-OBS-RL/1001` | `canonical-dynamic` | 200 | 176 | 24 shield-stuck | 9.1e-3 (4.7e-7 in the other episodes) |
| `VIENNA/RISK+GREEN-RL/extended_ft` | `vienna-primary` | 200 | 185 | 15 shield-stuck | 1.4e-6 |

In all 1,800 episodes every column the scorer computes is the environment's, bit for
bit, apart from the cargo health. The 83 episodes that ended for a reason the
positions do not show (78 shield-stuck endings and 5 planners without a path at the
start) end as `ended`, with the same steps and exposure. The estimated cargo health
stays within 1.4e-6 of the environment's except in two episodes that flew along the
edge of the workspace (seed 5135 for 9 positions, seed 5158 for 5): there the
environment clips the position at the edge but keeps the velocity pointing out of
the workspace, so its cargo model sees accelerations that the positions do not have.
No mission-acceptability verdict changed. All 1,800 episodes are within the
vehicle's speed limits and workspace. The scorer counts 7 steps, in 6 episodes (two
each of the three policies), as moving into a hazard between two positions: in each
the environment's geometric shield had shortened the step, and the samples of the
flown segment found a sliver of a sensitive zone below its overflight altitude that
the shield's samples along the commanded segment had skipped
(`scripts/migration/shield_steps.py` replays them; docs/TRAJECTORIES.md).

```bash
python scripts/migration/trajectory_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/trajectory_parity \
    --vienna /path/to/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz --policies
```

`tests/test_trajectories.py` repeats the comparison in the three canonical protocols
with a simple policy, so it needs neither the archive nor PyTorch. The archive
itself keeps the flown paths of two episodes only, of the Warsaw full planner
(`primary_no_terminal_Warsaw_FULL_RH_Astar_seed7000/paths/`; the `planned_paths` of
the M9 static planner are waypoint plans, not flights), so the paths are flown
again here; the two Warsaw paths can be scored once the Warsaw world is available
(below).

## Real-GIS worlds (Vienna, Warsaw)

The processed Vienna and Warsaw worlds are derived from OpenStreetMap data and are
not distributed (docs/OSM_ATTRIBUTION.md). The archived real-GIS evaluations used
their footway-only variants, in which the sidewalk layer holds the OpenStreetMap
footways minus roads and buildings. `legacy_parity.py` gives a world passed with
`--real-world` to the notebook code under the footway-only file name the notebook
looks for first, and records the world's SHA-256 and layers checksum
(`UrbanWorldLayers.layers_sha256()`, docs/WORLD_NPZ_SCHEMA.md) in its report. Only
the world of the city being re-run is needed:

```bash
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method VIENNA_FULL_RH_Astar --episodes 20 \
    --real-world vienna=/path/to/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz
```

**Vienna.** The check used a copy of the footway-only world kept from the archived
runs (file SHA-256 `d63f0622…`, layers checksum `fa99c571…`). It reproduces the 20
archived primary planner episodes VIENNA_FULL_RH_Astar 7000–7019 (table above). In
`RuscuavEnv` (`--env refactored`), episodes 7000–7004 give, character for
character, the rows of the notebook environment. The package planner reproduces
all 200 episodes, 7000–7199 ("The planners"):

```bash
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method VIENNA_FULL_RH_Astar --episodes 200 --code package \
    --real-world vienna=/path/to/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz
```

This copy is a later build of the world than the one in the v1 release. The file
of the same name in the archive's v1 asset index (built 2 July 2026) has other
bytes (13,061,714, SHA-256 `7044b314…`), and the world metadata archived on 1 July
2026 (`worlds/real_gis/metadata/`) describes slightly different layers: this copy
has 141 fewer road cells, 539 more sidewalk cells and one more building cell. The
v2.0.0 primary Vienna evaluations, run one after another on 21 July 2026, used this
later build. The M12C fine-tuning evaluations of early July 2026 ran on the earlier
one: of the first five episodes of their stage C planner evaluation
(`m12c_vienna_footway_only_m10_receding_astar_stageC_200`), this copy reproduces
four exactly, and in seed 7001 the route differs.

The missions of seeds 7000–7199 in this world, those of the primary evaluations,
are recorded in
`scenarios/real_gis/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_seed7000_n200.csv`:

```bash
python scripts/materialize_legacy_scenarios.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/scenarios \
    --world /path/to/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz \
    --output scenarios/real_gis/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_seed7000_n200.csv
```

For every seed, 19 environments sampled the same mission: the planner environment
in stages B and C (the first one twice, from its factory and from `env.reset`), the
four learned-policy evaluation environments of the primary regime, the four M12C
environment variants (RISK+GREEN-RL and GEO-RL fine-tuning, stages B and C), and
`RuscuavEnv` for all eight method presets and for the planner environment in the
secondary regime. The JSON file next to the CSV records the world file and its
layers checksum. The missions belong to a world with that checksum. A world
rebuilt from current OpenStreetMap data will usually have another one; it can be
evaluated with the same seeds, but its missions and results are then its own.
With `RUSCUAV_VIENNA_WORLD` pointing to the world, `tests/test_env.py` checks the
first missions against the CSV.

**Warsaw.** The archived Warsaw runs used a footway-only world that the notebook
built in memory (`real_world_footway_only`) and never saved. Its footway layer
came from the OpenStreetMap download, not from the saved base world, so it cannot
be rebuilt from the base world file, which reproduces only one of four archived
planner episodes. The Warsaw missions can be recorded once a copy of the
footway-only world is found.

## Building worlds from OpenStreetMap

`ruscuav_benchmark.gis` and `ruscuav build-aoi` (docs/BUILDING_WORLDS.md) port the
notebook's `build_real_world_for_aoi`, which built the Vienna world, and the
footway-only variant of its Vienna cells. `scripts/migration/gis_parity.py` builds
each area with the notebook's code (run by `ruscuav_benchmark.legacy`) and with the
package from the same OSM XML file and compares the two worlds: every layer, the
identifiers and the area's UTM bounds, for both sidewalk variants. The notebook
downloaded its features with OSMnx; in the comparison its download function reads
them from the file with the OSMnx function the package uses (`features_from_xml`,
filtered by the area's bounding box as the download was). For layers without
features in the area (none in its bounding box, or none left after clipping to the
square), the notebook's empty-layer branches (a GeoDataFrame with the CRS name
"LOCAL_RUSCUAV", which GeoPandas 1 refuses) are given an empty frame without CRS.

| Area | OpenStreetMap data | Share of the area in the file | Features in the area (buildings, highways, green, sensitive) | Terrain | Worlds identical |
|---|---|---|---|---|---|
| Vienna Prater (the paper's area) | synthetic town | 100% | 128, 32, 20, 12 | fallback | 2 of 2 |
| Warsaw centre (the paper's area) | synthetic town | 100% | 127, 32, 20, 12 | fallback | 2 of 2 |
| Cape Town (UTM zone 34 south) | synthetic town | 100% | 137, 34, 20, 12 | DEM (WGS 84) and DSM (UTM) | 2 of 2 |
| Vienna Prater | green areas and sensitive facilities only between the bounding box and the square | 100% | 1, 1, 0, 0 | fallback | 2 of 2 |
| Germany, 48.136 N 10.0695 E | OSMnx test extract | 4.7% | 32, 15, 0, 0 | fallback | 2 of 2 |
| West Oakland | OSMnx test extract (`.osm.bz2`) | 12% | 23, 38, 2, 2 | fallback | 2 of 2 |
| Vaduz | Liechtenstein, 2013 | 100% | 7, 174, 2, 2 | fallback | 2 of 2 |
| Schaan | Liechtenstein, 2013 | 100% | 192, 188, 3, 4 | fallback | 2 of 2 |

With the GIS libraries of the archived Colab environment (`requirements-reproducibility.txt`:
OSMnx 2.1.1, GeoPandas 1.1.3, Shapely 2.1.2 with GEOS 3.13.1, Rasterio 1.5.0 with
GDAL 3.12.1, PyProj 3.7.2 with PROJ 9.5.1, NumPy 2.0.2), every layer of every world
is the notebook's, bit for bit, and so are the identifiers and the UTM bounds; the
DEM and DSM of the Cape Town case are read with the notebook's absolute heights
(`terrain_datum="absolute"`). The comparison holds on the notebook's grid (1 km at
1 m) and with its terrain; relative DEM heights (the package's default) and other
grids are the package's own. The OpenStreetMap files are test data of OSMnx (two small areas,
which cover only part of a 1 km square) and the 2013 Liechtenstein extract of
osm2pgsql's tests, converted to XML; they are not in the repository. With the newest
versions at the time of writing (OSMnx 2.1.1, GeoPandas 1.2.0, Shapely 2.2.0 with
GEOS 3.14.1, Rasterio 1.5.2 with GDAL 3.12.2, PyProj 3.8.0 with PROJ 9.8.1, NumPy
2.5.4) the package builds the same worlds of the synthetic Vienna town, West Oakland
and Schaan, bit for bit.

The archived Vienna world was built this way: it keeps every rule of the builder
(`check_builder_rules`: no sidewalk on roads or buildings, no green area or canopy
under buildings, canopy inside green areas, building heights of 3–150 m and canopy
heights of the rules (Vienna's are 6, 12 and 18 m) only on their masks, canopy
density 0.75, roofs, canopy tops and height above ground derived from their layers,
the surface the highest of terrain, roofs and canopy tops outside buildings), its
terrain is the fallback surface bit for bit, and the UTM bounds and bounding box of
`--preset vienna` are those of its archived metadata (`worlds/real_gis/metadata/`).
Its 77 building cells where the surface is 12 m above the terrain, higher than the
roof, are canopy of a park that the buildings replaced after the surface was
computed, as in the builder.

```bash
python scripts/migration/gis_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/gis_parity \
    --vienna /path/to/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz \
    --osm 37.807645,-122.300415=/path/to/West-Oakland.osm.bz2
```

`tests/test_gis.py` builds worlds from synthetic extracts (`tests/osm_fixtures.py`),
compares two of them with the notebook's builder when `RUSCUAV_ARCHIVE` is set and
checks the archived Vienna world's rules when `RUSCUAV_VIENNA_WORLD` is set.

## Full re-evaluation in Colab (original route)

1. Create an environment matching `requirements-reproducibility.txt` and the archived `pip_freeze_colab.txt`.
2. Load the synthetic world NPZs from `worlds/synthetic/`.
3. Run the definition blocks from `source_code/top_uav_submission_pipeline.py`.
4. Run the submission-repair cells and v3 one-click runner from `code/submission_repair/`.
5. Compare regenerated artifacts with `MANIFEST.csv` and `SHA256SUMS`.

PPO retraining is computationally expensive and not required for asset-level verification. The five trained seeds and exact claim-driving checkpoints are archived.
