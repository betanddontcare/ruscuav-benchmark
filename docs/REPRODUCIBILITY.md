# Reproducibility guide

## Asset-level verification

```bash
python scripts/verify_v2_release.py --root .
sha256sum -c SHA256SUMS
```

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
1.2.3, Stable-Baselines3 2.8.0 and PyTorch 2.11.0, matching the versions recorded in
the archived checkpoints):

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
tolerance of 1e-5 (`--rtol`), which these differences stay well within.

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
training. That gives 164 runs and 38,482 steps, with no difference. Of the
episodes, 153 ended in success, 3 in shield-stuck, 12 in stage C stops (sensitive
zone, building collision), and the rest at the 150-step cap.

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

**The archived evaluations.** With `--planners package`, `scripts/legacy_parity.py`
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
produced the archive, not from the port. The 200 episodes of one planner take
70–90 minutes on one core.

For the dynamic planner, `legacy_parity.py` applies the archived evaluation's
post-hoc step with the notebook's functions (docs/PLANNERS.md). From the archived
episode rows, `summarize_planner_rows` gives the values of the archived summaries
exactly: every statistic of the static and receding planners' summaries, and the
14 it computes of the dynamic planner's summary (which adds conditional
statistics of the post-hoc step). `tests/test_planner_outputs.py` checks this when
`RUSCUAV_ARCHIVE` points to the archive.

```bash
python scripts/legacy_parity.py --archive /path/to/RUSCUAV-v2.0.0 --work runs/parity \
    --method DYN_RH_Astar --episodes 200 --planners package
```

`tests/test_planner_episodes.py` replays ten episodes recorded from the notebook
planners (`scripts/migration/make_planner_golden.py`), so it needs no archive: the
first 30 steps of each planner, and of the dynamic planner also in stage C, a
static flight to the target, and planning failures at the start and in mid-flight.
It compares the rows, the plans and the reset info; with `RUSCUAV_EXACT_GOLDEN=1`
bit for bit.

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
    --method VIENNA_FULL_RH_Astar --episodes 200 --planners package \
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

## Full re-evaluation in Colab (original route)

1. Create an environment matching `requirements-reproducibility.txt` and the archived `pip_freeze_colab.txt`.
2. Load the synthetic world NPZs from `worlds/synthetic/`.
3. Run the definition blocks from `source_code/top_uav_submission_pipeline.py`.
4. Run the submission-repair cells and v3 one-click runner from `code/submission_repair/`.
5. Compare regenerated artifacts with `MANIFEST.csv` and `SHA256SUMS`.

PPO retraining is computationally expensive and not required for asset-level verification. The five trained seeds and exact claim-driving checkpoints are archived.
