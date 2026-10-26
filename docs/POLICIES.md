# Policies

`ruscuav_benchmark.policies` evaluates learned policies in the RUSCUAV environment
and writes the per-episode rows and summaries of the paper's evaluations. It knows
the PPO checkpoints of the v2.0.0 archive and the evaluations that used them, and
it trains new PPO policies with the paper's settings.

```bash
pip install -e ".[rl]"         # Stable-Baselines3 and PyTorch, plus the environment
```

## Evaluating a policy

```python
from ruscuav_benchmark import UrbanWorldLayers
from ruscuav_benchmark.policies import archived_checkpoint, evaluate_policy

world = UrbanWorldLayers.load_npz("worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz")

# An archived checkpoint (from the extracted Zenodo archive), 200 missions from seed 5000
ckpt = archived_checkpoint("RISK+GREEN-RL/4001", archive="/data/RUSCUAV-v2.0.0")
result = evaluate_policy(world, ckpt, "RISK+GREEN-RL", output_dir="runs/rg4001")
print(result.summary["success_rate"], result.summary["mission_acceptability_rate"])

# Your own Stable-Baselines3 checkpoint, on recorded missions, in the stopping regime
from ruscuav_benchmark.scenarios import missions_from_csv
missions = missions_from_csv("scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv", world)
result = evaluate_policy(world, "my_ppo.zip", "OBS-RL", missions=missions,
                         regime="secondary_operational_stopping")

# Any object with predict(obs, deterministic=...), or a function of the observation
from ruscuav_benchmark.policies import TargetSeekingPolicy
result = evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", n_episodes=20)
```

The third argument is the method preset of the environment
(`ruscuav_benchmark.env.METHOD_PRESETS`): `GEO-RL`, `OBS-RL`, `RISK-RL`,
`GREEN-RL`, `RISK+GREEN-RL`, `RISK+OBS-RL`, `DYN-RISK-RL` or `DYN-OBS-RL`. It fixes
the observation, which has to be the one the policy was trained with: usually the
preset it was trained in, but the paper also flew the static-trained RISK+GREEN-RL
policies in `DYN-RISK-RL`, which has the same observation. A checkpoint whose
observation size differs is refused before anything is written. The preset also
fixes the reward the environment reports and, for the `DYN-` presets, the dynamic
risk field. Each episode runs in a fresh environment. The policy acts
deterministically unless `deterministic=False`; a Stable-Baselines3 model is then
seeded with the episode seed, so that sampled runs repeat.

Episodes and regimes work as for the planners (docs/PLANNERS.md). With `output_dir`
the evaluation writes `<prefix>_episodes.csv` (rewritten after every episode),
`<prefix>_summary.json` and `<prefix>_config.json`, which records the policy (a
checkpoint's path and SHA-256, or the policy's class), the row format and label, and
the full environment configuration; with `save_paths` also the flown paths, in
`<prefix>_paths/`. The default prefix is `<method>_<world>_<regime>_<checksum>`, the
checksum of those settings, so that evaluations of different checkpoints or
settings do not overwrite each other (policies of the same class with the same
settings share it; give `prefix` to keep them apart). `env_overrides` changes other
environment settings, for example `{"lidar": LidarConfig(urban=False)}`; the
regime, the risk profile and the dynamic risk have their own arguments.

From the command line:

```bash
ruscuav evaluate --world worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz \
    --checkpoint RISK+GREEN-RL/4001 --archive /data/RUSCUAV-v2.0.0 --output-dir runs/rg4001
ruscuav evaluate --world my_city.npz --policy my_ppo.zip --method OBS-RL --episodes 50 \
    --regime secondary_operational_stopping --output-dir runs/my_city
ruscuav evaluate --evaluation dynamic/DYN-OBS-RL/1001 --archive /data/RUSCUAV-v2.0.0 \
    --output-dir runs/archived
```

`--checkpoint` flies an archived checkpoint in the preset it was trained in (or
`--method`), with the settings it was trained with (`checkpoint_env_overrides`: the
urban LiDAR off for `RISK+OBS-RL/no_urban_lidar`). `--evaluation` re-runs one of the
archived evaluations below with all its settings. `--format`, `--label` and
`--stochastic` set the row format, the method label and sampled actions.

## Episode rows

The rows have the columns of the archived evaluations, in one of three formats
(`row_format`):

- `"posthoc"` (default): the primary evaluation of the learned policies. The
  episode metrics (`total_reward`, `steps`, `termination`, `is_success`, `dist_xy`,
  `cargo_health`, mean, maximum and cumulative risk, the seconds over roads,
  pedestrian ways, buildings, green areas, canopy and sensitive zones, collision and
  hard-violation steps), the flown path length (`path_length_xy`,
  `path_length_3d`), and mission acceptability recomputed from the row with the
  paper's budgets (`mission_acceptable`; the environment's own verdict is
  `mission_acceptable_logged`). Then the derived columns of the post-hoc step: risk
  per step, second and metre, and the fraction of the flight over each layer.
- `"dynamic"` (default for, and only for, presets with dynamic risk): the
  dynamic-risk evaluations. The
  episode metrics with the dynamic risk (`dynamic_risk_mean`, `dynamic_risk_max`,
  `dynamic_cumulative_risk`, `dynamic_budget_ok`) and the environment's mission
  acceptability with and without the dynamic budget (`mission_acceptable`,
  `mission_acceptable_static_only`).
- `"transfer"`: the real-world transfer evaluations of the Vienna fine-tuning
  experiments. The episode metrics and the environment's mission acceptability;
  the method column is `policy_kind` (`reward_only` or `geometry_only`).

The budgets (`ruscuav_benchmark.posthoc.DEFAULT_BUDGETS`) are a cargo health of at
least 0.80 and at most 20 s over roads, 5 s over pedestrian ways, 15 s over
buildings and 0 s in sensitive zones, no hard violation, and, with dynamic risk, a
cumulative dynamic risk of at most 10. The summaries have the statistics of the
archived summaries of each format: success and mission-acceptability rates, means of
the episode metrics and the termination counts; in the posthoc format also the means
over the successful and over the acceptable episodes, in the other two the standard
deviation of the return (`std_reward`) and the median final distance.

## The archived checkpoints and evaluations

`ARCHIVED_CHECKPOINTS` lists the 41 PPO checkpoints of the archive by key: the
method preset they were trained in and the training seed, `<method>/<seed>` for the
five-seed runs (`GEO-RL`, `OBS-RL`, `RISK+GREEN-RL`, `RISK-RL`, `GREEN-RL`,
`DYN-OBS-RL`, seeds 1001–5001), `<method>/single` for single runs (among them
`RISK+OBS-RL/single` and `DYN-RISK-RL/single`), `RISK+OBS-RL/no_urban_lidar`, and
`VIENNA/...` for the Vienna fine-tuning (`VIENNA/RISK+GREEN-RL/short_ft`,
`extended_ft`, `final`, `VIENNA/GEO-RL/final`). Each entry has the path inside the
extracted archive and the SHA-256, which `archived_checkpoint(key, archive)` checks.

`ARCHIVED_EVALUATIONS` lists the 31 primary (no-terminal) policy evaluations of the
paper, each 200 deterministic episodes:

| Names | Checkpoints | Environment | Format |
|---|---|---|---|
| `static/<GEO-RL, OBS-RL, RISK+GREEN-RL>/<seed>`, `static/RISK+OBS-RL/single` | the same | the same, canonical world, seeds 5000–5199 | posthoc |
| `dynamic/RISK+GREEN-RL/<seed>` | the static-trained `RISK+GREEN-RL/<seed>` | `DYN-RISK-RL` | dynamic |
| `dynamic/DYN-OBS-RL/<seed>`, `dynamic/DYN-RISK-RL/single` | the same | the same | dynamic |
| `vienna/RISK+GREEN-RL/<zero_shot, short_finetuned, extended_finetuned>` | `RISK+GREEN-RL/1001`, `VIENNA/RISK+GREEN-RL/short_ft`, `extended_ft` | `RISK+GREEN-RL`, Vienna footway-only world, seeds 7000–7199 | transfer |
| `vienna/GEO-RL/1001` | `GEO-RL/1001` | `GEO-RL`, Vienna (label `VIENNA-GEO`) | posthoc |

`evaluate_archived(name, archive)` re-runs one of them with its checkpoint and
settings (`n_episodes`, `seed0` and the output can change). The Vienna evaluations
need `world=`, the processed world, which is not distributed
(docs/OSM_ATTRIBUTION.md); the world's layers checksum must be the archived one
(`check_world=False` flies another). On another machine the network's float32
results can differ in the last bit, and a few percent of the flights then take
another route: re-run on a machine other than the archive's, the 31 evaluations
matched 96.8 % of the archived rows to within a relative 1e-5, and their success
and acceptability rates to within one episode (docs/REPRODUCIBILITY.md).

## Training

```python
from ruscuav_benchmark.policies import train_ppo

paths = train_ppo(world, "RISK+GREEN-RL", "runs/train_rg", seed=1001)
result = evaluate_policy(world, paths["final"], "RISK+GREEN-RL", n_episodes=200)
```

`train_ppo` follows the archived training runs: Stable-Baselines3 PPO
(`PPO_SETTINGS`: `MlpPolicy` with two hidden layers of 256 units, learning rate
3e-4, batch size 256, `n_steps` 512 with four or more environments, entropy
coefficient 0.01, γ = 0.99, GAE λ = 0.95, clip range 0.2) on four parallel
environments in the `fast_train` sensing mode, through the curriculum of
`TRAINING_SCHEDULES` (stages A, B and C of 50,000, 75,000 and 75,000 steps for most
methods; 200,000 steps of stage A for GEO-RL). Each stage continues from the
previous stage's model; the environments of stage k are seeded from
`seed + 100 k`. The archived checkpoints were not re-trained: training is not
reproducible bit for bit across machines, so the package only checks that training
runs, saves and reloads. Evaluation uses the `strict_eval` mode.

## Relation to the v2.0.0 notebook

The rollouts and rows follow the notebook's `rollout_ppo_episode` (with
`add_derived_metrics`), `rollout_m11_dynamic_rl_episode`,
`rollout_m11c_dynamic_obs_rl_episode` and `rollout_m12c_real_policy_episode`; the
summaries follow `summarize_episode_dataframe`, `summarize_m11_dynamic_eval_df` and
`summarize_m12c_real_policy_eval_df`. Differences that do not change any archived
result: an episode resets its environment once (the notebook's factories reset it
twice with the same seed, to the same state); the summaries record the package's
settings instead of the notebook's paths; missing values are written as JSON `null`
instead of `NaN`.
