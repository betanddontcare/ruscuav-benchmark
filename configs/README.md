# Configurations

These YAML files record the settings of the archived experiments; no code reads
them. The package holds the paper's settings in code, and that is what its commands
and functions use:

| File | What it records | In the package |
|---|---|---|
| `ppo_multiseed_v0.yaml` | the five-seed PPO runs of the paper: training seeds, curriculum, evaluation missions, the reported dynamic risk | `ruscuav_benchmark.policies.TRAINING_SCHEDULES` and `PPO_SETTINGS`, `ruscuav_benchmark.env.dynamic_risk.REPORTED_DYNAMIC_CONFIG`: the same values (docs/POLICIES.md); the package does not save intermediate checkpoints (`save_freq_steps`) |
| `receding_planners_v0.yaml` | the paper's planners: static (M9), receding-horizon (M10) and dynamic (M11) risk-cost A* | `ruscuav_benchmark.planners.PLANNER_PRESETS` (docs/PLANNERS.md): `static_M9` and `receding_M10` have the values of `STATIC_Astar` and `LOCAL_RH_Astar`. `dynamic_M11` inherits `receding_M10`, but the archived dynamic runs (`DYN_RH_Astar`) replanned every 3 steps on the full static risk map (`prior_risk_mode` perfect) with the dynamic risk sensed within 75 m |
| `ruscuav_v0_static.yaml` | the early RUSCUAV-v0 example: static risk-cost A* on a 10 m grid in AOI_SYNTH_001–003, whose 30-mission results the archive keeps in `results/secondary_operational_stopping/multiworld_static_astar_archive/` | its mission budgets are the benchmark's (`MissionConfig`, `ExposureBudgets`); its planner settings are not the paper's (`PLANNER_PRESETS`: 5 m grid, λ = 0.35) |
| `ruscuav_v0_dynamic.yaml` | the early v0 proxy of dynamic risk: Gaussian hotspots switched on and off periodically (`ruscuav_benchmark.dynamic`) | superseded by the dynamic risk of the paper (`REPORTED_DYNAMIC_CONFIG`: 16 hotspots on roads and sidewalks, smooth pulses, budget 10; docs/ENVIRONMENT.md) |
| `real_gis_pipeline_v0.yaml` | the real-GIS area profile of the paper: 1 km at 1 m, footway-only sidewalks (the same file as `worlds/real_gis/real_gis_pipeline_v0.yaml`) | `ruscuav build-aoi --preset vienna` or `warsaw`, with the areas' centres (`ruscuav_benchmark.gis.PAPER_AOIS`, docs/BUILDING_WORLDS.md) |

The method names of the files are the notebook's: `M7_geometry_only` is `GEO-RL`,
`M8_obs_only` `OBS-RL`, `M8_reward_only` `RISK+GREEN-RL`,
`M11C_dynamic_obs_reward_only` `DYN-OBS-RL` and `M6_full_semantic` `RISK+OBS-RL`;
`static_M9` is `STATIC_Astar` and `receding_M10` `LOCAL_RH_Astar` (with the full map,
`FULL_RH_Astar`), `dynamic_M11` `DYN_RH_Astar`.
