# RUSCUAV v2.0.0 benchmark specification

## 1. World representation

A world is a tuple of aligned raster layers:

```text
W = (Z_T, Z_S, B, R, P, G, C, Q)
```

where `Z_T` and `Z_S` are terrain and surface elevation, and `B`, `R`, `P`, `G`, `C`, and `Q` denote buildings, roads, pedestrian ways, green areas, canopy, and sensitive zones. Main worlds use 1 m cells over a 1 km × 1 km AOI.

Semantic class, physical obstacle and risk field are distinct. Roads and pedestrian ways are exposure classes, buildings and canopy are also physical obstacles, and sensitive zones can be soft overlays or low-altitude hard constraints.

## 2. Altitude-aware semantic risk

Road, pedestrian and sensitive contributions are altitude attenuated. Buildings and canopy depend on clearance above roofs/canopy. The frozen balanced profile uses weights 0.10 (green), 0.20 (canopy), 3 (building), 6 (road), 12 (pedestrian) and 50 (sensitive). Exact scales and multipliers are listed in the supplementary PDF and `configs/`.

## 3. Action and dynamics

The PPO action is a normalized two-dimensional horizontal acceleration command. Vertical motion is produced by an internal altitude controller. The risk model is altitude-aware, but the learned policies do not freely command altitude and the A* planners are 2.5-D fixed-planner-altitude methods.

## 4. Primary no-terminal evaluation

Soft budget exceedances do not terminate an episode. An episode ends at target-region completion, hard physical violation, irrecoverable shield failure, or timeout. Success therefore measures target-region completion independently of soft budget compliance.

Mission acceptability is computed post hoc:

```text
acceptable = success
             and cargo_health >= 0.80
             and road_time <= 20 s
             and pedestrian_time <= 5 s
             and building_time <= 15 s
             and sensitive_time <= 0 s
             and hard_violation_steps == 0
```

Dynamic-risk evaluations additionally report `DynOK = dynamic_cumulative_risk <= 10`.

## 5. Secondary operational-stopping evaluation

The strict Stage-C regime terminates selected missions after a soft-budget exceedance. Its completion rate is labelled strict/operational completion and must not be pooled with primary no-terminal SR.

## 6. Required reporting

Every method should provide:

- per-episode CSV and summary JSON;
- world and scenario seed identifiers;
- exact method/environment/planner configuration;
- SR, MAR and AGS;
- cumulative risk plus risk conditional on success/acceptability;
- risk per step/time/distance where available;
- semantic exposure times and fractions;
- cargo health, hard violations and termination causes.

Learned-policy comparisons use training-seed uncertainty as primary; episode bootstrap is secondary fixed-policy uncertainty. Fixed-checkpoint paired tests are exploratory and use complete, declared Holm families.
