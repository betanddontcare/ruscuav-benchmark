"""Evaluate a planner on a world: one episode per seed or mission, a per-episode CSV
and a summary.

Each episode runs in a fresh environment of the planner's method preset
(``RISK+OBS-RL``; ``DYN-RISK-RL`` with the reported dynamic risk for the dynamic
planner) in the chosen regime, as in the archived evaluations. Needs the ``env``
extra.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..scenarios import Episode, episode_list
from ..tables import Row, is_true, jsonable, mean_of, median_of, std_of, termination_counts, write_rows_csv
from ..world import UrbanWorldLayers, canonical_aoi_id
from .config import PLANNER_PRESETS, PlannerSpec, get_planner
from .dynamic import DynamicRecedingPlanner
from .grid import RiskCostGrid
from .receding import RecedingHorizonPlanner
from .rollout import run_dynamic_episode, run_receding_episode, run_static_episode
from .static import StaticPlanner



@dataclass
class PlannerEvaluation:
    """Result of :func:`evaluate_planner`."""

    planner: PlannerSpec
    rows: List[Row]
    summary: Dict[str, Any]
    csv_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    paths: Dict[int, np.ndarray] = field(default_factory=dict)


def make_planner(spec: PlannerSpec, world: UrbanWorldLayers, risk_profile: str = "balanced",
                 grid: Optional[RiskCostGrid] = None):
    """The planner object of a :class:`PlannerSpec`."""
    if spec.kind == "static":
        return StaticPlanner(world, spec.config, risk_profile, grid)
    if spec.kind == "receding":
        return RecedingHorizonPlanner(world, spec.config, risk_profile, grid)
    return DynamicRecedingPlanner(world, spec.config, risk_profile, spec.dynamic_prior_mode, grid)


def _env_factory(spec: PlannerSpec, world: UrbanWorldLayers, regime: Optional[str], stage: Optional[str],
                 mode: Optional[str], risk_profile: str, dynamic_risk: Any) -> Tuple[Callable[[], Any], Any]:
    from .. import env as env_package

    overrides: Dict[str, Any] = {"risk_profile": risk_profile}
    if dynamic_risk is not None:
        if spec.kind != "dynamic":
            raise ValueError("dynamic_risk applies to the dynamic planner only")
        overrides["dynamic_risk"] = dynamic_risk
    if stage is not None or mode is not None:
        if regime is not None:
            raise ValueError("give a regime or a stage/mode, not both")
        config = env_package.make_config(spec.env_method, stage=None if stage is None else str(stage).upper(),
                                         mode=None if mode is None else str(mode).lower(), **overrides)
    else:
        config = env_package.make_config(spec.env_method, regime=regime or "primary_no_terminal", **overrides)
    env_class = env_package.RuscuavEnv  # needs the "env" extra
    return (lambda: env_class(world, config)), config


def evaluate_planner(
    world: UrbanWorldLayers,
    planner: Union[str, PlannerSpec] = "FULL_RH_Astar",
    *,
    regime: Optional[str] = None,
    stage: Optional[str] = None,
    mode: Optional[str] = None,
    seeds: Optional[Iterable[int]] = None,
    missions: Optional[Sequence[Episode]] = None,
    n_episodes: Optional[int] = None,
    seed0: int = 5000,
    risk_profile: Optional[str] = None,
    dynamic_risk: Any = None,
    protocol: Any = None,
    output_dir: Union[str, Path, None] = None,
    prefix: Optional[str] = None,
    save_paths: bool = False,
    progress: Optional[Callable[[int, int, Row], None]] = None,
    **config_changes: Any,
) -> PlannerEvaluation:
    """Fly ``planner`` (a preset name or a :class:`PlannerSpec`) on episodes of ``world``.

    Episodes are ``missions`` (``(seed, reset options)``, e.g. from
    :func:`missions_from_csv`), ``seeds`` (missions sampled by the environment) or,
    by default, ``n_episodes`` seeds from ``seed0`` (200 by default). ``n_episodes``
    also limits ``missions`` and ``seeds``. The regime is ``"primary_no_terminal"``
    (default) or ``"secondary_operational_stopping"``; alternatively give ``stage``
    and ``mode``. Keyword arguments ``config_changes`` change planner configuration
    fields, e.g. ``planner_agl_m=30.0``. ``dynamic_risk`` replaces the dynamic risk
    configuration of the dynamic planner's environment.

    ``protocol`` (a name or file, see :mod:`ruscuav_benchmark.protocols`) sets the
    missions (all, or the first ``n_episodes``), the regime, the risk profile, the
    acceptability budgets, the step length and the dynamic risk, which must then not
    be given; the planner's ``max_steps`` must be the protocol's step limit, and the
    world must be the protocol's.

    The environment and the planner use the risk profile ``risk_profile``
    (``"balanced"`` by default; the archived runs used it throughout).

    With ``output_dir``, ``<prefix>_episodes.csv`` (rewritten after every episode),
    ``<prefix>_summary.json`` and ``<prefix>_config.json`` are written there. The
    default prefix is ``<planner>_<world>_<regime>``, with a checksum of the
    configuration appended when it differs from the preset, so that runs with other
    settings do not overwrite each other. With ``save_paths`` the flown paths go to
    ``<prefix>_paths/episode_<i>_seed_<seed>_actual_path.npy`` (static planner: the
    planned paths also to ``<prefix>_planned_paths/``).
    """
    spec = get_planner(planner) if isinstance(planner, str) else planner
    if config_changes:
        spec = spec.with_config(**config_changes)
    protocol_record = chosen_protocol = None
    if protocol is not None:
        from ..protocols import load_protocol

        chosen_protocol = load_protocol(protocol)
        if int(spec.config.max_steps) != int(chosen_protocol.max_steps):
            raise ValueError(f"max_steps={spec.config.max_steps}: protocol {chosen_protocol.name!r} has a step limit "
                             f"of {chosen_protocol.max_steps}")
        fixed = {"regime": regime, "stage": stage, "mode": mode, "seeds": seeds, "missions": missions,
                 "risk_profile": risk_profile, "dynamic_risk": dynamic_risk}
        chosen, protocol_record = chosen_protocol.apply(world, dynamic_method=spec.kind == "dynamic",
                                                        n_episodes=n_episodes, given=fixed)
        missions, regime, risk_profile = chosen["missions"], chosen["regime"], chosen["risk_profile"]
        dynamic_risk, n_episodes = chosen.get("dynamic_risk"), None
    risk_profile = "balanced" if risk_profile is None else risk_profile
    episodes = episode_list(seeds, missions, n_episodes, seed0)
    if chosen_protocol is not None:
        from .. import env as env_package

        env_config = chosen_protocol.environment_config(spec.env_method, dynamic_risk=spec.kind == "dynamic")
        chosen_protocol.check_environment(env_config)

        def new_env():
            return env_package.RuscuavEnv(world, env_config)
    else:
        new_env, env_config = _env_factory(spec, world, regime, stage, mode, risk_profile, dynamic_risk)
    grid = RiskCostGrid(world, spec.config, risk_profile)
    planner_obj = make_planner(spec, world, risk_profile, grid)

    world_id = canonical_aoi_id(world.aoi_id)
    regime_label = (regime or "primary_no_terminal") if stage is None and mode is None else \
        f"stage_{env_config.stage}_{env_config.mode}"
    settings = {
        "planner": spec.to_dict(),
        "world_id": world_id,
        "regime": regime_label,
        "environment": {
            "method": spec.env_method, "stage": env_config.stage, "mode": env_config.mode,
            "operational_stopping": bool(env_config.resolved_terminate_on_urban_constraint),
            "risk_profile": risk_profile,
            "dynamic_risk": None if env_config.dynamic_risk is None else asdict(env_config.dynamic_risk),
        },
    }
    if protocol_record is not None:
        settings["protocol"] = protocol_record
    if prefix is None:
        preset = PLANNER_PRESETS.get(spec.name)
        if protocol_record is not None:
            prefix = f"{spec.name}_{protocol_record['name']}"
            if not protocol_record["complete"]:
                prefix += f"_n{protocol_record['missions_flown']}"
            custom = preset is None or preset.to_dict() != spec.to_dict()
        else:
            prefix = f"{spec.name}_{world_id}_{regime_label}"
            custom = preset is None or preset.to_dict() != spec.to_dict() or dynamic_risk is not None \
                or risk_profile != "balanced"
        if custom:
            digest = hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
            prefix += f"_cfg{digest[:8]}"
    out_dir = None if output_dir is None else Path(output_dir)
    csv_path = summary_path = None
    paths_dir = planned_dir = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{prefix}_episodes.csv"
        summary_path = out_dir / f"{prefix}_summary.json"
        (out_dir / f"{prefix}_config.json").write_text(json.dumps(_jsonable(settings), indent=2) + "\n",
                                                       encoding="utf-8")
        if save_paths:
            paths_dir = out_dir / f"{prefix}_paths"
            paths_dir.mkdir(exist_ok=True)
            if spec.kind == "static":
                planned_dir = out_dir / f"{prefix}_planned_paths"
                planned_dir.mkdir(exist_ok=True)

    rows: List[Row] = []
    paths: Dict[int, np.ndarray] = {}
    for index, (seed, options) in enumerate(episodes):
        env = new_env()
        try:
            if spec.kind == "static":
                row, path, plan_result = run_static_episode(env, planner_obj, seed, options)
                if planned_dir is not None and plan_result.get("path_xy") is not None:
                    np.save(planned_dir / f"episode_{index:05d}_seed_{seed}_planned_path.npy", plan_result["path_xy"])
            elif spec.kind == "receding":
                row, path = run_receding_episode(env, planner_obj, seed, options)
            else:
                row, path = run_dynamic_episode(env, planner_obj, seed, options)
        finally:
            env.close()
        row["episode"] = index
        rows.append(row)
        if path is not None:
            paths[index] = path
            if paths_dir is not None:
                np.save(paths_dir / f"episode_{index:05d}_seed_{seed}_actual_path.npy", path)
        if csv_path is not None:
            write_rows_csv(csv_path, rows)
        if progress is not None:
            progress(index, len(episodes), row)

    summary = summarize_planner_rows(rows, spec.kind)
    summary.update({
        **settings,
        "legacy_world_id": world.aoi_id,
        "seeds": [row["seed"] for row in rows],
        "missions_given": missions is not None,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    if summary_path is not None:
        summary["csv_path"] = str(csv_path)
        summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    return PlannerEvaluation(spec, rows, summary, csv_path, summary_path, paths)


# ---------------------------------------------------------------------- output and summary
# Shared with the policy evaluations (ruscuav_benchmark.tables).
_jsonable = jsonable
_mean = mean_of
_std = std_of
_median = median_of
_termination_counts = termination_counts


def summarize_planner_rows(rows: List[Row], kind: str) -> Dict[str, Any]:
    """Summary of a planner evaluation, with the keys of the archived summaries of the
    planner kind (success and mission-acceptability rates, means of the episode
    metrics, termination counts)."""
    if not rows:
        return {"n_episodes": 0}
    successes = [row for row in rows if is_true(row.get("is_success"))]
    s: Dict[str, Any] = {"n_episodes": len(rows)}
    if kind == "static":
        s["planner_path_found_rate"] = _mean(rows, "planner_success")
    s["success_rate"] = _mean(rows, "is_success")
    s["mission_acceptability_rate"] = _mean(rows, "mission_acceptable")
    if kind == "dynamic":
        s["static_only_acceptability_rate"] = _mean(rows, "mission_acceptable_static_only")
    s.update({
        "mean_reward": _mean(rows, "total_reward"),
        "std_reward": _std(rows, "total_reward"),
        "mean_steps": _mean(rows, "steps"),
        "mean_dist_xy": _mean(rows, "dist_xy"),
        "median_dist_xy": _median(rows, "dist_xy"),
        "mean_cargo_health": _mean(rows, "cargo_health"),
        "mean_urban_risk": _mean(rows, "urban_mean_risk"),
    })
    if kind == "dynamic":
        s.update({
            "mean_urban_cumulative_risk": _mean(rows, "urban_cumulative_risk"),
            "mean_dynamic_risk": _mean(rows, "dynamic_risk_mean"),
            "mean_dynamic_risk_max": _mean(rows, "dynamic_risk_max"),
            "mean_dynamic_cumulative_risk": _mean(rows, "dynamic_cumulative_risk"),
            "dynamic_budget_ok_rate": _mean(rows, "dynamic_budget_ok"),
            "mean_road_time": _mean(rows, "road_time"),
            "mean_sidewalk_time": _mean(rows, "sidewalk_time"),
            "mean_green_time": _mean(rows, "green_time"),
            "mean_sensitive_time": _mean(rows, "sensitive_time"),
            "mean_hard_violation_steps": _mean(rows, "urban_hard_violation_steps"),
        })
    else:
        s.update({
            "mean_urban_max_risk": _mean(rows, "urban_max_risk"),
            "mean_cumulative_risk": _mean(rows, "urban_cumulative_risk"),
            "mean_road_time": _mean(rows, "road_time"),
            "mean_sidewalk_time": _mean(rows, "sidewalk_time"),
            "mean_building_time": _mean(rows, "building_time"),
            "mean_green_time": _mean(rows, "green_time"),
            "mean_canopy_time": _mean(rows, "canopy_time"),
            "mean_sensitive_time": _mean(rows, "sensitive_time"),
            "mean_hard_violation_steps": _mean(rows, "urban_hard_violation_steps"),
        })
    if kind == "static":
        costs = [dict(planner_cost=(np.nan if not math.isfinite(row["planner_cost"]) else row["planner_cost"]))
                 for row in rows]
        s.update({
            "mean_planner_cost": _mean(costs, "planner_cost"),
            "mean_planner_expanded": _mean(rows, "planner_expanded"),
            "mean_planned_path_length": _mean(rows, "planned_path_length"),
        })
    elif kind == "receding":
        s.update({
            "mean_replan_count": _mean(rows, "replan_count"),
            "mean_replan_success_count": _mean(rows, "replan_success_count"),
            "mean_replan_fail_count": _mean(rows, "replan_fail_count"),
            "mean_total_expanded": _mean(rows, "total_expanded"),
            "mean_expanded_per_replan": _mean(rows, "mean_expanded_per_replan"),
            "mean_total_plan_time_s": _mean(rows, "total_plan_time_s"),
            "mean_plan_time_s": _mean(rows, "mean_plan_time_s"),
            "mean_final_known_fraction": _mean(rows, "final_known_fraction"),
        })
    s["termination_counts"] = _termination_counts(rows)
    s["acceptable_given_success_rate"] = _mean(successes, "mission_acceptable") if successes else 0.0
    return s
