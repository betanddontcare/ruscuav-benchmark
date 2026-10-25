"""Evaluate a planner on a world: one episode per seed or mission, a per-episode CSV
and a summary.

Each episode runs in a fresh environment of the planner's method preset
(``RISK+OBS-RL``; ``DYN-RISK-RL`` with the reported dynamic risk for the dynamic
planner) in the chosen regime, as in the archived evaluations. Needs the ``env``
extra.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..scenarios import mission_from_row, read_scenarios, scenario_seed
from ..world import UrbanWorldLayers, canonical_aoi_id
from .config import PLANNER_PRESETS, PlannerSpec, get_planner
from .dynamic import DynamicRecedingPlanner
from .grid import RiskCostGrid
from .receding import RecedingHorizonPlanner
from .rollout import run_dynamic_episode, run_receding_episode, run_static_episode
from .static import StaticPlanner

Row = Dict[str, Any]
#: (seed, reset options or None) of one episode.
Episode = Tuple[Optional[int], Optional[Dict[str, Any]]]


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


def missions_from_csv(path: Union[str, Path], world: Optional[UrbanWorldLayers] = None,
                      seed0: int = 5000) -> List[Episode]:
    """Episodes of a scenario CSV (rows of ``world`` only, when given).

    The episode seed is the row's ``seed``, or ``seed0`` plus the row index when the
    file has none. The seed sets the environment's random state and the dynamic risk
    field, so rows that share a seed (files written with one generation seed) share
    one dynamic field; a warning says so.
    """
    rows = read_scenarios(path, world_id=None if world is None else world.aoi_id)
    episodes = []
    for index, row in enumerate(rows):
        seed = scenario_seed(row)
        episodes.append((seed0 + index if seed is None else seed, mission_from_row(row)))
    seeds = [seed for seed, _ in episodes]
    if len(set(seeds)) < len(seeds):
        warnings.warn(f"{path}: {len(seeds) - len(set(seeds))} rows repeat a seed; episodes with the same seed "
                      "have the same random state and dynamic risk field", stacklevel=2)
    return episodes


def _episodes(seeds: Optional[Iterable[int]], missions: Optional[Sequence[Episode]], n_episodes: Optional[int],
              seed0: int) -> List[Episode]:
    if seeds is not None and missions is not None:
        raise ValueError("give seeds or missions, not both")
    if n_episodes is not None and int(n_episodes) < 1:
        raise ValueError(f"n_episodes must be at least 1, got {n_episodes}")
    if missions is not None:
        episodes = [(None if s is None else int(s), o) for s, o in missions]
    elif seeds is not None:
        episodes = [(int(s), None) for s in seeds]
    else:
        episodes = [(int(seed0) + i, None) for i in range(200 if n_episodes is None else int(n_episodes))]
    if missions is not None or seeds is not None:
        if n_episodes is not None:
            episodes = episodes[: int(n_episodes)]
    if not episodes:
        raise ValueError("no episodes to evaluate")
    return episodes


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
    risk_profile: str = "balanced",
    dynamic_risk: Any = None,
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

    The environment and the planner use the risk profile ``risk_profile``; the
    archived runs used ``"balanced"`` throughout.

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
    episodes = _episodes(seeds, missions, n_episodes, seed0)
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
    if prefix is None:
        prefix = f"{spec.name}_{world_id}_{regime_label}"
        preset = PLANNER_PRESETS.get(spec.name)
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


# ---------------------------------------------------------------------- output
def _missing(value: Any) -> bool:
    return value is None or (isinstance(value, (float, np.floating)) and math.isnan(float(value)))


def _column_kind(values: List[Any]) -> str:
    """The dtype pandas would give a column of these values: bool, int, float or object."""
    present = [v for v in values if not _missing(v)]
    if present and all(isinstance(v, (bool, np.bool_)) for v in present):
        return "bool" if len(present) == len(values) else "object"
    numbers = [v for v in present if isinstance(v, (int, float, np.integer, np.floating))
               and not isinstance(v, (bool, np.bool_))]
    if len(numbers) == len(present):
        if len(present) == len(values) and all(isinstance(v, (int, np.integer)) for v in present):
            return "int"
        return "float"
    return "object"


def _csv_value(value: Any, kind: str) -> str:
    if _missing(value):
        return ""
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    if kind == "float" or isinstance(value, (float, np.floating)):
        return repr(float(value))
    return str(value)


def write_rows_csv(path: Union[str, Path], rows: List[Row]) -> Path:
    """Write episode rows as ``pandas.DataFrame(rows).to_csv(index=False)`` would:
    columns in order of first appearance, empty cells for missing values and NaN,
    integer columns with gaps written as floats."""
    columns: List[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    kinds = {key: _column_kind([row.get(key) for row in rows]) for key in columns}
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_value(row.get(key), kinds[key]) for key in columns])
    return path


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    return obj


# ---------------------------------------------------------------------- summary
def _column(rows: List[Row], key: str) -> np.ndarray:
    values = []
    for row in rows:
        value = row.get(key)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=np.float64)


def _mean(rows: List[Row], key: str) -> float:
    """``pandas.Series.mean``: NaN set to 0 for the sum, divided by the count of values."""
    col = _column(rows, key)
    mask = np.isnan(col)
    count = int((~mask).sum())
    if count == 0:
        return float("nan")
    return float(np.where(mask, 0.0, col).sum() / count)


def _std(rows: List[Row], key: str) -> float:
    """Population standard deviation by a sequential two-pass sum over the values, as
    pandas computes ``Series.std(ddof=0)`` with bottleneck (the archived summaries)."""
    values = [v for v in _column(rows, key).tolist() if not math.isnan(v)]
    if not values:
        return float("nan")
    total = 0.0
    for v in values:
        total += v
    mean = total / len(values)
    squares = 0.0
    for v in values:
        squares += (v - mean) * (v - mean)
    return math.sqrt(squares / len(values))


def _median(rows: List[Row], key: str) -> float:
    col = _column(rows, key)
    col = col[~np.isnan(col)]
    return float(np.median(col)) if col.size else float("nan")


def _termination_counts(rows: List[Row]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        key = "None" if row.get("termination") is None else str(row["termination"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def summarize_planner_rows(rows: List[Row], kind: str) -> Dict[str, Any]:
    """Summary of a planner evaluation, with the keys of the archived summaries of the
    planner kind (success and mission-acceptability rates, means of the episode
    metrics, termination counts)."""
    if not rows:
        return {"n_episodes": 0}
    successes = [row for row in rows if row.get("is_success") is True]
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
