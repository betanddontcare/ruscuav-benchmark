"""One planner episode in a :class:`~ruscuav_benchmark.env.RuscuavEnv`.

Each function resets the environment (``env.reset(seed=seed, options=options)``),
flies the planner's path with the waypoint controller and returns the episode row
with the columns of the archived v2.0.0 evaluations of that planner kind, together
with the flown path (positions, float32, one per step plus the start). The rows
take their metrics from the environment's final ``info``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .controller import waypoint_action
from .dynamic import DynamicRecedingPlanner
from .grid import path_length_xy
from .receding import RecedingHorizonPlanner
from .static import StaticPlanner

Row = Dict[str, Any]


def _seed_value(seed: Optional[int]) -> Optional[int]:
    return None if seed is None else int(seed)


def _remaining_xy(env) -> float:
    return float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))


def _exposure_columns(final_info: Dict[str, Any]) -> Row:
    return {
        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(final_info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),
        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),
        "building_collision_steps": int(final_info.get("urban_building_collision_steps", 0)),
        "canopy_collision_steps": int(final_info.get("urban_canopy_collision_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(
            final_info.get("urban_sensitive_low_altitude_violation_steps", 0)),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),
    }


def _outcome_columns(final_info: Dict[str, Any], rewards: List[float], steps: int) -> Row:
    return {
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),
        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),
    }


def run_static_episode(env, planner: StaticPlanner, seed: Optional[int] = None,
                       options: Optional[Dict[str, Any]] = None,
                       max_steps: Optional[int] = None) -> Tuple[Row, Optional[np.ndarray], Dict[str, Any]]:
    """Plan once from start to target, then fly the path. Returns (row, flown path or
    None if no path was found, plan result)."""
    cfg = planner.config
    max_steps = int(cfg.max_steps) if max_steps is None else int(max_steps)
    env.reset(seed=seed, options=options)
    start_xy = env.start_point[:2].copy()
    target_xy = env.target_point[:2].copy()
    plan_result = planner.plan(start_xy, target_xy)

    if not plan_result["success"]:
        row = {
            "episode": None, "seed": _seed_value(seed),
            "planner_success": False, "planner_reason": plan_result["reason"],
            "planner_cost": float(plan_result["cost"]), "planner_expanded": int(plan_result["expanded"]),
            "planned_path_length": np.nan,
            "termination": "planner_no_path", "is_success": False, "mission_acceptable": False,
            "total_reward": 0.0, "steps": 0, "dist_xy": float(np.linalg.norm(target_xy - start_xy)),
            "cargo_health": np.nan, "urban_mean_risk": np.nan, "urban_max_risk": np.nan,
            "urban_cumulative_risk": np.nan,
            "road_time": 0.0, "sidewalk_time": 0.0, "building_time": 0.0, "green_time": 0.0, "canopy_time": 0.0,
            "sensitive_time": 0.0, "urban_hard_violation_steps": 0,
        }
        return row, None, plan_result

    path_xy = plan_result["path_xy"]
    path_actual = [env.current_pos.copy()]
    rewards: List[float] = []
    infos: List[Dict[str, Any]] = []
    wp_idx = 1 if len(path_xy) > 1 else 0
    done, steps = False, 0
    while not done and steps < max_steps:
        action, wp_idx = waypoint_action(env, path_xy, wp_idx, cfg)
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        infos.append(dict(info))
        path_actual.append(env.current_pos.copy())
        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}
    if not done:
        final_info = dict(final_info)
        final_info["termination"] = "manual_max_steps"
    if final_info.get("dist_xy") is None:
        final_info["dist_xy"] = _remaining_xy(env)

    row = {
        "episode": None, "seed": _seed_value(seed),
        "planner_success": True, "planner_reason": plan_result["reason"],
        "planner_cost": float(plan_result["cost"]), "planner_expanded": int(plan_result["expanded"]),
        "planned_path_length": float(path_length_xy(path_xy)),
        **_outcome_columns(final_info, rewards, steps),
        **_exposure_columns(final_info),
    }
    return row, np.asarray(path_actual, dtype=np.float32), plan_result


def run_receding_episode(env, planner: RecedingHorizonPlanner, seed: Optional[int] = None,
                         options: Optional[Dict[str, Any]] = None,
                         max_steps: Optional[int] = None) -> Tuple[Row, np.ndarray]:
    """Sense and replan every ``replan_every_steps`` steps while flying. If the first
    plan fails the episode ends with ``termination="planner_no_path"``; a later
    failed replan keeps the previous plan. Returns (row, flown path)."""
    cfg = planner.config
    max_steps = int(cfg.max_steps) if max_steps is None else int(max_steps)
    env.reset(seed=seed, options=options)
    planner.reset_belief()
    planner.reveal_around_xy(env.current_pos[:2])
    target_xy = env.target_point[:2].copy()

    current_plan = None
    wp_idx = 0
    path_actual = [env.current_pos.copy()]
    rewards: List[float] = []
    infos: List[Dict[str, Any]] = []
    replan_success_count = replan_fail_count = 0
    last_plan_expanded, last_plan_time_s = 0, 0.0
    last_known_fraction = planner.known_fraction()
    no_path_info: Dict[str, Any] = {}

    done, steps = False, 0
    while not done and steps < max_steps:
        if current_plan is None or steps % int(cfg.replan_every_steps) == 0:
            planner.reveal_around_xy(env.current_pos[:2])
            result = planner.plan(start_xy=env.current_pos[:2].copy(), target_xy=target_xy)
            last_plan_expanded = int(result.get("expanded", 0))
            last_plan_time_s = float(result.get("plan_time_s", 0.0))
            last_known_fraction = float(result.get("known_fraction", planner.known_fraction()))
            if result["success"]:
                current_plan = result["path_xy"]
                wp_idx = 1 if len(current_plan) > 1 else 0
                replan_success_count += 1
            else:
                replan_fail_count += 1
                if current_plan is None:
                    no_path_info = {"termination": "planner_no_path", "is_success": False,
                                    "mission_acceptable": False, "dist_xy": _remaining_xy(env)}
                    break

        action, wp_idx = waypoint_action(env, current_plan, wp_idx, cfg)
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        infos.append(dict(info))
        path_actual.append(env.current_pos.copy())
        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else no_path_info
    if not done and final_info.get("termination", None) is None:
        final_info = dict(final_info)
        final_info.update(termination="manual_max_steps", is_success=False, mission_acceptable=False,
                          dist_xy=_remaining_xy(env))
    if final_info.get("dist_xy") is None:
        final_info["dist_xy"] = _remaining_xy(env)

    replan_count = int(planner.replan_count)
    total_plan_time_s = float(planner.total_plan_time_s)
    row = {
        "episode": None, "seed": _seed_value(seed),
        "planner_type": "M10_receding_horizon_Astar",
        "prior_risk_mode": cfg.prior_risk_mode,
        "sensing_radius_m": float(cfg.sensing_radius_m),
        "replan_every_steps": int(cfg.replan_every_steps),
        "risk_lambda": float(cfg.risk_lambda),
        "replan_count": replan_count,
        "replan_success_count": int(replan_success_count),
        "replan_fail_count": int(replan_fail_count),
        "total_expanded": int(planner.total_expanded),
        "mean_expanded_per_replan": float(planner.total_expanded / max(1, replan_count)),
        "total_plan_time_s": total_plan_time_s,
        "mean_plan_time_s": float(total_plan_time_s / max(1, replan_count)),
        "last_plan_expanded": int(last_plan_expanded),
        "last_plan_time_s": float(last_plan_time_s),
        "final_known_fraction": float(planner.known_fraction()),
        "last_known_fraction": float(last_known_fraction),
        **_outcome_columns(final_info, rewards, steps),
        **_exposure_columns(final_info),
    }
    return row, np.asarray(path_actual, dtype=np.float32)


def run_dynamic_episode(env, planner: DynamicRecedingPlanner, seed: Optional[int] = None,
                        options: Optional[Dict[str, Any]] = None,
                        max_steps: Optional[int] = None) -> Tuple[Row, np.ndarray]:
    """Receding episode of the dynamic planner in an environment with dynamic risk
    (``env.dynamic_field``). Returns (row, flown path).

    As in the archived runs, an episode whose first plan fails ends after zero steps
    with ``termination="manual_max_steps"``.
    """
    if env.dynamic_config is None:
        raise ValueError("the dynamic planner needs an environment with dynamic risk (e.g. DYN-RISK-RL)")
    cfg = planner.config
    max_steps = int(cfg.max_steps) if max_steps is None else int(max_steps)
    env.reset(seed=seed, options=options)
    planner.reset_belief(env.dynamic_field)
    planner.reveal_around_xy(env.current_pos[:2], step=env.current_step)
    target_xy = env.target_point[:2].copy()

    current_plan = None
    wp_idx = 0
    rewards: List[float] = []
    infos: List[Dict[str, Any]] = []
    path_actual = [env.current_pos.copy()]
    replan_success_count = replan_fail_count = 0

    done, steps = False, 0
    while not done and steps < max_steps:
        if current_plan is None or steps % int(cfg.replan_every_steps) == 0:
            planner.reveal_around_xy(env.current_pos[:2], step=env.current_step)
            result = planner.plan(start_xy=env.current_pos[:2].copy(), target_xy=target_xy)
            if result["success"]:
                current_plan = result["path_xy"]
                wp_idx = 1 if len(current_plan) > 1 else 0
                replan_success_count += 1
            else:
                replan_fail_count += 1
                if current_plan is None:
                    break

        action, wp_idx = waypoint_action(env, current_plan, wp_idx, cfg)
        _, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        infos.append(dict(info))
        path_actual.append(env.current_pos.copy())
        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}
    if not done and final_info.get("termination", None) is None:
        final_info = dict(final_info)
        final_info.update(termination="manual_max_steps", is_success=False, mission_acceptable=False,
                          dist_xy=_remaining_xy(env))
    if final_info.get("dist_xy") is None:
        final_info["dist_xy"] = _remaining_xy(env)

    outcome = _outcome_columns(final_info, rewards, steps)
    row = {
        "episode": None, "seed": _seed_value(seed),
        "method": "M11_dynamic_receding_Astar",
        "dynamic_prior_mode": planner.dynamic_prior_mode,
        "total_reward": outcome["total_reward"], "steps": outcome["steps"],
        "termination": outcome["termination"], "is_success": outcome["is_success"],
        "mission_acceptable": outcome["mission_acceptable"],
        "mission_acceptable_static_only": bool(final_info.get("mission_acceptable_static_only", False)),
        "dist_xy": outcome["dist_xy"], "cargo_health": outcome["cargo_health"],
        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),
        "dynamic_risk_mean": float(final_info.get("dynamic_risk_mean", np.nan)),
        "dynamic_risk_max": float(final_info.get("dynamic_risk_max", np.nan)),
        "dynamic_cumulative_risk": float(final_info.get("dynamic_cumulative_risk", np.nan)),
        "dynamic_budget_ok": bool(final_info.get("dynamic_budget_ok", False)),
        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        # not in the archive's rows, whose post-hoc acceptability therefore left out the building budget
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),
        "replan_count": int(planner.replan_count),
        "replan_success_count": int(replan_success_count),
        "replan_fail_count": int(replan_fail_count),
        "total_expanded": int(planner.total_expanded),
        "mean_expanded_per_replan": float(planner.total_expanded / max(1, planner.replan_count)),
        "total_plan_time_s": float(planner.total_plan_time_s),
        "mean_plan_time_s": float(planner.total_plan_time_s / max(1, planner.replan_count)),
        "final_known_fraction": float(planner.known_fraction()),
        "final_dynamic_known_fraction": float(planner.dynamic_known_fraction()),
    }
    return row, np.asarray(path_actual, dtype=np.float32)


def run_episode(env, planner, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None,
                max_steps: Optional[int] = None) -> Tuple[Row, Optional[np.ndarray]]:
    """Episode of any planner kind: (row, flown path)."""
    if isinstance(planner, DynamicRecedingPlanner):
        return run_dynamic_episode(env, planner, seed, options, max_steps)
    if isinstance(planner, RecedingHorizonPlanner):
        return run_receding_episode(env, planner, seed, options, max_steps)
    if isinstance(planner, StaticPlanner):
        row, path, _ = run_static_episode(env, planner, seed, options, max_steps)
        return row, path
    raise TypeError(f"not a RUSCUAV planner: {type(planner).__name__}")
