"""Fly a policy for one episode and build the episode row.

The rows have the columns of the archived evaluations, in one of three formats:

``"posthoc"``
    The primary evaluation of the learned policies (the notebook's
    ``rollout_ppo_episode``): the episode metrics, the flown path length, and
    mission acceptability recomputed post hoc from the row
    (:mod:`ruscuav_benchmark.posthoc`); the environment's own verdict is
    ``mission_acceptable_logged``.
``"dynamic"``
    The dynamic-risk evaluations (``rollout_m11_dynamic_rl_episode`` and
    ``rollout_m11c_dynamic_obs_rl_episode``): the episode metrics with the dynamic
    risk and the environment's mission acceptability, with and without the dynamic
    budget.
``"transfer"``
    The real-world transfer evaluations (``rollout_m12c_real_policy_episode``):
    the episode metrics and the environment's mission acceptability.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ..posthoc import DEFAULT_BUDGETS, DT_S, static_acceptability

Row = Dict[str, Any]

ROW_FORMATS = ("posthoc", "dynamic", "transfer")


def policy_action(policy: Any, obs: np.ndarray, deterministic: bool) -> Any:
    """The action of a policy: an object with ``predict(obs, deterministic=...)`` (as a
    Stable-Baselines3 model; the first returned value is the action) or a callable
    ``policy(obs)``."""
    predict = getattr(policy, "predict", None)
    if predict is not None:
        action, _ = predict(obs, deterministic=deterministic)
        return action
    if callable(policy):
        return policy(obs)
    raise TypeError("policy must have a predict(obs, deterministic=...) method or be callable")


def fly(env: Any, policy: Any, seed: Optional[int], options: Optional[Dict[str, Any]] = None,
        deterministic: bool = True, max_steps: int = 1000,
        on_step: Optional[Callable[[np.ndarray, Any], None]] = None) -> Tuple[Dict[str, Any], List[float], Dict[str, Any], np.ndarray, int]:
    """Reset ``env`` (with ``seed`` and reset ``options``) and fly ``policy`` until the
    episode ends or ``max_steps``. Returns (reset info, rewards, final step info,
    path as float32 positions including the start, steps)."""
    obs, reset_info = env.reset(seed=None if seed is None else int(seed), options=options)
    path = [np.asarray(env.current_pos, dtype=np.float32).copy()]
    rewards: List[float] = []
    final_info: Dict[str, Any] = {}
    done = False
    steps = 0
    while not done and steps < int(max_steps):
        action = policy_action(policy, obs, deterministic)
        if on_step is not None:
            on_step(obs, action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        final_info = dict(info)
        path.append(np.asarray(env.current_pos, dtype=np.float32).copy())
        done = bool(terminated or truncated)
        steps += 1
    return dict(reset_info), rewards, final_info, np.asarray(path, dtype=np.float32), steps


def _final_dist_xy(env: Any, final_info: Dict[str, Any]) -> float:
    if final_info.get("dist_xy") is None:
        return float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))
    return float(final_info["dist_xy"])


def _step_s(env: Any) -> float:
    """The environment's step length (``DT_S``, 0.5 s, unless configured otherwise)."""
    config = getattr(env, "config", None)
    return float(config.dynamics.dt) if config is not None else DT_S


def _total_reward(rewards: List[float]) -> float:
    return float(np.sum(rewards)) if rewards else 0.0


def posthoc_row(env: Any, seed: Optional[int], label: str, regime: Any, reset_info: Dict[str, Any],
                rewards: List[float], info: Dict[str, Any], path: np.ndarray, steps: int,
                budgets: Mapping[str, float] = DEFAULT_BUDGETS) -> Row:
    """Row of the primary policy evaluation (before the derived metrics), with mission
    acceptability recomputed with ``budgets``."""
    xy = path[:, :2]
    row: Row = {
        "episode": None,
        "seed": None if seed is None else int(seed),
        "method": str(label),
        "evaluation_regime": regime.name,
        "stage": regime.stage,
        "mode": regime.mode,
        "total_reward": _total_reward(rewards),
        "steps": int(steps),
        "episode_duration_s": float(steps * _step_s(env)),
        "termination": info.get("termination"),
        "is_success": bool(info.get("is_success", False)),
        "mission_acceptable_logged": bool(info.get("mission_acceptable", False)),
        "dist_xy": float(info["dist_xy"]) if "dist_xy" in info else float(
            np.linalg.norm(env.target_point[:2] - env.current_pos[:2])),
        "path_length_xy": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        "path_length_3d": float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
        "cargo_health": float(info.get("cargo_health", np.nan)),
        "urban_mean_risk": float(info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(info.get("urban_cumulative_risk", np.nan)),
        "road_time": float(info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(info.get("urban_building_time", 0.0)),
        "green_time": float(info.get("urban_green_time", 0.0)),
        "canopy_time": float(info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(info.get("urban_sensitive_time", 0.0)),
        "building_collision_steps": int(info.get("urban_building_collision_steps", 0)),
        "canopy_collision_steps": int(info.get("urban_canopy_collision_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(info.get("urban_sensitive_low_altitude_violation_steps", 0)),
        "urban_hard_violation_steps": int(info.get("urban_hard_violation_steps", 0)),
        "reset_safe_start_target": bool(reset_info.get("urban_safe_start_target", False)),
    }
    row["mission_acceptable"] = static_acceptability(row, budgets)
    return row


def dynamic_row(env: Any, seed: Optional[int], label: str, rewards: List[float], info: Dict[str, Any],
                steps: int) -> Row:
    """Row of the dynamic-risk policy evaluations."""
    return {
        "episode": None,
        "seed": None if seed is None else int(seed),
        "method": str(label),
        "total_reward": _total_reward(rewards),
        "steps": int(steps),
        "termination": info.get("termination", None),
        "is_success": bool(info.get("is_success", False)),
        "mission_acceptable": bool(info.get("mission_acceptable", False)),
        "mission_acceptable_static_only": bool(info.get("mission_acceptable_static_only", False)),
        "dist_xy": _final_dist_xy(env, info),
        "cargo_health": float(info.get("cargo_health", np.nan)),
        "urban_mean_risk": float(info.get("urban_mean_risk", np.nan)),
        "urban_cumulative_risk": float(info.get("urban_cumulative_risk", np.nan)),
        "dynamic_risk_mean": float(info.get("dynamic_risk_mean", np.nan)),
        "dynamic_risk_max": float(info.get("dynamic_risk_max", np.nan)),
        "dynamic_cumulative_risk": float(info.get("dynamic_cumulative_risk", np.nan)),
        "dynamic_budget_ok": bool(info.get("dynamic_budget_ok", False)),
        "road_time": float(info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(info.get("urban_building_time", 0.0)),
        "green_time": float(info.get("urban_green_time", 0.0)),
        "canopy_time": float(info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(info.get("urban_sensitive_time", 0.0)),
        "urban_hard_violation_steps": int(info.get("urban_hard_violation_steps", 0)),
    }


def transfer_row(env: Any, seed: Optional[int], label: str, rewards: List[float], info: Dict[str, Any],
                 steps: int) -> Row:
    """Row of the real-world transfer evaluations."""
    return {
        "episode": None,
        "seed": None if seed is None else int(seed),
        "policy_kind": str(label),
        "total_reward": _total_reward(rewards),
        "steps": int(steps),
        "termination": info.get("termination", None),
        "is_success": bool(info.get("is_success", False)),
        "mission_acceptable": bool(info.get("mission_acceptable", False)),
        "dist_xy": _final_dist_xy(env, info),
        "cargo_health": float(info.get("cargo_health", np.nan)),
        "urban_mean_risk": float(info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(info.get("urban_cumulative_risk", np.nan)),
        "road_time": float(info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(info.get("urban_building_time", 0.0)),
        "green_time": float(info.get("urban_green_time", 0.0)),
        "canopy_time": float(info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(info.get("urban_sensitive_time", 0.0)),
        "building_collision_steps": int(info.get("urban_building_collision_steps", 0)),
        "canopy_collision_steps": int(info.get("urban_canopy_collision_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(info.get("urban_sensitive_low_altitude_violation_steps", 0)),
        "urban_hard_violation_steps": int(info.get("urban_hard_violation_steps", 0)),
    }


def run_policy_episode(env: Any, policy: Any, seed: Optional[int], *, row_format: str, label: str,
                       regime: Any = None, options: Optional[Dict[str, Any]] = None, deterministic: bool = True,
                       max_steps: int = 1000, budgets: Mapping[str, float] = DEFAULT_BUDGETS) -> Tuple[Row, np.ndarray]:
    """Fly one episode of ``policy`` in ``env`` and return (row, flown path).

    ``label`` fills the ``method`` column (``policy_kind`` in the transfer format);
    the posthoc format also records ``regime`` (an ``EvaluationRegime``) and
    recomputes mission acceptability with ``budgets`` (the other formats keep the
    environment's verdict, from its mission configuration).
    """
    if row_format not in ROW_FORMATS:
        raise ValueError(f"row_format={row_format!r}; expected one of {list(ROW_FORMATS)}")
    if row_format == "posthoc" and regime is None:
        raise ValueError("the posthoc format records the evaluation regime; give regime")
    reset_info, rewards, info, path, steps = fly(env, policy, seed, options, deterministic, max_steps)
    if row_format == "posthoc":
        row = posthoc_row(env, seed, label, regime, reset_info, rewards, info, path, steps, budgets)
    elif row_format == "dynamic":
        row = dynamic_row(env, seed, label, rewards, info, steps)
    else:
        row = transfer_row(env, seed, label, rewards, info, steps)
    return row, path
