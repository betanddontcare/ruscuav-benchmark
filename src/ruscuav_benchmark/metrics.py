from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .risk import UrbanRiskModel
from .world import UrbanWorldLayers


@dataclass(frozen=True)
class ExposureBudgets:
    cargo_min: float = 0.80
    road_budget_s: float = 20.0
    sidewalk_budget_s: float = 5.0
    building_budget_s: float = 15.0
    sensitive_budget_s: float = 0.0
    dynamic_budget: float = 10.0
    success_radius_m: float = 30.0


def densify_path_xy(path_xy: np.ndarray, step_m: float = 5.0) -> np.ndarray:
    path_xy = np.asarray(path_xy, dtype=np.float32)
    if path_xy.ndim != 2 or path_xy.shape[1] != 2:
        raise ValueError("path_xy must have shape (N, 2)")
    if len(path_xy) <= 1:
        return path_xy.copy()
    step_m = max(float(step_m), 1e-6)
    pts = [path_xy[0]]
    for p0, p1 in zip(path_xy[:-1], path_xy[1:]):
        d = float(np.linalg.norm(p1 - p0))
        if d < 1e-9:
            continue
        n = max(1, int(np.ceil(d / step_m)))
        for k in range(1, n + 1):
            a = k / n
            pts.append(((1.0 - a) * p0 + a * p1).astype(np.float32))
    return np.asarray(pts, dtype=np.float32)


def mission_acceptable(record: Dict[str, float], budgets: ExposureBudgets = ExposureBudgets()) -> bool:
    return bool(
        record.get("success", False)
        and float(record.get("cargo_health", 0.0)) >= budgets.cargo_min
        and float(record.get("road_time_s", 0.0)) <= budgets.road_budget_s
        and float(record.get("sidewalk_time_s", 0.0)) <= budgets.sidewalk_budget_s
        and float(record.get("building_time_s", 0.0)) <= budgets.building_budget_s
        and float(record.get("sensitive_time_s", 0.0)) <= budgets.sensitive_budget_s
        and int(record.get("hard_violation_steps", 0)) == 0
    )


def evaluate_path(
    world: UrbanWorldLayers,
    risk_model: UrbanRiskModel,
    path_xy: Optional[np.ndarray],
    scenario_id: str = "scenario",
    target_xy: Optional[np.ndarray] = None,
    agl_m: float = 40.0,
    speed_mps: float = 10.0,
    dt: float = 0.5,
    budgets: ExposureBudgets = ExposureBudgets(),
) -> Dict[str, float | int | bool | str]:
    """Evaluate a waypoint path under RUSCUAV static mission-acceptability metrics.

    The path is sampled at ``speed_mps * dt`` spacing. This is intentionally a
    lightweight planner-evaluation helper, not a high-fidelity UAV dynamics
    simulation. Learned-policy environment evaluation should use the full Gym
    environment from the research code.
    """
    if path_xy is None or len(path_xy) == 0:
        return {
            "scenario_id": scenario_id,
            "success": False,
            "mission_acceptable": False,
            "termination": "no_path",
            "path_length_m": float("nan"),
            "steps": 0,
            "cargo_health": 1.0,
            "cumulative_risk": float("inf"),
            "road_time_s": 0.0,
            "sidewalk_time_s": 0.0,
            "building_time_s": 0.0,
            "green_time_s": 0.0,
            "canopy_time_s": 0.0,
            "sensitive_time_s": 0.0,
            "hard_violation_steps": 0,
            "final_distance_m": float("inf"),
        }
    sample_step_m = float(speed_mps) * float(dt)
    samples = densify_path_xy(path_xy, step_m=sample_step_m)
    total_risk = 0.0
    road_time = sidewalk_time = building_time = green_time = canopy_time = sensitive_time = 0.0
    hard_steps = 0
    for xy in samples:
        x, y = float(xy[0]), float(xy[1])
        z = risk_model.terrain_at(x, y) + float(agl_m)
        flags = risk_model.flags_at(x, y)
        cons = risk_model.constraints_at(x, y, z)
        total_risk += risk_model.risk_at(x, y, z) * float(dt)
        road_time += float(dt) if flags["road"] else 0.0
        sidewalk_time += float(dt) if flags["sidewalk"] else 0.0
        building_time += float(dt) if flags["building"] else 0.0
        green_time += float(dt) if flags["green"] else 0.0
        canopy_time += float(dt) if flags["canopy"] else 0.0
        sensitive_time += float(dt) if flags["sensitive"] else 0.0
        hard_steps += int(cons["any_hard_violation"])
    diffs = np.diff(np.asarray(path_xy, dtype=np.float32), axis=0)
    path_length = float(np.sum(np.linalg.norm(diffs, axis=1))) if len(path_xy) > 1 else 0.0
    final_distance = 0.0 if target_xy is None else float(np.linalg.norm(
        np.asarray(path_xy[-1], dtype=np.float32) - np.asarray(target_xy, dtype=np.float32)))
    success = bool(final_distance <= budgets.success_radius_m)
    record: Dict[str, float | int | bool | str] = {
        "scenario_id": scenario_id,
        "success": success,
        "termination": "success",
        "path_length_m": path_length,
        "steps": int(len(samples)),
        "cargo_health": 1.0,
        "cumulative_risk": float(total_risk),
        "road_time_s": float(road_time),
        "sidewalk_time_s": float(sidewalk_time),
        "building_time_s": float(building_time),
        "green_time_s": float(green_time),
        "canopy_time_s": float(canopy_time),
        "sensitive_time_s": float(sensitive_time),
        "hard_violation_steps": int(hard_steps),
        "final_distance_m": final_distance,
    }
    record["mission_acceptable"] = mission_acceptable(record, budgets=budgets)
    return record


def summarize_episode_records(records: List[Dict[str, float | int | bool | str]]) -> Dict[str, float | int]:
    n = len(records)
    if n == 0:
        return {"episodes": 0}
    def mean_float(key: str) -> float:
        vals = [float(r[key]) for r in records if key in r and np.isfinite(float(r[key]))]
        return float(np.mean(vals)) if vals else float("nan")
    success_count = int(sum(bool(r.get("success", False)) for r in records))
    acceptable_count = int(sum(bool(r.get("mission_acceptable", False)) for r in records))
    return {
        "episodes": n,
        "success_count": success_count,
        "acceptable_count": acceptable_count,
        "success_rate": success_count / n,
        "mission_acceptability_rate": acceptable_count / n,
        "acceptable_given_success": acceptable_count / max(1, success_count),
        "mean_cumulative_risk": mean_float("cumulative_risk"),
        "mean_path_length_m": mean_float("path_length_m"),
        "mean_steps": mean_float("steps"),
        "mean_road_time_s": mean_float("road_time_s"),
        "mean_sidewalk_time_s": mean_float("sidewalk_time_s"),
        "mean_building_time_s": mean_float("building_time_s"),
        "mean_green_time_s": mean_float("green_time_s"),
        "mean_canopy_time_s": mean_float("canopy_time_s"),
        "mean_sensitive_time_s": mean_float("sensitive_time_s"),
        "mean_hard_violation_steps": mean_float("hard_violation_steps"),
    }


def write_episode_csv(path: str | Path, records: List[Dict[str, float | int | bool | str]]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return str(path)
    keys = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)
    return str(path)
