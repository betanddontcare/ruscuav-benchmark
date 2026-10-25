from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np

from .metrics import ExposureBudgets, evaluate_path, summarize_episode_records, write_episode_csv
from .planners import RiskAStarConfig, RiskCostAStarPlanner
from .risk import UrbanRiskModel
from .world import UrbanWorldLayers


def read_scenarios(path: str | Path, world_id: str | None = None) -> List[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if world_id is not None:
        rows = [r for r in rows if r.get("world_id") == world_id]
    return rows


def run_static_astar_evaluation(
    world_path: str | Path,
    scenarios_csv: str | Path,
    output_dir: str | Path,
    max_episodes: int | None = None,
    profile: str = "balanced",
    plan_res_m: float = 10.0,
    agl_m: float = 40.0,
    risk_lambda: float = 0.05,
) -> dict:
    world = UrbanWorldLayers.load_npz(world_path)
    risk_model = UrbanRiskModel(world, profile=profile, hard_sensitive_zones=True)
    planner = RiskCostAStarPlanner(
        world,
        risk_model,
        config=RiskAStarConfig(plan_res_m=plan_res_m, planner_agl_m=agl_m, risk_lambda=risk_lambda),
    )
    rows = read_scenarios(scenarios_csv, world_id=world.aoi_id)
    if max_episodes is not None:
        rows = rows[: int(max_episodes)]
    records = []
    for row in rows:
        start_xy = np.array([float(row["start_x"]), float(row["start_y"])], dtype=np.float32)
        target_xy = np.array([float(row["target_x"]), float(row["target_y"])], dtype=np.float32)
        path = planner.plan(start_xy, target_xy)
        rec = evaluate_path(world, risk_model, path, scenario_id=row["scenario_id"], target_xy=target_xy, agl_m=agl_m, budgets=ExposureBudgets())
        rec.update({"world_id": world.aoi_id, "method": "M9_static_risk_Astar"})
        records.append(rec)
    summary = summarize_episode_records(records)
    summary.update({"world_id": world.aoi_id, "method": "M9_static_risk_Astar", "profile": profile, "agl_m": agl_m, "plan_res_m": plan_res_m})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_episode_csv(out_dir / f"{world.aoi_id}_static_astar_episodes.csv", records)
    with (out_dir / f"{world.aoi_id}_static_astar_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run static RUSCUAV risk-cost A* evaluation.")
    parser.add_argument("--world", required=True, help="Path to urban_world_layers.npz")
    parser.add_argument("--scenarios", required=True, help="CSV with scenario seeds and start/target pairs")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument("--plan-res-m", type=float, default=10.0)
    parser.add_argument("--agl-m", type=float, default=40.0)
    parser.add_argument("--risk-lambda", type=float, default=0.05)
    args = parser.parse_args()
    summary = run_static_astar_evaluation(
        args.world, args.scenarios, args.output_dir,
        max_episodes=args.max_episodes,
        profile=args.profile,
        plan_res_m=args.plan_res_m,
        agl_m=args.agl_m,
        risk_lambda=args.risk_lambda,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
