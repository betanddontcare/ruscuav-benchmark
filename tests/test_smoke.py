from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruscuav_benchmark.synthetic import create_synthetic_world
from ruscuav_benchmark.risk import UrbanRiskModel
from ruscuav_benchmark.planners import RiskCostAStarPlanner
from ruscuav_benchmark.metrics import evaluate_path


def test_synthetic_world_gate_and_planner_smoke():
    world = create_synthetic_world(aoi_id="TEST", seed=7, width=200, height=200)
    risk = UrbanRiskModel(world)
    planner = RiskCostAStarPlanner(world, risk)
    start = np.array([20.0, 20.0], dtype=np.float32)
    target = np.array([180.0, 180.0], dtype=np.float32)
    path = planner.plan(start, target)
    rec = evaluate_path(world, risk, path, scenario_id="smoke")
    assert "success" in rec
    assert "mission_acceptable" in rec
