"""Tests for the planner configurations, the planning grid and A* (NumPy only).

Episodes are tested in tests/test_planner_episodes.py.
"""

from dataclasses import asdict
from pathlib import Path
import math
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark import UrbanWorldLayers  # noqa: E402
from ruscuav_benchmark.planners import (  # noqa: E402
    PLANNER_PRESETS,
    REPORTED_DYN_RH_CONFIG,
    REPORTED_FULL_RH_CONFIG,
    REPORTED_LOCAL_RH_CONFIG,
    REPORTED_STATIC_CONFIG,
    PlannerConfig,
    RecedingPlannerConfig,
    RiskCostGrid,
    StaticPlanner,
    astar,
    get_planner,
)
from ruscuav_benchmark.planners.grid import NEIGHBOURS, nearest_free_cell  # noqa: E402

WORLD_PATH = REPO / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz"


@pytest.fixture(scope="module")
def world():
    return UrbanWorldLayers.load_npz(WORLD_PATH)


@pytest.fixture(scope="module")
def grid(world):
    return RiskCostGrid(world, PLANNER_PRESETS["STATIC_Astar"].config)


# ---------------------------------------------------------------------- configuration
def test_presets_have_the_reported_settings():
    expected = {
        "STATIC_Astar": ("static", REPORTED_STATIC_CONFIG),
        "LOCAL_RH_Astar": ("receding", REPORTED_LOCAL_RH_CONFIG),
        "FULL_RH_Astar": ("receding", REPORTED_FULL_RH_CONFIG),
        "DYN_RH_Astar": ("dynamic", REPORTED_DYN_RH_CONFIG),
    }
    assert list(PLANNER_PRESETS) == list(expected)
    for name, (kind, reported) in expected.items():
        spec = PLANNER_PRESETS[name]
        assert spec.kind == kind
        config = asdict(spec.config)
        assert {k: config[k] for k in reported} == reported
    # The reported settings of the archive (REPORTED_M9_CONFIG ... REPORTED_M11_PLANNER_CONFIG).
    assert REPORTED_STATIC_CONFIG == {"plan_res_m": 5.0, "planner_agl_m": 40.0, "risk_lambda": 0.35, "use_astar": True,
                                      "waypoint_radius_m": 8.0, "target_speed_mps": 12.0, "max_steps": 1000}
    assert REPORTED_LOCAL_RH_CONFIG["prior_risk_mode"] == "mean"
    assert REPORTED_FULL_RH_CONFIG["prior_risk_mode"] == "perfect"
    assert REPORTED_DYN_RH_CONFIG["replan_every_steps"] == 3
    assert PLANNER_PRESETS["DYN_RH_Astar"].dynamic_prior_mode == "zero"
    assert PLANNER_PRESETS["DYN_RH_Astar"].env_method == "DYN-RISK-RL"
    assert PLANNER_PRESETS["FULL_RH_Astar"].env_method == "RISK+OBS-RL"


def test_configuration_is_validated():
    with pytest.raises(ValueError, match="plan_res_m"):
        PlannerConfig(plan_res_m=0.0)
    with pytest.raises(ValueError, match="prior_risk_mode"):
        RecedingPlannerConfig(prior_risk_mode="optimistic")
    with pytest.raises(ValueError, match="replan_every_steps"):
        RecedingPlannerConfig(replan_every_steps=0)
    with pytest.raises(ValueError, match="unknown"):
        get_planner("FULL_RH_Astar", sensor_range=10.0)
    with pytest.raises(ValueError, match="Unknown planner"):
        get_planner("RRT")
    assert get_planner("FULL_RH_Astar", planner_agl_m=30.0).config.planner_agl_m == 30.0


# ---------------------------------------------------------------------- grid and A*
def test_paths_avoid_blocked_cells_and_corners(world, grid):
    planner = StaticPlanner(world, grid.config, grid=grid)
    rng = np.random.default_rng(3)
    found = 0
    for _ in range(6):
        start, target = rng.uniform(0, 1000, 2).astype(np.float32), rng.uniform(0, 1000, 2).astype(np.float32)
        result = planner.plan(start, target)
        if not result["success"]:
            continue
        found += 1
        cells = result["path_cells"]
        assert np.array_equal(result["path_xy"][0], start) and np.array_equal(result["path_xy"][-1], target)
        assert not any(grid.block[r, c] for r, c in cells)
        for (r0, c0), (r1, c1) in zip(cells, cells[1:]):
            assert (r1 - r0, c1 - c0) in NEIGHBOURS
            if r1 != r0 and c1 != c0:
                assert not grid.block[r0, c1] and not grid.block[r1, c0]
    assert found >= 4


def test_astar_and_dijkstra_find_equally_cheap_paths(grid):
    free = np.argwhere(~grid.block)
    rng = np.random.default_rng(5)
    for _ in range(3):
        a, b = (tuple(int(v) for v in free[i]) for i in rng.choice(len(free), 2, replace=False))
        cells_a, cost_a, expanded_a = astar(grid.cell_cost, grid.block, a, b, grid.plan_res_m, use_astar=True)
        cells_d, cost_d, expanded_d = astar(grid.cell_cost, grid.block, a, b, grid.plan_res_m, use_astar=False)
        assert (cells_a is None) == (cells_d is None)
        if cells_a is not None:
            assert math.isclose(cost_a, cost_d, rel_tol=1e-9)
            assert expanded_a <= expanded_d


def test_blocked_start_moves_to_the_nearest_free_cell(grid):
    blocked = tuple(int(v) for v in np.argwhere(grid.block)[0])
    free = nearest_free_cell(blocked, grid.block, grid.cell_cost)
    assert free is not None and not grid.block[free]
    assert nearest_free_cell(free, grid.block, grid.cell_cost) == free


def test_grid_reuse_requires_the_same_settings(world, grid):
    StaticPlanner(world, grid.config, grid=grid)
    with pytest.raises(ValueError, match="planner_agl_m"):
        StaticPlanner(world, PlannerConfig(planner_agl_m=30.0), grid=grid)
