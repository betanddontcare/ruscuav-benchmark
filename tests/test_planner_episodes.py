"""Planner episodes in ruscuav_benchmark.env (needs the "env" extra).

The golden test flies the cases recorded from the archived notebook code
(tests/data, written by scripts/migration/make_planner_golden.py) and compares every
plan, action, position and the episode row. Floats are compared with a 1e-6
tolerance unless RUSCUAV_EXACT_GOLDEN=1 (identical bits; true on the reference
platform). The exhaustive comparison is scripts/migration/planner_parity.py.
"""

from pathlib import Path
import csv
import json
import os
import sys

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark import UrbanWorldLayers  # noqa: E402
from ruscuav_benchmark.env import make_env  # noqa: E402
from ruscuav_benchmark.planners import (  # noqa: E402
    DynamicRecedingPlanner,
    RecedingHorizonPlanner,
    StaticPlanner,
    evaluate_planner,
    get_planner,
    missions_from_csv,
    run_episode,
)
from ruscuav_benchmark.planners.rollout import (  # noqa: E402
    run_dynamic_episode,
    run_receding_episode,
    run_static_episode,
)

WORLD_PATH = REPO / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz"
GOLDEN_PATH = REPO / "tests" / "data" / "planner_golden_AOI_SYNTH_000.json"
MISSIONS_PATH = REPO / "scenarios" / "canonical" / "AOI_SYNTH_000_seed5000_n200.csv"
EXACT = os.environ.get("RUSCUAV_EXACT_GOLDEN") == "1"
TIMING_KEYS = {"total_plan_time_s", "mean_plan_time_s", "last_plan_time_s"}


@pytest.fixture(scope="module")
def world():
    return UrbanWorldLayers.load_npz(WORLD_PATH)


class RecordingEnv:
    """Forwards to an environment and records the actions and positions."""

    def __init__(self, env):
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "actions", [])
        object.__setattr__(self, "positions", [])
        object.__setattr__(self, "reset_info", None)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        object.__setattr__(self, "reset_info", dict(out[1]))
        return out

    def step(self, action):
        self.actions.append(np.array(action, dtype=np.float32))
        out = self._env.step(action)
        self.positions.append(np.array(self._env.current_pos))
        return out


def record_plans(planner):
    plans = []
    original = planner.plan

    def plan(start_xy, target_xy):
        result = original(start_xy, target_xy)
        plans.append((np.array(start_xy, dtype=np.float32), result))
        return result

    planner.plan = plan
    return plans


def modified_world(world, target_xy, kind):
    """The world with buildings over the goal (within 125 m, Chebyshev distance:
    "goal_blocked") or in a wall 60 to 80 m around it ("goal_enclosed"), as in
    scripts/migration/planner_parity.py."""
    rows, cols = world.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    cheb = np.maximum(np.abs(xx + 0.5 - float(target_xy[0])), np.abs(yy + 0.5 - float(target_xy[1])))
    area = cheb <= 125.0 if kind == "goal_blocked" else (cheb >= 60.0) & (cheb <= 80.0)
    copy = UrbanWorldLayers.__new__(UrbanWorldLayers)
    copy.__dict__.update(world.__dict__)
    copy.building_mask = np.where(area, 1.0, world.building_mask).astype(np.float32)
    copy.building_height = np.where(area, 200.0, world.building_height).astype(np.float32)
    copy.building_roof_z = (world.terrain_z + copy.building_height).astype(np.float32)
    return copy


def _close(a, b):
    if EXACT:
        return np.array_equal(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), equal_nan=True)
    return np.allclose(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), rtol=1e-6, atol=1e-6,
                       equal_nan=True)


def _golden_cases():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case_index", range(len(_golden_cases())))
def test_golden_episodes_of_the_notebook_planners(world, case_index):
    case = _golden_cases()[case_index]
    spec = get_planner(case["planner"], **case["changes"])
    env = RecordingEnv(make_env(world, spec.env_method, stage=case["stage"], mode=case["mode"]))
    planner_world = world
    if case["world_mod"] is not None:
        probe = make_env(world, "RISK+OBS-RL", stage=case["stage"], mode=case["mode"])
        probe.reset(seed=case["seed"])
        planner_world = modified_world(world, probe.target_point[:2], case["world_mod"])
    if spec.kind == "static":
        planner = StaticPlanner(planner_world, spec.config)
        plans = record_plans(planner)
        row, _, _ = run_static_episode(env, planner, case["seed"])
    elif spec.kind == "receding":
        planner = RecedingHorizonPlanner(planner_world, spec.config)
        plans = record_plans(planner)
        row, _ = run_receding_episode(env, planner, case["seed"])
    else:
        planner = DynamicRecedingPlanner(planner_world, spec.config, dynamic_prior_mode=case["dynamic_prior_mode"])
        plans = record_plans(planner)
        row, _ = run_dynamic_episode(env, planner, case["seed"])

    expected = case["row"]
    keys = list(expected)
    if spec.kind == "dynamic":
        # the package's rows of the dynamic planner add the building time, which the notebook's left out
        keys.insert(keys.index("sidewalk_time") + 1, "building_time")
        assert row["building_time"] == float(env.exposure.building_time)
    assert list(row) == keys
    for key, value in expected.items():
        if key in TIMING_KEYS:
            continue
        if isinstance(value, float) or isinstance(row[key], float):
            assert _close(row[key], value), key
        else:
            assert row[key] == value, key
    assert len(env.actions) == len(case["actions"])
    assert _close(env.actions, case["actions"]) and _close(env.positions, case["positions"])
    assert len(plans) == len(case["plans"])
    for (start, got), want in zip(plans, case["plans"]):
        assert got["reason"] == want["reason"] and got["expanded"] == want["expanded"]
        assert _close(start, want["start"]) and _close(got["cost"], want["cost"])
        assert (got["path_xy"] is None) == (want["path"] is None)
        if want["path"] is not None:
            assert _close(got["path_xy"], want["path"])
    assert env.reset_info == case["reset_info"]


def test_options_reproduce_the_recorded_mission_exactly(world):
    with MISSIONS_PATH.open(newline="", encoding="utf-8") as f:
        row = next(csv.DictReader(f))
    (seed, options), = missions_from_csv(MISSIONS_PATH, world)[:1]
    assert seed == int(row["seed"])
    plain, given = make_env(world, "RISK+OBS-RL"), make_env(world, "RISK+OBS-RL")
    obs_a, info_a = plain.reset(seed=seed)
    obs_b, info_b = given.reset(seed=seed, options=options)
    assert info_b.pop("mission_source") == "options"
    assert np.array_equal(obs_a, obs_b) and info_a == info_b
    assert np.array_equal(plain.start_point, given.start_point)
    assert np.array_equal(plain.target_point, given.target_point)
    for _ in range(5):
        out_a, out_b = plain.step([0.3, -0.2]), given.step([0.3, -0.2])
        assert np.array_equal(out_a[0], out_b[0]) and out_a[1] == out_b[1] and out_a[4] == out_b[4]


def test_options_fly_any_mission(world):
    env = make_env(world, "RISK+OBS-RL")
    _, info = env.reset(seed=1, options={"start": (100.0, 120.0), "target": [800.5, 900.0, 60.0]})
    assert info["mission_source"] == "options"
    assert env.start_point[:2].tolist() == [100.0, 120.0]
    assert env.target_point.tolist() == pytest.approx([800.5, 900.0, 60.0])
    ground = env.dynamics.ground_at(100.0, 120.0)
    assert env.start_point[2] == pytest.approx(ground + env.config.dynamics.cruise_agl_m, abs=1e-4)
    assert np.array_equal(env.current_pos, env.start_point)
    with pytest.raises(ValueError, match="outside the world"):
        env.reset(seed=1, options={"start": (-5.0, 10.0), "target": (500.0, 500.0)})
    with pytest.raises(ValueError, match="both"):
        env.reset(seed=1, options={"start": (5.0, 10.0)})
    with pytest.raises(ValueError, match="unknown reset options"):
        env.reset(seed=1, options={"start": (5.0, 10.0), "target": (500.0, 500.0), "wind": 3})


def test_evaluate_planner_writes_the_episode_files(world, tmp_path):
    missions = missions_from_csv(MISSIONS_PATH, world)[:2]
    result = evaluate_planner(world, "LOCAL_RH_Astar", missions=missions, max_steps=8, output_dir=tmp_path,
                              save_paths=True)
    assert [row["seed"] for row in result.rows] == [5000, 5001]
    assert [row["episode"] for row in result.rows] == [0, 1]
    assert all(row["termination"] == "manual_max_steps" and row["steps"] == 8 for row in result.rows)
    # A configuration that differs from the preset gets its own file names.
    assert result.csv_path.name.startswith("LOCAL_RH_Astar_AOI_SYNTH_000_primary_no_terminal_cfg")
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["n_episodes"] == 2 and summary["planner"]["config"]["max_steps"] == 8
    assert summary["termination_counts"] == {"manual_max_steps": 2}
    assert summary["environment"]["method"] == "RISK+OBS-RL" and summary["environment"]["stage"] == "B"
    config = json.loads(result.csv_path.with_name(result.csv_path.name.replace("_episodes.csv", "_config.json"))
                        .read_text(encoding="utf-8"))
    assert config["environment"]["operational_stopping"] is False and config["environment"]["dynamic_risk"] is None
    with result.csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0])[:4] == ["episode", "seed", "planner_type", "prior_risk_mode"]
    assert rows[1]["seed"] == "5001" and rows[1]["mission_acceptable"] == "False"
    assert len(list(tmp_path.glob("*_paths/*.npy"))) == 2
    assert result.paths[0].shape == (9, 3)


def test_evaluate_planner_settings(world, tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        evaluate_planner(world, "STATIC_Astar", n_episodes=0)
    with pytest.raises(ValueError, match="not both"):
        evaluate_planner(world, "STATIC_Astar", regime="primary_no_terminal", stage="C")
    with pytest.raises(ValueError, match="dynamic planner only"):
        evaluate_planner(world, "STATIC_Astar", dynamic_risk={"n_hotspots": 2})
    result = evaluate_planner(world, "DYN_RH_Astar", seeds=[5000], stage="c", mode="STRICT_EVAL", max_steps=2,
                              dynamic_prior_mode="mean", output_dir=tmp_path, prefix="dyn")
    assert result.summary["regime"] == "stage_C_strict_eval"
    assert result.summary["environment"]["operational_stopping"] is True
    assert result.summary["environment"]["dynamic_risk"]["n_hotspots"] == 16
    assert result.rows[0]["dynamic_prior_mode"] == "mean"
    assert (tmp_path / "dyn_episodes.csv").exists()


def test_run_episode_dispatches_on_the_planner_kind(world):
    spec = get_planner("STATIC_Astar", max_steps=3)
    row, path = run_episode(make_env(world, spec.env_method), StaticPlanner(world, spec.config), seed=5002)
    assert row["steps"] == 3 and path.shape == (4, 3) and row["planner_success"] is True
    with pytest.raises(TypeError):
        run_episode(make_env(world, spec.env_method), object(), seed=5002)


def test_dynamic_planner_needs_a_dynamic_environment(world):
    spec = get_planner("DYN_RH_Astar", max_steps=3)
    planner = DynamicRecedingPlanner(world, spec.config)
    with pytest.raises(ValueError, match="dynamic risk"):
        run_dynamic_episode(make_env(world, "RISK+OBS-RL"), planner, seed=5000)


def test_cli_evaluate(tmp_path, capsys):
    from ruscuav_benchmark.cli import main

    code = main(["evaluate", "--world", str(WORLD_PATH), "--planner", "STATIC_Astar", "--episodes", "1",
                 "--set", "max_steps=4", "--set", "use_astar=no", "--output-dir", str(tmp_path), "--prefix", "quick",
                 "--quiet"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["planner"] == "STATIC_Astar" and out["n_episodes"] == 1
    config = json.loads((tmp_path / "quick_config.json").read_text(encoding="utf-8"))
    assert config["planner"]["config"]["use_astar"] is False and config["planner"]["config"]["max_steps"] == 4
    assert (tmp_path / "quick_episodes.csv").exists()
    base = ["evaluate", "--world", str(WORLD_PATH), "--output-dir", str(tmp_path), "--quiet"]
    for bad in (["--set", "nonsense=1"], ["--set", "max_steps=many"], ["--set", "use_astar=maybe"],
                ["--set", "max_steps"], ["--set", "max_steps=0"], ["--episodes", "0"],
                ["--planner", "STATIC_Astar", "--set", "dynamic_prior_mode=mean"]):
        assert main(base + bad) == 2, bad
    assert "ruscuav evaluate:" in capsys.readouterr().err
