"""Tests for the RUSCUAV environment (ruscuav_benchmark.env).

The golden-trajectory test replays actions recorded from the archived notebook
environment (tests/data, written by scripts/migration/make_env_golden.py), so the
refactored environment is checked against the notebook code without the archive.
Floats are compared with a 1e-6 tolerance, since NumPy's vectorised math may round
differently on other CPUs; set RUSCUAV_EXACT_GOLDEN=1 to require identical bits
(true on the reference platform, x86-64 with NumPy 2.0.2). The exhaustive
step-by-step comparison is scripts/migration/env_parity.py.
"""

from dataclasses import replace
from pathlib import Path
import csv
import json
import os
import sys
import warnings

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark import UrbanWorldLayers  # noqa: E402
from ruscuav_benchmark.env import (  # noqa: E402
    METHOD_PRESETS,
    EpisodeConfig,
    RuscuavEnv,
    ShieldConfig,
    make_config,
    make_env,
)
from ruscuav_benchmark.env import legacy_factories  # noqa: E402

WORLD_PATH = REPO / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz"
GOLDEN_PATH = REPO / "tests" / "data" / "env_golden_AOI_SYNTH_000.json"
MISSIONS_PATH = REPO / "scenarios" / "canonical" / "AOI_SYNTH_000_seed5000_n200.csv"
VIENNA_MISSIONS_PATH = (REPO / "scenarios" / "real_gis"
                        / "vienna_prater_leopoldstadt_osm_1km_v1_footway_only_seed7000_n200.csv")
#: The processed Vienna world is not distributed (OpenStreetMap ODbL); point this to a
#: copy of vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz.
VIENNA_WORLD = os.environ.get("RUSCUAV_VIENNA_WORLD")
EXACT = os.environ.get("RUSCUAV_EXACT_GOLDEN") == "1"
TOL = {"rtol": 0.0, "atol": 0.0} if EXACT else {"rtol": 1e-6, "atol": 1e-6}

#: Golden-trajectory variants (scripts/migration/env_parity.py names) and the public API call for each.
GOLDEN_ENVS = {
    "RISK+GREEN-RL_primary": lambda w, **kw: make_env(w, "RISK+GREEN-RL", regime="primary_no_terminal", **kw),
    "RISK+GREEN-RL_secondary": lambda w, **kw: make_env(w, "RISK+GREEN-RL",
                                                        regime="secondary_operational_stopping", **kw),
    "OBS-RL_secondary": lambda w, **kw: make_env(w, "OBS-RL", regime="secondary_operational_stopping", **kw),
    "no_shields_diag_C": lambda w, **kw: make_env(w, "RISK+GREEN-RL", stage="C", mode="strict_eval",
                                                  shield=ShieldConfig(action_shield=False, geometric_shield=False),
                                                  **kw),
    "M11C_dynamic_obs_B": lambda w, **kw: make_env(w, "DYN-OBS-RL", regime="primary_no_terminal", **kw),
}


@pytest.fixture(scope="module")
def world():
    return UrbanWorldLayers.load_npz(WORLD_PATH)


def _golden_cases():
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))["cases"]


def _assert_recorded_missions(world, missions_path: Path, n: int) -> None:
    with missions_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))[:n]
    env = make_env(world, "RISK+GREEN-RL", regime="primary_no_terminal")
    for row in rows:
        _, info = env.reset(seed=int(row["seed"]))
        expected_start = [float(row[k]) for k in ("start_x", "start_y", "start_z")]
        expected_target = [float(row[k]) for k in ("target_x", "target_y", "target_z")]
        np.testing.assert_allclose(env.start_point, expected_start, atol=1e-5)
        np.testing.assert_allclose(env.target_point, expected_target, atol=1e-5)
        assert info["urban_safe_start_target"] == (row["safe_start_target"] == "True")


def test_reset_gives_the_recorded_canonical_missions(world):
    _assert_recorded_missions(world, MISSIONS_PATH, 4)


@pytest.mark.skipif(not VIENNA_WORLD, reason="RUSCUAV_VIENNA_WORLD is not set (the Vienna world is not distributed)")
def test_reset_gives_the_recorded_vienna_missions():
    provenance = json.loads(VIENNA_MISSIONS_PATH.with_suffix(".json").read_text(encoding="utf-8"))
    vienna = UrbanWorldLayers.load_npz(VIENNA_WORLD)
    assert vienna.layers_sha256() == provenance["world_layers_sha256"], (
        f"{VIENNA_WORLD} is not the world the missions were recorded in")
    _assert_recorded_missions(vienna, VIENNA_MISSIONS_PATH, 8)


@pytest.mark.parametrize("case_index", range(len(_golden_cases())))
def test_golden_trajectories_of_the_notebook_environment(world, case_index):
    case = _golden_cases()[case_index]
    overrides = {}
    if case.get("episode"):
        overrides["episode"] = replace(EpisodeConfig(), **case["episode"])
    env = GOLDEN_ENVS[case["variant"]](world, **overrides)
    obs, info = env.reset(seed=case["seed"])
    np.testing.assert_array_equal(env.start_point, np.array(case["start_point"], dtype=np.float32))
    np.testing.assert_array_equal(env.target_point, np.array(case["target_point"], dtype=np.float32))
    np.testing.assert_allclose(obs, case["reset_obs"], **TOL)
    assert info["urban_reset_attempts"] == case["reset_attempts"]
    for i, action in enumerate(case["actions"]):
        obs, reward, terminated, truncated, info = env.step(np.array(action, dtype=np.float32))
        np.testing.assert_allclose(env.current_pos, case["positions"][i], rtol=TOL["rtol"],
                                   atol=0.0 if EXACT else 1e-4)
        np.testing.assert_allclose(reward, case["rewards"][i], **TOL)
        assert info["termination"] == case["terminations"][i]
    np.testing.assert_allclose(obs, case["final_obs"], **TOL)
    for key, expected in case["final_info"].items():
        if isinstance(expected, float):
            np.testing.assert_allclose(info[key], expected, rtol=TOL["rtol"], atol=0.0 if EXACT else 1e-9,
                                       err_msg=key)
        else:
            assert info[key] == expected, key


@pytest.mark.parametrize("method", sorted(METHOD_PRESETS))
def test_every_method_runs(world, method):
    env = make_env(world, method, regime="primary_no_terminal")
    obs, info = env.reset(seed=5001)
    assert obs.shape == env.observation_space.shape and obs.dtype == np.float32
    assert ("dynamic_seed" in info) == (env.config.dynamic_risk is not None)
    for _ in range(3):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.all(np.isfinite(obs)) and np.isfinite(reward)
    for key in ("urban_cumulative_risk", "urban_road_time", "urban_sidewalk_time", "mission_acceptable",
                "urban_hard_violation_steps", "urban_shielded_any"):
        assert key in info
    assert ("dynamic_cumulative_risk" in info) == (env.config.dynamic_risk is not None)


def test_lidar_ray_count_sets_the_observation_size(world):
    from ruscuav_benchmark.env import LidarConfig

    env = make_env(world, "GEO-RL", lidar=LidarConfig(n_rays=16))
    obs, _ = env.reset(seed=5000)
    assert obs.shape == (32,) == env.observation_space.shape
    obs, *_ = env.step(np.zeros(2, dtype=np.float32))
    assert obs.shape == (32,)


def test_unseeded_reset_continues_deterministically(world):
    a = make_env(world, "DYN-RISK-RL", regime="primary_no_terminal", seed=7)
    b = make_env(world, "DYN-RISK-RL", regime="primary_no_terminal", seed=7)
    obs_a, info_a = a.reset()
    obs_b, info_b = b.reset()
    np.testing.assert_array_equal(obs_a, obs_b)
    assert info_a == info_b and info_a["dynamic_seed"] == info_b["dynamic_seed"]


def test_timeout_and_stuck_terminations(world):
    env = RuscuavEnv(world, make_config("RISK+GREEN-RL", episode=EpisodeConfig(max_steps=4)))
    env.reset(seed=5000)
    for _ in range(4):
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
    assert truncated and not terminated and info["termination"] == "timeout"

    env = RuscuavEnv(world, make_config("RISK+GREEN-RL", episode=EpisodeConfig(stuck_patience_steps=3)))
    env.reset(seed=5000)
    results = [env.step(np.zeros(2, dtype=np.float32)) for _ in range(3)]
    assert results[-1][2] and results[-1][4]["termination"] == "stuck"


def _world_like(world, **arrays):
    fields = {name: getattr(world, name) for name in world.__dataclass_fields__}
    fields.update(arrays)
    return type(world)(**fields)


def test_worlds_without_feasible_missions_are_rejected(world):
    high = _world_like(world, terrain_z=world.terrain_z + 200.0, surface_z=world.surface_z + 200.0,
                       building_roof_z=world.building_roof_z + 200.0, canopy_top_z=world.canopy_top_z + 200.0)
    with pytest.raises(ValueError, match="ceiling"):
        make_env(high, "GEO-RL")
    small = _world_like(world, **{name: getattr(world, name)[:300, :300] for name in (
        "terrain_z", "surface_z", "height_above_ground", "building_mask", "building_height", "building_roof_z",
        "road_mask", "sidewalk_mask", "green_mask", "canopy_mask", "canopy_height", "canopy_top_z",
        "canopy_density", "sensitive_mask")})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="too small"):
            make_env(small, "GEO-RL")


def test_legacy_factories_build_the_matching_presets(world):
    env = legacy_factories.make_static_eval_env("RISK-RL", world, legacy_factories.PRIMARY_NO_TERMINAL, 5000)
    assert (env.config.observation, env.config.urban_reward, env.config.stage) == ("geometry", "risk", "B")
    env = legacy_factories.make_ruscuav_m8_ablation_env_m4_mode(world, "no_action_shield", stage="C",
                                                                mode="strict_eval")
    assert env.config.observation == "semantic" and not env.config.shield.action_shield
    assert legacy_factories.M11DynamicRiskConfig(**legacy_factories.REPORTED_DYNAMIC_CONFIG) == \
        legacy_factories.reported_dynamic_config()
    with pytest.raises(ValueError, match="obstacles"):
        legacy_factories.make_ruscuav_env_m4_mode(world, n_obstacles=3)
