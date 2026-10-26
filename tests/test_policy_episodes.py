"""Policy episodes in ruscuav_benchmark.env: golden episodes of the notebook's rollout
functions, and evaluate_policy (files, formats, missions, checks).

The golden episodes (tests/data/policy_golden_AOI_SYNTH_000.json, written by
scripts/migration/make_policy_golden.py) fly the simple policies of
ruscuav_benchmark.policies.simple, so the test needs neither PyTorch nor the archive.
With RUSCUAV_EXACT_GOLDEN=1 rows and digests must be identical bit for bit.
"""

from pathlib import Path
import csv
import hashlib
import json
import math
import os
import sys

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark import UrbanWorldLayers  # noqa: E402
from ruscuav_benchmark.cli import main as cli_main  # noqa: E402
from ruscuav_benchmark.env import REGIMES, RuscuavEnv, make_config  # noqa: E402
from ruscuav_benchmark.policies import (  # noqa: E402
    ConstantPolicy,
    TargetSeekingPolicy,
    evaluate_policy,
    run_policy_episode,
)
from ruscuav_benchmark.posthoc import with_derived_metrics  # noqa: E402
from ruscuav_benchmark.scenarios import missions_from_csv  # noqa: E402

WORLD = REPO / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz"
GOLDEN = json.loads((REPO / "tests" / "data" / "policy_golden_AOI_SYNTH_000.json").read_text(encoding="utf-8"))
MISSIONS = REPO / "scenarios" / "canonical" / "AOI_SYNTH_000_seed5000_n200.csv"
EXACT = os.environ.get("RUSCUAV_EXACT_GOLDEN") == "1"
POLICIES = {
    "seek": lambda: TargetSeekingPolicy(),
    "seek_undamped": lambda: TargetSeekingPolicy(damping=0.0),
    "constant": lambda: ConstantPolicy([1.0, 0.2]),
}


@pytest.fixture(scope="module")
def world():
    return UrbanWorldLayers.load_npz(WORLD)


class Digest:
    def __init__(self, policy):
        self.policy, self.calls = policy, 0
        self.obs, self.actions = hashlib.sha256(), hashlib.sha256()

    def predict(self, obs, deterministic=True):
        action, state = self.policy.predict(obs, deterministic=deterministic)
        self.obs.update(np.ascontiguousarray(obs).tobytes())
        self.actions.update(np.ascontiguousarray(action).tobytes())
        self.calls += 1
        return action, state


def _close(expected, actual):
    if expected is None:
        return actual is None or (isinstance(actual, float) and math.isnan(actual))
    if isinstance(expected, bool) or isinstance(expected, str):
        return expected == actual
    if isinstance(expected, int) and not isinstance(actual, float):
        return expected == actual
    expected, actual = float(expected), float(actual)
    if EXACT:
        return expected == actual
    return math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9)


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
def test_golden_episodes(world, case):
    regime = REGIMES["primary_no_terminal" if case["regime"] == "primary" else "secondary_operational_stopping"]
    env = RuscuavEnv(world, make_config(case["method"], regime=regime.name))
    policy = Digest(POLICIES[case["policy"]]())
    row, path = run_policy_episode(env, policy, case["seed"], row_format=case["row_format"], label=case["label"],
                                   regime=regime, max_steps=case["max_steps"])
    row["episode"] = 0
    if case["row_format"] == "posthoc":
        row = with_derived_metrics([row])[0]
    expected = case["row"]
    assert list(row) == list(expected)
    for key, value in expected.items():
        assert _close(value, row[key]), (key, value, row[key])
    assert policy.calls == case["policy_calls"] and len(path) == case["path_points"]
    if EXACT:
        assert policy.obs.hexdigest() == case["obs_sha256"]
        assert policy.actions.hexdigest() == case["actions_sha256"]
        assert hashlib.sha256(np.ascontiguousarray(path).tobytes()).hexdigest() == case["path_sha256"]


def test_golden_episodes_cover_every_format_and_ending():
    formats = {c["row_format"] for c in GOLDEN["cases"]}
    endings = {c["row"]["termination"] for c in GOLDEN["cases"]}
    assert formats == {"posthoc", "dynamic", "transfer"}
    assert {"success", "shield_stuck", "stuck", None} <= endings
    assert any(c["regime"] == "secondary" for c in GOLDEN["cases"])


def test_evaluate_policy_writes_the_archived_files(world, tmp_path):
    result = evaluate_policy(world, TargetSeekingPolicy(), "RISK+GREEN-RL", seeds=[5000, 5011], output_dir=tmp_path,
                             save_paths=True)
    prefix = result.csv_path.name[: -len("_episodes.csv")]
    assert prefix.startswith("RISK+GREEN-RL_AOI_SYNTH_000_primary_no_terminal_") and len(prefix.split("_")[-1]) == 8
    with result.csv_path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header[:6] == ["episode", "seed", "method", "evaluation_regime", "stage", "mode"]
    assert header[-4:] == ["urban_risk_per_m_xy", "road_time_per_m_xy", "ped_time_per_m_xy", "building_time_per_m_xy"]
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    config = json.loads((tmp_path / f"{prefix}_config.json").read_text(encoding="utf-8"))
    assert summary["n_episodes"] == 2 and summary["seeds"] == [5000, 5011]
    assert summary["termination_counts"] == {"success": 1, "shield_stuck": 1}
    assert summary["mission_acceptability_rate"] == 0.5 and summary["mean_steps_given_success"] == 114.0
    assert config["policy"]["kind"] == "TargetSeekingPolicy" and config["regime"]["stage"] == "B"
    assert config["environment"]["resolved"]["terminate_on_urban_constraint"] is False
    assert sorted(p.name for p in (tmp_path / f"{prefix}_paths").iterdir()) == [
        "episode_00000_seed_5000_path.npy", "episode_00001_seed_5011_path.npy"]
    assert [r["mission_acceptable"] for r in result.rows] == [r["mission_acceptable_posthoc_static"] for r in result.rows]


def test_csv_equals_pandas(world, tmp_path):
    pd = pytest.importorskip("pandas")
    result = evaluate_policy(world, ConstantPolicy([0.0, 0.0]), "DYN-OBS-RL", seeds=[5003], max_steps=3,
                             output_dir=tmp_path)
    assert result.row_format == "dynamic" and result.rows[0]["method"] == "RL_dynamic_obs_reward_only"
    expected = pd.DataFrame(result.rows).to_csv(index=False, lineterminator="\n")
    assert result.csv_path.read_text(encoding="utf-8") == expected


def test_missions_reproduce_seeded_episodes(world):
    missions = missions_from_csv(MISSIONS, world)[:2]
    a = evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", missions=missions)
    b = evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", seeds=[5000, 5001])
    assert a.rows == b.rows
    assert a.summary["missions_given"] is True and b.summary["missions_given"] is False


def test_formats_labels_and_regimes(world):
    r = evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", seeds=[5003], row_format="transfer", max_steps=5)
    assert r.rows[0]["policy_kind"] == "geometry_only" and "method" not in r.rows[0]
    r = evaluate_policy(world, TargetSeekingPolicy(), "DYN-RISK-RL", seeds=[5003], max_steps=5, label="mine")
    assert r.row_format == "dynamic" and r.rows[0]["method"] == "mine"
    r = evaluate_policy(world, TargetSeekingPolicy(), "OBS-RL", seeds=[5003], max_steps=5, stage="c", mode="strict_eval")
    assert r.summary["regime"]["name"] == "stage_C_strict_eval" and r.rows[0]["stage"] == "C"
    r = evaluate_policy(world, lambda obs: np.zeros(2, dtype=np.float32), "RISK+OBS-RL", seeds=[5003], max_steps=3)
    assert r.rows[0]["steps"] == 3 and r.summary["policy"]["kind"] == "function"


class _WrongShape:
    class observation_space:
        shape = (5,)

    def predict(self, obs, deterministic=True):
        return np.zeros(2, dtype=np.float32), None


def test_evaluate_policy_rejects_bad_settings(world):
    seek = TargetSeekingPolicy()
    with pytest.raises(ValueError, match="row_format"):
        evaluate_policy(world, seek, "GEO-RL", seeds=[5000], row_format="paper")
    with pytest.raises(ValueError, match="regime or a stage"):
        evaluate_policy(world, seek, "GEO-RL", seeds=[5000], regime="primary_no_terminal", stage="B")
    with pytest.raises(ValueError, match="Unknown regime"):
        evaluate_policy(world, seek, "GEO-RL", seeds=[5000], regime="primary")
    with pytest.raises(ValueError, match="n_episodes"):
        evaluate_policy(world, seek, "GEO-RL", n_episodes=0)
    with pytest.raises(ValueError, match="shape"):
        evaluate_policy(world, _WrongShape(), "GEO-RL", seeds=[5000])
    with pytest.raises(ValueError, match="Unknown method"):
        evaluate_policy(world, seek, "PPO", seeds=[5000])
    env = RuscuavEnv(world, make_config("GEO-RL"))
    with pytest.raises(ValueError, match="regime"):
        run_policy_episode(env, seek, 5000, row_format="posthoc", label="x")
    with pytest.raises(TypeError, match="predict"):
        run_policy_episode(env, object(), 5000, row_format="dynamic", label="x")


@pytest.mark.parametrize("extra, message", [
    (["--policy", "p.zip"], "needs --method"),
    (["--policy", "p.zip", "--method", "GEO-RL", "--set", "planner_agl_m=30"], "--set"),
    (["--checkpoint", "GEO-RL/1001"], "needs --archive"),
    (["--checkpoint", "GEO-RL/9", "--archive", "."], "unknown checkpoint"),
    (["--policy", "p.zip", "--checkpoint", "GEO-RL/1001"], "give one of"),
    (["--evaluation", "static/GEO-RL/1001"], "needs --archive"),
    (["--evaluation", "static/GEO-RL/9", "--archive", "."], "unknown archived evaluation"),
    (["--evaluation", "static/GEO-RL/1001", "--archive", ".", "--method", "OBS-RL"], "fixes these settings"),
    (["--evaluation", "vienna/GEO-RL/1001", "--archive", "."], "not in the archive"),
    (["--planner", "STATIC_Astar", "--policy", "p.zip", "--method", "GEO-RL"], "not both"),
    (["--planner", "STATIC_Astar", "--method", "GEO-RL"], "apply to --policy"),
    (["--policy", "p.zip", "--method", "PPO"], "unknown method"),
    (["--policy", "missing.zip", "--method", "GEO-RL"], "not found"),
])
def test_cli_policy_errors(tmp_path, capsys, extra, message):
    world = [] if "--evaluation" in extra else ["--world", str(WORLD)]
    code = cli_main(["evaluate", *world, "--output-dir", str(tmp_path), "--episodes", "1", *extra])
    assert code == 2
    assert message in capsys.readouterr().err


def test_default_names_do_not_collide_and_nothing_is_written_on_errors(world, tmp_path):
    a = evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", seeds=[5003], max_steps=3, output_dir=tmp_path)
    b = evaluate_policy(world, TargetSeekingPolicy(damping=0.0), "GEO-RL", seeds=[5003], max_steps=3, output_dir=tmp_path)
    c = evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", seeds=[5003], max_steps=3, output_dir=tmp_path,
                        row_format="transfer")
    assert a.csv_path == b.csv_path  # the same class and settings: give a prefix to keep both
    assert a.csv_path != c.csv_path
    empty = tmp_path / "empty"
    with pytest.raises(ValueError, match="shape"):
        evaluate_policy(world, _WrongShape(), "GEO-RL", seeds=[5000], output_dir=empty)
    assert not empty.exists()


def test_dynamic_format_overrides_and_step_length(world):
    from ruscuav_benchmark.env import DynamicsConfig

    with pytest.raises(ValueError, match="dynamic risk"):
        evaluate_policy(world, TargetSeekingPolicy(), "OBS-RL", seeds=[5000], row_format="dynamic")
    with pytest.raises(ValueError, match="env_overrides cannot set"):
        evaluate_policy(world, TargetSeekingPolicy(), "OBS-RL", seeds=[5000], env_overrides={"stage": "C"})
    r = evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", seeds=[5003], max_steps=8,
                        env_overrides={"dynamics": DynamicsConfig(dt=0.25)})
    row = r.rows[0]
    assert row["episode_duration_s"] == 2.0 and row["urban_risk_per_s"] == row["urban_cumulative_risk"] / 2.0


def test_posthoc_csv_equals_the_notebooks(world, tmp_path):
    pytest.importorskip("pandas")
    sys.path.insert(0, str(REPO / "tests"))
    import notebook_reference as nb

    result = evaluate_policy(world, TargetSeekingPolicy(), "RISK+GREEN-RL", seeds=[5000, 5011], output_dir=tmp_path)
    zero = evaluate_policy(world, TargetSeekingPolicy(), "RISK+GREEN-RL", seeds=[5003], max_steps=0)
    derived = set(with_derived_metrics([{}])[0]) - {"episode_duration_s"}
    base = [k for k in result.rows[0] if k not in derived]
    raw = [{k: r[k] for k in base} for r in result.rows] + [{k: zero.rows[0][k] for k in base}]
    expected = nb.add_derived_metrics(nb.pd.DataFrame(raw)).to_csv(index=False, lineterminator="\n")
    from ruscuav_benchmark.tables import write_rows_csv

    ours = write_rows_csv(tmp_path / "ours.csv", with_derived_metrics(raw)).read_text(encoding="utf-8")
    assert ours == expected
    assert zero.rows[0]["steps"] == 0 and math.isnan(zero.rows[0]["urban_risk_per_step"])
