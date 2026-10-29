"""PPO training and checkpoint evaluation (needs the "rl" extra: Stable-Baselines3).

Training is only checked to run, save, reload and evaluate; with RUSCUAV_ARCHIVE an
archived checkpoint is also flown on its first archived missions.
"""

from pathlib import Path
import json
import math
import os
import sys

import pytest

pytest.importorskip("gymnasium")
sb3 = pytest.importorskip("stable_baselines3")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark import UrbanWorldLayers  # noqa: E402
from ruscuav_benchmark.cli import main as cli_main  # noqa: E402
from ruscuav_benchmark.policies import (  # noqa: E402
    ARCHIVED_EVALUATIONS,
    PPO_SETTINGS,
    TRAINING_SCHEDULES,
    evaluate_archived,
    evaluate_policy,
    load_policy,
    train_ppo,
)

WORLD = REPO / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz"
ARCHIVE = os.environ.get("RUSCUAV_ARCHIVE")
TINY = {"n_steps": 32, "batch_size": 32, "n_epochs": 1}


@pytest.fixture(scope="module")
def world():
    return UrbanWorldLayers.load_npz(WORLD)


@pytest.fixture(scope="module")
def trained(world, tmp_path_factory):
    out = tmp_path_factory.mktemp("ppo")
    return train_ppo(world, "RISK+GREEN-RL", out, seed=7, schedule=(("A", 64), ("B", 32)), n_envs=2,
                     vec_env="dummy", ppo_overrides=TINY)


def test_schedules_and_settings_are_the_papers():
    assert TRAINING_SCHEDULES["RISK+GREEN-RL"] == (("A", 50_000), ("B", 75_000), ("C", 75_000))
    assert TRAINING_SCHEDULES["GEO-RL"] == (("A", 200_000),)
    assert PPO_SETTINGS["policy_kwargs"] == {"net_arch": [256, 256]} and PPO_SETTINGS["batch_size"] == 256


def test_train_ppo_runs_saves_and_reloads(world, trained):
    assert sorted(trained) == ["0_A", "1_B", "final"] and trained["final"] == trained["1_B"]
    model = load_policy(trained["final"])
    assert model.num_timesteps >= 96 and model.n_steps == 32
    assert model.policy_kwargs["net_arch"] == [256, 256]
    result = evaluate_policy(world, trained["final"], "RISK+GREEN-RL", seeds=[5003], max_steps=10)
    assert result.rows[0]["steps"] == 10 and result.summary["policy"]["kind"] == "stable_baselines3_checkpoint"


def test_sampled_actions_repeat(world, trained):
    runs = [evaluate_policy(world, trained["final"], "RISK+GREEN-RL", seeds=[5003, 5011], max_steps=15,
                            deterministic=False).rows for _ in range(2)]
    deterministic = evaluate_policy(world, trained["final"], "RISK+GREEN-RL", seeds=[5003, 5011], max_steps=15).rows
    assert runs[0] == runs[1]
    assert runs[0] != deterministic


def test_training_with_subprocess_environments(world, tmp_path):
    paths = train_ppo(world, "GEO-RL", tmp_path, seed=3, schedule=(("A", 64),), n_envs=2, vec_env="subproc",
                      ppo_overrides=TINY)
    assert sorted(paths) == ["0_A", "final"] and load_policy(paths["final"]).num_timesteps >= 64


def test_checkpoint_needs_its_observation(world, trained):
    with pytest.raises(ValueError, match="shape"):
        evaluate_policy(world, trained["final"], "OBS-RL", seeds=[5003], max_steps=5)


def test_cli_flies_a_policy(world, trained, tmp_path, capsys):
    code = cli_main(["evaluate", "--world", str(WORLD), "--policy", str(trained["final"]), "--method", "RISK+GREEN-RL",
                     "--episodes", "1", "--seed0", "5003", "--output-dir", str(tmp_path), "--quiet"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["policy"] == str(trained["final"]) and out["n_episodes"] == 1
    csv_path = Path(out["csv_path"])
    assert csv_path.is_file() and csv_path.name.startswith("RISK+GREEN-RL_AOI_SYNTH_000_primary_no_terminal_")


def test_checkpoints_of_other_algorithms(world, tmp_path, capsys):
    from ruscuav_benchmark.cli import build_parser
    from ruscuav_benchmark.env import make_env
    from ruscuav_benchmark.policies import ALGORITHMS

    model = sb3.SAC("MlpPolicy", make_env(world, "RISK+GREEN-RL"), seed=0, learning_starts=4, batch_size=4,
                    buffer_size=64, policy_kwargs={"net_arch": [16, 16]})
    path = tmp_path / "sac.zip"
    model.learn(total_timesteps=8).save(path)
    result = evaluate_policy(world, path, "RISK+GREEN-RL", seeds=[5003], max_steps=5, algorithm="SAC")
    assert result.rows[0]["steps"] == 5 and result.summary["policy"]["algorithm"] == "SAC"
    assert type(load_policy(path, algorithm="SAC")).__name__ == "SAC"
    with pytest.raises(ValueError, match="could not be loaded with PPO"):
        load_policy(path)
    with pytest.raises(ValueError, match="unknown algorithm"):
        load_policy(path, algorithm="DQN")
    code = cli_main(["evaluate", "--world", str(WORLD), "--policy", str(path), "--algorithm", "SAC", "--method",
                     "RISK+GREEN-RL", "--episodes", "1", "--seed0", "5003", "--output-dir", str(tmp_path), "--quiet"])
    assert code == 0 and json.loads(capsys.readouterr().out)["n_episodes"] == 1
    evaluate = build_parser()._subparsers._group_actions[0].choices["evaluate"]
    assert next(a for a in evaluate._actions if a.dest == "algorithm").choices == ALGORITHMS


@pytest.mark.skipif(not ARCHIVE, reason="needs RUSCUAV_ARCHIVE (the extracted v2.0.0 archive)")
@pytest.mark.parametrize("name", ["static/RISK+GREEN-RL/4001", "dynamic/DYN-OBS-RL/1001"])
def test_archived_checkpoint_reproduces_its_first_episodes(name):
    pd = pytest.importorskip("pandas")
    entry = ARCHIVED_EVALUATIONS[name]
    result = evaluate_archived(name, ARCHIVE, n_episodes=2)
    archived = pd.read_csv(Path(ARCHIVE) / entry.episodes_csv, float_precision="round_trip").head(2)
    assert list(archived.columns) == list(result.rows[0])
    for i, row in enumerate(result.rows):
        for key, value in row.items():
            expected = archived[key].iloc[i]
            if key == "episode":
                continue
            if isinstance(value, (bool, str)) or value is None or key in ("seed", "steps"):
                assert (value is None and pd.isna(expected)) or value == expected, key
            else:
                # float32 policy outputs differ between PyTorch builds (docs/REPRODUCIBILITY.md)
                assert math.isclose(float(value), float(expected), rel_tol=1e-5, abs_tol=1e-9), key
