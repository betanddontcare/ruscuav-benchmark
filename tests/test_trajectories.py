"""Tests for trajectory scoring (ruscuav_benchmark.trajectories) and the scoring commands.

Episodes flown in the environment are scored again from their saved paths: every
column must come out as the environment recorded it, apart from the cargo health,
which the scorer estimates from the positions, and the shield-stuck ending, which
it cannot see; and none of their steps may count as unflyable (on these missions;
see docs/TRAJECTORIES.md for the shield's shortened steps). Synthetic
trajectories check the scoring rules and what the environment's vehicle could not
fly. scripts/migration/trajectory_parity.py does the first for the archived
evaluations.
"""

from pathlib import Path
import json
import math
import sys

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark import UrbanWorldLayers  # noqa: E402
from ruscuav_benchmark.cli import main  # noqa: E402
from ruscuav_benchmark.policies import TargetSeekingPolicy, evaluate_policy  # noqa: E402
from ruscuav_benchmark.protocols import load_protocol  # noqa: E402
from ruscuav_benchmark.trajectories import (Trajectory, evaluate_trajectories, read_trajectories,  # noqa: E402
                                            score_trajectories, score_trajectory, trajectory_env,
                                            write_trajectories_csv)

WORLD = REPO / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz"
NOT_COMPARED = {"episode", "method", "total_reward"}


@pytest.fixture(scope="module")
def world():
    return UrbanWorldLayers.load_npz(WORLD)


def _flown(world, tmp_path, protocol, method, n):
    result = evaluate_policy(world, TargetSeekingPolicy(), method, protocol=protocol, n_episodes=n,
                             output_dir=tmp_path / protocol, save_paths=True)
    return result, result.csv_path.parent / (result.csv_path.name[: -len("_episodes.csv")] + "_paths")


@pytest.mark.parametrize("protocol, method", [("canonical-primary", "GEO-RL"), ("canonical-secondary", "RISK+OBS-RL"),
                                              ("canonical-dynamic", "DYN-OBS-RL")])
def test_scored_paths_reproduce_the_flown_rows(world, tmp_path, protocol, method):
    result, paths = _flown(world, tmp_path, protocol, method, 4)
    scored = score_trajectories(protocol, paths, world, label="T", n_episodes=4)
    assert [r["seed"] for r in scored] == [r["seed"] for r in result.rows]
    for flown, ours in zip(result.rows, scored):
        assert ours["cargo_health_source"] == "estimated" and ours["positions_ignored"] == 0
        assert abs(ours["cargo_health"] - flown["cargo_health"]) < 1e-5
        assert ours["hazard_segment_steps"] == 0 and ours["within_workspace"] and ours["termination_reported"] is None
        if flown["termination"] == "shield_stuck":
            assert ours["termination"] == "ended" and not ours["is_success"]
        for key, value in flown.items():
            if key in NOT_COMPARED or key in ("cargo_health", "termination") or key not in ours:
                continue
            assert ours[key] == value or (isinstance(value, float) and math.isnan(value) and math.isnan(ours[key])), \
                (flown["seed"], key, value, ours[key])


def test_trajectory_files_round_trip(world, tmp_path):
    _, paths = _flown(world, tmp_path, "canonical-primary", "GEO-RL", 2)
    from_dir = read_trajectories(paths)
    assert sorted(from_dir) == [5000, 5001]
    t = from_dir[5000]
    t.cargo_health = np.linspace(1.0, 0.9, len(t.positions) - 1)
    t.termination = "landed"
    csv_path = write_trajectories_csv(tmp_path / "t.csv", from_dir.values())
    from_csv = read_trajectories(csv_path)
    assert np.array_equal(from_csv[5000].positions, t.positions.astype(np.float64))
    assert np.array_equal(from_csv[5000].cargo_health, t.cargo_health) and from_csv[5000].termination == "landed"
    assert from_csv[5001].cargo_health is None and from_csv[5001].termination is None
    np.savez(tmp_path / "t.npz", **{f"seed_{s}": v.positions for s, v in from_dir.items()})
    assert sorted(read_trajectories(tmp_path / "t.npz")) == [5000, 5001]
    rows = score_trajectories("canonical-primary", from_csv, world, n_episodes=2)
    assert rows[0]["cargo_health_source"] == "given" and rows[0]["cargo_health"] == 0.9
    assert rows[0]["termination_reported"] == "landed" and rows[1]["termination_reported"] is None
    bad = tmp_path / "bad.csv"
    bad.write_text("seed,step,x,y,z\n5000,0,1,2,3\n5000,2,1,2,3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not 0, 1, 2"):
        read_trajectories(bad)
    bad.write_text("seed,step,x,y,z\n5000,0,1,2,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2: z 'x' is not a number"):
        read_trajectories(bad)
    # a byte-order mark, blank lines and rows that leave out their empty trailing fields
    loose = tmp_path / "loose.csv"
    loose.write_bytes("\ufeffseed,step,x,y,z,cargo_health,termination\n5000,0,1,2,3\n   \n\n5000,1,1,2,4,0.5\n"
                      .encode("utf-8"))
    read = read_trajectories(loose)[5000]
    assert read.positions.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 4.0]] and read.cargo_health.tolist() == [0.5]
    assert read.termination is None


def _line(start, target, speed=10.0, dt=0.5, extra=5, z=None):
    start, target = np.asarray(start, float), np.asarray(target, float)
    d = target[:2] - start[:2]
    n = int(np.ceil(np.linalg.norm(d) / (speed * dt))) + extra
    out = [start.copy()]
    for k in range(1, n + 1):
        p = start.copy()
        p[:2] = start[:2] + d * min(1.0, k * speed * dt / np.linalg.norm(d))
        if z is not None:
            p[2] = z(k)
        out.append(p)
    return np.array(out)


def _ramp(start, target, acc=2.0, speed=10.0, dt=0.5):
    """Positions along the straight line, accelerating from rest at ``acc`` up to ``speed``."""
    start, target = np.asarray(start, float), np.asarray(target, float)
    d = target[:2] - start[:2]
    length = float(np.linalg.norm(d))
    t_acc = speed / acc
    out, k = [start.copy()], 0
    while True:
        k += 1
        t = k * dt
        s = 0.5 * acc * t * t if t <= t_acc else 0.5 * acc * t_acc * t_acc + speed * (t - t_acc)
        p = start.copy()
        p[:2] = start[:2] + d * min(1.0, s / length)
        out.append(p)
        if s >= length:
            return np.array(out)


def test_scoring_rules_on_synthetic_trajectories(world):
    protocol = load_protocol("canonical-primary")
    env = trajectory_env(world, protocol)
    mission = protocol.missions[0]
    options = mission.episode()[1]
    line = _line(mission.start, mission.target)
    row = score_trajectory(env, mission.seed, line, options=options)
    assert row["termination"] == "success" and row["is_success"] and row["positions_ignored"] > 0
    assert row["dist_xy"] < 30.0 and row["within_speed_limits"] and row["total_reward"] is None
    assert row["steps"] == len(line) - 1 - row["positions_ignored"]
    # full speed from rest on the first step: 20 m/s^2, beyond the vehicle's commands
    assert not row["within_acceleration_limits"] and abs(row["max_acc_xy_mps2"] - 20.0) < 1e-3
    gentle = _ramp(mission.start, mission.target)
    smooth = score_trajectory(env, mission.seed, gentle, options=options)
    assert smooth["is_success"] and smooth["within_acceleration_limits"] and smooth["within_speed_limits"]
    assert smooth["max_acc_xy_mps2"] <= 2.0 + 1e-3 and smooth["max_acc_z_mps2"] < 1e-3
    climb = gentle.copy()
    climb[3:, 2] += 1.5  # a 1.5 m step up: 6 m/s^2 vertically, more than the autopilot's 3 m/s^2
    climbing = score_trajectory(env, mission.seed, climb, options=options)
    assert climbing["within_speed_limits"] and not climbing["within_acceleration_limits"]
    assert abs(climbing["max_acc_z_mps2"] - 6.0) < 1e-3
    # into the ground on the third step
    ground = lambda k: (mission.start[2] if k < 3 else -50.0)  # noqa: E731
    row = score_trajectory(env, mission.seed, _line(mission.start, mission.target, z=ground), options=options)
    assert row["termination"] == "terrain_collision" and row["steps"] == 3 and not row["is_success"]
    # stopping short of the target: the method's own ending is reported, not taken
    row = score_trajectory(env, mission.seed, line[:4], options=options, termination="landed")
    assert row["termination"] == "ended" and row["termination_reported"] == "landed" and row["steps"] == 3
    assert not row["mission_acceptable"]
    for short in (line[:1], line[:5]):  # claiming success does not make one
        row = score_trajectory(env, mission.seed, short, options=options, termination="success")
        assert row["termination"] == "ended" and not row["is_success"] and not row["mission_acceptable"]
        assert row["termination_reported"] == "success" and row["dist_xy"] > 700.0
    still = score_trajectory(env, mission.seed, line[:1], options=options)  # no step: as the environment's rows
    assert still["termination"] == "ended" and still["steps"] == 0 and math.isnan(still["cargo_health"])
    assert math.isnan(still["urban_cumulative_risk"]) and still["road_time"] == 0.0
    fast = _line(mission.start, mission.target, speed=40.0)
    assert not score_trajectory(env, mission.seed, fast, options=options)["within_speed_limits"]
    outside = line[:6].copy()
    outside[5, 0] = -20.0  # beyond the workspace, which the environment's vehicle cannot leave
    assert score_trajectory(env, mission.seed, outside, options=options, cargo_health=[1.0] * 5)[
        "within_workspace"] is False
    with pytest.raises(ValueError, match="starts at"):
        score_trajectory(env, mission.seed, line + 5.0, options=options)
    with pytest.raises(ValueError, match="shape"):
        score_trajectory(env, mission.seed, line[:, :2], options=options)
    with pytest.raises(ValueError, match="cargo-health"):
        score_trajectory(env, mission.seed, line, options=options, cargo_health=[1.0])
    for value in (float("nan"), 1.5, -0.1):
        with pytest.raises(ValueError, match="from 0 to 1"):
            score_trajectory(env, mission.seed, line[:3], options=options, cargo_health=[1.0, value])
    env.close()
    items = {mission.seed: Trajectory(mission.seed, line)}
    with pytest.raises(ValueError, match="no trajectory"):
        score_trajectories(protocol, items, world, n_episodes=2)
    assert len(score_trajectories(protocol, items, world, n_episodes=2, require_all=False)) == 1
    with pytest.raises(ValueError, match="not missions"):
        score_trajectories(protocol, {1: Trajectory(1, line)}, world)
    for n in (0, -1, 201):
        with pytest.raises(ValueError, match="200 missions"):
            score_trajectories(protocol, items, world, n_episodes=n)


def _across_a_building(world, hazards):
    """Two safe positions 5 m above the ground on either side of a building (the
    segment between them enters it)."""
    res = float(world.resolution_m)
    ys, xs = np.nonzero(world.building_mask > 0.5)
    for k in np.random.default_rng(0).permutation(len(xs))[:300]:
        x, y = (xs[k] + 0.5) * res, (ys[k] + 0.5) * res
        z = float(world.terrain_z[ys[k], xs[k]]) + 5.0
        if not hazards.is_unsafe(np.array([x, y, z], np.float32)):
            continue
        ends = []
        for sign in (-1.0, 1.0):
            for d in np.arange(1.0, 80.0, 1.0):
                p = np.array([x + sign * d, y, z], np.float32)
                if not hazards.is_unsafe(p) and not hazards.is_unsafe(p + np.float32([sign * 2.0, 0.0, 0.0])):
                    ends.append(p + np.float32([sign * 2.0, 0.0, 0.0]))
                    break
        if len(ends) == 2:
            return ends[0], ends[1]
    raise AssertionError("no building to cross")


def test_unflyable_steps_are_flagged(world):
    protocol = load_protocol("canonical-primary")
    env = trajectory_env(world, protocol)
    mission = protocol.missions[0]
    options = mission.episode()[1]
    a, b = _across_a_building(world, env.hazards)
    start = np.asarray(mission.start, np.float32)
    # given cargo health: the jumps would break the estimated one
    to_a = score_trajectory(env, mission.seed, np.array([start, a]), options=options, cargo_health=[1.0])
    through = score_trajectory(env, mission.seed, np.array([start, a, b]), options=options, cargo_health=[1.0, 1.0])
    assert through["steps"] == 2 and through["hazard_segment_steps"] == to_a["hazard_segment_steps"] + 1
    assert through["building_collision_steps"] == to_a["building_collision_steps"]  # the positions are outside
    assert not through["within_speed_limits"] and not through["within_acceleration_limits"]
    env.close()


def test_flown_accelerations_exceed_the_limits_only_where_the_shield_or_the_bounds_act(world, tmp_path, monkeypatch):
    """The environment's own flights keep to the acceleration limits of its commands, except on the steps that
    its geometric shield, or the bounds of its workspace, stop or shorten at once."""
    from ruscuav_benchmark.env import RuscuavEnv
    from ruscuav_benchmark.trajectories import acceleration_limits

    acted = {}
    reset, step = RuscuavEnv.reset, RuscuavEnv.step

    def recording_reset(self, *args, **kwargs):
        out = reset(self, *args, **kwargs)
        self._seed_for_test = kwargs.get("seed")
        acted[self._seed_for_test] = [False]
        return out

    def recording_step(self, action):
        out = step(self, action)
        at_bound = np.isclose(self.current_pos, self.bounds_low, atol=1e-3) | \
            np.isclose(self.current_pos, self.bounds_high, atol=1e-3)
        acted[self._seed_for_test].append(bool(out[4]["urban_geom_shielded"]) or bool(at_bound.any()))
        return out

    monkeypatch.setattr(RuscuavEnv, "reset", recording_reset)
    monkeypatch.setattr(RuscuavEnv, "step", recording_step)
    _, paths = _flown(world, tmp_path, "canonical-primary", "GEO-RL", 10)
    monkeypatch.undo()
    scored = score_trajectories("canonical-primary", paths, world, n_episodes=10)
    env = trajectory_env(world, load_protocol("canonical-primary"))
    limits = acceleration_limits(env.config.dynamics)
    env.close()
    flown = read_trajectories(paths)
    for row in scored:
        p = flown[row["seed"]].positions.astype(np.float32).astype(np.float64)[: row["steps"] + 1]
        v = np.vstack([np.zeros(3), np.diff(p, axis=0) / 0.5])
        a = np.diff(v, axis=0) / 0.5  # a[k - 1]: the acceleration of step k
        over = np.flatnonzero((np.hypot(a[:, 0], a[:, 1]) > limits["xy"] + 1e-2)
                              | (np.abs(a[:, 2]) > limits["z"] + 1e-2)) + 1
        steps = acted[row["seed"]]
        assert all(steps[k] or steps[k - 1] for k in over), (row["seed"], over.tolist())
        assert row["within_acceleration_limits"] == (len(over) == 0), row["seed"]
    assert any(not row["within_acceleration_limits"] for row in scored)  # the shield's steps exceed them


def test_zero_step_rows_in_a_dynamic_protocol(world):
    protocol = load_protocol("canonical-dynamic")
    env = trajectory_env(world, protocol)
    mission = protocol.missions[0]
    row = score_trajectory(env, mission.seed, np.array([mission.start]), options=mission.episode()[1])
    assert row["steps"] == 0 and math.isnan(row["dynamic_cumulative_risk"]) and math.isnan(row["dynamic_risk_mean"])
    assert row["dynamic_budget_ok"] is False and not row["mission_acceptable"] and not row["mission_acceptable_logged"]
    line = _line(mission.start, mission.target)
    flown = score_trajectory(env, mission.seed, line, options=mission.episode()[1])
    assert flown["is_success"] and flown["dynamic_cumulative_risk"] >= 0.0
    env.close()


def test_protocol_budgets_decide_acceptability_everywhere(world, tmp_path):
    from ruscuav_benchmark.planners import evaluate_planner
    from ruscuav_benchmark.posthoc import DEFAULT_BUDGETS, static_acceptability
    from ruscuav_benchmark.protocols import make_protocol

    strict = make_protocol(world, "strict", n_episodes=2, budgets={"road_s": 0.0, "ped_s": 0.0, "building_s": 0.0})
    flown = evaluate_policy(world, TargetSeekingPolicy(), "RISK+OBS-RL", protocol=strict, output_dir=tmp_path / "s",
                            save_paths=True)
    paths = flown.csv_path.parent / (flown.csv_path.name[: -len("_episodes.csv")] + "_paths")
    scored = evaluate_trajectories(paths, strict, world)
    for a, b in zip(flown.rows, scored.rows):
        assert a["mission_acceptable"] == b["mission_acceptable"]
        assert a["mission_acceptable_logged"] == b["mission_acceptable_logged"]
        exposed = a["road_time"] > 0 or a["sidewalk_time"] > 0 or a["building_time"] > 0
        assert not (exposed and a["mission_acceptable"])
    assert flown.summary["mission_acceptability_rate"] == scored.summary["mission_acceptability_rate"]
    # the planners' rows carry the environment's verdict, with the protocol's budgets
    planner = evaluate_planner(world, "STATIC_Astar", protocol=strict, n_episodes=2)
    for row in planner.rows:
        assert row["mission_acceptable"] == static_acceptability(row, strict.budgets)
    assert strict.budgets["cargo_min"] == DEFAULT_BUDGETS["cargo_min"]


def test_evaluate_and_score_from_the_command_line(world, tmp_path, capsys):
    _, paths = _flown(world, tmp_path, "canonical-primary", "GEO-RL", 3)
    result = evaluate_trajectories(paths, "canonical-primary", world, label="MINE", n_episodes=3,
                                   output_dir=tmp_path / "eval")
    assert result.csv_path.name == "MINE_canonical-primary_n3_episodes.csv" and result.summary["n_episodes"] == 3
    assert "urban_risk_per_step" in result.rows[0] and not result.summary["protocol"]["complete"]
    assert main(["evaluate", "--protocol", "canonical-primary", "--trajectories", str(paths), "--label", "CLI",
                 "--episodes", "3", "--output-dir", str(tmp_path / "cli"), "--quiet"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["label"] == "CLI" and printed["protocol"] == "canonical-primary"
    assert main(["evaluate", "--trajectories", str(paths), "--output-dir", str(tmp_path / "x")]) == 2
    assert "needs --protocol" in capsys.readouterr().err
    other = evaluate_policy(world, TargetSeekingPolicy(gain=0.5), "GEO-RL", protocol="canonical-primary", n_episodes=3,
                            output_dir=tmp_path / "other")
    code = main(["score", "--protocol", "canonical-primary", f"MINE={result.csv_path}", f"OTHER={other.csv_path}",
                 "--compare", "MINE:OTHER:success", "--output-dir", str(tmp_path / "score")])
    assert code == 0
    out = capsys.readouterr()
    assert "MINE_vs_OTHER_success" in out.out and "do not cover all missions" in out.err
    report = json.loads((tmp_path / "score" / "score.json").read_text(encoding="utf-8"))
    assert [r["label"] for r in report["runs"]] == ["MINE", "OTHER"] and len(report["paired_tests"]) == 1
    assert (tmp_path / "score" / "run_summaries.csv").is_file()
    assert main(["score", "--protocol", "canonical-dynamic", f"MINE={result.csv_path}",
                 "--output-dir", str(tmp_path / "s2")]) == 2
    assert "no dynamic risk" in capsys.readouterr().err
    assert main(["protocol", "list"]) == 0 and "canonical-primary" in capsys.readouterr().out
    assert main(["protocol", "show", "vienna-primary"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["n_missions"] == 200 and shown["world"]["distributed"] is False
    assert main(["protocol", "create", "--world", str(WORLD), "--name", "tiny", "--episodes", "2",
                 "--output", str(tmp_path / "tiny.json")]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["missions"] == 2 and load_protocol(tmp_path / "tiny.json").seeds == [5000, 5001]
    # the world file is recorded relative to the protocol file, and found from it
    from ruscuav_benchmark.protocols import find_world

    assert not Path(created["world_file"]).is_absolute()
    assert find_world(load_protocol(tmp_path / "tiny.json")).resolve() == WORLD.resolve()


def test_score_refuses_runs_of_another_protocol(world, tmp_path, capsys):
    import hashlib

    from ruscuav_benchmark.protocols import make_protocol

    result, _ = _flown(world, tmp_path, "canonical-primary", "GEO-RL", 2)
    out = str(tmp_path / "score")
    # the rows' regime
    assert main(["score", "--protocol", "canonical-secondary", f"MINE={result.csv_path}", "--output-dir", out]) == 2
    assert "regime primary_no_terminal" in capsys.readouterr().err
    # the protocol recorded next to the CSV: same missions, regime and world, another protocol
    tiny = make_protocol(world, "tiny", n_episodes=2).save(tmp_path / "tiny.json")
    assert main(["score", "--protocol", str(tiny), f"MINE={result.csv_path}", "--output-dir", out]) == 2
    assert "flown under the protocol 'canonical-primary'" in capsys.readouterr().err
    # dynamic risk where the protocol has none
    dyn = evaluate_policy(world, TargetSeekingPolicy(), "DYN-OBS-RL", protocol="canonical-dynamic", n_episodes=1,
                          output_dir=tmp_path / "dyn")
    assert main(["score", "--protocol", "canonical-primary", f"D={dyn.csv_path}", "--output-dir", out]) == 2
    assert "has dynamic risk" in capsys.readouterr().err
    assert main(["score", "--protocol", "canonical-dynamic", f"D={dyn.csv_path}", "--output-dir", out]) == 0
    capsys.readouterr()
    # runs of a registry are read, and hashed, from the registry's folder
    registry = tmp_path / "reg" / "registry.csv"
    (registry.parent / "runs").mkdir(parents=True)
    copy = registry.parent / "runs" / "geo_episodes.csv"
    copy.write_bytes(result.csv_path.read_bytes())
    registry.write_text("label,csv_path\nGEO,runs/geo_episodes.csv\n", encoding="utf-8")
    assert main(["score", "--registry", str(registry), "--output-dir", out]) == 0
    capsys.readouterr()
    report = json.loads((tmp_path / "score" / "score.json").read_text(encoding="utf-8"))
    assert report["runs"][0]["csv_path"] == "runs/geo_episodes.csv"
    assert report["runs"][0]["sha256"] == hashlib.sha256(copy.read_bytes()).hexdigest()


def test_evaluate_notes_trajectories_the_vehicle_could_not_fly(world, tmp_path, capsys):
    protocol = load_protocol("canonical-primary")
    mission = protocol.missions[0]
    fast = _line(mission.start, mission.target, speed=40.0)
    csv_path = write_trajectories_csv(tmp_path / "fast.csv", [Trajectory(mission.seed, fast)])
    assert main(["evaluate", "--protocol", "canonical-primary", "--trajectories", str(csv_path), "--episodes", "1",
                 "--output-dir", str(tmp_path / "fast"), "--quiet"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["episodes_outside_speed_limits"] == 1
    assert "could not have been flown" in out.err and "episodes_outside_speed_limits: 1" in out.err
    # from rest to 40 m/s in one step also exceeds the acceleration limits, which are noted apart
    assert json.loads(out.out)["episodes_outside_acceleration_limits"] == 1
    assert "1 trajectory accelerates faster" in out.err
    gentle = write_trajectories_csv(tmp_path / "gentle.csv", [Trajectory(mission.seed, _ramp(mission.start,
                                                                                               mission.target))])
    assert main(["evaluate", "--protocol", "canonical-primary", "--trajectories", str(gentle), "--episodes", "1",
                 "--output-dir", str(tmp_path / "gentle"), "--quiet"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["episodes_outside_acceleration_limits"] == 0 and "accelerate faster" not in out.err
