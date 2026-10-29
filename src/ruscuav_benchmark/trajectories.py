"""Scoring flown trajectories with the benchmark's accounting.

A method that does not run in ``ruscuav_benchmark.env`` (another simulator, a
planner with its own controller, a recorded flight) is scored from its trajectories:
for each mission of a protocol, the position at the start and after every step of
``dt`` (0.5 s), x and y in metres in the world's grid frame and z the absolute
altitude in the height frame of the world's terrain (docs/TRAJECTORIES.md).

:func:`score_trajectory` accounts the positions exactly as the environment does at
the end of each step: the exposure times over each layer, the semantic and the
dynamic risk, the hard violations and, in the secondary regime, operational
stopping. It ends the episode where the environment would: on reaching the target
region (success), on terrain contact, after 300 steps without progress (stuck or
near success), when the cargo health reaches zero, or at the step limit; positions
after the end are ignored. A trajectory that stops before any of these ends with
``"ended"``; the method's own ending, if it gives one, is kept as
``termination_reported`` and decides nothing. The row has the columns of the
primary evaluations (the ``posthoc`` format), so it is scored like any other run.

Cargo health is taken from the trajectory when it has it, otherwise estimated with
the environment's cargo model from the accelerations of the positions (finite
differences, starting at rest). The rows also report what the environment's vehicle
could not have flown: speeds above its limits, positions outside its workspace and
steps whose motion enters a hazard between two positions, which the environment's
geometric shield would have stopped; and accelerations above the limits of its
commands, which the environment's own flights reach only where that shield, or the
bounds of the workspace, stop or shorten a step. The action shield of the
environment has no counterpart here, so ``shield_stuck`` cannot occur.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from .posthoc import DEFAULT_BUDGETS, dynamic_acceptability, static_acceptability
from .tables import Row

__all__ = ["Trajectory", "read_trajectories", "write_trajectories_csv", "score_trajectory", "score_trajectories",
           "trajectory_env", "acceleration_limits", "TrajectoryEvaluation", "evaluate_trajectories"]

_SEED_IN_NAME = re.compile(r"seed_?(-?\d+)")
#: Tolerance of the workspace bounds (m), for positions rounded to float32.
_WORKSPACE_TOLERANCE_M = 1e-3
#: Tolerance of the acceleration limits (m/s^2): finite differences of float32 positions.
_ACCELERATION_TOLERANCE = 1e-2


def acceleration_limits(dynamics: Any) -> Dict[str, float]:
    """The largest accelerations (m/s^2) the environment's vehicle can fly with the
    dynamics configuration ``dynamics``: horizontally the norm of its command box
    (each axis up to ``a_max_xy``), vertically ``a_max_z``. The low-pass filter and
    the jerk limit keep the commands within these bounds, and the speed limits only
    shorten an acceleration. A separate jerk check would add little at the
    environment's step: two accelerations within these bounds differ by at most
    ``2 * sqrt(2) * a_max_xy`` per step, 34 m/s^3 with the defaults, close to the
    30 m/s^3 jerk limit of the commands."""
    return {"xy": float(np.sqrt(2.0) * dynamics.a_max_xy), "z": float(dynamics.a_max_z)}


@dataclass
class Trajectory:
    """Positions (n + 1, 3) of one episode: the start, then one per step; optionally
    the cargo health after each step (n values) and the method's own ending, which
    is reported (``termination_reported``) but does not decide the row's ending."""

    seed: int
    positions: np.ndarray
    cargo_health: Optional[np.ndarray] = None
    termination: Optional[str] = None


def _cell(row: Dict[str, Optional[str]], key: str) -> str:
    """A CSV cell, empty when the row leaves it out."""
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _number(path: Path, line: int, key: str, text: str, kind=float):
    try:
        return kind(text)
    except ValueError:
        raise ValueError(f"{path}, line {line}: {key} {text!r} is not {'an integer' if kind is int else 'a number'}"
                         ) from None


def read_trajectories(path: Union[str, Path]) -> Dict[int, Trajectory]:
    """Trajectories by seed from

    - a folder of ``.npy`` files with ``seed_<seed>`` in their name, each an
      (n + 1, 3) array (as ``save_paths`` of the evaluations writes them; files with
      ``planned`` in their name are skipped);
    - an ``.npz`` file with one (n + 1, 3) array per key ``seed_<seed>``;
    - a CSV (UTF-8, with or without a byte-order mark) with the columns ``seed``,
      ``step`` (0 for the start), ``x``, ``y``, ``z`` and optionally ``cargo_health``
      (empty at step 0) and ``termination``; blank lines are skipped.
    """
    path = Path(path)
    out: Dict[int, Trajectory] = {}

    def add(seed: int, trajectory: Trajectory) -> None:
        if seed in out:
            raise ValueError(f"{path}: two trajectories of seed {seed}")
        out[seed] = trajectory

    if path.is_dir():
        for file in sorted(path.glob("*.npy")):
            if "planned" in file.name:
                continue
            match = _SEED_IN_NAME.search(file.name)
            if match is None:
                raise ValueError(f"{file}: no seed_<seed> in the file name")
            add(int(match.group(1)), Trajectory(int(match.group(1)), np.load(file)))
    elif path.suffix == ".npz":
        with np.load(path) as data:
            for key in data.files:
                match = _SEED_IN_NAME.fullmatch(key)
                if match is None:
                    raise ValueError(f"{path}: key {key!r} is not seed_<seed>")
                add(int(match.group(1)), Trajectory(int(match.group(1)), np.asarray(data[key])))
    else:
        rows: Dict[int, List[Dict[str, Any]]] = {}
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields = [name.strip() for name in reader.fieldnames or []]
            reader.fieldnames = fields
            missing = [c for c in ("seed", "step", "x", "y", "z") if c not in fields]
            if missing:
                raise ValueError(f"{path}: columns {missing} missing")
            for row in reader:
                if not any(_cell(row, key) for key in fields):
                    continue  # a blank line
                line = reader.line_num
                parsed = {"line": line, "step": _number(path, line, "step", _cell(row, "step"), int),
                          "xyz": [_number(path, line, key, _cell(row, key)) for key in ("x", "y", "z")],
                          "cargo_health": _cell(row, "cargo_health"), "termination": _cell(row, "termination")}
                rows.setdefault(_number(path, line, "seed", _cell(row, "seed"), int), []).append(parsed)
        for seed, items in rows.items():
            items.sort(key=lambda r: r["step"])
            steps = [r["step"] for r in items]
            if steps != list(range(len(steps))):
                raise ValueError(f"{path}: the steps of seed {seed} are not 0, 1, 2, ...")
            positions = np.array([r["xyz"] for r in items], dtype=np.float64)
            cargo = None
            values = [r["cargo_health"] for r in items[1:]]
            if any(values):
                if not all(values):
                    raise ValueError(f"{path}: seed {seed} has cargo health for some steps only")
                cargo = np.array([_number(path, r["line"], "cargo_health", r["cargo_health"]) for r in items[1:]])
            ends = {r["termination"] for r in items} - {""}
            if len(ends) > 1:
                raise ValueError(f"{path}: seed {seed} has several terminations {sorted(ends)}")
            add(seed, Trajectory(seed, positions, cargo, ends.pop() if ends else None))
    return out


def write_trajectories_csv(path: Union[str, Path], trajectories: Iterable[Trajectory]) -> Path:
    """Write trajectories in the CSV format :func:`read_trajectories` reads."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["seed", "step", "x", "y", "z", "cargo_health", "termination"])
        for t in trajectories:
            for k, (x, y, z) in enumerate(np.asarray(t.positions, dtype=np.float64)):
                cargo = "" if t.cargo_health is None or k == 0 else repr(float(t.cargo_health[k - 1]))
                writer.writerow([t.seed, k, repr(float(x)), repr(float(y)), repr(float(z)), cargo, t.termination or ""])
    return path


def trajectory_env(world: Any, protocol: Any) -> Any:
    """The environment whose accounting scores trajectories of ``protocol``: the
    configuration of the planners (``RISK+OBS-RL``) under the protocol, with its
    regime, risk profile, step length, step limit, budgets and dynamic risk
    (:meth:`~ruscuav_benchmark.protocols.Protocol.environment_config`). Needs the
    "env" extra."""
    from .env import RuscuavEnv

    return RuscuavEnv(world, protocol.environment_config())


def score_trajectory(env: Any, seed: int, positions: Any, *, options: Optional[Dict[str, Any]] = None,
                     regime: Any = None, label: str = "TRAJECTORY", cargo_health: Optional[Sequence[float]] = None,
                     termination: Optional[str] = None, budgets: Optional[Dict[str, float]] = None,
                     start_tolerance_m: float = 0.05) -> Row:
    """The episode row of one trajectory, accounted in ``env`` (:func:`trajectory_env`)
    on the mission of ``seed`` (``options``: the protocol's mission). ``positions`` is
    (n + 1, 3), starting at the mission's start (within ``start_tolerance_m``);
    positions are taken as float32, as the environment holds them. ``cargo_health``
    (n values in [0, 1]) replaces the estimate; ``termination``, the method's own
    ending, is reported as ``termination_reported`` only. The row's
    ``mission_acceptable`` is the post-hoc verdict with ``budgets`` (the paper's by
    default; with dynamic risk including the dynamic budget),
    ``mission_acceptable_logged`` the environment's own."""
    from .env.reward import ProgressCounters
    from .policies.rollout import posthoc_row

    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) < 1:
        raise ValueError(f"seed {seed}: positions must have shape (n + 1, 3), got {pos.shape}")
    if not np.isfinite(pos).all():
        raise ValueError(f"seed {seed}: positions must be finite")
    given_cargo = None if cargo_health is None else np.asarray(cargo_health, dtype=np.float64).reshape(-1)
    if given_cargo is not None:
        if len(given_cargo) != len(pos) - 1:
            raise ValueError(f"seed {seed}: {len(given_cargo)} cargo-health values for {len(pos) - 1} steps")
        if not (np.isfinite(given_cargo).all() and (given_cargo >= 0.0).all() and (given_cargo <= 1.0).all()):
            raise ValueError(f"seed {seed}: cargo health must be a number from 0 to 1 after every step")
    _, reset_info = env.reset(seed=int(seed), options=options)
    start = env.start_point
    if float(np.max(np.abs(pos[0] - start))) > float(start_tolerance_m):
        raise ValueError(f"seed {seed}: the trajectory starts at {pos[0].tolist()}, the mission at {start.tolist()}")
    cfg = env.config
    ep, mission, d = cfg.episode, cfg.mission, cfg.dynamics
    dt = float(d.dt)
    e = env.exposure
    hazards, shield = env.hazards, env.shield
    counters = ProgressCounters(best_dist_xy=env.dist_xy(start))
    health = 1.0
    prev = start.astype(np.float64)
    prev32 = np.asarray(start, dtype=np.float32)
    vel_prev = np.zeros(3)
    max_vxy = max_vz = max_axy = max_az = 0.0
    hazard_segments = 0
    end: Optional[str] = None
    info: Dict[str, Any] = {}
    steps = 0
    for k in range(1, len(pos)):
        p = pos[k]
        vel = (p.astype(np.float64) - prev) / dt
        acc = (vel - vel_prev) / dt
        max_vxy = max(max_vxy, float(np.hypot(vel[0], vel[1])))
        max_vz = max(max_vz, abs(float(vel[2])))
        max_axy = max(max_axy, float(np.hypot(acc[0], acc[1])))
        max_az = max(max_az, abs(float(acc[2])))
        # the motion of the step as the environment's geometric shield checks it
        if not hazards.is_unsafe(prev32, include_clearance_violation=True) and \
                shield.first_hit_on_segment(prev32, p) is not None:
            hazard_segments += 1
        if given_cargo is None:
            health = env.cargo.step(health, acc)[0]
        else:
            health = float(given_cargo[k - 1])
        dist = env.dist_xy(p)
        terminated = truncated = False
        # the environment's navigation endings, in its order
        if p[2] <= env.dynamics.ground_at(p[0], p[1]) + ep.terrain_margin_m:
            terminated, end = True, "terrain_collision"
        if dist < (counters.best_dist_xy - ep.improve_eps_m):
            counters.best_dist_xy = dist
            counters.steps_since_improve = 0
        else:
            counters.steps_since_improve += 1
        if not terminated and dist < ep.success_radius_m:
            terminated, end = True, "success"
        if not terminated and counters.steps_since_improve >= ep.stuck_patience_steps:
            terminated = True
            end = "near_success" if ep.success_radius_m < dist <= ep.near_success_radius_m else "stuck"
        if not terminated and health <= 0.0:
            terminated, end = True, "cargo"
        if not terminated and k >= ep.max_steps:
            truncated, end = True, "timeout"
        urban = e.update(env.risk_model, p, dt)
        if env.terminate_on_urban_constraint and not terminated and not truncated:
            stop = e.operational_stop(urban["constraints"], mission)
            if stop is not None:
                terminated, end = True, stop
        if env.dynamic_config is not None:
            e.update_dynamic(0.0 if env.dynamic_field is None else float(env.dynamic_field.risk_at_pos(p, k)), dt)
        steps = k
        prev, prev32, vel_prev = p.astype(np.float64), p, vel
        if terminated or truncated:
            break
    if end is None:
        end = "ended"
    success = end == "success"
    acceptable = e.mission_acceptable(end, health, mission)
    info.update({
        "termination": end, "is_success": success, "dist_xy": env.dist_xy(pos[steps]), "cargo_health": float(health),
        "mission_acceptable": acceptable, "urban_mean_risk": e.risk_sum / max(1e-6, e.steps * dt),
        "urban_max_risk": e.risk_max, "urban_cumulative_risk": e.risk_sum, "urban_road_time": e.road_time,
        "urban_sidewalk_time": e.sidewalk_time, "urban_building_time": e.building_time,
        "urban_green_time": e.green_time, "urban_canopy_time": e.canopy_time, "urban_sensitive_time": e.sensitive_time,
        "urban_building_collision_steps": e.building_collision_steps,
        "urban_canopy_collision_steps": e.canopy_collision_steps,
        "urban_sensitive_low_altitude_violation_steps": e.sensitive_low_altitude_violation_steps,
        "urban_hard_violation_steps": e.hard_violation_steps,
    })
    if steps == 0:
        # no step: the environment's rows have no cargo health and no risk either
        for key in ("cargo_health", "urban_mean_risk", "urban_max_risk", "urban_cumulative_risk"):
            info.pop(key)
    if regime is None:
        from .env.config import REGIMES

        regime = next(r for r in REGIMES.values() if r.stage == cfg.stage and r.mode == cfg.mode)
    flown = pos[: steps + 1]
    row = posthoc_row(env, seed, label, regime, reset_info, [], info, flown, steps)
    row["total_reward"] = None
    budgets = DEFAULT_BUDGETS if budgets is None else budgets
    row["mission_acceptable"] = static_acceptability(row, budgets)
    if env.dynamic_config is not None:
        # as in the dynamic evaluations: acceptability with the dynamic budget, and without it
        dc = env.dynamic_config
        budget_ok = steps > 0 and e.dynamic_risk_sum <= float(dc.dynamic_cumulative_risk_budget)
        row["mission_acceptable_logged"] = bool(acceptable and budget_ok)
        nan = float("nan")
        row.update({
            "dynamic_risk_mean": e.dynamic_risk_sum / max(1e-6, e.dynamic_steps * dt) if steps else nan,
            "dynamic_risk_max": e.dynamic_risk_max if steps else nan,
            "dynamic_cumulative_risk": e.dynamic_risk_sum if steps else nan,
            "dynamic_budget_ok": bool(budget_ok),
            "mission_acceptable_static_only": row["mission_acceptable"],
        })
        row["mission_acceptable"] = dynamic_acceptability(row, budgets)
    low = np.asarray(env.bounds_low, dtype=np.float64) - _WORKSPACE_TOLERANCE_M
    high = np.asarray(env.bounds_high, dtype=np.float64) + _WORKSPACE_TOLERANCE_M
    limits = acceleration_limits(d)
    row.update({
        "termination_reported": termination or None,
        "cargo_health_source": "estimated" if given_cargo is None else "given",
        "positions_ignored": int(len(pos) - 1 - steps),
        "max_speed_xy_mps": max_vxy, "max_speed_z_mps": max_vz, "max_acc_xy_mps2": max_axy,
        "max_acc_z_mps2": max_az,
        "within_speed_limits": bool(max_vxy <= d.v_max_xy + 1e-3 and max_vz <= d.v_max_z + 1e-3),
        "within_acceleration_limits": bool(max_axy <= limits["xy"] + _ACCELERATION_TOLERANCE
                                           and max_az <= limits["z"] + _ACCELERATION_TOLERANCE),
        "within_workspace": bool(((flown >= low) & (flown <= high)).all()),
        "hazard_segment_steps": int(hazard_segments),
    })
    return row


def score_trajectories(protocol: Any, trajectories: Union[str, Path, Dict[int, Trajectory]], world: Any = None, *,
                       label: str = "TRAJECTORY", n_episodes: Optional[int] = None,
                       require_all: bool = True) -> List[Row]:
    """Rows of the trajectories of a protocol's missions (in the protocol's order).
    With ``require_all`` every mission (of the first ``n_episodes``) needs a
    trajectory; trajectories of other seeds are refused."""
    from .protocols import load_protocol, protocol_world

    protocol = load_protocol(protocol)
    missions = protocol.first_missions(n_episodes)
    layers = protocol_world(protocol, world)
    items = trajectories if isinstance(trajectories, dict) else read_trajectories(trajectories)
    unknown = sorted(set(items) - set(protocol.seeds))
    if unknown:
        raise ValueError(f"trajectories of seeds that are not missions of {protocol.name!r}: {unknown[:5]}")
    absent = [m.seed for m in missions if m.seed not in items]
    if absent and require_all:
        raise ValueError(f"no trajectory for {len(absent)} missions of {protocol.name!r}, e.g. seed {absent[0]}")
    env = trajectory_env(layers, protocol)
    rows = []
    try:
        for index, mission in enumerate(m for m in missions if m.seed in items):
            t = items[mission.seed]
            row = score_trajectory(env, mission.seed, t.positions, options=mission.episode()[1], label=label,
                                   cargo_health=t.cargo_health, termination=t.termination,
                                   budgets=protocol.budgets)
            row["episode"] = index
            rows.append(row)
    finally:
        env.close()
    return rows


@dataclass
class TrajectoryEvaluation:
    """Rows and summary of :func:`evaluate_trajectories`."""

    label: str
    rows: List[Row]
    summary: Dict[str, Any]
    csv_path: Optional[Path] = None
    summary_path: Optional[Path] = None


def evaluate_trajectories(trajectories: Union[str, Path, Dict[int, Trajectory]], protocol: Any, world: Any = None, *,
                          label: str = "TRAJECTORY", n_episodes: Optional[int] = None,
                          output_dir: Union[str, Path, None] = None, prefix: Optional[str] = None
                          ) -> TrajectoryEvaluation:
    """Score the trajectories of a method on the missions of ``protocol`` (all, or the
    first ``n_episodes``) and summarize them as the primary evaluations are: rates,
    post-hoc acceptability with the protocol's budgets (dynamic acceptability in a
    dynamic protocol) and conditional means. The summary also counts the episodes
    the environment's vehicle could not have flown (outside its speed limits or its
    workspace, or with steps into a hazard), those that accelerate faster than its
    commands can, and those with estimated cargo health.
    With ``output_dir``, ``<prefix>_episodes.csv`` (with the derived columns),
    ``<prefix>_summary.json`` and ``<prefix>_config.json`` are written there; the
    default prefix is ``<label>_<protocol>``."""
    import json
    from datetime import datetime, timezone

    from .posthoc import summarize_posthoc, with_derived_metrics
    from .protocols import load_protocol, protocol_world
    from .tables import jsonable, write_rows_csv

    chosen = load_protocol(protocol)
    layers = protocol_world(chosen, world)
    rows = score_trajectories(chosen, trajectories, layers, label=label, n_episodes=n_episodes)
    n_run = len(rows)
    record = {"name": chosen.name, "digest": chosen.digest(), "missions": chosen.n_episodes, "missions_flown": n_run,
              "complete": n_run == chosen.n_episodes, "acceptability": chosen.acceptability,
              "budgets": dict(chosen.budgets)}
    settings = {"label": label, "world_id": chosen.world_id, "regime": chosen.regime, "protocol": record,
                "trajectories": trajectories if isinstance(trajectories, (str, Path)) else "<in memory>",
                "scoring": "ruscuav_benchmark.trajectories.score_trajectory"}
    summary = summarize_posthoc(rows, dynamic=chosen.dynamic, budgets=chosen.budgets, dt_s=chosen.dt_s)
    summary.update({**jsonable(settings), "seeds": [row["seed"] for row in rows],
                    "cargo_health_estimated": sum(row["cargo_health_source"] == "estimated" for row in rows),
                    "episodes_outside_speed_limits": sum(not row["within_speed_limits"] for row in rows),
                    "episodes_outside_acceleration_limits": sum(not row["within_acceleration_limits"]
                                                                for row in rows),
                    "episodes_outside_workspace": sum(not row["within_workspace"] for row in rows),
                    "episodes_with_hazard_segments": sum(row["hazard_segment_steps"] > 0 for row in rows),
                    "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    out = TrajectoryEvaluation(label, with_derived_metrics(rows, dt_s=chosen.dt_s, budgets=chosen.budgets), summary)
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        if prefix is None:
            prefix = f"{label}_{chosen.name}" + ("" if record["complete"] else f"_n{n_run}")
        out.csv_path = write_rows_csv(directory / f"{prefix}_episodes.csv", out.rows)
        (directory / f"{prefix}_config.json").write_text(json.dumps(jsonable(settings), indent=2) + "\n",
                                                          encoding="utf-8")
        out.summary_path = directory / f"{prefix}_summary.json"
        summary["csv_path"] = str(out.csv_path)
        out.summary_path.write_text(json.dumps(jsonable(summary), indent=2) + "\n", encoding="utf-8")
    return out
