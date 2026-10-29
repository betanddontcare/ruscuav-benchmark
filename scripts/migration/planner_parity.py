#!/usr/bin/env python3
"""Step-by-step parity between the notebook's planners and ruscuav_benchmark.planners.

For every case the notebook rollout (rollout_m9_static_planner_episode,
rollout_m10_receding_planner_episode or rollout_m11_dynamic_receding_planner_episode,
in the notebook environment or, with --env refactored, in ruscuav_benchmark.env) and
the package rollout (in ruscuav_benchmark.env) fly the same mission. Both record
every plan (start, path, cost, expanded cells, reason), every action and every
position. A case passes when all of these, the reset info and the episode row are
identical, bit for bit; only the planning wall-clock times are not compared.

Cases: the four planners of the paper on several missions in both regimes;
configuration variants (Dijkstra, the four priors, unknown blockers, replanning
interval, sensing radius, planning altitudes, grid sizes, blocking switches, risk
weights, the "mean" dynamic prior, controller settings, step limits); and planning
failures, flown with a planner world in which the goal is buried in buildings or
walled in. The grid, the beliefs and single plans are also compared function by
function on random positions.

Example::

    python scripts/migration/planner_parity.py --archive /data/RUSCUAV-v2.0.0
    python scripts/migration/planner_parity.py --archive /data/RUSCUAV-v2.0.0 --cases presets --env refactored
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from ruscuav_benchmark.env import make_env  # noqa: E402
from ruscuav_benchmark.env.dynamic_risk import REPORTED_DYNAMIC_CONFIG, DynamicRiskField  # noqa: E402
from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402
from ruscuav_benchmark.planners import (  # noqa: E402
    PLANNER_PRESETS,
    DynamicRecedingPlanner,
    PlannerConfig,
    RecedingHorizonPlanner,
    RecedingPlannerConfig,
    RiskCostGrid,
    StaticPlanner,
    run_dynamic_episode,
    run_receding_episode,
    run_static_episode,
)
from ruscuav_benchmark.world import UrbanWorldLayers  # noqa: E402

TIMING_KEYS = {"total_plan_time_s", "mean_plan_time_s", "last_plan_time_s", "plan_time_s"}
ENV_METHOD = {"static": "RISK+OBS-RL", "receding": "RISK+OBS-RL", "dynamic": "DYN-RISK-RL"}


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


# ---------------------------------------------------------------------- recording
class Recorder:
    def __init__(self) -> None:
        self.actions: List[np.ndarray] = []
        self.positions: List[np.ndarray] = []
        self.plans: List[tuple] = []
        self.reset_infos: List[dict] = []


class EnvProxy:
    """Forwards to an environment and records actions, positions and reset infos."""

    def __init__(self, env, rec: Recorder) -> None:
        object.__setattr__(self, "_env", env)
        object.__setattr__(self, "_rec", rec)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        setattr(self._env, name, value)

    def reset(self, *args, **kwargs):
        out = self._env.reset(*args, **kwargs)
        self._rec.reset_infos.append(dict(out[1]))
        return out

    def step(self, action):
        self._rec.actions.append(np.array(action, dtype=np.float32, copy=True))
        out = self._env.step(action)
        self._rec.positions.append(np.array(self._env.current_pos, copy=True))
        return out


def record_plans(planner, rec: Recorder) -> None:
    original = planner.plan

    def plan(start_xy, target_xy):
        result = original(start_xy, target_xy)
        path = result.get("path_xy")
        rec.plans.append((np.asarray(start_xy, dtype=np.float32).copy(), None if path is None else path.copy(),
                          result["cost"], result["expanded"], result["reason"]))
        return result

    planner.plan = plan


# ---------------------------------------------------------------------- comparison
def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and \
            a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return type(a) is type(b) and a == b


#: Columns the package's rows add to the notebook's: the dynamic planner's building time
#: (docs/PLANNERS.md).
ADDED_KEYS = ("building_time",)


def compare_rows(ref: dict, new: dict) -> List[str]:
    problems = []
    new = {key: value for key, value in new.items() if key in ref or key not in ADDED_KEYS}
    if list(ref) != list(new):
        problems.append(f"row keys differ: {list(ref)} vs {list(new)}")
    for key in ref:
        if key in TIMING_KEYS or key not in new:
            continue
        if not _same(ref[key], new[key]):
            problems.append(f"row[{key!r}]: {ref[key]!r} vs {new[key]!r}")
    return problems


def compare_records(ref: Recorder, new: Recorder) -> List[str]:
    problems = []
    if len(ref.actions) != len(new.actions):
        problems.append(f"{len(ref.actions)} vs {len(new.actions)} steps")
    for i, (a, b) in enumerate(zip(ref.actions, new.actions)):
        if not _same(a, b):
            problems.append(f"action {i}: {a} vs {b}")
            break
    for i, (a, b) in enumerate(zip(ref.positions, new.positions)):
        if not _same(a, b):
            problems.append(f"position {i}: {a} vs {b}")
            break
    if len(ref.plans) != len(new.plans):
        problems.append(f"{len(ref.plans)} vs {len(new.plans)} plans")
    for i, (a, b) in enumerate(zip(ref.plans, new.plans)):
        for name, x, y in zip(("start", "path", "cost", "expanded", "reason"), a, b):
            if x is None or y is None:
                same = x is None and y is None
            elif isinstance(x, np.ndarray):
                same = _same(x, y)
            else:
                same = (x == y) or (isinstance(x, float) and math.isinf(x) and math.isinf(y))
            if not same:
                problems.append(f"plan {i} {name}: {x!r} vs {y!r}")
                break
    ref_info = ref.reset_infos[-1] if ref.reset_infos else {}
    new_info = new.reset_infos[-1] if new.reset_infos else {}
    if ref_info != new_info:
        problems.append(f"reset info: {ref_info} vs {new_info}")
    return problems


# ---------------------------------------------------------------------- worlds
def modified_worlds(legacy_world, package_world, target_xy, kind: str):
    """Copies of both worlds in which the goal is buried in a building block or walled in."""
    res = float(package_world.resolution_m)
    rows, cols = package_world.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    cx, cy = (xx + 0.5) * res, (yy + 0.5) * res
    cheb = np.maximum(np.abs(cx - float(target_xy[0])), np.abs(cy - float(target_xy[1])))
    if kind == "goal_blocked":
        area = cheb <= 125.0             # more than 20 planning cells of 5 m around the goal
    else:
        area = (cheb >= 60.0) & (cheb <= 80.0)
    out = []
    for world in (legacy_world, package_world):
        building_mask = np.array(world.building_mask, copy=True)
        building_height = np.array(world.building_height, copy=True)
        building_mask[area] = 1.0
        building_height[area] = 200.0
        world_copy = type(world).__new__(type(world))
        world_copy.__dict__.update(world.__dict__)
        world_copy.building_mask = building_mask.astype(np.float32)
        world_copy.building_height = building_height.astype(np.float32)
        world_copy.building_roof_z = (world.terrain_z + world_copy.building_height).astype(np.float32)
        out.append(world_copy)
    return out


# ---------------------------------------------------------------------- cases
def preset_fields(name: str) -> dict:
    return asdict(PLANNER_PRESETS[name].config)


def case(name, planner, seeds, stage="B", mode="strict_eval", dynamic_prior_mode="zero", world_mod=None, **changes):
    spec = PLANNER_PRESETS[planner]
    fields = {**preset_fields(planner), **changes}
    return {"name": name, "kind": spec.kind, "fields": fields, "seeds": list(seeds), "stage": stage, "mode": mode,
            "dynamic_prior_mode": dynamic_prior_mode, "world_mod": world_mod}


def build_cases(group: str) -> List[dict]:
    presets = [case(f"{p}_primary", p, (5000, 5003, 5011)) for p in PLANNER_PRESETS]
    presets += [case(f"{p}_secondary", p, (5003, 5011), stage="C") for p in ("FULL_RH_Astar", "DYN_RH_Astar")]
    short = dict(max_steps=60)
    variants = [
        case("STATIC_dijkstra", "STATIC_Astar", (5003,), use_astar=False, **short),
        case("FULL_dijkstra", "FULL_RH_Astar", (5003,), use_astar=False, **short),
        case("LOCAL_prior_zero", "LOCAL_RH_Astar", (5003,), prior_risk_mode="zero", **short),
        case("LOCAL_prior_blurred", "LOCAL_RH_Astar", (5003,), prior_risk_mode="blurred", **short),
        case("LOCAL_blockers_unknown", "LOCAL_RH_Astar", (5003,), hard_blockers_known=False, **short),
        case("LOCAL_replan_every_step", "LOCAL_RH_Astar", (5003,), replan_every_steps=1, max_steps=30),
        case("LOCAL_sensing_30m", "LOCAL_RH_Astar", (5003,), sensing_radius_m=30.0, **short),
        case("LOCAL_slow_controller", "LOCAL_RH_Astar", (5003,), target_speed_mps=6.0, waypoint_radius_m=4.0, **short),
        case("STATIC_agl90_no_sensitive_block", "STATIC_Astar", (5003,), planner_agl_m=90.0, **short),
        case("STATIC_agl25", "STATIC_Astar", (5003,), planner_agl_m=25.0, **short),
        case("STATIC_no_building_canopy_block", "STATIC_Astar", (5003,), block_buildings=False, block_canopy=False,
             **short),
        case("STATIC_lambda0", "STATIC_Astar", (5003,), risk_lambda=0.0, **short),
        case("STATIC_lambda3", "STATIC_Astar", (5003,), risk_lambda=3.0, **short),
        case("STATIC_res10", "STATIC_Astar", (5003,), plan_res_m=10.0, **short),
        case("STATIC_res3", "STATIC_Astar", (5003,), plan_res_m=3.0, **short),
        case("FULL_res3", "FULL_RH_Astar", (5003,), plan_res_m=3.0, **short),
        case("DYN_prior_mean", "DYN_RH_Astar", (5003,), dynamic_prior_mode="mean", **short),
        case("DYN_replan7_sensing120", "DYN_RH_Astar", (5003,), replan_every_steps=7, sensing_radius_m=120.0, **short),
    ]
    failures = [
        case(f"{p}_{mod}", p, (5003,), world_mod=mod, **short)
        for mod in ("goal_blocked", "goal_enclosed") for p in ("STATIC_Astar", "LOCAL_RH_Astar", "DYN_RH_Astar")
    ]
    groups = {"presets": presets, "variants": variants, "failures": failures}
    if group == "all":
        return presets + variants + failures
    return groups[group]


# ---------------------------------------------------------------------- runs
def run_reference(ns, c, seed, legacy_env_world, legacy_planner_world, rec: Recorder):
    fields = c["fields"]
    stage, mode = c["stage"], c["mode"]
    factory_name = "make_ruscuav_env_m11_dynamic_mode" if c["kind"] == "dynamic" else "make_ruscuav_env_m4_mode"
    original_factory = getattr(ns, factory_name)
    setattr(ns, factory_name, lambda *a, **k: EnvProxy(original_factory(*a, **k), rec))
    original_m11 = ns.M11DynamicRecedingRiskPlanner
    try:
        if c["kind"] == "static":
            known = ns.M9PlannerConfig.__dataclass_fields__
            cfg = ns.M9PlannerConfig(**{k: v for k, v in fields.items() if k in known})
            planner = ns.M9StaticRiskCostPlanner(world=legacy_planner_world, config=cfg, risk_profile="balanced")
            record_plans(planner, rec)
            row, _, _ = ns.rollout_m9_static_planner_episode(
                world=legacy_env_world, planner=planner, seed=seed, cfg=cfg, stage=stage, mode=mode, n_obstacles=0,
                max_steps=cfg.max_steps)
        elif c["kind"] == "receding":
            cfg = ns.M10RecedingHorizonPlannerConfig(**fields)
            planner = ns.M10RecedingHorizonRiskPlanner(world=legacy_planner_world, config=cfg, risk_profile="balanced")
            record_plans(planner, rec)
            row, _ = ns.rollout_m10_receding_planner_episode(
                world=legacy_env_world, planner=planner, seed=seed, cfg=cfg, stage=stage, mode=mode, n_obstacles=0,
                max_steps=cfg.max_steps)
        else:
            cfg = ns.M10RecedingHorizonPlannerConfig(**fields)

            class Recorded(original_m11):
                def __init__(self, world, *args, **kwargs):
                    super().__init__(legacy_planner_world, *args, **kwargs)
                    record_plans(self, rec)

            ns.M11DynamicRecedingRiskPlanner = Recorded
            row, _ = ns.rollout_m11_dynamic_receding_planner_episode(
                world=legacy_env_world, seed=seed, cfg=cfg, dynamic_config=ns.reported_dynamic_config(),
                dynamic_prior_mode=c["dynamic_prior_mode"], stage=stage, mode=mode, n_obstacles=0,
                max_steps=cfg.max_steps)
    finally:
        setattr(ns, factory_name, original_factory)
        ns.M11DynamicRecedingRiskPlanner = original_m11
    return row


def run_candidate(c, seed, env_world, planner_world, rec: Recorder):
    fields = c["fields"]
    env = EnvProxy(make_env(env_world, ENV_METHOD[c["kind"]], stage=c["stage"], mode=c["mode"]), rec)
    if c["kind"] == "static":
        planner = StaticPlanner(planner_world, PlannerConfig(**{k: v for k, v in fields.items()
                                                                if k in PlannerConfig.__dataclass_fields__}))
        record_plans(planner, rec)
        row, _, _ = run_static_episode(env, planner, seed)
    elif c["kind"] == "receding":
        planner = RecedingHorizonPlanner(planner_world, RecedingPlannerConfig(**fields))
        record_plans(planner, rec)
        row, _ = run_receding_episode(env, planner, seed)
    else:
        planner = DynamicRecedingPlanner(planner_world, RecedingPlannerConfig(**fields),
                                         dynamic_prior_mode=c["dynamic_prior_mode"])
        record_plans(planner, rec)
        row, _ = run_dynamic_episode(env, planner, seed)
    return row


def run_cases(ns, cases, legacy_world, package_world, env_label) -> List[dict]:
    results = []
    for c in cases:
        for seed in c["seeds"]:
            t0 = time.time()
            legacy_planner_world, package_planner_world = legacy_world, package_world
            if c["world_mod"] is not None:
                probe = make_env(package_world, "RISK+OBS-RL", stage=c["stage"], mode=c["mode"])
                probe.reset(seed=seed)
                legacy_planner_world, package_planner_world = modified_worlds(
                    legacy_world, package_world, probe.target_point[:2], c["world_mod"])
            ref, new = Recorder(), Recorder()
            with quiet():
                ref_row = run_reference(ns, c, seed, legacy_world, legacy_planner_world, ref)
            new_row = run_candidate(c, seed, package_world, package_planner_world, new)
            problems = compare_rows(ref_row, new_row) + compare_records(ref, new)
            result = {
                "case": c["name"], "seed": seed, "env": env_label, "ok": not problems, "problems": problems[:10],
                "steps": len(ref.actions), "plans": len(ref.plans), "termination": ref_row.get("termination"),
                "planner_reason": ref.plans[-1][4] if ref.plans else None, "elapsed_s": round(time.time() - t0, 1),
            }
            results.append(result)
            status = "ok  " if result["ok"] else "DIFF"
            print(f"{status} {c['name']:36s} seed {seed}: {result['steps']:4d} steps, {result['plans']:3d} plans, "
                  f"{result['termination']}, last plan {result['planner_reason']} ({result['elapsed_s']} s)",
                  flush=True)
            for problem in problems[:5]:
                print("      ", problem)
    return results


# ---------------------------------------------------------------------- function level
def compare_functions(ns, legacy_world, package_world, n_random: int = 40, seed: int = 0) -> List[dict]:
    """Grids, beliefs and single plans on random positions."""
    rng = np.random.default_rng(seed)
    results = []
    configs = [
        ("preset", {}), ("dijkstra", {"use_astar": False}), ("agl25", {"planner_agl_m": 25.0}),
        ("agl90", {"planner_agl_m": 90.0}), ("res3", {"plan_res_m": 3.0}), ("res10", {"plan_res_m": 10.0}),
        ("lambda0", {"risk_lambda": 0.0}), ("no_block", {"block_buildings": False, "block_canopy": False,
                                                         "block_sensitive_low_altitude": False}),
    ]
    for label, changes in configs:
        fields = {**preset_fields("STATIC_Astar"), **changes}
        legacy = ns.M9StaticRiskCostPlanner(world=legacy_world, config=ns.M9PlannerConfig(**fields))
        grid = RiskCostGrid(package_world, PlannerConfig(**fields))
        problems = [f"grid {name} differs" for name in ("risk_low", "risk_norm", "block", "cell_cost")
                    if not _same(getattr(legacy, name), getattr(grid, name))]
        if legacy.plan_res_m != grid.plan_res_m or (legacy.grid_h, legacy.grid_w) != (grid.grid_h, grid.grid_w):
            problems.append("grid geometry differs")
        planner = StaticPlanner(package_world, PlannerConfig(**fields), grid=grid)
        for _ in range(n_random // 4):
            a = rng.uniform(0.0, 1000.0, size=2).astype(np.float32)
            b = rng.uniform(0.0, 1000.0, size=2).astype(np.float32)
            p, q = legacy.plan(a, b), planner.plan(a, b)
            if not (p["reason"] == q["reason"] and _same(p["path_xy"], q["path_xy"]) and p["expanded"] == q["expanded"]
                    and (p["cost"] == q["cost"] or (math.isinf(p["cost"]) and math.isinf(q["cost"])))):
                problems.append(f"plan {a} -> {b}: {p['reason']}/{p['cost']} vs {q['reason']}/{q['cost']}")
        results.append({"check": f"grid+plans_{label}", "ok": not problems, "problems": problems[:5]})

    for prior in ("zero", "mean", "blurred", "perfect"):
        for known in (True, False):
            fields = {**preset_fields("LOCAL_RH_Astar"), "prior_risk_mode": prior, "hard_blockers_known": known}
            legacy = ns.M10RecedingHorizonRiskPlanner(world=legacy_world,
                                                      config=ns.M10RecedingHorizonPlannerConfig(**fields))
            new = RecedingHorizonPlanner(package_world, RecedingPlannerConfig(**fields))
            problems = []
            for step in range(6):
                xy = rng.uniform(0.0, 1000.0, size=2).astype(np.float32)
                legacy.reveal_around_xy(xy)
                new.reveal_around_xy(xy)
                for name in ("known_mask", "known_block", "current_cell_cost", "current_block", "prior_risk_norm"):
                    if not _same(getattr(legacy, name), getattr(new, name)):
                        problems.append(f"after reveal {step}: {name} differs")
                b = rng.uniform(0.0, 1000.0, size=2).astype(np.float32)
                p, q = legacy.plan(xy, b), new.plan(xy, b)
                if not (p["reason"] == q["reason"] and _same(p["path_xy"], q["path_xy"])):
                    problems.append(f"plan after reveal {step} differs")
            if (legacy.replan_count, legacy.total_expanded) != (new.replan_count, new.total_expanded):
                problems.append("planning counters differ")
            results.append({"check": f"belief_{prior}_blockers_{'known' if known else 'sensed'}", "ok": not problems,
                            "problems": problems[:5]})

    dyn_cfg = ns.reported_dynamic_config()
    for mode in ("zero", "mean"):
        for episode_seed in (5003, 5011):
            fields = preset_fields("DYN_RH_Astar")
            legacy = ns.M11DynamicRecedingRiskPlanner(
                world=legacy_world, config=ns.M10RecedingHorizonPlannerConfig(**fields), dynamic_config=dyn_cfg,
                episode_seed=episode_seed, dynamic_prior_mode=mode)
            legacy.reset_belief()
            new = DynamicRecedingPlanner(package_world, RecedingPlannerConfig(**fields), dynamic_prior_mode=mode)
            new.reset_belief(DynamicRiskField(package_world, REPORTED_DYNAMIC_CONFIG, seed=episode_seed))
            problems = []
            for step in (0, 40, 90, 150, 240):
                xy = rng.uniform(0.0, 1000.0, size=2).astype(np.float32)
                legacy.reveal_around_xy(xy, step=step)
                new.reveal_around_xy(xy, step=step)
                for name in ("known_mask", "dynamic_known_mask", "current_dynamic_true_norm", "current_cell_cost"):
                    if not _same(getattr(legacy, name), getattr(new, name)):
                        problems.append(f"step {step}: {name} differs")
                b = rng.uniform(0.0, 1000.0, size=2).astype(np.float32)
                p, q = legacy.plan(xy, b), new.plan(xy, b)
                if not (p["reason"] == q["reason"] and _same(p["path_xy"], q["path_xy"])):
                    problems.append(f"plan at step {step} differs")
            results.append({"check": f"dynamic_belief_{mode}_seed{episode_seed}", "ok": not problems,
                            "problems": problems[:5]})
    for r in results:
        print(f"{'ok  ' if r['ok'] else 'DIFF'} {r['check']}", *r["problems"][:3], flush=True)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path, help="Extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--work", type=Path, default=Path("runs/planner_parity"))
    p.add_argument("--cases", default="all", choices=("all", "presets", "variants", "failures", "functions", "none"),
                   help="episode cases to fly ('functions' or 'none': only the function-level checks)")
    p.add_argument("--no-functions", action="store_true", help="skip the function-level checks")
    p.add_argument("--env", choices=("legacy", "refactored"), default="legacy",
                   help="environment of the notebook rollouts (the package always uses ruscuav_benchmark.env)")
    p.add_argument("--report", type=Path, default=None, help="JSON report (default: <work>/planner_parity.json)")
    args = p.parse_args()

    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with quiet():
        ns = load_legacy_namespace(args.archive.resolve(), work)
    if args.env == "refactored":
        from legacy_parity import use_refactored_env

        use_refactored_env(ns)
    legacy_world = ns.UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)
    package_world = UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)

    t0 = time.time()
    report: Dict[str, Any] = {"env": args.env}
    if not args.no_functions:
        report["functions"] = compare_functions(ns, legacy_world, package_world)
    if args.cases not in ("functions", "none"):
        report["episodes"] = run_cases(ns, build_cases(args.cases), legacy_world, package_world, args.env)
    report["elapsed_s"] = round(time.time() - t0, 1)
    checks = report.get("functions", []) + report.get("episodes", [])
    report["ok"] = all(r["ok"] for r in checks)
    path = args.report or work / f"planner_parity_{args.cases}_{args.env}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    n_bad = sum(not r["ok"] for r in checks)
    print(f"\n{len(checks) - n_bad}/{len(checks)} checks identical; report: {path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
