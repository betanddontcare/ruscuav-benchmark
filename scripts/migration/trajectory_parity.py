#!/usr/bin/env python3
"""Protocol evaluations against the archive, and trajectory scoring against the environment.

For each case a method is flown under a protocol (ruscuav_benchmark.protocols) with
its flown paths saved. Two comparisons follow:

1. the protocol run against the archived evaluation of the same method: the rows
   are compared as scripts/legacy_parity.py compares them (integers, flags and text
   exactly, other numbers to a relative 1e-5; planning times are not compared);
2. the saved paths, scored with ruscuav_benchmark.trajectories, against the rows of
   the flown episodes: every column the two have is compared exactly, except the
   cargo health, which the scorer estimates from the positions (its largest
   difference and any acceptability verdict it changes are reported), and the
   endings of the method's own that positions do not show (the action shield's
   shield-stuck, a planner without a path): they end as "ended". None of the
   episodes may leave the speed limits or the workspace. Steps counted as moving
   into a hazard are reported: on a step that the environment's geometric shield
   shortened, the samples of the shortened segment can find a hazard sliver that
   the shield's samples along the commanded segment skipped. So are episodes
   outside the acceleration limits: the geometric shield stops or shortens a step
   at once, faster than the vehicle's commands can.

Cases: the static, local and full receding-horizon A* in canonical-primary, the
dynamic planner in canonical-dynamic, the full planner in canonical-secondary (no
archived counterpart) and, with --vienna, the full planner at 30 m in vienna-primary;
with --policies (Stable-Baselines3 and --archive) also GEO-RL/1001 in
canonical-primary, DYN-OBS-RL/1001 in canonical-dynamic and, with --vienna, the
extended fine-tuned RISK+GREEN-RL in vienna-primary.

    python scripts/migration/trajectory_parity.py --archive /data/RUSCUAV-v2.0.0 --work runs/trajectory_parity \\
        --vienna /data/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz --policies
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.protocols import load_protocol, protocol_world  # noqa: E402
from ruscuav_benchmark.tables import jsonable, read_rows_csv  # noqa: E402
from ruscuav_benchmark.trajectories import score_trajectories  # noqa: E402

_PLAN = "experiments/paperA_primary_no_terminal_static_planners/eval/primary_no_terminal_{0}_n200_seed5000/" \
        "primary_no_terminal_{0}_n200_seed5000_episodes.csv"
PLANNER_CASES = [
    # (case, protocol, planner, planner settings, archived per-episode CSV or None)
    ("STATIC_Astar", "canonical-primary", "STATIC_Astar", {}, _PLAN.format("STATIC_Astar")),
    ("LOCAL_RH_Astar", "canonical-primary", "LOCAL_RH_Astar", {}, _PLAN.format("LOCAL_RH_Astar")),
    ("FULL_RH_Astar", "canonical-primary", "FULL_RH_Astar", {}, _PLAN.format("FULL_RH_Astar")),
    ("DYN_RH_Astar", "canonical-dynamic", "DYN_RH_Astar", {},
     "experiments/paperA_primary_no_terminal_dynamic/eval/primary_no_terminal_dynamic_receding_Astar/"
     "primary_no_terminal_dynamic_receding_Astar_episodes.csv"),
    ("FULL_RH_Astar_secondary", "canonical-secondary", "FULL_RH_Astar", {}, None),
    ("VIENNA_FULL_RH_Astar", "vienna-primary", "FULL_RH_Astar", {"planner_agl_m": 30.0},
     "experiments/paperA_primary_no_terminal_vienna/eval/primary_no_terminal_Vienna_FULL_RH_Astar/"
     "primary_no_terminal_Vienna_FULL_RH_Astar_episodes.csv"),
]
POLICY_CASES = [
    # (case, protocol, checkpoint, method preset, row format, label, archived evaluation)
    ("GEO-RL/1001", "canonical-primary", "GEO-RL/1001", "GEO-RL", "posthoc", "GEO-RL", "static/GEO-RL/1001"),
    ("DYN-OBS-RL/1001", "canonical-dynamic", "DYN-OBS-RL/1001", "DYN-OBS-RL", "dynamic",
     "RL_dynamic_obs_reward_only", "dynamic/DYN-OBS-RL/1001"),
    ("VIENNA/RISK+GREEN-RL/extended_ft", "vienna-primary", "VIENNA/RISK+GREEN-RL/extended_ft", "RISK+GREEN-RL",
     "transfer", "reward_only", "vienna/RISK+GREEN-RL/extended_finetuned"),
]
_TIMES = {"episode", "total_plan_time_s", "mean_plan_time_s", "last_plan_time_s"}
_NOT_SCORED = _TIMES | {"method", "policy_kind", "total_reward"}  # the label and the return are not scored
#: Endings the scorer reproduces; any other ending of the method's own (shield-stuck, no path) is "ended".
_POSITION_ENDINGS = {"success", "terrain_collision", "stuck", "near_success", "cargo", "timeout"}


def _number(v: Any) -> Optional[float]:
    if v is None or isinstance(v, (bool, str)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _same(a: Any, b: Any) -> bool:
    if _missing(a) or _missing(b):
        return _missing(a) and _missing(b)
    x, y = _number(a), _number(b)
    if x is None or y is None:
        return str(a) == str(b)
    return x == y


def _close(a: Any, b: Any, rtol: float = 1e-5, atol: float = 1e-9) -> bool:
    x, y = _number(a), _number(b)
    if _missing(a) or _missing(b) or x is None or y is None or isinstance(a, int) and isinstance(b, int):
        return _same(a, b)
    return abs(x - y) <= atol + rtol * abs(y)


def compare_with_archive(rows: List[Dict[str, Any]], archived: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_seed = {r["seed"]: r for r in archived}
    columns = [c for c in rows[0] if c in archived[0] and c not in _TIMES]
    identical = within = 0
    for row in rows:
        ref = by_seed[row["seed"]]
        identical += all(_same(row[c], ref[c]) for c in columns)
        within += all(_close(row[c], ref[c]) for c in columns)
    return {"columns": len(columns), "rows": len(rows), "identical": identical, "within_tolerance": within}


def compare_scored(flown: List[Dict[str, Any]], scored: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": len(flown), "identical_apart_from_cargo": 0, "cargo_max_abs_diff": 0.0,
                           "acceptability_changed_by_cargo": 0, "other_differences": {}, "endings": {},
                           "unflyable": {"outside_speed_limits": 0, "outside_acceleration_limits": 0,
                                         "outside_workspace": 0, "hazard_segments": 0}}
    for a, b in zip(flown, scored):
        assert a["seed"] == b["seed"]
        out["unflyable"]["outside_speed_limits"] += not b["within_speed_limits"]
        out["unflyable"]["outside_acceleration_limits"] += not b["within_acceleration_limits"]
        out["unflyable"]["outside_workspace"] += not b["within_workspace"]
        out["unflyable"]["hazard_segments"] += b["hazard_segment_steps"] > 0
        diffs = [c for c in a if c in b and c not in _NOT_SCORED and not _same(a[c], b[c])]
        if "cargo_health" in diffs:
            if _missing(a["cargo_health"]) or _missing(b["cargo_health"]):
                out["other_differences"]["cargo_health_missing"] = out["other_differences"].get(
                    "cargo_health_missing", 0) + 1
            else:
                out["cargo_max_abs_diff"] = max(out["cargo_max_abs_diff"], abs(float(a["cargo_health"]) -
                                                                                float(b["cargo_health"])))
            diffs.remove("cargo_health")
        acceptability = [c for c in diffs if c.startswith("mission_acceptable")]
        if acceptability:
            out["acceptability_changed_by_cargo"] += 1
            diffs = [c for c in diffs if c not in acceptability]
        if a["termination"] != b["termination"]:
            key = f"{a['termination']} -> {b['termination']}"
            out["endings"][key] = out["endings"].get(key, 0) + 1
            diffs = [c for c in diffs if c != "termination"]
        for c in diffs:
            out["other_differences"][c] = out["other_differences"].get(c, 0) + 1
        out["identical_apart_from_cargo"] += not diffs and a["termination"] == b["termination"]
    return out


def _paths_dir(csv_path: Path) -> Path:
    return csv_path.parent / (csv_path.name[: -len("_episodes.csv")] + "_paths")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path, help="extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--work", type=Path, default=Path("runs/trajectory_parity"))
    p.add_argument("--vienna", type=Path, default=None, help="processed Vienna footway-only world (not distributed)")
    p.add_argument("--policies", action="store_true", help="also the policy cases (Stable-Baselines3)")
    p.add_argument("--cases", nargs="*", default=None, help="run only these cases")
    p.add_argument("--episodes", type=int, default=None, help="first missions only (default all 200)")
    args = p.parse_args()

    from ruscuav_benchmark.planners import evaluate_planner

    report: Dict[str, Any] = {"episodes": args.episodes, "cases": {}}
    t_all = time.time()
    for case, protocol_name, planner, settings, archived_csv in PLANNER_CASES:
        if args.cases and case not in args.cases or (protocol_name.startswith("vienna") and args.vienna is None):
            continue
        t0 = time.time()
        protocol = load_protocol(protocol_name)
        world = protocol_world(protocol, args.vienna if protocol_name.startswith("vienna") else None)
        result = evaluate_planner(world, planner, protocol=protocol, n_episodes=args.episodes,
                                  output_dir=args.work / case, save_paths=True, **settings)
        entry: Dict[str, Any] = {"protocol": protocol_name, "planner": planner}
        if archived_csv is not None:
            entry["archive"] = compare_with_archive(result.rows, read_rows_csv(args.archive / archived_csv,
                                                                               exact_floats=True))
        scored = score_trajectories(protocol, _paths_dir(result.csv_path), world, label=planner,
                                    n_episodes=args.episodes)
        entry["trajectories"] = compare_scored(result.rows, scored)
        entry["elapsed_s"] = round(time.time() - t0, 1)
        report["cases"][case] = entry
        print(case, json.dumps(entry), flush=True)

    if args.policies:
        from ruscuav_benchmark.policies import ARCHIVED_EVALUATIONS, archived_checkpoint, evaluate_policy

        for case, protocol_name, checkpoint, method, row_format, label, evaluation in POLICY_CASES:
            if args.cases and case not in args.cases or (protocol_name.startswith("vienna") and args.vienna is None):
                continue
            t0 = time.time()
            protocol = load_protocol(protocol_name)
            world = protocol_world(protocol, args.vienna if protocol_name.startswith("vienna") else None)
            result = evaluate_policy(world, archived_checkpoint(checkpoint, args.archive), method, protocol=protocol,
                                     n_episodes=args.episodes, row_format=row_format, label=label,
                                     output_dir=args.work / case.replace("/", "_"), save_paths=True)
            archived = read_rows_csv(args.archive / ARCHIVED_EVALUATIONS[evaluation].episodes_csv, exact_floats=True)
            entry = {"protocol": protocol_name, "checkpoint": checkpoint,
                     "archive": compare_with_archive(result.rows, archived)}
            scored = score_trajectories(protocol, _paths_dir(result.csv_path), world, label=label,
                                        n_episodes=args.episodes)
            entry["trajectories"] = compare_scored(result.rows, scored)
            entry["elapsed_s"] = round(time.time() - t0, 1)
            report["cases"][case] = entry
            print(case, json.dumps(entry), flush=True)

    report["elapsed_s"] = round(time.time() - t_all, 1)
    args.work.mkdir(parents=True, exist_ok=True)
    out = args.work / "trajectory_parity.json"
    out.write_text(json.dumps(jsonable(report), indent=2) + "\n", encoding="utf-8")
    def unexpected(endings):
        return [k for k in endings if not (k.endswith(" -> ended") and k.split(" -> ")[0] not in _POSITION_ENDINGS
                                           and not k.startswith("urban_"))]

    bad = [c for c, e in report["cases"].items() if e["trajectories"]["other_differences"]
           or unexpected(e["trajectories"]["endings"]) or e["trajectories"]["unflyable"]["outside_speed_limits"]
           or e["trajectories"]["unflyable"]["outside_workspace"]]
    print(f"{len(report['cases'])} cases; report: {out}" + (f"; unexpected differences in {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
