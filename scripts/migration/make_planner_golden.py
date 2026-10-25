#!/usr/bin/env python3
"""Record reference planner episodes of the archived notebook code for tests.

tests/test_planner_episodes.py flies the same cases with ruscuav_benchmark.planners and
compares every plan, action, position and the episode row with these records, so
the planners stay checked against the notebook code without the Zenodo archive.

Usage: python scripts/migration/make_planner_golden.py --archive /data/RUSCUAV-v2.0.0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from planner_parity import Recorder, case, modified_worlds, run_reference  # noqa: E402
from ruscuav_benchmark.env import make_env  # noqa: E402
from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402
from ruscuav_benchmark.world import UrbanWorldLayers  # noqa: E402

#: (planner, scenario seed, configuration changes, planner-world modification, stage).
#: Short episodes of the four planners (one in stage C), a full episode that the
#: environment ends (success), and planning failures: the goal buried in buildings
#: (static, dynamic), walled in (receding: no path at the start), and walled in with
#: blockers sensed instead of known (receding: the wall is found during the flight,
#: later replans fail and the last plan is kept).
CASES = [
    ("STATIC_Astar", 5000, {"max_steps": 30}, None, "B"),
    ("LOCAL_RH_Astar", 5003, {"max_steps": 30}, None, "B"),
    ("FULL_RH_Astar", 5011, {"max_steps": 30}, None, "B"),
    ("DYN_RH_Astar", 5003, {"max_steps": 30}, None, "B"),
    ("DYN_RH_Astar", 5003, {"max_steps": 30}, None, "C"),
    ("STATIC_Astar", 5121, {}, None, "B"),
    ("STATIC_Astar", 5003, {"max_steps": 30}, "goal_blocked", "B"),
    ("DYN_RH_Astar", 5011, {"max_steps": 30}, "goal_blocked", "B"),
    ("LOCAL_RH_Astar", 5003, {"max_steps": 30}, "goal_enclosed", "B"),
    ("LOCAL_RH_Astar", 5003, {"max_steps": 8, "hard_blockers_known": False, "sensing_radius_m": 580.0,
                              "replan_every_steps": 1}, "goal_enclosed", "B"),
]
OUTPUT = REPO / "tests" / "data" / "planner_golden_AOI_SYNTH_000.json"


def _list(a):
    return None if a is None else np.asarray(a).tolist()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--work", type=Path, default=Path("runs/planner_golden"))
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()

    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_legacy_namespace(args.archive.resolve(), args.work.resolve())
    legacy_world = ns.UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)
    package_world = UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)

    records = []
    for planner, seed, changes, world_mod, stage in CASES:
        c = case(f"{planner}_{seed}", planner, (seed,), stage=stage, world_mod=world_mod, **changes)
        legacy_planner_world = legacy_world
        if world_mod is not None:
            probe = make_env(package_world, "RISK+OBS-RL", stage=c["stage"], mode=c["mode"])
            probe.reset(seed=seed)
            legacy_planner_world, _ = modified_worlds(legacy_world, package_world, probe.target_point[:2], world_mod)
        rec = Recorder()
        with contextlib.redirect_stdout(io.StringIO()):
            row = run_reference(ns, c, seed, legacy_world, legacy_planner_world, rec)
        records.append({
            "planner": planner, "seed": seed, "changes": changes, "world_mod": world_mod,
            "stage": c["stage"], "mode": c["mode"], "dynamic_prior_mode": c["dynamic_prior_mode"],
            "row": row,
            "actions": [_list(a) for a in rec.actions],
            "positions": [_list(x) for x in rec.positions],
            "plans": [{"start": _list(start), "path": _list(path), "cost": cost, "expanded": expanded,
                       "reason": reason} for start, path, cost, expanded, reason in rec.plans],
            "reset_info": rec.reset_infos[-1],
        })
        print(f"{planner} seed {seed} stage {stage} {world_mod or ''}: {len(rec.actions)} steps, "
              f"{len(rec.plans)} plans ({', '.join(sorted({p[4] for p in rec.plans}))}), {row['termination']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "description": "Planner episodes of the archived RUSCUAV v2.0.0 notebook code "
                       "(scripts/migration/make_planner_golden.py), in the notebook environment.",
        "world": "AOI_SYNTH_000",
        "cases": records,
    }, separators=(",", ":")) + "\n", encoding="utf-8")
    print("wrote", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
