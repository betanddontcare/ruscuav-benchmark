#!/usr/bin/env python3
"""Write the canonical-world missions of RUSCUAV v2.0.0 to a scenario CSV.

In the archived runs, start and target points were not stored: the legacy
environment sampled them in ``env.reset(seed)``. NumPy's Generator does not promise
identical draws across versions, so this script records them once, using the legacy
code loaded outside Colab, and cross-checks them:

* every evaluation environment variant (planners, GEO-RL, OBS-RL, RISK+GREEN-RL,
  RISK+OBS-RL) must produce the same start and target for a seed;
* start and target must equal the endpoints of the archived M9 planned paths.

Example::

    python scripts/materialize_legacy_scenarios.py --archive /data/RUSCUAV-v2.0.0 \\
        --work runs/scenarios --output scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402
from ruscuav_benchmark.world import canonical_aoi_id  # noqa: E402

POLICY_VARIANTS = ("GEO-RL", "OBS-RL", "RISK+GREEN-RL", "RISK+OBS-RL")
M9_PLANNED_PATHS = (
    "experiments/m9_static_planner_baseline/eval/"
    "m9_b0_static_astar_res5_agl40_lambda035_strict_eval_200/planned_paths/"
    "episode_{episode:05d}_seed_{seed}_planned_path.npy"
)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed0", type=int, default=5000)
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    archive = args.archive.resolve()
    with quiet():
        ns = load_legacy_namespace(archive, args.work)
    world = ns.UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)
    world_id = canonical_aoi_id(world.aoi_id)
    regime = ns.PRIMARY_NO_TERMINAL

    rows, problems, checked_paths = [], [], 0
    for episode in range(args.n):
        seed = args.seed0 + episode
        with quiet():
            env = ns.make_ruscuav_env_m4_mode(
                world=world, stage=regime.stage, mode=regime.mode,
                risk_profile="balanced", n_obstacles=0, seed=seed,
            )
            _, info = env.reset(seed=seed)
        start = np.asarray(env.start_point, dtype=np.float64).copy()
        target = np.asarray(env.target_point, dtype=np.float64).copy()
        env.close()

        for method in POLICY_VARIANTS:
            with quiet():
                other = ns.make_static_eval_env(method, world, regime, seed)
                other.reset(seed=seed)
            if not (np.array_equal(other.start_point, start) and np.array_equal(other.target_point, target)):
                problems.append(f"seed {seed}: {method} samples a different start/target")
            other.close()

        path_file = archive / M9_PLANNED_PATHS.format(episode=episode, seed=seed)
        if path_file.exists():
            planned = np.load(path_file, allow_pickle=False)
            checked_paths += 1
            if not (np.allclose(planned[0], start[:2], atol=1e-3) and np.allclose(planned[-1], target[:2], atol=1e-3)):
                problems.append(f"seed {seed}: archived M9 planned path endpoints differ")

        rows.append({
            "scenario_id": f"{world_id}_seed{seed}",
            "world_id": world_id,
            "seed": seed,
            "start_x": f"{start[0]:.3f}", "start_y": f"{start[1]:.3f}", "start_z": f"{start[2]:.6f}",
            "target_x": f"{target[0]:.3f}", "target_y": f"{target[1]:.3f}", "target_z": f"{target[2]:.6f}",
            "safe_start_target": bool(info.get("urban_safe_start_target", False)),
        })
        if (episode + 1) % 20 == 0:
            print(f"{episode + 1}/{args.n} seeds done, problems so far: {len(problems)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    provenance = {
        "description": "Canonical-world missions of the RUSCUAV v2.0.0 primary evaluation, recorded from the legacy env.reset(seed).",
        "world_id": world_id,
        "legacy_world_id": world.aoi_id,
        "seeds": [args.seed0, args.seed0 + args.n - 1],
        "environment": f"make_ruscuav_env_m4_mode(stage={regime.stage!r}, mode={regime.mode!r}, risk_profile='balanced', n_obstacles=0)",
        "variants_cross_checked": list(POLICY_VARIANTS),
        "archived_planned_paths_checked": checked_paths,
        "problems": problems,
        "z_note": "start_z/target_z are absolute altitudes in metres, in the same height frame as the world's terrain_z.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} missions); archived paths checked: {checked_paths}; problems: {len(problems)}")
    for line in problems[:20]:
        print("  ", line)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
