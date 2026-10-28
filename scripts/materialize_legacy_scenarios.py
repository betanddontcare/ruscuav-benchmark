#!/usr/bin/env python3
"""Write the missions of archived RUSCUAV v2.0.0 evaluations to a scenario CSV.

In the archived runs, start and target points were not stored: the legacy
environment sampled them in ``env.reset(seed)``. NumPy's Generator does not promise
identical draws across versions, so this script records them once, using the legacy
code loaded outside Colab, and cross-checks them:

* every environment the archived evaluations used in the world must produce the
  same start and target for a seed: the planner environment and the learned-policy
  evaluation environments (GEO-RL, OBS-RL, RISK+GREEN-RL, RISK+OBS-RL), for a
  processed real-GIS world (``--world``) also the M12C fine-tuning environments and
  the planner environment in the strict stage C;
* so must ``ruscuav_benchmark.env`` for every method preset;
* in the canonical world, start and target must equal the endpoints of the archived
  M9 planned paths.

The JSON file written next to the CSV identifies the world: file name, size and
SHA-256, and the checksum of its layers (``UrbanWorldLayers.layers_sha256``), which
a rebuilt world must match for the missions to apply.

Examples::

    python scripts/materialize_legacy_scenarios.py --archive /data/RUSCUAV-v2.0.0 \\
        --work runs/scenarios --output scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv
    python scripts/materialize_legacy_scenarios.py --archive /data/RUSCUAV-v2.0.0 \\
        --work runs/scenarios --world /data/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz \\
        --output scenarios/real_gis/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_seed7000_n200.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ruscuav_benchmark.env import make_env  # noqa: E402
from ruscuav_benchmark.env.config import METHOD_PRESETS, PRIMARY_NO_TERMINAL, SECONDARY_OPERATIONAL_STOP  # noqa: E402
from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402
from ruscuav_benchmark.world import UrbanWorldLayers, canonical_aoi_id  # noqa: E402

POLICY_VARIANTS = ("GEO-RL", "OBS-RL", "RISK+GREEN-RL", "RISK+OBS-RL")
M9_PLANNED_PATHS = (
    "experiments/m9_static_planner_baseline/eval/"
    "m9_b0_static_astar_res5_agl40_lambda035_strict_eval_200/planned_paths"
)
#: Index of the archived project's files with sizes and SHA-256 sums.
ASSET_INDEX = "reproducibility/v1_environment_and_audit/manifests/ASSET_INDEX.csv"


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def file_info(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": h.hexdigest()}


def archived_world_file(archive: Path, name: str):
    """The asset-index entry of the archived project for a world file name, if any."""
    index = archive / ASSET_INDEX
    if not index.exists():
        return None
    with index.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["relpath"] == f"worlds/{name}":
                return {"path": row["relpath"], "bytes": int(row["bytes"]), "sha256": row["sha256"],
                        "listed_in": ASSET_INDEX}
    return None


def legacy_variants(ns, world, real_gis: bool) -> list:
    """(label, factory(seed)) of the archived evaluation environments; each factory
    returns an environment reset with the seed."""
    primary, secondary = ns.PRIMARY_NO_TERMINAL, ns.SECONDARY_OPERATIONAL_STOP
    variants = [
        (f"make_static_eval_env({method!r}, {primary.name})",
         lambda seed, method=method: ns.make_static_eval_env(method, world, primary, seed))
        for method in POLICY_VARIANTS
    ]
    if real_gis:
        for regime in (primary, secondary):
            for kind in ("reward_only", "geometry_only"):
                variants.append((
                    f"make_m12c_real_vienna_env_mode({kind!r}, stage={regime.stage!r}, mode={regime.mode!r})",
                    lambda seed, kind=kind, regime=regime: ns.make_m12c_real_vienna_env_mode(
                        world, policy_kind=kind, stage=regime.stage, mode=regime.mode, seed=seed),
                ))
        variants.append((
            f"make_ruscuav_env_m4_mode(stage={secondary.stage!r}, mode={secondary.mode!r})",
            lambda seed: ns.make_ruscuav_env_m4_mode(world=world, stage=secondary.stage, mode=secondary.mode,
                                                     risk_profile="balanced", n_obstacles=0, seed=seed),
        ))
    return variants


def refactored_variants(world: UrbanWorldLayers) -> list:
    """(label, factory(seed)) of ruscuav_benchmark.env: every method preset in the
    primary regime, the planner environment (RISK+OBS-RL) also in the secondary one."""
    configs = [(method, PRIMARY_NO_TERMINAL.name) for method in METHOD_PRESETS]
    configs.append(("RISK+OBS-RL", SECONDARY_OPERATIONAL_STOP.name))
    return [
        (f"ruscuav_benchmark.env.make_env({method!r}, regime={regime!r})",
         lambda seed, method=method, regime=regime: make_env(world, method, regime=regime, seed=seed))
        for method, regime in configs
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path, help="Extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--work", required=True, type=Path, help="Writable work directory")
    p.add_argument("--output", required=True, type=Path, help="Scenario CSV to write; the JSON goes next to it")
    p.add_argument("--world", type=Path, default=None,
                   help="processed real-GIS world NPZ, which the archive does not contain "
                        "(default: the canonical world of the archive)")
    p.add_argument("--seed0", type=int, default=None, help="first seed (default 5000; 7000 with --world)")
    p.add_argument("--n", type=int, default=200, help="number of seeds (default 200)")
    p.add_argument("--note", action="append", default=[], help="text to record in the JSON (repeatable)")
    args = p.parse_args()

    archive = args.archive.resolve()
    real_gis = args.world is not None
    seed0 = args.seed0 if args.seed0 is not None else (7000 if real_gis else 5000)
    with quiet():
        ns = load_legacy_namespace(archive, args.work)
    # absolute(), not resolve(): the file name recorded is the one given, even for a link.
    world_path = args.world.absolute() if real_gis else Path(ns.CANONICAL_WORLD_PATH)
    with quiet():
        world = ns.UrbanWorldLayers.load_npz(world_path)
    world_layers = UrbanWorldLayers.load_npz(world_path)
    world_id = canonical_aoi_id(world.aoi_id)
    regime = ns.PRIMARY_NO_TERMINAL
    variants = legacy_variants(ns, world, real_gis) + refactored_variants(world_layers)

    rows, problems, checked_paths, t0 = [], [], 0, time.time()
    for episode in range(args.n):
        seed = seed0 + episode
        with quiet():
            env = ns.make_ruscuav_env_m4_mode(
                world=world, stage=regime.stage, mode=regime.mode,
                risk_profile="balanced", n_obstacles=0, seed=seed,
            )
            first_start, first_target = env.start_point.copy(), env.target_point.copy()
            _, info = env.reset(seed=seed)
        start = np.asarray(env.start_point, dtype=np.float64).copy()
        target = np.asarray(env.target_point, dtype=np.float64).copy()
        env.close()
        if not (np.array_equal(first_start, start) and np.array_equal(first_target, target)):
            problems.append(f"seed {seed}: the factory's reset and env.reset(seed) sample different missions")

        for label, factory in variants:
            with quiet():
                other = factory(seed)
            if not (np.array_equal(other.start_point, start) and np.array_equal(other.target_point, target)):
                problems.append(f"seed {seed}: {label} samples a different start/target")
            other.close()

        if not real_gis:
            for path_file in sorted((archive / M9_PLANNED_PATHS).glob(f"episode_*_seed_{seed}_planned_path.npy")):
                planned = np.load(path_file, allow_pickle=False)
                checked_paths += 1
                if not (np.allclose(planned[0], start[:2], atol=1e-3)
                        and np.allclose(planned[-1], target[:2], atol=1e-3)):
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
            print(f"{episode + 1}/{args.n} seeds done in {time.time() - t0:.0f} s, problems so far: {len(problems)}",
                  flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    if real_gis:
        description = (f"Missions of the RUSCUAV v2.0.0 evaluations in the processed real-GIS world {world_id}, "
                       "recorded from the legacy env.reset(seed).")
    else:
        description = ("Canonical-world missions of the RUSCUAV v2.0.0 primary evaluation, recorded from the "
                       "legacy env.reset(seed).")
    archived_file = archived_world_file(archive, world_path.name)
    provenance = {
        "description": description,
        "world_id": world_id,
        "legacy_world_id": world.aoi_id,
        "seeds": [seed0, seed0 + args.n - 1],
        "world_file": file_info(world_path),
        "world_layers_sha256": world_layers.layers_sha256(),
        **({"archived_world_file": archived_file} if archived_file else {}),
        "environment": f"make_ruscuav_env_m4_mode(stage={regime.stage!r}, mode={regime.mode!r}, "
                       "risk_profile='balanced', n_obstacles=0)",
        "variants_cross_checked": [label for label, _ in variants],
        "archived_planned_paths_checked": checked_paths,
        "problems": problems,
        **({"notes": args.note} if args.note else {}),
        "z_note": "start_z/target_z are absolute altitudes in metres, in the same height frame as the world's "
                  "terrain_z.",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(rows)} missions); variants cross-checked: {len(variants)}; "
          f"archived paths checked: {checked_paths}; problems: {len(problems)}")
    for line in problems[:20]:
        print("  ", line)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
