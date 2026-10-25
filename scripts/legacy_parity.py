#!/usr/bin/env python3
"""Re-run archived RUSCUAV v2.0.0 evaluations with the legacy notebook code and
compare the per-episode results with the archived CSVs.

The legacy code is loaded outside Colab with ``ruscuav_benchmark.legacy.runtime``.
Episodes are re-run with the exact configuration of the archived primary
no-terminal runs, then every shared column is compared episode by episode.

Examples::

    python scripts/legacy_parity.py --archive /data/RUSCUAV-v2.0.0 --work runs/parity \
        --method STATIC_Astar --episodes 5
    python scripts/legacy_parity.py --archive /data/RUSCUAV-v2.0.0 --work runs/parity \
        --method PPO:RISK+GREEN-RL:4001 --episodes 5

Requires the reproducibility dependencies (requirements-reproducibility.txt).
Exit code 0 means every compared value matched within the tolerance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ruscuav_benchmark.legacy.runtime import DRIVE_LITERAL, load_legacy_namespace  # noqa: E402

PLANNERS = {
    # name: (legacy evaluator, reported-config global, config class)
    "STATIC_Astar": ("evaluate_m9_static_planner_drive", "REPORTED_M9_CONFIG", "M9PlannerConfig"),
    "LOCAL_RH_Astar": ("evaluate_m10_receding_planner_drive", "REPORTED_M10_LOCAL_CONFIG", "M10RecedingHorizonPlannerConfig"),
    "FULL_RH_Astar": ("evaluate_m10_receding_planner_drive", "REPORTED_M10_FULL_CONFIG", "M10RecedingHorizonPlannerConfig"),
}
PLANNER_ARCHIVE = "experiments/paperA_primary_no_terminal_static_planners/eval/primary_no_terminal_{name}_n200_seed5000/primary_no_terminal_{name}_n200_seed5000_episodes.csv"
PPO_ARCHIVE = "experiments/paperA_submission_repair_v1/primary_no_terminal/static/{method}/{run}"

#: Wall-clock measurements differ between machines and are not compared.
IGNORED_COLUMNS = {"total_plan_time_s", "mean_plan_time_s", "last_plan_time_s"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _as_float(value):
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out


def _as_text(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return str(value).strip().lower()


def compare_episodes(archived: pd.DataFrame, rerun: pd.DataFrame, atol: float, rtol: float = 0.0) -> dict:
    """Compare shared columns seed by seed: numbers within atol + rtol*|archived|, text exactly."""
    archived = archived.set_index("seed")
    rerun = rerun.set_index("seed")
    columns = [c for c in rerun.columns if c in archived.columns and c not in IGNORED_COLUMNS]
    mismatches, max_abs_diff, exact_seeds = [], {}, set(rerun.index)
    for seed in rerun.index:
        if seed not in archived.index:
            mismatches.append({"seed": int(seed), "column": "<row>", "archived": "missing", "rerun": "present"})
            continue
        for col in columns:
            a, b = archived.at[seed, col], rerun.at[seed, col]
            fa, fb = _as_float(a), _as_float(b)
            if fa is not None and fb is not None:
                if math.isnan(fa) and math.isnan(fb):
                    continue
                diff = abs(fa - fb)
                max_abs_diff[col] = max(max_abs_diff.get(col, 0.0), diff)
                if diff > atol:
                    exact_seeds.discard(seed)
                if not diff <= atol + rtol * abs(fa):
                    mismatches.append({"seed": int(seed), "column": col, "archived": a, "rerun": b})
            elif _as_text(a) != _as_text(b):
                exact_seeds.discard(seed)
                mismatches.append({"seed": int(seed), "column": col, "archived": str(a), "rerun": str(b)})
    return {
        "episodes_compared": int(len(rerun)),
        "columns_compared": columns,
        "columns_not_compared": sorted(set(rerun.columns) - set(columns)),
        "mismatches": mismatches,
        "max_abs_diff": {k: v for k, v in sorted(max_abs_diff.items()) if v > 0},
        "episodes_fully_matching": int(len(set(rerun.index) - {m["seed"] for m in mismatches})),
        "episodes_exact_within_atol": int(len(exact_seeds)),
    }


def run_planner(ns, world, name: str, n: int, seed0: int, archive: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    evaluator, cfg_name, cfg_cls = PLANNERS[name]
    cfg = getattr(ns, cfg_cls)(**getattr(ns, cfg_name))
    dirs = ns.setup_ruscuav_drive_project(drive_root=str(ns.RUSCUAV_VIRTUAL_DRIVE), experiment_name="legacy_parity")
    regime = ns.PRIMARY_NO_TERMINAL
    kwargs = dict(
        world=world, drive_dirs=dirs, risk_profile="balanced", n_episodes=n, seed0=seed0,
        stage=regime.stage, mode=regime.mode, n_obstacles=0,
        prefix=f"parity_{name}_n{n}_seed{seed0}_{int(time.time())}", save_paths=True,
    )
    kwargs["planner_config" if evaluator.startswith("evaluate_m9") else "cfg"] = cfg
    rerun, _, csv_path, _ = getattr(ns, evaluator)(**kwargs)
    archived = pd.read_csv(archive / PLANNER_ARCHIVE.format(name=name))
    return archived, rerun, {"rerun_csv": str(csv_path), "config": getattr(ns, cfg_name)}


def run_ppo(ns, world, method: str, run: str, n: int, archive: Path, work: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    run_dir = archive / PPO_ARCHIVE.format(method=method, run=run)
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    model_path = Path(protocol["model_path"].replace(DRIVE_LITERAL, str(archive)))
    actual_sha = sha256_file(model_path)
    if actual_sha != protocol["model_sha256"]:
        raise RuntimeError(f"Checkpoint hash mismatch for {model_path}")
    out_dir = work / "ppo" / method / f"{run}_{int(time.time())}"
    rerun, _, csv_path, _ = ns.evaluate_ppo_resume(
        model_path=model_path,
        env_factory=lambda seed: ns.make_static_eval_env(method, world, ns.PRIMARY_NO_TERMINAL, seed),
        method=method,
        regime=ns.PRIMARY_NO_TERMINAL,
        output_dir=out_dir,
        n_episodes=n,
        seed0=int(protocol["seed0"]),
        deterministic=bool(protocol["deterministic"]),
        save_paths=True,
    )
    archived = pd.read_csv(run_dir / "episodes.csv")
    return archived, rerun, {"rerun_csv": str(csv_path), "model_path": str(model_path), "model_sha256": actual_sha}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path, help="Extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--work", required=True, type=Path, help="Writable work directory")
    p.add_argument("--method", required=True,
                   help="STATIC_Astar | LOCAL_RH_Astar | FULL_RH_Astar | PPO:<method>:<train seed or 'single'>")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed0", type=int, default=5000, help="First scenario seed (planners)")
    p.add_argument("--atol", type=float, default=1e-9, help="absolute tolerance for numeric values")
    p.add_argument("--rtol", type=float, default=1e-5,
                   help="relative tolerance; absorbs float32 differences between PyTorch builds "
                        "(discrete outcomes, counts and exposure times still have to match)")
    p.add_argument("--rerun-csv", type=Path, default=None,
                   help="compare an existing re-run CSV instead of running episodes again")
    args = p.parse_args()

    archive, work = args.archive.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.rerun_csv is not None:
        rerun = pd.read_csv(args.rerun_csv)
        if args.method in PLANNERS:
            archived = pd.read_csv(archive / PLANNER_ARCHIVE.format(name=args.method))
        else:
            _, method, seed = args.method.split(":", 2)
            run = "single_checkpoint" if seed == "single" else f"trainseed_{int(seed)}"
            archived = pd.read_csv(archive / PPO_ARCHIVE.format(method=method, run=run) / "episodes.csv")
        info = {"rerun_csv": str(args.rerun_csv)}
    else:
        ns = load_legacy_namespace(archive, work)
        world = ns.UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)
        if args.method in PLANNERS:
            archived, rerun, info = run_planner(ns, world, args.method, args.episodes, args.seed0, archive)
        elif args.method.startswith("PPO:"):
            _, method, seed = args.method.split(":", 2)
            run = "single_checkpoint" if seed == "single" else f"trainseed_{int(seed)}"
            archived, rerun, info = run_ppo(ns, world, method, run, args.episodes, archive, work)
        else:
            p.error(f"Unknown method {args.method!r}")

    report = {"method": args.method, "episodes": int(len(rerun)), "atol": args.atol, "rtol": args.rtol, **info,
              **compare_episodes(archived, rerun, args.atol, args.rtol), "elapsed_s": round(time.time() - t0, 1)}
    report_path = work / f"parity_{args.method.replace(':', '_').replace('+', 'p')}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    ok = not report["mismatches"]
    print(f"\n{args.method}: {report['episodes_fully_matching']}/{report['episodes_compared']} episodes match "
          f"on {len(report['columns_compared'])} columns (atol={args.atol}, rtol={args.rtol}); "
          f"{report['episodes_exact_within_atol']} bit-for-bit within atol; "
          f"max |diff| per column: {report['max_abs_diff'] or 'all exactly equal'}")
    if not ok:
        for m in report["mismatches"][:20]:
            print("  mismatch:", m)
    print("report:", report_path)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
