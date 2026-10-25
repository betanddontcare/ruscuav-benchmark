#!/usr/bin/env python3
"""Re-run archived RUSCUAV v2.0.0 evaluations with the legacy notebook code and
compare the per-episode results with the archived CSVs.

The legacy code is loaded outside Colab with ``ruscuav_benchmark.legacy.runtime``.
Episodes are re-run with the exact configuration of the archived runs (primary
no-terminal planners and PPO policies, strict reward-factor evaluations), then every
shared column is compared episode by episode.

Examples::

    python scripts/legacy_parity.py --archive /data/RUSCUAV-v2.0.0 --work runs/parity \
        --method STATIC_Astar --episodes 5
    python scripts/legacy_parity.py --archive /data/RUSCUAV-v2.0.0 --work runs/parity \
        --method PPO:RISK+GREEN-RL:4001 --episodes 5
    python scripts/legacy_parity.py --archive /data/RUSCUAV-v2.0.0 --work runs/parity \
        --method REWARD_FACTOR:risk_only:1001 --episodes 3

With ``--env refactored`` the legacy evaluation code drives the refactored
environment (ruscuav_benchmark.env) instead of the notebook classes: the
notebook's environment factories are replaced by those of
ruscuav_benchmark.env.legacy_factories before the episodes run.

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

#: Real-GIS full-map receding A* runs (planner altitude 30 m, seeds from 7000). The
#: processed worlds are not in the archive; pass them with --real-world.
REAL_PLANNERS = {
    "VIENNA_FULL_RH_Astar": (
        "vienna",
        "experiments/paperA_primary_no_terminal_vienna/eval/primary_no_terminal_Vienna_FULL_RH_Astar/"
        "primary_no_terminal_Vienna_FULL_RH_Astar_episodes.csv",
    ),
    "WARSAW_FULL_RH_Astar": (
        "warsaw",
        "experiments/paperA_primary_no_terminal_warsaw_scenario_aligned/eval/"
        "primary_no_terminal_Warsaw_FULL_RH_Astar_seed7000/primary_no_terminal_Warsaw_FULL_RH_Astar_seed7000_episodes.csv",
    ),
}
#: File names the notebook used for the processed real-GIS worlds on Drive.
REAL_WORLD_DRIVE_NAMES = {
    "vienna": "worlds/vienna_prater_leopoldstadt_osm_1km_v1_urban_world_layers.npz",
    "warsaw": "worlds/warsaw_centrum_osm_1km_v1_urban_world_layers.npz",
}
PPO_ARCHIVE = "experiments/paperA_submission_repair_v1/primary_no_terminal/static/{method}/{run}"
#: Strict (stage C) evaluations of the reward-factor ablation (RISK-RL = risk_only,
#: GREEN-RL = green_only), episodes from seed 5000.
REWARD_FACTOR_ARCHIVE = "experiments/paperA_reward_factor_{variant}_seed{seed}"
REWARD_FACTOR_PREFIX = "paperA_reward_factor_{variant}_trainseed{seed}_strict_eval_200"

#: Not compared: wall-clock measurements differ between machines, and "episode" is a
#: run-local counter (rows are matched by seed, so a run may start at any seed).
IGNORED_COLUMNS = {"episode", "total_plan_time_s", "mean_plan_time_s", "last_plan_time_s"}

#: Notebook environment factories replaced by --env refactored.
ENV_FACTORIES = (
    "make_ruscuav_env_m4", "make_ruscuav_env_m4_mode", "make_ruscuav_m8_ablation_env_m4_mode",
    "make_ruscuav_geo_baseline_env_m4_mode",
    "make_ruscuav_env_m11_dynamic_mode", "make_ruscuav_env_m11c_dynamic_obs_mode", "make_m12c_real_vienna_env_mode",
    "make_geo_eval_env", "make_static_eval_env",
)


def use_refactored_env(ns) -> None:
    """Make the legacy evaluation code create ruscuav_benchmark.env environments."""
    from ruscuav_benchmark.env import legacy_factories

    for name in ENV_FACTORIES:
        setattr(ns, name, getattr(legacy_factories, name))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path) -> pd.DataFrame:
    """Read a CSV with exact float parsing.

    pandas' default parser can be off in the last bit, which would show up as
    spurious 1e-16 differences between identical values.
    """
    return pd.read_csv(path, float_precision="round_trip")


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
    """Compare a re-run with the archive, matching rows by seed.

    Integer columns (counts, steps), booleans and text must match exactly; other
    numbers must agree within atol + rtol*|archived|. A column present on only one
    side (apart from IGNORED_COLUMNS) fails the comparison.
    """
    archived = archived.set_index("seed")
    rerun = rerun.set_index("seed")
    columns = [c for c in rerun.columns if c in archived.columns and c not in IGNORED_COLUMNS]
    only_rerun = sorted(set(rerun.columns) - set(archived.columns) - IGNORED_COLUMNS)
    only_archive = sorted(set(archived.columns) - set(rerun.columns) - IGNORED_COLUMNS)
    exact_columns = {c for c in columns if archived[c].dtype.kind in "iub"}

    mismatches, max_abs_diff, inexact = [], {}, set()
    for col in only_archive + only_rerun:
        mismatches.append({
            "seed": None, "column": col,
            "archived": "present" if col in only_archive else "missing",
            "rerun": "present" if col in only_rerun else "missing",
        })
    for seed in rerun.index:
        if seed not in archived.index:
            inexact.add(seed)
            mismatches.append({"seed": int(seed), "column": "<row>", "archived": "missing", "rerun": "present"})
            continue
        for col in columns:
            a, b = archived.at[seed, col], rerun.at[seed, col]
            fa, fb = _as_float(a), _as_float(b)
            if fa is not None and fb is not None:
                if fa == fb or (math.isnan(fa) and math.isnan(fb)):
                    continue
                diff = abs(fa - fb)
                if math.isnan(diff):  # exactly one side is NaN
                    diff = math.inf
                max_abs_diff[col] = max(max_abs_diff.get(col, 0.0), diff)
                if diff > atol:
                    inexact.add(seed)
                tolerance = atol if col in exact_columns else atol + rtol * abs(fa)
                if not diff <= tolerance:
                    mismatches.append({"seed": int(seed), "column": col, "archived": a, "rerun": b})
            elif _as_text(a) != _as_text(b):
                inexact.add(seed)
                mismatches.append({"seed": int(seed), "column": col, "archived": str(a), "rerun": str(b)})

    structural = bool(only_archive or only_rerun)
    seeds = set(rerun.index)
    return {
        "episodes_compared": int(len(rerun)),
        "columns_compared": columns,
        "columns_only_in_archive": only_archive,
        "columns_only_in_rerun": only_rerun,
        "mismatches": mismatches,
        "max_abs_diff": {k: v for k, v in sorted(max_abs_diff.items()) if v > 0},
        "episodes_fully_matching": 0 if structural else int(len(seeds - {m["seed"] for m in mismatches})),
        "episodes_exact_within_atol": 0 if structural else int(len(seeds - inexact)),
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
    archived = read_csv(archive / PLANNER_ARCHIVE.format(name=name))
    return archived, rerun, {"rerun_csv": str(csv_path), "config": getattr(ns, cfg_name)}


def run_real_planner(ns, name: str, n: int, seed0: int, archive: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    city, archived_csv = REAL_PLANNERS[name]
    (vienna, vienna_src), (warsaw, warsaw_src) = ns.get_real_worlds()  # the archived run's code path
    world, source = (vienna, vienna_src) if city == "vienna" else (warsaw, warsaw_src)
    cfg = ns.M10RecedingHorizonPlannerConfig(**{**ns.REPORTED_M10_FULL_CONFIG, "planner_agl_m": 30.0})
    dirs = ns.setup_ruscuav_drive_project(drive_root=str(ns.RUSCUAV_VIRTUAL_DRIVE), experiment_name="legacy_parity")
    regime = ns.PRIMARY_NO_TERMINAL
    rerun, _, csv_path, _ = ns.evaluate_m10_receding_planner_drive(
        world=world, drive_dirs=dirs, cfg=cfg, risk_profile="balanced", n_episodes=n, seed0=seed0,
        stage=regime.stage, mode=regime.mode, n_obstacles=0,
        prefix=f"parity_{name}_n{n}_seed{seed0}_{int(time.time())}", save_paths=True,
    )
    archived = read_csv(archive / archived_csv)
    return archived, rerun, {"rerun_csv": str(csv_path), "world_source": str(source), "world_aoi_id": world.aoi_id}


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
    archived = read_csv(run_dir / "episodes.csv")
    return archived, rerun, {"rerun_csv": str(csv_path), "model_path": str(model_path), "model_sha256": actual_sha}


def reward_factor_paths(archive: Path, variant: str, seed: int) -> tuple[Path, Path]:
    exp = archive / REWARD_FACTOR_ARCHIVE.format(variant=variant, seed=seed)
    prefix = REWARD_FACTOR_PREFIX.format(variant=variant, seed=seed)
    return exp / "models" / f"ppo_ruscuav_m4_after_stage_M8_{variant}_C.zip", exp / "eval" / prefix / f"{prefix}_episodes.csv"


def run_reward_factor(ns, world, variant: str, seed: int, n: int, archive: Path,
                      work: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Re-run the first episodes of a reward-factor evaluation with the notebook's rollout."""
    from stable_baselines3 import PPO

    model_path, archived_csv = reward_factor_paths(archive, variant, seed)
    model = PPO.load(str(model_path), device="cpu")
    rows = []
    for episode in range(n):
        row, _ = ns.rollout_m8_ablation_eval_episode(
            model=model, world=world, variant_name=variant, seed=5000 + episode, stage="C", mode="strict_eval",
            n_obstacles=0, deterministic=True, max_steps=1000,
        )
        row["episode"] = episode
        rows.append(row)
    csv_path = work / "reward_factor" / f"{variant}_trainseed{seed}_{int(time.time())}_episodes.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return read_csv(archived_csv), read_csv(csv_path), {"rerun_csv": str(csv_path), "model_path": str(model_path)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path, help="Extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--work", required=True, type=Path, help="Writable work directory")
    p.add_argument("--method", required=True,
                   help="STATIC_Astar | LOCAL_RH_Astar | FULL_RH_Astar | VIENNA_FULL_RH_Astar | "
                        "WARSAW_FULL_RH_Astar | PPO:<method>:<train seed or 'single'> | "
                        "REWARD_FACTOR:<risk_only|green_only>:<train seed>")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed0", type=int, default=None,
                   help="first scenario seed for planners (default 5000; 7000 for real-GIS worlds)")
    p.add_argument("--real-world", action="append", default=[], metavar="CITY=NPZ",
                   help="processed real-GIS world for VIENNA_/WARSAW_ methods, e.g. vienna=/data/vienna.npz")
    p.add_argument("--atol", type=float, default=1e-9, help="absolute tolerance for numeric values")
    p.add_argument("--rtol", type=float, default=1e-5,
                   help="relative tolerance for non-integer numbers; absorbs float32 differences between "
                        "PyTorch builds. Integer columns, booleans and text always have to match exactly.")
    p.add_argument("--rerun-csv", type=Path, default=None,
                   help="compare an existing re-run CSV instead of running episodes again")
    p.add_argument("--env", choices=("legacy", "refactored"), default="legacy",
                   help="environment the episodes run in: the notebook classes or ruscuav_benchmark.env")
    args = p.parse_args()

    archive, work = args.archive.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.rerun_csv is not None:
        rerun = read_csv(args.rerun_csv)
        if args.method in PLANNERS:
            archived = read_csv(archive / PLANNER_ARCHIVE.format(name=args.method))
        elif args.method in REAL_PLANNERS:
            archived = read_csv(archive / REAL_PLANNERS[args.method][1])
        elif args.method.startswith("REWARD_FACTOR:"):
            _, variant, seed = args.method.split(":", 2)
            archived = read_csv(reward_factor_paths(archive, variant, int(seed))[1])
        else:
            _, method, seed = args.method.split(":", 2)
            run = "single_checkpoint" if seed == "single" else f"trainseed_{int(seed)}"
            archived = read_csv(archive / PPO_ARCHIVE.format(method=method, run=run) / "episodes.csv")
        info = {"rerun_csv": str(args.rerun_csv)}
    else:
        extra = {}
        for item in args.real_world:
            city, _, path = item.partition("=")
            if city not in REAL_WORLD_DRIVE_NAMES or not path:
                p.error(f"--real-world expects vienna=PATH or warsaw=PATH, got {item!r}")
            extra[path] = REAL_WORLD_DRIVE_NAMES[city]
        ns = load_legacy_namespace(archive, work, extra_inputs=extra)
        if args.env == "refactored":
            use_refactored_env(ns)
        world = ns.UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)
        if args.method in PLANNERS:
            seed0 = 5000 if args.seed0 is None else args.seed0
            archived, rerun, info = run_planner(ns, world, args.method, args.episodes, seed0, archive)
        elif args.method in REAL_PLANNERS:
            seed0 = 7000 if args.seed0 is None else args.seed0
            archived, rerun, info = run_real_planner(ns, args.method, args.episodes, seed0, archive)
        elif args.method.startswith("PPO:"):
            _, method, seed = args.method.split(":", 2)
            run = "single_checkpoint" if seed == "single" else f"trainseed_{int(seed)}"
            archived, rerun, info = run_ppo(ns, world, method, run, args.episodes, archive, work)
        elif args.method.startswith("REWARD_FACTOR:"):
            _, variant, seed = args.method.split(":", 2)
            if variant not in ("risk_only", "green_only"):
                p.error(f"REWARD_FACTOR variant must be risk_only or green_only, got {variant!r}")
            archived, rerun, info = run_reward_factor(ns, world, variant, int(seed), args.episodes, archive, work)
        else:
            p.error(f"Unknown method {args.method!r}")

    env = None if args.rerun_csv is not None else args.env
    report = {"method": args.method, "env": env, "episodes": int(len(rerun)), "atol": args.atol,
              "rtol": args.rtol, **info, **compare_episodes(archived, rerun, args.atol, args.rtol),
              "elapsed_s": round(time.time() - t0, 1)}
    report_path = work / f"parity_{args.method.replace(':', '_').replace('+', 'p')}_{env or 'rerun_csv'}.json"
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
