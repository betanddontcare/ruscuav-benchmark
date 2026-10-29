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
    python scripts/legacy_parity.py --archive /data/RUSCUAV-v2.0.0 --work runs/parity \
        --method POLICY:dynamic/DYN-OBS-RL/1001 --episodes 200 --code package

With ``--env refactored`` the legacy evaluation code drives the refactored
environment (ruscuav_benchmark.env) instead of the notebook classes: the
notebook's environment factories are replaced by those of
ruscuav_benchmark.env.legacy_factories before the episodes run.

``--method POLICY:<name>`` re-runs one of the archived primary policy evaluations
of ``ruscuav_benchmark.policies.ARCHIVED_EVALUATIONS`` (static, dynamic and Vienna),
e.g. ``POLICY:dynamic/DYN-OBS-RL/1001``, with the notebook's evaluator for it.

With ``--code package`` the planner methods fly the planners of
ruscuav_benchmark.planners, and the policy evaluations run with
ruscuav_benchmark.policies, in ruscuav_benchmark.env, instead of the notebook's
code. The dynamic planner's rows then get the post-hoc columns of the archived
evaluation (derived metrics, mission acceptability recomputed) from the notebook's
own functions.

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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ruscuav_benchmark.legacy.runtime import DRIVE_LITERAL, load_legacy_namespace  # noqa: E402
from ruscuav_benchmark.world import UrbanWorldLayers  # noqa: E402

PLANNERS = {
    # name: (legacy evaluator, reported-config global, config class)
    "STATIC_Astar": ("evaluate_m9_static_planner_drive", "REPORTED_M9_CONFIG", "M9PlannerConfig"),
    "LOCAL_RH_Astar": ("evaluate_m10_receding_planner_drive", "REPORTED_M10_LOCAL_CONFIG",
                       "M10RecedingHorizonPlannerConfig"),
    "FULL_RH_Astar": ("evaluate_m10_receding_planner_drive", "REPORTED_M10_FULL_CONFIG",
                      "M10RecedingHorizonPlannerConfig"),
    "DYN_RH_Astar": ("evaluate_dynamic_planner_mode", "REPORTED_M11_PLANNER_CONFIG", "M10RecedingHorizonPlannerConfig"),
}
PLANNER_ARCHIVE = ("experiments/paperA_primary_no_terminal_static_planners/eval/"
                   "primary_no_terminal_{name}_n200_seed5000/primary_no_terminal_{name}_n200_seed5000_episodes.csv")
DYNAMIC_PLANNER_ARCHIVE = (
    "experiments/paperA_primary_no_terminal_dynamic/eval/primary_no_terminal_dynamic_receding_Astar/"
    "primary_no_terminal_dynamic_receding_Astar_episodes.csv"
)


def planner_archive(name: str) -> str:
    return DYNAMIC_PLANNER_ARCHIVE if name == "DYN_RH_Astar" else PLANNER_ARCHIVE.format(name=name)

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
#: Drive names under which --real-world files are given to the notebook code. The
#: archived runs used the footway-only worlds (sidewalk = footway - road - building),
#: which the notebook's get_real_worlds() looks for first. The Vienna name is the
#: notebook's; the Warsaw footway-only world was built in memory and never saved, so
#: its name only has to match the notebook's search pattern.
REAL_WORLD_DRIVE_NAMES = {
    "vienna": "worlds/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz",
    "warsaw": "worlds/warsaw_centrum_osm_1km_v1_footway_only_urban_world_layers.npz",
}
#: First global name get_real_worlds() checks for each city.
REAL_WORLD_GLOBALS = {"vienna": "vienna_world_footway_only", "warsaw": "real_world_footway_only"}
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
    prefix = f"parity_{name}_n{n}_seed{seed0}_{int(time.time())}"
    if name == "DYN_RH_Astar":
        _, _, csv_path, _ = ns.evaluate_dynamic_planner_mode(
            world=world, drive_dirs=dirs, cfg=cfg, dynamic_config=ns.reported_dynamic_config(), regime=regime,
            n_episodes=n, seed0=seed0, prefix=prefix, dynamic_prior_mode="zero", save_paths=True,
        )
        rerun = read_csv(csv_path)
    else:
        kwargs = dict(
            world=world, drive_dirs=dirs, risk_profile="balanced", n_episodes=n, seed0=seed0,
            stage=regime.stage, mode=regime.mode, n_obstacles=0, prefix=prefix, save_paths=True,
        )
        kwargs["planner_config" if evaluator.startswith("evaluate_m9") else "cfg"] = cfg
        rerun, _, csv_path, _ = getattr(ns, evaluator)(**kwargs)
    archived = read_csv(archive / planner_archive(name))
    return archived, rerun, {"rerun_csv": str(csv_path), "config": getattr(ns, cfg_name)}


def run_package_planner(ns, world_path, name: str, n: int, seed0: int, work: Path,
                        **config_changes) -> tuple[pd.DataFrame, dict]:
    """Episodes of ruscuav_benchmark.planners in ruscuav_benchmark.env, in the archived
    evaluation's format."""
    from ruscuav_benchmark.planners import evaluate_planner

    world = UrbanWorldLayers.load_npz(world_path)
    out_dir = work / "package_planners" / f"{name}_n{n}_seed{seed0}_{int(time.time())}"
    result = evaluate_planner(world, name, seeds=range(seed0, seed0 + n), output_dir=out_dir, save_paths=True,
                              **config_changes)
    csv_path = result.csv_path
    if name == "DYN_RH_Astar":
        # The archived primary dynamic evaluation (evaluate_dynamic_planner_mode) adds the
        # regime and the flown path length, then the post-hoc columns.
        rows = []
        for index, row in enumerate(result.rows):
            row = dict(row)
            # the archived rows have no building time (the package's add it, docs/PLANNERS.md), so the archive's
            # derived metrics and acceptability were computed without it: compare with them as they were
            row.pop("building_time", None)
            row["evaluation_regime"] = ns.PRIMARY_NO_TERMINAL.name
            path = result.paths[index]
            row["path_length_xy"] = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
            rows.append(row)
        df = ns.add_derived_metrics(pd.DataFrame(rows).sort_values("episode"))
        df["mission_acceptable"] = ns.recompute_dynamic_acceptability(df)
        csv_path = out_dir / f"{name}_posthoc_episodes.csv"
        df.to_csv(csv_path, index=False)
    return read_csv(csv_path), {"rerun_csv": str(csv_path), "planner": result.planner.to_dict(),
                                "world_layers_sha256": world.layers_sha256()}


def run_real_planner(ns, name: str, n: int, seed0: int, archive: Path, given_cities: set,
                     package_work: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    city, archived_csv = REAL_PLANNERS[name]
    # get_real_worlds() loads both cities; a city without --real-world gets a placeholder,
    # so that a Vienna run needs no Warsaw world and vice versa.
    for other, global_name in REAL_WORLD_GLOBALS.items():
        if other != city and other not in given_cities:
            setattr(ns, global_name, None)
    (vienna, vienna_src), (warsaw, warsaw_src) = ns.get_real_worlds()  # the archived run's code path
    world, source = (vienna, vienna_src) if city == "vienna" else (warsaw, warsaw_src)
    archived = read_csv(archive / archived_csv)
    if package_work is not None:
        rerun, info = run_package_planner(ns, source, "FULL_RH_Astar", n, seed0, package_work, planner_agl_m=30.0)
        info.update({"world_source": str(source), "world_sha256": sha256_file(Path(str(source)))})
        return archived, rerun, info
    cfg = ns.M10RecedingHorizonPlannerConfig(**{**ns.REPORTED_M10_FULL_CONFIG, "planner_agl_m": 30.0})
    dirs = ns.setup_ruscuav_drive_project(drive_root=str(ns.RUSCUAV_VIRTUAL_DRIVE), experiment_name="legacy_parity")
    regime = ns.PRIMARY_NO_TERMINAL
    rerun, _, csv_path, _ = ns.evaluate_m10_receding_planner_drive(
        world=world, drive_dirs=dirs, cfg=cfg, risk_profile="balanced", n_episodes=n, seed0=seed0,
        stage=regime.stage, mode=regime.mode, n_obstacles=0,
        prefix=f"parity_{name}_n{n}_seed{seed0}_{int(time.time())}", save_paths=True,
    )
    info = {"rerun_csv": str(csv_path), "world_source": str(source), "world_aoi_id": world.aoi_id}
    if Path(str(source)).is_file():
        info["world_sha256"] = sha256_file(Path(str(source)))
        info["world_layers_sha256"] = UrbanWorldLayers.load_npz(source).layers_sha256()
    return archived, rerun, info


def run_ppo(ns, world, method: str, run: str, n: int, archive: Path,
            work: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
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
    model = exp / "models" / f"ppo_ruscuav_m4_after_stage_M8_{variant}_C.zip"
    return model, exp / "eval" / prefix / f"{prefix}_episodes.csv"


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


def run_policy(ns, name: str, n: int, seed0, archive: Path, work: Path, package: bool,
               real_worlds: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """An archived policy evaluation, re-run with the notebook's evaluator or the package."""
    from ruscuav_benchmark.policies import ARCHIVED_EVALUATIONS, archived_checkpoint, evaluate_archived

    entry = ARCHIVED_EVALUATIONS[name]
    seed0 = entry.seed0 if seed0 is None else int(seed0)
    world_path = Path(str(ns.CANONICAL_WORLD_PATH)) if entry.world == "canonical" else Path(real_worlds[entry.world])
    model_path = archived_checkpoint(entry.checkpoint, archive)
    info = {"checkpoint": entry.checkpoint, "model_path": str(model_path), "model_sha256": sha256_file(model_path),
            "world_source": str(world_path),
            "world_layers_sha256": UrbanWorldLayers.load_npz(world_path).layers_sha256()}
    archived = read_csv(archive / entry.episodes_csv)
    tag = f"{name.replace('/', '_')}_n{n}_seed{seed0}_{int(time.time())}"
    if package:
        result = evaluate_archived(name, archive, world=UrbanWorldLayers.load_npz(world_path), n_episodes=n,
                                   seed0=seed0, output_dir=work / "package_policies" / tag, save_paths=True)
        return archived, read_csv(result.csv_path), {**info, "rerun_csv": str(result.csv_path)}
    world = ns.UrbanWorldLayers.load_npz(world_path)
    regime = ns.PRIMARY_NO_TERMINAL
    if entry.row_format == "posthoc":
        _, _, csv_path, _ = ns.evaluate_ppo_resume(
            model_path=model_path,
            env_factory=lambda seed: ns.make_static_eval_env(entry.method, world, regime, seed),
            method=entry.label, regime=regime, output_dir=work / "ppo" / tag, n_episodes=n, seed0=seed0,
            deterministic=True, save_paths=True,
        )
    else:
        dirs = ns.setup_ruscuav_drive_project(drive_root=str(ns.RUSCUAV_VIRTUAL_DRIVE), experiment_name="legacy_parity")
        if entry.row_format == "dynamic" and entry.method == "DYN-OBS-RL":
            _, _, csv_path, _ = ns.evaluate_m11c_dynamic_obs_rl_drive(
                model_path=model_path, world=world, drive_dirs=dirs, dynamic_config=ns.reported_dynamic_config(),
                n_episodes=n, seed0=seed0, prefix=f"parity_{tag}", stage=regime.stage, mode=regime.mode,
                save_paths=False)
        elif entry.row_format == "dynamic":
            _, _, csv_path, _ = ns.evaluate_m11_dynamic_rl_drive(
                model_path=model_path, world=world, drive_dirs=dirs, dynamic_config=ns.reported_dynamic_config(),
                n_episodes=n, seed0=seed0, prefix=f"parity_{tag}", stage=regime.stage, mode=regime.mode,
                reward_only_obs=True, save_paths=False)
        else:
            _, _, csv_path, _ = ns.evaluate_m12c_real_policy_drive(
                model_path=model_path, world=world, policy_kind=entry.label, drive_dirs=dirs, n_episodes=n,
                seed0=seed0, prefix=f"parity_{tag}", stage=regime.stage, mode=regime.mode, deterministic=True,
                save_paths=True)
    return archived, read_csv(csv_path), {**info, "rerun_csv": str(csv_path)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path, help="Extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--work", required=True, type=Path, help="Writable work directory")
    p.add_argument("--method", required=True,
                   help="STATIC_Astar | LOCAL_RH_Astar | FULL_RH_Astar | DYN_RH_Astar | VIENNA_FULL_RH_Astar | "
                        "WARSAW_FULL_RH_Astar | PPO:<method>:<train seed or 'single'> | "
                        "REWARD_FACTOR:<risk_only|green_only>:<train seed> | POLICY:<archived evaluation, e.g. "
                        "static/GEO-RL/1001, dynamic/DYN-OBS-RL/1001, vienna/RISK+GREEN-RL/short_finetuned>")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed0", type=int, default=None,
                   help="first scenario seed for planners and POLICY evaluations (default: that of the archived "
                        "run, 5000 or 7000 for real-GIS worlds)")
    p.add_argument("--real-world", action="append", default=[], metavar="CITY=NPZ",
                   help="processed footway-only real-GIS world for VIENNA_/WARSAW_ methods, e.g. "
                        "vienna=/data/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz")
    p.add_argument("--atol", type=float, default=1e-9, help="absolute tolerance for numeric values")
    p.add_argument("--rtol", type=float, default=1e-5,
                   help="relative tolerance for non-integer numbers; absorbs float32 differences between "
                        "PyTorch builds. Integer columns, booleans and text always have to match exactly.")
    p.add_argument("--rerun-csv", type=Path, default=None,
                   help="compare an existing re-run CSV instead of running episodes again")
    p.add_argument("--env", choices=("legacy", "refactored"), default="legacy",
                   help="environment the episodes run in: the notebook classes or ruscuav_benchmark.env")
    p.add_argument("--code", "--planners", dest="code", choices=("notebook", "package"), default="notebook",
                   help="planner methods and POLICY evaluations: the notebook's code, or ruscuav_benchmark.planners "
                        "and ruscuav_benchmark.policies (which always run in ruscuav_benchmark.env)")
    args = p.parse_args()

    archive, work = args.archive.resolve(), args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.rerun_csv is not None:
        rerun = read_csv(args.rerun_csv)
        if args.method in PLANNERS:
            archived = read_csv(archive / planner_archive(args.method))
        elif args.method in REAL_PLANNERS:
            archived = read_csv(archive / REAL_PLANNERS[args.method][1])
        elif args.method.startswith("REWARD_FACTOR:"):
            _, variant, seed = args.method.split(":", 2)
            archived = read_csv(reward_factor_paths(archive, variant, int(seed))[1])
        elif args.method.startswith("POLICY:"):
            from ruscuav_benchmark.policies import ARCHIVED_EVALUATIONS

            name = args.method.split(":", 1)[1]
            if name not in ARCHIVED_EVALUATIONS:
                p.error(f"unknown archived policy evaluation {name!r}; available: {', '.join(ARCHIVED_EVALUATIONS)}")
            archived = read_csv(archive / ARCHIVED_EVALUATIONS[name].episodes_csv)
        else:
            _, method, seed = args.method.split(":", 2)
            run = "single_checkpoint" if seed == "single" else f"trainseed_{int(seed)}"
            archived = read_csv(archive / PPO_ARCHIVE.format(method=method, run=run) / "episodes.csv")
        info = {"rerun_csv": str(args.rerun_csv)}
    else:
        extra, given_cities = {}, set()
        for item in args.real_world:
            city, _, path = item.partition("=")
            if city not in REAL_WORLD_DRIVE_NAMES or not path:
                p.error(f"--real-world expects vienna=PATH or warsaw=PATH, got {item!r}")
            extra[path] = REAL_WORLD_DRIVE_NAMES[city]
            given_cities.add(city)
        if args.method in REAL_PLANNERS and REAL_PLANNERS[args.method][0] not in given_cities:
            p.error(f"{args.method} needs --real-world {REAL_PLANNERS[args.method][0]}=PATH")
        package = args.code == "package"
        policy_name = args.method.split(":", 1)[1] if args.method.startswith("POLICY:") else None
        if policy_name is not None:
            from ruscuav_benchmark.policies import ARCHIVED_EVALUATIONS

            if policy_name not in ARCHIVED_EVALUATIONS:
                p.error(f"unknown archived policy evaluation {policy_name!r}; available: "
                        f"{', '.join(ARCHIVED_EVALUATIONS)}")
            city = ARCHIVED_EVALUATIONS[policy_name].world
            if city != "canonical" and city not in given_cities:
                p.error(f"{args.method} needs --real-world {city}=PATH")
        if package and not (args.method in PLANNERS or args.method in REAL_PLANNERS or policy_name is not None):
            p.error("--code package applies to the planner methods and POLICY evaluations only")
        ns = load_legacy_namespace(archive, work, extra_inputs=extra)
        if args.env == "refactored":
            use_refactored_env(ns)
        world = ns.UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)
        if args.method in PLANNERS:
            seed0 = 5000 if args.seed0 is None else args.seed0
            if package:
                archived = read_csv(archive / planner_archive(args.method))
                rerun, info = run_package_planner(ns, ns.CANONICAL_WORLD_PATH, args.method, args.episodes, seed0, work)
            else:
                archived, rerun, info = run_planner(ns, world, args.method, args.episodes, seed0, archive)
        elif args.method in REAL_PLANNERS:
            seed0 = 7000 if args.seed0 is None else args.seed0
            archived, rerun, info = run_real_planner(ns, args.method, args.episodes, seed0, archive, given_cities,
                                                     package_work=work if package else None)
        elif args.method.startswith("PPO:"):
            _, method, seed = args.method.split(":", 2)
            run = "single_checkpoint" if seed == "single" else f"trainseed_{int(seed)}"
            archived, rerun, info = run_ppo(ns, world, method, run, args.episodes, archive, work)
        elif policy_name is not None:
            real_worlds = {item.partition("=")[0]: item.partition("=")[2] for item in args.real_world}
            archived, rerun, info = run_policy(ns, policy_name, args.episodes, args.seed0, archive, work, package,
                                               real_worlds)
        elif args.method.startswith("REWARD_FACTOR:"):
            _, variant, seed = args.method.split(":", 2)
            if variant not in ("risk_only", "green_only"):
                p.error(f"REWARD_FACTOR variant must be risk_only or green_only, got {variant!r}")
            archived, rerun, info = run_reward_factor(ns, world, variant, int(seed), args.episodes, archive, work)
        else:
            p.error(f"Unknown method {args.method!r}")

    if args.rerun_csv is not None:
        env = None
    elif args.code == "package":
        env = "package"
    else:
        env = args.env
    report = {"method": args.method, "env": env, "episodes": int(len(rerun)), "atol": args.atol,
              "rtol": args.rtol, **info, **compare_episodes(archived, rerun, args.atol, args.rtol),
              "elapsed_s": round(time.time() - t0, 1)}
    method_tag = args.method.replace(':', '_').replace('+', 'p').replace('/', '_')
    report_path = work / f"parity_{method_tag}_{env or 'rerun_csv'}.json"
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
