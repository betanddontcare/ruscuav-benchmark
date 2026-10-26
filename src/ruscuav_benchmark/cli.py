"""Command-line interface: ``ruscuav <command>``.

Commands:
  validate   check world NPZ files against the ruscuav-world schema
  evaluate   fly a planner of the paper or a policy on a world (same as ``ruscuav-evaluate``)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from . import __version__


def _cmd_validate(args: argparse.Namespace) -> int:
    from .validate import validate_world_npz

    reports = [validate_world_npz(path) for path in args.paths]
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        print("\n\n".join(r.format() for r in reports))
    return 0 if all(r.ok for r in reports) else 1


def _config_changes(items: List[str], kind: str) -> Dict[str, Any]:
    """``--set FIELD=VALUE`` items, parsed by the type of each configuration field."""
    from dataclasses import fields

    from .planners import PlannerConfig, RecedingPlannerConfig

    config_class = PlannerConfig if kind == "static" else RecedingPlannerConfig
    types = {f.name: (f.type if isinstance(f.type, str) else f.type.__name__) for f in fields(config_class)}
    if kind == "dynamic":
        types["dynamic_prior_mode"] = "str"
    changes: Dict[str, Any] = {}
    for item in items:
        key, sep, text = (part.strip() for part in item.partition("="))
        if not sep or not key:
            raise ValueError(f"--set expects FIELD=VALUE, got {item!r}")
        if key not in types:
            raise ValueError(f"--set {key}: not a setting of this planner; settings: {', '.join(types)}")
        kind_of = types[key]
        try:
            if kind_of == "bool":
                lowered = text.lower()
                if lowered not in ("true", "false", "1", "0", "yes", "no"):
                    raise ValueError
                value: Any = lowered in ("true", "1", "yes")
            elif kind_of == "int":
                value = int(text)
            elif kind_of == "float":
                value = float(text)
            else:
                value = text
        except ValueError:
            expected = "true or false" if kind_of == "bool" else f"a {kind_of}"
            raise ValueError(f"--set {key}={text!r}: expected {expected}") from None
        changes[key] = value
    return changes


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .scenarios import missions_from_csv
    from .world import UrbanWorldLayers

    sources = [name for name, value in (("--policy", args.policy), ("--checkpoint", args.checkpoint),
                                        ("--evaluation", args.evaluation)) if value is not None]
    policy_mode = bool(sources)
    try:
        if len(sources) > 1:
            raise ValueError(f"give one of --policy, --checkpoint and --evaluation, not {' and '.join(sources)}")
        if policy_mode and args.planner is not None:
            raise ValueError(f"give --planner or {sources[0]}, not both")
        if policy_mode and args.set:
            raise ValueError("--set changes planner settings; it does not apply to policies")
        if not policy_mode and (args.method or args.format or args.label or args.stochastic or args.archive):
            raise ValueError("--method, --format, --label, --stochastic and --archive apply to "
                             "--policy, --checkpoint and --evaluation")
        if args.evaluation is not None:
            name, result = _evaluate_archived(args)
        else:
            if args.world is None:
                raise ValueError("--world is required")
            if args.seed0 is None:
                args.seed0 = 5000
            world = UrbanWorldLayers.load_npz(args.world)
            missions = None
            if args.scenarios is not None:
                missions = missions_from_csv(args.scenarios, world, seed0=args.seed0)
                if not missions:
                    print(f"{args.scenarios} has no missions for world {world.aoi_id}", file=sys.stderr)
                    return 2
            if policy_mode:
                name, result = _evaluate_policy(args, world, missions)
            else:
                name, result = _evaluate_planner(args, world, missions)
    except (ValueError, TypeError, OSError, ModuleNotFoundError) as exc:
        print(f"ruscuav evaluate: {exc}", file=sys.stderr)
        return 2
    s = result.summary
    keys = ["n_episodes", "success_rate", "mission_acceptability_rate", "acceptable_given_success_rate",
            "mean_steps", "termination_counts"]
    regime = s["regime"]["name"] if isinstance(s.get("regime"), dict) else s.get("regime")
    print(json.dumps({("policy" if policy_mode else "planner"): name, "world_id": s["world_id"], "regime": regime,
                      **{k: s[k] for k in keys if k in s},
                      "csv_path": None if result.csv_path is None else str(result.csv_path)}, indent=2))
    return 0


def _progress(args: argparse.Namespace, name: str):
    def progress(index: int, total: int, row: dict) -> None:
        if not args.quiet and ((index + 1) % 10 == 0 or index == 0 or index + 1 == total):
            print(f"[{name}] episode {index + 1}/{total}: seed {row['seed']}, {row['termination']}, "
                  f"{row['steps']} steps", file=sys.stderr, flush=True)

    return progress


def _evaluate_planner(args: argparse.Namespace, world, missions):
    from .planners import evaluate_planner, get_planner

    spec = get_planner(args.planner or "FULL_RH_Astar")
    spec = spec.with_config(**_config_changes(args.set, spec.kind))
    result = evaluate_planner(
        world, spec, regime=args.regime, missions=missions, n_episodes=args.episodes, seed0=args.seed0,
        risk_profile=args.profile, output_dir=args.output_dir, prefix=args.prefix, save_paths=args.save_paths,
        progress=_progress(args, spec.name),
    )
    return spec.name, result


def _evaluate_policy(args: argparse.Namespace, world, missions):
    from .env import METHOD_PRESETS
    from .policies import ARCHIVED_CHECKPOINTS, archived_checkpoint, checkpoint_env_overrides, evaluate_policy

    overrides = None
    if args.checkpoint is not None:
        if args.archive is None:
            raise ValueError("--checkpoint needs --archive (the extracted v2.0.0 archive)")
        if args.checkpoint not in ARCHIVED_CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {args.checkpoint!r}; available: {', '.join(sorted(ARCHIVED_CHECKPOINTS))}")
        policy = archived_checkpoint(args.checkpoint, args.archive)
        method = args.method or ARCHIVED_CHECKPOINTS[args.checkpoint].method
        overrides = checkpoint_env_overrides(args.checkpoint)
        name = args.checkpoint
    else:
        if args.archive is not None:
            raise ValueError("--archive applies to --checkpoint and --evaluation")
        if args.method is None:
            raise ValueError("--policy needs --method, the method preset whose observation the policy was trained with")
        policy, method, name = args.policy, args.method, args.policy
    if method not in METHOD_PRESETS:
        raise ValueError(f"unknown method {method!r}; available: {', '.join(METHOD_PRESETS)}")
    result = evaluate_policy(
        world, policy, method, regime=args.regime, missions=missions, n_episodes=args.episodes, seed0=args.seed0,
        row_format=args.format, label=args.label, deterministic=not args.stochastic, risk_profile=args.profile,
        env_overrides=overrides, output_dir=args.output_dir, prefix=args.prefix, save_paths=args.save_paths,
        progress=_progress(args, name),
    )
    return name, result


def _evaluate_archived(args: argparse.Namespace):
    from .policies import ARCHIVED_EVALUATIONS, evaluate_archived
    from .world import UrbanWorldLayers

    if args.archive is None:
        raise ValueError("--evaluation needs --archive (the extracted v2.0.0 archive)")
    if args.evaluation not in ARCHIVED_EVALUATIONS:
        raise ValueError(f"unknown archived evaluation {args.evaluation!r}; available: "
                         f"{', '.join(ARCHIVED_EVALUATIONS)}")
    given = [flag for flag, value in (("--method", args.method), ("--format", args.format), ("--label", args.label),
                                      ("--scenarios", args.scenarios)) if value is not None]
    given += ["--stochastic"] if args.stochastic else []
    given += ["--regime"] if args.regime != "primary_no_terminal" else []
    given += ["--profile"] if args.profile != "balanced" else []
    if given:
        raise ValueError(f"{', '.join(given)}: the archived evaluation fixes these settings")
    world = None if args.world is None else UrbanWorldLayers.load_npz(args.world)
    kwargs = {"seed0": args.seed0} if args.seed0 is not None else {}
    if args.episodes is not None:
        kwargs["n_episodes"] = args.episodes
    result = evaluate_archived(args.evaluation, args.archive, world=world, output_dir=args.output_dir,
                               prefix=args.prefix, save_paths=args.save_paths,
                               progress=_progress(args, args.evaluation), **kwargs)
    return args.evaluation, result


def build_parser() -> argparse.ArgumentParser:
    from .planners import PLANNER_PRESETS

    parser = argparse.ArgumentParser(prog="ruscuav", description="RUSCUAV benchmark tools.")
    parser.add_argument("--version", action="version", version=f"ruscuav {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate world NPZ files (exit code 1 if any is invalid)")
    p.add_argument("paths", nargs="+", help="world NPZ files")
    p.add_argument("--json", action="store_true", help="print machine-readable reports")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser(
        "evaluate", help="fly a planner of the paper or a policy on a world (needs the 'env' extra)",
        description="Fly a planner of the paper, or a PPO policy, on a world, one episode per mission, and write "
                    "the per-episode CSV and the summary. Missions come from --scenarios or are sampled by the "
                    "environment from seeds --seed0, --seed0+1, ... Policies need the 'rl' extra.",
    )
    p.add_argument("--world", default=None,
                   help="world NPZ file (for --evaluation: only for the Vienna evaluations, whose world is not in the "
                        "archive)")
    p.add_argument("--planner", default=None, choices=list(PLANNER_PRESETS),
                   help="planner (default FULL_RH_Astar unless a policy is given)")
    p.add_argument("--policy", default=None, help="Stable-Baselines3 PPO checkpoint (.zip) to fly; needs --method")
    p.add_argument("--checkpoint", default=None,
                   help="archived checkpoint to fly, e.g. RISK+GREEN-RL/4001 (see docs/POLICIES.md); needs --archive")
    p.add_argument("--evaluation", default=None,
                   help="re-run an archived policy evaluation, e.g. static/RISK+GREEN-RL/4001 or "
                        "dynamic/DYN-OBS-RL/1001 (see docs/POLICIES.md); needs --archive")
    p.add_argument("--archive", default=None, help="extracted RUSCUAV v2.0.0 archive, for --checkpoint and --evaluation")
    p.add_argument("--method", default=None,
                   help="method preset of the policy's environment, e.g. RISK+GREEN-RL (default for --checkpoint: "
                        "the preset it was trained in)")
    p.add_argument("--format", default=None, choices=["posthoc", "dynamic", "transfer"],
                   help="row format of a policy evaluation (default posthoc; dynamic for DYN-* presets)")
    p.add_argument("--label", default=None, help="method label written in the rows of a policy evaluation")
    p.add_argument("--stochastic", action="store_true", help="sample the policy's actions (default deterministic)")
    p.add_argument("--scenarios", default=None,
                   help="scenario CSV with start/target columns and, optionally, seed and world_id (rows of "
                        "this world are flown; rows sharing a seed share the dynamic risk field)")
    p.add_argument("--episodes", type=int, default=None,
                   help="number of episodes (default: every scenario row, or 200 seeds)")
    p.add_argument("--seed0", type=int, default=None,
                   help="first seed when no scenario file is given, or for rows without a seed (default 5000; for "
                        "--evaluation that of the archived run)")
    p.add_argument("--regime", default="primary_no_terminal",
                   choices=["primary_no_terminal", "secondary_operational_stopping"],
                   help="evaluation regime (default primary_no_terminal)")
    p.add_argument("--profile", default="balanced", help="risk profile (default balanced)")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="change a planner setting, e.g. --set planner_agl_m=30 (repeatable; see docs/PLANNERS.md)")
    p.add_argument("--output-dir", required=True, help="directory for <prefix>_episodes.csv, _summary.json, ...")
    p.add_argument("--prefix", default=None,
                   help="file name prefix (default <planner or method>_<world>_<regime>; for planners plus a "
                        "configuration checksum when settings differ from the preset)")
    p.add_argument("--save-paths", action="store_true", help="also save the flown (and planned) paths")
    p.add_argument("--quiet", action="store_true", help="no progress messages")
    p.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
