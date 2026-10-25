"""Command-line interface: ``ruscuav <command>``.

Commands:
  validate   check world NPZ files against the ruscuav-world schema
  evaluate   fly one of the paper's planners on a world (same as ``ruscuav-evaluate``)
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
    from .planners import evaluate_planner, get_planner, missions_from_csv
    from .world import UrbanWorldLayers

    try:
        world = UrbanWorldLayers.load_npz(args.world)
        spec = get_planner(args.planner)
        spec = spec.with_config(**_config_changes(args.set, spec.kind))
        missions = None
        if args.scenarios is not None:
            missions = missions_from_csv(args.scenarios, world, seed0=args.seed0)
            if not missions:
                print(f"{args.scenarios} has no missions for world {world.aoi_id}", file=sys.stderr)
                return 2

        def progress(index: int, total: int, row: dict) -> None:
            if not args.quiet and ((index + 1) % 10 == 0 or index == 0 or index + 1 == total):
                print(f"[{spec.name}] episode {index + 1}/{total}: seed {row['seed']}, {row['termination']}, "
                      f"{row['steps']} steps", file=sys.stderr, flush=True)

        result = evaluate_planner(
            world, spec, regime=args.regime, missions=missions, n_episodes=args.episodes, seed0=args.seed0,
            risk_profile=args.profile, output_dir=args.output_dir, prefix=args.prefix, save_paths=args.save_paths,
            progress=progress,
        )
    except (ValueError, TypeError, OSError, ModuleNotFoundError) as exc:
        print(f"ruscuav evaluate: {exc}", file=sys.stderr)
        return 2
    s = result.summary
    keys = ["n_episodes", "success_rate", "mission_acceptability_rate", "acceptable_given_success_rate",
            "mean_steps", "termination_counts"]
    print(json.dumps({"planner": spec.name, "world_id": s["world_id"], "regime": s["regime"],
                      **{k: s[k] for k in keys if k in s}, "csv_path": s.get("csv_path")}, indent=2))
    return 0


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
        "evaluate", help="fly a planner of the paper on a world (needs the 'env' extra)",
        description="Fly a planner of the paper on a world, one episode per mission, and write the per-episode "
                    "CSV and the summary. Missions come from --scenarios or are sampled by the environment "
                    "from seeds --seed0, --seed0+1, ...",
    )
    p.add_argument("--world", required=True, help="world NPZ file")
    p.add_argument("--planner", default="FULL_RH_Astar", choices=list(PLANNER_PRESETS),
                   help="planner (default FULL_RH_Astar)")
    p.add_argument("--scenarios", default=None,
                   help="scenario CSV with start/target columns and, optionally, seed and world_id (rows of "
                        "this world are flown; rows sharing a seed share the dynamic risk field)")
    p.add_argument("--episodes", type=int, default=None,
                   help="number of episodes (default: every scenario row, or 200 seeds)")
    p.add_argument("--seed0", type=int, default=5000,
                   help="first seed when no scenario file is given, or for rows without a seed (default 5000)")
    p.add_argument("--regime", default="primary_no_terminal",
                   choices=["primary_no_terminal", "secondary_operational_stopping"],
                   help="evaluation regime (default primary_no_terminal)")
    p.add_argument("--profile", default="balanced", help="risk profile (default balanced)")
    p.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                   help="change a planner setting, e.g. --set planner_agl_m=30 (repeatable; see docs/PLANNERS.md)")
    p.add_argument("--output-dir", required=True, help="directory for <prefix>_episodes.csv, _summary.json, ...")
    p.add_argument("--prefix", default=None,
                   help="file name prefix (default <planner>_<world>_<regime>, plus a configuration checksum "
                        "when settings differ from the preset)")
    p.add_argument("--save-paths", action="store_true", help="also save the flown (and planned) paths")
    p.add_argument("--quiet", action="store_true", help="no progress messages")
    p.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
