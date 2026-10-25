"""Command-line interface: ``ruscuav <command>``.

Commands:
  validate   check world NPZ files against the ruscuav-world schema
  evaluate   run the static risk-cost A* evaluation (same as ``ruscuav-evaluate``)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__


def _cmd_validate(args: argparse.Namespace) -> int:
    from .validate import validate_world_npz

    reports = [validate_world_npz(path) for path in args.paths]
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        print("\n\n".join(r.format() for r in reports))
    return 0 if all(r.ok for r in reports) else 1


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from .evaluate import run_static_astar_evaluation

    summary = run_static_astar_evaluation(
        args.world, args.scenarios, args.output_dir,
        max_episodes=args.max_episodes, profile=args.profile,
        plan_res_m=args.plan_res_m, agl_m=args.agl_m, risk_lambda=args.risk_lambda,
    )
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ruscuav", description="RUSCUAV benchmark tools.")
    parser.add_argument("--version", action="version", version=f"ruscuav {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate world NPZ files (exit code 1 if any is invalid)")
    p.add_argument("paths", nargs="+", help="world NPZ files")
    p.add_argument("--json", action="store_true", help="print machine-readable reports")
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser("evaluate", help="run the static risk-cost A* evaluation on one world")
    p.add_argument("--world", required=True, help="world NPZ file")
    p.add_argument("--scenarios", required=True, help="scenario CSV with start/target pairs")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--profile", default="balanced")
    p.add_argument("--plan-res-m", type=float, default=10.0)
    p.add_argument("--agl-m", type=float, default=40.0)
    p.add_argument("--risk-lambda", type=float, default=0.05)
    p.set_defaults(func=_cmd_evaluate)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
