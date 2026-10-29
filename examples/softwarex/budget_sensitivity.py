#!/usr/bin/env python3
"""Mission acceptability of runs under the protocol's budgets and with another pedestrian-way budget.

The last sentence of Example 1: with 10 s over pedestrian ways, the stopping budget of the secondary regime
(canonical-secondary), instead of the 5 s of the acceptability budgets.

    python examples/softwarex/budget_sensitivity.py --ped-s 10 \\
        GEO_SHORTEST=runs/geo/GEO_SHORTEST_canonical-primary_episodes.csv \\
        FULL_RH_Astar=runs/full/FULL_RH_Astar_canonical-primary_episodes.csv
"""

from __future__ import annotations

import argparse

from ruscuav_benchmark.posthoc import DEFAULT_BUDGETS
from ruscuav_benchmark.scoring import load_run, score_runs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", metavar="LABEL=CSV", help="per-episode files of the runs")
    ap.add_argument("--ped-s", type=float, default=10.0, help="pedestrian-way budget to try, in seconds (default 10)")
    args = ap.parse_args()
    runs = [load_run(path, label) for label, path in (item.split("=", 1) for item in args.runs)]
    variants = [(f"pedestrian budget {DEFAULT_BUDGETS['ped_s']:g} s (the protocol's)", DEFAULT_BUDGETS),
                (f"pedestrian budget {args.ped_s:g} s", {**DEFAULT_BUDGETS, "ped_s": args.ped_s})]
    for name, budgets in variants:
        rates = ", ".join(f"{row['label']} {100 * row['mission_acceptability_rate']:.1f}%"
                          for row in score_runs(runs, budgets=budgets).summaries)
        print(f"{name}: {rates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
