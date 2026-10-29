#!/usr/bin/env python3
"""The data of Fig. 3: the missions of each run by outcome.

Each mission falls in one part: acceptable; successful, pedestrian budget exceeded; successful, other budget
exceeded (no pedestrian excess); not successful. Success is the sum of the first three parts, acceptability the
first. Budgets are the acceptability budgets of the protocols used (the package defaults).

    python examples/softwarex/fig3_data.py \\
        synthetic:FULL_RH_Astar=runs/full/FULL_RH_Astar_canonical-primary_episodes.csv ... \\
        city:FULL_RH_Astar=runs/city/FULL_RH_Astar_30m_episodes.csv ... > fig3_data.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter

from ruscuav_benchmark.posthoc import DEFAULT_BUDGETS, as_bool, static_acceptability
from ruscuav_benchmark.scoring import load_run

PARTS = ("acceptable", "successful, pedestrian budget exceeded", "successful, other budget exceeded",
         "not successful")


def part(row) -> str:
    if static_acceptability(row, DEFAULT_BUDGETS):
        return PARTS[0]
    if not as_bool(row["is_success"]):
        return PARTS[3]
    return PARTS[1] if float(row["sidewalk_time"]) > DEFAULT_BUDGETS["ped_s"] else PARTS[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", metavar="WORLD:LABEL=CSV", help="per-episode file of a run in a world")
    args = ap.parse_args()
    out = csv.writer(sys.stdout, lineterminator="\n")
    out.writerow(["world", "method", "part", "n"])
    for item in args.runs:
        key, path = item.split("=", 1)
        world, label = key.split(":", 1)
        counts = Counter(part(row) for row in load_run(path, label).rows)
        for p in PARTS:
            out.writerow([world, label, p, counts[p]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
