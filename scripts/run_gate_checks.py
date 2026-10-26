#!/usr/bin/env python3
"""Run the notebook's world gate checks (M1) and write quality_report.json next to each world.

The gate wants mutually exclusive classes, as in the synthetic worlds, and sane
heights. Worlds built from OpenStreetMap have overlapping classes (roads under
building passages, roads and footways through parks) and print OVERLAP: that is
legal, `ruscuav validate` accepts them. CHECK means a height check failed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ruscuav_benchmark.world import UrbanWorldLayers, run_world_gate_checks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("worlds", nargs="+", help="Paths to urban_world_layers.npz files")
    p.add_argument("--profile", default="balanced")
    args = p.parse_args()
    for world_path in args.worlds:
        world = UrbanWorldLayers.load_npz(world_path)
        report = run_world_gate_checks(world, risk_profile=args.profile)
        out = Path(world_path).with_name("quality_report.json")
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        status = "PASS" if report["passed"] else ("OVERLAP" if report["height"]["passed"] else "CHECK")
        print(world.aoi_id, status, "->", out)


if __name__ == "__main__":
    main()
