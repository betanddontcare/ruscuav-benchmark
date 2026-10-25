#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ruscuav_benchmark.world import UrbanWorldLayers, run_world_gate_checks


def main() -> None:
    p = argparse.ArgumentParser(description="Run RUSCUAV world gate checks.")
    p.add_argument("worlds", nargs="+", help="Paths to urban_world_layers.npz files")
    p.add_argument("--profile", default="balanced")
    args = p.parse_args()
    for world_path in args.worlds:
        world = UrbanWorldLayers.load_npz(world_path)
        report = run_world_gate_checks(world, risk_profile=args.profile)
        out = Path(world_path).with_name("quality_report.json")
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(world.aoi_id, "PASS" if report["passed"] else "CHECK", "->", out)


if __name__ == "__main__":
    main()
