#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ruscuav_benchmark.scenarios import sample_static_scenarios, write_scenarios_csv
from ruscuav_benchmark.world import UrbanWorldLayers


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate the fixed AOI_SYNTH_001-003 scenario CSV. The shipped file is "
        "scenarios/legacy_v1/static_eval_seeds.csv; the default output keeps it untouched."
    )
    p.add_argument("--world-root", default="worlds/synthetic/AOI_SYNTH_001_003")
    p.add_argument("--output", default="runs/scenarios/static_eval_seeds.csv")
    p.add_argument("--n-per-world", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260701)
    args = p.parse_args()
    all_rows = []
    for idx, world_path in enumerate(sorted(Path(args.world_root).glob("AOI_SYNTH_*/urban_world_layers.npz"))):
        world = UrbanWorldLayers.load_npz(world_path)
        rows = sample_static_scenarios(world, n=args.n_per_world, seed=args.seed + idx)
        all_rows.extend(rows)
        print(world.aoi_id, len(rows))
    out = write_scenarios_csv(args.output, all_rows)
    print("saved", out, "rows", len(all_rows))


if __name__ == "__main__":
    main()
