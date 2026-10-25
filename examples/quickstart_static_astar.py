"""Quickstart: fly the paper's static risk-cost A* planner on the canonical world.

Run from the repository root after ``pip install -e ".[env]"``::

    python examples/quickstart_static_astar.py

The script flies STATIC_Astar on the first 3 recorded canonical-world missions
(``scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv``) in the primary no-terminal
regime, which takes about a minute, and writes the per-episode CSV and the summary to
``runs/quickstart_static_astar/``. The rows are the first rows of the archived
STATIC_Astar evaluation; see docs/PLANNERS.md for the other planners and for your
own worlds.
"""

from pathlib import Path

from ruscuav_benchmark import UrbanWorldLayers
from ruscuav_benchmark.planners import evaluate_planner, missions_from_csv

ROOT = Path(__file__).resolve().parents[1]
world_path = ROOT / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz"
missions_path = ROOT / "scenarios" / "canonical" / "AOI_SYNTH_000_seed5000_n200.csv"
out_dir = ROOT / "runs" / "quickstart_static_astar"

for path in (world_path, missions_path):
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}; run the quickstart from a full repository checkout.")

world = UrbanWorldLayers.load_npz(world_path)
missions = missions_from_csv(missions_path, world)[:3]
result = evaluate_planner(world, "STATIC_Astar", missions=missions, output_dir=out_dir)

for row in result.rows:
    print(f"seed {row['seed']}: {row['termination']} after {row['steps']} steps, "
          f"mission acceptable: {row['mission_acceptable']}, road {row['road_time']} s, "
          f"pedestrian {row['sidewalk_time']} s")
s = result.summary
print(f"success rate {s['success_rate']:.0%}, mission acceptability rate {s['mission_acceptability_rate']:.0%}")
print(f"Per-episode results written to {result.csv_path}")
