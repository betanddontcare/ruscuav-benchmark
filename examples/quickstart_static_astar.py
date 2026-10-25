"""Quickstart: evaluate the static risk-cost A* baseline on one bundled world.

Run from the repository root after ``pip install -e .``::

    python examples/quickstart_static_astar.py

The script evaluates the first 10 scenarios of AOI_SYNTH_001 and writes the
per-episode CSV and summary JSON to ``runs/quickstart_static_astar/``.

It uses the compact RUSCUAV-v0 static profile (``configs/ruscuav_v0_static.yaml``).
The planner configuration reported in the paper is reproduced through the legacy
runtime; see ``scripts/legacy_parity.py``.
"""

import json
from pathlib import Path

from ruscuav_benchmark.evaluate import run_static_astar_evaluation

ROOT = Path(__file__).resolve().parents[1]
world = ROOT / "worlds" / "synthetic" / "AOI_SYNTH_001_003" / "AOI_SYNTH_001" / "urban_world_layers.npz"
scenarios = ROOT / "scenarios" / "legacy_v1" / "static_eval_seeds.csv"
out_dir = ROOT / "runs" / "quickstart_static_astar"

for path in (world, scenarios):
    if not path.exists():
        raise SystemExit(f"Missing {path.relative_to(ROOT)}; run the quickstart from a full repository checkout.")

summary = run_static_astar_evaluation(
    world_path=world,
    scenarios_csv=scenarios,
    output_dir=out_dir,
    max_episodes=10,
)
print(json.dumps(summary, indent=2))
print(f"Per-episode results written to {out_dir}")
