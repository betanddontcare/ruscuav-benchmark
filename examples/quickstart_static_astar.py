from pathlib import Path

from ruscuav_benchmark.evaluate import run_static_astar_evaluation

ROOT = Path(__file__).resolve().parents[1]
world = ROOT / "data" / "ruscuav_v0" / "worlds" / "AOI_SYNTH_001" / "urban_world_layers.npz"
scenarios = ROOT / "data" / "ruscuav_v0" / "scenarios" / "static_eval_seeds.csv"
out_dir = ROOT / "runs" / "quickstart_static_astar"

summary = run_static_astar_evaluation(
    world_path=world,
    scenarios_csv=scenarios,
    output_dir=out_dir,
    max_episodes=10,
)
print(summary)
