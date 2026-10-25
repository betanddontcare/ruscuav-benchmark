"""Episode CSVs and summaries of planner evaluations (no environment needed).

The CSV writer must write what ``pandas.DataFrame(rows).to_csv(index=False)`` writes,
and the summaries must give the values of the archived summaries; the latter is
checked against the archive when RUSCUAV_ARCHIVE points to it.
"""

from pathlib import Path
import json
import math
import os
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.planners import summarize_planner_rows, write_rows_csv  # noqa: E402

ARCHIVE = os.environ.get("RUSCUAV_ARCHIVE")
STATIC_PLANNER_RUN = "experiments/paperA_primary_no_terminal_static_planners/eval/primary_no_terminal_{0}_n200_seed5000/" \
                     "primary_no_terminal_{0}_n200_seed5000"
#: The primary dynamic evaluation; its summary adds conditional statistics of the
#: post-hoc step, which summarize_planner_rows does not compute.
DYNAMIC_PLANNER_RUN = "experiments/paperA_primary_no_terminal_dynamic/eval/primary_no_terminal_dynamic_receding_Astar/" \
                      "primary_no_terminal_dynamic_receding_Astar"


def _rows():
    """A failed static plan (fewer columns, NaN) between two flown episodes."""
    flown = {"episode": 0, "seed": 5000, "planner_success": True, "planner_cost": 880.25, "total_reward": 12167.9,
             "steps": 143, "termination": "success", "is_success": True, "mission_acceptable": True,
             "dist_xy": 24.58, "cargo_health": 0.9758395625308733, "road_time": 2.0, "building_collision_steps": 0}
    failed = {"episode": 1, "seed": None, "planner_success": False, "planner_cost": float("inf"),
              "total_reward": 0.0, "steps": 0, "termination": "planner_no_path", "is_success": False,
              "mission_acceptable": False, "dist_xy": 612.5, "cargo_health": float("nan"), "road_time": 0.0}
    other = dict(flown, episode=2, seed=5002, total_reward=1e-05, cargo_health=0.1 + 0.2, mission_acceptable=False,
                 building_collision_steps=3, termination=None)
    return [flown, failed, other]


def test_csv_matches_pandas(tmp_path):
    pd = pytest.importorskip("pandas")
    rows = _rows()
    path = write_rows_csv(tmp_path / "rows.csv", rows)
    expected = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n")
    assert path.read_text(encoding="utf-8") == expected


def _sequential_std(values):
    mean = 0.0
    for v in values:
        mean += v
    mean /= len(values)
    total = 0.0
    for v in values:
        total += (v - mean) * (v - mean)
    return math.sqrt(total / len(values))


def test_summary_statistics_follow_pandas():
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    rows = []
    for i in range(37):
        rows.append({"is_success": bool(i % 3), "mission_acceptable": bool(i % 4 == 1), "planner_success": True,
                     "planner_cost": float(rng.uniform(500, 900)), "planner_expanded": int(rng.integers(100, 9000)),
                     "planned_path_length": float(rng.uniform(500, 900)), "total_reward": float(rng.normal(1e4, 3e3)),
                     "steps": int(rng.integers(80, 300)), "dist_xy": float(rng.uniform(0, 40)),
                     "cargo_health": float("nan") if i % 5 == 0 else float(rng.uniform(0.8, 1.0)),
                     "urban_mean_risk": float(rng.uniform(0, 1)), "urban_max_risk": float(rng.uniform(0, 8)),
                     "urban_cumulative_risk": float(rng.uniform(0, 60)), "termination": "success" if i % 3 else "stuck"})
    s = summarize_planner_rows(rows, "static")
    df = pd.DataFrame(rows)
    assert s["mean_cargo_health"] == df["cargo_health"].mean()
    assert s["mean_dist_xy"] == df["dist_xy"].mean() and s["median_dist_xy"] == df["dist_xy"].median()
    assert s["success_rate"] == df["is_success"].mean()
    assert s["std_reward"] == _sequential_std(df["total_reward"].tolist())
    assert math.isclose(s["std_reward"], df["total_reward"].std(ddof=0), rel_tol=1e-12)
    success = df[df["is_success"]]
    assert s["acceptable_given_success_rate"] == success["mission_acceptable"].mean()
    assert s["termination_counts"] == {"success": 24, "stuck": 13}


@pytest.mark.skipif(not ARCHIVE, reason="needs RUSCUAV_ARCHIVE (the extracted v2.0.0 archive)")
@pytest.mark.parametrize("name, kind, min_compared", [("STATIC_Astar", "static", 20), ("LOCAL_RH_Astar", "receding", 20),
                                                      ("FULL_RH_Astar", "receding", 20), ("DYN_RH_Astar", "dynamic", 14)])
def test_summaries_equal_the_archived_ones(name, kind, min_compared):
    pd = pytest.importorskip("pandas")
    stem = Path(ARCHIVE) / (DYNAMIC_PLANNER_RUN if kind == "dynamic" else STATIC_PLANNER_RUN.format(name))
    df = pd.read_csv(f"{stem}_episodes.csv", float_precision="round_trip")
    rows = [{k: (None if isinstance(v, float) and math.isnan(v) else (v.item() if hasattr(v, "item") else v))
             for k, v in record.items()} for record in df.to_dict("records")]
    ours = summarize_planner_rows(rows, kind)
    archived = json.loads(Path(f"{stem}_summary.json").read_text(encoding="utf-8"))
    compared = 0
    for key, value in archived.items():
        if key in ours and isinstance(value, (int, float)) and not isinstance(value, bool):
            assert ours[key] == value, key
            compared += 1
    assert compared >= min_compared
    assert ours["termination_counts"] == archived["termination_counts"]
