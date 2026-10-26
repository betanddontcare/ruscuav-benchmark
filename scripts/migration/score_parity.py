#!/usr/bin/env python3
"""The paper's tables, rebuilt by ruscuav_benchmark.scoring, against the archived files.

1. The eight primary tables of the post-processing (run summaries, conditional and
   normalized risk, causes of non-acceptability, the dynamic-budget sweep, the two
   training-seed tables and the Holm families) are rebuilt from the archived
   per-episode CSVs (ruscuav_benchmark.paper_tables) and compared with the archived
   files line by line, as text. With SciPy other than the archive's 1.16.3 the
   training-seed intervals may differ in the last digits (reported; accepted below
   1e-13 relative).
2. The secondary episode-bootstrap table of the multi-seed runs is recomputed with
   ruscuav_benchmark.stats.episode_bootstrap_ci and compared value by value.
3. With pandas installed (--pandas), ruscuav_benchmark.tables.read_rows_csv is
   compared with pandas.read_csv, value and type of every cell, in both float modes,
   on every CSV of the archive.

    python scripts/migration/score_parity.py --archive /data/RUSCUAV-v2.0.0 --pandas
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.paper_tables import TABLE_FILES, write_paper_tables  # noqa: E402
from ruscuav_benchmark.stats import episode_bootstrap_ci  # noqa: E402
from ruscuav_benchmark.tables import read_rows_csv  # noqa: E402

BOOTSTRAP_TABLE = "paper_ready_v2/tables/paper_table_multiseed_rl_episode_bootstrap_secondary.csv"
# ms_episode_bootstrap_secondary_table: metric -> column, binary ones first (bootstrap seed 10000 + train seed + index)
_BINARY = {"success_rate": "is_success", "mission_acceptability_rate": "mission_acceptable",
           "static_only_acceptability_rate": "mission_acceptable_static_only",
           "dynamic_budget_ok_rate": "dynamic_budget_ok"}
_CONTINUOUS = {"mean_reward": "total_reward", "mean_steps": "steps", "mean_dist_xy": "dist_xy",
               "mean_cargo_health": "cargo_health", "mean_cumulative_risk": "urban_cumulative_risk",
               "mean_urban_cumulative_risk": "urban_cumulative_risk",
               "mean_dynamic_cumulative_risk": "dynamic_cumulative_risk", "mean_dynamic_risk_max": "dynamic_risk_max",
               "mean_road_time": "road_time", "mean_sidewalk_time": "sidewalk_time",
               "mean_building_time": "building_time", "mean_green_time": "green_time",
               "mean_canopy_time": "canopy_time", "mean_sensitive_time": "sensitive_time",
               "mean_hard_violation_steps": "urban_hard_violation_steps"}


def _max_relative_difference(ours: list, theirs: list) -> float:
    worst = 0.0
    for a, b in zip(ours, theirs):
        for key, x in a.items():
            y = b.get(key)
            if isinstance(x, float) and isinstance(y, float) and x != y and not (math.isnan(x) and math.isnan(y)):
                worst = max(worst, abs(x - y) / max(abs(y), 1e-300))
    return worst


def check_tables(archive: Path) -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        written = write_paper_tables(archive, tmp)
        for name, path in written.items():
            ours = path.read_text(encoding="utf-8").splitlines()
            theirs = (archive / TABLE_FILES[name]).read_text(encoding="utf-8").splitlines()
            differing = sum(a != b for a, b in zip(ours, theirs)) + abs(len(ours) - len(theirs))
            out[name] = {"rows": len(theirs) - 1, "lines_differing": differing}
            if differing:
                out[name]["max_relative_difference"] = _max_relative_difference(
                    read_rows_csv(path, exact_floats=True), read_rows_csv(archive / TABLE_FILES[name], exact_floats=True))
    return out


def _as_bool(value) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def check_bootstrap(archive: Path) -> dict:
    rows = read_rows_csv(archive / BOOTSTRAP_TABLE, exact_floats=True)
    runs = {}
    for row in rows:
        runs.setdefault((row["method"], int(row["train_seed"])), []).append(row)
    identical = compared = 0
    worst = 0.0
    for (method, seed), expected in runs.items():
        files = sorted((archive / "experiments").glob(f"paperA_multiseed_rl_v1_{method}_trainseed{seed}/eval/*/"
                                                      "*episodes*.csv"))
        data = read_rows_csv(files[0], exact_floats=True)
        columns = set(data[0])
        specs = [(m, np.array([_as_bool(d[c]) for d in data], dtype=float)) for m, c in _BINARY.items()
                 if c in columns]
        specs += [(m, np.array([d[c] for d in data if d[c] is not None], dtype=float)) for m, c in _CONTINUOUS.items()
                  if c in columns]
        ours = {m: episode_bootstrap_ci(v, n_boot=2000, seed=10_000 + seed + i) for i, (m, v) in enumerate(specs)}
        for row in expected:
            ref = (float(row["estimate"]), float(row["ci95_low"]), float(row["ci95_high"]))
            got = ours[row["metric"]]
            compared += 1
            identical += got == ref
            worst = max(worst, max(abs(a - b) / max(abs(b), 1e-300) for a, b in zip(got, ref)))
    return {"runs": len(runs), "rows": compared, "identical": identical, "max_relative_difference": worst}


def check_reader(archive: Path) -> dict:
    import pandas as pd

    files = sorted(archive.rglob("*.csv"))
    cells = mismatched = unreadable = 0
    for path in files:
        for exact in (False, True):
            try:
                df = pd.read_csv(path, float_precision="round_trip" if exact else None)
            except Exception:  # noqa: BLE001 - files pandas cannot read are skipped
                unreadable += 1
                break
            try:
                ours = read_rows_csv(path, exact_floats=exact)
            except ValueError:
                mismatched += 1
                continue
            if len(ours) != len(df) or (ours and list(ours[0]) != [str(c) for c in df.columns]):
                mismatched += 1
                continue
            for column in df.columns:
                for value, row in zip(df[column].tolist(), ours):
                    cells += 1
                    mine = row[str(column)]
                    if isinstance(value, float) and math.isnan(value):
                        ok = mine is None
                    else:
                        value = value.item() if hasattr(value, "item") else value
                        ok = type(mine) is type(value) and mine == value
                    mismatched += not ok
    return {"files": len(files), "files_pandas_cannot_read": unreadable, "cells_compared": cells,
            "mismatches": mismatched}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path, help="extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--pandas", action="store_true", help="also compare the CSV reader with pandas on every CSV")
    p.add_argument("--report", type=Path, default=None, help="JSON report")
    args = p.parse_args()
    import scipy

    t0 = time.time()
    report = {"versions": {"numpy": np.__version__, "scipy": scipy.__version__},
              "tables": check_tables(args.archive), "bootstrap": check_bootstrap(args.archive)}
    if args.pandas:
        report["reader"] = check_reader(args.archive)
    report["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(report, indent=2))
    if args.report is not None:
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # with the archive's SciPy (1.16.3) every table is identical; later versions compute the Student-t quantile
    # more accurately, which moves the last digits of the training-seed intervals
    exact = scipy.__version__ == "1.16.3"
    ok = all(t["lines_differing"] == 0 or (not exact and "training_seed" in name
                                           and t.get("max_relative_difference", 1.0) < 1e-13)
             for name, t in report["tables"].items())
    ok = ok and report["bootstrap"]["max_relative_difference"] < 1e-15
    ok = ok and ("reader" not in report or report["reader"]["mismatches"] == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
