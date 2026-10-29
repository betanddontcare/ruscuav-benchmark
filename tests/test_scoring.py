"""Tests for scoring: CSV reading, the statistics, the paper's tables and protocols.

The tables are checked against the notebook's own pandas code (tests/notebook_reference.py)
on synthetic runs, and, with RUSCUAV_ARCHIVE pointing to the extracted v2.0.0 archive,
against the archived table files.
"""

from pathlib import Path
import csv
import io
import json
import math
import os
import random
import sys

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

from ruscuav_benchmark import scoring  # noqa: E402
from ruscuav_benchmark.protocols import FORMAT, Mission, Protocol, list_protocols, load_protocol  # noqa: E402
from ruscuav_benchmark.scenarios import read_scenarios  # noqa: E402
from ruscuav_benchmark.tables import parse_float, read_rows_csv, write_rows_csv  # noqa: E402


def _pandas_2():
    """pandas, for the comparisons with the archive's post-processing, which read its CSVs
    with pandas 2.2.2; the package reads them as pandas 2 does, and pandas 3 reads
    some numbers differently."""
    pd = pytest.importorskip("pandas")
    if int(pd.__version__.split(".")[0]) >= 3:
        pytest.skip(f"the comparison is with pandas 2 (pandas {pd.__version__} is installed)")
    return pd

ARCHIVE = os.environ.get("RUSCUAV_ARCHIVE")


# ---------------------------------------------------------------- reading CSVs
@pytest.mark.parametrize("text, pandas_value, exact_value", [
    ("98.77162609100341", 98.7716260910034, 98.77162609100341),
    ("0.1234567890123456789", 0.1234567890123456, 0.12345678901234568),
    ("0.000000000000000000001", 0.0, 1e-21),
    ("0.30000000000000004", 0.3, 0.30000000000000004),
    ("0.0024589003220967243", 0.0024589003220967, 0.0024589003220967243),
    ("0.9483319529766342", 0.9483319529766342, 0.9483319529766342),
    ("5e-324", 5e-324, 5e-324),
    ("-2.5E+3", -2500.0, -2500.0),
    ("inf", math.inf, math.inf),
    ("-Infinity", -math.inf, -math.inf),
])
def test_parse_float_reads_numbers_as_pandas_does(text, pandas_value, exact_value):
    assert parse_float(text) == pandas_value
    assert parse_float(text, exact=True) == exact_value


@pytest.mark.parametrize("text", ["", "abc", "1e", "1e400", "1.2.3", "--1", "0x10"])
def test_parse_float_refuses_what_pandas_does_not_read(text):
    assert parse_float(text) is None


def test_read_rows_csv_types_columns_as_pandas(tmp_path):
    path = tmp_path / "rows.csv"
    path.write_text(
        "i,i_gap,f,b,b_gap,s,big,huge\n"
        "1,7,0.5,True,False,success,9223372036854775808,58048222443720826880.000\n"
        "-2,,1e3,false,,,5,2.5\n"
        "3,9,inf,TRUE,True,stuck,12,1\n", encoding="utf-8")
    rows = read_rows_csv(path)
    assert [r["i"] for r in rows] == [1, -2, 3] and all(type(r["i"]) is int for r in rows)
    assert [r["i_gap"] for r in rows] == [7.0, None, 9.0] and type(rows[0]["i_gap"]) is float
    assert [r["f"] for r in rows] == [0.5, 1000.0, math.inf]
    assert [r["b"] for r in rows] == [True, False, True]
    assert [r["b_gap"] for r in rows] == [False, None, True]
    assert [r["s"] for r in rows] == ["success", None, "stuck"]
    assert [r["big"] for r in rows] == [9223372036854775808, 5, 12]  # read as uint64
    assert [r["huge"] for r in rows] == ["58048222443720826880.000", "2.5", "1"]  # overflow: kept as text


def test_read_rows_csv_reads_what_pandas_reads(tmp_path):
    path = tmp_path / "edge.csv"
    # a byte-order mark, lines of white space, padded numbers, booleans in any case, the smallest int64 next to
    # a gap, an overflow to -infinity
    path.write_bytes("\ufeffi,f,b,e\n-9223372036854775808, 1.5 ,tRUE,-1e400\n   \n\t\n5,2.5,False,1.5 \n,3.5,true,2\n"
                     .encode("utf-8"))
    rows = read_rows_csv(path)
    assert list(rows[0]) == ["i", "f", "b", "e"] and len(rows) == 3
    assert [r["i"] for r in rows] == [None, 5.0, None]  # pandas marks gaps with the smallest int64
    assert [r["f"] for r in rows] == [1.5, 2.5, 3.5] and [r["b"] for r in rows] == [True, False, True]
    assert [r["e"] for r in rows] == ["-1e400", "1.5 ", "2"]  # the default parser does not overflow
    exact = read_rows_csv(path, exact_floats=True)
    assert [r["e"] for r in exact] == [-math.inf, 1.5, 2.0]  # round trip: -infinity, but +infinity is an error
    assert parse_float(" 1.5 ", exact=True) == 1.5 and parse_float("1e400", exact=True) is None
    pd = _pandas_2()
    for exact_floats, rows_read in ((False, rows), (True, exact)):
        df = pd.read_csv(path, float_precision="round_trip" if exact_floats else None)
        for column in df.columns:
            for value, row in zip(df[column].tolist(), rows_read):
                if isinstance(value, float) and math.isnan(value):
                    assert row[column] is None, column
                else:
                    assert type(row[column]) is type(value) and row[column] == value, (column, value)


def _random_csv(rng):
    n = rng.randint(1, 25)

    def number():
        x = rng.uniform(-1e3, 1e3) * 10 ** rng.randint(-12, 12)
        return rng.choice([repr(x), "%.17g" % x, "%.3f" % x, "%.6e" % x, "%g" % x, str(np.float32(x))])

    cols = {
        "i": [str(rng.randint(-10 ** 6, 10 ** 6)) for _ in range(n)],
        "gap": [rng.choice(["", str(rng.randint(0, 99))]) for _ in range(n)],
        "f": [number() for _ in range(n)],
        "na": [rng.choice(["", "nan", "NaN", "NA", number()]) for _ in range(n)],
        "b": [rng.choice(["True", "False", "TRUE", "false", ""]) for _ in range(n)],
        "s": [rng.choice(["success", "", "a,b", 'q"x']) for _ in range(n)],
        "mix": [rng.choice(["1", "2.5", "inf", "-inf", "1e5", ".5", "5.", "-0.0"]) for _ in range(n)],
        "big": [rng.choice(["9223372036854775808", "18446744073709551616", "-3", "", "7"]) for _ in range(n)],
    }
    buf = io.StringIO()
    if rng.random() < 0.3:
        buf.write("\ufeff")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(cols)
    for k in range(n):
        writer.writerow([cols[c][k] for c in cols])
        if rng.random() < 0.1:
            buf.write(rng.choice(["\n", "  \n", "\t\n"]))
    return buf.getvalue()


def test_read_rows_csv_agrees_with_pandas(tmp_path):
    pd = _pandas_2()
    rng = random.Random(11)
    for k in range(60):
        path = tmp_path / f"r{k}.csv"
        path.write_text(_random_csv(rng), encoding="utf-8")
        for exact in (False, True):
            df = pd.read_csv(path, float_precision="round_trip" if exact else None)
            ours = read_rows_csv(path, exact_floats=exact)
            for column in df.columns:
                for value, row in zip(df[column].tolist(), ours):
                    if isinstance(value, float) and math.isnan(value):
                        assert row[column] is None, (k, column)
                    else:
                        value = value.item() if hasattr(value, "item") else value
                        assert type(row[column]) is type(value) and row[column] == value, (k, column, value)


# ------------------------------------------------------------------- statistics
def test_statistics_give_the_archived_values():
    scipy = pytest.importorskip("scipy")
    from ruscuav_benchmark.stats import holm_adjust, mcnemar_exact_p, training_seed_ci

    # GEO-RL success rates of the five training seeds (paper_table_primary_no_terminal_static_training_seed_ci.csv);
    # the interval is the archive's bit for bit with its SciPy (1.16.3): later versions compute the t quantile
    # more accurately, which moves the last digits
    ci = training_seed_ci([0.775, 0.77, 0.815, 0.82, 0.785])
    archived = {"n": 5, "mean": 0.7929999999999999, "std": 0.02307596151842862,
                "ci95_low": 0.7643474093586315, "ci95_high": 0.8216525906413683}
    if scipy.__version__ == "1.16.3":
        assert ci == archived
    else:
        assert ci == pytest.approx(archived, rel=1e-13) and ci["mean"] == archived["mean"]
    assert training_seed_ci([0.5])["n"] == 1 and math.isnan(training_seed_ci([0.5])["std"])
    assert training_seed_ci([None, "x"])["n"] == 0
    # the static Holm family (paper_table_holm_full_families.csv)
    raw = [mcnemar_exact_p(25, 13), mcnemar_exact_p(29, 20), mcnemar_exact_p(33, 12), mcnemar_exact_p(36, 19),
           mcnemar_exact_p(3, 33), mcnemar_exact_p(4, 49)]
    assert raw == [0.07295138851623051, 0.2528697301676033, 0.002458900322096724, 0.030028945411892487,
                   2.2721360437572002e-07, 7.053979622639872e-11]
    assert holm_adjust(raw).tolist() == [0.14590277703246102, 0.2528697301676033, 0.009835601288386895,
                                         0.09008683623567745, 1.1360680218786001e-06, 4.2323877735839233e-10]
    assert mcnemar_exact_p(0, 0) == 1.0 and mcnemar_exact_p(6, 6) == 1.0
    with pytest.raises(ValueError):
        mcnemar_exact_p(-1, 2)


# ------------------------------------------------------------- the tables
def _synthetic_rows(rng, n=60, dynamic=False, offset=0.0, seed0=5000):
    rows = []
    for k in range(n):
        success = rng.random() < 0.8 - offset
        steps = rng.randint(0, 400) if rng.random() < 0.97 else 0
        row = {
            "episode": k, "seed": seed0 + k, "method": "M", "total_reward": rng.uniform(-500, 1500), "steps": steps,
            "termination": "success" if success else rng.choice(["stuck", "shield_stuck", "timeout"]),
            "is_success": success, "dist_xy": rng.uniform(0, 300),
            "path_length_xy": rng.choice([0.0, rng.uniform(1, 900)]),
            "cargo_health": rng.uniform(0.7, 1.0), "urban_cumulative_risk": rng.uniform(0, 80),
            "road_time": rng.choice([0.0, rng.uniform(0, 30)]), "sidewalk_time": rng.uniform(0, 9),
            "building_time": rng.uniform(0, 20), "green_time": rng.uniform(0, 40), "canopy_time": rng.uniform(0, 10),
            "sensitive_time": rng.choice([0.0, 0.0, 0.0, 0.5]),
            "urban_hard_violation_steps": rng.choice([0, 0, 0, 0, 2]),
        }
        if dynamic:
            row["dynamic_cumulative_risk"] = rng.uniform(0, 25)
        rows.append(row)
    return rows


def _write(tmp_path, name, rows):
    path = tmp_path / f"{name}.csv"
    write_rows_csv(path, rows)
    return path


def _assert_same(ours, theirs):
    assert ours.keys() == theirs.keys(), set(ours) ^ set(theirs)
    for key, value in theirs.items():
        mine = ours[key]
        if isinstance(value, float) and math.isnan(value):
            assert mine is None or math.isnan(mine), key
        else:
            assert mine == value and type(mine) in (type(value), int, float, bool), (key, mine, value)


def _records(df):
    out = []
    for rec in df.to_dict("records"):
        out.append({k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()})
    return out


def test_tables_agree_with_the_notebook_code(tmp_path):
    pd = _pandas_2()
    pytest.importorskip("scipy")
    import notebook_reference as nb

    rng = random.Random(5)
    runs, registry = [], {}
    for label, seeds, dynamic in (("A", (1001, 2001, 3001), False), ("B", (1001, 2001), False), ("D", (None,), True)):
        for train_seed in seeds:
            name = f"{label}_{train_seed}"
            path = _write(tmp_path, name, _synthetic_rows(rng, dynamic=dynamic, offset=0.1 * len(runs) % 0.3))
            runs.append(scoring.load_run(path, label, dynamic=dynamic, train_seed=train_seed, scope="s"))
            registry[name] = path
    for run in runs:
        df = pd.read_csv(run.source)
        summary = nb.summarize_episode_dataframe(df, dynamic=run.dynamic)
        _assert_same({k: v for k, v in scoring.summary_row(run).items() if k not in run.fields()},
                     {k: v for k, v in summary.items() if not isinstance(v, dict)})
        _assert_same_rows = [dict(r) for r in scoring.cause_rows(run)]
        for ours, theirs in zip(_assert_same_rows, _records(nb.cause_decomposition(df, run.label, run.dynamic))):
            _assert_same(ours, theirs)
        if run.dynamic:
            reference = _records(nb.dynamic_budget_sensitivity(df, run.label))
            for ours, theirs in zip(scoring.dynamic_budget_rows(run), reference):
                _assert_same({k: v for k, v in ours.items() if k not in ("train_seed", "csv_path")}, theirs)
    # training-seed intervals over the summaries
    summaries = scoring.summary_table(runs)
    reference = _records(nb.training_seed_ci_all_scopes(pd.DataFrame(summaries)))
    ours = scoring.training_seed_table(summaries)
    assert len(ours) == len(reference) == 6
    for a, b in zip(ours, reference):
        _assert_same(a, b)
    # paired tests with Holm's correction
    labels = {"A": registry["A_1001"], "B": registry["B_1001"], "D": registry["D_None"]}
    spec = [("f1", "ab_success", "A", "B", "success", True), ("f1", "ab_mar", "A", "B", "MAR_static", False),
            ("f1", "ad_mar", "A", "D", "MAR_static", True), ("f2", "dd", "D", "D", "DynOK", True)]
    template = pd.DataFrame(spec, columns=["family", "comparison_id", "method_a", "method_b", "metric",
                                           "claim_driving"])
    reference = _records(nb.compute_holm_families(template, {k: str(v) for k, v in labels.items()}))
    comparisons = [dict(zip(template.columns, s)) for s in spec]
    ours = scoring.holm_table(comparisons, {k: scoring.load_run(v, k) for k, v in labels.items()})
    assert [list(r) for r in ours] == [list(r) for r in reference]
    for a, b in zip(ours, reference):
        _assert_same(a, b)
    # the episode bootstrap
    from ruscuav_benchmark.stats import episode_bootstrap_ci

    values = [r["urban_cumulative_risk"] for r in runs[0].rows]
    assert episode_bootstrap_ci(values, n_boot=300, seed=7) == nb.ms_bootstrap_ci(values, n_boot=300, seed=7)
    flags = np.array([r["is_success"] for r in runs[0].rows])
    assert episode_bootstrap_ci(flags, n_boot=200, seed=3) == nb.ms_bootstrap_ci(flags, n_boot=200, seed=3)


def test_score_runs_writes_the_tables(tmp_path):
    pytest.importorskip("scipy")
    rng = random.Random(9)
    runs = [scoring.Run("A", _synthetic_rows(rng), train_seed=1001.0, scope="x"),
            scoring.Run("A", _synthetic_rows(rng), train_seed=2001.0, scope="x"),
            scoring.Run("B", _synthetic_rows(rng, dynamic=True), dynamic=True, scope="x")]
    with pytest.raises(ValueError, match="name several runs"):
        scoring.score_runs(runs, [{"family": "f", "comparison_id": "c", "method_a": "A", "method_b": "B",
                                   "metric": "success"}])
    tables = scoring.score_runs(runs[1:], [{"family": "f", "comparison_id": "c", "method_a": "A", "method_b": "B",
                                            "metric": "MAR_static", "claim_driving": True}])
    assert len(tables.summaries) == 2 and not tables.training_seed_ci and len(tables.causes) == 16
    assert len(tables.dynamic_budget) == len(scoring.DYNAMIC_BUDGET_SWEEP) and len(tables.paired_tests) == 1
    written = tables.write(tmp_path)
    assert sorted(p.name for p in written.values()) == ["dynamic_budget_sensitivity.csv",
                                                        "non_acceptability_causes.csv", "paired_tests_holm.csv",
                                                        "run_summaries.csv"]
    assert len(scoring.score_runs(runs).training_seed_ci) == 3
    # runs in another regime are not pooled with the primary ones
    secondary = scoring.Run("A", _synthetic_rows(rng), train_seed=3001.0, scope="x",
                            regime="secondary_operational_stopping")
    seed_table = scoring.score_runs(runs[:2] + [secondary]).training_seed_ci
    assert {r["regime"] for r in seed_table} == {"primary_no_terminal"} and {r["n"] for r in seed_table} == {2}
    # comparisons from several sources need not have the same columns
    mixed = scoring.holm_table([{"family": "f", "comparison_id": "c1", "method_a": "A", "method_b": "B",
                                 "metric": "success", "claim_driving": False},
                                {"family": "f", "comparison_id": "c2", "method_a": "A", "method_b": "B",
                                 "metric": "MAR_static"}], {"A": runs[1], "B": runs[2]})
    assert list(mixed[0]) == list(mixed[1]) and mixed[1]["claim_driving"] is None and mixed[1]["family_size"] == 2
    # comparison files: a byte-order mark and blank lines are fine
    declared = tmp_path / "comparisons.csv"
    declared.write_bytes("\ufefffamily,comparison_id,method_a,method_b,metric,claim_driving\n\n"
                         "f,c1,A,B,success,True\n   \n".encode("utf-8"))
    assert scoring.read_comparisons(declared) == [{"family": "f", "comparison_id": "c1", "method_a": "A",
                                                   "method_b": "B", "metric": "success", "claim_driving": True}]
    with pytest.raises(ValueError, match="not available"):
        scoring.paired_test(runs[0], runs[2], "DynOK")
    with pytest.raises(KeyError):
        scoring.holm_table([{"family": "f", "comparison_id": "c", "method_a": "A", "method_b": "Z",
                             "metric": "success"}], {"A": runs[0]})
    duplicate = scoring.Run("C", runs[0].rows + runs[0].rows[:1])
    with pytest.raises(ValueError, match="several episodes"):
        scoring.paired_outcomes(duplicate)


def test_runs_are_checked_before_scoring(tmp_path, capsys):
    """A missing column or cell would count as within its budget (as in the archive's post-processing):
    the runs given to ruscuav score are refused instead, those of a registry noted."""
    from ruscuav_benchmark.cli import main

    rows = _synthetic_rows(random.Random(3), n=20)
    good = scoring.Run("A", rows)
    scoring.check_run(good)
    assert scoring.missing_columns(good) == [] and scoring.trajectory_checks(good) is None
    for key in ("is_success", "cargo_health", "sidewalk_time", "urban_hard_violation_steps"):
        run = scoring.Run("A", [{k: v for k, v in r.items() if k != key} for r in rows])
        assert scoring.missing_columns(run) == [key]
        with pytest.raises(ValueError, match=f"no column {key}"):
            scoring.check_run(run)
    other_name = [{**{k: v for k, v in r.items() if k != "urban_hard_violation_steps"},
                   "hard_violation_steps": r["urban_hard_violation_steps"]} for r in rows]
    scoring.check_run(scoring.Run("A", other_name))  # the archive's other name of the count
    assert scoring.missing_columns(scoring.Run("D", rows, dynamic=True)) == ["dynamic_cumulative_risk"]
    # without a protocol a run is dynamic when it has dynamic_cumulative_risk: a misspelled one would drop the
    # dynamic budget, so the other columns of the dynamic evaluations require it
    misspelled = scoring.Run("D", [dict(r, dynamic_budget_ok=False, dynamic_cumulative_risks=50.0) for r in rows])
    assert not misspelled.dynamic and scoring.missing_columns(misspelled) == ["dynamic_cumulative_risk"]
    with pytest.raises(ValueError, match="no column dynamic_cumulative_risk"):
        scoring.check_run(misspelled)
    successful = next(i for i, r in enumerate(rows) if r["is_success"])
    for key, value, message in (("is_success", "1.0", "not true or false"), ("is_success", None, "not true or false"),
                                ("road_time", None, "without a number in road_time"),
                                ("road_time", float("nan"), "without a number in road_time"),
                                ("sidewalk_time", "n/a", "without a number in sidewalk_time"),
                                ("sidewalk_time", float("inf"), "without a number in sidewalk_time"),
                                ("building_time", True, "without a number in building_time"),
                                ("cargo_health", 1.2, "not from 0 to 1"),
                                ("building_time", -1.0, "negative building_time")):
        bad = [dict(r) for r in rows]
        bad[successful][key] = value
        with pytest.raises(ValueError, match=message):
            scoring.check_run(scoring.Run("A", bad))
    loose = [dict(r) for r in rows]
    loose[next(i for i, r in enumerate(rows) if not r["is_success"])]["cargo_health"] = None
    scoring.check_run(scoring.Run("A", loose))  # an unsuccessful episode is not acceptable whatever its cells
    # scored trajectories: without the infeasible column, rows of a development version
    with pytest.raises(ValueError, match="development version"):
        scoring.check_run(scoring.Run("T", [dict(r, cargo_health_source="estimated") for r in rows]))
    scored = scoring.Run("T", [dict(r, cargo_health_source="capped" if i % 2 else "given",
                                    infeasible="speed" if i < 3 else None, within_acceleration_limits=i >= 5)
                               for i, r in enumerate(rows)])
    scoring.check_run(scored)
    assert scoring.trajectory_checks(scored) == {
        "episodes": 20, "earlier_version": False, "infeasible": 3, "infeasible_by_reason": {"speed": 3},
        "outside_acceleration_limits": 5, "cargo_health": {"capped": 10, "given": 10}}
    # on the command line
    full = _write(tmp_path, "full_episodes", rows)
    short = _write(tmp_path, "short_episodes", [{k: v for k, v in r.items() if k != "building_time"} for r in rows])
    out = tmp_path / "score"
    assert main(["score", f"A={full}", f"B={short}", "--output-dir", str(out)]) == 2
    assert "run 'B' has no column building_time" in capsys.readouterr().err
    old = _write(tmp_path, "old_episodes", [dict(r, cargo_health_source="given") for r in rows])
    registry = tmp_path / "registry.csv"
    registry.write_text(f"label,csv_path\nA,{full.name}\nB,{short.name}\nT,{old.name}\n", encoding="utf-8")
    assert main(["score", "--registry", str(registry), "--output-dir", str(out)]) == 0
    err = capsys.readouterr().err
    assert "run 'B' of the registry has no building_time; a missing column" in err
    assert "run 'T': trajectories scored by a development version" in err
    report = json.loads((out / "score.json").read_text(encoding="utf-8"))
    assert [r["missing_columns"] for r in report["runs"]] == [[], ["building_time"], []]
    assert [r["trajectories"] for r in report["runs"]][:2] == [None, None]
    assert report["runs"][2]["trajectories"]["earlier_version"] is True


@pytest.mark.skipif(not ARCHIVE, reason="needs RUSCUAV_ARCHIVE (the extracted v2.0.0 archive)")
def test_paper_tables_reproduce_the_archived_files(tmp_path):
    scipy = pytest.importorskip("scipy")
    from ruscuav_benchmark.paper_tables import TABLE_FILES, write_paper_tables

    written = write_paper_tables(ARCHIVE, tmp_path)
    assert set(written) == set(TABLE_FILES)
    for name, path in written.items():
        archived = Path(ARCHIVE) / TABLE_FILES[name]
        if scipy.__version__ == "1.16.3" or "training_seed" not in name:
            assert path.read_text(encoding="utf-8") == archived.read_text(encoding="utf-8"), name
            continue
        # other SciPy versions: the t quantile, and with it the interval bounds, differ in the last digits
        ours, theirs = read_rows_csv(path, exact_floats=True), read_rows_csv(archived, exact_floats=True)
        assert len(ours) == len(theirs)
        for a, b in zip(ours, theirs):
            assert {k: v for k, v in a.items() if not k.startswith("ci95")} == \
                {k: v for k, v in b.items() if not k.startswith("ci95")}
            assert a["ci95_low"] == pytest.approx(b["ci95_low"], rel=1e-13, nan_ok=True)
            assert a["ci95_high"] == pytest.approx(b["ci95_high"], rel=1e-13, nan_ok=True)


# -------------------------------------------------------------------- protocols
WORLD = REPO / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz"


def test_builtin_protocols_are_the_recorded_missions():
    assert list_protocols() == ["canonical-dynamic", "canonical-primary", "canonical-secondary", "vienna-primary"]
    for name in list_protocols():
        p = load_protocol(name)
        source = REPO / p.missions_source["file"]
        assert p.missions_source["sha256"] == __import__("hashlib").sha256(source.read_bytes()).hexdigest()
        rows = read_scenarios(source)
        assert [m.seed for m in p.missions] == [int(r["seed"]) for r in rows]
        assert [m.start for m in p.missions] == [tuple(float(r[f"start_{a}"]) for a in "xyz") for r in rows]
        assert [m.target for m in p.missions] == [tuple(float(r[f"target_{a}"]) for a in "xyz") for r in rows]
        meta = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
        assert p.world_layers_sha256 == meta["world_layers_sha256"] and p.n_episodes == 200
        assert p.budgets == scoring.DEFAULT_BUDGETS
    assert load_protocol("canonical-dynamic").dynamic and not load_protocol("canonical-primary").dynamic
    assert load_protocol("canonical-secondary").regime == "secondary_operational_stopping"


def test_protocol_files_round_trip_and_are_checked(tmp_path):
    p = load_protocol("canonical-dynamic")
    path = p.save(tmp_path / "p.json")
    q = load_protocol(path)
    assert q.to_dict() == p.to_dict() and q.digest() == p.digest() and q.dynamic_risk == p.dynamic_risk
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == FORMAT
    for change, message in ((lambda d: d.update(format="x"), "format"),
                            (lambda d: d["missions"].append(d["missions"][0]), "repeat a seed"),
                            (lambda d: d.update(dynamic_risk=None), "dynamic risk"),
                            (lambda d: d.update(regime="stage_D"), "regime"),
                            (lambda d: d.update(missions=[]), "no missions"),
                            (lambda d: d["budgets"].update(road=1.0), "unknown budgets"),
                            (lambda d: d["budgets"].update(road_s=-1.0), "at least 0"),
                            (lambda d: d["budgets"].update(cargo_min=1.5), "cargo_min"),
                            (lambda d: d["budgets"].update(dynamic=12.0), "dynamic_cumulative_risk_budget"),
                            (lambda d: d["dynamic_risk"].pop("dynamic_cumulative_risk_budget"),
                             "dynamic_cumulative_risk_budget"),
                            (lambda d: d.update(max_steps=0), "max_steps"),
                            (lambda d: d.update(dt_s=0.0), "dt_s")):
        broken = json.loads(json.dumps(data))
        change(broken)
        with pytest.raises(ValueError, match=message):
            Protocol.from_dict(broken)
    with pytest.raises(FileNotFoundError):
        load_protocol("no-such-protocol")
    # a relative world file is looked for next to the protocol file first
    from ruscuav_benchmark.protocols import find_world

    folder = tmp_path / "elsewhere"
    folder.mkdir()
    (folder / "w.npz").write_bytes(WORLD.read_bytes())
    data["world"]["file"] = "w.npz"
    (folder / "p.json").write_text(json.dumps(data), encoding="utf-8")
    moved = load_protocol(folder / "p.json")
    assert find_world(moved) == folder / "w.npz"
    # the digest is the content's: neither the world file's location nor the description count
    assert moved.world_file == "w.npz" and moved.digest() == p.digest()
    described = json.loads(json.dumps(data))
    described["description"] = "another description"
    assert Protocol.from_dict(described).digest() == p.digest()
    for change in (lambda d: d["missions"][0]["target"].__setitem__(0, d["missions"][0]["target"][0] + 1.0),
                   lambda d: d["budgets"].update(ped_s=10.0),
                   lambda d: d["world"].update(layers_sha256="0" * 64),
                   lambda d: d.update(name="another-name")):
        changed = json.loads(json.dumps(data))
        change(changed)
        assert Protocol.from_dict(changed).digest() != p.digest()


def test_protocol_settings_and_seed_checks():
    from ruscuav_benchmark import UrbanWorldLayers

    p = load_protocol("canonical-primary")
    world = UrbanWorldLayers.load_npz(WORLD)
    settings, record = p.apply(world, dynamic_method=False, n_episodes=3)
    assert len(settings["missions"]) == 3 and settings["regime"] == "primary_no_terminal"
    assert record["complete"] is False and record["missions_flown"] == 3 and record["digest"] == p.digest()
    assert p.apply(world, dynamic_method=False)[1]["complete"] is True
    with pytest.raises(ValueError, match="fixes"):
        p.apply(world, dynamic_method=False, given={"regime": "primary_no_terminal"})
    with pytest.raises(ValueError, match="no dynamic risk"):
        p.apply(world, dynamic_method=True)
    with pytest.raises(ValueError, match="dynamic risk"):
        load_protocol("canonical-dynamic").apply(world, dynamic_method=False)
    other = UrbanWorldLayers.load_npz(REPO / "worlds" / "synthetic" / "AOI_SYNTH_001_003" / "AOI_SYNTH_001" /
                                      "urban_world_layers.npz")
    with pytest.raises(ValueError, match="layers checksum"):
        p.apply(other, dynamic_method=False)
    assert p.check_seeds(range(5000, 5200)) == {"complete": True, "missing": []}
    assert p.check_seeds([5000, 5001])["missing"][:2] == [5002, 5003]
    with pytest.raises(ValueError, match="not missions"):
        p.check_seeds([4999])
    with pytest.raises(ValueError, match="repeats"):
        p.check_seeds([5000, 5000])
    assert p.check_seeds([5000.0, "5001"])["missing"][0] == 5002
    for seeds, message in (([None], "without a seed"), ([float("nan")], "without a seed"), (["x"], "integers"),
                           ([5000.5], "integers")):
        with pytest.raises(ValueError, match=message):
            p.check_seeds(seeds)
    assert len(p.episodes(3)) == 3 and len(p.episodes()) == 200
    for n in (0, -1, 201, 2.5):
        with pytest.raises(ValueError, match="200 missions"):
            p.episodes(n)
    assert Mission(5000, (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)).episode() == (5000, {"start": [1.0, 2.0, 3.0],
                                                                               "target": [4.0, 5.0, 6.0]})


def test_make_protocol_records_the_environment_missions(tmp_path):
    pytest.importorskip("gymnasium")
    pytest.importorskip("scipy")
    from ruscuav_benchmark.protocols import make_protocol

    made = make_protocol(WORLD, "mine", n_episodes=4, seed0=5000, dynamic=True)
    reference = load_protocol("canonical-primary").missions[:4]
    for ours, recorded in zip(made.missions, reference):
        assert ours.seed == recorded.seed
        assert np.array_equal(np.float32(ours.start), np.float32(recorded.start))
        assert np.array_equal(np.float32(ours.target), np.float32(recorded.target))
    assert made.dynamic and made.world_layers_sha256 == load_protocol("canonical-primary").world_layers_sha256
    assert load_protocol(made.save(tmp_path / "mine.json")).digest() == made.digest()
    assert made.budgets == scoring.DEFAULT_BUDGETS and made.world_file == WORLD.as_posix()
    # budgets: the paper's, with the given ones replaced; one dynamic budget
    from dataclasses import asdict

    from ruscuav_benchmark.env.dynamic_risk import REPORTED_DYNAMIC_CONFIG

    strict = make_protocol(WORLD, "strict", n_episodes=1, budgets={"road_s": 0.0})
    assert strict.budgets == {**scoring.DEFAULT_BUDGETS, "road_s": 0.0} and not strict.dynamic
    dyn = make_protocol(WORLD, "dyn12", n_episodes=1, dynamic=True, budgets={"dynamic": 12.0})
    assert dyn.budgets["dynamic"] == 12.0 and dyn.dynamic_risk["dynamic_cumulative_risk_budget"] == 12.0
    loose = make_protocol(WORLD, "dyn35", n_episodes=1,
                          dynamic_risk={**asdict(REPORTED_DYNAMIC_CONFIG), "dynamic_cumulative_risk_budget": 35.0})
    assert loose.budgets["dynamic"] == 35.0
    with pytest.raises(ValueError, match="differ"):
        make_protocol(WORLD, "both", n_episodes=1, budgets={"dynamic": 12.0},
                      dynamic_risk={**asdict(REPORTED_DYNAMIC_CONFIG), "dynamic_cumulative_risk_budget": 35.0})
    with pytest.raises(ValueError, match="at least 1"):
        make_protocol(WORLD, "none", n_episodes=0)


# ------------------------------------------------------------ evaluation under a protocol
def test_protocol_evaluations_equal_plain_ones():
    pytest.importorskip("gymnasium")
    pytest.importorskip("scipy")
    from ruscuav_benchmark import UrbanWorldLayers
    from ruscuav_benchmark.planners import evaluate_planner
    from ruscuav_benchmark.policies import TargetSeekingPolicy, evaluate_policy

    world = UrbanWorldLayers.load_npz(WORLD)
    a = evaluate_planner(world, "STATIC_Astar", protocol="canonical-primary", n_episodes=2)
    b = evaluate_planner(world, "STATIC_Astar", n_episodes=2, seed0=5000)
    keys = [k for k in b.rows[0] if "plan_time" not in k]
    assert [[r[k] for k in keys] for r in a.rows] == [[r[k] for k in keys] for r in b.rows]
    assert a.summary["protocol"]["name"] == "canonical-primary" and not a.summary["protocol"]["complete"]
    c = evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", protocol="canonical-primary", n_episodes=2,
                        max_steps=1000)
    d = evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", n_episodes=2, seed0=5000)
    assert c.rows == d.rows
    with pytest.raises(ValueError, match="fixes"):
        evaluate_planner(world, "STATIC_Astar", protocol="canonical-primary", regime="primary_no_terminal")
    with pytest.raises(ValueError, match="max_steps"):
        evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", protocol="canonical-primary", max_steps=5)
    with pytest.raises(ValueError, match="dynamic risk"):
        evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", protocol="canonical-dynamic", n_episodes=1)
    # a dynamic protocol scores dynamic acceptability: the posthoc format has no dynamic risk
    with pytest.raises(ValueError, match="row format"):
        evaluate_policy(world, TargetSeekingPolicy(), "DYN-OBS-RL", protocol="canonical-dynamic", n_episodes=1,
                        row_format="posthoc")
    # the protocol fixes the environment's accounting; the method's own settings may change
    from ruscuav_benchmark.env import make_config
    from ruscuav_benchmark.env.config import DynamicsConfig, EpisodeConfig, LidarConfig, MissionConfig

    with pytest.raises(ValueError, match="fixes these environment settings"):
        evaluate_policy(world, TargetSeekingPolicy(), "GEO-RL", protocol="canonical-primary", n_episodes=1,
                        env_overrides={"episode": EpisodeConfig(max_steps=3)})
    with pytest.raises(ValueError, match="max_steps"):
        evaluate_planner(world, "STATIC_Astar", protocol="canonical-primary", n_episodes=1, max_steps=3)
    p = load_protocol("canonical-primary")
    p.check_environment(p.environment_config("GEO-RL", lidar=LidarConfig(urban=False)))
    p.check_environment(make_config("GEO-RL", regime="primary_no_terminal"))
    for changed, what in ((make_config("GEO-RL", regime="primary_no_terminal", dynamics=DynamicsConfig(dt=0.25)),
                           "dynamics"),
                          (make_config("GEO-RL", regime="primary_no_terminal",
                                       mission=MissionConfig(accept_road_s=30.0)), "mission"),
                          (make_config("GEO-RL", regime="secondary_operational_stopping"), "regime"),
                          (make_config("GEO-RL", regime="primary_no_terminal", risk_profile="safety_first"),
                           "risk_profile"),
                          (make_config("DYN-RISK-RL", regime="primary_no_terminal"), "dynamic_risk")):
        with pytest.raises(ValueError, match=what):
            p.check_environment(changed)
