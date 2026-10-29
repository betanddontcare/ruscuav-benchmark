"""Scoring evaluation results: the paper's tables for any set of runs.

A *run* is the per-episode rows of one evaluation: one planner, one policy checkpoint
or any other method flown on the missions of one protocol (a CSV written by
``ruscuav evaluate``, by the archived evaluations, or by your own code with the same
columns). From a list of runs this module computes the tables of the paper's
post-processing, with the archive's arithmetic:

- :func:`summary_table`: one row per run with the success rate, the mission
  acceptability recomputed post hoc with the budgets, and the conditional and
  normalized risk and exposure means (``paper_table_primary_no_terminal_all_run_summaries``);
- :func:`training_seed_table`: mean, standard deviation and Student-t interval over
  the training seeds of each method (``..._training_seed_ci_all_scopes``);
- :func:`cause_table`: why episodes are not acceptable
  (``paper_table_non_acceptability_cause_decomposition``);
- :func:`dynamic_budget_table`: acceptability over a sweep of dynamic-risk budgets
  (``paper_table_dynamic_budget_sensitivity``);
- :func:`holm_table`: exact paired (McNemar) tests of declared comparisons with
  Holm's correction within each family (``paper_table_holm_full_families``).

Runs are dynamic when their acceptability includes the dynamic-risk budget (the
archived dynamic evaluations; by default, runs whose rows have
``dynamic_cumulative_risk``). Missions are paired by the ``seed`` column.

The tables read the cells as the archive's post-processing did: a missing column or
cell counts as within its budget. :func:`check_run` refuses runs whose rows cannot
be scored as they are (``ruscuav score`` checks the runs given on its command line
with it), and :func:`missing_columns` names the columns the acceptability needs and
a run lacks.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .posthoc import DEFAULT_BUDGETS, DT_S, _hard_key, _number, as_bool, dynamic_acceptability, \
    static_acceptability, summarize_posthoc
from .stats import holm_adjust, mcnemar_exact_p, training_seed_ci
from .tables import Row, is_missing, pandas_mean, read_rows_csv, write_rows_csv

__all__ = [
    "Run", "load_run", "read_registry", "ACCEPTABILITY_COLUMNS", "missing_columns", "check_run", "trajectory_checks",
    "CAUSES", "DYNAMIC_BUDGET_SWEEP", "SEED_CI_METRICS", "PAIRED_METRICS", "PAPER_HOLM_COMPARISONS", "summary_row",
    "summary_table", "training_seed_table", "cause_rows", "cause_table", "dynamic_budget_rows",
    "dynamic_budget_table", "paired_outcomes", "paired_test", "holm_table", "read_comparisons", "ScoreTables",
    "score_runs",
]

#: Columns that mission acceptability reads, besides a count of hard violations
#: (``urban_hard_violation_steps`` or ``hard_violation_steps``) and, in dynamic runs,
#: ``dynamic_cumulative_risk``.
ACCEPTABILITY_COLUMNS = ("is_success", "cargo_health", "road_time", "sidewalk_time", "building_time",
                         "sensitive_time")
_HARD_COLUMNS = ("urban_hard_violation_steps", "hard_violation_steps")
#: The values ``is_success`` may have (``as_bool`` reads the first five as true).
_BOOLEAN_TEXT = {"true", "1", "yes", "y", "t", "false", "0", "no", "n", "f"}

#: Causes of non-acceptability, in the archive's order. An episode can have several.
CAUSES = (
    "not_success", "cargo_low", "road_budget_exceeded", "ped_budget_exceeded", "building_budget_exceeded",
    "sensitive_budget_exceeded", "hard_violation", "dynamic_budget_exceeded",
)
#: Dynamic-risk budgets of the archived sensitivity sweep.
DYNAMIC_BUDGET_SWEEP = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 35.0)
#: Statistics summarized over training seeds by default.
SEED_CI_METRICS = ("success_rate", "mission_acceptability_rate", "acceptable_given_success_rate")
#: Binary outcomes the paired tests compare.
PAIRED_METRICS = ("success", "MAR_static", "DynOK", "MAR_dynamic")

#: The paper's declared families of paired comparisons (Table IV): method labels
#: refer to runs (see :mod:`ruscuav_benchmark.paper_tables` for the archived ones).
PAPER_HOLM_COMPARISONS: Tuple[Dict[str, Any], ...] = tuple(
    dict(zip(("family", "comparison_id", "method_a", "method_b", "metric", "claim_driving"), spec)) for spec in (
        ("static", "riskgreen_vs_geo_SR", "RISK+GREEN-RL", "GEO-RL", "success", True),
        ("static", "riskgreen_vs_geo_MAR", "RISK+GREEN-RL", "GEO-RL", "MAR_static", True),
        ("static", "riskobs_vs_geo_SR", "RISK+OBS-RL", "GEO-RL", "success", False),
        ("static", "riskobs_vs_geo_MAR", "RISK+OBS-RL", "GEO-RL", "MAR_static", False),
        ("static", "riskgreen_vs_fullplanner_SR", "RISK+GREEN-RL", "FULL-RH-A*", "success", True),
        ("static", "riskgreen_vs_fullplanner_MAR", "RISK+GREEN-RL", "FULL-RH-A*", "MAR_static", True),
        ("dynamic", "staticRL_vs_dynplanner_DynOK", "STATIC-RL-DYN", "DYN-RH-A*", "DynOK", False),
        ("dynamic", "dynriskRL_vs_dynplanner_DynOK", "DYN-RISK-RL", "DYN-RH-A*", "DynOK", False),
        ("dynamic", "dynobsRL_vs_dynplanner_SR", "DYN-OBS-RL", "DYN-RH-A*", "success", True),
        ("dynamic", "dynobsRL_vs_dynplanner_MAR", "DYN-OBS-RL", "DYN-RH-A*", "MAR_dynamic", True),
        ("dynamic", "dynobsRL_vs_dynplanner_DynOK", "DYN-OBS-RL", "DYN-RH-A*", "DynOK", True),
        ("vienna", "zero_vs_extended_MAR", "VIENNA-ZERO", "VIENNA-EXT", "MAR_static", True),
        ("vienna", "extended_vs_planner_SR", "VIENNA-EXT", "VIENNA-PLANNER", "success", True),
        ("vienna", "extended_vs_planner_MAR", "VIENNA-EXT", "VIENNA-PLANNER", "MAR_static", True),
        ("vienna", "geo_vs_extended_MAR", "VIENNA-GEO", "VIENNA-EXT", "MAR_static", False),
    )
)


@dataclass
class Run:
    """The per-episode rows of one evaluation and how the tables label it.

    ``train_seed`` groups runs of one learned method over training seeds; ``scope``
    groups methods (for example "static", "dynamic", "vienna"); ``source`` is the CSV
    path written into the tables and ``path`` the file the rows were read from (a
    registry may record other paths than the local ones)."""

    label: str
    rows: List[Row]
    dynamic: bool = False
    train_seed: Optional[float] = None
    scope: str = ""
    regime: str = "primary_no_terminal"
    source: str = ""
    path: Optional[str] = None

    def fields(self) -> Row:
        """The registry columns of the run (as the archive's tables have them)."""
        return {"label": self.label, "csv_path": self.source, "dynamic": bool(self.dynamic),
                "train_seed": None if self.train_seed is None else float(self.train_seed), "scope": self.scope,
                "regime": self.regime}


def _has_column(rows: Sequence[Row], key: str) -> bool:
    return any(key in row for row in rows)


def load_run(path: Union[str, Path], label: Optional[str] = None, *, dynamic: Optional[bool] = None,
             train_seed: Optional[float] = None, scope: str = "", regime: str = "primary_no_terminal",
             exact_floats: bool = False, source: Optional[str] = None) -> Run:
    """A run from a per-episode CSV. The label defaults to the rows' ``method`` when
    they have one value, else the file name; ``dynamic`` defaults to whether the rows
    have ``dynamic_cumulative_risk``. Numbers are read as ``pandas.read_csv`` reads them
    (the archive's tables did), or correctly rounded with ``exact_floats``."""
    path = Path(path)
    rows = read_rows_csv(path, exact_floats=exact_floats)
    if label is None:
        methods = {str(r.get("method")) for r in rows if not is_missing(r.get("method"))}
        label = methods.pop() if len(methods) == 1 else path.stem.replace("_episodes", "")
    if dynamic is None:
        dynamic = _has_column(rows, "dynamic_cumulative_risk")
    return Run(label=str(label), rows=rows, dynamic=bool(dynamic), train_seed=train_seed, scope=scope,
               regime=regime, source=str(path) if source is None else source, path=str(path))


def read_registry(path: Union[str, Path], base: Union[str, Path, None] = None,
                  replace_prefix: Optional[Tuple[str, Union[str, Path]]] = None,
                  exact_floats: bool = False) -> List[Run]:
    """Runs listed in a registry CSV with the columns ``label`` and ``csv_path`` and,
    optionally, ``dynamic``, ``train_seed``, ``scope`` and ``regime`` (the archive's
    ``paper_table_primary_no_terminal_csv_registry.csv``). Relative paths are taken
    from ``base`` (default: the registry's folder); ``replace_prefix=(old, new)``
    reads recorded paths that start with ``old`` from ``new`` instead, keeping the
    recorded path in the tables."""
    path = Path(path)
    base = path.parent if base is None else Path(base)
    runs = []
    for entry in read_rows_csv(path):
        recorded = str(entry["csv_path"])
        local = recorded
        if replace_prefix is not None and recorded.startswith(replace_prefix[0]):
            local = str(Path(replace_prefix[1]) / recorded[len(replace_prefix[0]):])
        file = Path(local) if Path(local).is_absolute() else base / local
        seed = entry.get("train_seed")
        dynamic = entry.get("dynamic")
        runs.append(load_run(
            file, str(entry["label"]), dynamic=None if dynamic is None else as_bool(dynamic),
            train_seed=None if is_missing(seed) else float(seed), scope=str(entry.get("scope") or ""),
            regime=str(entry.get("regime") or "primary_no_terminal"), exact_floats=exact_floats, source=recorded))
    return runs


def _columns(run: Run) -> List[str]:
    names: List[str] = []
    for row in run.rows:
        names.extend(key for key in row if key not in names)
    return names


def _looks_dynamic(names: Iterable[str]) -> bool:
    """Columns that only the dynamic evaluations write (``dynamic_risk_mean``,
    ``dynamic_budget_ok``, ``mission_acceptable_static_only``, ...)."""
    return any(name.startswith("dynamic_") or name == "mission_acceptable_static_only" for name in names)


def missing_columns(run: Run, dynamic: Optional[bool] = None) -> List[str]:
    """The columns mission acceptability reads that the run's rows do not have:
    :data:`ACCEPTABILITY_COLUMNS`, a count of hard violations and, for a dynamic run
    (``dynamic``, default the run's own flag, or rows with columns of the dynamic
    evaluations), ``dynamic_cumulative_risk``. The tables count a missing column as
    within its budget, as the archive's post-processing did."""
    names = set(_columns(run))
    missing = [key for key in ACCEPTABILITY_COLUMNS if key not in names]
    if not names & set(_HARD_COLUMNS):
        missing.append(_HARD_COLUMNS[0])
    dynamic = (run.dynamic or _looks_dynamic(names)) if dynamic is None else dynamic
    if dynamic and "dynamic_cumulative_risk" not in names:
        missing.append("dynamic_cumulative_risk")
    return missing


def _is_boolean(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, (int, np.integer)):
        return int(value) in (0, 1)
    return isinstance(value, str) and value.strip().lower() in _BOOLEAN_TEXT


def _finite(value: Any) -> Optional[float]:
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def check_run(run: Run, dynamic: Optional[bool] = None) -> None:
    """Refuse (``ValueError``) a run whose rows the tables cannot score as they are:
    a column of :func:`missing_columns` is absent (``dynamic_cumulative_risk`` also
    when other columns of the dynamic evaluations are there: a misspelled column
    would drop the dynamic budget); an ``is_success`` is not true or false
    (``True``/``False``, ``1``/``0``, ``yes``/``no``, ...); a successful episode has no
    finite number in a column the acceptability reads, a cargo health outside 0 to 1,
    or a negative time or count; or the rows are scored trajectories without the
    ``infeasible`` column, scored by a development version of 2.1.0 that did not end
    infeasible trajectories (score the trajectories again)."""
    what = f"run {run.label!r}"
    missing = missing_columns(run, dynamic)
    if missing:
        raise ValueError(f"{what} has no column {', '.join(missing)}: mission acceptability needs "
                         f"{', '.join(ACCEPTABILITY_COLUMNS)}, a count of hard violations "
                         f"({' or '.join(_HARD_COLUMNS)}) and, when dynamic, dynamic_cumulative_risk "
                         "(docs/SCORING.md)")
    columns = _columns(run)
    if _stale_trajectories(columns):
        raise ValueError(f"{what}: trajectories scored by a development version of 2.1.0, which did not end the "
                         "trajectories the vehicle could not fly; score them again with ruscuav evaluate "
                         "--trajectories")
    dynamic = (run.dynamic or _looks_dynamic(columns)) if dynamic is None else dynamic
    for row in run.rows:
        where = f"{what}, seed {row.get('seed')}"
        if not _is_boolean(row.get("is_success")):
            raise ValueError(f"{where}: is_success {row.get('is_success')!r} is not true or false")
        if not as_bool(row.get("is_success")):
            continue
        keys = list(ACCEPTABILITY_COLUMNS[1:]) + [_hard_key(row)] + (["dynamic_cumulative_risk"] if dynamic else [])
        values = {key: _finite(row.get(key)) for key in keys}
        empty = [key for key, value in values.items() if value is None]
        if empty:
            raise ValueError(f"{where}: a successful episode without a number in {', '.join(empty)}")
        if not 0.0 <= values["cargo_health"] <= 1.0:
            raise ValueError(f"{where}: cargo health {values['cargo_health']!r} is not from 0 to 1")
        negative = [key for key, value in values.items() if key != "cargo_health" and value < 0.0]
        if negative:
            raise ValueError(f"{where}: negative {', '.join(negative)}")


def _stale_trajectories(columns: Iterable[str]) -> bool:
    """Rows of scored trajectories without the ``infeasible`` column: a development
    version of 2.1.0 scored them, which did not end infeasible trajectories."""
    columns = set(columns)
    return "cargo_health_source" in columns and "infeasible" not in columns


def trajectory_checks(run: Run) -> Optional[Dict[str, Any]]:
    """For a run of scored trajectories (rows with ``cargo_health_source``, as
    ``ruscuav evaluate --trajectories`` writes them), the episodes ended as infeasible,
    by reason, those outside the acceleration limits and the sources of the cargo
    health, and whether a development version of 2.1.0 scored them
    (``earlier_version``: its rows have no ``infeasible`` column, and their counts of
    infeasible episodes mean nothing); None for other runs."""
    rows = [row for row in run.rows if not is_missing(row.get("cargo_health_source"))]
    if not rows:
        return None
    reasons = [str(row.get("infeasible")) for row in rows if not is_missing(row.get("infeasible"))]
    sources = [str(row["cargo_health_source"]) for row in rows]
    return {
        "episodes": len(rows),
        "earlier_version": _stale_trajectories(_columns(run)),
        "infeasible": len(reasons),
        "infeasible_by_reason": {reason: reasons.count(reason) for reason in sorted(set(reasons))},
        "outside_acceleration_limits": sum(not as_bool(row.get("within_acceleration_limits")) for row in rows),
        "cargo_health": {source: sources.count(source) for source in sorted(set(sources))},
    }


def summary_row(run: Run, budgets: Optional[Mapping[str, float]] = None, dt_s: float = DT_S) -> Row:
    """The run's registry columns and its post-hoc summary (without the termination
    counts), as a row of the archive's ``..._all_run_summaries`` table."""
    summary = summarize_posthoc(run.rows, dynamic=run.dynamic, budgets=budgets, dt_s=dt_s)
    return {**run.fields(), **{k: v for k, v in summary.items() if not isinstance(v, dict)}}


def summary_table(runs: Iterable[Run], budgets: Optional[Mapping[str, float]] = None,
                  dt_s: float = DT_S) -> List[Row]:
    return [summary_row(run, budgets, dt_s) for run in runs]


def training_seed_table(summaries: Sequence[Row], metrics: Sequence[str] = SEED_CI_METRICS,
                        group_by: Sequence[str] = ("scope", "label"), method_key: str = "label",
                        min_seeds: int = 2) -> List[Row]:
    """Per group of summary rows with at least ``min_seeds`` training seeds, the
    :func:`~ruscuav_benchmark.stats.training_seed_ci` of each metric. Rows without a
    training seed are left out; groups are sorted, as pandas' ``groupby`` sorts them.
    The output names the group columns as ``group_by`` does, except that
    ``method_key`` is written as ``method``."""
    groups: Dict[Tuple[Any, ...], List[Row]] = {}
    for row in summaries:
        if is_missing(row.get("train_seed")):
            continue
        groups.setdefault(tuple(row.get(k) for k in group_by), []).append(row)
    out = []
    for key in sorted(groups):
        rows = groups[key]
        if len({float(r["train_seed"]) for r in rows}) < int(min_seeds):
            continue
        names = {("method" if k == method_key else k): v for k, v in zip(group_by, key)}
        for metric in metrics:
            if any(metric in r for r in rows):
                out.append({**names, "metric": metric, **training_seed_ci(r.get(metric) for r in rows)})
    return out


def _cause_flags(row: Row, dynamic: bool, budgets: Mapping[str, float]) -> Dict[str, bool]:
    success = as_bool(row.get("is_success"))
    acceptable = dynamic_acceptability(row, budgets) if dynamic else static_acceptability(row, budgets)
    return {
        "not_success": not success,
        "cargo_low": _number(row, "cargo_health", 1.0) < float(budgets["cargo_min"]),
        "road_budget_exceeded": _number(row, "road_time") > float(budgets["road_s"]),
        "ped_budget_exceeded": _number(row, "sidewalk_time") > float(budgets["ped_s"]),
        "building_budget_exceeded": _number(row, "building_time") > float(budgets["building_s"]),
        "sensitive_budget_exceeded": _number(row, "sensitive_time") > float(budgets["sensitive_s"]),
        "hard_violation": _number(row, _hard_key(row)) > 0,
        "dynamic_budget_exceeded": dynamic and _number(row, "dynamic_cumulative_risk") > float(budgets["dynamic"]),
        "unacceptable": not acceptable,
        "successful_but_unacceptable": success and not acceptable,
    }


def cause_rows(run: Run, budgets: Optional[Mapping[str, float]] = None) -> List[Row]:
    """How often each cause occurs among the unacceptable and among the successful
    but unacceptable episodes of the run (the archive's ``cause_decomposition``)."""
    budgets = DEFAULT_BUDGETS if budgets is None else budgets
    flags = [_cause_flags(row, run.dynamic, budgets) for row in run.rows]
    n_unacceptable = sum(f["unacceptable"] for f in flags)
    n_sbu = sum(f["successful_but_unacceptable"] for f in flags)
    out = []
    for cause in CAUSES:
        among_u = sum(f[cause] and f["unacceptable"] for f in flags)
        among_sbu = sum(f[cause] and f["successful_but_unacceptable"] for f in flags)
        out.append({
            "label": run.label, "cause": cause, "n_episodes": len(flags), "n_unacceptable": n_unacceptable,
            "n_successful_but_unacceptable": n_sbu, "count_among_unacceptable": among_u,
            "share_among_unacceptable": among_u / n_unacceptable if n_unacceptable > 0 else float("nan"),
            "count_among_successful_but_unacceptable": among_sbu,
            "share_among_successful_but_unacceptable": among_sbu / n_sbu if n_sbu > 0 else float("nan"),
        })
    return out


def cause_table(runs: Iterable[Run], budgets: Optional[Mapping[str, float]] = None) -> List[Row]:
    return [row for run in runs for row in cause_rows(run, budgets)]


def _rate(values: Sequence[bool]) -> float:
    return pandas_mean(np.asarray(values, dtype=np.float64)) if len(values) else float("nan")


def dynamic_budget_rows(run: Run, betas: Sequence[float] = DYNAMIC_BUDGET_SWEEP,
                        budgets: Optional[Mapping[str, float]] = None) -> List[Row]:
    """Success, static acceptability, the share within the dynamic budget and the
    dynamic acceptability of the run for each dynamic-risk budget ``beta_dyn``."""
    budgets = DEFAULT_BUDGETS if budgets is None else budgets
    success = [as_bool(row.get("is_success")) for row in run.rows]
    static_ok = [static_acceptability(row, budgets) for row in run.rows]
    dyn = [_number(row, "dynamic_cumulative_risk") for row in run.rows]
    out = []
    for beta in betas:
        dyn_ok = [d <= float(beta) for d in dyn]
        full = [s and d for s, d in zip(static_ok, dyn_ok)]
        given = [f for f, s in zip(full, success) if s]
        out.append({
            "label": run.label, "beta_dyn": float(beta), "success_rate": _rate(success),
            "static_acceptability_rate": _rate(static_ok), "dynamic_budget_ok_rate": _rate(dyn_ok),
            "dynamic_mission_acceptability_rate": _rate(full),
            "acceptable_given_success_rate": _rate(given) if given else float("nan"),
            "train_seed": None if run.train_seed is None else float(run.train_seed), "csv_path": run.source,
        })
    return out


def dynamic_budget_table(runs: Iterable[Run], betas: Sequence[float] = DYNAMIC_BUDGET_SWEEP,
                         budgets: Optional[Mapping[str, float]] = None) -> List[Row]:
    return [row for run in runs if run.dynamic for row in dynamic_budget_rows(run, betas, budgets)]


def paired_outcomes(run: Run, budgets: Optional[Mapping[str, float]] = None) -> Dict[str, Dict[Any, bool]]:
    """The binary outcomes of each mission (by seed): ``success``, ``MAR_static`` and,
    when the rows have dynamic risk, ``DynOK`` (within the dynamic budget) and
    ``MAR_dynamic``."""
    budgets = DEFAULT_BUDGETS if budgets is None else budgets
    out: Dict[str, Dict[Any, bool]] = {"success": {}, "MAR_static": {}}
    dynamic = _has_column(run.rows, "dynamic_cumulative_risk")
    if dynamic:
        out["DynOK"], out["MAR_dynamic"] = {}, {}
    for row in run.rows:
        seed = row.get("seed")
        if seed in out["success"]:
            raise ValueError(f"run {run.label!r} has several episodes of seed {seed}")
        out["success"][seed] = as_bool(row.get("is_success"))
        out["MAR_static"][seed] = static_acceptability(row, budgets)
        if dynamic:
            out["DynOK"][seed] = _number(row, "dynamic_cumulative_risk") <= float(budgets["dynamic"])
            out["MAR_dynamic"][seed] = dynamic_acceptability(row, budgets)
    return out


def paired_test(run_a: Run, run_b: Run, metric: str, budgets: Optional[Mapping[str, float]] = None) -> Row:
    """Exact McNemar test of ``metric`` (one of :data:`PAIRED_METRICS`) between two
    runs on their common missions; the rates are those of each whole run."""
    a, b = paired_outcomes(run_a, budgets), paired_outcomes(run_b, budgets)
    for name, outcomes in ((run_a.label, a), (run_b.label, b)):
        if metric not in outcomes:
            raise ValueError(f"{metric!r} is not available for run {name!r}; available: {list(outcomes)}")
    common = [seed for seed in a[metric] if seed in b[metric]]
    n_ab = sum(a[metric][s] and not b[metric][s] for s in common)
    n_ba = sum(b[metric][s] and not a[metric][s] for s in common)
    rate_a, rate_b = _rate(list(a[metric].values())), _rate(list(b[metric].values()))
    return {"n_paired": len(common), "n_ab": n_ab, "n_ba": n_ba, "rate_a": rate_a, "rate_b": rate_b,
            "effect_a_minus_b_pp": 100.0 * (rate_a - rate_b), "p_raw": mcnemar_exact_p(n_ab, n_ba)}


def holm_table(comparisons: Iterable[Mapping[str, Any]], runs: Mapping[str, Run],
               budgets: Optional[Mapping[str, float]] = None) -> List[Row]:
    """Paired tests of declared comparisons (rows with ``family``, ``comparison_id``,
    ``method_a``, ``method_b``, ``metric`` and optionally ``claim_driving``; methods are
    keys of ``runs``) with Holm-adjusted p-values within each family and the decision
    at the 0.05 level (the archive's ``compute_holm_families``)."""
    rows = []
    for spec in comparisons:
        missing = [m for m in (spec["method_a"], spec["method_b"]) if m not in runs]
        if missing:
            raise KeyError(f"comparison {spec.get('comparison_id')!r}: no run labelled {missing[0]!r}")
        rows.append({**dict(spec), **paired_test(runs[spec["method_a"]], runs[spec["method_b"]], str(spec["metric"]),
                                                 budgets)})
    families: Dict[Any, List[int]] = {}
    for index, row in enumerate(rows):
        families.setdefault(row["family"], []).append(index)
    for indices in families.values():
        adjusted = holm_adjust([rows[i]["p_raw"] for i in indices])
        for i, p in zip(indices, adjusted):
            rows[i]["family_size"] = len(indices)
            rows[i]["p_holm"] = float(p)
    for row in rows:
        row["significant_holm_005"] = bool(row["p_holm"] < 0.05)
    # the archive's column order: the comparison, the counts and rates, then family size and Holm
    # (comparisons from several sources may have different columns: all of them, in order of appearance)
    order: List[str] = []
    for row in rows:
        order.extend(key for key in row if key not in order)
    for key in ("family_size", "p_holm", "significant_holm_005"):
        if key in order:
            order.remove(key)
            order.append(key)
    return [{k: row.get(k) for k in order} for row in rows]


@dataclass
class ScoreTables:
    """The tables of :func:`score_runs`, as lists of rows."""

    summaries: List[Row] = field(default_factory=list)
    training_seed_ci: List[Row] = field(default_factory=list)
    causes: List[Row] = field(default_factory=list)
    dynamic_budget: List[Row] = field(default_factory=list)
    paired_tests: List[Row] = field(default_factory=list)

    FILES = {
        "summaries": "run_summaries.csv",
        "training_seed_ci": "training_seed_ci.csv",
        "causes": "non_acceptability_causes.csv",
        "dynamic_budget": "dynamic_budget_sensitivity.csv",
        "paired_tests": "paired_tests_holm.csv",
    }

    def write(self, directory: Union[str, Path]) -> Dict[str, Path]:
        """Write the non-empty tables as CSV files into ``directory``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, filename in self.FILES.items():
            rows = getattr(self, name)
            if rows:
                written[name] = write_rows_csv(directory / filename, rows)
        return written


def score_runs(runs: Sequence[Run], comparisons: Optional[Iterable[Mapping[str, Any]]] = None,
               budgets: Optional[Mapping[str, float]] = None, dt_s: float = DT_S,
               seed_ci_metrics: Sequence[str] = SEED_CI_METRICS) -> ScoreTables:
    """All tables for a list of runs: summaries, training-seed intervals (methods with
    several seeds, by scope, regime and label), causes of non-acceptability, the
    dynamic-budget sweep (dynamic runs) and, with ``comparisons``, the paired tests
    with Holm's correction. Methods in ``comparisons`` are run labels; a label with
    several runs (training seeds) is ambiguous there and refused."""
    summaries = summary_table(runs, budgets, dt_s)
    seed_table = training_seed_table(summaries, seed_ci_metrics, group_by=("scope", "regime", "label"))
    tables = ScoreTables(summaries=summaries, training_seed_ci=seed_table,
                         causes=cause_table(runs, budgets), dynamic_budget=dynamic_budget_table(runs, budgets=budgets))
    if comparisons is not None:
        comparisons = list(comparisons)
        by_label: Dict[str, Run] = {}
        repeated = set()
        for run in runs:
            if run.label in by_label:
                repeated.add(run.label)
            by_label[run.label] = run
        used = {spec[k] for spec in comparisons for k in ("method_a", "method_b")}
        if used & repeated:
            raise ValueError(f"labels {sorted(used & repeated)} name several runs; give each compared run its "
                             "own label")
        tables.paired_tests = holm_table(comparisons, by_label, budgets)
    return tables


def read_comparisons(path: Union[str, Path]) -> List[Row]:
    """Declared comparisons from a CSV with the columns of :data:`PAPER_HOLM_COMPARISONS`."""
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.DictReader(f) if any(str(v or "").strip() for v in row.values())]
    for row in rows:
        missing = [k for k in ("family", "comparison_id", "method_a", "method_b", "metric") if not row.get(k)]
        if missing:
            raise ValueError(f"{path}: comparison {row} has no {missing[0]!r}")
        if row.get("metric") not in PAIRED_METRICS:
            raise ValueError(f"{path}: metric {row.get('metric')!r}; expected one of {list(PAIRED_METRICS)}")
        if "claim_driving" in row:
            row["claim_driving"] = as_bool(row["claim_driving"])
    return rows

