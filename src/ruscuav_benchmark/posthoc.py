"""The post-hoc step of the archived primary evaluations.

The v2.0.0 primary (no-terminal) evaluations of the learned policies and of the
dynamic planner recomputed mission acceptability from each finished episode row,
with the budgets below, and added derived metrics: risk per step, second and metre,
and the fraction of the flight time over each layer. Their summaries add the means
of the episode metrics over all, successful and acceptable episodes. These are the
notebook's ``recompute_static_acceptability``, ``add_derived_metrics`` and
``summarize_episode_dataframe``, row by row and with the same arithmetic, for rows
of Python numbers as the package builds them (pandas would keep float32 columns in
float32).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

from .tables import Row, pandas_max, pandas_mean, pandas_quantile, termination_counts

__all__ = [
    "DT_S", "DEFAULT_BUDGETS", "as_bool", "static_acceptability", "dynamic_acceptability", "derived_metrics",
    "with_derived_metrics", "summarize_posthoc",
]

#: Duration of an environment step (s).
DT_S = 0.5
#: Mission-acceptability budgets of the paper: minimum cargo health, maximum seconds
#: over roads, pedestrian ways, buildings and sensitive zones, maximum cumulative
#: dynamic risk.
DEFAULT_BUDGETS: Dict[str, float] = {
    "cargo_min": 0.80,
    "road_s": 20.0,
    "ped_s": 5.0,
    "building_s": 15.0,
    "sensitive_s": 0.0,
    "dynamic": 10.0,
}

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}


def as_bool(value: Any) -> bool:
    """A cell as the notebook's ``as_bool_series`` reads it."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in _TRUE_STRINGS


def _number(row: Row, key: str, default: float = 0.0) -> float:
    """``num_col``: the value as a number; missing, empty or non-numeric gives ``default``."""
    value = row.get(key)
    if value is None:
        return float(default)
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if math.isnan(number) else number


def _hard_key(row: Row) -> str:
    return "urban_hard_violation_steps" if "urban_hard_violation_steps" in row else "hard_violation_steps"


def static_acceptability(row: Row, budgets: Mapping[str, float] = DEFAULT_BUDGETS) -> bool:
    """Success, cargo health and the time budgets, and no hard violation."""
    return bool(
        as_bool(row.get("is_success"))
        and _number(row, "cargo_health", 1.0) >= float(budgets["cargo_min"])
        and _number(row, "road_time") <= float(budgets["road_s"])
        and _number(row, "sidewalk_time") <= float(budgets["ped_s"])
        and _number(row, "building_time") <= float(budgets["building_s"])
        and _number(row, "sensitive_time") <= float(budgets["sensitive_s"])
        and _number(row, _hard_key(row)) <= 0
    )


def dynamic_acceptability(row: Row, budgets: Mapping[str, float] = DEFAULT_BUDGETS) -> bool:
    """:func:`static_acceptability` and the dynamic-risk budget."""
    return static_acceptability(row, budgets) and _number(row, "dynamic_cumulative_risk") <= float(budgets["dynamic"])


def _div(a: float, b: float) -> float:
    """a / b, with b = 0 treated as missing (NaN), as the notebook did."""
    return float("nan") if b == 0.0 or math.isnan(b) else a / b


def derived_metrics(row: Row, dt_s: float = DT_S, budgets: Mapping[str, float] = DEFAULT_BUDGETS) -> Dict[str, Any]:
    """The columns ``add_derived_metrics`` sets on a row, in its order."""
    out: Dict[str, Any] = {"mission_acceptable_posthoc_static": static_acceptability(row, budgets)}
    dynamic = "dynamic_cumulative_risk" in row
    if dynamic:
        out["dynamic_budget_ok_posthoc"] = bool(_number(row, "dynamic_cumulative_risk") <= float(budgets["dynamic"]))
        out["mission_acceptable_posthoc_dynamic"] = dynamic_acceptability(row, budgets)
    steps = _number(row, "steps")
    duration = steps * float(dt_s)
    out["episode_duration_s"] = duration
    risk = _number(row, "urban_cumulative_risk") if "urban_cumulative_risk" in row else _number(row, "mean_cumulative_risk")
    out["urban_risk_per_step"] = _div(risk, steps)
    out["urban_risk_per_s"] = _div(risk, duration)
    if dynamic:
        dyn = _number(row, "dynamic_cumulative_risk")
        out["dynamic_risk_per_step"] = _div(dyn, steps)
        out["dynamic_risk_per_s"] = _div(dyn, duration)
    for key, label in (("road_time", "road_fraction"), ("sidewalk_time", "ped_fraction"),
                       ("building_time", "building_fraction"), ("green_time", "green_fraction"),
                       ("canopy_time", "canopy_fraction"), ("sensitive_time", "sensitive_fraction")):
        out[label] = _div(_number(row, key), duration)
    if "path_length_xy" in row:
        distance = _number(row, "path_length_xy")
        out["urban_risk_per_m_xy"] = _div(risk, distance)
        for key, label in (("road_time", "road_time_per_m_xy"), ("sidewalk_time", "ped_time_per_m_xy"),
                           ("building_time", "building_time_per_m_xy")):
            out[label] = _div(_number(row, key), distance)
    return out


def with_derived_metrics(rows: List[Row], dt_s: float = DT_S,
                         budgets: Mapping[str, float] = DEFAULT_BUDGETS) -> List[Row]:
    """``add_derived_metrics`` on a table of rows: every row gets every column of the
    table (None where it had none, an empty cell in pandas), followed by the derived
    columns (existing ones are replaced in place)."""
    columns: List[str] = []
    for row in rows:
        columns.extend(key for key in row if key not in columns)
    out = []
    for row in rows:
        new = {key: row.get(key) for key in columns}
        new.update(derived_metrics(new, dt_s, budgets))
        out.append(new)
    return out


_SCALAR_METRICS = (
    "steps", "dist_xy", "cargo_health", "urban_cumulative_risk", "urban_risk_per_step", "urban_risk_per_s",
    "path_length_xy", "urban_risk_per_m_xy", "road_time", "sidewalk_time", "building_time", "green_time",
    "canopy_time", "sensitive_time", "road_fraction", "ped_fraction", "building_fraction", "green_fraction",
)
_DYNAMIC_METRICS = ("dynamic_cumulative_risk", "dynamic_risk_per_step", "dynamic_risk_per_s")


def _numeric_column(rows: List[Row], key: str) -> np.ndarray:
    """``pd.to_numeric(column, errors="coerce")``."""
    values = []
    for row in rows:
        value = row.get(key)
        try:
            values.append(np.nan if value is None else float(value))
        except (TypeError, ValueError):
            values.append(np.nan)
    return np.asarray(values, dtype=np.float64)


def summarize_posthoc(rows: List[Row], dynamic: bool = False, budgets: Optional[Mapping[str, float]] = None,
                      dt_s: float = DT_S) -> Dict[str, Any]:
    """``summarize_episode_dataframe``: rates and conditional means of the post-hoc
    evaluation (mission acceptability from the post-hoc static or, with ``dynamic``,
    dynamic check)."""
    budgets = DEFAULT_BUDGETS if budgets is None else budgets
    d = with_derived_metrics(rows, dt_s=dt_s, budgets=budgets)
    success = np.array([as_bool(r.get("is_success")) for r in d], dtype=bool)
    acc_key = "mission_acceptable_posthoc_dynamic" if dynamic else "mission_acceptable_posthoc_static"
    acceptable = np.array([as_bool(r.get(acc_key)) for r in d], dtype=bool)
    n_success = int(success.sum())
    result: Dict[str, Any] = {
        "n_episodes": len(d),
        "success_rate": pandas_mean(success.astype(np.float64)) if d else float("nan"),
        "mission_acceptability_rate": pandas_mean(acceptable.astype(np.float64)) if d else float("nan"),
        "acceptable_given_success_rate": (
            pandas_mean(acceptable[success].astype(np.float64)) if n_success > 0 else float("nan")
        ),
    }
    columns = {key for r in d for key in r}
    metrics = _SCALAR_METRICS + (_DYNAMIC_METRICS if dynamic else ())
    for metric in metrics:
        if metric not in columns:
            continue
        values = _numeric_column(d, metric)
        result[f"mean_{metric}"] = pandas_mean(values)
        result[f"mean_{metric}_given_success"] = pandas_mean(values[success]) if n_success > 0 else float("nan")
        result[f"mean_{metric}_given_acceptable"] = (
            pandas_mean(values[acceptable]) if int(acceptable.sum()) > 0 else float("nan")
        )
    if dynamic:
        result["dynamic_budget_ok_rate"] = pandas_mean(
            np.array([as_bool(r.get("dynamic_budget_ok_posthoc")) for r in d], dtype=np.float64))
        dyn = _numeric_column(d, "dynamic_cumulative_risk")
        for q in (0.50, 0.75, 0.90, 0.95, 0.99):
            result[f"dynamic_risk_p{int(q * 100)}"] = pandas_quantile(dyn, q)
        result["dynamic_risk_max"] = pandas_max(dyn)
    result["termination_counts"] = termination_counts(d)
    return result

