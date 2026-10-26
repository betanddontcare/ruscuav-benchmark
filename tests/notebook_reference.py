"""The archived notebook's pandas code for episode tables, copied verbatim, as the
reference for the package's row-wise ports (tests only).

From code/submission_repair/RUSCUAV_submission_repair_cells.py: ``as_bool_series``,
``num_col``, ``recompute_static_acceptability``, ``recompute_dynamic_acceptability``,
``add_derived_metrics``, ``summarize_episode_dataframe``, ``training_seed_ci``,
``non_acceptability_flags``, ``cause_decomposition``, ``dynamic_budget_sensitivity``,
``paired_binary_counts``, ``paired_exact_p``, ``holm_adjust`` and
``compute_holm_families``. From src/ruscuav_benchmark/legacy/definitions.py:
``summarize_m11_dynamic_eval_df``, ``summarize_m12c_real_policy_eval_df`` and
``ms_bootstrap_ci``. ``training_seed_ci_all_scopes`` is the aggregation loop of the
one-click runner's ``stage_postprocess``
(code/submission_repair/RUSCUAV_submission_repair_one_click_runner_v3.py).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

try:  # the statistics need SciPy; the other functions do not
    from scipy.stats import binomtest, t as student_t
except ImportError:  # pragma: no cover
    binomtest = student_t = None

DT_S = 0.5
DEFAULT_BUDGETS = {
    "cargo_min": 0.80,
    "road_s": 20.0,
    "ped_s": 5.0,
    "building_s": 15.0,
    "sensitive_s": 0.0,
    "dynamic": 10.0,
}


def as_bool_series(series):
    if str(series.dtype) == "bool":
        return series.astype(bool)
    return series.map(
        lambda x: bool(x)
        if isinstance(x, (bool, np.bool_))
        else str(x).strip().lower() in {"true", "1", "yes", "y", "t"}
    ).astype(bool)


def num_col(df, name, default=0.0):
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)


def recompute_static_acceptability(df, budgets=DEFAULT_BUDGETS):
    success = as_bool_series(df["is_success"])
    hard_col = (
        "urban_hard_violation_steps"
        if "urban_hard_violation_steps" in df.columns
        else "hard_violation_steps"
    )
    return (
        success
        & (num_col(df, "cargo_health", 1.0) >= float(budgets["cargo_min"]))
        & (num_col(df, "road_time") <= float(budgets["road_s"]))
        & (num_col(df, "sidewalk_time") <= float(budgets["ped_s"]))
        & (num_col(df, "building_time") <= float(budgets["building_s"]))
        & (num_col(df, "sensitive_time") <= float(budgets["sensitive_s"]))
        & (num_col(df, hard_col) <= 0)
    )


def recompute_dynamic_acceptability(df, budgets=DEFAULT_BUDGETS):
    return recompute_static_acceptability(df, budgets) & (
        num_col(df, "dynamic_cumulative_risk") <= float(budgets["dynamic"])
    )


def add_derived_metrics(df, dt_s=DT_S, budgets=DEFAULT_BUDGETS):
    out = df.copy()
    out["mission_acceptable_posthoc_static"] = recompute_static_acceptability(out, budgets)
    if "dynamic_cumulative_risk" in out.columns:
        out["dynamic_budget_ok_posthoc"] = (
            num_col(out, "dynamic_cumulative_risk") <= float(budgets["dynamic"])
        )
        out["mission_acceptable_posthoc_dynamic"] = recompute_dynamic_acceptability(out, budgets)

    out["episode_duration_s"] = num_col(out, "steps") * float(dt_s)
    duration = out["episode_duration_s"].replace(0.0, np.nan)
    steps = num_col(out, "steps").replace(0.0, np.nan)

    if "urban_cumulative_risk" in out.columns:
        risk = num_col(out, "urban_cumulative_risk")
    else:
        risk = num_col(out, "mean_cumulative_risk")
    out["urban_risk_per_step"] = risk / steps
    out["urban_risk_per_s"] = risk / duration

    if "dynamic_cumulative_risk" in out.columns:
        dyn = num_col(out, "dynamic_cumulative_risk")
        out["dynamic_risk_per_step"] = dyn / steps
        out["dynamic_risk_per_s"] = dyn / duration

    for col, label in [
        ("road_time", "road_fraction"),
        ("sidewalk_time", "ped_fraction"),
        ("building_time", "building_fraction"),
        ("green_time", "green_fraction"),
        ("canopy_time", "canopy_fraction"),
        ("sensitive_time", "sensitive_fraction"),
    ]:
        out[label] = num_col(out, col) / duration

    if "path_length_xy" in out.columns:
        distance = num_col(out, "path_length_xy").replace(0.0, np.nan)
        out["urban_risk_per_m_xy"] = risk / distance
        for col, label in [
            ("road_time", "road_time_per_m_xy"),
            ("sidewalk_time", "ped_time_per_m_xy"),
            ("building_time", "building_time_per_m_xy"),
        ]:
            out[label] = num_col(out, col) / distance

    return out


def summarize_episode_dataframe(df, dynamic=False):
    d = add_derived_metrics(df)
    success = as_bool_series(d["is_success"])
    acc_col = "mission_acceptable_posthoc_dynamic" if dynamic else "mission_acceptable_posthoc_static"
    acceptable = as_bool_series(d[acc_col])

    result = {
        "n_episodes": int(len(d)),
        "success_rate": float(success.mean()),
        "mission_acceptability_rate": float(acceptable.mean()),
        "acceptable_given_success_rate": (
            float(acceptable[success].mean()) if int(success.sum()) > 0 else np.nan
        ),
    }

    scalar_metrics = [
        "steps",
        "dist_xy",
        "cargo_health",
        "urban_cumulative_risk",
        "urban_risk_per_step",
        "urban_risk_per_s",
        "path_length_xy",
        "urban_risk_per_m_xy",
        "road_time",
        "sidewalk_time",
        "building_time",
        "green_time",
        "canopy_time",
        "sensitive_time",
        "road_fraction",
        "ped_fraction",
        "building_fraction",
        "green_fraction",
    ]
    if dynamic:
        scalar_metrics.extend(
            [
                "dynamic_cumulative_risk",
                "dynamic_risk_per_step",
                "dynamic_risk_per_s",
            ]
        )

    for metric in scalar_metrics:
        if metric not in d.columns:
            continue
        values = pd.to_numeric(d[metric], errors="coerce")
        result[f"mean_{metric}"] = float(values.mean())
        result[f"mean_{metric}_given_success"] = (
            float(values[success].mean()) if int(success.sum()) > 0 else np.nan
        )
        result[f"mean_{metric}_given_acceptable"] = (
            float(values[acceptable].mean()) if int(acceptable.sum()) > 0 else np.nan
        )

    if dynamic:
        result["dynamic_budget_ok_rate"] = float(
            as_bool_series(d["dynamic_budget_ok_posthoc"]).mean()
        )
        dyn = pd.to_numeric(d["dynamic_cumulative_risk"], errors="coerce")
        for q in [0.50, 0.75, 0.90, 0.95, 0.99]:
            result[f"dynamic_risk_p{int(q * 100)}"] = float(dyn.quantile(q))
        result["dynamic_risk_max"] = float(dyn.max())

    result["termination_counts"] = {
        str(k): int(v)
        for k, v in d["termination"].fillna("None").value_counts().to_dict().items()
    }
    return result


def summarize_m11_dynamic_eval_df(df):
    if len(df) == 0:
        return {"n_episodes": 0}

    summary = {
        "n_episodes": int(len(df)),
        "success_rate": float(df["is_success"].mean()),
        "mission_acceptability_rate": float(df["mission_acceptable"].mean()),
        "static_only_acceptability_rate": float(df["mission_acceptable_static_only"].mean()) if "mission_acceptable_static_only" in df else np.nan,

        "mean_reward": float(df["total_reward"].mean()),
        "std_reward": float(df["total_reward"].std(ddof=0)),

        "mean_steps": float(df["steps"].mean()),
        "mean_dist_xy": float(df["dist_xy"].mean()),
        "median_dist_xy": float(df["dist_xy"].median()),
        "mean_cargo_health": float(df["cargo_health"].mean()),

        "mean_urban_risk": float(df["urban_mean_risk"].mean()),
        "mean_urban_cumulative_risk": float(df["urban_cumulative_risk"].mean()),

        "mean_dynamic_risk": float(df["dynamic_risk_mean"].mean()),
        "mean_dynamic_risk_max": float(df["dynamic_risk_max"].mean()),
        "mean_dynamic_cumulative_risk": float(df["dynamic_cumulative_risk"].mean()),
        "dynamic_budget_ok_rate": float(df["dynamic_budget_ok"].mean()),

        "mean_road_time": float(df["road_time"].mean()),
        "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
        "mean_green_time": float(df["green_time"].mean()),
        "mean_sensitive_time": float(df["sensitive_time"].mean()),
        "mean_hard_violation_steps": float(df["urban_hard_violation_steps"].mean()),

        "termination_counts": {
            str(k): int(v)
            for k, v in df["termination"].fillna("None").value_counts().to_dict().items()
        },
    }

    success_df = df[df["is_success"] == True]  # noqa: E712
    if len(success_df) > 0:
        summary["acceptable_given_success_rate"] = float(success_df["mission_acceptable"].mean())
    else:
        summary["acceptable_given_success_rate"] = 0.0

    return summary


def summarize_m12c_real_policy_eval_df(df):
    if len(df) == 0:
        return {"n_episodes": 0}

    summary = {
        "n_episodes": int(len(df)),
        "success_rate": float(df["is_success"].mean()),
        "mission_acceptability_rate": float(df["mission_acceptable"].mean()),

        "mean_reward": float(df["total_reward"].mean()),
        "std_reward": float(df["total_reward"].std(ddof=0)),

        "mean_steps": float(df["steps"].mean()),
        "mean_dist_xy": float(df["dist_xy"].mean()),
        "median_dist_xy": float(df["dist_xy"].median()),

        "mean_cargo_health": float(df["cargo_health"].mean()),

        "mean_urban_risk": float(df["urban_mean_risk"].mean()),
        "mean_urban_max_risk": float(df["urban_max_risk"].mean()),
        "mean_cumulative_risk": float(df["urban_cumulative_risk"].mean()),

        "mean_road_time": float(df["road_time"].mean()),
        "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
        "mean_building_time": float(df["building_time"].mean()),
        "mean_green_time": float(df["green_time"].mean()),
        "mean_canopy_time": float(df["canopy_time"].mean()),
        "mean_sensitive_time": float(df["sensitive_time"].mean()),

        "mean_hard_violation_steps": float(df["urban_hard_violation_steps"].mean()),

        "termination_counts": {
            str(k): int(v)
            for k, v in df["termination"].fillna("None").value_counts().to_dict().items()
        },
    }

    success_df = df[df["is_success"] == True]  # noqa: E712

    if len(success_df) > 0:
        summary["acceptable_given_success_rate"] = float(success_df["mission_acceptable"].mean())
    else:
        summary["acceptable_given_success_rate"] = 0.0

    return summary


def training_seed_ci(values: Iterable[float]) -> dict[str, float]:
    vals = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(float)
    n = int(len(vals))
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(vals.mean())
    if n == 1:
        return {"n": 1, "mean": mean, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    std = float(vals.std(ddof=1))
    half = float(student_t.ppf(0.975, n - 1) * std / math.sqrt(n))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


CAUSE_ORDER = [
    "not_success",
    "cargo_low",
    "road_budget_exceeded",
    "ped_budget_exceeded",
    "building_budget_exceeded",
    "sensitive_budget_exceeded",
    "hard_violation",
    "dynamic_budget_exceeded",
]


def non_acceptability_flags(df: pd.DataFrame, dynamic: bool = False) -> pd.DataFrame:
    d = add_derived_metrics(df)
    flags = pd.DataFrame(index=d.index)
    success = as_bool_series(d["is_success"])
    flags["not_success"] = ~success
    flags["cargo_low"] = num_col(d, "cargo_health", 1.0) < DEFAULT_BUDGETS["cargo_min"]
    flags["road_budget_exceeded"] = num_col(d, "road_time") > DEFAULT_BUDGETS["road_s"]
    flags["ped_budget_exceeded"] = num_col(d, "sidewalk_time") > DEFAULT_BUDGETS["ped_s"]
    flags["building_budget_exceeded"] = num_col(d, "building_time") > DEFAULT_BUDGETS["building_s"]
    flags["sensitive_budget_exceeded"] = num_col(d, "sensitive_time") > DEFAULT_BUDGETS["sensitive_s"]
    hard_col = "urban_hard_violation_steps" if "urban_hard_violation_steps" in d.columns else "hard_violation_steps"
    flags["hard_violation"] = num_col(d, hard_col) > 0
    flags["dynamic_budget_exceeded"] = (
        num_col(d, "dynamic_cumulative_risk") > DEFAULT_BUDGETS["dynamic"]
        if dynamic
        else False
    )
    acceptable = recompute_dynamic_acceptability(d) if dynamic else recompute_static_acceptability(d)
    flags["unacceptable"] = ~acceptable
    flags["successful_but_unacceptable"] = success & (~acceptable)
    return flags


def cause_decomposition(df: pd.DataFrame, label: str, dynamic: bool = False) -> pd.DataFrame:
    flags = non_acceptability_flags(df, dynamic=dynamic)
    n_unacceptable = int(flags["unacceptable"].sum())
    n_sbu = int(flags["successful_but_unacceptable"].sum())
    rows = []
    for cause in CAUSE_ORDER:
        if cause not in flags.columns:
            continue
        rows.append(
            {
                "label": label,
                "cause": cause,
                "n_episodes": int(len(flags)),
                "n_unacceptable": n_unacceptable,
                "n_successful_but_unacceptable": n_sbu,
                "count_among_unacceptable": int((flags[cause] & flags["unacceptable"]).sum()),
                "share_among_unacceptable": (
                    float((flags[cause] & flags["unacceptable"]).sum() / n_unacceptable)
                    if n_unacceptable > 0
                    else np.nan
                ),
                "count_among_successful_but_unacceptable": int(
                    (flags[cause] & flags["successful_but_unacceptable"]).sum()
                ),
                "share_among_successful_but_unacceptable": (
                    float(
                        (flags[cause] & flags["successful_but_unacceptable"]).sum()
                        / n_sbu
                    )
                    if n_sbu > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


DYNAMIC_BUDGET_SWEEP = [2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 35.0]


def dynamic_budget_sensitivity(
    df: pd.DataFrame,
    label: str,
    budgets: Iterable[float] = DYNAMIC_BUDGET_SWEEP,
) -> pd.DataFrame:
    d = add_derived_metrics(df)
    static_ok = recompute_static_acceptability(d)
    success = as_bool_series(d["is_success"])
    dyn = num_col(d, "dynamic_cumulative_risk")
    rows = []
    for beta in budgets:
        dyn_ok = dyn <= float(beta)
        full = static_ok & dyn_ok
        rows.append(
            {
                "label": label,
                "beta_dyn": float(beta),
                "success_rate": float(success.mean()),
                "static_acceptability_rate": float(static_ok.mean()),
                "dynamic_budget_ok_rate": float(dyn_ok.mean()),
                "dynamic_mission_acceptability_rate": float(full.mean()),
                "acceptable_given_success_rate": (
                    float(full[success].mean()) if int(success.sum()) > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_binary_counts(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str) -> dict[str, int]:
    a = df_a[["seed", metric]].copy()
    b = df_b[["seed", metric]].copy()
    a[metric] = as_bool_series(a[metric])
    b[metric] = as_bool_series(b[metric])
    paired = a.merge(b, on="seed", suffixes=("_a", "_b"), how="inner")
    n_ab = int((paired[f"{metric}_a"] & ~paired[f"{metric}_b"]).sum())
    n_ba = int((~paired[f"{metric}_a"] & paired[f"{metric}_b"]).sum())
    return {"n_paired": int(len(paired)), "n_ab": n_ab, "n_ba": n_ba}


def paired_exact_p(n_ab: int, n_ba: int) -> float:
    discordant = int(n_ab + n_ba)
    if discordant == 0:
        return 1.0
    return float(binomtest(k=min(n_ab, n_ba), n=discordant, p=0.5, alternative="two-sided").pvalue)


def holm_adjust(raw_p: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(raw_p), dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted_sorted[rank] = running
    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def compute_holm_families(
    comparison_spec: pd.DataFrame,
    csv_registry: Mapping[str, str | Path],
) -> pd.DataFrame:
    """
    comparison_spec columns:
        family, comparison_id, method_a, method_b, metric, claim_driving
    csv_registry maps method labels to per-episode CSVs.
    """
    rows = []
    cache: dict[str, pd.DataFrame] = {}

    for _, spec in comparison_spec.iterrows():
        a_label, b_label = str(spec["method_a"]), str(spec["method_b"])
        for label in [a_label, b_label]:
            if label not in cache:
                p = Path(csv_registry[label])
                if not p.exists():
                    raise FileNotFoundError(p)
                df = add_derived_metrics(pd.read_csv(p))
                # Ensure canonical binary metric aliases.
                df["success"] = as_bool_series(df["is_success"])
                df["MAR_static"] = recompute_static_acceptability(df)
                if "dynamic_cumulative_risk" in df.columns:
                    df["DynOK"] = num_col(df, "dynamic_cumulative_risk") <= DEFAULT_BUDGETS["dynamic"]
                    df["MAR_dynamic"] = recompute_dynamic_acceptability(df)
                cache[label] = df

        metric = str(spec["metric"])
        counts = paired_binary_counts(cache[a_label], cache[b_label], metric)
        raw_p = paired_exact_p(counts["n_ab"], counts["n_ba"])
        mean_a = float(as_bool_series(cache[a_label][metric]).mean())
        mean_b = float(as_bool_series(cache[b_label][metric]).mean())
        rows.append(
            {
                **spec.to_dict(),
                **counts,
                "rate_a": mean_a,
                "rate_b": mean_b,
                "effect_a_minus_b_pp": 100.0 * (mean_a - mean_b),
                "p_raw": raw_p,
            }
        )

    result = pd.DataFrame(rows)
    result["family_size"] = result.groupby("family")["comparison_id"].transform("count")
    result["p_holm"] = np.nan
    for family, idx in result.groupby("family").groups.items():
        result.loc[idx, "p_holm"] = holm_adjust(result.loc[idx, "p_raw"])
    result["significant_holm_005"] = result["p_holm"] < 0.05
    return result


def ms_bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=12345):
    """
    Episode-level bootstrap CI. This is secondary uncertainty only.
    """
    arr = np.asarray(values)

    if arr.dtype.kind in {"b"}:
        arr = arr.astype(float)
    else:
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().to_numpy(dtype=float)

    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(int(seed))
    n = len(arr)
    boots = np.empty(int(n_boot), dtype=float)

    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(stat_fn(arr[idx]))

    alpha = 1.0 - float(ci)
    estimate = float(stat_fn(arr))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))

    return estimate, lo, hi


def training_seed_ci_all_scopes(summary_df):
    agg_rows = []
    metrics = ["success_rate", "mission_acceptability_rate", "acceptable_given_success_rate"]
    for (scope, label), group in summary_df.dropna(subset=["train_seed"]).groupby(["scope", "label"]):
        if group["train_seed"].nunique() < 2:
            continue
        for metric in metrics:
            if metric in group.columns:
                agg_rows.append({"scope": scope, "method": label, "metric": metric, **training_seed_ci(group[metric])})
    return pd.DataFrame(agg_rows)
