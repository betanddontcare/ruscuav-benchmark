"""The archived notebook's pandas code for episode tables, copied verbatim, as the
reference for the package's row-wise ports (tests only).

From code/submission_repair/RUSCUAV_submission_repair_cells.py: ``as_bool_series``,
``num_col``, ``recompute_static_acceptability``, ``recompute_dynamic_acceptability``,
``add_derived_metrics`` and ``summarize_episode_dataframe``. From
src/ruscuav_benchmark/legacy/definitions.py: ``summarize_m11_dynamic_eval_df`` and
``summarize_m12c_real_policy_eval_df``.
"""

import numpy as np
import pandas as pd

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
