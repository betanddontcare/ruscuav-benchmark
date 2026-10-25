# %% [markdown]
# RUSCUAV submission-repair package
# ==================================
#
# Purpose
# -------
# This file contains Colab-ready cells for the remaining submission-blocking fixes:
# 1. exact run provenance and checksums;
# 2. primary no-terminal reevaluation, with strict Stage C retained as a secondary
#    operational-stopping regime;
# 3. scenario-aligned Warsaw planner/policy evaluation;
# 4. conditional and normalized risk metrics;
# 5. decomposition of non-acceptability causes;
# 6. dynamic-budget (beta_dyn) sensitivity;
# 7. complete, reproducible Holm families;
# 8. validation/regeneration of the Warsaw trajectory figure;
# 9. release-readiness manifest for Zenodo.
#
# Where to run
# ------------
# Run these cells after all class/function-definition cells from TOP_UAV have been
# executed and after Google Drive has been mounted. The cells intentionally do not
# retrain PPO. They only reevaluate existing checkpoints and rebuild statistics.
#
# Execution order
# ---------------
# A. Run Sections 0-3 first (setup, provenance, exact configs, helpers).
# B. Run Section 4 for the primary no-terminal evaluation.
# C. Run Sections 5-8 for derived metrics, Holm families, figures, and release audit.
#
# All RUN_* toggles default to False. Enable one block at a time.

# %%
# ============================================================
# 0. Imports, paths, and runtime checks
# ============================================================

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import shutil
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

try:
    from scipy.stats import binomtest, t as student_t
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("scipy is required for exact paired tests and t intervals") from exc

try:
    from stable_baselines3 import PPO
except Exception as exc:  # pragma: no cover - dependency guard
    raise RuntimeError("stable-baselines3 must be installed in the Colab runtime") from exc

try:
    from IPython.display import display
except Exception:  # pragma: no cover
    def display(obj):
        print(obj)


DRIVE_ROOT = Path("/content/drive/MyDrive/RUSCUAV")
PAPER_ROOT = DRIVE_ROOT / "paper_ready_v1"
TABLE_DIR = PAPER_ROOT / "tables"
FIG_DIR = PAPER_ROOT / "figures_tits"
SUBMISSION_REPAIR_ROOT = DRIVE_ROOT / "experiments" / "paperA_submission_repair_v1"
PROVENANCE_DIR = SUBMISSION_REPAIR_ROOT / "provenance"
PRIMARY_EVAL_DIR = SUBMISSION_REPAIR_ROOT / "primary_no_terminal"
SECONDARY_EVAL_DIR = SUBMISSION_REPAIR_ROOT / "secondary_operational_stopping"
SUPPLEMENT_DIR = SUBMISSION_REPAIR_ROOT / "supplement"
RELEASE_DIR = SUBMISSION_REPAIR_ROOT / "release_manifest"

for directory in [
    TABLE_DIR,
    FIG_DIR,
    PROVENANCE_DIR,
    PRIMARY_EVAL_DIR,
    SECONDARY_EVAL_DIR,
    SUPPLEMENT_DIR,
    RELEASE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

DT_S = 0.5
DEFAULT_BUDGETS = {
    "cargo_min": 0.80,
    "road_s": 20.0,
    "ped_s": 5.0,
    "building_s": 15.0,
    "sensitive_s": 0.0,
    "dynamic": 10.0,
}

# These are the exact settings used by the reported receding planners and M11-v1 runs.
REPORTED_M9_CONFIG = {
    "plan_res_m": 5.0,
    "planner_agl_m": 40.0,
    "risk_lambda": 0.35,
    "use_astar": True,
    "waypoint_radius_m": 8.0,
    "target_speed_mps": 12.0,
    "max_steps": 1000,
}
REPORTED_M10_LOCAL_CONFIG = {
    **REPORTED_M9_CONFIG,
    "sensing_radius_m": 75.0,
    "replan_every_steps": 5,
    "prior_risk_mode": "mean",
    "hard_blockers_known": True,
}
REPORTED_M10_FULL_CONFIG = {
    **REPORTED_M9_CONFIG,
    "sensing_radius_m": 75.0,
    "replan_every_steps": 5,
    "prior_risk_mode": "perfect",
    "hard_blockers_known": True,
}
REPORTED_M11_PLANNER_CONFIG = {
    **REPORTED_M9_CONFIG,
    "sensing_radius_m": 75.0,
    "replan_every_steps": 3,
    "prior_risk_mode": "perfect",
    "hard_blockers_known": True,
}
REPORTED_DYNAMIC_CONFIG = {
    "n_hotspots": 16,
    "hotspot_mode": "road_sidewalk",
    "radius_m_min": 60.0,
    "radius_m_max": 140.0,
    "amplitude_min": 15.0,
    "amplitude_max": 40.0,
    "start_step_min": 0,
    "start_step_max": 180,
    "duration_min_steps": 120,
    "duration_max_steps": 350,
    "temporal_mode": "hann",
    "altitude_h_scale_m": 70.0,
    "dynamic_cumulative_risk_budget": 10.0,
    "dynamic_risk_lambda": 0.20,
    "dynamic_norm_denom": 10.0,
}
REPORTED_CARGO_HEALTH_CONFIG = {
    "g0": 9.81,
    "overload_soft_g": 0.30,
    "overload_hard_g_diagnostic": 1.20,
    "health_decay_k": 0.005,
}
REPORTED_TRAINING_BUDGET = {
    "stage_A_steps": 50_000,
    "stage_B_steps": 75_000,
    "stage_C_steps": 75_000,
    "total_steps": 200_000,
    "vienna_extended_stage_C_steps": 250_000,
}

REQUIRED_GLOBALS = [
    "UrbanWorldLayers",
    "RealisticUrbanEnvConfig",
    "RealisticUrbanCargoUAVEnvM4",
    "make_ruscuav_m8_ablation_env_m4_mode",
    "make_ruscuav_env_m4_mode",
    "M9PlannerConfig",
    "M10RecedingHorizonPlannerConfig",
    "evaluate_m9_static_planner_drive",
    "evaluate_m10_receding_planner_drive",
    "M11DynamicRiskConfig",
    "evaluate_m11_dynamic_rl_drive",
    "evaluate_m11c_dynamic_obs_rl_drive",
    "rollout_m11_dynamic_receding_planner_episode",
    "summarize_m11_dynamic_eval_df",
]

missing_globals = [name for name in REQUIRED_GLOBALS if name not in globals()]
if missing_globals:
    print("[STOP] Run the TOP_UAV definition cells first. Missing:")
    for name in missing_globals:
        print("  -", name)
else:
    print("Runtime definitions found.")


# %% [markdown]
# 1. Provenance manifest and exact-configuration audit
# ----------------------------------------------------
# This section scans paper tables and experiment folders, hashes all claim-driving
# artifacts, extracts run configurations from JSON, and flags configuration conflicts.

# %%
# ============================================================
# 1. Provenance helpers
# ============================================================


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: os.PathLike[str] | str, chunk_size: int = 1024 * 1024) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def safe_read_json(path: os.PathLike[str] | str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {"value": obj}


def flatten_dict(obj: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in obj.items():
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(flatten_dict(value, prefix=name, sep=sep))
        else:
            out[name] = value
    return out


def existing_path(value: Any) -> Optional[Path]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    p = Path(str(value))
    return p if p.exists() else None


def find_related_jsons(csv_path: Path) -> list[Path]:
    stem = csv_path.name.replace("_episodes.csv", "")
    candidates = [
        csv_path.with_name(stem + "_summary.json"),
        csv_path.with_name(stem + "_config.json"),
        csv_path.with_name(stem + "_planner_config.json"),
        csv_path.with_name(stem + "_dynamic_config.json"),
        csv_path.parent / "protocol.json",
    ]
    return [p for p in candidates if p.exists()]


def collect_claim_artifacts(table_dir: Path = TABLE_DIR) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    path_columns = [
        "csv_path",
        "summary_path",
        "model_path",
        "final_model_path",
        "world_path",
        "checkpoint_path",
    ]

    for table_path in sorted(table_dir.glob("*.csv")):
        try:
            table = pd.read_csv(table_path)
        except Exception as exc:
            print(f"[WARN] Cannot read {table_path}: {exc!r}")
            continue

        for row_index, row in table.iterrows():
            label = str(
                row.get(
                    "label",
                    row.get("method", row.get("variant", row.get("planner", table_path.stem))),
                )
            )
            for column in path_columns:
                if column not in table.columns:
                    continue
                p = existing_path(row.get(column))
                if p is None:
                    continue
                record = {
                    "source_table": table_path.name,
                    "source_row": int(row_index),
                    "label": label,
                    "path_role": column,
                    "path": str(p),
                    "size_bytes": int(p.stat().st_size),
                    "sha256": sha256_file(p),
                    "modified_utc": datetime.fromtimestamp(
                        p.stat().st_mtime, tz=timezone.utc
                    ).replace(microsecond=0).isoformat(),
                }
                rows.append(record)

                if column == "csv_path":
                    for jp in find_related_jsons(p):
                        rows.append(
                            {
                                "source_table": table_path.name,
                                "source_row": int(row_index),
                                "label": label,
                                "path_role": "related_json",
                                "path": str(jp),
                                "size_bytes": int(jp.stat().st_size),
                                "sha256": sha256_file(jp),
                                "modified_utc": datetime.fromtimestamp(
                                    jp.stat().st_mtime, tz=timezone.utc
                                ).replace(microsecond=0).isoformat(),
                            }
                        )

    return pd.DataFrame(rows).drop_duplicates(subset=["path", "sha256"])


def extract_run_config_rows(artifact_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    json_paths = artifact_df.loc[
        artifact_df["path"].astype(str).str.endswith(".json"), "path"
    ].drop_duplicates()

    for path_str in json_paths:
        p = Path(path_str)
        obj = safe_read_json(p)
        flat = flatten_dict(obj)
        for key, value in flat.items():
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, sort_keys=True)
            rows.append(
                {
                    "json_path": str(p),
                    "json_sha256": sha256_file(p),
                    "parameter": key,
                    "value": value,
                }
            )
    return pd.DataFrame(rows)


def compare_expected_config(
    config_rows: pd.DataFrame,
    expected: Mapping[str, Any],
    path_contains: str,
    parameter_suffix_map: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    parameter_suffix_map = parameter_suffix_map or {}
    subset = config_rows[
        config_rows["json_path"].astype(str).str.contains(path_contains, case=False, regex=False)
    ].copy()
    results: list[dict[str, Any]] = []

    for expected_key, expected_value in expected.items():
        suffix = parameter_suffix_map.get(expected_key, expected_key)
        hits = subset[subset["parameter"].astype(str).str.endswith(suffix)]
        observed_values = hits["value"].dropna().astype(str).unique().tolist()
        expected_str = str(expected_value)
        results.append(
            {
                "path_filter": path_contains,
                "expected_parameter": expected_key,
                "expected_value": expected_str,
                "observed_values": " | ".join(observed_values),
                "status": "MATCH" if expected_str in observed_values else "CHECK",
            }
        )
    return pd.DataFrame(results)


RUN_PROVENANCE_AUDIT = False

if RUN_PROVENANCE_AUDIT:
    artifacts = collect_claim_artifacts(TABLE_DIR)
    artifact_path = PROVENANCE_DIR / "claim_artifact_manifest.csv"
    artifacts.to_csv(artifact_path, index=False)

    config_rows = extract_run_config_rows(artifacts)
    config_rows_path = PROVENANCE_DIR / "run_config_parameters_long.csv"
    config_rows.to_csv(config_rows_path, index=False)

    audit_parts = [
        compare_expected_config(config_rows, REPORTED_M10_LOCAL_CONFIG, "receding"),
        compare_expected_config(config_rows, REPORTED_DYNAMIC_CONFIG, "dynamic"),
    ]
    config_audit = pd.concat(audit_parts, ignore_index=True)
    config_audit_path = PROVENANCE_DIR / "reported_configuration_audit.csv"
    config_audit.to_csv(config_audit_path, index=False)

    print("Saved:", artifact_path)
    print("Saved:", config_rows_path)
    print("Saved:", config_audit_path)
    display(config_audit)


# %% [markdown]
# 2. Evaluation-regime definitions and derived metrics
# ----------------------------------------------------
# Primary evaluation: hard physical/sensitive constraints remain active, but soft
# road/pedestrian/building exposure exceedances do not terminate the episode.
# MAR is computed post hoc. Secondary evaluation retains Stage C operational stopping.

# %%
# ============================================================
# 2. Regime and metric helpers
# ============================================================

@dataclass(frozen=True)
class EvaluationRegime:
    name: str
    stage: str
    mode: str = "strict_eval"
    terminate_on_soft_budgets: bool = False
    description: str = ""


PRIMARY_NO_TERMINAL = EvaluationRegime(
    name="primary_no_terminal",
    stage="B",
    mode="strict_eval",
    terminate_on_soft_budgets=False,
    description=(
        "Hard physical and low-altitude sensitive-zone constraints are active; "
        "road, pedestrian, building, and dynamic budget exceedances are logged "
        "but do not terminate the episode."
    ),
)
SECONDARY_OPERATIONAL_STOP = EvaluationRegime(
    name="secondary_operational_stopping",
    stage="C",
    mode="strict_eval",
    terminate_on_soft_budgets=True,
    description=(
        "Stage C operational stress test: selected semantic-budget exceedances "
        "can terminate the episode."
    ),
)


def as_bool_series(series: pd.Series) -> pd.Series:
    if str(series.dtype) == "bool":
        return series.astype(bool)
    return series.map(
        lambda x: bool(x)
        if isinstance(x, (bool, np.bool_))
        else str(x).strip().lower() in {"true", "1", "yes", "y", "t"}
    ).astype(bool)


def num_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").fillna(default)


def recompute_static_acceptability(
    df: pd.DataFrame,
    budgets: Mapping[str, float] = DEFAULT_BUDGETS,
) -> pd.Series:
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


def recompute_dynamic_acceptability(
    df: pd.DataFrame,
    budgets: Mapping[str, float] = DEFAULT_BUDGETS,
) -> pd.Series:
    return recompute_static_acceptability(df, budgets) & (
        num_col(df, "dynamic_cumulative_risk") <= float(budgets["dynamic"])
    )


def add_derived_metrics(
    df: pd.DataFrame,
    dt_s: float = DT_S,
    budgets: Mapping[str, float] = DEFAULT_BUDGETS,
) -> pd.DataFrame:
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


def summarize_episode_dataframe(df: pd.DataFrame, dynamic: bool = False) -> dict[str, Any]:
    d = add_derived_metrics(df)
    success = as_bool_series(d["is_success"])
    acc_col = "mission_acceptable_posthoc_dynamic" if dynamic else "mission_acceptable_posthoc_static"
    acceptable = as_bool_series(d[acc_col])

    result: dict[str, Any] = {
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


# %% [markdown]
# 3. Generic resume-safe no-terminal policy evaluator
# ---------------------------------------------------
# The existing M8/M11 evaluators hard-code some Stage C choices. The generic
# evaluator below accepts an environment factory, records actual path length, and
# always recomputes MAR post hoc from the final per-episode record.

# %%
# ============================================================
# 3. Generic PPO evaluation
# ============================================================


def make_geo_eval_env(
    world,
    regime: EvaluationRegime,
    seed: int,
    n_obstacles: int = 0,
):
    """Geometry-only evaluation with strict geometric checks in both regimes."""
    cfg = RealisticUrbanEnvConfig(
        risk_profile="balanced",
        include_urban_vector_obs=False,
        use_urban_risk_costs=False,
        urban_risk_lambda=0.0,
        green_bonus_lambda=0.0,
        use_urban_altitude_reference=True,
        terminate_on_urban_constraint=bool(regime.terminate_on_soft_budgets),
        hard_sensitive_zones=True,
        default_n_obstacles=int(n_obstacles),
        mission_accept_cargo_min=DEFAULT_BUDGETS["cargo_min"],
        mission_accept_road_budget_s=DEFAULT_BUDGETS["road_s"],
        mission_accept_sidewalk_budget_s=DEFAULT_BUDGETS["ped_s"],
        mission_accept_building_budget_s=DEFAULT_BUDGETS["building_s"],
        mission_accept_sensitive_budget_s=DEFAULT_BUDGETS["sensitive_s"],
        terminal_road_budget_s=30.0,
        terminal_sidewalk_budget_s=10.0,
        terminal_building_budget_s=20.0,
        terminal_sensitive_budget_s=0.0,
    )
    env = RealisticUrbanCargoUAVEnvM4(
        world=world,
        config=cfg,
        include_shield_features=False,
        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,
        urban_lidar_step_m=3.0,
        urban_segment_check_step_m=1.0,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )
    env.reset(seed=int(seed))
    return env


def make_static_eval_env(
    method: str,
    world,
    regime: EvaluationRegime,
    seed: int,
    n_obstacles: int = 0,
):
    method = str(method)
    if method == "GEO-RL":
        return make_geo_eval_env(world, regime, seed, n_obstacles=n_obstacles)
    if method == "OBS-RL":
        return make_ruscuav_m8_ablation_env_m4_mode(
            world=world,
            variant_name="obs_only",
            stage=regime.stage,
            mode=regime.mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=seed,
        )
    if method == "RISK+GREEN-RL":
        return make_ruscuav_m8_ablation_env_m4_mode(
            world=world,
            variant_name="reward_only",
            stage=regime.stage,
            mode=regime.mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=seed,
        )
    if method == "RISK-RL":
        return make_ruscuav_m8_ablation_env_m4_mode(
            world=world,
            variant_name="risk_only",
            stage=regime.stage,
            mode=regime.mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=seed,
        )
    if method == "GREEN-RL":
        return make_ruscuav_m8_ablation_env_m4_mode(
            world=world,
            variant_name="green_only",
            stage=regime.stage,
            mode=regime.mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=seed,
        )
    if method == "RISK+OBS-RL":
        return make_ruscuav_env_m4_mode(
            world=world,
            stage=regime.stage,
            mode=regime.mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=seed,
        )
    raise ValueError(f"Unsupported static policy method: {method}")


def rollout_ppo_episode(
    model,
    env,
    seed: int,
    method: str,
    regime: EvaluationRegime,
    deterministic: bool = True,
    max_steps: int = 1000,
) -> tuple[dict[str, Any], np.ndarray]:
    obs, reset_info = env.reset(seed=int(seed))
    path = [np.asarray(env.current_pos, dtype=np.float32).copy()]
    rewards: list[float] = []
    final_info: dict[str, Any] = {}
    done = False
    steps = 0

    while not done and steps < int(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        final_info = dict(info)
        path.append(np.asarray(env.current_pos, dtype=np.float32).copy())
        done = bool(terminated or truncated)
        steps += 1

    if "dist_xy" not in final_info:
        final_info["dist_xy"] = float(
            np.linalg.norm(env.target_point[:2] - env.current_pos[:2])
        )

    path_arr = np.asarray(path, dtype=np.float32)
    path_length_xy = float(np.linalg.norm(np.diff(path_arr[:, :2], axis=0), axis=1).sum())
    path_length_3d = float(np.linalg.norm(np.diff(path_arr, axis=0), axis=1).sum())

    row = {
        "episode": None,
        "seed": int(seed),
        "method": str(method),
        "evaluation_regime": regime.name,
        "stage": regime.stage,
        "mode": regime.mode,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "episode_duration_s": float(steps * DT_S),
        "termination": final_info.get("termination"),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable_logged": bool(final_info.get("mission_acceptable", False)),
        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "path_length_xy": path_length_xy,
        "path_length_3d": path_length_3d,
        "cargo_health": float(final_info.get("cargo_health", np.nan)),
        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(final_info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),
        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),
        "building_collision_steps": int(final_info.get("urban_building_collision_steps", 0)),
        "canopy_collision_steps": int(final_info.get("urban_canopy_collision_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(
            final_info.get("urban_sensitive_low_altitude_violation_steps", 0)
        ),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),
        "reset_safe_start_target": bool(reset_info.get("urban_safe_start_target", False)),
    }
    one = add_derived_metrics(pd.DataFrame([row]))
    row["mission_acceptable"] = bool(one.loc[0, "mission_acceptable_posthoc_static"])
    env.close()
    return row, path_arr


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    with tmp.open("w", encoding="utf-8") as f:
        json.dump(convert(obj), f, indent=2)
    os.replace(tmp, path)


def evaluate_ppo_resume(
    model_path: os.PathLike[str] | str,
    env_factory: Callable[[int], Any],
    method: str,
    regime: EvaluationRegime,
    output_dir: os.PathLike[str] | str,
    n_episodes: int,
    seed0: int,
    deterministic: bool = True,
    save_paths: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any], Path, Path]:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths_dir = output_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "episodes.csv"
    summary_path = output_dir / "summary.json"
    protocol_path = output_dir / "protocol.json"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        rows = existing.to_dict("records")
        completed = set(existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming {method}/{regime.name}: {len(completed)}/{n_episodes}")
    else:
        rows = []
        completed = set()

    atomic_json(
        protocol_path,
        {
            "method": method,
            "regime": asdict(regime),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "n_episodes": int(n_episodes),
            "seed0": int(seed0),
            "deterministic": bool(deterministic),
            "created_or_resumed_at": utc_now_iso(),
        },
    )

    model = PPO.load(str(model_path), device="cpu")
    for episode in range(int(n_episodes)):
        if episode in completed:
            continue
        seed = int(seed0 + episode)
        env = env_factory(seed)
        row, path = rollout_ppo_episode(
            model=model,
            env=env,
            seed=seed,
            method=method,
            regime=regime,
            deterministic=deterministic,
        )
        row["episode"] = int(episode)
        rows.append(row)
        if save_paths:
            np.save(paths_dir / f"episode_{episode:05d}_seed_{seed}_path.npy", path)

        current = add_derived_metrics(pd.DataFrame(rows).sort_values("episode"))
        current.to_csv(csv_path, index=False)
        summary = summarize_episode_dataframe(current, dynamic=False)
        summary.update(
            {
                "status": "partial" if len(current) < int(n_episodes) else "complete",
                "method": method,
                "regime": asdict(regime),
                "model_path": str(model_path),
                "model_sha256": sha256_file(model_path),
                "updated_at": utc_now_iso(),
                "csv_path": str(csv_path),
            }
        )
        atomic_json(summary_path, summary)
        if episode == 0 or (episode + 1) % 10 == 0:
            print(
                f"[{method}/{regime.name}] {episode+1}/{n_episodes} "
                f"SR={summary['success_rate']:.1%} MAR={summary['mission_acceptability_rate']:.1%} "
                f"term={row['termination']}"
            )

    final_df = add_derived_metrics(pd.DataFrame(rows).sort_values("episode"))
    final_df.to_csv(csv_path, index=False)
    final_summary = summarize_episode_dataframe(final_df, dynamic=False)
    final_summary.update(
        {
            "status": "complete",
            "finished_at": utc_now_iso(),
            "method": method,
            "regime": asdict(regime),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
        }
    )
    atomic_json(summary_path, final_summary)
    return final_df, final_summary, csv_path, summary_path


# %% [markdown]
# 4. Primary no-terminal reevaluation plan
# ----------------------------------------
# Enable one RUN_* switch at a time. No PPO training is performed.

# %%
# ============================================================
# 4A. Artifact discovery and experiment registries
# ============================================================

MULTISEED_TABLE = TABLE_DIR / "paper_table_multiseed_rl_seed_level.csv"


def load_multiseed_table() -> pd.DataFrame:
    if not MULTISEED_TABLE.exists():
        raise FileNotFoundError(
            f"Missing {MULTISEED_TABLE}. Copy it from the archived paper_ready_v1 tables."
        )
    df = pd.read_csv(MULTISEED_TABLE)
    complete = df[df["run_status"].astype(str).eq("complete")].copy()
    return complete


def model_registry_from_multiseed() -> dict[tuple[str, int], Path]:
    df = load_multiseed_table()
    registry: dict[tuple[str, int], Path] = {}
    for _, row in df.iterrows():
        p = existing_path(row.get("final_model_path"))
        if p is not None:
            registry[(str(row["method"]), int(row["train_seed"]))] = p
    return registry


def load_world_from_candidate_names(
    global_names: Iterable[str],
    path_patterns: Iterable[str | Path],
):
    for name in global_names:
        if name in globals():
            return globals()[name], f"global:{name}"
    for pattern in path_patterns:
        pattern_str = str(pattern)
        if any(token in pattern_str for token in ("*", "?", "[")):
            matches = [Path(item) for item in sorted(glob.glob(pattern_str, recursive=True))]
        else:
            matches = [Path(pattern_str)]
        for path in matches:
            if path.exists():
                return UrbanWorldLayers.load_npz(path), str(path)
    raise FileNotFoundError(f"World not found. globals={list(global_names)}, patterns={list(path_patterns)}")


MODEL_REGISTRY = model_registry_from_multiseed() if MULTISEED_TABLE.exists() else {}

# Optional single-checkpoint paths. Populate only if auto-discovery cannot find them.
SINGLE_CHECKPOINTS: dict[str, Optional[Path]] = {
    "RISK+OBS-RL": None,          # M6 final checkpoint
    "DYN-RISK-RL": None,         # M11B final checkpoint
    "VIENNA_ZERO_SHOT": None,     # typically one RISK+GREEN checkpoint
    "VIENNA_SHORT_FT": None,
    "VIENNA_EXTENDED_FT": None,
}

CANONICAL_WORLD_PATH = DRIVE_ROOT / "worlds" / "AOI_FAKE_001_urban_world_layers.npz"


def require_model(method_key: str, train_seed: int) -> Path:
    key = (method_key, int(train_seed))
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Model not found in multiseed registry: {key}")
    return MODEL_REGISTRY[key]


# %%
# ============================================================
# 4B. Static canonical-world no-terminal evaluation
# ============================================================

RUN_STATIC_PRIMARY_RL = False
RUN_STATIC_PRIMARY_PLANNERS = False
STATIC_TRAIN_SEEDS = [1001, 2001, 3001, 4001, 5001]
STATIC_EVAL_SEED0 = 5000
STATIC_N_EPISODES = 200

if RUN_STATIC_PRIMARY_RL:
    canonical_world = UrbanWorldLayers.load_npz(CANONICAL_WORLD_PATH)
    method_mapping = {
        "GEO-RL": "M7_geometry_only",
        "OBS-RL": "M8_obs_only",
        "RISK+GREEN-RL": "M8_reward_only",
    }
    static_seed_rows: list[dict[str, Any]] = []

    for short_name, registry_name in method_mapping.items():
        for train_seed in STATIC_TRAIN_SEEDS:
            model_path = require_model(registry_name, train_seed)
            out_dir = PRIMARY_EVAL_DIR / "static" / short_name / f"trainseed_{train_seed}"
            df_eval, summary, csv_path, summary_path = evaluate_ppo_resume(
                model_path=model_path,
                env_factory=lambda seed, m=short_name: make_static_eval_env(
                    m, canonical_world, PRIMARY_NO_TERMINAL, seed
                ),
                method=short_name,
                regime=PRIMARY_NO_TERMINAL,
                output_dir=out_dir,
                n_episodes=STATIC_N_EPISODES,
                seed0=STATIC_EVAL_SEED0,
                deterministic=True,
                save_paths=False,
            )
            static_seed_rows.append(
                {
                    "method": short_name,
                    "train_seed": train_seed,
                    "csv_path": str(csv_path),
                    "summary_path": str(summary_path),
                    **{k: v for k, v in summary.items() if not isinstance(v, dict)},
                }
            )

    static_seed_df = pd.DataFrame(static_seed_rows)
    static_seed_df.to_csv(TABLE_DIR / "paper_table_primary_no_terminal_static_seed_level.csv", index=False)

    aggregate_rows: list[dict[str, Any]] = []
    metrics = [
        "success_rate",
        "mission_acceptability_rate",
        "acceptable_given_success_rate",
        "mean_urban_cumulative_risk",
        "mean_urban_cumulative_risk_given_success",
        "mean_urban_risk_per_step",
        "mean_urban_risk_per_s",
        "mean_urban_risk_per_m_xy",
        "mean_road_fraction",
        "mean_ped_fraction",
        "mean_building_fraction",
    ]
    for method, group in static_seed_df.groupby("method"):
        for metric in metrics:
            if metric not in group.columns:
                continue
            ci = training_seed_ci(group[metric])
            aggregate_rows.append({"method": method, "metric": metric, **ci})
    pd.DataFrame(aggregate_rows).to_csv(
        TABLE_DIR / "paper_table_primary_no_terminal_static_training_seed_ci.csv",
        index=False,
    )

if RUN_STATIC_PRIMARY_PLANNERS:
    canonical_world = UrbanWorldLayers.load_npz(CANONICAL_WORLD_PATH)
    dirs = setup_ruscuav_drive_project(
        drive_root=str(DRIVE_ROOT),
        experiment_name="paperA_primary_no_terminal_static_planners",
    )
    m9_cfg = M9PlannerConfig(**REPORTED_M9_CONFIG)
    m10_local_cfg = M10RecedingHorizonPlannerConfig(**REPORTED_M10_LOCAL_CONFIG)
    m10_full_cfg = M10RecedingHorizonPlannerConfig(**REPORTED_M10_FULL_CONFIG)

    evaluate_m9_static_planner_drive(
        world=canonical_world,
        drive_dirs=dirs,
        planner_config=m9_cfg,
        risk_profile="balanced",
        n_episodes=STATIC_N_EPISODES,
        seed0=STATIC_EVAL_SEED0,
        stage=PRIMARY_NO_TERMINAL.stage,
        mode=PRIMARY_NO_TERMINAL.mode,
        n_obstacles=0,
        prefix="primary_no_terminal_STATIC_Astar_n200_seed5000",
        save_paths=True,
    )
    evaluate_m10_receding_planner_drive(
        world=canonical_world,
        drive_dirs=dirs,
        cfg=m10_local_cfg,
        risk_profile="balanced",
        n_episodes=STATIC_N_EPISODES,
        seed0=STATIC_EVAL_SEED0,
        stage=PRIMARY_NO_TERMINAL.stage,
        mode=PRIMARY_NO_TERMINAL.mode,
        n_obstacles=0,
        prefix="primary_no_terminal_LOCAL_RH_Astar_n200_seed5000",
        save_paths=True,
    )
    evaluate_m10_receding_planner_drive(
        world=canonical_world,
        drive_dirs=dirs,
        cfg=m10_full_cfg,
        risk_profile="balanced",
        n_episodes=STATIC_N_EPISODES,
        seed0=STATIC_EVAL_SEED0,
        stage=PRIMARY_NO_TERMINAL.stage,
        mode=PRIMARY_NO_TERMINAL.mode,
        n_obstacles=0,
        prefix="primary_no_terminal_FULL_RH_Astar_n200_seed5000",
        save_paths=True,
    )


# %%
# ============================================================
# 4C. Dynamic no-terminal evaluation
# ============================================================

RUN_DYNAMIC_PRIMARY = False
DYNAMIC_EVAL_SEED0 = 5000
DYNAMIC_N_EPISODES = 200
DYNAMIC_M11C_TRAIN_SEEDS = [1001, 2001, 3001, 4001, 5001]


def reported_dynamic_config():
    return M11DynamicRiskConfig(**REPORTED_DYNAMIC_CONFIG)


def evaluate_dynamic_planner_mode(
    world,
    drive_dirs,
    cfg,
    dynamic_config,
    regime: EvaluationRegime,
    n_episodes: int,
    seed0: int,
    prefix: str,
    dynamic_prior_mode: str = "zero",
    save_paths: bool = True,
):
    """Resume-safe dynamic planner evaluator with explicit stage/regime."""
    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)
    paths_dir = eval_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)
    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"

    if csv_path.exists():
        existing = pd.read_csv(csv_path)
        rows = existing.to_dict("records")
        completed = set(existing["episode"].dropna().astype(int).tolist())
    else:
        rows, completed = [], set()

    for episode in range(int(n_episodes)):
        if episode in completed:
            continue
        seed = int(seed0 + episode)
        row, path = rollout_m11_dynamic_receding_planner_episode(
            world=world,
            seed=seed,
            cfg=cfg,
            dynamic_config=dynamic_config,
            dynamic_prior_mode=dynamic_prior_mode,
            stage=regime.stage,
            mode=regime.mode,
            n_obstacles=0,
            max_steps=cfg.max_steps,
        )
        row["episode"] = episode
        row["evaluation_regime"] = regime.name
        path = np.asarray(path, dtype=np.float32)
        row["path_length_xy"] = float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum())
        rows.append(row)
        if save_paths:
            np.save(paths_dir / f"episode_{episode:05d}_seed_{seed}_path.npy", path)

        current = add_derived_metrics(pd.DataFrame(rows).sort_values("episode"))
        current["mission_acceptable"] = recompute_dynamic_acceptability(current)
        current.to_csv(csv_path, index=False)
        summary = summarize_episode_dataframe(current, dynamic=True)
        summary.update(
            {
                "status": "partial" if len(current) < n_episodes else "complete",
                "evaluation_regime": asdict(regime),
                "planner_config": cfg.__dict__,
                "dynamic_config": dynamic_config.__dict__,
                "dynamic_prior_mode": dynamic_prior_mode,
                "updated_at": utc_now_iso(),
            }
        )
        atomic_json(summary_path, summary)

    final = add_derived_metrics(pd.DataFrame(rows).sort_values("episode"))
    final["mission_acceptable"] = recompute_dynamic_acceptability(final)
    final.to_csv(csv_path, index=False)
    summary = summarize_episode_dataframe(final, dynamic=True)
    summary.update(
        {
            "status": "complete",
            "evaluation_regime": asdict(regime),
            "planner_config": cfg.__dict__,
            "dynamic_config": dynamic_config.__dict__,
            "dynamic_prior_mode": dynamic_prior_mode,
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "finished_at": utc_now_iso(),
        }
    )
    atomic_json(summary_path, summary)
    return final, summary, csv_path, summary_path


if RUN_DYNAMIC_PRIMARY:
    canonical_world = UrbanWorldLayers.load_npz(CANONICAL_WORLD_PATH)
    dyn_cfg = reported_dynamic_config()
    dirs = setup_ruscuav_drive_project(
        drive_root=str(DRIVE_ROOT),
        experiment_name="paperA_primary_no_terminal_dynamic",
    )

    # Static-trained RISK+GREEN-RL: evaluate all five seed checkpoints under dynamic risk.
    for train_seed in STATIC_TRAIN_SEEDS:
        model_path = require_model("M8_reward_only", train_seed)
        evaluate_m11_dynamic_rl_drive(
            model_path=model_path,
            world=canonical_world,
            drive_dirs=dirs,
            dynamic_config=dyn_cfg,
            n_episodes=DYNAMIC_N_EPISODES,
            seed0=DYNAMIC_EVAL_SEED0,
            prefix=f"primary_no_terminal_static_trained_RISK_GREEN_seed{train_seed}",
            stage=PRIMARY_NO_TERMINAL.stage,
            mode=PRIMARY_NO_TERMINAL.mode,
            reward_only_obs=True,
            save_paths=False,
        )

    # M11C: all five training seeds.
    for train_seed in DYNAMIC_M11C_TRAIN_SEEDS:
        model_path = require_model("M11C_dynamic_obs_reward_only", train_seed)
        evaluate_m11c_dynamic_obs_rl_drive(
            model_path=model_path,
            world=canonical_world,
            drive_dirs=dirs,
            dynamic_config=dyn_cfg,
            n_episodes=DYNAMIC_N_EPISODES,
            seed0=DYNAMIC_EVAL_SEED0,
            prefix=f"primary_no_terminal_M11C_seed{train_seed}",
            stage=PRIMARY_NO_TERMINAL.stage,
            mode=PRIMARY_NO_TERMINAL.mode,
            save_paths=False,
        )

    # M11B is a single-checkpoint reference unless a replicated registry exists.
    if SINGLE_CHECKPOINTS["DYN-RISK-RL"] is not None:
        evaluate_m11_dynamic_rl_drive(
            model_path=SINGLE_CHECKPOINTS["DYN-RISK-RL"],
            world=canonical_world,
            drive_dirs=dirs,
            dynamic_config=dyn_cfg,
            n_episodes=DYNAMIC_N_EPISODES,
            seed0=DYNAMIC_EVAL_SEED0,
            prefix="primary_no_terminal_M11B_single_checkpoint",
            stage=PRIMARY_NO_TERMINAL.stage,
            mode=PRIMARY_NO_TERMINAL.mode,
            reward_only_obs=True,
            save_paths=False,
        )

    planner_cfg = M10RecedingHorizonPlannerConfig(**REPORTED_M11_PLANNER_CONFIG)
    evaluate_dynamic_planner_mode(
        world=canonical_world,
        drive_dirs=dirs,
        cfg=planner_cfg,
        dynamic_config=dyn_cfg,
        regime=PRIMARY_NO_TERMINAL,
        n_episodes=DYNAMIC_N_EPISODES,
        seed0=DYNAMIC_EVAL_SEED0,
        prefix="primary_no_terminal_dynamic_receding_Astar",
        dynamic_prior_mode="zero",
        save_paths=False,
    )


# %%
# ============================================================
# 4D. Vienna and Warsaw scenario-aligned no-terminal evaluation
# ============================================================

RUN_VIENNA_PRIMARY = False
RUN_WARSAW_PRIMARY = False
REAL_EVAL_SEED0 = 7000
REAL_N_EPISODES = 200


def get_real_worlds():
    vienna, vienna_source = load_world_from_candidate_names(
        ["vienna_world_footway_only", "vienna_world"],
        [
            DRIVE_ROOT / "worlds" / "*vienna*footway*urban_world_layers*.npz",
            DRIVE_ROOT / "worlds" / "*vienna*urban_world_layers*.npz",
        ],
    )
    warsaw, warsaw_source = load_world_from_candidate_names(
        ["real_world_footway_only", "warsaw_world_footway_only", "real_world"],
        [
            DRIVE_ROOT / "worlds" / "*warsaw*footway*urban_world_layers*.npz",
            DRIVE_ROOT / "worlds" / "*warsaw*urban_world_layers*.npz",
        ],
    )
    return (vienna, vienna_source), (warsaw, warsaw_source)


if RUN_VIENNA_PRIMARY:
    (vienna_world, vienna_source), _ = get_real_worlds()
    dirs = setup_ruscuav_drive_project(
        drive_root=str(DRIVE_ROOT),
        experiment_name="paperA_primary_no_terminal_vienna",
    )

    # Existing Vienna evaluator accepts stage='B'. Populate checkpoint paths first.
    for label, checkpoint in [
        ("Vienna_zero_shot_RISK_GREEN", SINGLE_CHECKPOINTS["VIENNA_ZERO_SHOT"]),
        ("Vienna_short_finetuned_RISK_GREEN", SINGLE_CHECKPOINTS["VIENNA_SHORT_FT"]),
        ("Vienna_extended_finetuned_RISK_GREEN", SINGLE_CHECKPOINTS["VIENNA_EXTENDED_FT"]),
    ]:
        if checkpoint is None:
            print(f"[SKIP] Set the checkpoint path for {label} in SINGLE_CHECKPOINTS before Vienna evaluation")
            continue
        evaluate_m12c_real_policy_drive(
            model_path=checkpoint,
            world=vienna_world,
            policy_kind="reward_only",
            drive_dirs=dirs,
            n_episodes=REAL_N_EPISODES,
            seed0=REAL_EVAL_SEED0,
            prefix=f"primary_no_terminal_{label}",
            stage=PRIMARY_NO_TERMINAL.stage,
            mode=PRIMARY_NO_TERMINAL.mode,
            deterministic=True,
            save_paths=True,
        )

    vienna_planner_cfg = M10RecedingHorizonPlannerConfig(
        **{**REPORTED_M10_FULL_CONFIG, "planner_agl_m": 30.0}
    )
    evaluate_m10_receding_planner_drive(
        world=vienna_world,
        drive_dirs=dirs,
        cfg=vienna_planner_cfg,
        risk_profile="balanced",
        n_episodes=REAL_N_EPISODES,
        seed0=REAL_EVAL_SEED0,
        stage=PRIMARY_NO_TERMINAL.stage,
        mode=PRIMARY_NO_TERMINAL.mode,
        n_obstacles=0,
        prefix="primary_no_terminal_Vienna_FULL_RH_Astar",
        save_paths=True,
    )

if RUN_WARSAW_PRIMARY:
    _, (warsaw_world, warsaw_source) = get_real_worlds()
    dirs = setup_ruscuav_drive_project(
        drive_root=str(DRIVE_ROOT),
        experiment_name="paperA_primary_no_terminal_warsaw_scenario_aligned",
    )

    method_mapping = {
        "GEO-RL": "M7_geometry_only",
        "OBS-RL": "M8_obs_only",
        "RISK+GREEN-RL": "M8_reward_only",
    }
    for short_name, registry_name in method_mapping.items():
        for train_seed in STATIC_TRAIN_SEEDS:
            model_path = require_model(registry_name, train_seed)
            out_dir = Path(dirs["eval"]) / short_name / f"trainseed_{train_seed}"
            evaluate_ppo_resume(
                model_path=model_path,
                env_factory=lambda seed, m=short_name: make_static_eval_env(
                    m, warsaw_world, PRIMARY_NO_TERMINAL, seed
                ),
                method=short_name,
                regime=PRIMARY_NO_TERMINAL,
                output_dir=out_dir,
                n_episodes=REAL_N_EPISODES,
                seed0=REAL_EVAL_SEED0,
                deterministic=True,
                save_paths=True,
            )

    warsaw_planner_cfg = M10RecedingHorizonPlannerConfig(
        **{**REPORTED_M10_FULL_CONFIG, "planner_agl_m": 30.0}
    )
    evaluate_m10_receding_planner_drive(
        world=warsaw_world,
        drive_dirs=dirs,
        cfg=warsaw_planner_cfg,
        risk_profile="balanced",
        n_episodes=REAL_N_EPISODES,
        seed0=REAL_EVAL_SEED0,
        stage=PRIMARY_NO_TERMINAL.stage,
        mode=PRIMARY_NO_TERMINAL.mode,
        n_obstacles=0,
        prefix="primary_no_terminal_Warsaw_FULL_RH_Astar_seed7000",
        save_paths=True,
    )

    # Retain the corresponding Stage C operational-stopping planner on the same scenarios.
    evaluate_m10_receding_planner_drive(
        world=warsaw_world,
        drive_dirs=dirs,
        cfg=warsaw_planner_cfg,
        risk_profile="balanced",
        n_episodes=REAL_N_EPISODES,
        seed0=REAL_EVAL_SEED0,
        stage=SECONDARY_OPERATIONAL_STOP.stage,
        mode=SECONDARY_OPERATIONAL_STOP.mode,
        n_obstacles=0,
        prefix="secondary_operational_stopping_Warsaw_FULL_RH_Astar_seed7000",
        save_paths=True,
    )


# %% [markdown]
# 5. Non-acceptability causes, conditional risk, and beta_dyn sensitivity

# %%
# ============================================================
# 5A. Cause decomposition
# ============================================================

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


def build_derived_tables_from_csv_registry(registry: pd.DataFrame) -> None:
    """registry columns: label, csv_path, dynamic (bool), train_seed (optional)."""
    summaries, causes = [], []
    for _, row in registry.iterrows():
        p = Path(row["csv_path"])
        if not p.exists():
            print("[WARN] Missing", p)
            continue
        df = add_derived_metrics(pd.read_csv(p))
        dynamic = bool(row.get("dynamic", False))
        label = str(row["label"])
        summary = summarize_episode_dataframe(df, dynamic=dynamic)
        summaries.append({**row.to_dict(), **{k: v for k, v in summary.items() if not isinstance(v, dict)}})
        causes.append(cause_decomposition(df, label=label, dynamic=dynamic))

    pd.DataFrame(summaries).to_csv(
        TABLE_DIR / "paper_table_conditional_normalized_risk_metrics.csv", index=False
    )
    if causes:
        pd.concat(causes, ignore_index=True).to_csv(
            TABLE_DIR / "paper_table_non_acceptability_cause_decomposition.csv",
            index=False,
        )


# %%
# ============================================================
# 5B. Dynamic-budget sensitivity (post-hoc, no rerun required)
# ============================================================

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


# %% [markdown]
# 6. Reproducible paired tests and complete Holm families

# %%
# ============================================================
# 6. Exact paired tests and Holm correction
# ============================================================


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


# Explicit families. Update labels to match the primary no-terminal CSV registry.
HOLM_COMPARISON_TEMPLATE = pd.DataFrame(
    [
        # Synthetic static family (m=6)
        ("static", "riskgreen_vs_geo_SR", "RISK+GREEN-RL", "GEO-RL", "success", True),
        ("static", "riskgreen_vs_geo_MAR", "RISK+GREEN-RL", "GEO-RL", "MAR_static", True),
        ("static", "riskobs_vs_geo_SR", "RISK+OBS-RL", "GEO-RL", "success", False),
        ("static", "riskobs_vs_geo_MAR", "RISK+OBS-RL", "GEO-RL", "MAR_static", False),
        ("static", "riskgreen_vs_fullplanner_SR", "RISK+GREEN-RL", "FULL-RH-A*", "success", True),
        ("static", "riskgreen_vs_fullplanner_MAR", "RISK+GREEN-RL", "FULL-RH-A*", "MAR_static", True),
        # Dynamic family (m=5)
        ("dynamic", "staticRL_vs_dynplanner_DynOK", "STATIC-RL-DYN", "DYN-RH-A*", "DynOK", False),
        ("dynamic", "dynriskRL_vs_dynplanner_DynOK", "DYN-RISK-RL", "DYN-RH-A*", "DynOK", False),
        ("dynamic", "dynobsRL_vs_dynplanner_SR", "DYN-OBS-RL", "DYN-RH-A*", "success", True),
        ("dynamic", "dynobsRL_vs_dynplanner_MAR", "DYN-OBS-RL", "DYN-RH-A*", "MAR_dynamic", True),
        ("dynamic", "dynobsRL_vs_dynplanner_DynOK", "DYN-OBS-RL", "DYN-RH-A*", "DynOK", True),
        # Vienna family (m=4)
        ("vienna", "zero_vs_extended_MAR", "VIENNA-ZERO", "VIENNA-EXT", "MAR_static", True),
        ("vienna", "extended_vs_planner_SR", "VIENNA-EXT", "VIENNA-PLANNER", "success", True),
        ("vienna", "extended_vs_planner_MAR", "VIENNA-EXT", "VIENNA-PLANNER", "MAR_static", True),
        ("vienna", "geo_vs_extended_MAR", "VIENNA-GEO", "VIENNA-EXT", "MAR_static", False),
    ],
    columns=["family", "comparison_id", "method_a", "method_b", "metric", "claim_driving"],
)

# Example after result CSVs are available:
# CSV_REGISTRY = {
#     "GEO-RL": "/.../episodes.csv",
#     "RISK+GREEN-RL": "/.../episodes.csv",
#     "RISK+OBS-RL": "/.../episodes.csv",
#     "FULL-RH-A*": "/.../episodes.csv",
#     ...
# }
# holm_full = compute_holm_families(HOLM_COMPARISON_TEMPLATE, CSV_REGISTRY)
# holm_full.to_csv(TABLE_DIR / "paper_table_holm_full_families.csv", index=False)
# holm_full[holm_full["claim_driving"]].to_csv(
#     TABLE_DIR / "paper_table_holm_claim_driving_subset.csv", index=False
# )


# %% [markdown]
# 7. Warsaw Fig. 9 status validation and regeneration

# %%
# ============================================================
# 7. Validate/select Warsaw trajectories from no-terminal output
# ============================================================


def validate_episode_acceptability(row: pd.Series) -> tuple[bool, dict[str, bool]]:
    checks = {
        "success": bool(row["is_success"]),
        "cargo": float(row.get("cargo_health", 1.0)) >= DEFAULT_BUDGETS["cargo_min"],
        "road": float(row.get("road_time", 0.0)) <= DEFAULT_BUDGETS["road_s"],
        "pedestrian": float(row.get("sidewalk_time", 0.0)) <= DEFAULT_BUDGETS["ped_s"],
        "building": float(row.get("building_time", 0.0)) <= DEFAULT_BUDGETS["building_s"],
        "sensitive": float(row.get("sensitive_time", 0.0)) <= DEFAULT_BUDGETS["sensitive_s"],
        "hard": float(row.get("urban_hard_violation_steps", 0.0)) <= 0.0,
    }
    return all(checks.values()), checks


def select_warsaw_figure_episodes(df: pd.DataFrame) -> dict[str, pd.Series]:
    d = add_derived_metrics(df)
    d["accepted_verified"] = recompute_static_acceptability(d)
    accepted = d[d["accepted_verified"]].copy()
    successful_unacceptable = d[
        as_bool_series(d["is_success"]) & ~as_bool_series(d["accepted_verified"])
    ].copy()
    if accepted.empty:
        raise RuntimeError("No verified accepted Warsaw episode in this CSV.")
    if successful_unacceptable.empty:
        raise RuntimeError("No successful-but-unacceptable Warsaw episode in this CSV.")

    accepted = accepted.sort_values(
        ["sidewalk_time", "road_time", "urban_cumulative_risk"], ascending=True
    )
    successful_unacceptable = successful_unacceptable.sort_values(
        ["sidewalk_time", "road_time", "urban_cumulative_risk"], ascending=False
    )
    return {
        "accepted": accepted.iloc[0],
        "successful_unacceptable": successful_unacceptable.iloc[0],
    }


def path_file_for_episode(paths_dir: Path, row: pd.Series) -> Path:
    episode = int(row["episode"])
    seed = int(row["seed"])
    candidates = [
        paths_dir / f"episode_{episode:05d}_seed_{seed}_actual_path.npy",
        paths_dir / f"episode_{episode:05d}_seed_{seed}_path.npy",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Path file missing for episode={episode}, seed={seed}")


def regenerate_warsaw_fig9(
    world,
    episodes_csv: os.PathLike[str] | str,
    paths_dir: os.PathLike[str] | str,
    output_stem: os.PathLike[str] | str,
    agl_m: float = 30.0,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = pd.read_csv(episodes_csv)
    selected = select_warsaw_figure_episodes(df)
    paths_dir = Path(paths_dir)
    accepted_path = np.load(path_file_for_episode(paths_dir, selected["accepted"]))
    rejected_path = np.load(path_file_for_episode(paths_dir, selected["successful_unacceptable"]))

    risk_model = UrbanRiskModel(
        world=world,
        profile="balanced",
        hard_sensitive_zones=True,
        include_global_altitude_band_cost=True,
    )
    risk = risk_model.risk_map_at_agl(float(agl_m))

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.6))
    extent = [0, world.width, world.height, 0]

    # Semantic panel.
    axes[0].imshow(world.terrain_z, origin="upper", extent=extent, cmap="Greys", alpha=0.18)
    overlays = [
        (world.green_mask, "green", 0.25, "green"),
        (world.road_mask, "orange", 0.38, "road"),
        (world.sidewalk_mask, "gold", 0.40, "pedestrian"),
        (world.building_mask, "gray", 0.55, "building"),
        (world.sensitive_mask, "red", 0.36, "sensitive"),
    ]
    for mask, color, alpha, _ in overlays:
        rgba = np.zeros(mask.shape + (4,), dtype=float)
        active = np.asarray(mask) > 0.5
        rgb = plt.matplotlib.colors.to_rgb(color)
        rgba[active] = (*rgb, alpha)
        axes[0].imshow(rgba, origin="upper", extent=extent, interpolation="nearest")

    axes[0].plot(accepted_path[:, 0], accepted_path[:, 1], linewidth=1.8, label="Accepted")
    axes[0].plot(
        rejected_path[:, 0],
        rejected_path[:, 1],
        linewidth=1.8,
        linestyle="--",
        label="Successful but unacceptable",
    )
    axes[0].set_title("Warsaw semantic layers")
    axes[0].legend(fontsize=6, loc="upper left")

    im = axes[1].imshow(risk, origin="upper", extent=extent, cmap="inferno")
    axes[1].plot(accepted_path[:, 0], accepted_path[:, 1], linewidth=1.8, label="Accepted")
    axes[1].plot(
        rejected_path[:, 0],
        rejected_path[:, 1],
        linewidth=1.8,
        linestyle="--",
        label="Successful but unacceptable",
    )
    axes[1].set_title(f"Warsaw risk at {agl_m:.0f} m AGL")
    axes[1].legend(fontsize=6, loc="upper left")
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("risk", fontsize=7)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(
            output_stem.with_suffix(f".{ext}"),
            dpi=600 if ext == "png" else None,
            bbox_inches="tight",
        )
    plt.close(fig)

    report = {
        "accepted_episode": int(selected["accepted"]["episode"]),
        "accepted_seed": int(selected["accepted"]["seed"]),
        "accepted_checks": validate_episode_acceptability(selected["accepted"])[1],
        "successful_unacceptable_episode": int(selected["successful_unacceptable"]["episode"]),
        "successful_unacceptable_seed": int(selected["successful_unacceptable"]["seed"]),
        "successful_unacceptable_checks": validate_episode_acceptability(
            selected["successful_unacceptable"]
        )[1],
        "output_stem": str(output_stem),
    }
    atomic_json(output_stem.with_name(output_stem.name + "_selection.json"), report)
    return report


# %% [markdown]
# 8. Release manifest and submission readiness

# %%
# ============================================================
# 8. Release package audit
# ============================================================

REQUIRED_RELEASE_PATTERNS = [
    "paper_ready_v1/tables/paper_table_primary_no_terminal_static_training_seed_ci.csv",
    "paper_ready_v1/tables/paper_table_conditional_normalized_risk_metrics.csv",
    "paper_ready_v1/tables/paper_table_non_acceptability_cause_decomposition.csv",
    "paper_ready_v1/tables/paper_table_holm_full_families.csv",
    "paper_ready_v1/tables/paper_table_dynamic_budget_sensitivity.csv",
    "experiments/paperA_submission_repair_v1/provenance/claim_artifact_manifest.csv",
]


def release_readiness(root: Path = DRIVE_ROOT) -> pd.DataFrame:
    rows = []
    for relative in REQUIRED_RELEASE_PATTERNS:
        p = root / relative
        rows.append(
            {
                "relative_path": relative,
                "exists": p.exists(),
                "size_bytes": int(p.stat().st_size) if p.exists() else np.nan,
                "sha256": sha256_file(p) if p.exists() else "",
            }
        )
    return pd.DataFrame(rows)


def create_release_manifest(root: Path = DRIVE_ROOT) -> tuple[Path, Path]:
    readiness = release_readiness(root)
    readiness_path = RELEASE_DIR / "submission_release_readiness.csv"
    readiness.to_csv(readiness_path, index=False)

    all_files = []
    for p in sorted(SUBMISSION_REPAIR_ROOT.rglob("*")):
        if p.is_file():
            all_files.append(
                {
                    "relative_path": str(p.relative_to(DRIVE_ROOT)),
                    "size_bytes": int(p.stat().st_size),
                    "sha256": sha256_file(p),
                }
            )
    manifest_path = RELEASE_DIR / "submission_repair_file_manifest.csv"
    pd.DataFrame(all_files).to_csv(manifest_path, index=False)
    return readiness_path, manifest_path


RUN_RELEASE_AUDIT = False
if RUN_RELEASE_AUDIT:
    ready_path, manifest_path = create_release_manifest(DRIVE_ROOT)
    print("Saved:", ready_path)
    print("Saved:", manifest_path)
    display(pd.read_csv(ready_path))


# %% [markdown]
# Final manuscript-update mapping
# --------------------------------
# Use the generated files as follows:
#
# - Table I / Fig. 4: primary_no_terminal static seed-level and planner outputs.
# - Table II / Fig. 6: primary_no_terminal dynamic outputs plus beta_dyn sensitivity.
# - Table III: primary_no_terminal Vienna outputs.
# - Table IV: paper_table_holm_full_families.csv; the main table may show only the
#   claim-driving subset, but the supplement must contain every family member.
# - Tables XII-XIII: rerun with primary_no_terminal if retained as primary claims;
#   otherwise relabel existing strict results as operational-stopping stress tests.
# - Fig. 9: regenerate from a scenario-aligned Warsaw no-terminal planner CSV using
#   regenerate_warsaw_fig9().
# - Appendix A6: use the exact REPORTED_* dictionaries above, not example/quickstart
#   configuration values.
# - Data Availability: publish all checkpoint, world, scenario, config, per-episode,
#   provenance, and checksum artifacts before inserting the final DOI/version.
