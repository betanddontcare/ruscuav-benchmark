# -*- coding: utf-8 -*-
"""
RUSCUAV submission-repair: one-click master runner v3
==================================================

Usage in the OLD TOP_UAV Colab notebook:
1. Mount Google Drive.
2. Run TOP_UAV definition cells.
3. Run all RUSCUAV_submission_repair_cells definitions once (RUN_* may stay False).
4. Optionally set explicit checkpoint paths in CHECKPOINT_OVERRIDES below.
5. Execute this entire file in the current notebook namespace, e.g.:

   %run -i /content/drive/MyDrive/RUSCUAV/code/RUSCUAV_submission_repair_one_click_runner_v3.py

The runner is resume-safe. Re-running it skips completed master stages and the
underlying evaluators resume from per-episode CSV files.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd


# ============================================================================
# USER CONFIGURATION — normally this is the only section you may need to edit.
# ============================================================================

RUNNER_VERSION = "v3"
AUTO_START = True
FAIL_FAST = True
SKIP_COMPLETED_MASTER_STAGES = True
FORCE_MASTER_STAGES: set[str] = set()  # e.g. {"postprocess", "release_audit"}

# Set a path only when automatic discovery cannot identify one unambiguously.
CHECKPOINT_OVERRIDES: dict[str, Optional[str]] = {
    "RISK+OBS-RL": None,          # M6 full-semantic checkpoint
    "DYN-RISK-RL": None,         # M11B dynamic-trained, no dynamic obs.
    "VIENNA_ZERO_SHOT": None,     # defaults to M8 reward-only seed 1001
    "VIENNA_SHORT_FT": None,
    "VIENNA_EXTENDED_FT": None,
}

# All stages needed for the submission repair. Set a stage to False only if you
# deliberately want to omit it. No PPO training is performed.
MASTER_PLAN: dict[str, bool] = {
    "provenance": True,
    "static_primary_rl": True,
    "static_primary_planners": True,
    "dynamic_primary": True,
    "vienna_primary": True,
    "warsaw_primary": True,
    "postprocess": True,
    "holm": True,
    "warsaw_fig9": True,
    "release_audit": True,
}

# If True, unresolved M6/M11B/Vienna fine-tuned checkpoints stop the pipeline
# before any expensive evaluation begins. This is safer for a final submission run.
REQUIRE_COMPLETE_CHECKPOINT_SET = True

# The historical Vienna short-fine-tuning checkpoint may be absent because the
# same Stage-C path was later continued for the extended fine-tuning run. v3
# first searches numbered immutable checkpoints near 75k Stage-C steps. If none
# exists, the short-FT row is skipped rather than silently reusing the extended
# model. All other repair stages continue.
ALLOW_MISSING_VIENNA_SHORT_FT = True
VIENNA_SHORT_TARGET_STEPS = 75_000
VIENNA_SHORT_MAX_DISTANCE_STEPS = 35_000
VIENNA_EXTENDED_TARGET_STEPS = 250_000
VIENNA_EXTENDED_MAX_DISTANCE_STEPS = 100_000


# ============================================================================
# PREFLIGHT
# ============================================================================

REQUIRED_ONE_CLICK_GLOBALS = [
    # Repair definitions
    "DRIVE_ROOT", "TABLE_DIR", "FIG_DIR", "SUBMISSION_REPAIR_ROOT",
    "PROVENANCE_DIR", "PRIMARY_EVAL_DIR", "RELEASE_DIR",
    "PRIMARY_NO_TERMINAL", "SECONDARY_OPERATIONAL_STOP",
    "collect_claim_artifacts", "extract_run_config_rows", "compare_expected_config",
    "REPORTED_M10_LOCAL_CONFIG", "REPORTED_DYNAMIC_CONFIG",
    "UrbanWorldLayers", "CANONICAL_WORLD_PATH", "MODEL_REGISTRY", "SINGLE_CHECKPOINTS",
    "require_model", "evaluate_ppo_resume", "make_static_eval_env",
    "setup_ruscuav_drive_project", "M9PlannerConfig", "M10RecedingHorizonPlannerConfig",
    "REPORTED_M9_CONFIG", "REPORTED_M10_LOCAL_CONFIG", "REPORTED_M10_FULL_CONFIG",
    "evaluate_m9_static_planner_drive", "evaluate_m10_receding_planner_drive",
    "reported_dynamic_config", "evaluate_m11_dynamic_rl_drive",
    "evaluate_m11c_dynamic_obs_rl_drive", "evaluate_dynamic_planner_mode",
    "REPORTED_M11_PLANNER_CONFIG", "get_real_worlds", "evaluate_m12c_real_policy_drive",
    "build_derived_tables_from_csv_registry", "dynamic_budget_sensitivity",
    "compute_holm_families", "HOLM_COMPARISON_TEMPLATE",
    "regenerate_warsaw_fig9", "create_release_manifest", "sha256_file",
    "summarize_episode_dataframe", "add_derived_metrics", "training_seed_ci",
    "utc_now_iso",
]

missing = [name for name in REQUIRED_ONE_CLICK_GLOBALS if name not in globals()]
if missing:
    raise RuntimeError(
        "One-click runner cannot start. First execute the TOP_UAV definition cells "
        "and all repair-definition cells. Missing:\n- " + "\n- ".join(missing)
    )

DRIVE_ROOT = Path(DRIVE_ROOT)
TABLE_DIR = Path(TABLE_DIR)
FIG_DIR = Path(FIG_DIR)
SUBMISSION_REPAIR_ROOT = Path(SUBMISSION_REPAIR_ROOT)
MASTER_DIR = SUBMISSION_REPAIR_ROOT / "master_run"
MASTER_DIR.mkdir(parents=True, exist_ok=True)
MASTER_STATE_PATH = MASTER_DIR / "master_run_state.json"
MASTER_LOG_PATH = MASTER_DIR / "master_run.log"
MASTER_RESOLVED_CHECKPOINTS = MASTER_DIR / "resolved_checkpoints.csv"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_state() -> dict[str, Any]:
    if MASTER_STATE_PATH.exists():
        try:
            return json.loads(MASTER_STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"created_at": _now(), "stages": {}}


def _save_state(state: dict[str, Any]) -> None:
    tmp = MASTER_STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, MASTER_STATE_PATH)


MASTER_STATE = _load_state()


def _existing_model(path_value: Optional[str | Path]) -> Optional[Path]:
    if not path_value:
        return None
    p = Path(path_value)
    return p if p.exists() and p.suffix.lower() == ".zip" else None


def _model_candidates_from_tables(keywords: list[str]) -> list[Path]:
    candidates: list[Path] = []
    for table_path in sorted(TABLE_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(table_path)
        except Exception:
            continue
        model_cols = [c for c in df.columns if "model" in c.lower() and "path" in c.lower()]
        if not model_cols:
            continue
        for _, row in df.iterrows():
            text = " ".join(str(v) for v in row.values).lower()
            if not all(k.lower() in text for k in keywords):
                continue
            for col in model_cols:
                p = _existing_model(row.get(col))
                if p is not None:
                    candidates.append(p)
    return sorted(set(candidates))


def _model_candidates_from_drive(keyword_sets: list[list[str]]) -> list[Path]:
    zips = list((DRIVE_ROOT / "experiments").rglob("*.zip"))
    scored: list[tuple[int, Path]] = []
    for p in zips:
        text = str(p).lower()
        best = -1
        for terms in keyword_sets:
            if all(term.lower() in text for term in terms):
                score = 10 * len(terms)
                if "final_c" in text or "final-c" in text:
                    score += 8
                elif "final" in text:
                    score += 5
                if "best" in text:
                    score += 2
                if "checkpoint" in text and "final" not in text:
                    score -= 2
                best = max(best, score)
        if best >= 0:
            scored.append((best, p))
    if not scored:
        return []
    max_score = max(score for score, _ in scored)
    return sorted({p for score, p in scored if score == max_score})


def _resolve_unique_model(
    key: str,
    table_keyword_sets: list[list[str]],
    drive_keyword_sets: list[list[str]],
    default: Optional[Path] = None,
) -> tuple[Optional[Path], list[Path]]:
    override = _existing_model(CHECKPOINT_OVERRIDES.get(key))
    if override is not None:
        return override, [override]

    existing = _existing_model(SINGLE_CHECKPOINTS.get(key))
    if existing is not None:
        return existing, [existing]

    if default is not None and default.exists():
        return default, [default]

    candidates: list[Path] = []
    for terms in table_keyword_sets:
        candidates.extend(_model_candidates_from_tables(terms))
    if not candidates:
        candidates.extend(_model_candidates_from_drive(drive_keyword_sets))
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0], candidates
    return None, candidates


def _newest_existing(paths: list[Path]) -> Optional[Path]:
    existing = [Path(p) for p in paths if Path(p).exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: (p.stat().st_mtime, str(p)))


def _newest_glob(directory: Path, pattern: str) -> Optional[Path]:
    if not directory.exists():
        return None
    return _newest_existing(list(directory.glob(pattern)))


def _completed_steps_from_checkpoint(path: Path) -> Optional[int]:
    """Extract immutable Stage-C completed-step count from checkpoint name."""
    match = re.search(r"completed_(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def _vienna_numbered_stage_candidates(m12c_root: Path) -> list[tuple[int, Path]]:
    """Find all immutable numbered Vienna reward-only Stage-C checkpoints."""
    rows: list[tuple[int, Path]] = []
    if not m12c_root.exists():
        return rows
    for path in m12c_root.rglob("*.zip"):
        low = str(path).lower()
        if "m12c_vienna_reward_only_c" not in low:
            continue
        completed = _completed_steps_from_checkpoint(path)
        if completed is None:
            continue
        rows.append((completed, path))
    # Deduplicate paths and order by completed steps, then modification time.
    unique = {str(path): (steps, path) for steps, path in rows}
    return sorted(unique.values(), key=lambda x: (x[0], x[1].stat().st_mtime, str(x[1])))


def _nearest_vienna_checkpoint(
    candidates: list[tuple[int, Path]],
    target_steps: int,
    max_distance_steps: int,
) -> Optional[Path]:
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda x: (abs(x[0] - int(target_steps)), -x[0], -x[1].stat().st_mtime),
    )
    steps, path = ranked[0]
    if abs(steps - int(target_steps)) > int(max_distance_steps):
        return None
    return path


def _checkpoint_resolution_note(path: Optional[Path], target: Optional[int] = None) -> str:
    if path is None:
        return "unresolved"
    completed = _completed_steps_from_checkpoint(path)
    if completed is None:
        return "mutable/final checkpoint; no completed-step count in filename"
    if target is None:
        return f"immutable numbered checkpoint; completed_stage_steps={completed}"
    return (
        f"immutable numbered checkpoint; completed_stage_steps={completed}; "
        f"target={target}; delta={completed-target:+d}"
    )


def resolve_checkpoints() -> dict[str, Optional[Path]]:
    zero_default = require_model("M8_reward_only", 1001)

    # Exact experiment-derived defaults. These are preferable to fuzzy matching:
    # - M6 full-semantic used the final Stage-C model in m5_m4_ppo_debug.
    # - M11B evaluation used m11b_paths["final_C"].
    # - Vienna short FT ended at 75,000 Stage-C steps.
    # - Vienna extended FT continued the same Stage-C run to 250,000 steps and
    #   was evaluated from the stage latest checkpoint.
    m6_default = _newest_existing([
        DRIVE_ROOT / "experiments" / "m5_m4_ppo_debug" / "models" / "ppo_ruscuav_m4_after_stage_C.zip",
    ])
    m11b_default = _newest_existing([
        DRIVE_ROOT / "experiments" / "m11b_dynamic_trained_reward_only" / "models" /
        "ppo_ruscuav_m4_after_stage_M11B_dynamic_reward_only_C.zip",
    ])

    m12c_root = DRIVE_ROOT / "experiments" / "m12C_real_vienna_footway_learning"
    m12c_stage_dir = m12c_root / "checkpoints" / "stage_M12C_vienna_reward_only_C"

    # SB3 completes rollouts in batches, so a nominal 75,000-step target often
    # produces an immutable checkpoint such as 75,776 or 77,824 rather than an
    # exact 75,000 filename. Search recursively and select the nearest numbered
    # checkpoint within a conservative tolerance.
    vienna_numbered = _vienna_numbered_stage_candidates(m12c_root)
    vienna_short_default = _nearest_vienna_checkpoint(
        vienna_numbered,
        target_steps=VIENNA_SHORT_TARGET_STEPS,
        max_distance_steps=VIENNA_SHORT_MAX_DISTANCE_STEPS,
    )
    vienna_extended_default = _nearest_vienna_checkpoint(
        vienna_numbered,
        target_steps=VIENNA_EXTENDED_TARGET_STEPS,
        max_distance_steps=VIENNA_EXTENDED_MAX_DISTANCE_STEPS,
    )
    if vienna_extended_default is None:
        vienna_extended_default = _newest_existing([
            m12c_stage_dir / "stage_M12C_vienna_reward_only_C_latest.zip",
            m12c_root / "models" / "ppo_ruscuav_m4_after_stage_M12C_vienna_reward_only_C.zip",
        ])

    # Never label the same mutable/final bytes as both short and extended FT.
    if (
        vienna_short_default is not None
        and vienna_extended_default is not None
        and vienna_short_default.resolve() == vienna_extended_default.resolve()
    ):
        vienna_short_default = None

    specs = {
        "RISK+OBS-RL": (
            [["m6"], ["full", "semantic"]],
            [["m6"], ["full_semantic"], ["full-semantic"]],
            m6_default,
        ),
        "DYN-RISK-RL": (
            [["m11b"], ["dynamic", "no dynamic obs"]],
            [["m11b"], ["dynamic_trained_reward_only"]],
            m11b_default,
        ),
        "VIENNA_ZERO_SHOT": (
            [["vienna", "zero"]],
            [["vienna", "zero"]],
            zero_default,
        ),
        "VIENNA_SHORT_FT": (
            [["vienna", "short"]],
            [["vienna", "short"], ["m12c", "short"]],
            vienna_short_default,
        ),
        "VIENNA_EXTENDED_FT": (
            [["vienna", "extended"]],
            [["vienna", "extended"], ["m12c", "extended"]],
            vienna_extended_default,
        ),
    }

    resolved: dict[str, Optional[Path]] = {}
    report_rows = []
    unresolved_required = []
    optional_warnings = []
    targets = {
        "VIENNA_SHORT_FT": VIENNA_SHORT_TARGET_STEPS,
        "VIENNA_EXTENDED_FT": VIENNA_EXTENDED_TARGET_STEPS,
    }
    for key, (table_terms, drive_terms, default) in specs.items():
        path, candidates = _resolve_unique_model(key, table_terms, drive_terms, default)
        resolved[key] = path
        if path is not None:
            SINGLE_CHECKPOINTS[key] = path
        elif key == "VIENNA_SHORT_FT" and ALLOW_MISSING_VIENNA_SHORT_FT:
            optional_warnings.append(
                "VIENNA_SHORT_FT was not recoverable as a distinct immutable checkpoint. "
                "The runner will skip this row and will not substitute the extended model."
            )
        else:
            unresolved_required.append(key)
        report_rows.append(
            {
                "key": key,
                "resolved_path": str(path) if path else "",
                "n_candidates": len(candidates),
                "completed_stage_steps": (
                    _completed_steps_from_checkpoint(path) if path is not None else np.nan
                ),
                "resolution_note": _checkpoint_resolution_note(path, targets.get(key)),
                "required": not (key == "VIENNA_SHORT_FT" and ALLOW_MISSING_VIENNA_SHORT_FT),
                "candidates": " | ".join(str(p) for p in candidates[:20]),
            }
        )

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(MASTER_RESOLVED_CHECKPOINTS, index=False)
    print("Checkpoint resolution:")
    print(report_df.to_string(index=False))

    warning_path = MASTER_DIR / "checkpoint_resolution_warnings.txt"
    if optional_warnings:
        warning_text = "\n".join("WARNING: " + x for x in optional_warnings)
        warning_path.write_text(warning_text + "\n", encoding="utf-8")
        print("\n" + warning_text)
        print("Saved warning:", warning_path)

    if unresolved_required and REQUIRE_COMPLETE_CHECKPOINT_SET:
        raise RuntimeError(
            "Unresolved required checkpoint paths: " + ", ".join(unresolved_required) +
            ". Set them once in CHECKPOINT_OVERRIDES at the top of the one-click runner. "
            f"Candidate report: {MASTER_RESOLVED_CHECKPOINTS}"
        )
    return resolved


RESOLVED_CHECKPOINTS = resolve_checkpoints()


# ============================================================================
# STAGE HELPERS
# ============================================================================


def run_stage(name: str, fn: Callable[[], Any], critical: bool = True) -> Any:
    enabled = MASTER_PLAN.get(name, False)
    if not enabled:
        print(f"[SKIP PLAN] {name}")
        return None

    prior = MASTER_STATE.get("stages", {}).get(name, {})
    if (
        SKIP_COMPLETED_MASTER_STAGES
        and name not in FORCE_MASTER_STAGES
        and prior.get("status") == "complete"
    ):
        print(f"[SKIP COMPLETE] {name} | finished_at={prior.get('finished_at')}")
        return None

    print("\n" + "=" * 110)
    print(f"MASTER STAGE START: {name} | {_now()}")
    print("=" * 110)
    MASTER_STATE.setdefault("stages", {})[name] = {
        "status": "running",
        "started_at": _now(),
    }
    _save_state(MASTER_STATE)

    try:
        result = fn()
        MASTER_STATE["stages"][name].update(
            {"status": "complete", "finished_at": _now()}
        )
        _save_state(MASTER_STATE)
        print(f"MASTER STAGE COMPLETE: {name} | {_now()}")
        return result
    except Exception as exc:
        MASTER_STATE["stages"][name].update(
            {
                "status": "failed",
                "failed_at": _now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _save_state(MASTER_STATE)
        print(f"MASTER STAGE FAILED: {name}: {exc}")
        traceback.print_exc()
        if critical and FAIL_FAST:
            raise
        return None


# ============================================================================
# STAGES
# ============================================================================


def stage_provenance():
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
    audit = pd.concat(audit_parts, ignore_index=True)
    audit_path = PROVENANCE_DIR / "reported_configuration_audit.csv"
    audit.to_csv(audit_path, index=False)
    print("Saved:", artifact_path)
    print("Saved:", config_rows_path)
    print("Saved:", audit_path)
    return audit


def _write_static_aggregate(static_seed_df: pd.DataFrame) -> None:
    metrics = [
        "success_rate", "mission_acceptability_rate", "acceptable_given_success_rate",
        "mean_urban_cumulative_risk", "mean_urban_cumulative_risk_given_success",
        "mean_urban_risk_per_step", "mean_urban_risk_per_s", "mean_urban_risk_per_m_xy",
        "mean_road_fraction", "mean_ped_fraction", "mean_building_fraction",
    ]
    rows = []
    for method, group in static_seed_df.groupby("method"):
        for metric in metrics:
            if metric in group.columns:
                rows.append({"method": method, "metric": metric, **training_seed_ci(group[metric])})
    pd.DataFrame(rows).to_csv(
        TABLE_DIR / "paper_table_primary_no_terminal_static_training_seed_ci.csv",
        index=False,
    )


def stage_static_primary_rl():
    world = UrbanWorldLayers.load_npz(CANONICAL_WORLD_PATH)
    mapping = {
        "GEO-RL": "M7_geometry_only",
        "OBS-RL": "M8_obs_only",
        "RISK+GREEN-RL": "M8_reward_only",
    }
    rows = []
    for short, registry_name in mapping.items():
        for train_seed in STATIC_TRAIN_SEEDS:
            model_path = require_model(registry_name, train_seed)
            out_dir = PRIMARY_EVAL_DIR / "static" / short / f"trainseed_{train_seed}"
            _, summary, csv_path, summary_path = evaluate_ppo_resume(
                model_path=model_path,
                env_factory=lambda seed, m=short: make_static_eval_env(
                    m, world, PRIMARY_NO_TERMINAL, seed
                ),
                method=short,
                regime=PRIMARY_NO_TERMINAL,
                output_dir=out_dir,
                n_episodes=STATIC_N_EPISODES,
                seed0=STATIC_EVAL_SEED0,
                deterministic=True,
                save_paths=False,
            )
            rows.append({
                "method": short, "train_seed": train_seed,
                "csv_path": str(csv_path), "summary_path": str(summary_path),
                **{k: v for k, v in summary.items() if not isinstance(v, dict)},
            })

    # M6 full-semantic: explicit single-checkpoint reference.
    m6 = RESOLVED_CHECKPOINTS.get("RISK+OBS-RL")
    if m6 is not None:
        out_dir = PRIMARY_EVAL_DIR / "static" / "RISK+OBS-RL" / "single_checkpoint"
        _, summary, csv_path, summary_path = evaluate_ppo_resume(
            model_path=m6,
            env_factory=lambda seed: make_static_eval_env(
                "RISK+OBS-RL", world, PRIMARY_NO_TERMINAL, seed
            ),
            method="RISK+OBS-RL",
            regime=PRIMARY_NO_TERMINAL,
            output_dir=out_dir,
            n_episodes=STATIC_N_EPISODES,
            seed0=STATIC_EVAL_SEED0,
            deterministic=True,
            save_paths=False,
        )
        pd.DataFrame([{
            "method": "RISK+OBS-RL", "train_seed": np.nan,
            "csv_path": str(csv_path), "summary_path": str(summary_path),
            **{k: v for k, v in summary.items() if not isinstance(v, dict)},
        }]).to_csv(
            TABLE_DIR / "paper_table_primary_no_terminal_static_single_checkpoint.csv",
            index=False,
        )

    seed_df = pd.DataFrame(rows)
    seed_path = TABLE_DIR / "paper_table_primary_no_terminal_static_seed_level.csv"
    seed_df.to_csv(seed_path, index=False)
    _write_static_aggregate(seed_df)
    print("Saved:", seed_path)
    return seed_df


def stage_static_primary_planners():
    world = UrbanWorldLayers.load_npz(CANONICAL_WORLD_PATH)
    dirs = setup_ruscuav_drive_project(
        drive_root=str(DRIVE_ROOT), experiment_name="paperA_primary_no_terminal_static_planners"
    )
    m9_cfg = M9PlannerConfig(**REPORTED_M9_CONFIG)
    local_cfg = M10RecedingHorizonPlannerConfig(**REPORTED_M10_LOCAL_CONFIG)
    full_cfg = M10RecedingHorizonPlannerConfig(**REPORTED_M10_FULL_CONFIG)

    evaluate_m9_static_planner_drive(
        world=world, drive_dirs=dirs, planner_config=m9_cfg, risk_profile="balanced",
        n_episodes=STATIC_N_EPISODES, seed0=STATIC_EVAL_SEED0,
        stage=PRIMARY_NO_TERMINAL.stage, mode=PRIMARY_NO_TERMINAL.mode,
        n_obstacles=0, prefix="primary_no_terminal_STATIC_Astar_n200_seed5000",
        save_paths=True,
    )
    for name, cfg in [("LOCAL_RH_Astar", local_cfg), ("FULL_RH_Astar", full_cfg)]:
        evaluate_m10_receding_planner_drive(
            world=world, drive_dirs=dirs, cfg=cfg, risk_profile="balanced",
            n_episodes=STATIC_N_EPISODES, seed0=STATIC_EVAL_SEED0,
            stage=PRIMARY_NO_TERMINAL.stage, mode=PRIMARY_NO_TERMINAL.mode,
            n_obstacles=0, prefix=f"primary_no_terminal_{name}_n200_seed5000",
            save_paths=True,
        )


def stage_dynamic_primary():
    world = UrbanWorldLayers.load_npz(CANONICAL_WORLD_PATH)
    dyn_cfg = reported_dynamic_config()
    dirs = setup_ruscuav_drive_project(
        drive_root=str(DRIVE_ROOT), experiment_name="paperA_primary_no_terminal_dynamic"
    )
    for train_seed in STATIC_TRAIN_SEEDS:
        evaluate_m11_dynamic_rl_drive(
            model_path=require_model("M8_reward_only", train_seed),
            world=world, drive_dirs=dirs, dynamic_config=dyn_cfg,
            n_episodes=DYNAMIC_N_EPISODES, seed0=DYNAMIC_EVAL_SEED0,
            prefix=f"primary_no_terminal_static_trained_RISK_GREEN_seed{train_seed}",
            stage=PRIMARY_NO_TERMINAL.stage, mode=PRIMARY_NO_TERMINAL.mode,
            reward_only_obs=True, save_paths=False,
        )
    for train_seed in DYNAMIC_M11C_TRAIN_SEEDS:
        evaluate_m11c_dynamic_obs_rl_drive(
            model_path=require_model("M11C_dynamic_obs_reward_only", train_seed),
            world=world, drive_dirs=dirs, dynamic_config=dyn_cfg,
            n_episodes=DYNAMIC_N_EPISODES, seed0=DYNAMIC_EVAL_SEED0,
            prefix=f"primary_no_terminal_M11C_seed{train_seed}",
            stage=PRIMARY_NO_TERMINAL.stage, mode=PRIMARY_NO_TERMINAL.mode,
            save_paths=False,
        )
    m11b = RESOLVED_CHECKPOINTS.get("DYN-RISK-RL")
    if m11b is not None:
        evaluate_m11_dynamic_rl_drive(
            model_path=m11b, world=world, drive_dirs=dirs, dynamic_config=dyn_cfg,
            n_episodes=DYNAMIC_N_EPISODES, seed0=DYNAMIC_EVAL_SEED0,
            prefix="primary_no_terminal_M11B_single_checkpoint",
            stage=PRIMARY_NO_TERMINAL.stage, mode=PRIMARY_NO_TERMINAL.mode,
            reward_only_obs=True, save_paths=False,
        )
    planner_cfg = M10RecedingHorizonPlannerConfig(**REPORTED_M11_PLANNER_CONFIG)
    evaluate_dynamic_planner_mode(
        world=world, drive_dirs=dirs, cfg=planner_cfg, dynamic_config=dyn_cfg,
        regime=PRIMARY_NO_TERMINAL, n_episodes=DYNAMIC_N_EPISODES,
        seed0=DYNAMIC_EVAL_SEED0, prefix="primary_no_terminal_dynamic_receding_Astar",
        dynamic_prior_mode="zero", save_paths=False,
    )


def stage_vienna_primary():
    (world, _), _ = get_real_worlds()
    dirs = setup_ruscuav_drive_project(
        drive_root=str(DRIVE_ROOT), experiment_name="paperA_primary_no_terminal_vienna"
    )
    for label, key in [
        ("Vienna_zero_shot_RISK_GREEN", "VIENNA_ZERO_SHOT"),
        ("Vienna_short_finetuned_RISK_GREEN", "VIENNA_SHORT_FT"),
        ("Vienna_extended_finetuned_RISK_GREEN", "VIENNA_EXTENDED_FT"),
    ]:
        checkpoint = RESOLVED_CHECKPOINTS.get(key)
        if checkpoint is None:
            if key == "VIENNA_SHORT_FT" and ALLOW_MISSING_VIENNA_SHORT_FT:
                print(
                    "[SKIP OPTIONAL] Vienna short fine-tuning checkpoint is not "
                    "recoverable as a distinct file. The extended checkpoint will "
                    "not be reused under the short-FT label."
                )
                continue
            raise RuntimeError(f"Missing Vienna checkpoint: {key}")
        evaluate_m12c_real_policy_drive(
            model_path=checkpoint, world=world, policy_kind="reward_only",
            drive_dirs=dirs, n_episodes=REAL_N_EPISODES, seed0=REAL_EVAL_SEED0,
            prefix=f"primary_no_terminal_{label}", stage=PRIMARY_NO_TERMINAL.stage,
            mode=PRIMARY_NO_TERMINAL.mode, deterministic=True, save_paths=True,
        )

    # Geometry-only fixed checkpoint for the complete exploratory Vienna Holm family.
    geo_model = require_model("M7_geometry_only", 1001)
    evaluate_ppo_resume(
        model_path=geo_model,
        env_factory=lambda seed: make_static_eval_env("GEO-RL", world, PRIMARY_NO_TERMINAL, seed),
        method="VIENNA-GEO", regime=PRIMARY_NO_TERMINAL,
        output_dir=Path(dirs["eval"]) / "VIENNA-GEO" / "trainseed_1001",
        n_episodes=REAL_N_EPISODES, seed0=REAL_EVAL_SEED0,
        deterministic=True, save_paths=True,
    )

    planner_cfg = M10RecedingHorizonPlannerConfig(
        **{**REPORTED_M10_FULL_CONFIG, "planner_agl_m": 30.0}
    )
    evaluate_m10_receding_planner_drive(
        world=world, drive_dirs=dirs, cfg=planner_cfg, risk_profile="balanced",
        n_episodes=REAL_N_EPISODES, seed0=REAL_EVAL_SEED0,
        stage=PRIMARY_NO_TERMINAL.stage, mode=PRIMARY_NO_TERMINAL.mode,
        n_obstacles=0, prefix="primary_no_terminal_Vienna_FULL_RH_Astar",
        save_paths=True,
    )


def stage_warsaw_primary():
    _, (world, _) = get_real_worlds()
    dirs = setup_ruscuav_drive_project(
        drive_root=str(DRIVE_ROOT),
        experiment_name="paperA_primary_no_terminal_warsaw_scenario_aligned",
    )
    mapping = {
        "GEO-RL": "M7_geometry_only",
        "OBS-RL": "M8_obs_only",
        "RISK+GREEN-RL": "M8_reward_only",
    }
    for short, registry_name in mapping.items():
        for train_seed in STATIC_TRAIN_SEEDS:
            evaluate_ppo_resume(
                model_path=require_model(registry_name, train_seed),
                env_factory=lambda seed, m=short: make_static_eval_env(
                    m, world, PRIMARY_NO_TERMINAL, seed
                ),
                method=short, regime=PRIMARY_NO_TERMINAL,
                output_dir=Path(dirs["eval"]) / short / f"trainseed_{train_seed}",
                n_episodes=REAL_N_EPISODES, seed0=REAL_EVAL_SEED0,
                deterministic=True, save_paths=True,
            )
    cfg = M10RecedingHorizonPlannerConfig(
        **{**REPORTED_M10_FULL_CONFIG, "planner_agl_m": 30.0}
    )
    for prefix, regime in [
        ("primary_no_terminal_Warsaw_FULL_RH_Astar_seed7000", PRIMARY_NO_TERMINAL),
        ("secondary_operational_stopping_Warsaw_FULL_RH_Astar_seed7000", SECONDARY_OPERATIONAL_STOP),
    ]:
        evaluate_m10_receding_planner_drive(
            world=world, drive_dirs=dirs, cfg=cfg, risk_profile="balanced",
            n_episodes=REAL_N_EPISODES, seed0=REAL_EVAL_SEED0,
            stage=regime.stage, mode=regime.mode, n_obstacles=0,
            prefix=prefix, save_paths=True,
        )


def _canonical_label(path: Path) -> str:
    text = str(path).lower()
    if "static_trained_risk_green" in text:
        return "STATIC-RL-DYN"
    if "m11c" in text:
        return "DYN-OBS-RL"
    if "m11b" in text:
        return "DYN-RISK-RL"
    if "dynamic_receding" in text:
        return "DYN-RH-A*"
    if "vienna_zero" in text:
        return "VIENNA-ZERO"
    if "vienna_extended" in text:
        return "VIENNA-EXT"
    if "vienna_short" in text:
        return "VIENNA-SHORT"
    if "vienna_full_rh" in text:
        return "VIENNA-PLANNER"
    if "vienna-geo" in text:
        return "VIENNA-GEO"
    if "risk+obs-rl" in text:
        return "RISK+OBS-RL"
    if "risk+green-rl" in text:
        return "RISK+GREEN-RL"
    if "obs-rl" in text:
        return "OBS-RL"
    if "geo-rl" in text:
        return "GEO-RL"
    if "full_rh_astar" in text or "full-rh-astar" in text:
        return "FULL-RH-A*"
    if "local_rh_astar" in text or "local-rh-astar" in text:
        return "LOCAL-RH-A*"
    if "static_astar" in text:
        return "STATIC-A*"
    return path.parent.name


def collect_primary_registry() -> pd.DataFrame:
    roots = [
        PRIMARY_EVAL_DIR,
        DRIVE_ROOT / "experiments" / "paperA_primary_no_terminal_static_planners" / "eval",
        DRIVE_ROOT / "experiments" / "paperA_primary_no_terminal_dynamic" / "eval",
        DRIVE_ROOT / "experiments" / "paperA_primary_no_terminal_vienna" / "eval",
        DRIVE_ROOT / "experiments" / "paperA_primary_no_terminal_warsaw_scenario_aligned" / "eval",
    ]
    rows = []
    seen = set()
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.csv")):
            if "episode" not in p.name.lower() and p.name != "episodes.csv":
                continue
            if p in seen:
                continue
            seen.add(p)
            text = str(p).lower()
            seed_match = re.search(r"(?:trainseed_|seed)(\d{4})", text)
            rows.append({
                "label": _canonical_label(p),
                "csv_path": str(p),
                "dynamic": "primary_no_terminal_dynamic" in text,
                "train_seed": int(seed_match.group(1)) if seed_match else np.nan,
                "scope": (
                    "vienna" if "vienna" in text else
                    "warsaw" if "warsaw" in text else
                    "dynamic" if "primary_no_terminal_dynamic" in text else
                    "static"
                ),
                "regime": "secondary_operational_stopping" if "secondary_operational" in text else "primary_no_terminal",
            })
    registry = pd.DataFrame(rows)
    registry_path = TABLE_DIR / "paper_table_primary_no_terminal_csv_registry.csv"
    registry.to_csv(registry_path, index=False)
    print("Saved:", registry_path)
    return registry


def stage_postprocess():
    registry = collect_primary_registry()
    primary = registry[registry["regime"].eq("primary_no_terminal")].copy()
    build_derived_tables_from_csv_registry(primary)

    # Dynamic-budget sensitivity for every primary dynamic run.
    beta_parts = []
    for _, row in primary[primary["dynamic"]].iterrows():
        df = pd.read_csv(row["csv_path"])
        part = dynamic_budget_sensitivity(df, label=str(row["label"]))
        part["train_seed"] = row.get("train_seed", np.nan)
        part["csv_path"] = row["csv_path"]
        beta_parts.append(part)
    beta = pd.concat(beta_parts, ignore_index=True) if beta_parts else pd.DataFrame()
    beta_path = TABLE_DIR / "paper_table_dynamic_budget_sensitivity.csv"
    beta.to_csv(beta_path, index=False)
    print("Saved:", beta_path)

    # One summary row per run and seed-level aggregate where applicable.
    summaries = []
    for _, row in primary.iterrows():
        df = pd.read_csv(row["csv_path"])
        summary = summarize_episode_dataframe(df, dynamic=bool(row["dynamic"]))
        summaries.append({**row.to_dict(), **{k: v for k, v in summary.items() if not isinstance(v, dict)}})
    summary_df = pd.DataFrame(summaries)
    summary_path = TABLE_DIR / "paper_table_primary_no_terminal_all_run_summaries.csv"
    summary_df.to_csv(summary_path, index=False)

    agg_rows = []
    metrics = ["success_rate", "mission_acceptability_rate", "acceptable_given_success_rate"]
    for (scope, label), group in summary_df.dropna(subset=["train_seed"]).groupby(["scope", "label"]):
        if group["train_seed"].nunique() < 2:
            continue
        for metric in metrics:
            if metric in group.columns:
                agg_rows.append({"scope": scope, "method": label, "metric": metric, **training_seed_ci(group[metric])})
    agg_path = TABLE_DIR / "paper_table_primary_no_terminal_training_seed_ci_all_scopes.csv"
    pd.DataFrame(agg_rows).to_csv(agg_path, index=False)
    print("Saved:", summary_path)
    print("Saved:", agg_path)
    return registry


def _first_glob(pattern: str) -> Optional[Path]:
    hits = sorted(Path("/").glob(pattern.lstrip("/"))) if "**" not in pattern else []
    if hits:
        return hits[0]
    import glob as _glob
    matches = sorted(_glob.glob(pattern, recursive=True))
    return Path(matches[0]) if matches else None


def _find_csv(root: Path, includes: list[str]) -> Optional[Path]:
    hits = []
    if root.exists():
        for p in root.rglob("*.csv"):
            text = str(p).lower()
            if ("episode" in p.name.lower() or p.name == "episodes.csv") and all(x.lower() in text for x in includes):
                hits.append(p)
    return sorted(hits)[0] if hits else None


def stage_holm():
    static_root = PRIMARY_EVAL_DIR / "static"
    static_planner_root = DRIVE_ROOT / "experiments" / "paperA_primary_no_terminal_static_planners" / "eval"
    dynamic_root = DRIVE_ROOT / "experiments" / "paperA_primary_no_terminal_dynamic" / "eval"
    vienna_root = DRIVE_ROOT / "experiments" / "paperA_primary_no_terminal_vienna" / "eval"

    registry = {
        "GEO-RL": static_root / "GEO-RL" / "trainseed_1001" / "episodes.csv",
        "RISK+GREEN-RL": static_root / "RISK+GREEN-RL" / "trainseed_1001" / "episodes.csv",
        "RISK+OBS-RL": static_root / "RISK+OBS-RL" / "single_checkpoint" / "episodes.csv",
        "FULL-RH-A*": _find_csv(static_planner_root, ["full_rh_astar"]),
        "STATIC-RL-DYN": _find_csv(dynamic_root, ["static_trained_risk_green_seed1001"]),
        "DYN-RISK-RL": _find_csv(dynamic_root, ["m11b"]),
        "DYN-OBS-RL": _find_csv(dynamic_root, ["m11c_seed1001"]),
        "DYN-RH-A*": _find_csv(dynamic_root, ["dynamic_receding"]),
        "VIENNA-ZERO": _find_csv(vienna_root, ["vienna_zero"]),
        "VIENNA-EXT": _find_csv(vienna_root, ["vienna_extended"]),
        "VIENNA-PLANNER": _find_csv(vienna_root, ["vienna_full_rh"]),
        "VIENNA-GEO": vienna_root / "VIENNA-GEO" / "trainseed_1001" / "episodes.csv",
    }
    missing = [k for k, p in registry.items() if p is None or not Path(p).exists()]
    if missing:
        pd.DataFrame({"missing_label": missing}).to_csv(
            TABLE_DIR / "paper_table_holm_missing_inputs.csv", index=False
        )
        raise RuntimeError("Holm inputs missing: " + ", ".join(missing))

    holm = compute_holm_families(HOLM_COMPARISON_TEMPLATE, {k: str(v) for k, v in registry.items()})
    full_path = TABLE_DIR / "paper_table_holm_full_families.csv"
    claim_path = TABLE_DIR / "paper_table_holm_claim_driving_subset.csv"
    holm.to_csv(full_path, index=False)
    holm[holm["claim_driving"]].to_csv(claim_path, index=False)
    print("Saved:", full_path)
    print("Saved:", claim_path)
    return holm


def stage_warsaw_fig9():
    _, (world, _) = get_real_worlds()
    root = DRIVE_ROOT / "experiments" / "paperA_primary_no_terminal_warsaw_scenario_aligned" / "eval"
    csv_path = _find_csv(root, ["primary_no_terminal_warsaw_full_rh_astar"])
    if csv_path is None:
        raise FileNotFoundError("Warsaw no-terminal planner episodes CSV not found")
    report = regenerate_warsaw_fig9(
        world=world,
        episodes_csv=csv_path,
        paths_dir=csv_path.parent / "paths",
        output_stem=FIG_DIR / "fig09_warsaw_verified_primary_no_terminal",
        agl_m=30.0,
    )
    print(json.dumps(report, indent=2))
    return report


def stage_release_audit():
    readiness_path, manifest_path = create_release_manifest(DRIVE_ROOT)
    print("Saved:", readiness_path)
    print("Saved:", manifest_path)
    print(pd.read_csv(readiness_path).to_string(index=False))
    return readiness_path, manifest_path


# ============================================================================
# MASTER EXECUTION
# ============================================================================


def run_all_submission_repairs():
    print("\n" + "#" * 110)
    print("RUSCUAV ONE-CLICK SUBMISSION REPAIR START")
    print("Started:", _now())
    print("Log:", MASTER_LOG_PATH)
    print("State:", MASTER_STATE_PATH)
    print("#" * 110)

    run_stage("provenance", stage_provenance)
    run_stage("static_primary_rl", stage_static_primary_rl)
    run_stage("static_primary_planners", stage_static_primary_planners)
    run_stage("dynamic_primary", stage_dynamic_primary)
    run_stage("vienna_primary", stage_vienna_primary)
    run_stage("warsaw_primary", stage_warsaw_primary)
    run_stage("postprocess", stage_postprocess)
    run_stage("holm", stage_holm)
    run_stage("warsaw_fig9", stage_warsaw_fig9, critical=False)
    run_stage("release_audit", stage_release_audit, critical=False)

    MASTER_STATE["finished_at"] = _now()
    _save_state(MASTER_STATE)
    print("\n" + "#" * 110)
    print("RUSCUAV ONE-CLICK SUBMISSION REPAIR FINISHED")
    print("Finished:", _now())
    print("State:", MASTER_STATE_PATH)
    print("#" * 110)
    return MASTER_STATE


if AUTO_START:
    with MASTER_LOG_PATH.open("a", encoding="utf-8", buffering=1) as log_file:
        tee = Tee(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            run_all_submission_repairs()
