"""Archived checkpoints, the evaluation registry, the post-hoc step and the policy
summaries (no environment needed).

The post-hoc step is compared with the notebook's pandas code (copied below), and,
when RUSCUAV_ARCHIVE points to the extracted v2.0.0 archive, with the archived
per-episode CSVs and summaries of every primary policy evaluation.
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

from ruscuav_benchmark.env.config import METHOD_PRESETS  # noqa: E402
from ruscuav_benchmark.policies import (  # noqa: E402
    ARCHIVED_CHECKPOINTS,
    ARCHIVED_EVALUATIONS,
    ROW_FORMATS,
    archived_checkpoint,
    summarize_policy_rows,
)
from ruscuav_benchmark.posthoc import (  # noqa: E402
    derived_metrics,
    dynamic_acceptability,
    static_acceptability,
    summarize_posthoc,
    with_derived_metrics,
)

ARCHIVE = os.environ.get("RUSCUAV_ARCHIVE")
PLANNER_DYN = "experiments/paperA_primary_no_terminal_dynamic/eval/primary_no_terminal_dynamic_receding_Astar/" \
              "primary_no_terminal_dynamic_receding_Astar"


# ---------------------------------------------------------------------- registry
def test_checkpoint_registry_is_consistent():
    assert len(ARCHIVED_CHECKPOINTS) == 41
    paths = set()
    for key, entry in ARCHIVED_CHECKPOINTS.items():
        assert entry.key == key and entry.method in METHOD_PRESETS
        assert len(entry.sha256) == 64 and entry.size_bytes > 0 and entry.path.endswith(".zip")
        paths.add(entry.path)
    assert len(paths) == len(ARCHIVED_CHECKPOINTS)
    for method in ("GEO-RL", "OBS-RL", "RISK+GREEN-RL", "RISK-RL", "GREEN-RL", "DYN-OBS-RL"):
        assert all(f"{method}/{seed}" in ARCHIVED_CHECKPOINTS for seed in (1001, 2001, 3001, 4001, 5001))
    # The final Vienna RISK+GREEN-RL model is the extended fine-tuning checkpoint.
    assert ARCHIVED_CHECKPOINTS["VIENNA/RISK+GREEN-RL/final"].sha256 == \
        ARCHIVED_CHECKPOINTS["VIENNA/RISK+GREEN-RL/extended_ft"].sha256


def test_evaluation_registry_is_consistent():
    assert len(ARCHIVED_EVALUATIONS) == 31
    counts = {"static": 0, "dynamic": 0, "vienna": 0}
    for name, entry in ARCHIVED_EVALUATIONS.items():
        counts[name.split("/")[0]] += 1
        assert entry.checkpoint in ARCHIVED_CHECKPOINTS and entry.method in METHOD_PRESETS
        assert entry.row_format in ROW_FORMATS and entry.world in ("canonical", "vienna")
        assert entry.n_episodes == 200 and entry.seed0 == (7000 if entry.world == "vienna" else 5000)
        # The evaluation environment gives the observation the policy was trained on.
        trained = ARCHIVED_CHECKPOINTS[entry.checkpoint].method
        assert METHOD_PRESETS[entry.method]["observation"] == METHOD_PRESETS[trained]["observation"]
    assert counts == {"static": 16, "dynamic": 11, "vienna": 4}


def test_archived_checkpoint_checks_the_file(tmp_path):
    entry = ARCHIVED_CHECKPOINTS["GEO-RL/1001"]
    with pytest.raises(FileNotFoundError):
        archived_checkpoint("GEO-RL/1001", tmp_path)
    path = tmp_path / entry.path
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not the checkpoint")
    with pytest.raises(ValueError, match="SHA-256"):
        archived_checkpoint("GEO-RL/1001", tmp_path)
    assert archived_checkpoint("GEO-RL/1001", tmp_path, verify=False) == path
    with pytest.raises(KeyError):
        archived_checkpoint("GEO-RL/9999", tmp_path)


@pytest.mark.skipif(not ARCHIVE, reason="needs RUSCUAV_ARCHIVE (the extracted v2.0.0 archive)")
def test_registry_matches_the_archive():
    for key in ARCHIVED_CHECKPOINTS:
        assert archived_checkpoint(key, ARCHIVE).stat().st_size == ARCHIVED_CHECKPOINTS[key].size_bytes
    for entry in ARCHIVED_EVALUATIONS.values():
        assert (Path(ARCHIVE) / entry.episodes_csv).is_file() and (Path(ARCHIVE) / entry.summary_json).is_file()


# ---------------------------------------------------------------------- against the notebook's pandas code
def _reference():
    """tests/notebook_reference.py (needs pandas)."""
    pytest.importorskip("pandas")
    sys.path.insert(0, str(REPO / "tests"))
    import notebook_reference

    return notebook_reference


def _synthetic_rows(dynamic: bool):
    rng = np.random.default_rng(3)
    rows = []
    for i in range(41):
        steps = 0 if i % 13 == 0 else int(rng.integers(40, 400))
        row = {"episode": i, "seed": 5000 + i, "steps": steps, "termination": [None, "success", "stuck"][i % 3],
               "is_success": bool(i % 3 == 1), "dist_xy": float(rng.uniform(0, 80)),
               "path_length_xy": 0.0 if i % 11 == 0 else float(rng.uniform(300, 900)),
               "cargo_health": float("nan") if i % 7 == 0 else float(rng.uniform(0.7, 1.0)),
               "urban_cumulative_risk": float(rng.uniform(0, 60)), "road_time": float(rng.choice([0.0, 4.5, 21.0])),
               "sidewalk_time": float(rng.choice([0.0, 2.5, 6.0])), "green_time": float(rng.uniform(0, 30)),
               "canopy_time": float(rng.uniform(0, 3)), "sensitive_time": float(rng.choice([0.0, 0.0, 0.5])),
               "urban_hard_violation_steps": int(rng.choice([0, 0, 0, 2]))}
        if i % 5 != 0:
            row["building_time"] = float(rng.choice([0.0, 3.0, 16.0]))
        if dynamic:
            row["dynamic_cumulative_risk"] = float(rng.uniform(0, 25))
        rows.append(row)
    return rows


def _same(a, b):
    a = a.item() if hasattr(a, "item") else a
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b and type(a) is type(b)


@pytest.mark.parametrize("dynamic", [False, True])
def test_post_hoc_step_follows_the_notebook(dynamic):
    nb = _reference()
    pd = nb.pd
    add_derived_metrics, summarize_episode_dataframe = nb.add_derived_metrics, nb.summarize_episode_dataframe
    rows = _synthetic_rows(dynamic)
    ref = add_derived_metrics(pd.DataFrame(rows))
    ours = with_derived_metrics(rows)
    assert list(ref.columns) == list(pd.DataFrame(ours).columns)
    original = {key for row in rows for key in row}
    new_columns = [c for c in ref.columns if c not in original or c == "episode_duration_s"]
    for i, row in enumerate(ours):
        for c in new_columns:
            assert _same(ref[c].iloc[i], row[c]), (i, c)
    ours_summary = summarize_posthoc(rows, dynamic)
    ref_summary = summarize_episode_dataframe(pd.DataFrame(rows), dynamic)
    assert list(ours_summary) == list(ref_summary)
    for key, value in ref_summary.items():
        assert (value == ours_summary[key] if isinstance(value, dict)
                else _same(float(value), float(ours_summary[key]))), key


def test_acceptability_budgets():
    row = {"is_success": True, "cargo_health": 0.8, "road_time": 20.0, "sidewalk_time": 5.0, "building_time": 15.0,
           "sensitive_time": 0.0, "urban_hard_violation_steps": 0, "dynamic_cumulative_risk": 10.0}
    assert static_acceptability(row) and dynamic_acceptability(row)
    for key, value in (("cargo_health", 0.79), ("road_time", 20.5), ("sidewalk_time", 5.5), ("building_time", 15.5),
                       ("sensitive_time", 0.5), ("urban_hard_violation_steps", 1), ("is_success", False)):
        assert not static_acceptability({**row, key: value}), key
    assert static_acceptability({**row, "cargo_health": None})  # missing cargo health counts as intact
    assert not dynamic_acceptability({**row, "dynamic_cumulative_risk": 10.5})
    assert static_acceptability({**row, "dynamic_cumulative_risk": 10.5})
    d = derived_metrics({**row, "steps": 0, "path_length_xy": 0.0})
    assert math.isnan(d["urban_risk_per_step"]) and math.isnan(d["road_fraction"])
    assert math.isnan(d["urban_risk_per_m_xy"])
    assert d["episode_duration_s"] == 0.0 and d["mission_acceptable_posthoc_dynamic"] is True


def _two_pass_std(values):
    values = [v for v in values if not math.isnan(v)]
    mean = 0.0
    for v in values:
        mean += v
    mean /= len(values)
    total = 0.0
    for v in values:
        total += (v - mean) ** 2
    return math.sqrt(total / len(values))


def _format_rows(row_format):
    rng = np.random.default_rng(11)
    rows = []
    for i in range(37):
        row = {"episode": i, "seed": 5000 + i, "total_reward": float(rng.normal(9000, 4000)),
               "steps": int(rng.integers(50, 400)), "termination": ["success", "stuck", None][i % 3],
               "is_success": bool(i % 3 == 0), "mission_acceptable": bool(i % 6 == 0),
               "dist_xy": float(rng.uniform(0, 90)),
               "cargo_health": float("nan") if i == 4 else float(rng.uniform(0.6, 1)),
               "urban_mean_risk": float(rng.uniform(0, 2)), "urban_max_risk": float(rng.uniform(0, 9)),
               "urban_cumulative_risk": float(rng.uniform(0, 80)), "road_time": float(rng.uniform(0, 30)),
               "sidewalk_time": float(rng.uniform(0, 9)), "building_time": float(rng.uniform(0, 20)),
               "green_time": float(rng.uniform(0, 30)), "canopy_time": float(rng.uniform(0, 4)),
               "sensitive_time": float(rng.choice([0.0, 0.5])), "urban_hard_violation_steps": int(rng.choice([0, 1]))}
        if row_format == "dynamic":
            row.update(mission_acceptable_static_only=bool(i % 4 == 0), dynamic_risk_mean=float(rng.uniform(0, 1)),
                       dynamic_risk_max=float(rng.uniform(0, 5)), dynamic_cumulative_risk=float(rng.uniform(0, 30)),
                       dynamic_budget_ok=bool(i % 2))
        rows.append(row)
    return rows


@pytest.mark.parametrize("row_format", ["dynamic", "transfer"])
def test_dynamic_and_transfer_summaries_follow_the_notebook(row_format):
    nb = _reference()
    rows = _format_rows(row_format)
    df = nb.pd.DataFrame(rows)
    ref = (nb.summarize_m11_dynamic_eval_df if row_format == "dynamic" else nb.summarize_m12c_real_policy_eval_df)(df)
    ours = summarize_policy_rows(rows, row_format)
    assert sorted(ours) == sorted(ref)
    for key, value in ref.items():
        if key == "termination_counts":
            assert ours[key] == value
        elif key == "std_reward":
            # The archived summaries computed std with bottleneck: a sequential two-pass sum.
            assert ours[key] == _two_pass_std([r["total_reward"] for r in rows])
            assert math.isclose(ours[key], value, rel_tol=1e-12)
        else:
            assert _same(float(value), float(ours[key])), key
    # No success: the archived summaries report 0, not NaN.
    failed = [dict(r, is_success=False) for r in rows]
    assert summarize_policy_rows(failed, row_format)["acceptable_given_success_rate"] == 0.0
    assert summarize_policy_rows([dict(r, is_success=np.bool_(r["is_success"])) for r in rows], row_format) == ours


def test_checkpoint_settings():
    from ruscuav_benchmark.policies import checkpoint_env_overrides

    assert checkpoint_env_overrides("RISK+GREEN-RL/4001") is None
    lidar = checkpoint_env_overrides("RISK+OBS-RL/no_urban_lidar")["lidar"]
    assert lidar.urban is False
    assert all(e.urban_lidar for k, e in ARCHIVED_CHECKPOINTS.items() if k != "RISK+OBS-RL/no_urban_lidar")
    assert {e.world_layers_sha256 for e in ARCHIVED_EVALUATIONS.values()} == {
        "be2bdac959eb9507598b77f8f0531dedf73e4f723bfacf162ea0cb9c4f785be4",
        "fa99c5710f9bcb2345901f986f0e97f59bddf26d775047c8c5f2e6079f692786",
    }


def test_evaluate_archived_refuses_other_settings_and_worlds(tmp_path):
    from ruscuav_benchmark import UrbanWorldLayers
    from ruscuav_benchmark.policies import evaluate_archived

    with pytest.raises(ValueError, match="fixed by the archived evaluation"):
        evaluate_archived("static/GEO-RL/1001", tmp_path, row_format="dynamic")
    with pytest.raises(ValueError, match="not in the archive"):
        evaluate_archived("vienna/GEO-RL/1001", tmp_path)
    other = UrbanWorldLayers.load_npz(REPO / "worlds" / "synthetic" / "AOI_SYNTH_001_003" / "AOI_SYNTH_001" /
                                      "urban_world_layers.npz")
    with pytest.raises(ValueError, match="layers checksum"):
        evaluate_archived("static/GEO-RL/1001", tmp_path, world=other)
    with pytest.raises(KeyError):
        evaluate_archived("static/GEO-RL/9", tmp_path)


# ---------------------------------------------------------------------- against the archive
def _archived_rows(stem_csv):
    pd = pytest.importorskip("pandas")
    df = pd.read_csv(stem_csv, float_precision="round_trip")
    return df, [{k: (None if isinstance(v, float) and math.isnan(v) else (v.item() if hasattr(v, "item") else v))
                 for k, v in r.items()} for r in df.to_dict("records")]


def _compare_summary(ours, archived):
    """Number of statistics compared; every statistic of the archived summary is."""
    compared = 0
    assert ours["termination_counts"] == archived["termination_counts"]
    for key, value in archived.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if key in ("obs_dim", "dynamic_obs_dim"):  # constants the dynamic-observation evaluations recorded
                continue
            assert key in ours, key
            assert _same(float(value), float(ours[key])), key
            compared += 1
    return compared


@pytest.mark.skipif(not ARCHIVE, reason="needs RUSCUAV_ARCHIVE (the extracted v2.0.0 archive)")
@pytest.mark.parametrize("name", [n for n, e in ARCHIVED_EVALUATIONS.items() if e.row_format == "posthoc"]
                         + ["planner/DYN_RH_Astar"])
def test_post_hoc_step_reproduces_the_archive(name):
    if name == "planner/DYN_RH_Astar":
        csv, summary, dynamic = f"{PLANNER_DYN}_episodes.csv", f"{PLANNER_DYN}_summary.json", True
    else:
        entry = ARCHIVED_EVALUATIONS[name]
        csv, summary, dynamic = entry.episodes_csv, entry.summary_json, False
    df, rows = _archived_rows(Path(ARCHIVE) / csv)
    for i, row in enumerate(rows):
        for key, value in derived_metrics(row).items():
            assert _same(df[key].iloc[i], value), (i, key)
    archived = json.loads((Path(ARCHIVE) / summary).read_text(encoding="utf-8"))
    ours = summarize_posthoc(rows, dynamic=dynamic)
    assert _compare_summary(ours, archived) == (68 if dynamic else 58)


@pytest.mark.skipif(not ARCHIVE, reason="needs RUSCUAV_ARCHIVE (the extracted v2.0.0 archive)")
@pytest.mark.parametrize("name", [n for n, e in ARCHIVED_EVALUATIONS.items() if e.row_format != "posthoc"])
def test_policy_summaries_reproduce_the_archive(name):
    entry = ARCHIVED_EVALUATIONS[name]
    _, rows = _archived_rows(Path(ARCHIVE) / entry.episodes_csv)
    archived = json.loads((Path(ARCHIVE) / entry.summary_json).read_text(encoding="utf-8"))
    ours = summarize_policy_rows(rows, entry.row_format)
    assert _compare_summary(ours, archived) == (22 if entry.row_format == "dynamic" else 20)
