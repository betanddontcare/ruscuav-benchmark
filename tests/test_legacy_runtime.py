"""Tests for the legacy runtime.

The source-rewriting tests need nothing extra. The parity test re-runs one archived
episode and needs the reproducibility dependencies plus an extracted RUSCUAV v2.0.0
Zenodo archive, pointed to by RUSCUAV_ARCHIVE; it is skipped otherwise.
"""

from pathlib import Path
import importlib.util
import os
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.legacy import runtime


def test_path_rewriting():
    text = 'a = "/content/drive/MyDrive/RUSCUAV/worlds/x.npz"\nb = "/content/ruscuav/data"\nc = "/content/data"\n'
    out = runtime.rewrite_paths(text, Path("/tmp/vd"), Path("/tmp/local"))
    assert "/content/" not in out
    assert '"/tmp/vd/worlds/x.npz"' in out and '"/tmp/local/ruscuav/data"' in out and '"/tmp/local/data"' in out


def test_path_rewriting_is_single_pass_and_respects_boundaries():
    text = 'a = "/content/drive/MyDrive/RUSCUAV/w.npz"\nb = "/content/ruscuav/x"\nc = "/content/database"\n'
    out = runtime.rewrite_paths(text, Path("/content/ruscuav_work/drive"), Path("/content/data_x/local"))
    assert '"/content/ruscuav_work/drive/w.npz"' in out  # not rewritten a second time
    assert '"/content/data_x/local/ruscuav/x"' in out
    assert '"/content/database"' in out  # /content/data only matches a whole path component


def test_virtual_drive_copies_inputs(tmp_path):
    archive = tmp_path / "archive"
    for rel in runtime.ARCHIVE_TO_DRIVE:
        (archive / rel).parent.mkdir(parents=True, exist_ok=True)
        (archive / rel).write_bytes(b"original")
    drive = runtime.prepare_virtual_drive(archive, tmp_path / "drive")
    for rel, drive_rel in runtime.ARCHIVE_TO_DRIVE.items():
        target = drive / drive_rel
        assert not target.is_symlink()
        target.write_bytes(b"changed by legacy code")
        assert (archive / rel).read_bytes() == b"original"
    runtime.prepare_virtual_drive(archive, tmp_path / "drive")  # refreshes the copy
    assert all((drive / d).read_bytes() == b"original" for d in runtime.ARCHIVE_TO_DRIVE.values())


def test_virtual_drive_takes_extra_inputs(tmp_path):
    archive = tmp_path / "archive"
    for rel in runtime.ARCHIVE_TO_DRIVE:
        (archive / rel).parent.mkdir(parents=True, exist_ok=True)
        (archive / rel).write_bytes(b"original")
    extra = tmp_path / "vienna.npz"
    extra.write_bytes(b"real world")
    drive = runtime.prepare_virtual_drive(archive, tmp_path / "drive", {extra: "worlds/vienna_x_urban_world_layers.npz"})
    assert (drive / "worlds/vienna_x_urban_world_layers.npz").read_bytes() == b"real world"


def test_state_patches_keep_line_numbers():
    source = runtime.DEFINITIONS_PATH.read_text(encoding="utf-8")
    patched = runtime.apply_notebook_state_patches(source)
    assert patched.count("\n") == source.count("\n")
    for anchor, code, _ in runtime.NOTEBOOK_STATE_PATCHES:
        line_before_anchor = patched.split("\n" + anchor, 1)[0].rsplit("\n", 1)[1]
        assert line_before_anchor.startswith(code.split("=")[0].strip())


ARCHIVE = os.environ.get("RUSCUAV_ARCHIVE")
HEAVY = all(importlib.util.find_spec(m) for m in ("torch", "stable_baselines3", "gymnasium", "scipy", "geopandas"))


@pytest.mark.skipif(not (ARCHIVE and HEAVY), reason="needs RUSCUAV_ARCHIVE and reproducibility dependencies")
def test_one_archived_planner_episode_reproduces(tmp_path):
    import pandas as pd

    ns = runtime.load_legacy_namespace(ARCHIVE, tmp_path)
    world = ns.UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)
    dirs = ns.setup_ruscuav_drive_project(drive_root=str(ns.RUSCUAV_VIRTUAL_DRIVE), experiment_name="pytest")
    rerun, *_ = ns.evaluate_m9_static_planner_drive(
        world=world, drive_dirs=dirs, planner_config=ns.M9PlannerConfig(**ns.REPORTED_M9_CONFIG),
        risk_profile="balanced", n_episodes=1, seed0=5000, stage="B", mode="strict_eval",
        n_obstacles=0, prefix="pytest_m9", save_paths=False,
    )
    archived = pd.read_csv(
        Path(ARCHIVE) / "experiments/paperA_primary_no_terminal_static_planners/eval/"
        "primary_no_terminal_STATIC_Astar_n200_seed5000/primary_no_terminal_STATIC_Astar_n200_seed5000_episodes.csv"
    ).iloc[0]
    row = rerun.iloc[0]
    for col in ("steps", "termination", "is_success", "mission_acceptable", "road_time", "sidewalk_time", "building_time"):
        assert str(row[col]) == str(archived[col]), col
    assert abs(row["urban_cumulative_risk"] - archived["urban_cumulative_risk"]) < 1e-9
