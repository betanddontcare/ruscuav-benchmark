"""Tests for scripts/upgrade_world_npz.py."""

from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.synthetic import create_synthetic_world  # noqa: E402
from ruscuav_benchmark.world import UrbanWorldLayers  # noqa: E402

SCRIPT = REPO / "scripts" / "upgrade_world_npz.py"


def run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def legacy_file(tmp_path, **overrides):
    payload = create_synthetic_world(aoi_id="OLD", seed=3, width=100, height=100).to_npz_dict()
    del payload["schema_version"]
    payload.update(overrides)
    path = tmp_path / "old.npz"
    np.savez_compressed(path, **payload)
    return path, payload


def test_upgrade_keeps_layers_and_renames(tmp_path):
    source, payload = legacy_file(tmp_path)
    out = tmp_path / "new.npz"
    result = run("--input", source, "--output", out, "--aoi-id", "NEW", "--crs", "LOCAL_TEST")
    assert result.returncode == 0, result.stdout + result.stderr
    world = UrbanWorldLayers.load_npz(out)
    assert (world.aoi_id, world.crs, world.meta["source_aoi_id"]) == ("NEW", "LOCAL_TEST", "OLD")
    with np.load(out, allow_pickle=False) as new:
        for key in ("terrain_z", "building_mask", "bbox"):
            assert np.array_equal(new[key], payload[key]) and new[key].dtype == payload[key].dtype


def test_upgrade_refuses_to_change_dtypes(tmp_path):
    source, payload = legacy_file(tmp_path)
    source, _ = legacy_file(tmp_path, terrain_z=payload["terrain_z"].astype(np.float64))
    out = tmp_path / "new.npz"
    result = run("--input", source, "--output", out)
    assert result.returncode == 1 and "terrain_z" in result.stdout
    assert not out.exists()
