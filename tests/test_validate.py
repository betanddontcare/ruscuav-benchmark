from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruscuav_benchmark.cli import main as cli_main
from ruscuav_benchmark.synthetic import create_synthetic_world
from ruscuav_benchmark.validate import validate_world_npz

REPO = Path(__file__).resolve().parents[1]
BUNDLED = sorted((REPO / "worlds" / "synthetic").glob("**/urban_world_layers.npz"))


@pytest.fixture()
def payload():
    world = create_synthetic_world(aoi_id="TEST", seed=7, width=120, height=120)
    return world.to_npz_dict()


def write(tmp_path, payload, name="w.npz"):
    path = tmp_path / name
    np.savez(path, **payload)
    return path


def test_valid_world_passes_with_grid_warning(tmp_path, payload):
    report = validate_world_npz(write(tmp_path, payload))
    assert report.ok, report.errors
    assert any("validated on 1000 × 1000" in w for w in report.warnings)
    assert set(report.info["coverage"]) >= {"building", "road", "sidewalk", "green", "canopy", "sensitive"}


def test_missing_key(tmp_path, payload):
    del payload["sensitive_mask"]
    report = validate_world_npz(write(tmp_path, payload))
    assert not report.ok and "sensitive_mask" in report.errors[0]


def test_pickled_object_rejected(tmp_path, payload):
    payload["aoi_id"] = np.array(["TEST", 1], dtype=object)
    report = validate_world_npz(write(tmp_path, payload))
    assert not report.ok and "pickled" in report.errors[0]


def test_shape_mismatch(tmp_path, payload):
    payload["road_mask"] = payload["road_mask"][:-1, :]
    report = validate_world_npz(write(tmp_path, payload))
    assert any("road_mask has shape" in e for e in report.errors)


def test_nan_rejected(tmp_path, payload):
    payload["terrain_z"] = payload["terrain_z"].copy()
    payload["terrain_z"][5, 5] = np.nan
    report = validate_world_npz(write(tmp_path, payload))
    assert any("terrain_z contains NaN" in e for e in report.errors)


def test_overlapping_masks_rejected(tmp_path, payload):
    road = payload["road_mask"] > 0.5
    payload["building_mask"] = np.where(road, 1.0, payload["building_mask"]).astype(np.float32)
    report = validate_world_npz(write(tmp_path, payload))
    assert any("building and road" in e for e in report.errors)


def test_inconsistent_derived_layer_rejected(tmp_path, payload):
    payload["height_above_ground"] = payload["height_above_ground"] + 2.0
    report = validate_world_npz(write(tmp_path, payload))
    assert any("height_above_ground must equal" in e for e in report.errors)


def test_fractional_mask_is_a_warning(tmp_path, payload):
    road = payload["road_mask"].copy()
    assert road.max() > 0.5  # the generated world has roads
    road[road > 0.5] = 0.9
    payload["road_mask"] = road
    report = validate_world_npz(write(tmp_path, payload))
    assert report.ok, report.errors
    assert any("road_mask is not binary" in w for w in report.warnings)


def test_bbox_must_match_grid(tmp_path, payload):
    payload["bbox"] = np.array([0.0, 0.0, 500.0, 120.0])
    report = validate_world_npz(write(tmp_path, payload))
    assert any("does not match the grid" in e for e in report.errors)


def test_local_bbox_with_epsg_label_warns(tmp_path, payload):
    payload["crs"] = np.array("EPSG:2180")
    report = validate_world_npz(write(tmp_path, payload))
    assert report.ok
    assert any("EPSG:2180" in w for w in report.warnings)


def test_missing_schema_version_warns(tmp_path, payload):
    del payload["schema_version"]
    report = validate_world_npz(write(tmp_path, payload))
    assert report.ok and any("no schema_version" in w for w in report.warnings)


def test_not_an_npz(tmp_path):
    path = tmp_path / "junk.npz"
    path.write_bytes(b"not a zip")
    assert not validate_world_npz(path).ok


def test_cli_exit_codes_and_json(tmp_path, payload, capsys):
    good = write(tmp_path, payload, "good.npz")
    bad_payload = dict(payload)
    del bad_payload["bbox"]
    bad = write(tmp_path, bad_payload, "bad.npz")
    assert cli_main(["validate", str(good)]) == 0
    capsys.readouterr()  # discard the text report
    assert cli_main(["validate", str(good), str(bad), "--json"]) == 1
    reports = json.loads(capsys.readouterr().out)
    assert [r["ok"] for r in reports] == [True, False]
    assert "missing required keys: bbox" in reports[1]["errors"][0]


@pytest.mark.skipif(not BUNDLED, reason="bundled worlds not present")
@pytest.mark.parametrize("path", BUNDLED, ids=lambda p: p.parent.name)
def test_bundled_worlds_are_valid(path):
    report = validate_world_npz(path)
    assert report.ok, report.errors
    assert report.warnings == []
