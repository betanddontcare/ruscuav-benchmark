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


def test_overlapping_masks_are_a_warning(tmp_path, payload):
    # worlds built from OpenStreetMap have overlapping classes, e.g. footways through parks
    sidewalk = payload["sidewalk_mask"] > 0.5
    assert sidewalk.any()
    payload["green_mask"] = np.where(sidewalk, 1.0, payload["green_mask"]).astype(np.float32)
    report = validate_world_npz(write(tmp_path, payload))
    assert not report.errors and any("sidewalk and green share" in w for w in report.warnings)


def test_terrain_above_the_workspace(tmp_path, payload):
    low, high = float(payload["terrain_z"].min()), float(payload["terrain_z"].max())
    assert high - low < 20.0

    def lifted(by):
        lifted_payload = dict(payload)
        for key in ("terrain_z", "surface_z", "building_roof_z", "canopy_top_z"):
            lifted_payload[key] = (payload[key] + np.float32(by)).astype(np.float32)
        return validate_world_npz(write(tmp_path, lifted_payload, f"lifted_{by:g}.npz"))

    # no room to cruise 30 m above the highest terrain: a warning
    report = lifted(125.0 - high)
    assert report.ok and any("give terrain heights relative" in w for w in report.warnings)
    # nowhere to fly 2 m above the lowest terrain below the 150 m ceiling: an error, as RuscuavEnv raises
    report = lifted(149.0 - low)
    assert not report.ok and any("lowest point is 149 m" in e for e in report.errors)


def test_workspace_constants_are_the_environments():
    pytest.importorskip("gymnasium")
    pytest.importorskip("scipy")
    from ruscuav_benchmark import validate
    from ruscuav_benchmark.env.config import DynamicsConfig

    d = DynamicsConfig()
    assert (validate.WORKSPACE_CEILING_M, validate.MIN_AGL_M, validate.CRUISE_AGL_M) == (
        d.z_max_m, d.min_agl_m, d.cruise_agl_m)


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


def _corrupt_member(path, payload, compressed):
    (np.savez_compressed if compressed else np.savez)(path, **payload)
    import zipfile
    data = bytearray(path.read_bytes())
    with zipfile.ZipFile(path) as z:
        info = z.getinfo("terrain_z.npy")
    offset = info.header_offset + 30 + len(info.filename) + len(info.extra) + info.compress_size // 2
    for i in range(8):
        data[offset + i] ^= 0xFF
    path.write_bytes(bytes(data))


def _non_npy_member(path, payload):
    import zipfile
    payload = dict(payload)
    del payload["aoi_id"]
    np.savez(path, **payload)
    with zipfile.ZipFile(path, "a") as z:
        z.writestr("aoi_id", "TEST")


MALFORMED = {
    "empty_file": lambda p, d: p.write_bytes(b""),
    "plain_npy": lambda p, d: np.save(open(p, "wb"), np.zeros((3, 3))),
    "zero_size_layers": lambda p, d: np.savez(p, **{k: (np.zeros((0, 0), np.float32) if getattr(v, "ndim", 0) == 2 else v) for k, v in d.items()}),
    "bad_meta_json": lambda p, d: np.savez(p, **{**d, "meta_json": np.array("{not json")}),
    "meta_json_not_object": lambda p, d: np.savez(p, **{**d, "meta_json": np.array("[1, 2]")}),
    "bbox_2x2": lambda p, d: np.savez(p, **{**d, "bbox": np.array([[0.0, 0.0], [120.0, 120.0]])}),
    "aoi_id_not_scalar": lambda p, d: np.savez(p, **{**d, "aoi_id": np.array(["TEST"])}),
    "resolution_text": lambda p, d: np.savez(p, **{**d, "resolution_m": np.array("one metre")}),
    "non_npy_member": _non_npy_member,
    "corrupt_member_stored": lambda p, d: _corrupt_member(p, d, False),
    "corrupt_member_compressed": lambda p, d: _corrupt_member(p, d, True),
}


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_files_are_reported_not_raised(tmp_path, payload, case):
    path = tmp_path / f"{case}.npz"
    MALFORMED[case](path, payload)
    report = validate_world_npz(path)  # must not raise
    assert not report.ok
    assert all("pickled" not in e for e in report.errors), report.errors
