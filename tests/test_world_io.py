from dataclasses import replace
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruscuav_benchmark.evaluate import read_scenarios
from ruscuav_benchmark.synthetic import create_synthetic_world
from ruscuav_benchmark.world import (
    WORLD_SCHEMA_VERSION,
    UrbanWorldLayers,
    canonical_aoi_id,
)

REPO = Path(__file__).resolve().parents[1]
BUNDLED = sorted((REPO / "worlds" / "synthetic").glob("**/urban_world_layers.npz"))
CANONICAL = REPO / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz"
#: layers_sha256() of the canonical world; the v2.0.0 file AOI_FAKE_001, whose aoi_id
#: and CRS differ, has the same. Scenario files record checksums of this kind, so the
#: function must not change.
CANONICAL_LAYERS_SHA256 = "be2bdac959eb9507598b77f8f0531dedf73e4f723bfacf162ea0cb9c4f785be4"
REAL_GIS_MISSIONS = sorted((REPO / "scenarios" / "real_gis").glob("*.csv"))


@pytest.fixture()
def small_world():
    return create_synthetic_world(aoi_id="TEST", seed=7, width=120, height=120)


def test_save_writes_schema_and_round_trips(tmp_path, small_world):
    small_world.meta = {"source": "unit test"}
    path = tmp_path / "w.npz"
    small_world.save_npz(path)
    with np.load(path, allow_pickle=False) as data:
        assert str(data["schema_version"]) == WORLD_SCHEMA_VERSION
    loaded = UrbanWorldLayers.load_npz(path)
    assert loaded.schema_version == WORLD_SCHEMA_VERSION
    assert loaded.meta == {"source": "unit test"}
    assert np.array_equal(loaded.building_mask, small_world.building_mask)


def test_load_refuses_pickled_objects(tmp_path, small_world):
    payload = small_world.to_npz_dict()
    payload["crs"] = np.array({"not": "allowed"}, dtype=object)
    path = tmp_path / "pickled.npz"
    np.savez(path, **payload)  # np.savez pickles object arrays
    with pytest.raises(ValueError, match="pickled Python objects"):
        UrbanWorldLayers.load_npz(path)


def test_load_reports_missing_keys(tmp_path, small_world):
    payload = small_world.to_npz_dict()
    del payload["canopy_top_z"]
    path = tmp_path / "missing.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="canopy_top_z"):
        UrbanWorldLayers.load_npz(path)


def test_layers_checksum_ignores_file_layout_and_metadata(tmp_path, small_world):
    payload = small_world.to_npz_dict()
    payload.update(aoi_id=np.array("OTHER"), crs=np.array("EPSG:2180"),
                   bbox=np.array([5.0, 5.0, 125.0, 125.0], dtype=np.float32),
                   meta_json=np.array('{"source": "elsewhere"}'))
    payload["road_mask"] = payload["road_mask"].astype(np.uint8)
    del payload["schema_version"]
    path = tmp_path / "other.npz"
    np.savez(path, **payload)  # uncompressed, other dtypes, identifiers and metadata
    assert UrbanWorldLayers.load_npz(path).layers_sha256() == small_world.layers_sha256()


def test_layers_checksum_changes_with_the_content(small_world):
    reference = small_world.layers_sha256()
    assert len(reference) == 64
    assert replace(small_world, resolution_m=2.0).layers_sha256() != reference
    swapped = replace(small_world, road_mask=small_world.sidewalk_mask, sidewalk_mask=small_world.road_mask)
    assert swapped.layers_sha256() != reference
    small_world.terrain_z[3, 4] += 0.5
    assert small_world.layers_sha256() != reference


@pytest.mark.skipif(not CANONICAL.exists(), reason="canonical world not present")
def test_layers_checksum_of_the_canonical_world_is_stable():
    assert UrbanWorldLayers.load_npz(CANONICAL).layers_sha256() == CANONICAL_LAYERS_SHA256


@pytest.mark.parametrize("path", REAL_GIS_MISSIONS, ids=lambda p: p.stem)
def test_recorded_real_gis_missions_name_their_world(path):
    provenance = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    rows = read_scenarios(path)
    first, last = provenance["seeds"]
    assert [int(r["seed"]) for r in rows] == list(range(first, last + 1))
    assert {r["world_id"] for r in rows} == {provenance["world_id"]}
    assert len(provenance["world_layers_sha256"]) == 64
    assert len(provenance["world_file"]["sha256"]) == 64
    assert provenance["problems"] == []


def test_legacy_world_alias():
    assert canonical_aoi_id("AOI_FAKE_001") == "AOI_SYNTH_000"
    assert canonical_aoi_id("AOI_SYNTH_001") == "AOI_SYNTH_001"


def test_scenario_filter_accepts_legacy_world_id(tmp_path):
    csv_path = tmp_path / "s.csv"
    csv_path.write_text(
        "scenario_id,world_id,start_x,start_y,target_x,target_y\n"
        "a,AOI_FAKE_001,1,1,2,2\n"
        "b,AOI_SYNTH_000,3,3,4,4\n"
        "c,AOI_SYNTH_001,5,5,6,6\n",
        encoding="utf-8",
    )
    assert [r["scenario_id"] for r in read_scenarios(csv_path, world_id="AOI_SYNTH_000")] == ["a", "b"]


@pytest.mark.skipif(not BUNDLED, reason="bundled worlds not present")
@pytest.mark.parametrize("path", BUNDLED, ids=lambda p: p.parent.name)
def test_bundled_worlds_use_current_schema(path):
    world = UrbanWorldLayers.load_npz(path)
    assert world.schema_version == WORLD_SCHEMA_VERSION
    assert world.crs == "LOCAL_METRIC_SYNTHETIC"
    assert world.meta and len(world.meta["source_sha256"]) == 64
