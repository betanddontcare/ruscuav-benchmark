"""Tests for the v0 dynamic-risk proxy (ruscuav_benchmark.dynamic) and the world quicklook
(ruscuav_benchmark.visualize). Neither needs an extra."""

from dataclasses import replace
from pathlib import Path
import math
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruscuav_benchmark.dynamic import (DynamicHotspot, DynamicRiskField,  # noqa: E402
                                       sample_hotspots_from_exposure_layers)
from ruscuav_benchmark.risk import UrbanRiskModel  # noqa: E402
from ruscuav_benchmark.synthetic import create_synthetic_world  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return create_synthetic_world("AOI_SYNTH_TEST", seed=3, width=400, height=300)


def test_temporal_activation_switches_with_the_duty_cycle():
    on = DynamicRiskField.temporal_activation
    assert on(0.0, 0.0, 100.0, 0.5) == 1.0 and on(50.0, 0.0, 100.0, 0.5) == 1.0
    assert on(51.0, 0.0, 100.0, 0.5) == 0.0 and on(99.0, 0.0, 100.0, 0.5) == 0.0
    assert on(100.0, 0.0, 100.0, 0.5) == 1.0  # the next period
    assert on(0.0, 60.0, 100.0, 0.5) == 0.0 and on(40.0, 60.0, 100.0, 0.5) == 1.0  # the phase shifts it
    assert on(73.0, 0.0, 100.0, 1.0) == 1.0  # always on


def test_risk_is_a_sum_of_gaussian_hotspots_decaying_with_altitude(world):
    model = UrbanRiskModel(world, profile="balanced")
    a = DynamicHotspot(x=100.0, y=120.0, sigma_m=10.0, amplitude=5.0, phase_s=0.0, period_s=60.0, duty=1.0)
    b = DynamicHotspot(x=130.0, y=120.0, sigma_m=20.0, amplitude=2.0, phase_s=0.0, period_s=60.0, duty=0.5)
    field = DynamicRiskField(world, [a], altitude_decay_m=60.0)
    ground = model.terrain_at(100.0, 120.0)
    assert field.risk_at(100.0, 120.0, ground, 0.0, model) == pytest.approx(5.0)
    assert field.risk_at(110.0, 120.0, model.terrain_at(110.0, 120.0), 0.0, model) == pytest.approx(
        5.0 * math.exp(-0.5))
    assert field.risk_at(100.0, 120.0, ground + 60.0, 0.0, model) == pytest.approx(5.0 * math.exp(-1.0))
    assert field.risk_at(100.0, 120.0, ground - 5.0, 0.0, model) == pytest.approx(5.0)  # below ground: as at 0 m
    both = DynamicRiskField(world, [a, b], altitude_decay_m=60.0)
    x, z = 115.0, model.terrain_at(115.0, 120.0) + 20.0
    alt = math.exp(-20.0 / 60.0)
    expected = 5.0 * math.exp(-15.0 ** 2 / 200.0) * alt + 2.0 * math.exp(-15.0 ** 2 / 800.0) * alt
    assert both.risk_at(x, 120.0, z, 10.0, model) == pytest.approx(expected)  # b is on (u = 1/6)
    assert both.risk_at(x, 120.0, z, 40.0, model) == pytest.approx(5.0 * math.exp(-15.0 ** 2 / 200.0) * alt)
    assert DynamicRiskField(world, []).risk_at(x, 120.0, z, 0.0, model) == 0.0


def test_hotspots_are_sampled_on_roads_and_sidewalks(world):
    hotspots = sample_hotspots_from_exposure_layers(world, seed=7, n_hotspots=12)
    assert len(hotspots) == 12
    assert hotspots == sample_hotspots_from_exposure_layers(world, seed=7, n_hotspots=12)
    assert hotspots != sample_hotspots_from_exposure_layers(world, seed=8, n_hotspots=12)
    exposure = (world.road_mask > 0.5) | (world.sidewalk_mask > 0.5)
    res = world.resolution_m
    for h in hotspots:
        assert exposure[int(round(h.y / res)), int(round(h.x / res))]
        assert 25.0 <= h.sigma_m <= 80.0 and 4.0 <= h.amplitude <= 18.0
        assert 0.0 <= h.phase_s <= 120.0 and 60.0 <= h.period_s <= 240.0 and 0.25 <= h.duty <= 0.75
    assert len({(h.x, h.y) for h in hotspots}) == 12  # cells are drawn without replacement
    few = int(exposure.sum())
    assert len(sample_hotspots_from_exposure_layers(world, seed=0, n_hotspots=few + 50)) == few
    empty = replace(world, road_mask=np.zeros_like(world.road_mask), sidewalk_mask=np.zeros_like(world.sidewalk_mask))
    assert sample_hotspots_from_exposure_layers(empty, seed=7) == []


def test_quicklook_writes_a_three_panel_png(world, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.image import imread

    from ruscuav_benchmark.visualize import save_world_quicklook

    path = save_world_quicklook(world, tmp_path / "new" / "look.png", agl_m=30.0)
    assert path == str(tmp_path / "new" / "look.png") and Path(path).is_file()
    image = imread(path)
    assert image.shape[:2] == (640, 1920)  # 12 x 4 inches at 160 dpi
    assert np.ptp(image[..., :3]) > 0.5  # something is drawn
