"""The vectorized LiDAR and hazard lookups against the sample-by-sample code, bit for bit.

``UrbanLidar.ray_hit_distances`` evaluates all samples of a scan at once;
``UrbanLidar.march`` is the sample-by-sample march it replaced (through
``UrbanHazardModel.is_unsafe``). The reference lookup is the expression
``UrbanRiskModel._ij_from_xy`` used before. Random states on two worlds, odd LiDAR
settings and every fallback branch must give identical results on any machine.
"""

from pathlib import Path
import math
import sys
import types

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("scipy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark import UrbanWorldLayers  # noqa: E402
from ruscuav_benchmark.env import LidarConfig, RuscuavEnv, make_config  # noqa: E402
from ruscuav_benchmark.risk import UrbanRiskModel  # noqa: E402

WORLDS = {
    "AOI_SYNTH_000": REPO / "worlds" / "synthetic" / "AOI_SYNTH_000" / "urban_world_layers.npz",
    "AOI_SYNTH_002": REPO / "worlds" / "synthetic" / "AOI_SYNTH_001_003" / "AOI_SYNTH_002" / "urban_world_layers.npz",
}


def _old_ij(self, x, y):
    res = float(self.world.resolution_m)
    xi = int(np.clip(np.floor(float(x) / res), 0, self.world.width - 1))
    yi = int(np.clip(np.floor(float(y) / res), 0, self.world.height - 1))
    return yi, xi


def _reference_scan(lidar, pos, target):
    """UrbanLidar.scan before the change: every ray marched sample by sample."""
    n = len(lidar.angles)
    scan_dist = float(lidar.range_m)
    origin = pos[:2].astype(np.float32)
    z_abs = float(pos[2])
    to_target = (target[:2] - origin).astype(np.float32)
    norm = float(np.linalg.norm(to_target) + 1e-9)
    to_target = to_target / norm if norm > 1e-9 else np.array([1.0, 0.0], dtype=np.float32)
    rays = np.ones(n, dtype=np.float32)
    for idx, ang in enumerate(lidar.angles):
        c, s = float(np.cos(ang)), float(np.sin(ang))
        ray_dir = np.array([c * to_target[0] - s * to_target[1], s * to_target[0] + c * to_target[1]], dtype=np.float32)
        ray_dir = ray_dir / float(np.linalg.norm(ray_dir) + 1e-9)
        hit = lidar.march(origin, ray_dir, z_abs, scan_dist)
        if hit is not None:
            rays[idx] = float(np.clip(hit / scan_dist, 0.0, 1.0))
    return rays, float(np.min(rays))


def _env(world, **lidar):
    config = make_config("RISK+OBS-RL", stage="B", mode="strict_eval", **({"lidar": LidarConfig(**lidar)} if lidar else {}))
    env = RuscuavEnv(world, config)
    # the reference path uses the old lookup throughout
    env.risk_model._ij_from_xy = types.MethodType(_old_ij, env.risk_model)
    return env


def _states(world, env, rng, n):
    for k in range(n):
        if k % 3 == 0:  # next to buildings, canopy or sensitive cells, low over them
            layer = (world.building_mask, world.canopy_mask, world.sensitive_mask)[k % 9 // 3]
            cells = np.argwhere(layer > 0.5)
            r, c = cells[rng.integers(len(cells))]
            xy = np.clip((np.array([c, r]) + rng.uniform(-4, 5, 2)) * world.resolution_m, 0, [env.x_max, env.y_max])
            z = float(world.terrain_z[int(r), int(c)]) + rng.uniform(0, 90)
        elif k % 3 == 1:  # on the workspace edge
            xy = np.array([rng.choice([0.0, env.x_max]), rng.uniform(0, env.y_max)])[rng.permutation(2)]
            z = rng.uniform(5, 140)
        else:
            xy = rng.uniform(0, [env.x_max, env.y_max])
            z = rng.uniform(0, 160)
        pos = np.array([xy[0], xy[1], z], dtype=np.float32)
        target = pos.copy() if k % 7 == 0 else np.array([*rng.uniform(0, [env.x_max, env.y_max]), z], dtype=np.float32)
        yield pos, target


@pytest.mark.parametrize("name", list(WORLDS))
@pytest.mark.parametrize("lidar", [{}, {"n_rays": 7, "half_angle_deg": 33.3, "range_m": 123.4, "step_m": 0.7}])
def test_scan_equals_the_sample_by_sample_march(name, lidar):
    world = UrbanWorldLayers.load_npz(WORLDS[name])
    env = _env(world, **lidar)
    new_model = UrbanRiskModel(world, hard_sensitive_zones=True)
    rng = np.random.default_rng(len(name) + len(lidar))
    hits = 0
    for pos, target in _states(world, env, rng, 24):
        rays, low = env.lidar.scan(pos, target)
        ref_rays, ref_low = _reference_scan(env.lidar, pos, target)
        assert rays.dtype == ref_rays.dtype and rays.tobytes() == ref_rays.tobytes()
        assert low == ref_low and type(low) is float
        hits += int((ref_rays < 1).sum())
        # the hazard check of many points against is_unsafe, with and without clearances
        pts = rng.uniform(-5, [env.x_max + 5, env.y_max + 5], size=(30, 2)).astype(np.float32)
        z = float(np.float32(pos[2]))
        for clearance in (True, False):
            mask = env.hazards.unsafe_at_altitude(pts[:, 0], pts[:, 1], z, include_clearance_violation=clearance)
            for (x, y), m in zip(pts, mask):
                assert bool(m) == env.hazards.is_unsafe(np.array([x, y, z], dtype=np.float32), clearance)
        # the new lookup against the old expression
        yi, xi = new_model.cells_from_xy(pts[:, 0], pts[:, 1])
        for (x, y), i, j in zip(pts, yi, xi):
            assert (int(i), int(j)) == _old_ij(new_model, x, y) == new_model._ij_from_xy(x, y)
    assert hits > 0


def test_lookup_edge_values():
    world = UrbanWorldLayers.load_npz(WORLDS["AOI_SYNTH_000"])
    model = UrbanRiskModel(world)
    w = world.width * world.resolution_m
    values = [0.0, -0.0, -1e-300, 1e-300, -0.5, 0.5, w, w - 1e-9, w + 1e-9, 1e30, -1e30, math.inf, -math.inf,
              float(np.nextafter(np.float32(w), np.float32(0))), np.float32(3.7), np.float64(12.25)]
    values += [k * world.resolution_m + e for k in range(0, int(world.width), 61) for e in (-1e-12, 0.0, 1e-12)]
    for x in values:
        for y in values[::5]:
            assert model._ij_from_xy(x, y) == _old_ij(model, x, y)
    for bad in (math.nan, np.float32("nan")):
        with pytest.raises(ValueError):
            model._ij_from_xy(bad, 1.0)
        with pytest.raises(ValueError):
            _old_ij(model, bad, 1.0)


def test_fallback_branches():
    world = UrbanWorldLayers.load_npz(WORLDS["AOI_SYNTH_000"])
    env = _env(world)
    lidar = env.lidar
    dirs = np.array([[1.0, 0.0], [0.0, -1.0], [0.6, 0.8]], dtype=np.float32)
    origin = np.array([100.0, 900.0], dtype=np.float32)
    z = 25.0
    expected = [lidar.march(origin, d, z, 250.0) for d in dirs]
    assert lidar.ray_hit_distances(origin, dirs, z, 250.0) == expected
    assert lidar.ray_hit_distance(origin, dirs[2], z, 250.0) == expected[2]
    # float64 directions: the products are computed in float64, then rounded as positions
    assert lidar.ray_hit_distances(origin, dirs.astype(np.float64), z, 250.0) == \
        [lidar.march(origin, d, z, 250.0) for d in dirs.astype(np.float64)]
    # integer directions take the sample-by-sample path
    int_dirs = np.array([[1, 0], [0, 1]], dtype=np.int64)
    assert lidar.ray_hit_distances(origin, int_dirs, z, 250.0) == [lidar.march(origin, d, z, 250.0) for d in int_dirs]
    # no samples within range
    assert lidar.ray_hit_distances(origin, dirs, z, 0.0) == [None, None, None]
    assert lidar.ray_hit_distances(origin, dirs[:0], z, 250.0) == []
    # NaN positions: both raise as the lookup does
    nan_origin = np.array([np.nan, 500.0], dtype=np.float32)
    with pytest.raises(ValueError):
        lidar.march(nan_origin, dirs[0], z, 250.0)
    with pytest.raises(ValueError):
        lidar.ray_hit_distances(nan_origin, dirs, z, 250.0)
