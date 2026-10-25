"""Planar urban LiDAR.

``n_rays`` rays fan out over ±``half_angle_deg`` around the direction to the
target. Each ray is marched in steps of ``step_m`` up to ``range_m`` at the current
altitude and stops at the first unsafe point (terrain, building or canopy below
the minimum clearance, sensitive volume). Readings are normalised to [0, 1]
(1 = nothing within range). Samples outside the workspace are skipped.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import LidarConfig
from .hazards import UrbanHazardModel


class UrbanLidar:
    def __init__(self, config: LidarConfig, hazards: UrbanHazardModel, step_m: float) -> None:
        self.config = config
        self.hazards = hazards
        self.range_m = float(config.range_m)
        self.step_m = float(step_m)
        half = float(config.half_angle_deg)
        self.angles = np.linspace(np.deg2rad(-half), np.deg2rad(half), int(config.n_rays)).astype(np.float32)
        self.bounds_low = hazards.bounds_low
        self.bounds_high = hazards.bounds_high

    def ray_hit_distance(self, origin_xy: np.ndarray, ray_dir_xy: np.ndarray, z_abs: float,
                         max_dist: float) -> Optional[float]:
        step = float(self.step_m)
        n_steps = int(np.ceil(float(max_dist) / step))
        min_x, min_y = self.bounds_low[0], self.bounds_low[1]
        max_x, max_y = self.bounds_high[0], self.bounds_high[1]
        for i in range(1, n_steps + 1):
            d = min(float(max_dist), i * step)
            p_xy = origin_xy + ray_dir_xy * d
            if p_xy[0] < min_x or p_xy[0] > max_x or p_xy[1] < min_y or p_xy[1] > max_y:
                continue
            pos = np.array([p_xy[0], p_xy[1], z_abs], dtype=np.float32)
            if self.hazards.is_unsafe(pos, include_clearance_violation=True):
                return float(d)
        return None

    def scan(self, pos: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (readings, minimum urban reading)."""
        n = len(self.angles)
        if not self.config.urban:
            return np.ones(n, dtype=np.float32), 1.0
        scan_dist = float(self.range_m)
        origin = pos[:2].astype(np.float32)
        z_abs = float(pos[2])
        to_target = (target[:2] - origin).astype(np.float32)
        norm = float(np.linalg.norm(to_target) + 1e-9)
        if norm > 1e-9:
            to_target = to_target / norm
        else:
            to_target = np.array([1.0, 0.0], dtype=np.float32)
        urban_rays = np.ones(n, dtype=np.float32)
        for idx, ang in enumerate(self.angles):
            c, s = float(np.cos(ang)), float(np.sin(ang))
            ray_dir = np.array(
                [c * to_target[0] - s * to_target[1], s * to_target[0] + c * to_target[1]],
                dtype=np.float32,
            )
            rn = float(np.linalg.norm(ray_dir) + 1e-9)
            ray_dir = ray_dir / rn
            hit = self.ray_hit_distance(origin_xy=origin, ray_dir_xy=ray_dir, z_abs=z_abs, max_dist=scan_dist)
            if hit is not None:
                urban_rays[idx] = float(np.clip(hit / scan_dist, 0.0, 1.0))
        return urban_rays, float(np.min(urban_rays))
