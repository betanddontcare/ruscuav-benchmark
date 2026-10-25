"""Episode-specific dynamic risk field (temporal Gaussian hotspots).

Dynamic risk is a soft overlay: it is not an obstacle, it adds to the reward
penalty, to the metrics and (for dynamic observations) to the observation. Each
episode draws its hotspots from its own generator, seeded with the episode seed.
This is the field of the paper's dynamic experiments; the periodic hotspots of
``ruscuav_benchmark.dynamic`` are a separate, simpler proxy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..world import UrbanWorldLayers

HOTSPOT_MODES = ("road_sidewalk", "sidewalk", "road", "mixed")
TEMPORAL_MODES = ("box", "hann")


@dataclass(frozen=True)
class DynamicRiskConfig:
    n_hotspots: int = 8
    #: Where hotspots are placed: "road_sidewalk", "sidewalk", "road" or "mixed"
    #: (road, sidewalk or green).
    hotspot_mode: str = "road_sidewalk"
    radius_m_min: float = 35.0
    radius_m_max: float = 90.0
    amplitude_min: float = 4.0
    amplitude_max: float = 14.0
    #: Activation window in environment steps.
    start_step_min: int = 0
    start_step_max: int = 250
    duration_min_steps: int = 60
    duration_max_steps: int = 220
    #: "box" (constant while active) or "hann" (smooth pulse).
    temporal_mode: str = "hann"
    altitude_h_scale_m: float = 55.0
    #: Mission acceptability: maximum time-integrated dynamic risk.
    dynamic_cumulative_risk_budget: float = 35.0
    #: Weight of the dynamic risk penalty in the reward.
    dynamic_risk_lambda: float = 0.10
    #: Normalisation of dynamic risk in observations and planner maps.
    dynamic_norm_denom: float = 10.0

    def __post_init__(self) -> None:
        if str(self.hotspot_mode).lower() not in HOTSPOT_MODES:
            raise ValueError(f"hotspot_mode={self.hotspot_mode!r}; expected one of {list(HOTSPOT_MODES)}")
        if self.temporal_mode not in TEMPORAL_MODES:
            raise ValueError(f"temporal_mode={self.temporal_mode!r}; expected one of {list(TEMPORAL_MODES)}")
        if int(self.n_hotspots) < 0:
            raise ValueError("n_hotspots must not be negative")
        if self.start_step_min > self.start_step_max or self.duration_min_steps > self.duration_max_steps:
            raise ValueError("dynamic risk step ranges need min <= max")


#: Dynamic risk configuration of the reported dynamic experiments.
REPORTED_DYNAMIC_CONFIG = DynamicRiskConfig(
    n_hotspots=16,
    hotspot_mode="road_sidewalk",
    radius_m_min=60.0,
    radius_m_max=140.0,
    amplitude_min=15.0,
    amplitude_max=40.0,
    start_step_min=0,
    start_step_max=180,
    duration_min_steps=120,
    duration_max_steps=350,
    temporal_mode="hann",
    altitude_h_scale_m=70.0,
    dynamic_cumulative_risk_budget=10.0,
    dynamic_risk_lambda=0.20,
    dynamic_norm_denom=10.0,
)


class DynamicRiskField:
    """Hotspots of one episode and the risk they produce."""

    def __init__(
        self,
        world: UrbanWorldLayers,
        config: DynamicRiskConfig,
        seed: int,
        bounds_xy: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        self.world = world
        self.config = config
        self.seed = int(seed)
        #: (min_x, min_y, max_x, max_y), used only if no candidate cell exists.
        self.bounds_xy = bounds_xy or (0.0, 0.0, world.width * world.resolution_m, world.height * world.resolution_m)
        self.rng = np.random.default_rng(self.seed)
        self.hotspots = self._sample_hotspots()

    def _sample_xy_from_mask(self, mask: np.ndarray):
        ys, xs = np.where(mask > 0.5)
        if len(xs) == 0:
            return None
        idx = int(self.rng.integers(0, len(xs)))
        x = float(xs[idx] * self.world.resolution_m)
        y = float(ys[idx] * self.world.resolution_m)
        return x, y

    def _candidate_mask(self) -> np.ndarray:
        mode = str(self.config.hotspot_mode).lower()
        if mode == "road_sidewalk":
            mask = ((self.world.road_mask > 0.5) | (self.world.sidewalk_mask > 0.5)).astype(np.float32)
        elif mode == "sidewalk":
            mask = (self.world.sidewalk_mask > 0.5).astype(np.float32)
        elif mode == "road":
            mask = (self.world.road_mask > 0.5).astype(np.float32)
        elif mode == "mixed":
            mask = (
                (self.world.road_mask > 0.5) | (self.world.sidewalk_mask > 0.5) | (self.world.green_mask > 0.5)
            ).astype(np.float32)
        else:
            raise ValueError("hotspot_mode must be road_sidewalk, sidewalk, road, or mixed")
        if np.sum(mask > 0.5) == 0:
            mask = np.ones_like(self.world.terrain_z, dtype=np.float32)
        return mask

    def _sample_hotspots(self) -> List[Dict[str, Any]]:
        mask = self._candidate_mask()
        min_x, min_y, max_x, max_y = self.bounds_xy
        hotspots = []
        for _ in range(int(self.config.n_hotspots)):
            xy = self._sample_xy_from_mask(mask)
            if xy is None:
                x = float(self.rng.uniform(min_x, max_x))
                y = float(self.rng.uniform(min_y, max_y))
            else:
                x, y = xy
            radius = float(self.rng.uniform(self.config.radius_m_min, self.config.radius_m_max))
            amplitude = float(self.rng.uniform(self.config.amplitude_min, self.config.amplitude_max))
            start = int(self.rng.integers(self.config.start_step_min, self.config.start_step_max + 1))
            duration = int(self.rng.integers(self.config.duration_min_steps, self.config.duration_max_steps + 1))
            end = start + duration
            hotspots.append({
                "x": float(x),
                "y": float(y),
                "radius_m": float(radius),
                "sigma_m": float(max(1e-6, radius / 2.0)),
                "amplitude": float(amplitude),
                "start_step": int(start),
                "end_step": int(end),
                "duration_steps": int(duration),
            })
        return hotspots

    def temporal_factor(self, step: int, h: Dict[str, Any]) -> float:
        step = int(step)
        if step < h["start_step"] or step > h["end_step"]:
            return 0.0
        if self.config.temporal_mode == "box":
            return 1.0
        if self.config.temporal_mode == "hann":
            u = (step - h["start_step"]) / max(1, h["duration_steps"])
            u = float(np.clip(u, 0.0, 1.0))
            return float(0.5 - 0.5 * np.cos(2.0 * np.pi * u))
        raise ValueError("temporal_mode must be box or hann")

    def risk_at_xy_agl(self, x: float, y: float, agl_m: float, step: int) -> float:
        x = float(x)
        y = float(y)
        agl_m = float(max(0.0, agl_m))
        step = int(step)
        alt_factor = float(np.exp(-agl_m / max(1e-6, self.config.altitude_h_scale_m)))
        total = 0.0
        for h in self.hotspots:
            tf = self.temporal_factor(step, h)
            if tf <= 0.0:
                continue
            dx = x - h["x"]
            dy = y - h["y"]
            d2 = dx * dx + dy * dy
            sigma2 = h["sigma_m"] ** 2
            spatial = float(np.exp(-0.5 * d2 / max(1e-6, sigma2)))
            total += h["amplitude"] * tf * spatial * alt_factor
        return float(total)

    def risk_at_pos(self, pos, step: int) -> float:
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        ground = float(self.world.terrain_z[
            int(np.clip(np.floor(y / self.world.resolution_m), 0, self.world.height - 1)),
            int(np.clip(np.floor(x / self.world.resolution_m), 0, self.world.width - 1)),
        ])
        agl_m = z - ground
        return self.risk_at_xy_agl(x, y, agl_m, step)

    def risk_lowres_map_at_step(self, step: int, plan_res_m: float, grid_h: int, grid_w: int,
                                planner_agl_m: float) -> np.ndarray:
        """Dynamic risk on a planner grid of cell size ``plan_res_m`` at constant AGL."""
        xs = (np.arange(grid_w, dtype=np.float32) + 0.5) * float(plan_res_m)
        ys = (np.arange(grid_h, dtype=np.float32) + 0.5) * float(plan_res_m)
        X, Y = np.meshgrid(xs, ys)
        out = np.zeros((grid_h, grid_w), dtype=np.float32)
        alt_factor = float(np.exp(-float(planner_agl_m) / max(1e-6, self.config.altitude_h_scale_m)))
        for h in self.hotspots:
            tf = self.temporal_factor(step, h)
            if tf <= 0.0:
                continue
            dx = X - h["x"]
            dy = Y - h["y"]
            d2 = dx * dx + dy * dy
            sigma2 = h["sigma_m"] ** 2
            out += (h["amplitude"] * tf * alt_factor * np.exp(-0.5 * d2 / max(1e-6, sigma2))).astype(np.float32)
        return out.astype(np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {"seed": int(self.seed), "config": dict(self.config.__dict__), "hotspots": self.hotspots}
