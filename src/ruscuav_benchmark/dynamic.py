from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .risk import UrbanRiskModel
from .world import UrbanWorldLayers


@dataclass(frozen=True)
class DynamicHotspot:
    x: float
    y: float
    sigma_m: float
    amplitude: float
    phase_s: float
    period_s: float
    duty: float = 0.5


class DynamicRiskField:
    """Calibrated spatio-temporal ground-risk proxy for RUSCUAV dynamic tests."""

    def __init__(self, world: UrbanWorldLayers, hotspots: List[DynamicHotspot], altitude_decay_m: float = 60.0) -> None:
        self.world = world
        self.hotspots = list(hotspots)
        self.altitude_decay_m = float(altitude_decay_m)

    @staticmethod
    def temporal_activation(t_s: float, phase_s: float, period_s: float, duty: float) -> float:
        u = ((float(t_s) + float(phase_s)) % float(period_s)) / float(period_s)
        return 1.0 if u <= float(duty) else 0.0

    def risk_at(self, x: float, y: float, z_abs: float, t_s: float, risk_model: UrbanRiskModel) -> float:
        h_agl = max(0.0, risk_model.z_agl_at(x, y, z_abs))
        alt = float(np.exp(-h_agl / max(1e-6, self.altitude_decay_m)))
        val = 0.0
        for hs in self.hotspots:
            d2 = (float(x) - hs.x) ** 2 + (float(y) - hs.y) ** 2
            spatial = np.exp(-d2 / (2.0 * hs.sigma_m ** 2))
            temporal = self.temporal_activation(t_s, hs.phase_s, hs.period_s, hs.duty)
            val += hs.amplitude * spatial * temporal * alt
        return float(val)


def sample_hotspots_from_exposure_layers(world: UrbanWorldLayers, seed: int = 0, n_hotspots: int = 12) -> List[DynamicHotspot]:
    rng = np.random.default_rng(int(seed))
    exposure = (world.road_mask > 0.5) | (world.sidewalk_mask > 0.5)
    ys, xs = np.where(exposure)
    if len(xs) == 0:
        return []
    idx = rng.choice(len(xs), size=min(int(n_hotspots), len(xs)), replace=False)
    hotspots = []
    for i in idx:
        hotspots.append(DynamicHotspot(
            x=float(xs[i]) * world.resolution_m,
            y=float(ys[i]) * world.resolution_m,
            sigma_m=float(rng.uniform(25.0, 80.0)),
            amplitude=float(rng.uniform(4.0, 18.0)),
            phase_s=float(rng.uniform(0.0, 120.0)),
            period_s=float(rng.uniform(60.0, 240.0)),
            duty=float(rng.uniform(0.25, 0.75)),
        ))
    return hotspots
