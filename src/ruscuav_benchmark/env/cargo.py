"""Cargo health.

The effective acceleration of each step (after shields) is converted to g. Every
step above the soft overload limit costs health quadratically in the excess; the
episode fails when health reaches zero.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import CargoConfig


class CargoModel:
    def __init__(self, config: CargoConfig) -> None:
        self.config = config

    def overload_g(self, eff_acc: np.ndarray) -> float:
        return float(np.linalg.norm(eff_acc) / self.config.g0)

    def step(self, health: float, eff_acc: np.ndarray) -> Tuple[float, float, float]:
        """Return (new health, overload in g, overload above the soft limit in g)."""
        over_g_eff = self.overload_g(eff_acc)
        excess = max(0.0, over_g_eff - self.config.overload_soft_g)
        if excess > 0.0:
            health = max(0.0, health - self.config.health_decay_k * (excess ** 2))
        return health, over_g_eff, excess
