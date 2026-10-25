"""Receding-horizon planner with a dynamic-risk overlay (M11 in the notebook)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..env.dynamic_risk import DynamicRiskField
from ..world import UrbanWorldLayers
from .config import DYNAMIC_PRIOR_MODES, RecedingPlannerConfig
from .grid import RiskCostGrid
from .receding import RecedingHorizonPlanner


class DynamicRecedingPlanner(RecedingHorizonPlanner):
    """:class:`RecedingHorizonPlanner` whose cell costs also include dynamic risk.

    The dynamic risk of the episode's field, at the planning altitude and the current
    step, divided by the field's ``dynamic_norm_denom``, is known within the sensing
    radius of the positions sensed so far; elsewhere it is zero
    (``dynamic_prior_mode="zero"``) or the mean of the current dynamic map ("mean").
    Static risk is handled as in the receding planner. Give the episode's field to
    :meth:`reset_belief` (the environment's ``dynamic_field`` after its reset).
    """

    kind = "dynamic"

    def __init__(self, world: UrbanWorldLayers, config: Optional[RecedingPlannerConfig] = None,
                 risk_profile: str = "balanced", dynamic_prior_mode: str = "zero",
                 grid: Optional[RiskCostGrid] = None) -> None:
        if dynamic_prior_mode not in DYNAMIC_PRIOR_MODES:
            raise ValueError("dynamic_prior_mode must be zero or mean")
        self.dynamic_prior_mode = str(dynamic_prior_mode)
        self.dynamic_field: Optional[DynamicRiskField] = None
        self.current_dynamic_true_norm: Optional[np.ndarray] = None
        self.dynamic_known_mask: Optional[np.ndarray] = None
        super().__init__(world, config, risk_profile, grid)
        self.dynamic_known_mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        self.current_dynamic_true_norm = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def reset_belief(self, dynamic_field: Optional[DynamicRiskField] = None) -> None:
        """Start an episode in ``dynamic_field`` (kept from the last call if None)."""
        if dynamic_field is not None:
            self.dynamic_field = dynamic_field
        super().reset_belief()
        self.dynamic_known_mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        self.current_dynamic_true_norm = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self._update_current_maps()

    def _dynamic_prior_norm_map(self) -> np.ndarray:
        if self.dynamic_prior_mode == "zero":
            return np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        mean_val = float(np.nanmean(self.current_dynamic_true_norm))
        return np.full((self.grid_h, self.grid_w), mean_val, dtype=np.float32)

    def update_dynamic_true_map(self, step: int) -> None:
        """Dynamic risk of the whole grid at ``step`` (what sensing reveals)."""
        if self.dynamic_field is None:
            raise RuntimeError("DynamicRecedingPlanner needs a dynamic field; call reset_belief(env.dynamic_field)")
        dyn_raw = self.dynamic_field.risk_lowres_map_at_step(
            step=int(step), plan_res_m=self.plan_res_m, grid_h=self.grid_h, grid_w=self.grid_w,
            planner_agl_m=self.config.planner_agl_m,
        )
        denom = max(1e-6, float(self.dynamic_field.config.dynamic_norm_denom))
        self.current_dynamic_true_norm = (dyn_raw / denom).astype(np.float32)

    def _reveal(self, rows: slice, cols: slice, mask: np.ndarray) -> None:
        super()._reveal(rows, cols, mask)
        local_dyn_known = self.dynamic_known_mask[rows, cols]
        local_dyn_known[mask] = True
        self.dynamic_known_mask[rows, cols] = local_dyn_known

    def reveal_around_xy(self, xy, step: int = 0) -> None:
        """Sense static and dynamic risk (at ``step``) within the sensing radius of ``xy``."""
        self.update_dynamic_true_map(step)
        super().reveal_around_xy(xy)

    def _estimated_risk(self) -> np.ndarray:
        static = super()._estimated_risk()
        if self.current_dynamic_true_norm is None:
            dynamic = np.zeros_like(static, dtype=np.float32)
        else:
            dynamic = np.where(self.dynamic_known_mask, self.current_dynamic_true_norm,
                               self._dynamic_prior_norm_map()).astype(np.float32)
        return static + dynamic

    def dynamic_known_fraction(self) -> float:
        return float(np.mean(self.dynamic_known_mask))
