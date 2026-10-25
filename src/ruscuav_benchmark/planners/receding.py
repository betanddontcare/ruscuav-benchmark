"""Receding-horizon planner with local sensing (M10 in the notebook)."""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..world import UrbanWorldLayers
from .config import RecedingPlannerConfig
from .grid import Cell, RiskCostGrid, cell_costs, grid_for, plan_path


class RecedingHorizonPlanner:
    """Replans from the current position on its current belief about the risk map.

    The true risk becomes known within ``sensing_radius_m`` of the positions where
    :meth:`reveal_around_xy` is called; elsewhere the prior of ``prior_risk_mode`` is
    used. Blocked cells are known everywhere (``hard_blockers_known``) or are revealed
    with the risk. :meth:`reset_belief` starts a new episode.
    """

    kind = "receding"

    def __init__(self, world: UrbanWorldLayers, config: Optional[RecedingPlannerConfig] = None,
                 risk_profile: str = "balanced", grid: Optional[RiskCostGrid] = None) -> None:
        self.config = config if config is not None else RecedingPlannerConfig()
        self.world = world
        self.risk_profile = risk_profile
        self.grid = grid_for(world, self.config, risk_profile, grid)
        self.plan_res_m = self.grid.plan_res_m
        self.grid_h, self.grid_w = self.grid.grid_h, self.grid.grid_w

        self.true_risk_norm = self.grid.risk_norm.astype(np.float32)
        self.true_block = self.grid.block.astype(bool)
        self.known_mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        if self.config.hard_blockers_known:
            self.known_block = self.true_block.copy()
        else:
            self.known_block = np.zeros_like(self.true_block, dtype=bool)
        self.prior_risk_norm = self._build_prior_risk_norm(self.config.prior_risk_mode)

        self.current_cell_cost: Optional[np.ndarray] = None
        self.current_block: Optional[np.ndarray] = None
        self.replan_count = 0
        self.total_expanded = 0
        self.total_plan_time_s = 0.0
        self._update_current_maps()

    # ------------------------------------------------------------------ belief
    def reset_belief(self) -> None:
        """Forget everything sensed and reset the planning counters."""
        self.known_mask[:] = False
        if self.config.hard_blockers_known:
            self.known_block = self.true_block.copy()
        else:
            self.known_block[:] = False
        self.replan_count = 0
        self.total_expanded = 0
        self.total_plan_time_s = 0.0
        self._update_current_maps()

    def _build_prior_risk_norm(self, mode: str) -> np.ndarray:
        mode = str(mode).lower()
        if mode == "zero":
            return np.zeros_like(self.true_risk_norm, dtype=np.float32)
        if mode == "mean":
            return np.full_like(self.true_risk_norm, float(np.nanmean(self.true_risk_norm)), dtype=np.float32)
        if mode == "blurred":
            try:
                from scipy.ndimage import gaussian_filter
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError('prior_risk_mode="blurred" needs scipy (the "env" extra)') from exc
            return gaussian_filter(self.true_risk_norm, sigma=5.0).astype(np.float32)
        if mode == "perfect":
            return self.true_risk_norm.copy().astype(np.float32)
        raise ValueError("prior_risk_mode must be 'zero', 'mean', 'blurred', or 'perfect'.")

    def _sensing_window(self, xy) -> Tuple[slice, slice, np.ndarray]:
        """Grid window around ``xy`` and the disc of sensed cells inside it."""
        cr, cc = self.grid.xy_to_cell(xy)
        radius_cells = max(1, int(np.ceil(float(self.config.sensing_radius_m) / self.plan_res_m)))
        r0, r1 = max(0, cr - radius_cells), min(self.grid_h - 1, cr + radius_cells)
        c0, c1 = max(0, cc - radius_cells), min(self.grid_w - 1, cc + radius_cells)
        rr, cc_grid = np.ogrid[r0:r1 + 1, c0:c1 + 1]
        mask = (rr - cr) ** 2 + (cc_grid - cc) ** 2 <= radius_cells ** 2
        return slice(r0, r1 + 1), slice(c0, c1 + 1), mask

    def _reveal(self, rows: slice, cols: slice, mask: np.ndarray) -> None:
        local_known = self.known_mask[rows, cols]
        local_known[mask] = True
        self.known_mask[rows, cols] = local_known
        if not self.config.hard_blockers_known:
            local_block = self.known_block[rows, cols]
            local_block[mask] = self.true_block[rows, cols][mask]
            self.known_block[rows, cols] = local_block

    def reveal_around_xy(self, xy) -> None:
        """Sense the true risk (and blockers) within the sensing radius of ``xy``."""
        rows, cols, mask = self._sensing_window(xy)
        self._reveal(rows, cols, mask)
        self._update_current_maps()

    def _estimated_risk(self) -> np.ndarray:
        return np.where(self.known_mask, self.true_risk_norm, self.prior_risk_norm).astype(np.float32)

    def _update_current_maps(self) -> None:
        current_block = (self.true_block if self.config.hard_blockers_known else self.known_block).copy()
        self.current_cell_cost = cell_costs(self._estimated_risk(), current_block, self.config.risk_lambda)
        self.current_block = current_block

    def known_fraction(self) -> float:
        return float(np.mean(self.known_mask))

    # ------------------------------------------------------------------ planning
    def xy_to_cell(self, xy) -> Cell:
        return self.grid.xy_to_cell(xy)

    def cell_to_xy(self, cell: Cell) -> np.ndarray:
        return self.grid.cell_to_xy(cell)

    def plan(self, start_xy, target_xy) -> Dict[str, Any]:
        """A* on the current belief; adds ``plan_time_s`` and ``known_fraction`` to the
        result of :func:`~ruscuav_benchmark.planners.grid.plan_path`. Searches (with or
        without a path) count towards ``replan_count`` and ``total_expanded``."""
        t0 = time.time()
        result = plan_path(self.grid, self.current_cell_cost, self.current_block, start_xy, target_xy,
                           self.config.use_astar)
        plan_time_s = time.time() - t0
        result["plan_time_s"] = float(plan_time_s)
        result["known_fraction"] = self.known_fraction()
        if result["reason"] != "start_or_goal_blocked":
            self.replan_count += 1
            self.total_expanded += int(result["expanded"])
            self.total_plan_time_s += float(plan_time_s)
        return result
