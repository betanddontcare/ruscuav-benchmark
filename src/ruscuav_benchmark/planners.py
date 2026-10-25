from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .risk import UrbanRiskModel
from .world import UrbanWorldLayers


@dataclass(frozen=True)
class RiskAStarConfig:
    plan_res_m: float = 10.0
    planner_agl_m: float = 40.0
    risk_lambda: float = 0.05
    block_buildings: bool = True
    block_canopy: bool = True
    block_sensitive_low_altitude: bool = True
    allow_diagonal: bool = True


def _block_reduce_mean(arr: np.ndarray, block: int) -> np.ndarray:
    h, w = arr.shape
    hh = h // block
    ww = w // block
    cropped = arr[: hh * block, : ww * block]
    return cropped.reshape(hh, block, ww, block).mean(axis=(1, 3))


def _block_reduce_max(arr: np.ndarray, block: int) -> np.ndarray:
    h, w = arr.shape
    hh = h // block
    ww = w // block
    cropped = arr[: hh * block, : ww * block]
    return cropped.reshape(hh, block, ww, block).max(axis=(1, 3))


class RiskCostAStarPlanner:
    """Static full-map risk-cost A* baseline for RUSCUAV-v0."""

    def __init__(self, world: UrbanWorldLayers, risk_model: UrbanRiskModel, config: RiskAStarConfig = RiskAStarConfig()) -> None:
        self.world = world
        self.risk_model = risk_model
        self.config = config
        res_ratio = float(config.plan_res_m) / float(world.resolution_m)
        block = int(round(res_ratio))
        if block <= 0 or abs(block - res_ratio) > 1e-6:
            raise ValueError("plan_res_m must be an integer multiple of world.resolution_m")
        self.block = block
        self.cost_grid, self.blocked_grid = self._build_grids()
        self.ny, self.nx = self.cost_grid.shape

    def _build_grids(self) -> tuple[np.ndarray, np.ndarray]:
        risk = self.risk_model.risk_map_at_agl(self.config.planner_agl_m)
        cost = _block_reduce_mean(risk, self.block).astype(np.float32)
        blocked = np.zeros_like(cost, dtype=bool)
        if self.config.block_buildings:
            # Block cells where the planner altitude cannot clear the roof.
            bad_b = (self.world.building_mask > 0.5) & (
                self.world.building_height + self.risk_model.profile.building_min_clearance_m > self.config.planner_agl_m
            )
            blocked |= _block_reduce_max(bad_b.astype(np.float32), self.block) > 0.5
        if self.config.block_canopy:
            bad_c = (self.world.canopy_mask > 0.5) & (
                self.world.canopy_height + self.risk_model.profile.canopy_min_clearance_m > self.config.planner_agl_m
            )
            blocked |= _block_reduce_max(bad_c.astype(np.float32), self.block) > 0.5
        if self.config.block_sensitive_low_altitude and self.config.planner_agl_m < self.risk_model.profile.sensitive_min_overflight_agl_m:
            blocked |= _block_reduce_max(self.world.sensitive_mask.astype(np.float32), self.block) > 0.5
        return cost, blocked

    def _world_to_cell(self, xy: np.ndarray) -> tuple[int, int]:
        x, y = float(xy[0]), float(xy[1])
        gx = int(np.clip(np.floor(x / self.config.plan_res_m), 0, self.nx - 1))
        gy = int(np.clip(np.floor(y / self.config.plan_res_m), 0, self.ny - 1))
        return gy, gx

    def _cell_to_world(self, cell: tuple[int, int]) -> np.ndarray:
        gy, gx = cell
        return np.array([(gx + 0.5) * self.config.plan_res_m, (gy + 0.5) * self.config.plan_res_m], dtype=np.float32)

    def _neighbors(self, cell: tuple[int, int]):
        gy, gx = cell
        if self.config.allow_diagonal:
            dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            dirs = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for dy, dx in dirs:
            yy, xx = gy + dy, gx + dx
            if 0 <= yy < self.ny and 0 <= xx < self.nx and not self.blocked_grid[yy, xx]:
                if dx != 0 and dy != 0:
                    # Avoid cutting through blocked corners.
                    if self.blocked_grid[gy, xx] or self.blocked_grid[yy, gx]:
                        continue
                yield (yy, xx), float(np.hypot(dx, dy)) * self.config.plan_res_m

    def plan(self, start_xy: np.ndarray, target_xy: np.ndarray) -> Optional[np.ndarray]:
        start = self._world_to_cell(np.asarray(start_xy, dtype=np.float32))
        goal = self._world_to_cell(np.asarray(target_xy, dtype=np.float32))
        if self.blocked_grid[start] or self.blocked_grid[goal]:
            return None
        def heuristic(a, b):
            return float(np.hypot((a[1] - b[1]) * self.config.plan_res_m, (a[0] - b[0]) * self.config.plan_res_m))
        open_heap = [(heuristic(start, goal), 0.0, start)]
        came_from: Dict[tuple[int, int], tuple[int, int]] = {}
        g_score: Dict[tuple[int, int], float] = {start: 0.0}
        visited = set()
        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()
                pts = [np.asarray(start_xy, dtype=np.float32)]
                pts += [self._cell_to_world(c) for c in cells[1:-1]]
                pts.append(np.asarray(target_xy, dtype=np.float32))
                return np.asarray(pts, dtype=np.float32)
            for nb, dist in self._neighbors(current):
                traversal = dist * (1.0 + self.config.risk_lambda * 0.5 * (self.cost_grid[current] + self.cost_grid[nb]))
                tentative = g_curr + float(traversal)
                if tentative < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    heapq.heappush(open_heap, (tentative + heuristic(nb, goal), tentative, nb))
        return None
