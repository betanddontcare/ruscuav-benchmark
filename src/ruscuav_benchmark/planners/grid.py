"""The planning grid (risk-cost map and blocked cells at the planning altitude) and A*.

The construction and the search are those of the archived M9 planner, operation for
operation, so a configuration gives the same paths as in the v2.0.0 runs.
"""

from __future__ import annotations

import heapq
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..risk import UrbanRiskModel
from ..world import UrbanWorldLayers
from .config import PlannerConfig

Cell = Tuple[int, int]

#: 8-connected moves (row, column), in the order the search expands them.
NEIGHBOURS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))

#: Start and goal cells that are blocked move to the nearest free cell within this many cells.
NEAREST_FREE_RADIUS_CELLS = 20

#: Configuration fields that determine a :class:`RiskCostGrid`.
GRID_FIELDS = ("plan_res_m", "planner_agl_m", "risk_lambda", "block_buildings", "block_canopy",
               "block_sensitive_low_altitude", "building_clearance_m", "canopy_clearance_m", "sensitive_min_agl_m")


def grid_for(world: UrbanWorldLayers, config: PlannerConfig, risk_profile: str,
             grid: Optional["RiskCostGrid"] = None) -> "RiskCostGrid":
    """``grid`` if it was built for ``world`` with the same grid settings, else a new grid."""
    if grid is None:
        return RiskCostGrid(world, config, risk_profile)
    different = [name for name in GRID_FIELDS if getattr(grid.config, name) != getattr(config, name)]
    if grid.world is not world or grid.risk_profile != risk_profile or different:
        raise ValueError("the given RiskCostGrid was built for another world, risk profile or settings "
                         f"({', '.join(different) or 'world/risk profile'})")
    return grid


def block_reduce_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Mean over ``factor`` x ``factor`` blocks (incomplete edge blocks are dropped)."""
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    reduced = arr[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))
    return reduced.astype(np.float32)


def block_reduce_max(arr: np.ndarray, factor: int) -> np.ndarray:
    """Maximum over ``factor`` x ``factor`` blocks (incomplete edge blocks are dropped)."""
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    reduced = arr[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).max(axis=(1, 3))
    return reduced.astype(np.float32)


class RiskCostGrid:
    """Risk, blocked cells and cell costs of a world on the planning grid.

    ``risk_low`` is the semantic risk at ``planner_agl_m`` averaged over each cell,
    ``risk_norm`` the same divided by its 95th percentile and clipped to [0, 10],
    ``block`` the blocked cells and ``cell_cost`` = 1 + ``risk_lambda`` * ``risk_norm``
    (infinite where blocked). Row index is y, column index is x.
    """

    def __init__(self, world: UrbanWorldLayers, config: Optional[PlannerConfig] = None,
                 risk_profile: str = "balanced") -> None:
        self.world = world
        self.config = config if config is not None else PlannerConfig()
        self.risk_profile = risk_profile
        self.factor = max(1, int(round(self.config.plan_res_m / float(world.resolution_m))))
        self.plan_res_m = self.factor * float(world.resolution_m)
        self.x_max = float(world.width * world.resolution_m)
        self.y_max = float(world.height * world.resolution_m)
        self.risk_model = UrbanRiskModel(
            world=world, profile=risk_profile, hard_sensitive_zones=True, include_global_altitude_band_cost=True,
        )
        self._build()

    def _build(self) -> None:
        cfg = self.config
        world = self.world
        full_risk = self.risk_model.risk_map_at_agl(cfg.planner_agl_m).astype(np.float32)
        risk_low = block_reduce_mean(full_risk, self.factor)
        p95 = float(np.nanpercentile(risk_low, 95))
        denom = max(1e-6, p95)
        risk_norm = np.clip(risk_low / denom, 0.0, 10.0).astype(np.float32)

        block = np.zeros_like(risk_low, dtype=bool)
        if cfg.block_buildings:
            building = (world.building_mask > 0.5) & ((world.building_height + cfg.building_clearance_m)
                                                      >= cfg.planner_agl_m)
            block |= block_reduce_max(building.astype(np.float32), self.factor) > 0.5
        if cfg.block_canopy:
            canopy = (world.canopy_mask > 0.5) & ((world.canopy_height + cfg.canopy_clearance_m)
                                                  >= cfg.planner_agl_m)
            block |= block_reduce_max(canopy.astype(np.float32), self.factor) > 0.5
        if cfg.block_sensitive_low_altitude and cfg.planner_agl_m < cfg.sensitive_min_agl_m:
            block |= block_reduce_max(world.sensitive_mask.astype(np.float32), self.factor) > 0.5

        self.risk_low = risk_low
        self.risk_norm = risk_norm
        self.block = block
        self.cell_cost = cell_costs(risk_norm, block, cfg.risk_lambda)
        self.grid_h, self.grid_w = self.cell_cost.shape

    def xy_to_cell(self, xy) -> Cell:
        x, y = float(xy[0]), float(xy[1])
        col = int(np.clip(np.floor(x / self.plan_res_m), 0, self.grid_w - 1))
        row = int(np.clip(np.floor(y / self.plan_res_m), 0, self.grid_h - 1))
        return row, col

    def cell_to_xy(self, cell: Cell) -> np.ndarray:
        """Centre of a cell, kept inside the world."""
        row, col = cell
        x = float(np.clip((col + 0.5) * self.plan_res_m, 0.0, self.x_max - 1e-3))
        y = float(np.clip((row + 0.5) * self.plan_res_m, 0.0, self.y_max - 1e-3))
        return np.array([x, y], dtype=np.float32)


def cell_costs(risk_norm: np.ndarray, block: np.ndarray, risk_lambda: float) -> np.ndarray:
    """1 + ``risk_lambda`` * risk, infinite on blocked cells (float32)."""
    cost = (1.0 + float(risk_lambda) * risk_norm).astype(np.float32)
    cost[block] = np.inf
    return cost


def nearest_free_cell(cell: Cell, block: np.ndarray, cell_cost: np.ndarray,
                      max_radius_cells: int = NEAREST_FREE_RADIUS_CELLS) -> Optional[Cell]:
    """``cell`` if it is free, else the nearest free cell in growing square rings, or None."""
    grid_h, grid_w = block.shape
    row, col = cell
    if 0 <= row < grid_h and 0 <= col < grid_w and not block[row, col] and np.isfinite(cell_cost[row, col]):
        return cell
    best, best_d2 = None, float("inf")
    for r in range(1, int(max_radius_cells) + 1):
        r0, r1 = max(0, row - r), min(grid_h - 1, row + r)
        c0, c1 = max(0, col - r), min(grid_w - 1, col + r)
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if block[rr, cc] or not np.isfinite(cell_cost[rr, cc]):
                    continue
                d2 = (rr - row) ** 2 + (cc - col) ** 2
                if d2 < best_d2:
                    best_d2, best = d2, (rr, cc)
        if best is not None:
            return best
    return None


def astar(cell_cost: np.ndarray, block: np.ndarray, start: Cell, goal: Cell, plan_res_m: float,
          use_astar: bool = True) -> Tuple[Optional[List[Cell]], float, int]:
    """Cheapest 8-connected path from ``start`` to ``goal``.

    A move costs its length times the mean cost of the two cells; diagonal moves may
    not cut the corner of a blocked cell. The heuristic is the straight-line distance
    (zero with ``use_astar=False``, which makes the search Dijkstra's). Returns the
    cells of the path (None if there is none), its cost (inf if none) and the number
    of expanded cells.
    """
    grid_h, grid_w = cell_cost.shape
    # Python lists for fast indexing; float32 -> float is exact, so costs are unchanged.
    cost = cell_cost.tolist()
    blocked = block.tolist()
    step_lengths = [float(np.hypot(dr, dc) * plan_res_m) for dr, dc in NEIGHBOURS]

    def heuristic(cell: Cell) -> float:
        if not use_astar:
            return 0.0
        dr = (cell[0] - goal[0]) * plan_res_m
        dc = (cell[1] - goal[1]) * plan_res_m
        return float(np.hypot(dr, dc))

    open_heap: List[Tuple[float, float, Cell]] = []
    heapq.heappush(open_heap, (heuristic(start), 0.0, start))
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}
    visited = set()
    expanded = 0
    while open_heap:
        _, g_curr, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        expanded += 1
        if current == goal:
            cells = [current]
            while current in came_from:
                current = came_from[current]
                cells.append(current)
            cells.reverse()
            return cells, float(g_curr), expanded
        cr, cc = current
        for (dr, dc), step_len in zip(NEIGHBOURS, step_lengths):
            nr, nc = cr + dr, cc + dc
            if nr < 0 or nr >= grid_h or nc < 0 or nc >= grid_w:
                continue
            if blocked[nr][nc] or not math.isfinite(cost[nr][nc]):
                continue
            if dr != 0 and dc != 0 and (blocked[cr][nc] or blocked[nr][cc]):
                continue
            step_cost = step_len * 0.5 * (cost[cr][cc] + cost[nr][nc])
            neighbour = (nr, nc)
            tentative_g = g_curr + step_cost
            if tentative_g < g_score.get(neighbour, float("inf")):
                g_score[neighbour] = tentative_g
                came_from[neighbour] = current
                heapq.heappush(open_heap, (tentative_g + heuristic(neighbour), tentative_g, neighbour))
    return None, float("inf"), expanded


def plan_path(grid: RiskCostGrid, cell_cost: np.ndarray, block: np.ndarray, start_xy, target_xy,
              use_astar: bool = True) -> Dict[str, Any]:
    """Plan from ``start_xy`` to ``target_xy`` on given costs and blocked cells.

    The path runs from the start position through the centres of the intermediate
    cells to the target position. Returns a dict with ``success``, ``reason`` ("ok",
    "start_or_goal_blocked" or "no_path"), ``path_xy`` (float32, N x 2), ``path_cells``,
    ``cost`` and ``expanded``, and the start and goal cells.
    """
    start_cell_raw = grid.xy_to_cell(start_xy)
    goal_cell_raw = grid.xy_to_cell(target_xy)
    start = nearest_free_cell(start_cell_raw, block, cell_cost)
    goal = nearest_free_cell(goal_cell_raw, block, cell_cost)
    if start is None or goal is None:
        return {"success": False, "reason": "start_or_goal_blocked", "path_xy": None, "path_cells": None,
                "cost": np.inf, "expanded": 0}
    cells, cost, expanded = astar(cell_cost, block, start, goal, grid.plan_res_m, use_astar)
    cells_info = {"start_cell": start, "goal_cell": goal, "start_cell_raw": start_cell_raw,
                  "goal_cell_raw": goal_cell_raw}
    if cells is None:
        return {"success": False, "reason": "no_path", "path_xy": None, "path_cells": None, "cost": np.inf,
                "expanded": int(expanded), **cells_info}
    path_xy = [np.asarray(start_xy, dtype=np.float32)]
    path_xy.extend(grid.cell_to_xy(cell) for cell in cells[1:-1])
    path_xy.append(np.asarray(target_xy, dtype=np.float32))
    return {"success": True, "reason": "ok", "path_xy": np.asarray(path_xy, dtype=np.float32), "path_cells": cells,
            "cost": float(cost), "expanded": int(expanded), **cells_info}


def path_length_xy(path_xy: Optional[np.ndarray]) -> float:
    """Length of a 2-D polyline (NaN for fewer than two points)."""
    if path_xy is None or len(path_xy) < 2:
        return float("nan")
    return float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))
