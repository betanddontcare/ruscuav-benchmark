"""Static planner: full-map risk-cost A*, planned once (M9 in the notebook)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..world import UrbanWorldLayers
from .config import PlannerConfig
from .grid import RiskCostGrid, grid_for, plan_path


class StaticPlanner:
    """Plans the cheapest path on the full risk map. It ignores the UAV dynamics and
    the cargo; the waypoint controller flies the path. ``grid`` reuses a grid built
    for the same world and settings."""

    kind = "static"

    def __init__(self, world: UrbanWorldLayers, config: Optional[PlannerConfig] = None,
                 risk_profile: str = "balanced", grid: Optional[RiskCostGrid] = None) -> None:
        self.config = config if config is not None else PlannerConfig()
        self.grid = grid_for(world, self.config, risk_profile, grid)
        self.world = world
        self.risk_profile = risk_profile

    def plan(self, start_xy, target_xy) -> Dict[str, Any]:
        """See :func:`ruscuav_benchmark.planners.grid.plan_path`."""
        return plan_path(self.grid, self.grid.cell_cost, self.grid.block, start_xy, target_xy, self.config.use_astar)
