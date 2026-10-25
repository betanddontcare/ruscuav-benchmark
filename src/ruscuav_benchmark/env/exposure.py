"""Per-episode urban exposure, mission acceptability and operational stopping.

Each step adds ``dt`` seconds to the exposure time of every class under the UAV
(road, sidewalk, building, green, canopy, sensitive), counts constraint
violations and integrates the semantic (and optionally dynamic) risk.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..risk import UrbanRiskModel
from .config import MissionConfig


class ExposureTracker:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.risk_sum = 0.0
        self.risk_max = 0.0
        self.steps = 0
        self.road_time = 0.0
        self.sidewalk_time = 0.0
        self.building_time = 0.0
        self.green_time = 0.0
        self.canopy_time = 0.0
        self.sensitive_time = 0.0
        self.building_collision_steps = 0
        self.building_clearance_violation_steps = 0
        self.canopy_collision_steps = 0
        self.canopy_clearance_violation_steps = 0
        self.sensitive_low_altitude_violation_steps = 0
        self.hard_violation_steps = 0
        self.reset_dynamic()

    def reset_dynamic(self) -> None:
        self.dynamic_risk_sum = 0.0
        self.dynamic_risk_max = 0.0
        self.dynamic_steps = 0

    def update(self, risk_model: UrbanRiskModel, pos, dt: float) -> Dict[str, Any]:
        """Account one step at position ``pos``; return the flags, constraints and risk there."""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        flags = risk_model.flags_at(x, y)
        constraints = risk_model.constraints_at(x, y, z)
        components = risk_model.cost_components_at(x, y, z)
        dt = float(dt)
        if flags["road"]:
            self.road_time += dt
        if flags["sidewalk"]:
            self.sidewalk_time += dt
        if flags["building"]:
            self.building_time += dt
        if flags["green"]:
            self.green_time += dt
        if flags["canopy"]:
            self.canopy_time += dt
        if flags["sensitive"]:
            self.sensitive_time += dt
        if constraints["building_collision"]:
            self.building_collision_steps += 1
        if constraints["building_clearance_violation"]:
            self.building_clearance_violation_steps += 1
        if constraints["canopy_collision"]:
            self.canopy_collision_steps += 1
        if constraints["canopy_clearance_violation"]:
            self.canopy_clearance_violation_steps += 1
        if constraints["sensitive_low_altitude_violation"]:
            self.sensitive_low_altitude_violation_steps += 1
        if constraints["any_hard_violation"]:
            self.hard_violation_steps += 1
        total_risk = float(components["total_risk"])
        self.risk_sum += total_risk * dt
        self.risk_max = max(self.risk_max, total_risk)
        self.steps += 1
        return {"flags": flags, "constraints": constraints, "components": components, "total_risk": total_risk}

    def update_dynamic(self, dynamic_risk: float, dt: float) -> None:
        self.dynamic_risk_sum += dynamic_risk * dt
        self.dynamic_risk_max = max(self.dynamic_risk_max, dynamic_risk)
        self.dynamic_steps += 1

    def mission_acceptable(self, termination: Optional[str], cargo_health: float, mission: MissionConfig) -> bool:
        """Success within the cargo and exposure budgets and without hard violations."""
        if termination != "success":
            return False
        if float(cargo_health) < mission.accept_cargo_min:
            return False
        if self.road_time > mission.accept_road_s:
            return False
        if self.sidewalk_time > mission.accept_sidewalk_s:
            return False
        if self.building_time > mission.accept_building_s:
            return False
        if self.sensitive_time > mission.accept_sensitive_s:
            return False
        if self.hard_violation_steps > 0:
            return False
        return True

    def operational_stop(self, constraints: Dict[str, bool], mission: MissionConfig) -> Optional[str]:
        """Termination reason under operational stopping (stage C), or None."""
        if constraints["terrain_collision"]:
            return "urban_terrain_constraint"
        if constraints["building_collision"]:
            return "urban_building_collision"
        if constraints["canopy_collision"]:
            return "urban_canopy_collision"
        if constraints["sensitive_low_altitude_violation"]:
            return "urban_sensitive_low_altitude_violation"
        if self.sidewalk_time > mission.terminal_sidewalk_s:
            return "urban_sidewalk_budget_exceeded"
        if self.road_time > mission.terminal_road_s:
            return "urban_road_budget_exceeded"
        if self.building_time > mission.terminal_building_s:
            return "urban_building_budget_exceeded"
        if self.sensitive_time > mission.terminal_sensitive_s:
            return "urban_sensitive_budget_exceeded"
        return None
