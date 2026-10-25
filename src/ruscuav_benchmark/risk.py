from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .world import UrbanWorldLayers


@dataclass(frozen=True)
class UrbanRiskProfile:
    name: str = "balanced"
    green_weight: float = 0.10
    canopy_weight: float = 0.20
    building_weight: float = 3.00
    road_weight: float = 6.00
    sidewalk_weight: float = 12.00
    sensitive_weight: float = 50.00
    green_h_scale_m: float = 35.0
    road_h_scale_m: float = 45.0
    sidewalk_h_scale_m: float = 55.0
    sensitive_h_scale_m: float = 80.0
    building_clearance_h_scale_m: float = 35.0
    canopy_clearance_h_scale_m: float = 20.0
    terrain_min_clearance_m: float = 2.0
    building_min_clearance_m: float = 5.0
    canopy_min_clearance_m: float = 3.0
    sensitive_min_overflight_agl_m: float = 80.0
    preferred_agl_low_m: float = 20.0
    preferred_agl_high_m: float = 80.0
    low_altitude_weight: float = 2.0
    high_altitude_weight: float = 0.50
    building_collision_multiplier: float = 8.0
    canopy_collision_multiplier: float = 4.0
    terrain_collision_multiplier: float = 10.0
    sensitive_low_altitude_multiplier: float = 5.0


RISK_PROFILES: Dict[str, UrbanRiskProfile] = {
    "efficiency_first": UrbanRiskProfile(
        name="efficiency_first", green_weight=0.05, canopy_weight=0.10,
        building_weight=1.50, road_weight=3.00, sidewalk_weight=6.00,
        sensitive_weight=30.00, preferred_agl_low_m=15.0, preferred_agl_high_m=70.0,
        high_altitude_weight=0.35,
    ),
    "balanced": UrbanRiskProfile(name="balanced"),
    "safety_first": UrbanRiskProfile(
        name="safety_first", green_weight=0.05, canopy_weight=0.10,
        building_weight=5.00, road_weight=10.00, sidewalk_weight=20.00,
        sensitive_weight=100.00, road_h_scale_m=60.0, sidewalk_h_scale_m=75.0,
        sensitive_h_scale_m=100.0, sensitive_min_overflight_agl_m=100.0,
        preferred_agl_low_m=30.0, preferred_agl_high_m=100.0, high_altitude_weight=0.70,
    ),
}


class UrbanRiskModel:
    """Altitude-aware semantic risk model for one UrbanWorldLayers instance."""

    def __init__(
        self,
        world: UrbanWorldLayers,
        profile: str | UrbanRiskProfile = "balanced",
        hard_sensitive_zones: bool = True,
        include_global_altitude_band_cost: bool = True,
    ) -> None:
        self.world = world
        if isinstance(profile, str):
            if profile not in RISK_PROFILES:
                raise ValueError(f"Unknown profile {profile!r}; available={list(RISK_PROFILES)}")
            self.profile = RISK_PROFILES[profile]
        elif isinstance(profile, UrbanRiskProfile):
            self.profile = profile
        else:
            raise TypeError("profile must be str or UrbanRiskProfile")
        self.hard_sensitive_zones = bool(hard_sensitive_zones)
        self.include_global_altitude_band_cost = bool(include_global_altitude_band_cost)

    def _ij_from_xy(self, x: float, y: float) -> tuple[int, int]:
        res = float(self.world.resolution_m)
        xi = int(np.clip(np.floor(float(x) / res), 0, self.world.width - 1))
        yi = int(np.clip(np.floor(float(y) / res), 0, self.world.height - 1))
        return yi, xi

    def terrain_at(self, x: float, y: float) -> float:
        yi, xi = self._ij_from_xy(x, y)
        return float(self.world.terrain_z[yi, xi])

    def z_agl_at(self, x: float, y: float, z_abs: float) -> float:
        return float(z_abs - self.terrain_at(x, y))

    def flags_at(self, x: float, y: float) -> Dict[str, bool]:
        yi, xi = self._ij_from_xy(x, y)
        return {
            "building": bool(self.world.building_mask[yi, xi] > 0.5),
            "road": bool(self.world.road_mask[yi, xi] > 0.5),
            "sidewalk": bool(self.world.sidewalk_mask[yi, xi] > 0.5),
            "green": bool(self.world.green_mask[yi, xi] > 0.5),
            "canopy": bool(self.world.canopy_mask[yi, xi] > 0.5),
            "sensitive": bool(self.world.sensitive_mask[yi, xi] > 0.5),
        }

    def heights_at(self, x: float, y: float) -> Dict[str, float]:
        yi, xi = self._ij_from_xy(x, y)
        return {
            "terrain_z": float(self.world.terrain_z[yi, xi]),
            "surface_z": float(self.world.surface_z[yi, xi]),
            "height_above_ground": float(self.world.height_above_ground[yi, xi]),
            "building_height": float(self.world.building_height[yi, xi]),
            "building_roof_z": float(self.world.building_roof_z[yi, xi]),
            "canopy_height": float(self.world.canopy_height[yi, xi]),
            "canopy_top_z": float(self.world.canopy_top_z[yi, xi]),
            "canopy_density": float(self.world.canopy_density[yi, xi]),
        }

    @staticmethod
    def _decay(value_m, scale_m: float):
        return np.exp(-np.maximum(value_m, 0.0) / max(float(scale_m), 1e-6))

    def clearances_at(self, x: float, y: float, z_abs: float) -> Dict[str, float]:
        h = self.heights_at(x, y)
        flags = self.flags_at(x, y)
        return {
            "terrain_clearance_m": float(z_abs - h["terrain_z"]),
            "building_clearance_m": float(z_abs - h["building_roof_z"]) if flags["building"] else 1e9,
            "canopy_clearance_m": float(z_abs - h["canopy_top_z"]) if flags["canopy"] else 1e9,
        }

    def constraints_at(self, x: float, y: float, z_abs: float) -> Dict[str, bool]:
        p = self.profile
        flags = self.flags_at(x, y)
        clear = self.clearances_at(x, y, z_abs)
        z_agl = self.z_agl_at(x, y, z_abs)
        terrain_collision = clear["terrain_clearance_m"] <= p.terrain_min_clearance_m
        building_collision = flags["building"] and clear["building_clearance_m"] <= 0.0
        building_clearance_violation = flags["building"] and clear["building_clearance_m"] < p.building_min_clearance_m
        canopy_collision = flags["canopy"] and clear["canopy_clearance_m"] <= 0.0
        canopy_clearance_violation = flags["canopy"] and clear["canopy_clearance_m"] < p.canopy_min_clearance_m
        sensitive_low_altitude_violation = (
            self.hard_sensitive_zones and flags["sensitive"] and z_agl < p.sensitive_min_overflight_agl_m
        )
        return {
            "terrain_collision": bool(terrain_collision),
            "building_collision": bool(building_collision),
            "building_clearance_violation": bool(building_clearance_violation),
            "canopy_collision": bool(canopy_collision),
            "canopy_clearance_violation": bool(canopy_clearance_violation),
            "sensitive_low_altitude_violation": bool(sensitive_low_altitude_violation),
            "any_hard_violation": bool(
                terrain_collision or building_collision or canopy_collision or sensitive_low_altitude_violation
            ),
        }

    def cost_components_at(self, x: float, y: float, z_abs: float) -> Dict[str, float]:
        p = self.profile
        flags = self.flags_at(x, y)
        heights = self.heights_at(x, y)
        clear = self.clearances_at(x, y, z_abs)
        constraints = self.constraints_at(x, y, z_abs)
        z_agl = max(0.0, self.z_agl_at(x, y, z_abs))
        road_cost = p.road_weight * self._decay(z_agl, p.road_h_scale_m) if flags["road"] else 0.0
        sidewalk_cost = p.sidewalk_weight * self._decay(z_agl, p.sidewalk_h_scale_m) if flags["sidewalk"] else 0.0
        sensitive_cost = p.sensitive_weight * self._decay(z_agl, p.sensitive_h_scale_m) if flags["sensitive"] else 0.0
        if constraints["sensitive_low_altitude_violation"]:
            sensitive_cost += p.sensitive_weight * p.sensitive_low_altitude_multiplier
        if flags["building"]:
            b_clear = clear["building_clearance_m"]
            if b_clear <= 0.0:
                building_cost = p.building_weight * p.building_collision_multiplier
            else:
                building_cost = p.building_weight * self._decay(b_clear, p.building_clearance_h_scale_m)
            if constraints["building_clearance_violation"]:
                building_cost += p.building_weight
        else:
            building_cost = 0.0
        if flags["canopy"]:
            c_clear = clear["canopy_clearance_m"]
            density = max(0.1, heights["canopy_density"])
            if c_clear <= 0.0:
                canopy_cost = p.canopy_weight * p.canopy_collision_multiplier * density
            else:
                canopy_cost = p.canopy_weight * density * self._decay(c_clear, p.canopy_clearance_h_scale_m)
            if constraints["canopy_clearance_violation"]:
                canopy_cost += p.canopy_weight * density
        else:
            canopy_cost = 0.0
        green_clean = flags["green"] and not flags["road"] and not flags["sidewalk"] and not flags["building"] and not flags["sensitive"]
        green_cost = p.green_weight * self._decay(z_agl, p.green_h_scale_m) if green_clean else 0.0
        terrain_cost = p.low_altitude_weight * p.terrain_collision_multiplier if constraints["terrain_collision"] else 0.0
        if self.include_global_altitude_band_cost:
            low_altitude_cost = p.low_altitude_weight * ((p.preferred_agl_low_m - z_agl) / max(1e-6, p.preferred_agl_low_m)) ** 2 if z_agl < p.preferred_agl_low_m else 0.0
            high_altitude_cost = p.high_altitude_weight * ((z_agl - p.preferred_agl_high_m) / max(1e-6, p.preferred_agl_high_m)) ** 2 if z_agl > p.preferred_agl_high_m else 0.0
        else:
            low_altitude_cost = high_altitude_cost = 0.0
        components = {
            "road_cost": float(road_cost),
            "sidewalk_cost": float(sidewalk_cost),
            "building_cost": float(building_cost),
            "green_cost": float(green_cost),
            "canopy_cost": float(canopy_cost),
            "sensitive_cost": float(sensitive_cost),
            "terrain_cost": float(terrain_cost),
            "low_altitude_cost": float(low_altitude_cost),
            "high_altitude_cost": float(high_altitude_cost),
        }
        components["total_risk"] = float(sum(components.values()))
        return components

    def risk_at(self, x: float, y: float, z_abs: float) -> float:
        return float(self.cost_components_at(x, y, z_abs)["total_risk"])

    def component_maps_at_agl(self, agl_m: float) -> Dict[str, np.ndarray]:
        p = self.profile
        agl = float(agl_m)
        road = self.world.road_mask
        sidewalk = self.world.sidewalk_mask
        building = self.world.building_mask
        green = self.world.green_mask
        canopy = self.world.canopy_mask
        sensitive = self.world.sensitive_mask
        road_cost = p.road_weight * road * self._decay(agl, p.road_h_scale_m)
        sidewalk_cost = p.sidewalk_weight * sidewalk * self._decay(agl, p.sidewalk_h_scale_m)
        sensitive_cost = p.sensitive_weight * sensitive * self._decay(agl, p.sensitive_h_scale_m)
        if self.hard_sensitive_zones and agl < p.sensitive_min_overflight_agl_m:
            sensitive_cost = sensitive_cost + sensitive * p.sensitive_weight * p.sensitive_low_altitude_multiplier
        building_clearance = agl - self.world.building_height
        building_cost = np.zeros(self.world.shape, dtype=np.float32)
        b_active = building > 0.5
        b_collision = b_active & (building_clearance <= 0.0)
        b_clearance_bad = b_active & (building_clearance > 0.0) & (building_clearance < p.building_min_clearance_m)
        b_ok = b_active & (building_clearance >= p.building_min_clearance_m)
        building_cost[b_collision] = p.building_weight * p.building_collision_multiplier
        building_cost[b_clearance_bad] = p.building_weight + p.building_weight * self._decay(building_clearance[b_clearance_bad], p.building_clearance_h_scale_m)
        building_cost[b_ok] = p.building_weight * self._decay(building_clearance[b_ok], p.building_clearance_h_scale_m)
        canopy_clearance = agl - self.world.canopy_height
        density = np.maximum(0.1, self.world.canopy_density)
        canopy_cost = np.zeros(self.world.shape, dtype=np.float32)
        c_active = canopy > 0.5
        c_collision = c_active & (canopy_clearance <= 0.0)
        c_clearance_bad = c_active & (canopy_clearance > 0.0) & (canopy_clearance < p.canopy_min_clearance_m)
        c_ok = c_active & (canopy_clearance >= p.canopy_min_clearance_m)
        canopy_cost[c_collision] = p.canopy_weight * p.canopy_collision_multiplier * density[c_collision]
        canopy_cost[c_clearance_bad] = p.canopy_weight * density[c_clearance_bad] + p.canopy_weight * density[c_clearance_bad] * self._decay(canopy_clearance[c_clearance_bad], p.canopy_clearance_h_scale_m)
        canopy_cost[c_ok] = p.canopy_weight * density[c_ok] * self._decay(canopy_clearance[c_ok], p.canopy_clearance_h_scale_m)
        green_clean = (green > 0.5) & (road <= 0.5) & (sidewalk <= 0.5) & (building <= 0.5) & (sensitive <= 0.5)
        green_cost = np.zeros(self.world.shape, dtype=np.float32)
        green_cost[green_clean] = p.green_weight * self._decay(agl, p.green_h_scale_m)
        low_altitude_cost = np.zeros(self.world.shape, dtype=np.float32)
        high_altitude_cost = np.zeros(self.world.shape, dtype=np.float32)
        if self.include_global_altitude_band_cost:
            if agl < p.preferred_agl_low_m:
                low_altitude_cost[:] = p.low_altitude_weight * ((p.preferred_agl_low_m - agl) / max(1e-6, p.preferred_agl_low_m)) ** 2
            if agl > p.preferred_agl_high_m:
                high_altitude_cost[:] = p.high_altitude_weight * ((agl - p.preferred_agl_high_m) / max(1e-6, p.preferred_agl_high_m)) ** 2
        maps = {
            "road_cost": road_cost.astype(np.float32),
            "sidewalk_cost": sidewalk_cost.astype(np.float32),
            "building_cost": building_cost.astype(np.float32),
            "green_cost": green_cost.astype(np.float32),
            "canopy_cost": canopy_cost.astype(np.float32),
            "sensitive_cost": sensitive_cost.astype(np.float32),
            "low_altitude_cost": low_altitude_cost,
            "high_altitude_cost": high_altitude_cost,
        }
        total = np.zeros(self.world.shape, dtype=np.float32)
        for v in maps.values():
            total += v.astype(np.float32)
        maps["total_risk"] = total
        return maps

    def risk_map_at_agl(self, agl_m: float) -> np.ndarray:
        return self.component_maps_at_agl(agl_m)["total_risk"]

    def normalized_risk_map_at_agl(self, agl_m: float) -> np.ndarray:
        risk = self.risk_map_at_agl(agl_m)
        maxv = float(np.nanmax(risk))
        return risk if maxv <= 1e-9 else (risk / maxv).astype(np.float32)
