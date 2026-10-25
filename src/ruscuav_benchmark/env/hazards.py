"""Urban hazard queries shared by the LiDAR, the shields and the metrics.

Hazards are a 2.5-D approximation: buildings and canopy are raster footprints with
a roof or canopy-top height, sensitive zones are no-fly volumes below the
overflight altitude, and the terrain has a minimum clearance. Footprint distances
come from signed distance fields (positive outside, negative inside, metres).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from ..risk import UrbanRiskModel


def signed_distance_field(mask: np.ndarray, resolution_m: float) -> np.ndarray:
    """Signed distance to the footprint ``mask > 0.5``: positive outside, negative inside."""
    mask = np.asarray(mask > 0.5, dtype=bool)
    outside_dist = distance_transform_edt(~mask) * resolution_m
    inside_dist = distance_transform_edt(mask) * resolution_m
    return (outside_dist - inside_dist).astype(np.float32)


class UrbanHazardModel:
    def __init__(self, risk_model: UrbanRiskModel, horizontal_margin_m: float, bounds_low: np.ndarray,
                 bounds_high: np.ndarray) -> None:
        self.risk_model = risk_model
        self.world = risk_model.world
        self.horizontal_margin_m = float(max(0.0, horizontal_margin_m))
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        res = float(self.world.resolution_m)
        self.building_sdf_m = signed_distance_field(self.world.building_mask, res)
        self.canopy_sdf_m = signed_distance_field(self.world.canopy_mask, res)
        self.sensitive_sdf_m = signed_distance_field(self.world.sensitive_mask, res)

    def sdf_at_xy(self, sdf: np.ndarray, x: float, y: float) -> float:
        yi, xi = self.risk_model._ij_from_xy(x, y)
        return float(sdf[yi, xi])

    def constraints_at(self, pos) -> dict:
        return self.risk_model.constraints_at(float(pos[0]), float(pos[1]), float(pos[2]))

    def is_unsafe(self, pos, include_clearance_violation: bool = True) -> bool:
        """True inside a hard hazard: terrain, building or canopy collision, or a
        sensitive zone below the overflight altitude; with
        ``include_clearance_violation`` also below the minimum clearance over a
        building or canopy."""
        c = self.constraints_at(pos)
        if c["terrain_collision"]:
            return True
        if c["building_collision"]:
            return True
        if c["canopy_collision"]:
            return True
        if c["sensitive_low_altitude_violation"]:
            return True
        if include_clearance_violation:
            if c["building_clearance_violation"]:
                return True
            if c["canopy_clearance_violation"]:
                return True
        return False

    def signed_clearance_m(self, pos) -> float:
        """Signed-clearance proxy (metres; negative inside a hazard).

        Minimum of the terrain clearance, the vertical clearance over a building or
        canopy footprint (footprint SDF otherwise) and, below the sensitive-zone
        overflight altitude, the sensitive-footprint SDF; each above its threshold.
        Not an exact 3-D distance, but stable and cheap enough to score shield
        candidates.
        """
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        p = self.risk_model.profile
        flags = self.risk_model.flags_at(x, y)
        clear = self.risk_model.clearances_at(x, y, z)
        z_agl = self.risk_model.z_agl_at(x, y, z)
        vals = [clear["terrain_clearance_m"] - p.terrain_min_clearance_m]
        if flags["building"]:
            vals.append(clear["building_clearance_m"] - p.building_min_clearance_m)
        else:
            vals.append(self.sdf_at_xy(self.building_sdf_m, x, y) - self.horizontal_margin_m)
        if flags["canopy"]:
            vals.append(clear["canopy_clearance_m"] - p.canopy_min_clearance_m)
        else:
            vals.append(self.sdf_at_xy(self.canopy_sdf_m, x, y) - self.horizontal_margin_m)
        if z_agl < p.sensitive_min_overflight_agl_m:
            vals.append(self.sdf_at_xy(self.sensitive_sdf_m, x, y) - self.horizontal_margin_m)
        return float(np.min(vals))

    def clearance_gradient_xy(self, pos, eps_m: float = 2.0) -> np.ndarray:
        """Unit finite-difference gradient of :meth:`signed_clearance_m` (zero if flat)."""
        pos = np.asarray(pos, dtype=np.float32)
        px = pos.copy()
        mx = pos.copy()
        py = pos.copy()
        my = pos.copy()
        px[0] += eps_m
        mx[0] -= eps_m
        py[1] += eps_m
        my[1] -= eps_m
        px[:2] = np.clip(px[:2], self.bounds_low[:2], self.bounds_high[:2])
        mx[:2] = np.clip(mx[:2], self.bounds_low[:2], self.bounds_high[:2])
        py[:2] = np.clip(py[:2], self.bounds_low[:2], self.bounds_high[:2])
        my[:2] = np.clip(my[:2], self.bounds_low[:2], self.bounds_high[:2])
        gx = self.signed_clearance_m(px) - self.signed_clearance_m(mx)
        gy = self.signed_clearance_m(py) - self.signed_clearance_m(my)
        g = np.array([gx, gy], dtype=np.float32)
        n = float(np.linalg.norm(g) + 1e-9)
        if n < 1e-6:
            return np.zeros(2, dtype=np.float32)
        return (g / n).astype(np.float32)
