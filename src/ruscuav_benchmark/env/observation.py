"""Observation vectors.

Geometry observation (16 + n_rays values, always present; indices for the default
64 LiDAR rays)::

    [0:3]   position / (x_max, y_max, z_max)
    [3:5]   target - position (x, y) / 1000 m
    [5]     horizontal distance to target / 1500 m, clipped to [0, 1]
    [6:8]   unit direction to target
    [8]     horizontal speed / v_max_xy, clipped to [0, 1]
    [9]     cosine between velocity and target direction (0 when hovering)
    [10]    height above terrain / 100 m
    [11:75] LiDAR readings (see lidar.py)
    [75:78] velocity / (v_max_xy, v_max_xy, v_max_z)
    [78]    overload proxy; always 0 (the archived environment never updated it,
            and the archived policies were trained with it at 0)
    [79]    cargo health

Semantic observation (20 values, appended for "semantic" observations): risk,
class flags and clearances under the UAV, risk and class shares sampled ahead,
risk sampled to the sides, and the exposure budgets used so far.

Dynamic observation (6 values, appended last for "dynamic" observations):
normalised dynamic risk here, ahead (mean, max), left and right (mean), and the
dynamic risk budget used so far.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from ..risk import UrbanRiskModel
from .config import DynamicsConfig, MissionConfig
from .dynamic_risk import DynamicRiskConfig, DynamicRiskField
from .exposure import ExposureTracker

URBAN_VECTOR_DIM = 20
DYNAMIC_VECTOR_DIM = 6
#: Look-ahead distances of the semantic and dynamic samples (metres).
AHEAD_DISTANCES_M = (20.0, 50.0, 100.0, 150.0, 220.0)
#: Risk normalisation of the semantic observation (sensitive low-altitude risk can be large).
URBAN_RISK_NORM = 300.0


def geometry_observation(pos: np.ndarray, target: np.ndarray, vel: np.ndarray, lidar: np.ndarray,
                         cargo_health: float, ground_z: float, dist_xy: float, dynamics: DynamicsConfig,
                         x_max: float, y_max: float, z_max: float) -> np.ndarray:
    pos_norm = pos / np.array([x_max, y_max, z_max], dtype=np.float32)
    rel_target_xy = (target[:2] - pos[:2]) / 1000.0
    dist_xy_norm = np.float32(np.clip(dist_xy / 1500.0, 0.0, 1.0))
    to_t = (target[:2] - pos[:2]).astype(np.float32)
    to_t_norm = float(np.linalg.norm(to_t) + 1e-9)
    dir_to_target = (to_t / to_t_norm).astype(np.float32)
    vxy = vel[:2].astype(np.float32)
    vxy_norm = float(np.linalg.norm(vxy) + 1e-9)
    speed_xy_norm = np.float32(np.clip(vxy_norm / dynamics.v_max_xy, 0.0, 1.0))
    if vxy_norm > 1e-6:
        heading_cos = np.float32(np.dot((vxy / vxy_norm).astype(np.float32), dir_to_target))
    else:
        heading_cos = np.float32(0.0)
    rel_h_m = float(pos[2] - ground_z)
    rel_h_norm = np.float32(rel_h_m / 100.0)
    vel_norm = vel / np.array([dynamics.v_max_xy, dynamics.v_max_xy, dynamics.v_max_z], dtype=np.float32)
    over_g_proxy = np.float32(0.0)
    return np.concatenate([
        pos_norm.astype(np.float32),
        rel_target_xy.astype(np.float32),
        np.array([dist_xy_norm], dtype=np.float32),
        dir_to_target.astype(np.float32),
        np.array([speed_xy_norm], dtype=np.float32),
        np.array([heading_cos], dtype=np.float32),
        np.array([rel_h_norm], dtype=np.float32),
        lidar.astype(np.float32),
        vel_norm.astype(np.float32),
        np.array([over_g_proxy], dtype=np.float32),
        np.array([np.float32(cargo_health)], dtype=np.float32),
    ], axis=0).astype(np.float32)


class SemanticSampler:
    """Risk and class shares sampled along a direction at the current altitude."""

    def __init__(self, risk_model: UrbanRiskModel, x_min: float, y_min: float, x_max: float, y_max: float) -> None:
        self.risk_model = risk_model
        self.x_min, self.y_min, self.x_max, self.y_max = float(x_min), float(y_min), float(x_max), float(y_max)

    def along(self, pos: np.ndarray, direction_xy, distances_m) -> dict:
        origin = pos[:2].astype(np.float32)
        d = np.asarray(direction_xy, dtype=np.float32)
        n = float(np.linalg.norm(d) + 1e-9)
        d = d / n
        z_abs = float(pos[2])
        risks, roads, sidewalks, buildings, greens, canopies, sensitives = [], [], [], [], [], [], []
        for dist in distances_m:
            p = origin + d * float(dist)
            x = float(np.clip(p[0], self.x_min, self.x_max - 1e-3))
            y = float(np.clip(p[1], self.y_min, self.y_max - 1e-3))
            risks.append(self.risk_model.risk_at(x, y, z_abs))
            flags = self.risk_model.flags_at(x, y)
            roads.append(float(flags["road"]))
            sidewalks.append(float(flags["sidewalk"]))
            buildings.append(float(flags["building"]))
            greens.append(float(flags["green"]))
            canopies.append(float(flags["canopy"]))
            sensitives.append(float(flags["sensitive"]))
        return {
            "risk_mean": float(np.mean(risks)) if risks else 0.0,
            "risk_max": float(np.max(risks)) if risks else 0.0,
            "road_mean": float(np.mean(roads)) if roads else 0.0,
            "sidewalk_mean": float(np.mean(sidewalks)) if sidewalks else 0.0,
            "building_mean": float(np.mean(buildings)) if buildings else 0.0,
            "green_mean": float(np.mean(greens)) if greens else 0.0,
            "canopy_mean": float(np.mean(canopies)) if canopies else 0.0,
            "sensitive_mean": float(np.mean(sensitives)) if sensitives else 0.0,
        }

    def observation(self, pos: np.ndarray, target: np.ndarray, exposure: ExposureTracker,
                    mission: MissionConfig) -> np.ndarray:
        rm = self.risk_model
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        flags = rm.flags_at(x, y)
        clear = rm.clearances_at(x, y, z)
        constraints = rm.constraints_at(x, y, z)
        components = rm.cost_components_at(x, y, z)
        risk_norm = float(np.clip(components["total_risk"] / URBAN_RISK_NORM, 0.0, 1.0))
        b_clear = clear["building_clearance_m"]
        c_clear = clear["canopy_clearance_m"]
        b_clear_norm = 1.0 if b_clear > 1e8 else float(np.clip(b_clear / 80.0, -1.0, 1.0))
        c_clear_norm = 1.0 if c_clear > 1e8 else float(np.clip(c_clear / 40.0, -1.0, 1.0))
        to_target = (target[:2] - pos[:2]).astype(np.float32)
        to_target_norm = float(np.linalg.norm(to_target) + 1e-9)
        forward = to_target / to_target_norm
        left = np.array([-forward[1], forward[0]], dtype=np.float32)
        right = -left
        distances = np.array(AHEAD_DISTANCES_M, dtype=np.float32)
        fwd = self.along(pos, forward, distances)
        lft = self.along(pos, left, distances[:4])
        rgt = self.along(pos, right, distances[:4])
        road_budget = max(1e-6, mission.accept_road_s)
        sidewalk_budget = max(1e-6, mission.accept_sidewalk_s)
        building_budget = max(1e-6, mission.accept_building_s)
        sensitive_budget = max(1.0, mission.accept_sensitive_s + 1.0)
        obs = np.array([
            risk_norm,
            float(flags["building"]),
            float(flags["road"]),
            float(flags["sidewalk"]),
            float(flags["green"]),
            float(flags["canopy"]),
            float(flags["sensitive"]),
            b_clear_norm,
            c_clear_norm,
            float(constraints["any_hard_violation"]),
            float(np.clip(fwd["risk_mean"] / URBAN_RISK_NORM, 0.0, 1.0)),
            float(np.clip(fwd["risk_max"] / URBAN_RISK_NORM, 0.0, 1.0)),
            fwd["road_mean"],
            fwd["sidewalk_mean"],
            fwd["building_mean"],
            fwd["green_mean"],
            float(np.clip(lft["risk_mean"] / URBAN_RISK_NORM, 0.0, 1.0)),
            float(np.clip(rgt["risk_mean"] / URBAN_RISK_NORM, 0.0, 1.0)),
            float(np.clip(exposure.road_time / road_budget, 0.0, 1.0)),
            float(np.clip(
                exposure.sidewalk_time / sidewalk_budget
                + exposure.building_time / building_budget
                + exposure.sensitive_time / sensitive_budget,
                0.0,
                1.0,
            )),
        ], dtype=np.float32)
        assert obs.shape[0] == URBAN_VECTOR_DIM
        return obs


class DynamicSampler:
    """Dynamic risk at the current altitude, here and along three directions."""

    def __init__(self, config: DynamicRiskConfig, ground_at: Callable[[float, float], float], x_min: float,
                 y_min: float, x_max: float, y_max: float) -> None:
        self.config = config
        self.ground_at = ground_at
        self.x_min, self.y_min, self.x_max, self.y_max = float(x_min), float(y_min), float(x_max), float(y_max)

    def risk_norm(self, risk_value: float) -> float:
        """Maps roughly 0..50 risk to 0..1 (denominator ``dynamic_norm_denom``, clip at 5)."""
        denom = max(1e-6, float(self.config.dynamic_norm_denom))
        x = float(risk_value) / denom
        x = np.clip(x, 0.0, 5.0) / 5.0
        return float(x)

    def risk_at_xy(self, field: Optional[DynamicRiskField], x: float, y: float, z_abs: float, step: int) -> float:
        if field is None:
            return 0.0
        x = float(np.clip(x, self.x_min, self.x_max - 1e-3))
        y = float(np.clip(y, self.y_min, self.y_max - 1e-3))
        z_abs = float(z_abs)
        ground = float(self.ground_at(x, y))
        agl_m = float(z_abs - ground)
        return float(field.risk_at_xy_agl(x=x, y=y, agl_m=agl_m, step=int(step)))

    def along(self, field: Optional[DynamicRiskField], pos: np.ndarray, direction_xy, distances_m,
              step: int) -> dict:
        if field is None:
            return {"mean": 0.0, "max": 0.0}
        pos_xy = pos[:2].astype(np.float32)
        d = np.asarray(direction_xy, dtype=np.float32)
        n = float(np.linalg.norm(d) + 1e-9)
        if n < 1e-8:
            d = np.array([1.0, 0.0], dtype=np.float32)
        else:
            d = d / n
        vals = []
        for dist in distances_m:
            p = pos_xy + d * float(dist)
            x = float(np.clip(p[0], self.x_min, self.x_max - 1e-3))
            y = float(np.clip(p[1], self.y_min, self.y_max - 1e-3))
            vals.append(self.risk_at_xy(field, x, y, float(pos[2]), step))
        if len(vals) == 0:
            return {"mean": 0.0, "max": 0.0}
        vals = np.asarray(vals, dtype=np.float32)
        return {"mean": float(np.mean(vals)), "max": float(np.max(vals))}

    def observation(self, field: Optional[DynamicRiskField], pos: np.ndarray, target: np.ndarray, step: int,
                    dynamic_risk_sum: float) -> np.ndarray:
        current = self.risk_at_xy(field, float(pos[0]), float(pos[1]), float(pos[2]), step)
        to_target = (target[:2] - pos[:2]).astype(np.float32)
        to_target_norm = float(np.linalg.norm(to_target) + 1e-9)
        if to_target_norm < 1e-8:
            forward = np.array([1.0, 0.0], dtype=np.float32)
        else:
            forward = to_target / to_target_norm
        left = np.array([-forward[1], forward[0]], dtype=np.float32)
        right = -left
        distances_fwd = np.array(AHEAD_DISTANCES_M, dtype=np.float32)
        distances_side = np.array(AHEAD_DISTANCES_M[:4], dtype=np.float32)
        fwd = self.along(field, pos, forward, distances_fwd, step)
        lft = self.along(field, pos, left, distances_side, step)
        rgt = self.along(field, pos, right, distances_side, step)
        budget = max(1e-6, float(self.config.dynamic_cumulative_risk_budget))
        budget_used = float(np.clip(dynamic_risk_sum / budget, 0.0, 1.0))
        obs = np.array([
            self.risk_norm(current),
            self.risk_norm(fwd["mean"]),
            self.risk_norm(fwd["max"]),
            self.risk_norm(lft["mean"]),
            self.risk_norm(rgt["mean"]),
            budget_used,
        ], dtype=np.float32)
        assert obs.shape[0] == DYNAMIC_VECTOR_DIM
        return obs
