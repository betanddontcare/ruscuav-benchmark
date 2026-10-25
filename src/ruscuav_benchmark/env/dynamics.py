"""Point-mass UAV dynamics with an altitude autopilot.

The policy commands a horizontal acceleration. It is low-pass filtered and jerk
limited; the vertical acceleration comes from a PD autopilot that tracks a
terrain-following altitude reference (raised over buildings and canopy). Speeds
are clamped and the position is integrated with a fixed time step.

All state arrays are float32, as in the archived environment; the arithmetic is
kept operation for operation so trajectories are bit-identical.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..risk import UrbanRiskModel
from .config import DynamicsConfig


class PointMassDynamics:
    def __init__(self, config: DynamicsConfig, risk_model: UrbanRiskModel, bounds_low: np.ndarray,
                 bounds_high: np.ndarray) -> None:
        self.config = config
        self.risk_model = risk_model
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.z_max = float(config.z_max_m)
        self.wind_xy = np.array(config.wind_xy, dtype=np.float32)

    # ------------------------------------------------------------------ terrain
    def ground_at(self, x, y) -> float:
        return self.risk_model.terrain_at(x, y)

    def reference_z_from_ground(self, ground: float) -> Optional[float]:
        """Cruise altitude over a ground height, or None if the altitude band is empty."""
        c = self.config
        floor = ground + c.min_agl_m
        ceiling = min(self.z_max, ground + c.max_agl_m)
        if floor > ceiling:
            return None
        target = ground + c.cruise_agl_m
        return float(np.clip(target, floor, ceiling))

    def z_reference(self, pos) -> float:
        """Altitude reference of the autopilot at the horizontal position of ``pos``."""
        c = self.config
        if not c.urban_altitude_reference:
            ground_here = self.ground_at(pos[0], pos[1])
            floor = ground_here + c.min_agl_m
            ceiling = min(self.z_max, ground_here + c.max_agl_m)
            if floor > ceiling:
                return float(ceiling)
            target = ground_here + c.cruise_agl_m
            return float(np.clip(target, floor, ceiling))
        x, y = float(pos[0]), float(pos[1])
        ground = self.ground_at(x, y)
        floor = ground + c.min_agl_m
        ceiling = min(self.z_max, ground + c.max_agl_m)
        if floor > ceiling:
            return float(ceiling)
        z_ref = ground + c.cruise_agl_m
        flags = self.risk_model.flags_at(x, y)
        heights = self.risk_model.heights_at(x, y)
        p = self.risk_model.profile
        if flags["building"]:
            z_ref = max(z_ref, heights["building_roof_z"] + p.building_min_clearance_m)
        if flags["canopy"]:
            z_ref = max(z_ref, heights["canopy_top_z"] + p.canopy_min_clearance_m)
        return float(np.clip(z_ref, floor, ceiling))

    # ------------------------------------------------------------------ one step
    def filter_acceleration_xy(self, acc_xy_f: np.ndarray, prev_acc_xy: np.ndarray,
                               desired_acc_xy: np.ndarray) -> np.ndarray:
        """Low-pass filter and jerk limit of the horizontal acceleration command."""
        c = self.config
        acc_xy_f = c.alpha_xy * acc_xy_f + (1.0 - c.alpha_xy) * desired_acc_xy
        jerk_xy = (acc_xy_f - prev_acc_xy) / c.dt
        jerk_xy_mag = float(np.linalg.norm(jerk_xy))
        if jerk_xy_mag > c.jerk_max:
            jerk_xy = jerk_xy * (c.jerk_max / (jerk_xy_mag + 1e-9))
            acc_xy_f = prev_acc_xy + jerk_xy * c.dt
        return acc_xy_f

    def altitude_command(self, z_ref: float, z: float, vz: float, acc_z_f: np.float32,
                         prev_acc_z: np.float32) -> np.float32:
        """PD autopilot towards ``z_ref``, filtered and jerk limited like the horizontal command."""
        c = self.config
        z_err = float(z_ref - float(z))
        vz = float(vz)
        az_cmd = c.kp_z * z_err - c.kd_z * vz
        az_cmd = float(np.clip(az_cmd, -c.a_max_z, c.a_max_z))
        acc_z_f = np.float32(c.alpha_z * float(acc_z_f) + (1.0 - c.alpha_z) * az_cmd)
        jerk_z = float((float(acc_z_f) - float(prev_acc_z)) / c.dt)
        if abs(jerk_z) > c.jerk_max:
            jerk_z = float(np.sign(jerk_z) * c.jerk_max)
            acc_z_f = np.float32(float(prev_acc_z) + jerk_z * c.dt)
        return acc_z_f

    def integrate(self, pos: np.ndarray, vel: np.ndarray, acc_xy_f: np.ndarray,
                  acc_z_f: np.float32) -> Tuple[np.ndarray, np.ndarray]:
        """New velocity (speed-clamped) and the commanded position, clipped to the workspace."""
        c = self.config
        acc_cmd = np.array([acc_xy_f[0], acc_xy_f[1], float(acc_z_f)], dtype=np.float32)
        vel = vel + acc_cmd * c.dt
        vxy = vel[:2]
        vxy_mag = float(np.linalg.norm(vxy))
        if vxy_mag > c.v_max_xy:
            vel[:2] = vxy * (c.v_max_xy / (vxy_mag + 1e-9))
        if abs(float(vel[2])) > c.v_max_z:
            vel[2] = np.float32(np.sign(vel[2]) * c.v_max_z)
        desired_pos = pos + vel * c.dt
        desired_pos[:2] = desired_pos[:2] + self.wind_xy * c.dt
        desired_pos = np.clip(desired_pos, self.bounds_low, self.bounds_high)
        return vel, desired_pos

    def predict_next_pos_xy(self, pos, vel, acc_xy) -> np.ndarray:
        """Horizontal position after one step with acceleration ``acc_xy`` (no filter)."""
        c = self.config
        pos = pos.astype(np.float32)
        vel = vel.astype(np.float32)
        acc_xy = acc_xy.astype(np.float32)
        v_next_xy = vel[:2] + acc_xy * c.dt
        v_mag = float(np.linalg.norm(v_next_xy))
        if v_mag > c.v_max_xy:
            v_next_xy = v_next_xy * (c.v_max_xy / (v_mag + 1e-9))
        p_next_xy = pos[:2] + v_next_xy * c.dt
        p_next_xy = np.clip(p_next_xy, self.bounds_low[:2], self.bounds_high[:2])
        return p_next_xy.astype(np.float32)
