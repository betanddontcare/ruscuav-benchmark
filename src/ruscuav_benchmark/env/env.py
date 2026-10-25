"""The RUSCUAV Gymnasium environment."""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..risk import UrbanRiskModel
from ..world import UrbanWorldLayers
from .cargo import CargoModel
from .config import EnvConfig
from .dynamic_risk import DynamicRiskField
from .dynamics import PointMassDynamics
from .exposure import ExposureTracker
from .hazards import UrbanHazardModel
from .lidar import UrbanLidar
from .observation import DynamicSampler, SemanticSampler, geometry_observation
from .reward import ProgressCounters, RewardModel
from .scenario import MissionSampler, endpoint_report
from .shield import UrbanShield

#: Value of ``info["min_clearance_safety_m"]``: the clearance to procedural
#: obstacles of the v2.0.0 environment, which no experiment used. Kept so episode
#: logs keep their columns.
NO_OBSTACLE_CLEARANCE_M = 1e9


class RuscuavEnv(gym.Env):
    """Low-altitude urban cargo-UAV navigation in a RUSCUAV world.

    Action: horizontal acceleration command in [-1, 1]^2 (scaled by ``a_max_xy``);
    altitude is held by the autopilot. Observation: see
    :mod:`ruscuav_benchmark.env.observation`. Reward and terminations: see
    :mod:`ruscuav_benchmark.env.reward`. Each ``step`` returns an ``info`` dict with
    the episode's exposure times, risk, constraint counts, mission acceptability
    and shield activity, from which the benchmark metrics are computed.

    The world must be in local metric coordinates with its origin at (0, 0); the
    workspace is the world extent horizontally and [0, ``dynamics.z_max_m``]
    vertically. The default parameters (normalisation scales, 500 m minimum mission
    distance, 150 m ceiling) were calibrated on 1000 m x 1000 m worlds.
    """

    metadata = {"render_modes": []}

    def __init__(self, world: UrbanWorldLayers, config: Optional[EnvConfig] = None) -> None:
        super().__init__()
        self.world = world
        self.config = cfg = config if config is not None else EnvConfig()

        # Workspace (scalars as Python floats, bounds arrays as float32).
        self.x_min, self.y_min, self.z_min = 0.0, 0.0, 0.0
        self.x_max = float(world.width * world.resolution_m)
        self.y_max = float(world.height * world.resolution_m)
        self.z_max = float(cfg.dynamics.z_max_m)
        self.bounds_low = np.array([self.x_min, self.y_min, self.z_min], dtype=np.float32)
        self.bounds_high = np.array([self.x_max, self.y_max, self.z_max], dtype=np.float32)

        # Components.
        self.risk_model = UrbanRiskModel(
            world=world,
            profile=cfg.risk_profile,
            hard_sensitive_zones=cfg.resolved_hard_sensitive_zones,
            include_global_altitude_band_cost=True,
        )
        self.dynamics = PointMassDynamics(cfg.dynamics, self.risk_model, self.bounds_low, self.bounds_high)
        self.hazards = UrbanHazardModel(self.risk_model, cfg.shield.horizontal_margin_m, self.bounds_low,
                                        self.bounds_high)
        self.lidar = UrbanLidar(cfg.lidar, self.hazards, cfg.resolved_lidar_step_m)
        self.shield = UrbanShield(self.hazards, self.dynamics, cfg.shield.action_shield, cfg.shield.geometric_shield,
                                  cfg.resolved_segment_check_step_m, cfg.shield.backoff_m)
        self.cargo = CargoModel(cfg.cargo)
        self.reward_model = RewardModel(cfg.rewards, cfg.episode, self.x_min, self.y_min, self.x_max, self.y_max)
        self.mission_sampler = MissionSampler(cfg.scenario, self.dynamics, self.x_max, self.y_max, self.z_max)
        self._check_world()
        self.exposure = ExposureTracker()
        self.semantic = SemanticSampler(self.risk_model, self.x_min, self.y_min, self.x_max, self.y_max)
        self.dynamic_config = cfg.dynamic_risk
        self.dynamic_sampler = (
            DynamicSampler(cfg.dynamic_risk, self.dynamics.ground_at, self.x_min, self.y_min, self.x_max, self.y_max)
            if cfg.dynamic_risk is not None else None
        )
        self.dynamic_field: Optional[DynamicRiskField] = None

        # Resolved urban settings.
        self.use_urban_risk_costs = cfg.use_urban_risk_costs
        self.urban_risk_lambda = cfg.resolved_urban_risk_lambda
        self.green_bonus_lambda = cfg.resolved_green_bonus_lambda
        self.terminate_on_urban_constraint = cfg.resolved_terminate_on_urban_constraint

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(cfg.observation_dim,),
                                            dtype=np.float32)

        # State. Before the first reset the mission is the diagonal of the spawn area.
        d = cfg.dynamics
        lo = self.mission_sampler.spawn_low
        hi_x, hi_y = self.mission_sampler.spawn_high_x, self.mission_sampler.spawn_high_y
        start_ground = self.dynamics.ground_at(lo, lo)
        target_ground = self.dynamics.ground_at(hi_x, hi_y)
        self.start_point = np.array(
            [lo, lo, min(self.z_max, max(start_ground + d.min_agl_m, start_ground + d.cruise_agl_m))],
            dtype=np.float32,
        )
        self.target_point = np.array(
            [hi_x, hi_y, min(self.z_max, max(target_ground + d.min_agl_m, target_ground + d.cruise_agl_m))],
            dtype=np.float32,
        )
        self._set_mission_axis()
        self.current_pos: Optional[np.ndarray] = None
        self.vel = np.zeros(3, dtype=np.float32)
        self.acc_xy_f = np.zeros(2, dtype=np.float32)
        self.prev_acc_xy = np.zeros(2, dtype=np.float32)
        self.acc_z_f = np.float32(0.0)
        self.prev_acc_z = np.float32(0.0)
        self.cargo_health = 1.0
        self.current_step = 0
        self.progress = ProgressCounters()
        self.cached_lidar = np.ones(cfg.lidar.n_rays, dtype=np.float32)
        self.last_urban_lidar_min_norm = 1.0
        self.last_urban_action_shielded = False
        self.last_urban_geom_shielded = False
        self.last_urban_shield_reason: Optional[str] = None

    # ------------------------------------------------------------------ properties
    @property
    def dt(self) -> float:
        return self.config.dynamics.dt

    @property
    def a_max_xy(self) -> float:
        return self.config.dynamics.a_max_xy

    @property
    def best_dist_xy(self) -> Optional[float]:
        return self.progress.best_dist_xy

    @best_dist_xy.setter
    def best_dist_xy(self, value: Optional[float]) -> None:
        self.progress.best_dist_xy = value

    @property
    def steps_since_improve(self) -> int:
        return self.progress.steps_since_improve

    @steps_since_improve.setter
    def steps_since_improve(self, value: int) -> None:
        self.progress.steps_since_improve = int(value)

    @property
    def shield_no_progress_steps(self) -> int:
        return self.progress.shield_no_progress_steps

    @shield_no_progress_steps.setter
    def shield_no_progress_steps(self, value: int) -> None:
        self.progress.shield_no_progress_steps = int(value)

    # ------------------------------------------------------------------ helpers
    def _check_world(self) -> None:
        """Fail early, with a clear message, on worlds where no mission can be sampled."""
        cfg = self.config
        sampler = self.mission_sampler
        if sampler.spawn_high_x <= sampler.spawn_low or sampler.spawn_high_y <= sampler.spawn_low:
            raise ValueError(
                f"World of {self.x_max:g} m x {self.y_max:g} m leaves no spawn area inside the "
                f"{cfg.scenario.spawn_margin_m} m margin (scenario.spawn_margin_m)")
        span_x = sampler.spawn_high_x - 1 - sampler.spawn_low
        span_y = sampler.spawn_high_y - 1 - sampler.spawn_low
        if float(np.hypot(span_x, span_y)) <= cfg.scenario.min_distance_m:
            raise ValueError(
                f"World of {self.x_max:g} m x {self.y_max:g} m is too small for missions at least "
                f"{cfg.scenario.min_distance_m:g} m long; lower scenario.min_distance_m")
        lowest = float(np.min(self.world.terrain_z))
        if lowest + cfg.dynamics.min_agl_m > self.z_max:
            raise ValueError(
                f"Terrain (lowest point {lowest:g} m) is above the workspace ceiling of {self.z_max:g} m "
                "(dynamics.z_max_m): give terrain heights relative to the area, or raise the ceiling")
        if abs(self.x_max - 1000.0) > 1e-6 or abs(self.y_max - 1000.0) > 1e-6:
            warnings.warn(
                f"RuscuavEnv defaults were calibrated on 1000 m x 1000 m worlds; this world is "
                f"{self.x_max:g} m x {self.y_max:g} m (observation scales and mission distances are unchanged)",
                stacklevel=3,
            )

    def _set_mission_axis(self) -> None:
        self.ideal_vec_2d = (self.target_point[:2] - self.start_point[:2]).astype(np.float32)
        self.ideal_len_2d = float(np.linalg.norm(self.ideal_vec_2d) + 1e-9)

    def dist_xy(self, pos) -> float:
        """Horizontal distance from ``pos`` to the target."""
        d = self.target_point[:2] - pos[:2]
        return float(np.linalg.norm(d))

    def cross_track_error(self) -> float:
        """Horizontal distance of the UAV from the start-target line."""
        p = self.current_pos[:2] - self.start_point[:2]
        cross = float(p[0] * self.ideal_vec_2d[1] - p[1] * self.ideal_vec_2d[0])
        return abs(cross) / self.ideal_len_2d

    def height_above_terrain(self, pos) -> float:
        return float(pos[2] - self.dynamics.ground_at(pos[0], pos[1]))

    def _observation(self) -> np.ndarray:
        pos = self.current_pos
        obs = geometry_observation(
            pos, self.target_point, self.vel, self.cached_lidar, self.cargo_health,
            self.dynamics.ground_at(pos[0], pos[1]), self.dist_xy(pos), self.config.dynamics,
            self.x_max, self.y_max, self.z_max,
        )
        parts = [obs]
        if self.config.semantic_observation:
            parts.append(self.semantic.observation(pos, self.target_point, self.exposure, self.config.mission))
        if self.config.dynamic_observation:
            parts.append(self.dynamic_sampler.observation(
                self.dynamic_field, pos, self.target_point, self.current_step, self.exposure.dynamic_risk_sum,
            ))
        if len(parts) == 1:
            return obs
        return np.concatenate(parts, axis=0).astype(np.float32)

    def _scan(self) -> None:
        self.cached_lidar, self.last_urban_lidar_min_norm = self.lidar.scan(self.current_pos, self.target_point)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
              ) -> Tuple[np.ndarray, Dict[str, Any]]:
        mission = self.config.mission
        attempts = mission.max_reset_attempts if mission.enforce_safe_start_target else 1
        safe, start_report, target_report = False, None, None
        for attempt in range(int(attempts)):
            # Unsafe endpoints: reseed deterministically and draw a new mission.
            attempt_seed = None if seed is None else int(seed) + 9973 * int(attempt)
            super().reset(seed=attempt_seed)
            self.start_point, self.target_point = self.mission_sampler.sample(self.np_random)
            self._set_mission_axis()
            start_report = endpoint_report(self.risk_model, self.start_point, mission.start_target_max_risk)
            target_report = endpoint_report(self.risk_model, self.target_point, mission.start_target_max_risk)
            safe = bool(start_report["safe"] and target_report["safe"])
            if (not mission.enforce_safe_start_target) or safe:
                break

        self.current_pos = self.start_point.copy()
        self.vel[:] = 0.0
        self.acc_xy_f[:] = 0.0
        self.prev_acc_xy[:] = 0.0
        self.acc_z_f = np.float32(0.0)
        self.prev_acc_z = np.float32(0.0)
        self.cargo_health = 1.0
        self.current_step = 0
        self.progress = ProgressCounters(best_dist_xy=self.dist_xy(self.current_pos))
        self.exposure.reset()
        self._scan()

        info: Dict[str, Any] = {
            "urban_safe_start_target": bool(safe),
            "urban_reset_attempts": int(attempt + 1),
            "start_urban_risk": float(start_report["risk"]),
            "target_urban_risk": float(target_report["risk"]),
            "start_urban_reason": start_report["reason"],
            "target_urban_reason": target_report["reason"],
        }

        if self.dynamic_config is not None:
            dyn_seed = int(seed) if seed is not None else int(self.np_random.integers(0, 2**31 - 1))
            self.dynamic_field = DynamicRiskField(world=self.world, config=self.dynamic_config, seed=dyn_seed,
                                                  bounds_xy=(self.x_min, self.y_min, self.x_max, self.y_max))
            info["dynamic_risk_enabled"] = True
            info["dynamic_hotspots"] = len(self.dynamic_field.hotspots)
            info["dynamic_seed"] = int(dyn_seed)

        obs = self._observation()
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape {obs.shape} != observation_space {self.observation_space.shape}")
        return obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        cfg = self.config
        d = cfg.dynamics
        self.last_urban_action_shielded = False
        self.last_urban_geom_shielded = False
        self.last_urban_shield_reason = None

        self.current_step += 1
        prev_pos = self.current_pos.copy()
        prev_vel = self.vel.copy()

        # Action -> shielded acceleration command -> dynamics.
        action = np.array(action, dtype=np.float32).reshape(2)
        desired_acc_xy = np.clip(action, -1.0, 1.0) * d.a_max_xy
        desired_acc_xy, reason = self.shield.filter_action(prev_pos, prev_vel, desired_acc_xy, self.dist_xy)
        if reason is not None:
            self.last_urban_action_shielded = True
            self.last_urban_shield_reason = reason

        self.acc_xy_f = self.dynamics.filter_acceleration_xy(self.acc_xy_f, self.prev_acc_xy, desired_acc_xy)
        self.prev_acc_xy = self.acc_xy_f.copy()
        z_ref = self.dynamics.z_reference(self.current_pos)
        self.acc_z_f = self.dynamics.altitude_command(z_ref, self.current_pos[2], self.vel[2], self.acc_z_f,
                                                      self.prev_acc_z)
        self.prev_acc_z = np.float32(self.acc_z_f)
        self.vel, desired_pos = self.dynamics.integrate(self.current_pos, self.vel, self.acc_xy_f, self.acc_z_f)

        # Geometric shield on the motion segment.
        safe_pos, shielded = self.shield.project_motion(prev_pos, desired_pos)
        if shielded:
            self.last_urban_geom_shielded = True
            self.last_urban_shield_reason = "urban_geometric_segment_hit"
        self.current_pos = safe_pos
        if shielded:
            self.vel = (self.current_pos - prev_pos) / d.dt
        eff_acc = (self.vel - prev_vel) / d.dt

        self._scan()
        self.cargo_health, _, excess = self.cargo.step(self.cargo_health, eff_acc)
        rel_h_m = self.height_above_terrain(self.current_pos)
        dist_xy = self.dist_xy(self.current_pos)
        prev_dist_xy = self.dist_xy(prev_pos)

        nav = self.reward_model.navigation(
            pos=self.current_pos, prev_pos=prev_pos, vel=self.vel, target=self.target_point,
            dist_xy=dist_xy, prev_dist_xy=prev_dist_xy, cross_track_m=self.cross_track_error(),
            lidar=self.cached_lidar, ground_z=self.dynamics.ground_at(self.current_pos[0], self.current_pos[1]),
            shielded=shielded, rel_h_m=rel_h_m, eff_acc=eff_acc, overload_excess_g=excess,
            cargo_health=self.cargo_health, step=self.current_step, counters=self.progress,
        )
        reward, terminated, truncated = nav.reward, nav.terminated, nav.truncated
        info: Dict[str, Any] = {
            "termination": nav.termination,
            "is_success": bool(nav.termination == "success"),
            "dist_xy": float(dist_xy),
            "cargo_health": float(self.cargo_health),
            "shielded": bool(shielded),
            "shield_no_progress_steps": int(self.progress.shield_no_progress_steps),
            "min_clearance_safety_m": float(NO_OBSTACLE_CLEARANCE_M),
            "rel_h_m": float(rel_h_m),
        }

        # Urban exposure, risk terms, operational stopping, mission acceptability.
        urban = self.exposure.update(self.risk_model, self.current_pos, d.dt)
        flags, components = urban["flags"], urban["components"]
        reward = self.reward_model.urban(reward, urban["total_risk"], flags, self.use_urban_risk_costs,
                                    self.urban_risk_lambda, self.green_bonus_lambda)
        if self.terminate_on_urban_constraint and (not terminated) and (not truncated):
            stop = self.exposure.operational_stop(urban["constraints"], cfg.mission)
            if stop is not None:
                terminated = True
                info["termination"] = stop
                info["is_success"] = False
                reward -= cfg.rewards.urban_failure_penalty
        termination = info.get("termination", None)
        mission_acceptable = self.exposure.mission_acceptable(termination, self.cargo_health, cfg.mission)
        if mission_acceptable and termination == "success":
            reward += float(cfg.rewards.acceptable_success_bonus)
        e = self.exposure
        mean_risk = e.risk_sum / max(1e-6, e.steps * d.dt)
        info.update({
            "urban_risk": float(urban["total_risk"]),
            "urban_mean_risk": float(mean_risk),
            "urban_max_risk": float(e.risk_max),
            "urban_cumulative_risk": float(e.risk_sum),
            "urban_road_cost": float(components["road_cost"]),
            "urban_sidewalk_cost": float(components["sidewalk_cost"]),
            "urban_building_cost": float(components["building_cost"]),
            "urban_green_cost": float(components["green_cost"]),
            "urban_canopy_cost": float(components["canopy_cost"]),
            "urban_sensitive_cost": float(components["sensitive_cost"]),
            "urban_road_time": float(e.road_time),
            "urban_sidewalk_time": float(e.sidewalk_time),
            "urban_building_time": float(e.building_time),
            "urban_green_time": float(e.green_time),
            "urban_canopy_time": float(e.canopy_time),
            "urban_sensitive_time": float(e.sensitive_time),
            "urban_building_collision_steps": int(e.building_collision_steps),
            "urban_building_clearance_violation_steps": int(e.building_clearance_violation_steps),
            "urban_canopy_collision_steps": int(e.canopy_collision_steps),
            "urban_canopy_clearance_violation_steps": int(e.canopy_clearance_violation_steps),
            "urban_sensitive_low_altitude_violation_steps": int(e.sensitive_low_altitude_violation_steps),
            "urban_hard_violation_steps": int(e.hard_violation_steps),
            "urban_building": bool(flags["building"]),
            "urban_road": bool(flags["road"]),
            "urban_sidewalk": bool(flags["sidewalk"]),
            "urban_green": bool(flags["green"]),
            "urban_canopy": bool(flags["canopy"]),
            "urban_sensitive": bool(flags["sensitive"]),
            "mission_acceptable": bool(mission_acceptable),
        })
        info.update({
            "urban_action_shielded": bool(self.last_urban_action_shielded),
            "urban_geom_shielded": bool(self.last_urban_geom_shielded),
            "urban_shielded_any": bool(self.last_urban_action_shielded or self.last_urban_geom_shielded),
            "urban_lidar_min_norm": float(self.last_urban_lidar_min_norm),
            "urban_shield_reason": self.last_urban_shield_reason,
            "urban_signed_clearance_proxy_m": float(self.hazards.signed_clearance_m(self.current_pos)),
        })

        # Dynamic risk.
        if self.dynamic_config is not None:
            dc = self.dynamic_config
            if self.dynamic_field is None:
                dyn_risk = 0.0
            else:
                dyn_risk = float(self.dynamic_field.risk_at_pos(self.current_pos, self.current_step))
            dt = float(d.dt)
            e.update_dynamic(dyn_risk, dt)
            reward = self.reward_model.dynamic(reward, dyn_risk, dc.dynamic_risk_lambda)
            base_acceptable = bool(info.get("mission_acceptable", False))
            dynamic_budget_ok = e.dynamic_risk_sum <= float(dc.dynamic_cumulative_risk_budget)
            dynamic_mission_acceptable = bool(base_acceptable and dynamic_budget_ok)
            info["mission_acceptable_static_only"] = base_acceptable
            info["mission_acceptable"] = dynamic_mission_acceptable
            info.update({
                "dynamic_risk": float(dyn_risk),
                "dynamic_risk_mean": float(e.dynamic_risk_sum / max(1e-6, e.dynamic_steps * dt)),
                "dynamic_risk_max": float(e.dynamic_risk_max),
                "dynamic_cumulative_risk": float(e.dynamic_risk_sum),
                "dynamic_budget_ok": bool(dynamic_budget_ok),
                "dynamic_mission_acceptable": bool(dynamic_mission_acceptable),
            })

        return self._observation(), float(reward), bool(terminated), bool(truncated), info
