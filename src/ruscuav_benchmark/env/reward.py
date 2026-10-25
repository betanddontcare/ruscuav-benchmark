"""Reward terms and termination conditions.

The step reward is accumulated in this order (the order matters for floating point
reproducibility of the archived results):

1. navigation shaping: progress towards the target, heading alignment, low-speed
   and time penalties, cross-track corridor penalty, LiDAR proximity penalty;
2. terrain collision (terminal), shield penalties and the shield-stuck termination;
3. altitude, acceleration, overload and workspace-boundary penalties;
4. success, near-success, stuck, cargo failure and timeout;
5. urban terms: semantic risk penalty and green-corridor bonus, operational
   stopping (stage C) and the bonus for an acceptable success;
6. dynamic risk penalty.

Steps 1-4 are :meth:`RewardModel.navigation`, 5 is :meth:`RewardModel.urban` and
6 is :meth:`RewardModel.dynamic`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import EpisodeConfig, RewardConfig


@dataclass
class NavigationOutcome:
    reward: float
    terminated: bool
    truncated: bool
    termination: Optional[str]


@dataclass
class ProgressCounters:
    """Episode counters for the stuck and shield-stuck terminations."""

    best_dist_xy: Optional[float] = None
    steps_since_improve: int = 0
    shield_no_progress_steps: int = 0


class RewardModel:
    def __init__(self, rewards: RewardConfig, episode: EpisodeConfig, x_min: float, y_min: float,
                 x_max: float, y_max: float) -> None:
        self.rewards = rewards
        self.episode = episode
        self.x_min, self.y_min, self.x_max, self.y_max = float(x_min), float(y_min), float(x_max), float(y_max)

    def boundary_penalty(self, pos_xy) -> float:
        """Quadratic penalty within ``boundary_margin_m`` of the workspace edge."""
        r = self.rewards
        x, y = float(pos_xy[0]), float(pos_xy[1])
        margin = r.boundary_margin_m
        k = r.boundary_k
        pen = 0.0
        if x > (self.x_max - margin):
            pen += k * (x - (self.x_max - margin)) ** 2
        elif x < (self.x_min + margin):
            pen += k * ((self.x_min + margin) - x) ** 2
        if y > (self.y_max - margin):
            pen += k * (y - (self.y_max - margin)) ** 2
        elif y < (self.y_min + margin):
            pen += k * ((self.y_min + margin) - y) ** 2
        return float(min(r.boundary_cap, pen))

    def navigation(self, *, pos: np.ndarray, prev_pos: np.ndarray, vel: np.ndarray, target: np.ndarray,
                   dist_xy: float, prev_dist_xy: float, cross_track_m: float, lidar: np.ndarray,
                   ground_z: float, shielded: bool, rel_h_m: float, eff_acc: np.ndarray, overload_excess_g: float,
                   cargo_health: float, step: int, counters: ProgressCounters) -> NavigationOutcome:
        """Navigation reward and terminations of one step; updates ``counters``."""
        r = self.rewards
        e = self.episode
        reward = 0.0
        terminated, truncated = False, False
        termination = None

        reward += r.progress_scale * (prev_dist_xy - dist_xy)

        to_t = (target[:2] - pos[:2]).astype(np.float32)
        to_t_norm = float(np.linalg.norm(to_t) + 1e-9)
        vxy_u = vel[:2] / (float(np.linalg.norm(vel[:2])) + 1e-9)
        align = float(np.dot(to_t / to_t_norm, vxy_u))
        reward += r.align_k * align

        if float(np.linalg.norm(vel[:2])) < r.speed_min_mps:
            reward -= r.low_speed_penalty

        reward -= r.time_penalty

        cte = cross_track_m
        cte_pen = (r.cte_quad_k * (max(0, cte - r.cte_soft_limit_m) ** 2)
                   + r.cte_lin_k * max(0, cte - r.cte_corridor_limit_m))
        reward -= float(np.clip(cte_pen, 0.0, r.cte_cap))

        min_l_norm = float(np.min(lidar))
        prox = max(0.0, r.lidar_trigger - min_l_norm)
        reward -= float(np.clip(r.lidar_k * (prox ** 2), 0.0, r.lidar_cap))

        if (not terminated) and (pos[2] <= ground_z + e.terrain_margin_m):
            reward -= r.terrain_collision_penalty
            terminated = True
            termination = "terrain_collision"

        step_xy_m = float(np.linalg.norm((pos[:2] - prev_pos[:2])))
        if shielded:
            reward -= r.shield_penalty
            if step_xy_m < r.shield_blocked_step_m:
                reward -= r.shield_blocked_penalty
            progressed = dist_xy < (prev_dist_xy - e.shield_progress_eps_m)
            if (not progressed) or (step_xy_m < e.shield_min_step_xy_m):
                counters.shield_no_progress_steps += 1
            else:
                counters.shield_no_progress_steps = 0
        else:
            counters.shield_no_progress_steps = 0

        if (not terminated) and (counters.shield_no_progress_steps >= e.shield_stuck_patience_steps):
            reward -= r.shield_stuck_penalty
            terminated = True
            termination = "shield_stuck"

        reward -= r.altitude_k * (abs(rel_h_m - r.altitude_ref_agl_m) / r.altitude_ref_agl_m)
        reward -= r.acceleration_k * float(np.dot(eff_acc, eff_acc))
        reward -= r.overload_k * (overload_excess_g ** 2)
        reward -= self.boundary_penalty(pos[:2])

        if counters.best_dist_xy is None:
            counters.best_dist_xy = dist_xy
        if dist_xy < (counters.best_dist_xy - e.improve_eps_m):
            counters.best_dist_xy = dist_xy
            counters.steps_since_improve = 0
        else:
            counters.steps_since_improve += 1

        if (not terminated) and (dist_xy < e.success_radius_m):
            reward += r.success_bonus
            terminated = True
            termination = "success"

        if (not terminated) and (counters.steps_since_improve >= e.stuck_patience_steps):
            if e.success_radius_m < dist_xy <= e.near_success_radius_m:
                reward += r.near_success_bonus
                terminated = True
                termination = "near_success"
            else:
                reward -= r.stuck_penalty
                terminated = True
                termination = "stuck"

        if (not terminated) and (cargo_health <= 0.0):
            reward -= r.cargo_failure_penalty
            terminated = True
            termination = "cargo"

        if (not terminated) and (step >= e.max_steps):
            truncated = True
            termination = "timeout"

        return NavigationOutcome(reward=reward, terminated=terminated, truncated=truncated, termination=termination)

    @staticmethod
    def clean_green(flags) -> bool:
        return bool(flags["green"] and not flags["road"] and not flags["sidewalk"]
                    and not flags["building"] and not flags["sensitive"])

    def urban(self, reward: float, total_risk: float, flags, use_risk_costs: bool, risk_lambda: float,
              green_bonus: float) -> float:
        """Semantic risk penalty and green-corridor bonus."""
        if use_risk_costs:
            reward -= float(risk_lambda) * total_risk
            if self.clean_green(flags):
                reward += float(green_bonus)
        return reward

    @staticmethod
    def dynamic(reward: float, dynamic_risk: float, dynamic_lambda: float) -> float:
        return reward - float(dynamic_lambda) * dynamic_risk
