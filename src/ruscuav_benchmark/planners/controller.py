"""Waypoint controller: turns a planned path into acceleration commands."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import PlannerConfig


def waypoint_action(env, path_xy: Optional[np.ndarray], wp_idx: int,
                    config: PlannerConfig) -> Tuple[np.ndarray, int]:
    """Action that steers ``env``'s UAV along ``path_xy``, and the updated waypoint index.

    Waypoints within ``waypoint_radius_m`` are passed. The desired velocity points at
    the current waypoint with the cruise speed, reduced near the end of the path to
    0.8 m/s per metre of remaining distance (at least 2 m/s); the action is the
    acceleration that reaches it in one step, scaled by ``a_max_xy`` and clipped to
    [-1, 1]. The environment's altitude hold flies the vertical profile.
    """
    pos_xy = env.current_pos[:2].astype(np.float32)
    if path_xy is None or len(path_xy) == 0:
        return np.zeros(env.action_space.shape, dtype=np.float32), wp_idx

    wp_idx = int(np.clip(wp_idx, 0, len(path_xy) - 1))
    while wp_idx < len(path_xy) - 1:
        if float(np.linalg.norm(path_xy[wp_idx] - pos_xy)) <= config.waypoint_radius_m:
            wp_idx += 1
        else:
            break

    delta = path_xy[wp_idx] - pos_xy
    dist = float(np.linalg.norm(delta) + 1e-9)
    if dist < 1e-6:
        desired_vel_xy = np.zeros(2, dtype=np.float32)
    else:
        direction = delta / dist
        final_dist = float(np.linalg.norm(path_xy[-1] - pos_xy))
        speed = min(float(config.target_speed_mps), max(2.0, 0.8 * final_dist))
        desired_vel_xy = direction.astype(np.float32) * speed

    acc_xy = (desired_vel_xy - env.vel[:2].astype(np.float32)) / float(env.dt)
    action = np.clip(acc_xy / float(env.a_max_xy), -1.0, 1.0).astype(np.float32)
    return action, wp_idx
