"""Random missions and safe endpoints.

A mission is a start and a target cell, drawn uniformly as integers in
[``spawn_margin_m``, extent - ``spawn_margin_m``) and at least ``min_distance_m``
apart, both at cruise altitude above terrain. A mission is feasible when an
8-connected grid path between the endpoints exists with a bounded detour and the
terrain along it can be followed within the vertical speed and acceleration
limits. The environment additionally rejects missions whose endpoints are on
buildings, roads, sidewalks or sensitive zones, violate a hard constraint or
carry more than ``start_target_max_risk`` risk, by reseeding the episode.

The draws, their order and the grid search are those of the archived
environment, so a seed gives the same mission.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..risk import UrbanRiskModel
from .config import ScenarioConfig
from .dynamics import PointMassDynamics


class MissionSampler:
    def __init__(self, config: ScenarioConfig, dynamics: PointMassDynamics, x_max: float, y_max: float,
                 z_max: float) -> None:
        self.config = config
        self.dynamics = dynamics
        self.x_max, self.y_max, self.z_max = float(x_max), float(y_max), float(z_max)
        margin = int(config.spawn_margin_m)
        self.spawn_low = margin
        self.spawn_high_x = int(self.x_max) - margin
        self.spawn_high_y = int(self.y_max) - margin

    # ------------------------------------------------------------------ sampling
    def sample(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """Draw (start_point, target_point), float32 (x, y, z) arrays."""
        c = self.config
        d = self.dynamics.config
        ground_at = self.dynamics.ground_at
        for _ in range(int(c.max_attempts)):
            s_x, s_y = rng.integers(self.spawn_low, self.spawn_high_x), rng.integers(self.spawn_low, self.spawn_high_y)
            t_x, t_y = rng.integers(self.spawn_low, self.spawn_high_x), rng.integers(self.spawn_low, self.spawn_high_y)
            ground_s = ground_at(s_x, s_y)
            ground_t = ground_at(t_x, t_y)
            min_start_z = ground_s + d.min_agl_m
            min_target_z = ground_t + d.min_agl_m
            if min_start_z > self.z_max or min_target_z > self.z_max:
                continue
            start_z = min(self.z_max, ground_s + d.cruise_agl_m)
            target_z = min(self.z_max, ground_t + d.cruise_agl_m)
            start_z = max(start_z, min_start_z)
            target_z = max(target_z, min_target_z)
            p1 = np.array([s_x, s_y, start_z], dtype=np.float32)
            p2 = np.array([t_x, t_y, target_z], dtype=np.float32)
            dist = np.linalg.norm(p1[:2] - p2[:2])
            if dist <= c.min_distance_m:
                continue
            if not self.is_feasible(p1, p2):
                continue
            return p1, p2
        raise RuntimeError(f"Could not sample a feasible mission in {c.max_attempts} attempts")

    # ------------------------------------------------------------------ feasibility
    def is_feasible(self, start_point: np.ndarray, target_point: np.ndarray) -> bool:
        c = self.config
        if not c.feasibility_check:
            return True
        start_xy = start_point[:2].astype(np.float32)
        target_xy = target_point[:2].astype(np.float32)
        straight_dist = float(np.linalg.norm(target_xy - start_xy))
        if straight_dist < 1e-6:
            return False
        path_xy = self.grid_path(start_xy, target_xy)
        if path_xy is None:
            return False
        path_len = path_length_xy(path_xy)
        detour_ratio = path_len / max(straight_dist, 1e-6)
        if detour_ratio > c.max_detour_ratio:
            return False
        return self.terrain_profile_is_feasible(path_xy)

    def _world_to_grid(self, xy):
        res = float(self.config.feasibility_grid_m)
        nx = int(self.x_max / res)
        ny = int(self.y_max / res)
        gx = int(np.clip(xy[0] / res, 0, nx - 1))
        gy = int(np.clip(xy[1] / res, 0, ny - 1))
        return gx, gy

    def _grid_to_world(self, ij):
        res = float(self.config.feasibility_grid_m)
        gx, gy = ij
        x = min(self.x_max - 1e-3, (gx + 0.5) * res)
        y = min(self.y_max - 1e-3, (gy + 0.5) * res)
        return np.array([x, y], dtype=np.float32)

    def grid_path(self, start_xy: np.ndarray, target_xy: np.ndarray) -> Optional[np.ndarray]:
        """A* path on the free ``feasibility_grid_m`` grid (8-connected), endpoints exact."""
        res = float(self.config.feasibility_grid_m)
        nx = int(self.x_max / res)
        ny = int(self.y_max / res)
        start = self._world_to_grid(start_xy)
        goal = self._world_to_grid(target_xy)
        neighbors = [
            (-1, -1), (0, -1), (1, -1),
            (-1, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1),
        ]

        def heuristic(a, b):
            dx = (a[0] - b[0]) * res
            dy = (a[1] - b[1]) * res
            return float(np.hypot(dx, dy))

        open_heap = []
        heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        visited = set()
        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            if current == goal:
                grid_path = [current]
                while current in came_from:
                    current = came_from[current]
                    grid_path.append(current)
                grid_path.reverse()
                world_path = [start_xy.astype(np.float32)]
                for cell in grid_path[1:-1]:
                    world_path.append(self._grid_to_world(cell))
                world_path.append(target_xy.astype(np.float32))
                return np.array(world_path, dtype=np.float32)
            cx, cy = current
            for dx, dy in neighbors:
                nx_i, ny_i = cx + dx, cy + dy
                if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                    continue
                step_cost = float(np.hypot(dx * res, dy * res))
                neighbor = (nx_i, ny_i)
                tentative_g = g_curr + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))
        return None

    def terrain_profile_is_feasible(self, path_xy: Optional[np.ndarray]) -> bool:
        """Can the cruise-altitude profile along the path be flown within the limits?"""
        c = self.config
        d = self.dynamics.config
        if path_xy is None or len(path_xy) < 2:
            return False
        dense_xy = densify_xy_path(path_xy, c.profile_step_m)
        grounds = np.array([self.dynamics.ground_at(p[0], p[1]) for p in dense_xy], dtype=np.float32)
        z_refs = []
        for g in grounds:
            z_ref = self.dynamics.reference_z_from_ground(float(g))
            if z_ref is None:
                return False
            z_refs.append(z_ref)
        z_refs = np.array(z_refs, dtype=np.float32)
        if np.any(grounds + d.min_agl_m > self.z_max):
            return False
        segs = np.diff(dense_xy, axis=0)
        ds = np.linalg.norm(segs, axis=1).astype(np.float32)
        nominal_vxy = float(max(1.0, c.nominal_speed_mps))
        dts = np.maximum(ds / nominal_vxy, 1e-3).astype(np.float32)
        dz = np.diff(z_refs).astype(np.float32)
        req_vz = dz / dts
        max_req_vz = float(c.max_vz_ratio * d.v_max_z)
        if np.any(np.abs(req_vz) > max_req_vz):
            return False
        if len(req_vz) >= 2:
            dt_mid = 0.5 * (dts[:-1] + dts[1:])
            dt_mid = np.maximum(dt_mid, 1e-3).astype(np.float32)
            req_az = np.diff(req_vz) / dt_mid
            max_req_az = float(c.max_az_ratio * d.a_max_z)
            if np.any(np.abs(req_az) > max_req_az):
                return False
        return True


def path_length_xy(path_xy) -> float:
    if path_xy is None or len(path_xy) < 2:
        return 0.0
    segs = np.diff(path_xy, axis=0)
    return float(np.sum(np.linalg.norm(segs, axis=1)))


def densify_xy_path(path_xy: np.ndarray, step_m: float) -> np.ndarray:
    if len(path_xy) <= 1:
        return path_xy.copy()
    dense = [path_xy[0].astype(np.float32)]
    step_m = float(max(1e-3, step_m))
    for i in range(len(path_xy) - 1):
        p0 = path_xy[i].astype(np.float32)
        p1 = path_xy[i + 1].astype(np.float32)
        d = float(np.linalg.norm(p1 - p0))
        if d < 1e-9:
            continue
        n_steps = max(1, int(np.ceil(d / step_m)))
        for k in range(1, n_steps + 1):
            a = k / n_steps
            pt = (1.0 - a) * p0 + a * p1
            dense.append(pt.astype(np.float32))
    return np.array(dense, dtype=np.float32)


def endpoint_report(risk_model: UrbanRiskModel, point, max_risk: float) -> Dict[str, Any]:
    """Is ``point`` an acceptable mission endpoint? Not on a building, road, sidewalk or
    sensitive cell, no hard violation, and risk at most ``max_risk``."""
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    risk = float(risk_model.risk_at(x, y, z))
    flags = risk_model.flags_at(x, y)
    constraints = risk_model.constraints_at(x, y, z)
    disallowed_flags = []
    for key in ["building", "road", "sidewalk", "sensitive"]:
        if flags.get(key, False):
            disallowed_flags.append(key)
    safe = risk <= max_risk and len(disallowed_flags) == 0 and not constraints["any_hard_violation"]
    return {
        "safe": bool(safe),
        "risk": risk,
        "flags": flags,
        "constraints": constraints,
        "disallowed_flags": disallowed_flags,
        "reason": "ok" if safe else f"risk={risk:.2f}, disallowed={disallowed_flags}, constraints={constraints}",
    }
