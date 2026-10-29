#!/usr/bin/env python3
"""A planner built outside RUSCUAV, scored by the benchmark from its trajectories.

The planner sees only the physical constraints at its cruise altitude: buildings and
canopy too tall to overfly with the benchmark's clearances, and sensitive zones,
treated as no-fly areas. It does not see roads or pedestrian ways. It finds the
shortest 8-connected path on a 4 m grid with SciPy, shortens it by line of sight,
rounds its corners, gives it a speed profile (12 m/s cruise, as the package's planners) and
writes the position every 0.5 s, 30 m above the terrain, in the trajectory CSV that `ruscuav evaluate
--trajectories` scores.

    python examples/softwarex/geometric_planner.py --protocol canonical-primary --output geo_paths.csv
"""

from __future__ import annotations

import argparse
import csv
import math

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

from ruscuav_benchmark.protocols import load_protocol, protocol_world

CRUISE_AGL_M = 30.0   # flight altitude above the terrain
BUILDING_CLEARANCE_M = 5.0
CANOPY_CLEARANCE_M = 3.0
GRID = 4              # world cells per planning cell
MARGIN_M = 6.0        # distance kept from blocked cells when planning
V_MAX = 12.0          # m/s, the cruise speed of the package's planners
A_LONG = 1.5          # m/s^2, along the path
A_LAT = 2.0           # m/s^2, in turns
DT = 0.5              # s, the protocol's step


def disk(radius_cells: float) -> np.ndarray:
    r = int(math.ceil(radius_cells))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= radius_cells * radius_cells


class Planner:
    def __init__(self, world):
        self.res = float(world.resolution_m)
        # the grid frame of the environment: metres from the corner of the first cell (the bbox is metadata)
        self.x0, self.y0 = 0.0, 0.0
        self.terrain = np.asarray(world.terrain_z, dtype=float)
        blocked = ((world.building_height > CRUISE_AGL_M - BUILDING_CLEARANCE_M - 1.0)
                   | (world.canopy_height > CRUISE_AGL_M - CANOPY_CLEARANCE_M - 1.0)
                   | (world.sensitive_mask > 0.5))
        self.los_mask = ndimage.binary_dilation(blocked, disk(3.0 / self.res))
        self.smooth_mask = ndimage.binary_dilation(blocked, disk(2.0 / self.res))
        plan_mask = ndimage.binary_dilation(blocked, disk(MARGIN_M / self.res))
        h, w = plan_mask.shape
        hc, wc = -(-h // GRID), -(-w // GRID)
        padded = np.ones((hc * GRID, wc * GRID), dtype=bool)
        padded[:h, :w] = plan_mask
        self.free = ~padded.reshape(hc, GRID, wc, GRID).any(axis=(1, 3))
        self.graph = self._graph()
        # nearest free planning cell, for starts and targets close to obstacles
        _, self.nearest = ndimage.distance_transform_edt(~self.free, return_indices=True)

    def _graph(self):
        hc, wc = self.free.shape
        idx = np.arange(hc * wc).reshape(hc, wc)
        rows, cols, weights = [], [], []
        step = GRID * self.res
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            ys = slice(max(0, -dy), hc - max(0, dy))
            xs = slice(max(0, -dx), wc - max(0, dx))
            yd = slice(max(0, dy), hc - max(0, -dy))
            xd = slice(max(0, dx), wc - max(0, -dx))
            ok = self.free[ys, xs] & self.free[yd, xd]
            if dy and dx:  # no corner cutting
                ok &= self.free[ys, xd] & self.free[yd, xs]
            rows.append(idx[ys, xs][ok])
            cols.append(idx[yd, xd][ok])
            weights.append(np.full(int(ok.sum()), step * math.hypot(dy, dx)))
        n = hc * wc
        return sparse.csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))),
                                 shape=(n, n))

    def _cell(self, x, y):
        r = int((y - self.y0) / self.res) // GRID
        c = int((x - self.x0) / self.res) // GRID
        r = min(max(r, 0), self.free.shape[0] - 1)
        c = min(max(c, 0), self.free.shape[1] - 1)
        return int(self.nearest[0, r, c]), int(self.nearest[1, r, c])

    def _blocked(self, mask, p):
        r = int((p[1] - self.y0) / self.res)
        c = int((p[0] - self.x0) / self.res)
        if not (0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]):
            return True
        return bool(mask[r, c])

    def _visible(self, a, b):
        n = max(2, int(np.hypot(*(b - a)) / 0.5) + 1)
        return not any(self._blocked(self.los_mask, a + (b - a) * t) for t in np.linspace(0.0, 1.0, n)[1:-1])

    def path(self, start_xy, target_xy):
        hc, wc = self.free.shape
        s, t = self._cell(*start_xy), self._cell(*target_xy)
        _, pred = csgraph.dijkstra(self.graph, indices=s[0] * wc + s[1], return_predecessors=True)
        node, cells = t[0] * wc + t[1], []
        while node >= 0:
            cells.append(divmod(int(node), wc))
            node = pred[node]
        if cells[-1] != s:
            raise RuntimeError("no path")
        centres = [np.array([self.x0 + (c + 0.5) * GRID * self.res, self.y0 + (r + 0.5) * GRID * self.res])
                   for r, c in reversed(cells)]
        points = [np.asarray(start_xy, float)] + centres[1:-1] + [np.asarray(target_xy, float)]
        # line-of-sight shortcuts
        out, i = [points[0]], 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self._visible(points[i], points[j]):
                j -= 1
            out.append(points[j])
            i = j
        return np.array(out)

    def smooth(self, poly):
        """The polyline resampled every metre, its corners rounded where the rounding stays clear."""
        seg = np.diff(poly, axis=0)
        s = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])
        q = np.arange(0.0, s[-1], 1.0)
        dense = np.column_stack([np.interp(q, s, poly[:, 0]), np.interp(q, s, poly[:, 1])])
        dense = np.vstack([dense, poly[-1]])
        for half in (12, 8, 5, 3, 0):
            if half == 0:
                return dense
            k = np.ones(2 * half + 1) / (2 * half + 1)
            padded = np.pad(dense, ((half, half), (0, 0)), mode="edge")
            cand = np.column_stack([np.convolve(padded[:, 0], k, "valid"), np.convolve(padded[:, 1], k, "valid")])
            cand[0], cand[-1] = dense[0], dense[-1]
            if not any(self._blocked(self.smooth_mask, p) for p in cand[3:-3]):
                return cand
        return dense

    def trajectory(self, start, target):
        xy = self.smooth(self.path(start[:2], target[:2]))
        d = np.hypot(*np.diff(xy, axis=0).T)
        keep = np.concatenate([[True], d > 1e-6])
        xy = xy[keep]
        d = np.hypot(*np.diff(xy, axis=0).T)
        heading = np.unwrap(np.arctan2(np.diff(xy[:, 1]), np.diff(xy[:, 0])))
        turn = np.abs(np.diff(heading)) / np.maximum(d[1:], 1e-6)          # curvature at inner points
        v_turn = np.full(len(xy), V_MAX)
        v_turn[1:-1] = np.minimum(V_MAX, np.sqrt(A_LAT / np.maximum(turn, 1e-9)))
        v = v_turn.copy()
        v[0] = 0.0
        for i in range(1, len(xy)):                                         # accelerate from rest
            v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * A_LONG * d[i - 1]))
        for i in range(len(xy) - 2, 0, -1):                                 # brake before turns
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * A_LONG * d[i]))
        t = np.concatenate([[0.0], np.cumsum(d / np.maximum(0.5 * (v[:-1] + v[1:]), 1e-3))])
        times = np.arange(0.0, t[-1] + DT, DT)
        x = np.interp(times, t, xy[:, 0])
        y = np.interp(times, t, xy[:, 1])
        z = np.empty_like(x)
        z[0] = start[2]
        for k in range(1, len(times)):
            r = min(max(int((y[k] - self.y0) / self.res), 0), self.terrain.shape[0] - 1)
            c = min(max(int((x[k] - self.x0) / self.res), 0), self.terrain.shape[1] - 1)
            ref = self.terrain[r, c] + CRUISE_AGL_M
            z[k] = z[k - 1] + float(np.clip(0.3 * (ref - z[k - 1]), -1.5, 1.5))
        x[0], y[0] = start[0], start[1]
        return np.column_stack([x, y, z])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--protocol", required=True, help="protocol name or file")
    ap.add_argument("--world", default=None, help="world NPZ (when the protocol's world is not distributed)")
    ap.add_argument("--output", required=True, help="trajectory CSV to write")
    ap.add_argument("--episodes", type=int, default=None, help="the first N missions")
    args = ap.parse_args()
    protocol = load_protocol(args.protocol)
    planner = Planner(protocol_world(protocol, args.world))
    missions = protocol.missions[:args.episodes] if args.episodes else protocol.missions
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(["seed", "step", "x", "y", "z"])
        no_path = 0
        for m in missions:
            start = np.asarray(m.start, float)
            try:
                traj = planner.trajectory(start, np.asarray(m.target, float))
            except RuntimeError:          # target unreachable at the cruise altitude: the UAV stays at the start
                traj, no_path = np.vstack([start, start]), no_path + 1
            for step, (x, y, z) in enumerate(traj):
                out.writerow([m.seed, step, f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"])
    print(f"{len(missions)} trajectories written to {args.output} ({no_path} without a path)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
