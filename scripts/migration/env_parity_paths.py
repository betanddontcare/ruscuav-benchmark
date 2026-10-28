#!/usr/bin/env python3
"""Lockstep checks of the paths that env_parity.py's flights rarely reach.

env_parity.py flies seek and random policies for up to a few hundred steps. This
script drives the notebook environment and the refactored one, in lockstep and
with identical actions, into the remaining paths, and compares observations,
rewards, flags, info and the full dynamic state after every step:

* every termination: stuck, near_success, timeout, cargo, terrain_collision,
  shield_stuck, and the stage C stops (sidewalk and road budgets, sensitive
  zones, building and canopy collisions); step limits are shortened identically
  in both environments where the default would take too long;
* flight to the workspace boundary, many unseeded resets with reset retries,
  the dynamic observation over unseeded episodes, the altitude reference
  without buildings and canopy;
* function by function on 1,200 random states near buildings, canopy and
  sensitive zones: action shield, geometric shield, clearance proxy and gradient,
  altitude reference, endpoint check, semantic observation, LiDAR, and the ground
  lookup on edge inputs.

Example::

    python scripts/migration/env_parity_paths.py --archive /data/RUSCUAV-v2.0.0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_parity import ARCHIVED_CANONICAL_WORLD, _same  # noqa: E402
from ruscuav_benchmark.env import (  # noqa: E402
    CargoConfig,
    DynamicsConfig,
    EpisodeConfig,
    LidarConfig,
    RuscuavEnv,
    make_config,
)
from ruscuav_benchmark.env import legacy_factories as N  # noqa: E402
from ruscuav_benchmark.env.scenario import endpoint_report  # noqa: E402
from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402

STATE = ("current_pos", "vel", "cargo_health", "current_step", "start_point", "target_point", "acc_xy_f",
         "prev_acc_xy", "acc_z_f", "prev_acc_z", "cached_lidar", "best_dist_xy", "steps_since_improve",
         "shield_no_progress_steps")


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def check(ref_env, cand_env, ref_out, cand_out, label):
    diff = _same(ref_out, cand_out, label)
    for field in STATE:
        if diff:
            break
        diff = _same(getattr(ref_env, field, None), getattr(cand_env, field, None), f"{label}.state.{field}")
    return diff


def run(ref_env, cand_env, policy, reset_kwargs, n, label):
    """Reset both, step with actions computed from the notebook environment's state."""
    rng = np.random.default_rng(123)
    with quiet():
        r1 = ref_env.reset(**reset_kwargs)
        r2 = cand_env.reset(**reset_kwargs)
    diff = check(ref_env, cand_env, r1, r2, f"{label}.reset")
    if diff:
        return diff, 0, None
    for i in range(1, n + 1):
        action = policy(ref_env, rng)
        with quiet():
            s1 = ref_env.step(action.copy())
            s2 = cand_env.step(action.copy())
        diff = check(ref_env, cand_env, s1, s2, f"{label}.step{i}")
        if diff:
            return diff, i, s1[4].get("termination")
        if s1[2] or s1[3]:
            return None, i, s1[4].get("termination")
    return None, n, "max_steps"


def to_point(point, v_max=12.0, gain=0.5):
    def policy(env, rng):
        delta = np.asarray(point, dtype=np.float64) - np.asarray(env.current_pos[:2], dtype=np.float64)
        dist = float(np.linalg.norm(delta))
        v_des = np.zeros(2) if dist < 1e-6 else delta / dist * min(v_max, gain * dist)
        acc = (v_des - np.asarray(env.vel[:2], dtype=np.float64)) / float(env.dt)
        return np.clip(acc / float(env.a_max_xy), -1.0, 1.0).astype(np.float32)
    return policy


def hover(env, rng):
    return np.zeros(2, dtype=np.float32)


def uniform(env, rng):
    return rng.uniform(-1.0, 1.0, 2).astype(np.float32)


def away(env, rng):
    d = np.asarray(env.current_pos[:2], np.float64) - np.asarray(env.target_point[:2], np.float64)
    return np.clip(d / (np.linalg.norm(d) + 1e-9) + rng.normal(0, 0.2, 2), -1, 1).astype(np.float32)


def nearest_cell(mask, xy, min_value=None, values=None):
    selected = mask > 0.5 if values is None else (mask > 0.5) & (values > min_value)
    ys, xs = np.where(selected)
    i = int(np.argmin((xs - xy[0]) ** 2 + (ys - xy[1]) ** 2))
    return np.array([xs[i] + 0.5, ys[i] + 0.5])


class Checks:
    def __init__(self, ns, ref_world, cand_world):
        self.L, self.LW, self.NW = ns, ref_world, cand_world
        self.failures = 0

    def report(self, name, result, t0):
        status = "OK  " if result[0] is None else "DIFF"
        self.failures += result[0] is not None
        print(f"{status} {name:58s} steps={result[1]:4d} end={str(result[2]):40s} {time.time() - t0:6.1f}s "
              f"{result[0] or ''}", flush=True)

    def m8(self, variant, stage, mode):
        with quiet():
            ref = self.L.make_ruscuav_m8_ablation_env_m4_mode(self.LW, variant, stage=stage, mode=mode)
            cand = N.make_ruscuav_m8_ablation_env_m4_mode(self.NW, variant, stage=stage, mode=mode)
        return ref, cand

    def rebuild(self, cand, **changes):
        return RuscuavEnv(self.NW, cand.config.with_changes(**changes))

    def terminations(self):
        for stage, seed in (("B", 5001), ("C", 5003)):
            t0 = time.time()
            ref, cand = self.m8("no_urban_lidar", stage, "fast_train")
            self.report(f"hover until stuck, stage {stage}, seed {seed}",
                        run(ref, cand, hover, {"seed": seed}, 310, "stuck"), t0)
        for seed in (5000, 5002):
            t0 = time.time()
            ref, cand = self.m8("no_urban_lidar", "B", "fast_train")
            with quiet():
                ref.reset(seed=seed)
            start, target = ref.start_point[:2].astype(np.float64), ref.target_point[:2].astype(np.float64)
            hold = target - 38.0 * (target - start) / np.linalg.norm(target - start)
            self.report(f"hold 38 m short until near_success, seed {seed}",
                        run(ref, cand, to_point(hold), {"seed": seed}, 700, "near"), t0)
        t0 = time.time()
        ref, cand = self.m8("no_urban_lidar", "C", "strict_eval")
        ref.max_steps = 25
        cand = self.rebuild(cand, episode=EpisodeConfig(max_steps=25))
        self.report("timeout (step limit 25), stage C", run(ref, cand, uniform, {"seed": 5004}, 40, "timeout"), t0)
        t0 = time.time()
        ref, cand = self.m8("no_urban_lidar", "B", "fast_train")
        ref.health_decay_k = 30.0
        cand = self.rebuild(cand, cargo=CargoConfig(health_decay_k=30.0))
        self.report("cargo failure (health decay 30)", run(ref, cand, uniform, {"seed": 5005}, 200, "cargo"), t0)
        t0 = time.time()
        ref, cand = self.m8("no_urban_lidar", "C", "fast_train")
        ref.safety_margin_z = 31.0
        cand = self.rebuild(cand, episode=EpisodeConfig(terrain_margin_m=31.0))
        self.report("terrain collision (margin 31 m)", run(ref, cand, uniform, {"seed": 5006}, 200, "terrain"), t0)
        for seed in (5000, 5001, 5002, 5007):
            t0 = time.time()
            ref, cand = self.m8("no_action_shield", "B", "fast_train")
            ref.urban_lidar_enabled = False
            ref.shield_stuck_patience_steps = 4
            cand = self.rebuild(cand, lidar=LidarConfig(urban=False),
                                episode=EpisodeConfig(shield_stuck_patience_steps=4))
            with quiet():
                ref.reset(seed=seed)
            target = ref.target_point[:2].astype(np.float64)
            self.report(f"geometric shield only, patience 4, seed {seed}",
                        run(ref, cand, to_point(target, v_max=18.0, gain=10.0), {"seed": seed}, 200, "shield"), t0)

    def stage_c_stops(self):
        for variant, layer, seed in (("reward_only_no_shields_diag", "sidewalk_mask", 5010),
                                     ("reward_only_no_shields_diag", "road_mask", 5011),
                                     ("reward_only_no_shields_diag", "sensitive_mask", 5012),
                                     ("no_geom_shield", "sensitive_mask", 5013)):
            t0 = time.time()
            ref, cand = self.m8(variant, "C", "fast_train")
            ref.urban_lidar_enabled = False
            cand = self.rebuild(cand, lidar=LidarConfig(urban=False))
            with quiet():
                ref.reset(seed=seed)
            goal = nearest_cell(getattr(self.NW, layer), ref.start_point[:2])
            self.report(f"stage C, {variant}, to {layer}, seed {seed}",
                        run(ref, cand, to_point(goal, v_max=10.0, gain=0.6), {"seed": seed}, 400, "stop"), t0)
        for kind, seed in (("building", 5014), ("building", 5015), ("canopy", 5016)):
            t0 = time.time()
            ref, cand = self.m8("reward_only_no_shields_diag", "C", "fast_train")
            ref.urban_lidar_enabled = False
            cand = self.rebuild(cand, lidar=LidarConfig(urban=False))
            with quiet():
                ref.reset(seed=seed)
            mask = self.NW.building_mask if kind == "building" else self.NW.canopy_mask
            heights = self.NW.building_height if kind == "building" else self.NW.canopy_height
            goal = nearest_cell(mask, ref.start_point[:2], 35.0 if kind == "building" else 20.0, heights)
            self.report(f"stage C, no shields, into a tall {kind}, seed {seed}",
                        run(ref, cand, to_point(goal, v_max=18.0, gain=5.0), {"seed": seed}, 400, "collide"), t0)

    def other_paths(self):
        t0 = time.time()
        with quiet():
            ref = self.L.make_ruscuav_env_m4_mode(self.LW, stage="C", mode="fast_train")
            cand = N.make_ruscuav_env_m4_mode(self.NW, stage="C", mode="fast_train")
        self.report("away to the workspace boundary, LiDAR on, stage C",
                    run(ref, cand, away, {"seed": 5008}, 120, "away"), t0)

        t0 = time.time()
        ref, cand = self.m8("no_urban_lidar", "B", "fast_train")
        result = run(ref, cand, uniform, {"seed": 4242}, 3, "unseeded0")
        for k in range(40):
            if result[0]:
                break
            result = run(ref, cand, uniform, {}, 3, f"unseeded{k + 1}")
        self.report("40 unseeded resets (with retries), 3 steps each", result, t0)

        t0 = time.time()
        L = self.L
        cfg = L.RealisticUrbanEnvConfig(risk_profile="balanced", include_urban_vector_obs=False,
                                        use_urban_risk_costs=True, urban_risk_lambda=0.10, green_bonus_lambda=0.03,
                                        terminate_on_urban_constraint=True, hard_sensitive_zones=True)
        ref = L.RealisticUrbanCargoUAVEnvM11CDynamicObs(self.LW, config=cfg, dynamic_config=L.reported_dynamic_config(),
                                                        urban_lidar_enabled=False, urban_lidar_step_m=3.0,
                                                        urban_segment_check_step_m=1.0)
        cand = RuscuavEnv(self.NW, make_config("DYN-OBS-RL", stage="C", mode="strict_eval",
                                               lidar=LidarConfig(urban=False)))
        result = run(ref, cand, uniform, {"seed": 777}, 30, "dyn0")
        for k in range(10):
            if result[0]:
                break
            result = run(ref, cand, uniform, {}, 30, f"dyn{k + 1}")
        self.report("dynamic observation, 10 unseeded episodes of 30 steps", result, t0)

        t0 = time.time()
        cfg = L.RealisticUrbanEnvConfig(risk_profile="balanced", include_urban_vector_obs=True, urban_risk_lambda=0.10,
                                        use_urban_altitude_reference=False, hard_sensitive_zones=True)
        ref = L.RealisticUrbanCargoUAVEnvM4(self.LW, config=cfg, urban_lidar_enabled=False, urban_lidar_step_m=3.0,
                                            urban_segment_check_step_m=1.0)
        cand = RuscuavEnv(self.NW, make_config("RISK+OBS-RL", stage="B", mode="strict_eval",
                                               lidar=LidarConfig(urban=False),
                                               dynamics=DynamicsConfig(urban_altitude_reference=False)))
        with quiet():
            ref.reset(seed=5009)
        target = ref.target_point[:2].astype(np.float64)
        self.report("altitude reference without buildings and canopy",
                    run(ref, cand, to_point(target, v_max=18.0, gain=10.0), {"seed": 5009}, 200, "altref"), t0)

    def functions(self, n_states=1200):
        t0 = time.time()
        with quiet():
            ref = self.L.make_ruscuav_env_m4_mode(self.LW, stage="B", mode="strict_eval")
            cand = N.make_ruscuav_env_m4_mode(self.NW, stage="B", mode="strict_eval")
            ref.reset(seed=5000)
            cand.reset(seed=5000)
        assert _same(ref.target_point, cand.target_point, "target") is None
        rng = np.random.default_rng(0)
        cells = []
        for mask in (self.NW.building_mask, self.NW.canopy_mask, self.NW.sensitive_mask):
            ys, xs = np.where(mask > 0.5)
            idx = rng.integers(0, len(xs), n_states // 3)
            cells += list(zip(xs[idx], ys[idx]))
        rng.shuffle(cells)

        def random_state(cx, cy):
            x = np.clip(cx + rng.uniform(-15, 15), 0, 1000)
            y = np.clip(cy + rng.uniform(-15, 15), 0, 1000)
            z = min(150.0, cand.dynamics.ground_at(x, y) + rng.uniform(1.0, 90.0))
            return np.array([x, y, z], dtype=np.float32)

        outcomes, first = Counter(), None
        for k, (cx, cy) in enumerate(cells):
            prev = random_state(cx, cy)
            vel = np.concatenate([rng.uniform(-18, 18, 2), rng.uniform(-6, 6, 1)]).astype(np.float32)
            acc = rng.uniform(-6, 6, 2).astype(np.float32)
            ref.last_urban_action_shielded, ref.last_urban_shield_reason = False, None
            a1 = ref._apply_action_shield_xy(prev.copy(), vel.copy(), acc.copy())
            ref_reason = ref.last_urban_shield_reason  # before the geometric shield overwrites it
            a2, reason = cand.shield.filter_action(prev.copy(), vel.copy(), acc.copy(), cand.dist_xy)
            outcomes[reason] += 1
            desired = np.clip(prev + rng.uniform(-10, 10, 3).astype(np.float32), ref.bounds_low,
                              ref.bounds_high).astype(np.float32)
            p1, s1 = ref._apply_safety_shield(prev.copy(), desired.copy())
            p2, s2 = cand.shield.project_motion(prev.copy(), desired.copy())
            ref.current_pos = prev.copy()
            cand.current_pos = prev.copy()
            diff = (_same(a1, a2, f"state{k}.action_shield") or _same(ref_reason, reason, f"state{k}.reason")
                    or _same(p1, p2, f"state{k}.geometric_shield") or _same(bool(s1), bool(s2), f"state{k}.shielded")
                    or _same(ref._urban_signed_clearance_proxy_m(prev), cand.hazards.signed_clearance_m(prev),
                             f"state{k}.clearance")
                    or _same(ref._urban_clearance_gradient_xy(prev), cand.hazards.clearance_gradient_xy(prev),
                             f"state{k}.gradient")
                    or _same(ref._z_reference(), cand.dynamics.z_reference(prev.copy()), f"state{k}.z_reference")
                    or _same(ref._endpoint_report(prev), endpoint_report(cand.risk_model, prev, 1.0),
                             f"state{k}.endpoint")
                    or _same(ref._get_urban_vector_obs(),
                             cand.semantic.observation(prev, cand.target_point, cand.exposure, cand.config.mission),
                             f"state{k}.semantic_observation"))
            if diff is None and k < 25:
                r1 = ref._compute_lidar_rays()
                r2, m2 = cand.lidar.scan(prev.copy(), cand.target_point)
                diff = (_same(r1, r2, f"state{k}.lidar")
                        or _same(ref.last_urban_lidar_min_norm, m2, f"state{k}.lidar_min"))
            if diff and first is None:
                first = diff
        for x in (-1e-7, -0.5, 0, 0.0, 0.4999, 0.9999999, 1.0, 498.99999, 999, 999.0, 999.4, 999.99994, 1000, 1000.0,
                  1000.5, 1e6):
            for y in (0.3, 999.7, 500.5):
                for conv in (float, np.float32, np.float64):
                    first = first or _same(ref._ground_at(conv(x), conv(y)), cand.dynamics.ground_at(conv(x), conv(y)),
                                           f"ground({x}, {y}, {conv.__name__})")
        self.report(f"functions on {len(cells)} states near hazards (action shield: "
                    f"{dict((str(k), v) for k, v in outcomes.items())})", (first, len(cells), None), t0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--work", type=Path, default=REPO / "runs" / "env_parity_paths")
    p.add_argument("--groups", nargs="+", default=["terminations", "stage_c_stops", "other_paths", "functions"])
    args = p.parse_args()
    with quiet():
        ns = load_legacy_namespace(args.archive, args.work)
    world_path = Path(args.archive) / ARCHIVED_CANONICAL_WORLD
    checks = Checks(ns, ns.UrbanWorldLayers.load_npz(world_path), N.UrbanWorldLayers.load_npz(world_path))
    for group in args.groups:
        getattr(checks, group)()
    print(f"{checks.failures} difference(s)")
    return 1 if checks.failures else 0


if __name__ == "__main__":
    sys.exit(main())
