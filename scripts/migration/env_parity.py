#!/usr/bin/env python3
"""Step-by-step parity between the legacy notebook environments and a candidate.

Both environments are created for the same variant and seed, reset, and stepped in
lockstep with the same actions. After the reset and after every step the
observation, reward, termination flags, the full info dict and the core state
(position, velocity, cargo health, step counter, start and target) must be
exactly equal: same dtypes, same bits. The first difference is reported.

The reference is the archived notebook code loaded by
ruscuav_benchmark.legacy.runtime. The candidate is any module exposing the same
factory names (make_ruscuav_env_m4_mode, make_static_eval_env, ...) and an
UrbanWorldLayers class: by default the refactored environment's
ruscuav_benchmark.env.legacy_factories, or a module file such as the verbatim
extraction written by extract_legacy_env.py.

The first episode of each run starts with reset(seed=...); later episodes start
with reset() and so check the unseeded path that training uses.

Example::

    python scripts/migration/env_parity.py --archive /data/RUSCUAV-v2.0.0 --variants all --seeds 5000
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import math
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402

ARCHIVED_CANONICAL_WORLD = "worlds/synthetic/AOI_FAKE_001_urban_world_layers.npz"


def _dyn(ns):
    return ns.reported_dynamic_config()


def _m4(stage, mode, profile="balanced"):
    return lambda ns, w, s: ns.make_ruscuav_env_m4_mode(
        world=w, stage=stage, mode=mode, risk_profile=profile, n_obstacles=0, seed=s)


def _static(method, regime):
    return lambda ns, w, s: ns.make_static_eval_env(method, w, getattr(ns, regime), s)


def _m8(variant, stage, mode):
    return lambda ns, w, s: ns.make_ruscuav_m8_ablation_env_m4_mode(
        world=w, variant_name=variant, stage=stage, mode=mode, risk_profile="balanced", n_obstacles=0, seed=s)


def _m11(stage, mode, reported=True, reward_only_obs=True):
    return lambda ns, w, s: ns.make_ruscuav_env_m11_dynamic_mode(
        world=w, stage=stage, mode=mode, risk_profile="balanced", n_obstacles=0, seed=s,
        dynamic_config=_dyn(ns) if reported else None, reward_only_obs=reward_only_obs)


def _m11c(stage, mode, reported=True):
    return lambda ns, w, s: ns.make_ruscuav_env_m11c_dynamic_obs_mode(
        world=w, stage=stage, mode=mode, risk_profile="balanced", n_obstacles=0, seed=s,
        dynamic_config=_dyn(ns) if reported else None)


def _m12c(kind, stage, mode):
    return lambda ns, w, s: ns.make_m12c_real_vienna_env_mode(world=w, policy_kind=kind, stage=stage, mode=mode, seed=s)


#: Variant name -> factory(namespace, world, seed). Each is an environment of a
#: reported evaluation, a planner rollout, a training stage or an ablation.
VARIANTS: Dict[str, Callable[[Any, Any, int], Any]] = {
    # Environment of the planners (full semantic observation and reward).
    "planner_B_strict": _m4("B", "strict_eval"),
    "planner_C_strict": _m4("C", "strict_eval"),
    "planner_A_fast": _m4("A", "fast_train"),
    "planner_B_safety_first": _m4("B", "strict_eval", "safety_first"),
    "planner_C_efficiency_first": _m4("C", "strict_eval", "efficiency_first"),
    # Static learned-policy evaluations in both regimes.
    "GEO-RL_primary": _static("GEO-RL", "PRIMARY_NO_TERMINAL"),
    "GEO-RL_secondary": _static("GEO-RL", "SECONDARY_OPERATIONAL_STOP"),
    "OBS-RL_primary": _static("OBS-RL", "PRIMARY_NO_TERMINAL"),
    "OBS-RL_secondary": _static("OBS-RL", "SECONDARY_OPERATIONAL_STOP"),
    "RISK+GREEN-RL_primary": _static("RISK+GREEN-RL", "PRIMARY_NO_TERMINAL"),
    "RISK+GREEN-RL_secondary": _static("RISK+GREEN-RL", "SECONDARY_OPERATIONAL_STOP"),
    "RISK-RL_primary": _static("RISK-RL", "PRIMARY_NO_TERMINAL"),
    "GREEN-RL_secondary": _static("GREEN-RL", "SECONDARY_OPERATIONAL_STOP"),
    "RISK+OBS-RL_primary": _static("RISK+OBS-RL", "PRIMARY_NO_TERMINAL"),
    "RISK+OBS-RL_secondary": _static("RISK+OBS-RL", "SECONDARY_OPERATIONAL_STOP"),
    # Training environments.
    "M4_train_A": lambda ns, w, s: ns.make_ruscuav_env_m4(world=w, stage="A", seed=s),
    "M4_no_lidar_C": lambda ns, w, s: ns.make_ruscuav_env_m4(world=w, stage="C", seed=s, urban_lidar_enabled=False),
    "GEO-RL_train": lambda ns, w, s: ns.make_ruscuav_geo_baseline_env_m4_mode(world=w, mode="fast_train", seed=s),
    "GEO-RL_strict": lambda ns, w, s: ns.make_ruscuav_geo_baseline_env_m4_mode(world=w, mode="strict_eval", seed=s),
    "OBS-RL_train_A": _m8("obs_only", "A", "fast_train"),
    "RISK+GREEN-RL_train_A": _m8("reward_only", "A", "fast_train"),
    "RISK-RL_train_B": _m8("risk_only", "B", "fast_train"),
    "GREEN-RL_train_C": _m8("green_only", "C", "fast_train"),
    # Safety-stack ablations and shield diagnostics.
    "no_urban_lidar_B": _m8("no_urban_lidar", "B", "strict_eval"),
    "no_action_shield_B": _m8("no_action_shield", "B", "strict_eval"),
    "no_geom_shield_C": _m8("no_geom_shield", "C", "strict_eval"),
    "no_shields_diag_B": _m8("reward_only_no_shields_diag", "B", "strict_eval"),
    "no_shields_diag_C": _m8("reward_only_no_shields_diag", "C", "strict_eval"),
    # Dynamic risk.
    "M11_dynamic_B": _m11("B", "strict_eval"),
    "M11_dynamic_C": _m11("C", "strict_eval"),
    "M11_dynamic_A_fast": _m11("A", "fast_train"),
    "M11_dynamic_semantic_B": _m11("B", "strict_eval", reward_only_obs=False),
    "M11_dynamic_default_C": _m11("C", "strict_eval", reported=False),
    "M11C_dynamic_obs_B": _m11c("B", "strict_eval"),
    "M11C_dynamic_obs_A_fast": _m11c("A", "fast_train"),
    "M11C_dynamic_obs_default_C": _m11c("C", "strict_eval", reported=False),
    # Real-world transfer environments (run here on the synthetic world).
    "M12C_reward_only_B": _m12c("reward_only", "B", "strict_eval"),
    "M12C_geometry_only_C": _m12c("geometry_only", "C", "strict_eval"),
    "M12C_reward_only_A_fast": _m12c("reward_only", "A", "fast_train"),
}

STATE_FIELDS = ("current_pos", "vel", "cargo_health", "current_step", "start_point", "target_point")
DEFAULT_CANDIDATE = "ruscuav_benchmark.env.legacy_factories"


def load_candidate(spec: str | Path) -> types.ModuleType:
    """Import a module by dotted name, or load a module file."""
    spec = str(spec)
    if not spec.endswith(".py"):
        return importlib.import_module(spec)
    path = Path(spec).resolve()
    name = f"ruscuav_env_candidate_{abs(hash(str(path)))}"
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[name] = module  # dataclasses resolve their module through sys.modules
    module_spec.loader.exec_module(module)
    return module


def _same(a: Any, b: Any, path: str) -> Optional[str]:
    """Exact comparison; returns a description of the first difference or None."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        if a_arr.dtype != b_arr.dtype or a_arr.shape != b_arr.shape:
            return f"{path}: dtype/shape {a_arr.dtype}{a_arr.shape} vs {b_arr.dtype}{b_arr.shape}"
        if a_arr.dtype.kind in "fc":
            differs = ~((a_arr == b_arr) | (np.isnan(a_arr) & np.isnan(b_arr)))
        else:
            differs = np.asarray(a_arr != b_arr)
        if np.any(differs):
            if a_arr.ndim == 0:
                return f"{path}: {a_arr!r} vs {b_arr!r}"
            first = tuple(int(i) for i in np.argwhere(differs)[0])
            return f"{path}{list(first)}: {a_arr[first]!r} vs {b_arr[first]!r}"
        return None
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            return (f"{path}: keys differ, only reference {sorted(set(a) - set(b))}, "
                    f"only candidate {sorted(set(b) - set(a))}")
        for key in a:
            diff = _same(a[key], b[key], f"{path}.{key}")
            if diff:
                return diff
        return None
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            diff = _same(x, y, f"{path}[{i}]")
            if diff:
                return diff
        return None
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} vs {type(b).__name__} ({a!r} vs {b!r})"
    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return None
    if a != b:
        return f"{path}: {a!r} vs {b!r}"
    return None


def _action(policy: str, env, rng: np.random.Generator) -> np.ndarray:
    if policy == "random":
        return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
    if policy == "hover":  # no command: low-speed penalties, then the stuck termination
        return np.zeros(2, dtype=np.float32)
    if policy in ("seek", "away"):  # towards (away from) the target, with noise to exercise the shields
        to_target = (np.asarray(env.target_point[:2], dtype=np.float64)
                     - np.asarray(env.current_pos[:2], dtype=np.float64))
        norm = float(np.linalg.norm(to_target))
        direction = to_target / norm if norm > 1e-9 else np.zeros(2)
        if policy == "away":  # reaches the workspace boundary
            direction = -direction
        return np.clip(direction + rng.normal(0.0, 0.3, size=2), -1.0, 1.0).astype(np.float32)
    raise ValueError(policy)


def compare_variant(ref_ns, candidate, variant: str, seed: int, policy: str, max_steps: int,
                    ref_world, cand_world, episodes: int = 1) -> Dict[str, Any]:
    """Run ``episodes`` episodes in lockstep; the first starts with reset(seed), the rest with reset()."""
    factory = VARIANTS[variant]
    with contextlib.redirect_stdout(io.StringIO()):
        ref_env = factory(ref_ns, ref_world, seed)
        cand_env = factory(candidate, cand_world, seed)
    result = {"variant": variant, "seed": seed, "policy": policy, "steps": 0, "episodes": 0,
              "terminations": [], "difference": None}

    def check(label: str, ref_vals, cand_vals) -> bool:
        diff = _same(ref_vals, cand_vals, label)
        if diff is None:
            for field in STATE_FIELDS:
                diff = _same(getattr(ref_env, field, None), getattr(cand_env, field, None), f"{label}.state.{field}")
                if diff:
                    break
        if diff:
            result["difference"] = diff
            return False
        return True

    if not check("observation_space", ref_env.observation_space.shape, cand_env.observation_space.shape):
        return result
    rng = np.random.default_rng(seed)
    for episode in range(1, episodes + 1):
        reset_kwargs = {"seed": seed} if episode == 1 else {}
        with contextlib.redirect_stdout(io.StringIO()):
            ref_out = ref_env.reset(**reset_kwargs)
            cand_out = cand_env.reset(**reset_kwargs)
        result["episodes"] = episode
        if not check(f"ep{episode}.reset", ref_out, cand_out):
            return result
        end = "max_steps"
        for step in range(1, max_steps + 1):
            action = _action(policy, ref_env, rng)
            with contextlib.redirect_stdout(io.StringIO()):
                ref_step = ref_env.step(action.copy())
                cand_step = cand_env.step(action.copy())
            result["steps"] += 1
            if not check(f"ep{episode}.step{step}", ref_step, cand_step):
                return result
            if ref_step[2] or ref_step[3]:
                end = ref_step[4].get("termination")
                break
        result["terminations"].append(end)
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--candidate", default=DEFAULT_CANDIDATE,
                   help="dotted module name or module file exposing the legacy factory names "
                        f"(default {DEFAULT_CANDIDATE})")
    p.add_argument("--work", type=Path, default=REPO / "runs" / "env_parity")
    p.add_argument("--variants", nargs="+", default=["all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[5000])
    p.add_argument("--policies", nargs="+", default=["seek", "random"])
    p.add_argument("--max-steps", type=int, default=200, help="steps per episode at most")
    p.add_argument("--episodes", type=int, default=2, help="episodes per run (the first seeded, the rest not)")
    args = p.parse_args()
    unknown = sorted(set(args.variants) - set(VARIANTS) - {"all"})
    if unknown:
        p.error(f"unknown variants {unknown}; available: {list(VARIANTS)}")

    variants = list(VARIANTS) if args.variants == ["all"] else args.variants
    with contextlib.redirect_stdout(io.StringIO()):
        ref_ns = load_legacy_namespace(args.archive, args.work)
    candidate = load_candidate(args.candidate)
    world_path = Path(args.archive) / ARCHIVED_CANONICAL_WORLD
    ref_world = ref_ns.UrbanWorldLayers.load_npz(world_path)
    cand_world = candidate.UrbanWorldLayers.load_npz(world_path)

    failures = 0
    for variant in variants:
        for seed in args.seeds:
            for policy in args.policies:
                t0 = time.time()
                r = compare_variant(ref_ns, candidate, variant, seed, policy, args.max_steps, ref_world, cand_world,
                                    episodes=args.episodes)
                status = "OK  " if r["difference"] is None else "DIFF"
                failures += r["difference"] is not None
                ends = ",".join(str(t) for t in r["terminations"])
                print(f"{status} {variant:28s} seed={seed} {policy:6s} episodes={r['episodes']} steps={r['steps']:4d} "
                      f"{ends:32s} {time.time() - t0:6.1f}s {r['difference'] or ''}", flush=True)
    print(f"{failures} difference(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
