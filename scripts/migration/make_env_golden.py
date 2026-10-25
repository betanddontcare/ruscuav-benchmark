#!/usr/bin/env python3
"""Record reference trajectories of the archived notebook environment for tests.

tests/test_env.py replays the same actions in ruscuav_benchmark.env and compares
with these trajectories, so the refactored environment stays checked against
the notebook code without needing the Zenodo archive at test time.

Usage: python scripts/migration/make_env_golden.py --archive /data/RUSCUAV-v2.0.0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from env_parity import ARCHIVED_CANONICAL_WORLD, VARIANTS, _action  # noqa: E402
from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402

#: (variant, scenario seed, policy, maximum steps, EpisodeConfig changes). The cases
#: cover a success, a stage C operational stop, a stuck termination after reset
#: retries, and the semantic and dynamic observations.
CASES = [
    ("RISK+GREEN-RL_primary", 5003, "seek", 200, {}),
    ("no_shields_diag_C", 5010, "to_sidewalk", 300, {}),
    ("RISK+GREEN-RL_primary", 5005, "hover", 60, {"stuck_patience_steps": 25}),  # 4 reset attempts
    ("OBS-RL_secondary", 5011, "random", 40, {}),
    ("M11C_dynamic_obs_B", 5003, "seek", 40, {}),
]
#: EpisodeConfig fields and the notebook attributes they correspond to.
NOTEBOOK_ATTRIBUTES = {"stuck_patience_steps": "stuck_patience_steps", "max_steps": "max_steps",
                       "shield_stuck_patience_steps": "shield_stuck_patience_steps"}
OUTPUT = REPO / "tests" / "data" / "env_golden_AOI_SYNTH_000.json"


def to_point_action(env, point, v_max=10.0, gain=0.6) -> np.ndarray:
    """Velocity controller towards ``point`` (horizontal), as an action."""
    delta = np.asarray(point, dtype=np.float64) - np.asarray(env.current_pos[:2], dtype=np.float64)
    dist = float(np.linalg.norm(delta))
    v_des = np.zeros(2) if dist < 1e-6 else delta / dist * min(v_max, gain * dist)
    acc = (v_des - np.asarray(env.vel[:2], dtype=np.float64)) / float(env.dt)
    return np.clip(acc / float(env.a_max_xy), -1.0, 1.0).astype(np.float32)


def nearest_cell(mask: np.ndarray, xy) -> np.ndarray:
    ys, xs = np.where(mask > 0.5)
    i = int(np.argmin((xs - xy[0]) ** 2 + (ys - xy[1]) ** 2))
    return np.array([xs[i] + 0.5, ys[i] + 0.5])


def record(ns, world, variant: str, seed: int, policy: str, steps: int, episode: dict) -> dict:
    with contextlib.redirect_stdout(io.StringIO()):
        env = VARIANTS[variant](ns, world, seed)
        for name, value in episode.items():
            setattr(env, NOTEBOOK_ATTRIBUTES[name], value)
        obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out = {
        "variant": variant, "seed": seed, "policy": policy, "episode": episode,
        "start_point": env.start_point.tolist(), "target_point": env.target_point.tolist(),
        "reset_obs": obs.tolist(), "reset_attempts": info["urban_reset_attempts"],
        "actions": [], "positions": [], "rewards": [], "terminations": [], "final_obs": None, "final_info": None,
    }
    goal = nearest_cell(world.sidewalk_mask, env.start_point[:2]) if policy == "to_sidewalk" else None
    for _ in range(steps):
        action = to_point_action(env, goal) if goal is not None else _action(policy, env, rng)
        with contextlib.redirect_stdout(io.StringIO()):
            obs, reward, terminated, truncated, info = env.step(action.copy())
        out["actions"].append(action.tolist())
        out["positions"].append(env.current_pos.tolist())
        out["rewards"].append(float(reward))
        out["terminations"].append(info["termination"])
        if terminated or truncated:
            break
    out["final_obs"] = obs.tolist()
    out["final_info"] = {k: v for k, v in info.items() if isinstance(v, (bool, int, float, str)) or v is None}
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--work", type=Path, default=REPO / "runs" / "env_golden")
    p.add_argument("--output", type=Path, default=OUTPUT)
    args = p.parse_args()
    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_legacy_namespace(args.archive, args.work)
    world = ns.UrbanWorldLayers.load_npz(args.archive / ARCHIVED_CANONICAL_WORLD)
    data = {
        "description": "Trajectories of the archived RUSCUAV v2.0.0 notebook environment on AOI_SYNTH_000 "
                       "(= archived AOI_FAKE_001); written by scripts/migration/make_env_golden.py.",
        "world": "worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz",
        "cases": [record(ns, world, *case) for case in CASES],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.output}: {[len(c['rewards']) for c in data['cases']]} steps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
