#!/usr/bin/env python3
"""Replay steps of an archived policy and show what the geometric shield did on them.

The trajectory scorer counts a step as moving into a hazard when the segment between
two positions, sampled every metre, has an unsafe sample (docs/TRAJECTORIES.md). On a
step that the environment's geometric shield shortened, the shield sampled the
commanded segment, and the scorer samples the shorter flown one at other points. This
script flies a checkpoint under a protocol up to the given steps and prints, for
each, whether the shield shortened it, the commanded and the flown length, and where
the first unsafe sample of each segment lies (as a fraction of the segment).

    python scripts/migration/shield_steps.py --archive /data/RUSCUAV-v2.0.0 --protocol canonical-primary \\
        --checkpoint GEO-RL/1001 5013:47 5185:166
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.env import RuscuavEnv  # noqa: E402
from ruscuav_benchmark.policies import (ARCHIVED_CHECKPOINTS, archived_checkpoint,  # noqa: E402
                                        checkpoint_env_overrides, load_policy)
from ruscuav_benchmark.policies.rollout import policy_action  # noqa: E402
from ruscuav_benchmark.protocols import load_protocol, protocol_world  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, help="extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--protocol", required=True)
    p.add_argument("--checkpoint", required=True, help="archived checkpoint, e.g. GEO-RL/1001")
    p.add_argument("--method", default=None, help="method preset (default: the checkpoint's)")
    p.add_argument("--world", default=None, help="the protocol's world, when it is not distributed")
    p.add_argument("steps", nargs="+", metavar="SEED:STEP")
    args = p.parse_args()

    protocol = load_protocol(args.protocol)
    world = protocol_world(protocol, args.world)
    method = args.method or ARCHIVED_CHECKPOINTS[args.checkpoint].method
    config = protocol.environment_config(method, dynamic_risk=method.startswith("DYN-"),
                                         **(checkpoint_env_overrides(args.checkpoint) or {}))
    policy = load_policy(archived_checkpoint(args.checkpoint, args.archive))
    missions = {m.seed: m for m in protocol.missions}
    for item in args.steps:
        seed, step = (int(v) for v in item.split(":"))
        env = RuscuavEnv(world, config)
        obs, _ = env.reset(seed=seed, options=missions[seed].episode()[1])
        checks = []
        project = env.shield.project_motion

        def recording(prev_pos, desired_pos, project=project, checks=checks):
            accepted, shielded = project(prev_pos, desired_pos)
            checks.append((np.array(prev_pos, np.float32), np.array(desired_pos, np.float32), accepted.copy(),
                           shielded))
            return accepted, shielded

        env.shield.project_motion = recording
        for _ in range(step):
            obs, _, terminated, truncated, _ = env.step(policy_action(policy, obs, True))
            if terminated or truncated:
                break
        if len(checks) < step:
            print(f"seed {seed}: the episode ended after {len(checks)} steps")
            continue
        a, b, accepted, shielded = checks[step - 1]
        shield = env.shield
        print(f"seed {seed} step {step}: shortened by the shield {shielded}; commanded {np.linalg.norm(b - a):.3f} m, "
              f"first unsafe sample at {shield.first_hit_on_segment(a, b)}; flown {np.linalg.norm(accepted - a):.3f} m, "
              f"first unsafe sample at {shield.first_hit_on_segment(a, accepted)}")
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
