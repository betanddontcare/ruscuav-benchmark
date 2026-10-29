#!/usr/bin/env python3
"""Listing 1 of the SoftwareX paper, shortened to check that it runs: SAC trained for STEPS steps and evaluated
on the first EPISODES missions of canonical-primary. The result says nothing about SAC (the listing trains for
100,000 steps and evaluates all 200 missions). Needs the "env" and "rl" extras.

    python examples/softwarex/listing_sac.py 2000 3 runs/softwarex/sac
"""

import sys

from stable_baselines3 import SAC

from ruscuav_benchmark.env import make_env
from ruscuav_benchmark.policies import evaluate_policy
from ruscuav_benchmark.protocols import load_protocol, protocol_world

steps, episodes, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
protocol = load_protocol("canonical-primary")
world = protocol_world(protocol)                  # layers checksum checked
env = make_env(world, "RISK+GREEN-RL")            # observation and reward preset
model = SAC("MlpPolicy", env, seed=0).learn(total_timesteps=steps)
result = evaluate_policy(world, model, "RISK+GREEN-RL", protocol=protocol, n_episodes=episodes,
                         label="SAC", output_dir=out)
summary = result.summary
print({key: summary[key] for key in ("n_episodes", "success_rate", "mission_acceptability_rate", "protocol")})
