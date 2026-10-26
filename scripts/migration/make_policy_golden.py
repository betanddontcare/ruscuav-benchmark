#!/usr/bin/env python3
"""Record golden policy-evaluation episodes from the notebook code for tests/test_policy_episodes.py.

The notebook's rollout functions (``rollout_ppo_episode`` with the post-hoc step,
``rollout_m11_dynamic_rl_episode``, ``rollout_m11c_dynamic_obs_rl_episode`` and
``rollout_m12c_real_policy_episode``) fly the simple policies of
``ruscuav_benchmark.policies.simple`` in the notebook environment, so the test needs
neither PyTorch nor the archive. For every case the file holds the episode row as
the archived evaluation wrote it (posthoc format: with the derived columns), the
number of policy calls and SHA-256 digests of the observations, the actions and the
flown path.

    python scripts/migration/make_policy_golden.py --archive /data/RUSCUAV-v2.0.0
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402
from ruscuav_benchmark.policies.simple import ConstantPolicy, TargetSeekingPolicy  # noqa: E402

OUT = REPO / "tests" / "data" / "policy_golden_AOI_SYNTH_000.json"

POLICIES = {
    "seek": lambda: TargetSeekingPolicy(),
    "seek_undamped": lambda: TargetSeekingPolicy(damping=0.0),
    "constant": lambda: ConstantPolicy([1.0, 0.2]),
}

#: (name, row format, method preset, regime, policy, seed, max_steps)
CASES = [
    ("posthoc_success", "posthoc", "RISK+GREEN-RL", "primary", "seek", 5000, 1000),
    ("posthoc_shield_stuck", "posthoc", "RISK+GREEN-RL", "primary", "seek", 5011, 1000),
    ("posthoc_secondary", "posthoc", "GEO-RL", "secondary", "seek_undamped", 5003, 1000),
    ("posthoc_stuck", "posthoc", "RISK+OBS-RL", "primary", "constant", 5003, 1000),
    ("posthoc_no_steps", "posthoc", "OBS-RL", "primary", "seek", 5003, 0),
    ("dynamic_obs", "dynamic", "DYN-OBS-RL", "primary", "seek", 5003, 1000),
    ("dynamic_reward", "dynamic", "DYN-RISK-RL", "primary", "seek", 5000, 1000),
    ("dynamic_secondary", "dynamic", "DYN-RISK-RL", "secondary", "seek", 5011, 1000),
    ("dynamic_no_steps", "dynamic", "DYN-OBS-RL", "primary", "seek", 5011, 0),
    ("transfer_reward_only", "transfer", "RISK+GREEN-RL", "primary", "seek", 5003, 1000),
    ("transfer_geometry_only", "transfer", "GEO-RL", "primary", "seek_undamped", 5000, 1000),
    ("transfer_truncated", "transfer", "RISK+GREEN-RL", "primary", "seek", 5011, 5),
]

LABELS = {("dynamic", "DYN-OBS-RL"): "RL_dynamic_obs_reward_only", ("dynamic", "DYN-RISK-RL"): "RL_reward_only_dynamic_eval",
          ("transfer", "GEO-RL"): "geometry_only", ("transfer", "RISK+GREEN-RL"): "reward_only"}


class Recording:
    def __init__(self, policy) -> None:
        self.policy = policy
        self.obs = hashlib.sha256()
        self.actions = hashlib.sha256()
        self.calls = 0

    def predict(self, obs, deterministic=True):
        action, state = self.policy.predict(obs, deterministic=deterministic)
        self.obs.update(np.ascontiguousarray(obs).tobytes())
        self.actions.update(np.ascontiguousarray(action).tobytes())
        self.calls += 1
        return action, state


def _plain(value):
    value = value.item() if hasattr(value, "item") else value
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def record(ns, world, name, row_format, method, regime_name, policy_name, seed, max_steps):
    regime = ns.PRIMARY_NO_TERMINAL if regime_name == "primary" else ns.SECONDARY_OPERATIONAL_STOP
    policy = Recording(POLICIES[policy_name]())
    if row_format == "posthoc":
        env = ns.make_static_eval_env(method, world, regime, seed)
        row, path = ns.rollout_ppo_episode(model=policy, env=env, seed=seed, method=method, regime=regime,
                                           deterministic=True, max_steps=max_steps)
        row["episode"] = 0
        frame = ns.add_derived_metrics(pd.DataFrame([row]))
        row = {k: _plain(frame.loc[0, k]) for k in frame.columns}
    elif row_format == "dynamic":
        dyn = ns.reported_dynamic_config()
        if method == "DYN-OBS-RL":
            row, path = ns.rollout_m11c_dynamic_obs_rl_episode(model=policy, world=world, seed=seed, dynamic_config=dyn,
                                                              stage=regime.stage, mode=regime.mode, max_steps=max_steps)
        else:
            row, path = ns.rollout_m11_dynamic_rl_episode(model=policy, world=world, seed=seed, dynamic_config=dyn,
                                                          stage=regime.stage, mode=regime.mode, max_steps=max_steps,
                                                          reward_only_obs=True)
        row["episode"] = 0
        row = {k: _plain(v) for k, v in row.items()}
    else:
        kind = LABELS[("transfer", method)]
        row, path = ns.rollout_m12c_real_policy_episode(model=policy, world=world, policy_kind=kind, seed=seed,
                                                        stage=regime.stage, mode=regime.mode, max_steps=max_steps)
        row["episode"] = 0
        row = {k: _plain(v) for k, v in row.items()}
    path = np.asarray(path, dtype=np.float32)
    return {
        "name": name, "row_format": row_format, "method": method, "regime": regime_name, "policy": policy_name,
        "seed": seed, "max_steps": max_steps, "label": LABELS.get((row_format, method), method),
        "row": row, "policy_calls": policy.calls, "obs_sha256": policy.obs.hexdigest(),
        "actions_sha256": policy.actions.hexdigest(), "path_points": int(len(path)),
        "path_sha256": hashlib.sha256(np.ascontiguousarray(path).tobytes()).hexdigest(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path)
    p.add_argument("--work", type=Path, default=Path("runs/policy_golden"))
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        ns = load_legacy_namespace(args.archive.resolve(), args.work.resolve())
    world = ns.UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH)
    cases = []
    for spec in CASES:
        with contextlib.redirect_stdout(io.StringIO()):
            c = record(ns, world, *spec)
        print(f"{c['name']:26s} {c['row']['steps']:4d} steps, {c['row']['termination']}")
        cases.append(c)
    doc = {"world": "AOI_SYNTH_000", "description": __doc__.split("\n\n")[0], "cases": cases}
    args.output.write_text(json.dumps(doc, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
