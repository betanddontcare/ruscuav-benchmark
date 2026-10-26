#!/usr/bin/env python3
"""Step-by-step parity between the notebook's policy evaluations and ruscuav_benchmark.policies.

For every case, an archived PPO checkpoint flies the same mission twice: with the
notebook's rollout function (``rollout_ppo_episode`` in ``make_static_eval_env``,
``rollout_m11_dynamic_rl_episode``, ``rollout_m11c_dynamic_obs_rl_episode`` or
``rollout_m12c_real_policy_episode``) in the notebook environment, and with
``ruscuav_benchmark.policies.run_policy_episode`` in ``ruscuav_benchmark.env``. Both
record every observation the policy receives and every action it returns. A case
passes when these, the flown path, the episode row and (posthoc format) the derived
post-hoc columns are identical, bit for bit.

Cases: the static, dynamic and transfer evaluation formats with checkpoints of every
method, in the primary and the secondary regime; with ``--vienna`` also the four
archived Vienna policy evaluations.

Example::

    python scripts/migration/policy_parity.py --archive /data/RUSCUAV-v2.0.0
    python scripts/migration/policy_parity.py --archive /data/RUSCUAV-v2.0.0 --vienna /data/vienna_footway_only.npz
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from ruscuav_benchmark.env import RuscuavEnv, make_config  # noqa: E402
from ruscuav_benchmark.legacy.runtime import load_legacy_namespace  # noqa: E402
from ruscuav_benchmark.policies import archived_checkpoint, load_policy, run_policy_episode  # noqa: E402
from ruscuav_benchmark.policies.evaluation import default_label  # noqa: E402
from ruscuav_benchmark.posthoc import derived_metrics  # noqa: E402
from ruscuav_benchmark.world import UrbanWorldLayers  # noqa: E402


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class RecordingPolicy:
    """Forwards ``predict`` to a model and records observations and actions."""

    def __init__(self, model) -> None:
        self.model = model
        self.observation_space = model.observation_space
        self.obs: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []

    def predict(self, obs, deterministic=True):
        action, state = self.model.predict(obs, deterministic=deterministic)
        self.obs.append(np.array(obs, copy=True))
        self.actions.append(np.array(action, copy=True))
        return action, state


def _same(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype and \
            a.shape == b.shape and np.array_equal(a, b, equal_nan=True)
    return type(a) is type(b) and a == b


def compare_rows(ref: dict, new: dict, what: str = "row", ordered: bool = True) -> List[str]:
    problems = []
    if (list(ref) if ordered else sorted(ref)) != (list(new) if ordered else sorted(new)):
        problems.append(f"{what} keys differ: {list(ref)} vs {list(new)}")
    for key in ref:
        if key in new and not _same(ref[key], new[key]):
            problems.append(f"{what}[{key!r}]: {ref[key]!r} vs {new[key]!r}")
    return problems


def compare_steps(ref: RecordingPolicy, new: RecordingPolicy) -> List[str]:
    problems = []
    if len(ref.obs) != len(new.obs):
        problems.append(f"{len(ref.obs)} vs {len(new.obs)} policy calls")
    for k, (a, b) in enumerate(zip(ref.obs, new.obs)):
        if not _same(a, b):
            problems.append(f"observation {k} differs (max |d| {float(np.max(np.abs(a - b))):.3g})")
            break
    for k, (a, b) in enumerate(zip(ref.actions, new.actions)):
        if not _same(a, b):
            problems.append(f"action {k} differs")
            break
    return problems


# ---------------------------------------------------------------------- cases
def case(name, checkpoint, method, row_format, seeds, regime="primary", world="canonical", label=None):
    return {"name": name, "checkpoint": checkpoint, "method": method, "row_format": row_format, "seeds": list(seeds),
            "regime": regime, "world": world, "label": label}


def build_cases(with_vienna: bool) -> List[dict]:
    cases = [
        case("static_GEO", "GEO-RL/1001", "GEO-RL", "posthoc", (5000, 5003)),
        case("static_OBS", "OBS-RL/2001", "OBS-RL", "posthoc", (5003,)),
        case("static_RISK+GREEN", "RISK+GREEN-RL/4001", "RISK+GREEN-RL", "posthoc", (5000, 5011)),
        case("static_RISK+OBS", "RISK+OBS-RL/single", "RISK+OBS-RL", "posthoc", (5003,)),
        case("static_GEO_secondary", "GEO-RL/1001", "GEO-RL", "posthoc", (5003,), regime="secondary"),
        case("static_RISK+GREEN_secondary", "RISK+GREEN-RL/4001", "RISK+GREEN-RL", "posthoc", (5011,), regime="secondary"),
        case("dynamic_static_trained", "RISK+GREEN-RL/3001", "DYN-RISK-RL", "dynamic", (5003,)),
        case("dynamic_M11C", "DYN-OBS-RL/5001", "DYN-OBS-RL", "dynamic", (5003, 5011)),
        case("dynamic_M11B", "DYN-RISK-RL/single", "DYN-RISK-RL", "dynamic", (5011,)),
        case("dynamic_M11C_secondary", "DYN-OBS-RL/1001", "DYN-OBS-RL", "dynamic", (5003,), regime="secondary"),
        case("transfer_reward_only", "RISK+GREEN-RL/1001", "RISK+GREEN-RL", "transfer", (5003,)),
        case("transfer_geometry_only", "GEO-RL/1001", "GEO-RL", "transfer", (5003,)),
    ]
    if with_vienna:
        cases += [
            case("vienna_zero_shot", "RISK+GREEN-RL/1001", "RISK+GREEN-RL", "transfer", (7000,), world="vienna"),
            case("vienna_short_ft", "VIENNA/RISK+GREEN-RL/short_ft", "RISK+GREEN-RL", "transfer", (7001,), world="vienna"),
            case("vienna_extended_ft", "VIENNA/RISK+GREEN-RL/extended_ft", "RISK+GREEN-RL", "transfer", (7002,),
                 world="vienna"),
            case("vienna_GEO", "GEO-RL/1001", "GEO-RL", "posthoc", (7003,), world="vienna", label="VIENNA-GEO"),
        ]
    return cases


def run_reference(ns, c, model, world, seed):
    regime = ns.PRIMARY_NO_TERMINAL if c["regime"] == "primary" else ns.SECONDARY_OPERATIONAL_STOP
    label = c["label"] or default_label(c["method"], c["row_format"])
    if c["row_format"] == "posthoc":
        env = ns.make_static_eval_env(c["method"], world, regime, seed)
        row, path = ns.rollout_ppo_episode(model=model, env=env, seed=seed, method=label, regime=regime,
                                           deterministic=True)
        derived = ns.add_derived_metrics(pd.DataFrame([row]))
        extra = {k: _plain(derived.loc[0, k]) for k in derived.columns if k not in row or k == "episode_duration_s"}
        return row, path, extra
    dyn_cfg = ns.reported_dynamic_config()
    if c["row_format"] == "dynamic":
        if c["method"] == "DYN-OBS-RL":
            row, path = ns.rollout_m11c_dynamic_obs_rl_episode(model=model, world=world, seed=seed, dynamic_config=dyn_cfg,
                                                              stage=regime.stage, mode=regime.mode)
        else:
            row, path = ns.rollout_m11_dynamic_rl_episode(model=model, world=world, seed=seed, dynamic_config=dyn_cfg,
                                                          stage=regime.stage, mode=regime.mode, reward_only_obs=True)
        return row, path, None
    kind = "geometry_only" if c["method"] == "GEO-RL" else "reward_only"
    row, path = ns.rollout_m12c_real_policy_episode(model=model, world=world, policy_kind=kind, seed=seed,
                                                    stage=regime.stage, mode=regime.mode)
    return row, path, None


def _plain(value):
    return value.item() if hasattr(value, "item") else value


def run_candidate(c, model, world, seed):
    regime = "primary_no_terminal" if c["regime"] == "primary" else "secondary_operational_stopping"
    from ruscuav_benchmark.env import REGIMES

    env = RuscuavEnv(world, make_config(c["method"], regime=regime))
    label = c["label"] or default_label(c["method"], c["row_format"])
    row, path = run_policy_episode(env, model, seed, row_format=c["row_format"], label=label,
                                   regime=REGIMES[regime])
    env.close()
    extra = None
    if c["row_format"] == "posthoc":
        derived = derived_metrics(row)
        extra = {k: v for k, v in derived.items() if k not in row or k == "episode_duration_s"}
    return row, path, extra


def run_cases(ns, cases, archive: Path, worlds: Dict[str, Any]) -> List[dict]:
    results = []
    for c in cases:
        model = load_policy(archived_checkpoint(c["checkpoint"], archive))
        legacy_world, package_world = worlds[c["world"]]
        for seed in c["seeds"]:
            t0 = time.time()
            ref_policy, new_policy = RecordingPolicy(model), RecordingPolicy(model)
            with quiet():
                ref_row, ref_path, ref_extra = run_reference(ns, c, ref_policy, legacy_world, seed)
            new_row, new_path, new_extra = run_candidate(c, new_policy, package_world, seed)
            problems = compare_rows(ref_row, new_row) + compare_steps(ref_policy, new_policy)
            if not _same(np.asarray(ref_path, dtype=np.float32), new_path):
                problems.append("flown path differs")
            if ref_extra is not None:
                problems += compare_rows(ref_extra, new_extra, "derived", ordered=False)
            result = {"case": c["name"], "checkpoint": c["checkpoint"], "seed": seed, "ok": not problems,
                      "problems": problems[:10], "steps": int(new_row["steps"]), "termination": new_row["termination"],
                      "elapsed_s": round(time.time() - t0, 1)}
            results.append(result)
            print(f"{'ok  ' if result['ok'] else 'DIFF'} {c['name']:30s} seed {seed}: {result['steps']:4d} steps, "
                  f"{result['termination']} ({result['elapsed_s']} s)", flush=True)
            for problem in problems[:5]:
                print("      ", problem)
    return results


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--archive", required=True, type=Path, help="Extracted RUSCUAV-v2.0.0 Zenodo archive")
    p.add_argument("--work", type=Path, default=Path("runs/policy_parity"))
    p.add_argument("--vienna", type=Path, default=None,
                   help="processed Vienna footway-only world, for the archived Vienna evaluations")
    p.add_argument("--report", type=Path, default=None, help="JSON report (default: <work>/policy_parity.json)")
    args = p.parse_args()

    archive = args.archive.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    with quiet():
        ns = load_legacy_namespace(archive, work)
    worlds = {"canonical": (ns.UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH),
                            UrbanWorldLayers.load_npz(ns.CANONICAL_WORLD_PATH))}
    if args.vienna is not None:
        worlds["vienna"] = (ns.UrbanWorldLayers.load_npz(args.vienna), UrbanWorldLayers.load_npz(args.vienna))

    t0 = time.time()
    report: Dict[str, Any] = {"episodes": run_cases(ns, build_cases(args.vienna is not None), archive, worlds)}
    report["elapsed_s"] = round(time.time() - t0, 1)
    report["ok"] = all(r["ok"] for r in report["episodes"])
    path = args.report or work / "policy_parity.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    n_bad = sum(not r["ok"] for r in report["episodes"])
    n = len(report["episodes"])
    steps = sum(r["steps"] for r in report["episodes"])
    print(f"\n{n - n_bad}/{n} episodes identical ({steps} steps); report: {path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
