#!/usr/bin/env python3
"""Run PPO multi-seed training for RUSCUAV headline learned baselines.

This script is a local-file wrapper around the original TOP_UAV definitions. It
requires the optional reproducibility dependencies and can be slow. The default
profile follows `configs/ppo_multiseed_v0.yaml`: five training seeds, 200 strict
evaluation episodes per trained model, and evaluation seed0=5000.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--world-path', required=True, help='UrbanWorldLayers .npz')
    ap.add_argument('--output-root', required=True, help='local output root replacing the original Colab Drive root')
    ap.add_argument('--methods', nargs='+', default=['M7_geometry_only','M8_reward_only','M8_obs_only','M11C_dynamic_obs_reward_only'])
    ap.add_argument('--train-seeds', nargs='+', type=int, default=[1001,2001,3001,4001,5001])
    args=ap.parse_args()

    from ruscuav_benchmark.legacy import definitions as R

    R.DRIVE_ROOT = Path(args.output_root).resolve()
    R.WORLD_PATH_DRIVE = str(Path(args.world_path).resolve())
    R.MULTISEED_ROOT = R.DRIVE_ROOT / 'experiments' / 'paperA_multiseed_rl_v1'
    R.MULTISEED_TABLE_DIR = R.DRIVE_ROOT / 'paper_ready_v1' / 'tables'
    R.MULTISEED_JSON_DIR = R.DRIVE_ROOT / 'paper_ready_v1' / 'json'
    R.SEED_LEVEL_PARTIAL_PATH = R.MULTISEED_ROOT / 'multiseed_rl_seed_level_partial.csv'
    R.EPISODE_CI_PARTIAL_PATH = R.MULTISEED_ROOT / 'multiseed_rl_episode_bootstrap_secondary_partial.csv'
    for d in [R.MULTISEED_ROOT, R.MULTISEED_TABLE_DIR, R.MULTISEED_JSON_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    world = R.UrbanWorldLayers.load_npz(args.world_path)
    R.world = world
    if 'M11C_dynamic_obs_reward_only' in args.methods:
        R.m11_dyn_cfg_v1 = R.M11DynamicRiskConfig(
            n_hotspots=16, hotspot_mode='road_sidewalk', radius_m_min=60.0, radius_m_max=140.0,
            amplitude_min=15.0, amplitude_max=40.0, start_step_min=0, start_step_max=180,
            duration_min_steps=120, duration_max_steps=350, temporal_mode='hann', altitude_h_scale_m=70.0,
            dynamic_cumulative_risk_budget=10.0, dynamic_risk_lambda=0.20, dynamic_norm_denom=10.0,
        )

    for method in args.methods:
        for seed in args.train_seeds:
            R.ms_run_one_train_eval(method, int(seed))

if __name__ == '__main__':
    main()
