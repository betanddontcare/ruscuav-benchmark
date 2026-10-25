#!/usr/bin/env python3
"""Evaluate existing PPO weights under the strict RUSCUAV protocol.

Place model `.zip` files according to `models/WEIGHTS_REQUIRED.csv`, then call
this script with a world path and an output directory. This wrapper intentionally
uses the legacy TOP_UAV evaluation functions so that reproduced CSVs stay aligned
with the draft-paper protocol.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--world-path', required=True)
    ap.add_argument('--release-root', default='.')
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed0', type=int, default=5000)
    args=ap.parse_args()

    from ruscuav_benchmark.legacy import definitions as R
    world = R.UrbanWorldLayers.load_npz(args.world_path)
    release_root=Path(args.release_root)
    out_root=Path(args.output_root); out_root.mkdir(parents=True, exist_ok=True)
    rows=list(csv.DictReader((release_root/'models/WEIGHTS_REQUIRED.csv').open(encoding='utf-8')))
    for row in rows:
        model_path=release_root / row['expected_path']
        if not model_path.exists():
            print('SKIP missing weight:', model_path)
            continue
        method=row['method']
        dirs=R.setup_ruscuav_drive_project(drive_root=str(out_root), experiment_name=f'eval_{method}_seed{row["train_seed"]}')
        prefix=f'{method}_trainseed{row["train_seed"]}_strict_eval_n{args.episodes}_evalseed{args.seed0}'
        if method == 'M7_geometry_only':
            R.evaluate_geo_baseline_m7_drive(str(model_path), world, dirs, n_episodes=args.episodes, seed0=args.seed0, n_obstacles=0, deterministic=True, prefix=prefix, mode='strict_eval', max_steps=1000, save_paths=False)
        elif method in {'M8_reward_only','M8_obs_only'}:
            variant='reward_only' if method=='M8_reward_only' else 'obs_only'
            R.evaluate_m8_ablation_variant_drive(str(model_path), world, variant, dirs, n_episodes=args.episodes, seed0=args.seed0, n_obstacles=0, deterministic=True, prefix=prefix, mode='strict_eval', max_steps=1000, save_paths=False)
        elif method == 'M11C_dynamic_obs_reward_only':
            dyn=R.M11DynamicRiskConfig(n_hotspots=16, hotspot_mode='road_sidewalk', radius_m_min=60.0, radius_m_max=140.0, amplitude_min=15.0, amplitude_max=40.0, start_step_min=0, start_step_max=180, duration_min_steps=120, duration_max_steps=350, temporal_mode='hann', altitude_h_scale_m=70.0, dynamic_cumulative_risk_budget=10.0, dynamic_risk_lambda=0.20, dynamic_norm_denom=10.0)
            R.evaluate_m11c_dynamic_obs_rl_drive(str(model_path), world, dirs, dyn, n_episodes=args.episodes, seed0=args.seed0, prefix=prefix, stage='C', mode='strict_eval', save_paths=False)
        else:
            print('No automatic evaluator registered for', method)

if __name__ == '__main__':
    main()
