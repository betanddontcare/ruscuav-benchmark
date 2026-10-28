#!/usr/bin/env python3
"""Run static and receding A* planner baselines using the TOP_UAV protocol."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--world-path', required=True)
    ap.add_argument('--output-root', required=True)
    ap.add_argument('--episodes', type=int, default=200)
    ap.add_argument('--seed0', type=int, default=5000)
    args=ap.parse_args()
    from ruscuav_benchmark.legacy import definitions as R
    world=R.UrbanWorldLayers.load_npz(args.world_path)
    dirs = R.setup_ruscuav_drive_project(
        drive_root=str(Path(args.output_root).resolve()), experiment_name='planner_reproduction'
    )
    # M9 static full-map risk-cost A*
    m9cfg = R.M9PlannerConfig(plan_res_m=5.0, planner_agl_m=40.0, risk_lambda=0.35, use_astar=True)
    R.evaluate_m9_static_planner_drive(
        world,
        dirs,
        planner_config=m9cfg,
        risk_profile='balanced',
        n_episodes=args.episodes,
        seed0=args.seed0,
        prefix='M9_static_risk_astar_eval200',
        stage='C',
        mode='strict_eval',
        n_obstacles=0,
        save_paths=False,
    )
    # M10 partial/local receding risk planner
    m10cfg = R.M10RecedingHorizonPlannerConfig(
        plan_res_m=5.0,
        planner_agl_m=40.0,
        risk_lambda=0.35,
        use_astar=True,
        sensing_radius_m=75.0,
        replan_every_steps=5,
        prior_risk_mode='mean',
        hard_blockers_known=True,
        waypoint_radius_m=8.0,
        target_speed_mps=12.0,
        max_steps=1000,
    )
    R.evaluate_m10_receding_planner_drive(
        world,
        dirs,
        cfg=m10cfg,
        risk_profile='balanced',
        n_episodes=args.episodes,
        seed0=args.seed0,
        prefix='M10_receding_risk_astar_eval200',
        stage='C',
        mode='strict_eval',
        n_obstacles=0,
        save_paths=False,
    )
    # M10 full-map upper reference: same code with perfect risk prior.
    m10full = R.M10RecedingHorizonPlannerConfig(
        plan_res_m=5.0,
        planner_agl_m=40.0,
        risk_lambda=0.35,
        use_astar=True,
        sensing_radius_m=1000.0,
        replan_every_steps=5,
        prior_risk_mode='perfect',
        hard_blockers_known=True,
        waypoint_radius_m=8.0,
        target_speed_mps=12.0,
        max_steps=1000,
    )
    R.evaluate_m10_receding_planner_drive(
        world,
        dirs,
        cfg=m10full,
        risk_profile='balanced',
        n_episodes=args.episodes,
        seed0=args.seed0,
        prefix='M10_full_map_receding_risk_astar_eval200',
        stage='C',
        mode='strict_eval',
        n_obstacles=0,
        save_paths=False,
    )


if __name__ == '__main__':
    main()
