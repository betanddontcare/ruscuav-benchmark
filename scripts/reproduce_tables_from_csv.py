#!/usr/bin/env python3
"""Recompute generic episode summaries from raw RUSCUAV CSV files.

This script does not fabricate or re-run PPO policies.  It verifies that archived
per-episode CSV files are internally consistent with their matching summary JSON
files and inventories the paper-ready table files used for Tables I-XI.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


METRIC_MAP = {
    'n_episodes': ('__len__', 'len'),
    'success_rate': ('is_success', 'mean_bool'),
    'mission_acceptability_rate': ('mission_acceptable', 'mean_bool'),
    'static_only_acceptability_rate': ('mission_acceptable_static_only', 'mean_bool'),
    'dynamic_budget_ok_rate': ('dynamic_budget_ok', 'mean_bool'),
    'mean_reward': ('total_reward', 'mean'),
    'std_reward': ('total_reward', 'std_population'),
    'mean_steps': ('steps', 'mean'),
    'mean_dist_xy': ('dist_xy', 'mean'),
    'median_dist_xy': ('dist_xy', 'median'),
    'mean_cargo_health': ('cargo_health', 'mean'),
    'mean_urban_risk': ('urban_mean_risk', 'mean'),
    'mean_urban_max_risk': ('urban_max_risk', 'mean'),
    'mean_cumulative_risk': ('urban_cumulative_risk', 'mean'),
    'mean_urban_cumulative_risk': ('urban_cumulative_risk', 'mean'),
    'mean_road_time': ('road_time', 'mean'),
    'mean_sidewalk_time': ('sidewalk_time', 'mean'),
    'mean_building_time': ('building_time', 'mean'),
    'mean_green_time': ('green_time', 'mean'),
    'mean_canopy_time': ('canopy_time', 'mean'),
    'mean_sensitive_time': ('sensitive_time', 'mean'),
    'mean_building_collision_steps': ('building_collision_steps', 'mean'),
    'mean_canopy_collision_steps': ('canopy_collision_steps', 'mean'),
    'mean_sensitive_low_altitude_violation_steps': ('sensitive_low_altitude_violation_steps', 'mean'),
    'mean_hard_violation_steps': ('urban_hard_violation_steps', 'mean'),
    'mean_dynamic_risk': ('dynamic_risk_mean', 'mean'),
    'mean_dynamic_risk_max': ('dynamic_risk_max', 'mean'),
    'mean_dynamic_cumulative_risk': ('dynamic_cumulative_risk', 'mean'),
    'mean_replan_count': ('replan_count', 'mean'),
    'mean_replan_success_count': ('replan_success_count', 'mean'),
    'mean_replan_fail_count': ('replan_fail_count', 'mean'),
    'mean_total_expanded': ('total_expanded', 'mean'),
    'mean_total_plan_time_s': ('total_plan_time_s', 'mean'),
    'mean_final_known_fraction': ('final_known_fraction', 'mean'),
}


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(bool)
    return s.astype(str).str.lower().isin(['true', '1', 'yes', 'y'])


def compute_metric(df: pd.DataFrame, col: str, op: str) -> float | int | None:
    if col == '__len__':
        return int(len(df))
    if col not in df.columns:
        return None
    if op == 'mean_bool':
        return float(to_bool_series(df[col]).mean())
    vals = pd.to_numeric(df[col], errors='coerce')
    if op == 'mean':
        return float(vals.mean())
    if op == 'median':
        return float(vals.median())
    if op == 'std_population':
        return float(vals.std(ddof=0))
    raise ValueError(op)


def recompute_summary(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric, (col, op) in METRIC_MAP.items():
        val = compute_metric(df, col, op)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            out[metric] = val
    if 'is_success' in df.columns and 'mission_acceptable' in df.columns:
        success = to_bool_series(df['is_success'])
        accept = to_bool_series(df['mission_acceptable'])
        out['acceptable_given_success_rate'] = float((accept & success).sum() / max(1, success.sum()))
    return out


def matching_summary_path(csv_path: Path) -> Path | None:
    name = csv_path.name
    if name.endswith('_episodes.csv'):
        cand = csv_path.with_name(name[:-len('_episodes.csv')] + '_summary.json')
        if cand.exists():
            return cand
    # Fallback: same directory with a single summary JSON
    summaries = list(csv_path.parent.glob('*_summary.json'))
    if len(summaries) == 1:
        return summaries[0]
    return None


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='Artifact root')
    ap.add_argument('--out', default='reproducibility/reports/table_reproduction', help='Output directory')
    ap.add_argument('--tolerance', type=float, default=1e-6)
    ap.add_argument('--strict', action='store_true', help='Exit non-zero on summary mismatches')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_csvs = sorted((root / 'experiments').rglob('*_episodes.csv'))
    summary_rows = []
    comparison_rows = []

    for csv_path in episode_csvs:
        rel_csv = csv_path.relative_to(root).as_posix()
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            comparison_rows.append({'csv_path': rel_csv, 'metric': '__read_csv__', 'status': 'ERROR',
                                    'error': repr(exc)})
            continue
        rec = recompute_summary(df)
        rec['csv_path'] = rel_csv
        summary_rows.append(rec)

        summary_path = matching_summary_path(csv_path)
        if summary_path is None:
            comparison_rows.append({'csv_path': rel_csv, 'metric': '__summary_json__',
                                    'status': 'MISSING_MATCHING_SUMMARY'})
            continue
        original = load_json(summary_path)
        rel_sum = summary_path.relative_to(root).as_posix()
        for metric, recomputed in rec.items():
            if metric == 'csv_path' or metric not in original:
                continue
            original_value = original.get(metric)
            try:
                diff = abs(float(recomputed) - float(original_value))
                status = 'OK' if diff <= args.tolerance else 'MISMATCH'
            except Exception:
                diff = None
                status = 'SKIP_NON_NUMERIC'
            comparison_rows.append({
                'csv_path': rel_csv,
                'summary_path': rel_sum,
                'metric': metric,
                'recomputed': recomputed,
                'summary_json': original_value,
                'abs_diff': diff,
                'status': status,
            })

    pd.DataFrame(summary_rows).to_csv(out_dir / 'episode_recomputed_summaries.csv', index=False)
    pd.DataFrame(comparison_rows).to_csv(out_dir / 'summary_comparison.csv', index=False)

    # Inventory paper table files.
    table_map_path = root / 'reproducibility' / 'manifests' / 'TABLE_FILE_MAP.csv'
    table_inventory = []
    if table_map_path.exists():
        with table_map_path.open(newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                for item in row.get('artifact_files', '').split(';'):
                    item = item.strip()
                    if not item:
                        continue
                    matches = sorted(root.glob(item)) if '*' in item else [root / item]
                    for p in matches:
                        table_inventory.append({
                            'paper_table': row.get('paper_table'),
                            'description': row.get('description'),
                            'artifact_file': p.relative_to(root).as_posix() if p.exists() else item,
                            'exists': p.exists(),
                            'rows': (int(pd.read_csv(p).shape[0])
                                     if p.exists() and p.suffix.lower() == '.csv' else None),
                            'columns': (int(pd.read_csv(p).shape[1])
                                        if p.exists() and p.suffix.lower() == '.csv' else None),
                        })
    pd.DataFrame(table_inventory).to_csv(out_dir / 'paper_table_inventory.csv', index=False)

    mismatch_count = sum(1 for r in comparison_rows if r.get('status') == 'MISMATCH')
    error_count = sum(1 for r in comparison_rows if str(r.get('status')).startswith('ERROR'))
    missing_summary_count = sum(1 for r in comparison_rows if r.get('status') == 'MISSING_MATCHING_SUMMARY')
    missing_table_count = sum(1 for r in table_inventory if not r.get('exists'))

    report = {
        'root': str(root),
        'episode_csv_files': len(episode_csvs),
        'recomputed_summary_rows': len(summary_rows),
        'summary_metric_comparisons': len(comparison_rows),
        'metric_mismatches': mismatch_count,
        'read_errors': error_count,
        'episode_csv_without_matching_summary_json': missing_summary_count,
        'paper_table_assets_checked': len(table_inventory),
        'paper_table_assets_missing': missing_table_count,
        'outputs': {
            'episode_recomputed_summaries': str(out_dir / 'episode_recomputed_summaries.csv'),
            'summary_comparison': str(out_dir / 'summary_comparison.csv'),
            'paper_table_inventory': str(out_dir / 'paper_table_inventory.csv'),
        },
        'strict_pass': mismatch_count == 0 and error_count == 0 and missing_table_count == 0,
    }
    (out_dir / 'REPRODUCTION_REPORT.json').write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                                       encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.strict and not report['strict_pass']:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
