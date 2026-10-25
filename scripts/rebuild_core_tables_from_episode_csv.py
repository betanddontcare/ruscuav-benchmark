#!/usr/bin/env python3
"""Rebuild core metric summaries from canonical per-episode CSV files.

This script recomputes the primary rates used in the paper tables from raw
per-episode CSVs. It is intentionally lightweight and does not rerun PPO or
planners. For exact manuscript tables including CIs and paired tests, compare
against the precomputed CSV/JSON artifacts in paper_ready_v1/ and experiments/.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from collections import Counter

NUMERIC_MEANS = [
    'total_reward','steps','dist_xy','cargo_health','urban_mean_risk',
    'urban_cumulative_risk','urban_max_risk','road_time','sidewalk_time',
    'building_time','green_time','canopy_time','sensitive_time',
    'urban_hard_violation_steps','building_collision_steps','canopy_collision_steps',
    'sensitive_low_altitude_violation_steps','dynamic_cumulative_risk','dynamic_risk',
    'dynamic_risk_max','replan_count','total_expanded','total_plan_time_s','plan_time_s'
]

def as_bool(v):
    return str(v).strip().lower() in {'1','true','yes','y'}

def as_float(v):
    try: return float(v)
    except Exception: return None

def summarize(path: Path):
    rows=list(csv.DictReader(path.open(newline='', encoding='utf-8')))
    n=len(rows)
    out={'n_episodes': n}
    if n==0: return out
    if 'is_success' in rows[0]:
        success=sum(as_bool(r.get('is_success')) for r in rows)
        out['success_rate']=success/n
    else:
        success=None
    if 'mission_acceptable' in rows[0]:
        acceptable=sum(as_bool(r.get('mission_acceptable')) for r in rows)
        out['mission_acceptability_rate']=acceptable/n
        if success:
            out['acceptable_given_success_rate']=acceptable/success
    for col in NUMERIC_MEANS:
        vals=[as_float(r.get(col,'')) for r in rows]
        vals=[v for v in vals if v is not None]
        if vals:
            out['mean_'+col]=sum(vals)/len(vals)
    if 'termination' in rows[0]:
        out['termination_counts']=json.dumps(dict(Counter(r.get('termination','') for r in rows)), sort_keys=True)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    ap.add_argument('--manifest', default='results/PER_EPISODE_CSV_REQUIRED.csv')
    ap.add_argument('--out', default='reproducibility/rebuilt_core_summary.csv')
    args=ap.parse_args()
    root=Path(args.root)
    rows=list(csv.DictReader((root/args.manifest).open(newline='', encoding='utf-8')))
    out_rows=[]
    all_fields=['table_or_claim','regime','method','expected_path']
    for row in rows:
        p=root/row['expected_path']
        s=summarize(p)
        rec={k:row.get(k,'') for k in ['table_or_claim','regime','method','expected_path']}
        rec.update(s)
        out_rows.append(rec)
        for k in rec:
            if k not in all_fields: all_fields.append(k)
    out_path=root/args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=all_fields)
        w.writeheader(); w.writerows(out_rows)
    print(out_path)

if __name__ == '__main__':
    main()
