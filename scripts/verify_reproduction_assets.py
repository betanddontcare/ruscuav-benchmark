#!/usr/bin/env python3
"""Verify integrity and completeness of the RUSCUAV v1.0.0 artifact."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def boolish(v) -> bool:
    return str(v).strip().lower() in {'1', 'true', 'yes', 'y'}


def check_exists(root: Path, paths: Iterable[str]) -> tuple[list[str], list[str]]:
    present, missing = [], []
    for r in paths:
        if not r:
            continue
        p = root / r
        (present if p.exists() else missing).append(r)
    return present, missing


def check_sb3_zip(path: Path) -> dict:
    required_any = {'data', 'policy.pth'}
    required_meta = {'_stable_baselines3_version', 'system_info.txt'}
    out = {'path': str(path), 'is_zip': False, 'missing_members': [], 'sb3_version': None, 'ok': False}
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            out['is_zip'] = True
            missing = sorted([x for x in required_any if x not in names] + [x for x in required_meta if x not in names])
            out['missing_members'] = missing
            if '_stable_baselines3_version' in names:
                out['sb3_version'] = z.read('_stable_baselines3_version').decode('utf-8', 'replace').strip()
            out['ok'] = not missing
    except Exception as exc:
        out['error'] = repr(exc)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.', help='Artifact root')
    ap.add_argument('--strict', action='store_true', help='Exit non-zero on missing critical assets')
    ap.add_argument('--check-sha', action='store_true', help='Verify SHA-256 digests from ASSET_INDEX.csv')
    ap.add_argument('--json-out', default=None, help='Optional JSON report path')
    args = ap.parse_args(argv)

    root = Path(args.root).resolve()
    report_dir = root / 'reproducibility' / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)

    core_required = [
        'README.md',
        'CITATION.cff',
        '.zenodo.json',
        'VERSION',
        'requirements-reproducibility.txt',
        'reproducibility/source/TOP_UAV_colab_export.py',
        'reproducibility/source/TOP_UAV_colab_export_no_magics.py',
        'src/ruscuav_benchmark/legacy/definitions.py',
        'reproducibility/environment/python_version.txt',
        'reproducibility/environment/pip_freeze_colab.txt',
        'reproducibility/manifests/TABLE_FILE_MAP.csv',
        'reproducibility/manifests/METHOD_ASSET_MAP.csv',
        'reproducibility/manifests/ASSET_INDEX.csv',
        'docs/DATA_AVAILABILITY.md',
        'docs/REPRODUCIBILITY.md',
    ]

    world_required = [
        'worlds/AOI_FAKE_001_urban_world_layers.npz',
        'worlds/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz',
        'worlds/vienna_prater_leopoldstadt_osm_1km_v1_urban_world_layers.npz',
        'worlds/warsaw_centrum_osm_1km_v1_urban_world_layers.npz',
    ]

    critical_checkpoint = ('experiments/m12C_real_vienna_footway_learning/checkpoints/stage_M12C_vienna_reward_only_C/'
                           'stage_M12C_vienna_reward_only_C_latest.zip')

    table_rows = read_csv(root / 'reproducibility' / 'manifests' / 'TABLE_FILE_MAP.csv')
    table_required = []
    for row in table_rows:
        for item in row.get('artifact_files', '').split(';'):
            item = item.strip()
            if not item:
                continue
            if '*' in item:
                matches = [rel(p, root) for p in root.glob(item)]
                if matches:
                    table_required.extend(matches)
                else:
                    table_required.append(item)
            else:
                table_required.append(item)

    weight_rows = read_csv(root / 'models' / 'WEIGHTS_REQUIRED.csv')
    csv_rows = read_csv(root / 'results' / 'PER_EPISODE_CSV_REQUIRED.csv')
    weight_required = [r.get('expected_path', '') for r in weight_rows]
    csv_required = [r.get('expected_path', '') for r in csv_rows]

    core_present, core_missing = check_exists(root, core_required)
    world_present, world_missing = check_exists(root, world_required)
    table_present, table_missing = check_exists(root, sorted(set(table_required)))
    weight_present, weight_missing = check_exists(root, weight_required)
    csv_present, csv_missing = check_exists(root, csv_required)
    checkpoint_present, checkpoint_missing = check_exists(root, [critical_checkpoint])

    counts = {
        'world_npz': len(list((root / 'worlds').glob('*urban_world_layers.npz'))),
        'paper_ready_csv_tables': len(list((root / 'paper_ready_v1' / 'tables').glob('*.csv'))),
        'paper_ready_appendix_tables': len(list((root / 'paper_ready_v1' / 'tables' / 'appendix').glob('*.csv'))),
        'episode_csv': len(list((root / 'experiments').rglob('*_episodes.csv'))),
        'summary_json': len(list((root / 'experiments').rglob('*_summary.json'))),
        'trajectory_npy': len(list((root / 'experiments').rglob('*.npy'))),
        'model_zip_in_models_dirs': len(list((root / 'experiments').glob('**/models/*.zip'))),
        'all_weight_like_zip': len([p for p in (root / 'experiments').rglob('*.zip')
                                    if ('/models/' in p.as_posix() or '/checkpoints/' in p.as_posix())]),
    }

    # Check SB3 archive structure for required weights.
    sb3_checks = []
    for r in sorted(set(weight_present + checkpoint_present)):
        p = root / r
        if p.suffix.lower() == '.zip':
            sb3_checks.append(check_sb3_zip(p))
    sb3_bad = [x for x in sb3_checks if not x.get('ok')]
    sb3_versions = sorted({x.get('sb3_version') for x in sb3_checks if x.get('sb3_version')})

    # Optional SHA check.
    sha_report = {'checked': 0, 'missing': [], 'mismatch': [], 'enabled': bool(args.check_sha)}
    if args.check_sha:
        idx_path = root / 'reproducibility' / 'manifests' / 'ASSET_INDEX.csv'
        rows = read_csv(idx_path)
        for row in rows:
            r = row.get('relpath', '')
            expected = row.get('sha256', '')
            if not r or not expected:
                continue
            p = root / r
            if not p.exists():
                sha_report['missing'].append(r)
                continue
            got = sha256_file(p)
            sha_report['checked'] += 1
            if got != expected:
                sha_report['mismatch'].append({'relpath': r, 'expected': expected, 'got': got})

    strict_pass = not any([
        core_missing,
        world_missing,
        table_missing,
        weight_missing,
        csv_missing,
        checkpoint_missing,
        sb3_bad,
        sha_report['missing'] if args.check_sha else [],
        sha_report['mismatch'] if args.check_sha else [],
    ])

    report = {
        'artifact_root': str(root),
        'version': (root / 'VERSION').read_text(encoding='utf-8').strip() if (root / 'VERSION').exists() else None,
        'counts': counts,
        'core_required': len(core_required),
        'core_missing': core_missing,
        'worlds_required': len(world_required),
        'worlds_missing': world_missing,
        'table_assets_required': len(set(table_required)),
        'table_assets_missing': table_missing,
        'weights_required': len(weight_required),
        'weights_missing': weight_missing,
        'per_episode_csv_required': len(csv_required),
        'per_episode_csv_missing': csv_missing,
        'critical_checkpoint_missing': checkpoint_missing,
        'sb3_archives_checked': len(sb3_checks),
        'sb3_versions': sb3_versions,
        'sb3_bad_archives': sb3_bad,
        'sha256': sha_report,
        'strict_pass': strict_pass,
    }

    out_path = Path(args.json_out) if args.json_out else report_dir / 'VERIFY_REPORT.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.strict and not strict_pass:
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
