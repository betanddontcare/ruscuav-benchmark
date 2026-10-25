#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='.')
    args = ap.parse_args()
    root = Path(args.root).resolve()
    errors = []

    req_path = root / 'manifests' / 'REQUIRED_ASSETS.json'
    if not req_path.exists():
        print('ERROR: missing', req_path)
        return 2
    req = json.loads(req_path.read_text(encoding='utf-8'))

    for rel in req['required']:
        p = root / rel
        if not p.is_file() or p.stat().st_size == 0:
            errors.append(f'missing/empty required asset: {rel}')

    # Real-GIS processed NPZ assets must be absent.
    for p in (root / 'worlds').rglob('*.npz'):
        low = p.name.lower()
        if 'vienna' in low or 'warsaw' in low:
            errors.append(f'forbidden processed real-GIS NPZ: {p.relative_to(root)}')

    # Check complete SHA256 list.
    sums = root / 'SHA256SUMS'
    if not sums.exists():
        errors.append('missing SHA256SUMS')
    else:
        for line in sums.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            expected, rel = line.split('  ', 1)
            p = root / rel
            if not p.exists():
                errors.append(f'SHA256SUMS references missing file: {rel}')
            elif sha256(p) != expected:
                errors.append(f'hash mismatch: {rel}')

    # Master run must be complete.
    state = root / 'provenance' / 'master_run' / 'master_run_state.json'
    if state.exists():
        data = json.loads(state.read_text(encoding='utf-8'))
        for stage, info in data.get('stages', {}).items():
            if info.get('status') != 'complete':
                errors.append(f'master run stage not complete: {stage}={info.get("status")}')

    if errors:
        print('RUSCUAV v2.0.0 verification FAILED')
        for e in errors:
            print(' -', e)
        return 1
    print('RUSCUAV v2.0.0 verification PASSED')
    print('Root:', root)
    print('Required assets:', len(req['required']))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
