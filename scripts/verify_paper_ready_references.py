#!/usr/bin/env python3
"""Verify that every csv_path, summary_path, and model_path in paper_ready_v1 exists inside the release."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

def normalize(s):
    prefix='/content/drive/MyDrive/RUSCUAV/'
    if s.startswith(prefix): return s[len(prefix):]
    if s.startswith(('experiments/','worlds/','paper_ready_v1/','data/','models/','results/')): return s
    return None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root', default='.')
    args=ap.parse_args(); root=Path(args.root)
    missing=[]; checked=[]
    for p in (root/'paper_ready_v1'/'tables').glob('**/*.csv'):
        rows=list(csv.DictReader(p.open(newline='', encoding='utf-8')))
        for row in rows:
            for col,val in row.items():
                if col.endswith('_path') or col in {'csv_path','summary_path','model_path'}:
                    nr=normalize(str(val).strip()) if val is not None else None
                    if nr:
                        checked.append(nr)
                        if not (root/nr).exists(): missing.append({'table':str(p.relative_to(root)), 'column':col, 'path':nr})
    print(json.dumps({'checked':len(checked), 'missing':missing, 'pass':not missing}, indent=2))
    return 0 if not missing else 2
if __name__ == '__main__':
    raise SystemExit(main())
