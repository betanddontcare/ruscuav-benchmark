#!/usr/bin/env python3
"""Verify that every csv_path, summary_path and model_path of the paper-ready tables exists inside the release.

The tables are those of ``paper_ready_v2/tables`` (the v2.0.0 archive) or, in a
v1 release, ``paper_ready_v1/tables``; ``--tables`` names another folder. Paths are
read relative to the release root (the notebook's ``/content/drive/MyDrive/RUSCUAV/``
prefix removed). Exits with 2 if a referenced file is missing, and with 1 if there
are no tables to check.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

PREFIXES = ('experiments/', 'worlds/', 'paper_ready_v1/', 'paper_ready_v2/', 'data/', 'models/', 'results/')


def shown(path, root):
    """``path`` relative to ``root`` when it is inside it."""
    return str(path.relative_to(root)) if path.is_relative_to(root) else str(path)


def normalize(s):
    prefix='/content/drive/MyDrive/RUSCUAV/'
    if s.startswith(prefix):
        return s[len(prefix):]
    if s.startswith(PREFIXES):
        return s
    return None

def main():
    ap=argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument('--root', default='.')
    ap.add_argument('--tables', default=None,
                    help='folder of the tables, relative to --root (default: paper_ready_v2/tables, else '
                         'paper_ready_v1/tables)')
    args=ap.parse_args()
    root=Path(args.root)
    if args.tables is not None:
        tables = root / args.tables
    else:
        tables = next((root / d for d in ('paper_ready_v2/tables', 'paper_ready_v1/tables') if (root / d).is_dir()),
                      root / 'paper_ready_v2/tables')
    files = sorted(tables.glob('**/*.csv'))
    if not files:
        print(f'no tables in {tables}')
        return 1
    missing=[]
    checked=[]
    for p in files:
        rows=list(csv.DictReader(p.open(newline='', encoding='utf-8')))
        for row in rows:
            for col,val in row.items():
                if col.endswith('_path') or col in {'csv_path','summary_path','model_path'}:
                    nr=normalize(str(val).strip()) if val is not None else None
                    if nr:
                        checked.append(nr)
                        if not (root/nr).exists():
                            missing.append({'table':shown(p, root), 'column':col, 'path':nr})
    print(json.dumps({'tables': shown(tables, root), 'checked':len(checked), 'missing':missing,
                      'pass':not missing}, indent=2))
    return 0 if not missing else 2
if __name__ == '__main__':
    raise SystemExit(main())
