#!/usr/bin/env python3
"""Rewrite an archived world NPZ in the ruscuav-world/1.0 layout.

The raster layers are copied unchanged. The output adds ``schema_version`` and a
``meta_json`` provenance record (source file name and SHA-256, previous world ID),
and can rename the world or correct its CRS label. The script refuses to write a
file whose layers, bbox or resolution differ from the input in value or dtype
(compared on the raw files), or that fails ``ruscuav validate``.

Example (canonical world of RUSCUAV v2.0.0)::

    python scripts/upgrade_world_npz.py \\
        --input /data/RUSCUAV-v2.0.0/worlds/synthetic/AOI_FAKE_001_urban_world_layers.npz \\
        --output worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz \\
        --aoi-id AOI_SYNTH_000 --crs LOCAL_METRIC_SYNTHETIC
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ruscuav_benchmark.validate import validate_world_npz  # noqa: E402
from ruscuav_benchmark.world import (  # noqa: E402
    OPTIONAL_LAYER_KEYS,
    REQUIRED_LAYER_KEYS,
    WORLD_SCHEMA_VERSION,
    UrbanWorldLayers,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def changed_arrays(original: Path, upgraded: Path) -> list[str]:
    """Names of layers, bbox or resolution that differ between two raw NPZ files."""
    changed = []
    with np.load(original, allow_pickle=False) as a, np.load(upgraded, allow_pickle=False) as b:
        keys = [k for k in REQUIRED_LAYER_KEYS + OPTIONAL_LAYER_KEYS + ("bbox", "resolution_m") if k in a.files]
        for key in keys:
            if key not in b.files:
                changed.append(f"{key} (missing)")
                continue
            x, y = a[key], b[key]
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y):
                changed.append(f"{key} ({x.dtype}{list(x.shape)} -> {y.dtype}{list(y.shape)})")
    return changed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--aoi-id", default=None, help="new world ID (the old one is kept in meta_json)")
    p.add_argument("--crs", default=None, help="corrected CRS label")
    p.add_argument("--note", action="append", default=[], help="extra provenance note (repeatable)")
    args = p.parse_args()

    source = UrbanWorldLayers.load_npz(args.input)
    meta = dict(source.meta or {})
    meta.update({
        "upgraded_to": WORLD_SCHEMA_VERSION,
        "source_file": args.input.name,
        "source_sha256": sha256_file(args.input),
        "source_aoi_id": source.aoi_id,
        "source_crs": source.crs,
    })
    if args.note:
        meta["notes"] = list(meta.get("notes", [])) + args.note
    world = replace(
        source,
        aoi_id=args.aoi_id or source.aoi_id,
        crs=args.crs or source.crs,
        schema_version=WORLD_SCHEMA_VERSION,
        meta=meta,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.stem + ".tmp.npz")
    world.save_npz(tmp)
    changed = changed_arrays(args.input, tmp)
    if changed:
        tmp.unlink()
        print(f"Refusing to write {args.output}: the upgrade would change {', '.join(changed)}")
        return 1
    report = validate_world_npz(tmp)
    if not report.ok:
        tmp.unlink()
        print(report.format())
        return 1
    tmp.replace(args.output)
    print(report.format().replace(str(tmp), str(args.output)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
