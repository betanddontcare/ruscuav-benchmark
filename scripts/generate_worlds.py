#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ruscuav_benchmark.synthetic import save_synthetic_world_bundle
from ruscuav_benchmark.visualize import save_world_quicklook
from ruscuav_benchmark.world import UrbanWorldLayers


def main() -> None:
    p = argparse.ArgumentParser(description="Generate bundled RUSCUAV-v0 synthetic worlds.")
    p.add_argument("--output-root", default="data/ruscuav_v0/worlds")
    p.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 2026])
    args = p.parse_args()
    out_root = Path(args.output_root)
    for idx, seed in enumerate(args.seeds, start=1):
        aoi_id = f"AOI_SYNTH_{idx:03d}"
        out_dir = out_root / aoi_id
        paths = save_synthetic_world_bundle(out_dir, aoi_id=aoi_id, seed=seed)
        quicklook = out_dir / "quicklook.png"
        save_world_quicklook(UrbanWorldLayers.load_npz(paths["world"]), quicklook)
        print(aoi_id, paths, "quicklook=", quicklook)


if __name__ == "__main__":
    main()
