#!/usr/bin/env python3
"""Build a real-GIS world from OpenStreetMap: the v2.0.0 options, on ``ruscuav build-aoi``.

Kept for the command line of v2.0.0, whose script called the notebook's
``build_real_world_for_aoi`` (and could not run outside the notebook). ``ruscuav
build-aoi`` (docs/BUILDING_WORLDS.md) runs the same pipeline and has more options.
Without ``--osm`` the features are downloaded from the Overpass API, as in v2.0.0;
like v2.0.0 the world keeps the sidewalk proxy unless ``--sidewalks footway-only``
(the worlds of the paper's evaluations) is given, and the heights of a DEM as they
are unless ``--terrain-datum relative`` is given (the environment flies between 0 and
150 m, so ``ruscuav validate`` refuses terrain above that). Needs the "gis" extra.

    python scripts/build_real_gis_world.py --aoi-id my_area --center-lat 48.2152 --center-lon 16.3975 \\
        --osm area.osm --out worlds/my_area.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ruscuav_benchmark.cli import main as ruscuav  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--aoi-id", required=True)
    ap.add_argument("--center-lat", type=float, required=True)
    ap.add_argument("--center-lon", type=float, required=True)
    ap.add_argument("--size-m", type=float, default=1000.0)
    ap.add_argument("--resolution-m", type=float, default=1.0)
    ap.add_argument("--dem-tif-path", default=None)
    ap.add_argument("--dsm-tif-path", default=None)
    ap.add_argument("--osm", default=None, help="OSM XML file (default: download from the Overpass API)")
    ap.add_argument("--sidewalks", default="proxy", choices=["proxy", "footway-only"])
    ap.add_argument("--terrain-datum", default="absolute", choices=["absolute", "relative"],
                    help="heights of the DEM and DSM: as the files have them (default, as v2.0.0) or relative to "
                         "the area's lowest whole metre")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    argv = ["build-aoi", "--aoi-id", args.aoi_id, f"--center={args.center_lat},{args.center_lon}",
            "--size", str(args.size_m), "--resolution", str(args.resolution_m), "--sidewalks", args.sidewalks,
            "--output", args.out]
    argv += ["--osm", args.osm] if args.osm else ["--download"]
    argv += ["--dem", args.dem_tif_path, "--terrain-datum", args.terrain_datum] if args.dem_tif_path else []
    argv += ["--dsm", args.dsm_tif_path] if args.dsm_tif_path else []
    return ruscuav(argv)


if __name__ == "__main__":
    raise SystemExit(main())
