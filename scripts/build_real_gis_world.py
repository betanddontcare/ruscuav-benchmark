#!/usr/bin/env python3
"""Build a real-GIS RUSCUAV world from OSM-derived layers.

Thin wrapper around the TOP_UAV real-GIS pipeline. The draft paper does not
expose exact AOI center/bbox values in the data-availability section, so this
script requires explicit center coordinates and AOI size.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--aoi-id', required=True)
    ap.add_argument('--center-lat', type=float, required=True)
    ap.add_argument('--center-lon', type=float, required=True)
    ap.add_argument('--size-m', type=float, default=1000.0)
    ap.add_argument('--resolution-m', type=float, default=1.0)
    ap.add_argument('--dem-tif-path', default=None)
    ap.add_argument('--dsm-tif-path', default=None)
    ap.add_argument('--out', required=True)
    args=ap.parse_args()
    from ruscuav_benchmark.legacy import definitions as R
    result = R.build_real_world_for_aoi(
        aoi_id=args.aoi_id,
        center_lat=args.center_lat,
        center_lon=args.center_lon,
        size_m=args.size_m,
        resolution_m=args.resolution_m,
        dem_tif_path=args.dem_tif_path,
        dsm_tif_path=args.dsm_tif_path,
        verbose=True,
    )
    # Notebook function returns (world, aoi_geo, layers_local, masks).
    world = result[0] if isinstance(result, tuple) else result
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    world.save_npz(args.out)
    print('saved', args.out)

if __name__ == '__main__':
    main()
