from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

from .world import AOIManifestEntry, UrbanWorldLayers, run_world_gate_checks


def _line_distance_to_segment(x: np.ndarray, y: np.ndarray, p0: Tuple[float, float], p1: Tuple[float, float]) -> np.ndarray:
    x0, y0 = p0
    x1, y1 = p1
    vx, vy = x1 - x0, y1 - y0
    wx, wy = x - x0, y - y0
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return np.hypot(x - x0, y - y0)
    t = np.clip((wx * vx + wy * vy) / denom, 0.0, 1.0)
    proj_x = x0 + t * vx
    proj_y = y0 + t * vy
    return np.hypot(x - proj_x, y - proj_y)


def _rect_mask(x: np.ndarray, y: np.ndarray, rect: Tuple[float, float, float, float]) -> np.ndarray:
    minx, miny, maxx, maxy = rect
    return (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)


def create_synthetic_world(
    aoi_id: str = "AOI_SYNTH_001",
    seed: int = 42,
    width: int = 1000,
    height: int = 1000,
    resolution_m: float = 1.0,
    crs: str = "LOCAL_METRIC_SYNTHETIC",
) -> UrbanWorldLayers:
    """Create a deterministic 1 km x 1 km synthetic RUSCUAV raster world.

    This generator is dependency-light and produces the canonical NPZ layer
    structure directly. It intentionally mirrors the semantics of the original
    Colab/GIS prototype: roads and sidewalks are exposure classes; buildings and
    canopy are 3-D obstacles; sensitive zones are overlays.
    """
    rng = np.random.default_rng(int(seed))
    yy, xx = np.mgrid[0:height, 0:width]
    x = xx.astype(np.float32) * float(resolution_m)
    y = yy.astype(np.float32) * float(resolution_m)

    terrain_z = (
        15.0
        + 3.0 * np.sin(x / 90.0)
        + 2.0 * np.cos(y / 120.0)
        + 1.5 * np.sin((x + y) / 150.0)
    ).astype(np.float32)

    road_lines = [
        ((200, 0), (200, 1000)),
        ((500, 0), (500, 1000)),
        ((780, 0), (780, 1000)),
        ((0, 180), (1000, 180)),
        ((0, 470), (1000, 470)),
        ((0, 720), (1000, 720)),
        ((100, 100), (900, 900)),
    ]
    road_mask = np.zeros((height, width), dtype=bool)
    road_outer = np.zeros_like(road_mask)
    road_buffer_m = 7.0
    sidewalk_buffer_m = 6.0
    for p0, p1 in road_lines:
        d = _line_distance_to_segment(x, y, p0, p1)
        road_mask |= d <= road_buffer_m
        road_outer |= d <= (road_buffer_m + sidewalk_buffer_m)

    green_rects = [
        (60, 480, 170, 690),
        (270, 735, 390, 910),
        (610, 780, 720, 980),
        (820, 520, 970, 760),
        (620, 190, 760, 330),
        (120, 190, 220, 310),
        (490, 0, 510, 1000),
        (0, 460, 1000, 480),
    ]
    green_mask = np.zeros((height, width), dtype=bool)
    for rect in green_rects:
        green_mask |= _rect_mask(x, y, rect)

    sidewalk_mask = road_outer & (~road_mask)

    building_mask = np.zeros((height, width), dtype=bool)
    building_height = np.zeros((height, width), dtype=np.float32)
    forbidden = road_outer.copy()
    forbidden |= green_mask

    n_buildings = 75
    attempts = 0
    placed = 0
    while placed < n_buildings and attempts < 1200:
        attempts += 1
        bw = int(rng.integers(22, 80))
        bh = int(rng.integers(22, 90))
        x0 = int(rng.integers(30, max(31, width - bw - 30)))
        y0 = int(rng.integers(30, max(31, height - bh - 30)))
        sl_y = slice(y0, y0 + bh)
        sl_x = slice(x0, x0 + bw)
        if forbidden[sl_y, sl_x].any() or building_mask[sl_y, sl_x].any():
            continue
        h_m = float(rng.uniform(8.0, 55.0))
        building_mask[sl_y, sl_x] = True
        building_height[sl_y, sl_x] = h_m
        placed += 1

    sensitive_centers = [(340, 840), (520, 600), (760, 620), (680, 270), (490, 370)]
    sensitive_radii = [52, 36, 48, 46, 40]
    sensitive_mask = np.zeros((height, width), dtype=bool)
    for (cx, cy), r in zip(sensitive_centers, sensitive_radii):
        sensitive_mask |= ((x - cx) ** 2 + (y - cy) ** 2) <= float(r) ** 2

    # Priority cleanup: buildings dominate ground semantics; green is not road;
    # sidewalks are neither road nor green; sensitive zones remain overlays.
    road_mask[building_mask] = False
    sidewalk_mask[building_mask] = False
    green_mask[building_mask] = False
    road_mask[green_mask] = False
    sidewalk_mask[road_mask] = False
    sidewalk_mask[green_mask] = False

    canopy_noise = rng.random((height, width), dtype=np.float32)
    canopy_mask = green_mask & (canopy_noise > 0.35)
    canopy_height = np.zeros((height, width), dtype=np.float32)
    canopy_height[canopy_mask] = rng.uniform(7.0, 24.0, size=int(canopy_mask.sum())).astype(np.float32)
    canopy_density = canopy_mask.astype(np.float32)

    obstacle_h = np.maximum(building_height, canopy_height).astype(np.float32)
    surface_z = terrain_z + obstacle_h
    height_above_ground = np.maximum(surface_z - terrain_z, 0.0).astype(np.float32)
    building_roof_z = terrain_z + building_height
    canopy_top_z = terrain_z + canopy_height

    bbox = (0.0, 0.0, float(width) * float(resolution_m), float(height) * float(resolution_m))
    return UrbanWorldLayers(
        aoi_id=aoi_id,
        crs=crs,
        bbox=bbox,
        resolution_m=resolution_m,
        terrain_z=terrain_z,
        surface_z=surface_z,
        height_above_ground=height_above_ground,
        building_mask=building_mask.astype(np.float32),
        building_height=building_height,
        building_roof_z=building_roof_z,
        road_mask=road_mask.astype(np.float32),
        sidewalk_mask=sidewalk_mask.astype(np.float32),
        green_mask=green_mask.astype(np.float32),
        canopy_mask=canopy_mask.astype(np.float32),
        canopy_height=canopy_height,
        canopy_top_z=canopy_top_z,
        canopy_density=canopy_density,
        sensitive_mask=sensitive_mask.astype(np.float32),
    )


def save_synthetic_world_bundle(
    output_dir: str | Path,
    aoi_id: str,
    seed: int,
    width: int = 1000,
    height: int = 1000,
    resolution_m: float = 1.0,
) -> Dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    world = create_synthetic_world(aoi_id=aoi_id, seed=seed, width=width, height=height, resolution_m=resolution_m)
    world_path = output_dir / "urban_world_layers.npz"
    world.save_npz(world_path)
    manifest = AOIManifestEntry(
        aoi_id=aoi_id,
        name=f"RUSCUAV synthetic benchmark world {aoi_id}",
        crs=world.crs,
        bbox=world.bbox,
        resolution_m=world.resolution_m,
        world_npz_path="urban_world_layers.npz",
        generator="ruscuav_benchmark.synthetic.create_synthetic_world",
        seed=int(seed),
        notes="Synthetic canonical raster world for RUSCUAV-v0 benchmark release.",
    )
    manifest_path = output_dir / "manifest.json"
    manifest.save_json(manifest_path)
    report = run_world_gate_checks(world, risk_profile="balanced")
    report_path = output_dir / "quality_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return {"world": str(world_path), "manifest": str(manifest_path), "quality_report": str(report_path)}
