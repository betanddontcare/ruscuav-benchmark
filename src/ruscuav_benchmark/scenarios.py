from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .risk import UrbanRiskModel
from .world import UrbanWorldLayers, canonical_aoi_id


def read_scenarios(path: str | Path, world_id: str | None = None) -> List[dict]:
    """Read a scenario CSV, optionally keeping only the rows of one world (legacy world
    IDs match too). A file without a ``world_id`` column is taken to be of that world."""
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        has_world_column = "world_id" in (reader.fieldnames or [])
    if world_id is not None and has_world_column:
        wanted = canonical_aoi_id(world_id)
        rows = [r for r in rows if canonical_aoi_id(r.get("world_id", "")) == wanted]
    return rows


def mission_from_row(row: Dict[str, Any]) -> Dict[str, List[float]]:
    """Reset options ``{"start": [...], "target": [...]}`` of a scenario row.

    The row needs ``start_x``, ``start_y``, ``target_x`` and ``target_y``; ``start_z`` and
    ``target_z`` are used when present and not empty (absolute altitudes in metres).
    """
    def point(prefix: str) -> List[float]:
        try:
            xyz = [float(row[f"{prefix}_x"]), float(row[f"{prefix}_y"])]
        except KeyError as exc:
            raise ValueError(f"scenario row has no {exc.args[0]!r} column") from exc
        z = row.get(f"{prefix}_z")
        if z not in (None, ""):
            xyz.append(float(z))
        return xyz

    return {"start": point("start"), "target": point("target")}


def scenario_seed(row: Dict[str, Any]) -> Optional[int]:
    """Episode seed of a scenario row (None if the row has none)."""
    value = row.get("seed")
    return None if value in (None, "") else int(value)


def sample_static_scenarios(
    world: UrbanWorldLayers,
    n: int = 200,
    seed: int = 0,
    min_distance_m: float = 500.0,
    agl_m: float = 40.0,
) -> List[dict]:
    """Sample deterministic low-risk start-target pairs for one world."""
    rng = np.random.default_rng(int(seed))
    risk_model = UrbanRiskModel(world, profile="balanced", hard_sensitive_zones=True)
    safe = (
        (world.building_mask <= 0.5)
        & (world.road_mask <= 0.5)
        & (world.sidewalk_mask <= 0.5)
        & (world.sensitive_mask <= 0.5)
        & (world.canopy_mask <= 0.5)
    )
    ys, xs = np.where(safe)
    if len(xs) < 2:
        raise RuntimeError(f"No enough safe cells in {world.aoi_id}")
    rows: List[dict] = []
    attempts = 0
    max_attempts = int(n) * 200
    while len(rows) < int(n) and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, len(xs), size=2)
        start = np.array([float(xs[i]) * world.resolution_m, float(ys[i]) * world.resolution_m], dtype=np.float32)
        target = np.array([float(xs[j]) * world.resolution_m, float(ys[j]) * world.resolution_m], dtype=np.float32)
        if float(np.linalg.norm(target - start)) < float(min_distance_m):
            continue
        # Make sure endpoints are not hard-violating at evaluation altitude.
        ok = True
        for xy in (start, target):
            z = risk_model.terrain_at(float(xy[0]), float(xy[1])) + float(agl_m)
            if risk_model.constraints_at(float(xy[0]), float(xy[1]), z)["any_hard_violation"]:
                ok = False
                break
        if not ok:
            continue
        rows.append({
            "scenario_id": f"{world.aoi_id}_S{len(rows):04d}",
            "world_id": world.aoi_id,
            "seed": int(seed),
            "start_x": f"{start[0]:.3f}",
            "start_y": f"{start[1]:.3f}",
            "target_x": f"{target[0]:.3f}",
            "target_y": f"{target[1]:.3f}",
            "min_distance_m": f"{float(min_distance_m):.3f}",
            "agl_m": f"{float(agl_m):.3f}",
        })
    if len(rows) < int(n):
        raise RuntimeError(f"Could only sample {len(rows)} scenarios for {world.aoi_id}")
    return rows


def write_scenarios_csv(path: str | Path, rows: List[dict]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return str(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(path)
