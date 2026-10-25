"""Validation of RUSCUAV world NPZ files against ``ruscuav-world/1.0``.

The validator never unpickles anything: files are opened with
``allow_pickle=False``, so no code from the file is executed. Errors make a world
unusable for the benchmark; warnings point at choices that are legal but untested
or easy to get wrong. See docs/WORLD_NPZ_SCHEMA.md.
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .world import (
    OPTIONAL_LAYER_KEYS,
    OPTIONAL_METADATA_KEYS,
    REQUIRED_LAYER_KEYS,
    REQUIRED_SCALAR_KEYS,
    WORLD_SCHEMA_VERSION,
    UrbanWorldLayers,
    run_world_gate_checks,
)

MASK_KEYS = ("building_mask", "road_mask", "sidewalk_mask", "green_mask", "canopy_mask", "sensitive_mask")
#: Grid the benchmark environment and planners were validated on.
VALIDATED_SHAPE = (1000, 1000)
VALIDATED_RESOLUTION_M = 1.0
HEIGHT_TOLERANCE_M = 1e-3


@dataclass
class ValidationReport:
    path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "ok": self.ok, "errors": self.errors, "warnings": self.warnings, "info": self.info}

    def format(self) -> str:
        lines = [f"{self.path}: {'OK' if self.ok else 'INVALID'}"]
        for key in ("aoi_id", "schema_version", "shape", "resolution_m", "crs"):
            if key in self.info:
                lines.append(f"  {key}: {self.info[key]}")
        lines += [f"  ERROR: {msg}" for msg in self.errors]
        lines += [f"  warning: {msg}" for msg in self.warnings]
        return "\n".join(lines)


def _scalar_text(arr: np.ndarray) -> str:
    return str(arr.item()) if arr.shape == () else str(arr)


def _check_derived(arrays: Dict[str, np.ndarray], report: ValidationReport) -> None:
    tz = arrays["terrain_z"].astype(np.float64)
    sz = arrays["surface_z"].astype(np.float64)
    bm = arrays["building_mask"] > 0.5
    cm = arrays["canopy_mask"] > 0.5
    bh = arrays["building_height"].astype(np.float64)
    ch = arrays["canopy_height"].astype(np.float64)

    def max_diff(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.max(np.abs(a - b))) if a.size else 0.0

    checks = [
        ("height_above_ground", "surface_z - terrain_z",
         max_diff(arrays["height_above_ground"].astype(np.float64), sz - tz)),
        ("building_roof_z", "terrain_z + building_height on building cells",
         max_diff(arrays["building_roof_z"][bm].astype(np.float64), (tz + bh)[bm])),
        ("canopy_top_z", "terrain_z + canopy_height on canopy cells",
         max_diff(arrays["canopy_top_z"][cm].astype(np.float64), (tz + ch)[cm])),
    ]
    for key, rule, diff in checks:
        if diff > HEIGHT_TOLERANCE_M:
            report.errors.append(f"{key} must equal {rule} (max deviation {diff:.4g} m)")

    density = arrays["canopy_density"]
    if density.min() < 0.0 or density.max() > 1.0:
        report.errors.append(
            f"canopy_density must lie in [0, 1] (found {float(density.min()):.4g} to {float(density.max()):.4g})"
        )
    if np.any(density[~cm] > 0.0):
        report.warnings.append("canopy_density is non-zero outside canopy_mask; it is ignored there")

    roof = np.where(bm, tz + bh, tz)
    canopy_top = np.where(cm, tz + ch, tz)
    surface_diff = max_diff(sz, np.maximum(np.maximum(tz, roof), canopy_top))
    if surface_diff > HEIGHT_TOLERANCE_M:
        report.warnings.append(
            f"surface_z differs from max(terrain, roofs, canopy tops) by up to {surface_diff:.4g} m "
            "(expected only for DSM-based surfaces)"
        )


def validate_world_npz(path: str | Path) -> ValidationReport:
    """Validate one world NPZ file and return a report (never raises on bad content)."""
    path = Path(path)
    report = ValidationReport(path=str(path))
    if not path.exists():
        report.errors.append("file not found")
        return report
    try:
        npz = np.load(path, allow_pickle=False)
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        report.errors.append(f"not a readable NPZ archive ({exc})")
        return report

    arrays: Dict[str, np.ndarray] = {}
    with npz:
        for key in npz.files:
            try:
                arrays[key] = npz[key]
            except ValueError:
                report.errors.append(
                    f"{key!r} contains pickled Python objects; world files may only contain plain arrays"
                )
    if report.errors:
        return report

    # Schema version
    if "schema_version" in arrays:
        version = _scalar_text(arrays["schema_version"])
        report.info["schema_version"] = version
        if version.split(".")[0] != WORLD_SCHEMA_VERSION.split(".")[0]:
            report.errors.append(f"unsupported schema_version {version!r} (expected {WORLD_SCHEMA_VERSION})")
    else:
        report.info["schema_version"] = None
        report.warnings.append(
            f"no schema_version (file predates {WORLD_SCHEMA_VERSION}); it is read with the {WORLD_SCHEMA_VERSION} rules"
        )

    missing = [k for k in REQUIRED_SCALAR_KEYS + REQUIRED_LAYER_KEYS if k not in arrays]
    if missing:
        report.errors.append(f"missing required keys: {', '.join(missing)}")
        return report
    known = set(REQUIRED_SCALAR_KEYS + REQUIRED_LAYER_KEYS + OPTIONAL_LAYER_KEYS + OPTIONAL_METADATA_KEYS)
    extra = sorted(set(arrays) - known)
    if extra:
        report.warnings.append(f"unknown keys are ignored: {', '.join(extra)}")

    # Scalar metadata
    aoi_id = _scalar_text(arrays["aoi_id"]).strip()
    crs = _scalar_text(arrays["crs"]).strip()
    report.info.update({"aoi_id": aoi_id, "crs": crs})
    if not aoi_id:
        report.errors.append("aoi_id is empty")
    if not crs:
        report.errors.append("crs is empty")
    try:
        bbox = np.asarray(arrays["bbox"], dtype=np.float64).ravel()
    except (TypeError, ValueError):
        bbox = np.array([np.nan])
    try:
        res_values = np.asarray(arrays["resolution_m"], dtype=np.float64).ravel()
        resolution = float(res_values[0]) if res_values.size == 1 else float("nan")
    except (TypeError, ValueError):
        resolution = float("nan")
    report.info["resolution_m"] = resolution
    if bbox.size != 4 or not np.all(np.isfinite(bbox)) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        report.errors.append(f"bbox must be four finite numbers (minx, miny, maxx, maxy) with max > min; got {bbox.tolist()}")
    if not (np.isfinite(resolution) and resolution > 0):
        report.errors.append(f"resolution_m must be a positive number; got {arrays['resolution_m']!r}")

    # Raster layers
    layer_keys = list(REQUIRED_LAYER_KEYS) + [k for k in OPTIONAL_LAYER_KEYS if k in arrays]
    shape = arrays["terrain_z"].shape
    report.info["shape"] = list(shape)
    for key in layer_keys:
        arr = arrays[key]
        if arr.ndim != 2:
            report.errors.append(f"{key} must be a 2-D raster; got shape {arr.shape}")
        elif arr.shape != shape:
            report.errors.append(f"{key} has shape {arr.shape}; terrain_z has {shape}")
        if not (np.issubdtype(arr.dtype, np.floating) or np.issubdtype(arr.dtype, np.integer)
                or arr.dtype == np.bool_):
            report.errors.append(f"{key} must be numeric; got dtype {arr.dtype}")
        elif not np.all(np.isfinite(arr.astype(np.float64))):
            report.errors.append(f"{key} contains NaN or infinite values")
    if report.errors:
        return report

    for key in MASK_KEYS + tuple(k for k in OPTIONAL_LAYER_KEYS if k in arrays):
        values = arrays[key].astype(np.float64)
        if values.min() < 0.0 or values.max() > 1.0:
            report.errors.append(f"{key} values must lie in [0, 1]")
        elif np.any((values > 0.0) & (values < 1.0)):
            report.warnings.append(f"{key} is not binary; cells are thresholded at 0.5 when loaded")

    if bbox.size == 4 and np.isfinite(resolution) and resolution > 0:
        width_m, height_m = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if abs(width_m - shape[1] * resolution) > 1e-6 * max(width_m, 1.0) or \
           abs(height_m - shape[0] * resolution) > 1e-6 * max(height_m, 1.0):
            report.errors.append(
                f"bbox extent {width_m:g} × {height_m:g} m does not match the grid "
                f"{shape[1]} × {shape[0]} cells at {resolution:g} m"
            )
        if crs.upper().startswith("EPSG:") and bbox[0] == 0.0 and bbox[1] == 0.0:
            report.warnings.append(
                f"bbox starts at (0, 0), which is typical of a local frame, but crs is {crs}; "
                "check that the CRS really describes these coordinates"
            )

    if tuple(shape) != VALIDATED_SHAPE or abs(resolution - VALIDATED_RESOLUTION_M) > 1e-9:
        report.warnings.append(
            f"the environment and planners are validated on {VALIDATED_SHAPE[0]} × {VALIDATED_SHAPE[1]} "
            f"grids at {VALIDATED_RESOLUTION_M:g} m; this world is {shape[0]} × {shape[1]} at {resolution:g} m"
        )

    if report.errors:
        return report
    _check_derived(arrays, report)

    # Benchmark gate checks: semantic mask exclusivity and height sanity
    world = UrbanWorldLayers.load_npz(path)
    gates = run_world_gate_checks(world)
    for name, row in gates["overlap"]["checks"].items():
        if not (row["hard_zero"] and row["pixels"] > 0):
            continue
        if name == "canopy_outside_green":
            report.errors.append(f"canopy must lie inside green areas: {row['pixels']} canopy cells are outside green_mask")
        else:
            a, b = name.split("__")
            report.errors.append(f"masks overlap: {a} and {b} share {row['pixels']} cells (must be disjoint)")
    for name, count in gates["height"]["checks"].items():
        if count:
            report.errors.append(f"height check failed: {name.replace('_', ' ')} ({count})")
    report.info["coverage"] = {
        k.replace("_mask", ""): round(v["coverage"], 4)
        for k, v in gates["summary"].items() if isinstance(v, dict) and "coverage" in v
    }
    return report
