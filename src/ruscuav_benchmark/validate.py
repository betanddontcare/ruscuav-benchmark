"""Validation of RUSCUAV world NPZ files against ``ruscuav-world/1.0``.

The validator never unpickles anything: files are opened with
``allow_pickle=False``, so no code from the file is executed. Errors make a world
unusable for the benchmark; warnings point at choices that are legal but untested
or easy to get wrong. See docs/WORLD_NPZ_SCHEMA.md.
"""

from __future__ import annotations

import json
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
#: The environment's workspace ceiling, and its lowest and cruise heights above the
#: terrain (m): the defaults of RuscuavEnv (``DynamicsConfig``), which protocols keep.
WORKSPACE_CEILING_M = 150.0
MIN_AGL_M = 2.0
CRUISE_AGL_M = 30.0


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
            "(expected for surfaces from a DSM, and in worlds built from OpenStreetMap where a building "
            "replaced canopy)"
        )
    lowest = float(tz.min()) if tz.size else 0.0
    highest = float(tz.max()) if tz.size else 0.0
    if lowest + MIN_AGL_M > WORKSPACE_CEILING_M:
        report.errors.append(
            f"the terrain's lowest point is {lowest:.4g} m: the environment cannot fly {MIN_AGL_M:g} m above it "
            f"below its ceiling of {WORKSPACE_CEILING_M:g} m (dynamics.z_max_m); give terrain heights relative to "
            "the area"
        )
    elif highest + CRUISE_AGL_M > WORKSPACE_CEILING_M:
        report.warnings.append(
            f"the terrain reaches {highest:.4g} m; the environment flies below {WORKSPACE_CEILING_M:g} m "
            f"(dynamics.z_max_m), cruising {CRUISE_AGL_M:g} m above the terrain: give terrain heights relative "
            "to the area"
        )


def _is_pickle_error(exc: BaseException) -> bool:
    return "allow_pickle" in str(exc)


def _scalar(arrays: Dict[str, Any], key: str, report: ValidationReport, kind: str) -> Any:
    """Return the value of a 0-d metadata array, or None after recording an error."""
    arr = arrays[key]
    if arr.ndim != 0:
        report.errors.append(f"{key} must be a scalar (0-d array); got shape {arr.shape}")
        return None
    if kind == "text":
        if arr.dtype.kind != "U":
            report.errors.append(f"{key} must be a string; got dtype {arr.dtype}")
            return None
        return str(arr.item())
    if arr.dtype.kind not in "iuf":
        report.errors.append(f"{key} must be a number; got dtype {arr.dtype}")
        return None
    return float(arr.item())


def validate_world_npz(path: str | Path) -> ValidationReport:
    """Validate one world NPZ file and return a report.

    It never raises because of the file's content: every problem, including an
    unreadable or truncated archive, is recorded in ``report.errors``.
    """
    path = Path(path)
    report = ValidationReport(path=str(path))
    try:
        _validate(path, report)
    except Exception as exc:  # last resort; specific checks above should catch problems first
        report.errors.append(f"could not be validated ({type(exc).__name__}: {exc})")
    return report


def _read_arrays(path: Path, report: ValidationReport) -> Dict[str, np.ndarray]:
    # NumPy takes any file that is neither a ZIP archive nor a .npy file for a pickle,
    # and refuses it as such; say what the file is instead
    try:
        with path.open("rb") as f:
            magic = f.read(6)
    except OSError as exc:
        report.errors.append(f"cannot be read ({type(exc).__name__}: {exc})")
        return {}
    if not magic:
        report.errors.append("not an NPZ archive: the file is empty")
        return {}
    if not (magic.startswith(b"PK\x03\x04") or magic.startswith(b"PK\x05\x06") or magic == b"\x93NUMPY"):
        report.errors.append("not an NPZ archive: the file is neither a ZIP archive nor a NumPy file")
        return {}
    try:
        npz = np.load(path, allow_pickle=False)
    except Exception as exc:
        report.errors.append(f"not a readable NPZ archive ({type(exc).__name__}: {exc})")
        return {}
    if not hasattr(npz, "files"):  # np.load returns a bare array for .npy content
        report.errors.append("not an NPZ archive: the file holds a single NumPy array, not named world layers")
        return {}
    arrays: Dict[str, np.ndarray] = {}
    with npz:
        for key in npz.files:
            try:
                value = npz[key]
            except Exception as exc:
                if _is_pickle_error(exc):
                    report.errors.append(
                        f"{key!r} contains pickled Python objects; world files may only contain plain arrays"
                    )
                else:
                    report.errors.append(f"{key!r} could not be read ({type(exc).__name__}: {exc})")
                continue
            if not isinstance(value, np.ndarray):
                report.errors.append(f"{key!r} is not a NumPy array (archive member without .npy data)")
                continue
            arrays[key] = value
    return arrays


def _validate(path: Path, report: ValidationReport) -> None:
    if not path.exists():
        report.errors.append("file not found")
        return
    arrays = _read_arrays(path, report)
    if report.errors:
        return

    # Schema version and provenance metadata
    if "schema_version" in arrays:
        version = _scalar(arrays, "schema_version", report, "text")
        report.info["schema_version"] = version
        if version is not None and version.split(".")[0] != WORLD_SCHEMA_VERSION.split(".")[0]:
            report.errors.append(f"unsupported schema_version {version!r} (expected {WORLD_SCHEMA_VERSION})")
    else:
        report.info["schema_version"] = None
        report.warnings.append(
            f"no schema_version (file predates {WORLD_SCHEMA_VERSION}); "
            f"it is read with the {WORLD_SCHEMA_VERSION} rules"
        )
    if "meta_json" in arrays:
        text = _scalar(arrays, "meta_json", report, "text")
        if text is not None:
            try:
                meta = json.loads(text)
            except ValueError as exc:
                report.errors.append(f"meta_json is not valid JSON ({exc})")
            else:
                if not isinstance(meta, dict):
                    report.errors.append("meta_json must hold a JSON object")

    missing = [k for k in REQUIRED_SCALAR_KEYS + REQUIRED_LAYER_KEYS if k not in arrays]
    if missing:
        report.errors.append(f"missing required keys: {', '.join(missing)}")
        return
    known = set(REQUIRED_SCALAR_KEYS + REQUIRED_LAYER_KEYS + OPTIONAL_LAYER_KEYS + OPTIONAL_METADATA_KEYS)
    extra = sorted(set(arrays) - known)
    if extra:
        report.warnings.append(f"unknown keys are ignored: {', '.join(extra)}")

    # Scalar metadata
    aoi_id = _scalar(arrays, "aoi_id", report, "text")
    crs = _scalar(arrays, "crs", report, "text")
    resolution = _scalar(arrays, "resolution_m", report, "number")
    report.info.update({"aoi_id": aoi_id, "crs": crs, "resolution_m": resolution})
    if aoi_id is not None and not aoi_id.strip():
        report.errors.append("aoi_id is empty")
    if crs is not None and not crs.strip():
        report.errors.append("crs is empty")
    if resolution is not None and not (np.isfinite(resolution) and resolution > 0):
        report.errors.append(f"resolution_m must be a positive number; got {resolution}")
    bbox_arr = arrays["bbox"]
    bbox = None
    if bbox_arr.shape != (4,) or bbox_arr.dtype.kind not in "iuf":
        report.errors.append(f"bbox must be 4 numbers (minx, miny, maxx, maxy); got shape {bbox_arr.shape}, "
                             f"dtype {bbox_arr.dtype}")
    else:
        bbox = bbox_arr.astype(np.float64)
        if not np.all(np.isfinite(bbox)) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            report.errors.append(f"bbox must be finite with max > min; got {bbox.tolist()}")

    # Raster layers
    layer_keys = list(REQUIRED_LAYER_KEYS) + [k for k in OPTIONAL_LAYER_KEYS if k in arrays]
    shape = arrays["terrain_z"].shape
    report.info["shape"] = list(shape)
    for key in layer_keys:
        arr = arrays[key]
        if arr.ndim != 2:
            report.errors.append(f"{key} must be a 2-D raster; got shape {arr.shape}")
            continue
        if arr.shape != shape:
            report.errors.append(f"{key} has shape {arr.shape}; terrain_z has {shape}")
        elif arr.size == 0:
            report.errors.append(f"{key} is empty (shape {arr.shape})")
        if arr.dtype.kind not in "iufb":
            report.errors.append(f"{key} must be numeric; got dtype {arr.dtype}")
        elif arr.size and not np.all(np.isfinite(arr.astype(np.float64))):
            report.errors.append(f"{key} contains NaN or infinite values")
    if report.errors:
        return

    for key in MASK_KEYS + tuple(k for k in OPTIONAL_LAYER_KEYS if k in arrays):
        values = arrays[key].astype(np.float64)
        if values.min() < 0.0 or values.max() > 1.0:
            report.errors.append(f"{key} values must lie in [0, 1]")
        elif np.any((values > 0.0) & (values < 1.0)):
            report.warnings.append(f"{key} is not binary; cells are thresholded at 0.5 when loaded")

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
        return
    _check_derived(arrays, report)

    # Benchmark gate checks: semantic classes and height sanity. Overlapping classes are
    # legal (the exposure of a cell counts towards each of its classes); worlds built
    # from OpenStreetMap, the paper's Vienna and Warsaw worlds among them, have roads
    # under building passages and roads and footways through green areas.
    world = UrbanWorldLayers.load_npz(path)
    gates = run_world_gate_checks(world)
    for name, row in gates["overlap"]["checks"].items():
        if not (row["hard_zero"] and row["pixels"] > 0):
            continue
        if name == "canopy_outside_green":
            report.warnings.append(f"{row['pixels']} canopy cells lie outside green_mask")
        else:
            a, b = name.split("__")
            report.warnings.append(f"masks overlap: {a} and {b} share {row['pixels']} cells, which count as both")
    for name, count in gates["height"]["checks"].items():
        if count:
            report.errors.append(f"height check failed: {name.replace('_', ' ')} ({count})")
    report.info["coverage"] = {
        k.replace("_mask", ""): round(v["coverage"], 4)
        for k, v in gates["summary"].items() if isinstance(v, dict) and "coverage" in v
    }
