from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: Version of the world NPZ layout written by :meth:`UrbanWorldLayers.save_npz`.
#: See docs/WORLD_NPZ_SCHEMA.md.
WORLD_SCHEMA_VERSION = "ruscuav-world/1.0"

#: World identifiers used in earlier releases, mapped to their current names.
#: AOI_FAKE_001 is the canonical synthetic world of RUSCUAV v2.0.0 and earlier.
LEGACY_AOI_ALIASES: Dict[str, str] = {"AOI_FAKE_001": "AOI_SYNTH_000"}


def canonical_aoi_id(aoi_id: str) -> str:
    """Return the current identifier for ``aoi_id``, resolving legacy aliases."""
    return LEGACY_AOI_ALIASES.get(str(aoi_id), str(aoi_id))


#: Array keys every world NPZ must contain.
REQUIRED_SCALAR_KEYS = ("aoi_id", "crs", "bbox", "resolution_m")
REQUIRED_LAYER_KEYS = (
    "terrain_z", "surface_z", "height_above_ground",
    "building_mask", "building_height", "building_roof_z",
    "road_mask", "sidewalk_mask", "green_mask",
    "canopy_mask", "canopy_height", "canopy_top_z", "canopy_density",
    "sensitive_mask",
)
OPTIONAL_LAYER_KEYS = ("water_mask", "railway_mask", "parking_mask")
OPTIONAL_METADATA_KEYS = ("schema_version", "meta_json")


@dataclass
class AOIManifestEntry:
    """Canonical metadata for one RUSCUAV area of interest.

    The bounding box is expressed in a metric local coordinate system:
    ``(minx, miny, maxx, maxy)``. For the bundled synthetic worlds the CRS is
    intentionally local and metric, not a real-world geodetic CRS.
    """

    aoi_id: str
    name: str
    crs: str
    bbox: Tuple[float, float, float, float]
    resolution_m: float
    world_npz_path: Optional[str] = None
    generator: Optional[str] = None
    seed: Optional[int] = None
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "AOIManifestEntry":
        d = dict(d)
        d["bbox"] = tuple(float(x) for x in d["bbox"])
        return AOIManifestEntry(**d)

    def save_json(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return str(path)

    @staticmethod
    def load_json(path: str | Path) -> "AOIManifestEntry":
        with Path(path).open("r", encoding="utf-8") as f:
            return AOIManifestEntry.from_dict(json.load(f))


@dataclass
class UrbanWorldLayers:
    """Aligned raster representation of an urban UAV benchmark world.

    The class intentionally separates semantic classes, physical obstacles, and
    risk fields. Roads and sidewalks are exposure classes. Buildings and canopy
    are both semantic classes and physical obstacles. Sensitive zones are
    semantic overlays and may overlap other masks.
    """

    aoi_id: str
    crs: str
    bbox: Tuple[float, float, float, float]
    resolution_m: float

    terrain_z: np.ndarray
    surface_z: np.ndarray
    height_above_ground: np.ndarray

    building_mask: np.ndarray
    building_height: np.ndarray
    building_roof_z: np.ndarray

    road_mask: np.ndarray
    sidewalk_mask: np.ndarray
    green_mask: np.ndarray

    canopy_mask: np.ndarray
    canopy_height: np.ndarray
    canopy_top_z: np.ndarray
    canopy_density: np.ndarray

    sensitive_mask: np.ndarray

    water_mask: Optional[np.ndarray] = None
    railway_mask: Optional[np.ndarray] = None
    parking_mask: Optional[np.ndarray] = None

    #: Schema of the file the world was loaded from (None for files older than 1.0).
    schema_version: Optional[str] = None
    #: Free-form provenance metadata, stored in the NPZ as a JSON string.
    meta: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.bbox = tuple(float(x) for x in self.bbox)
        self.resolution_m = float(self.resolution_m)

        float_layers = [
            "terrain_z", "surface_z", "height_above_ground",
            "building_height", "building_roof_z",
            "canopy_height", "canopy_top_z", "canopy_density",
        ]
        mask_layers = [
            "building_mask", "road_mask", "sidewalk_mask", "green_mask",
            "canopy_mask", "sensitive_mask",
        ]
        for name in float_layers:
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float32))
        for name in mask_layers:
            arr = np.asarray(getattr(self, name), dtype=np.float32)
            setattr(self, name, (arr > 0.5).astype(np.float32))
        for name in ["water_mask", "railway_mask", "parking_mask"]:
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, (np.asarray(arr, dtype=np.float32) > 0.5).astype(np.float32))
        self.validate_shapes()

    @property
    def shape(self) -> Tuple[int, int]:
        return self.terrain_z.shape

    @property
    def height(self) -> int:
        return int(self.terrain_z.shape[0])

    @property
    def width(self) -> int:
        return int(self.terrain_z.shape[1])

    def validate_shapes(self) -> bool:
        expected = self.terrain_z.shape
        required = [
            "surface_z", "height_above_ground", "building_mask", "building_height",
            "building_roof_z", "road_mask", "sidewalk_mask", "green_mask",
            "canopy_mask", "canopy_height", "canopy_top_z", "canopy_density",
            "sensitive_mask",
        ]
        for name in required:
            arr = getattr(self, name)
            if arr.shape != expected:
                raise ValueError(f"Layer {name!r} has shape {arr.shape}, expected {expected}.")
        for name in ["water_mask", "railway_mask", "parking_mask"]:
            arr = getattr(self, name)
            if arr is not None and arr.shape != expected:
                raise ValueError(f"Optional layer {name!r} has shape {arr.shape}, expected {expected}.")
        return True

    def layers_sha256(self) -> str:
        """SHA-256 checksum of the world content: the resolution and every layer array.

        The layers are hashed as loaded (float32, masks as 0/1), so the checksum does
        not depend on how the file was written (array dtypes, compression, key order)
        or on the identifiers and metadata (``aoi_id``, ``crs``, ``bbox``,
        ``schema_version``, ``meta``), none of which the environment reads. Two worlds
        with the same checksum therefore give the same simulation, and a world rebuilt
        from map data can be checked against the one a recorded run used.
        """
        h = hashlib.sha256(b"ruscuav-world-layers/1\n")
        h.update(np.asarray(self.resolution_m, dtype="<f8").tobytes())
        for name in REQUIRED_LAYER_KEYS + OPTIONAL_LAYER_KEYS:
            arr = getattr(self, name)
            if arr is None:
                continue
            arr = np.ascontiguousarray(arr, dtype="<f4")
            h.update(f"\n{name} {'x'.join(str(n) for n in arr.shape)}\n".encode("ascii"))
            h.update(arr.tobytes())
        return h.hexdigest()

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {
            "schema_version": np.array(WORLD_SCHEMA_VERSION),
            "aoi_id": np.array(self.aoi_id),
            "crs": np.array(self.crs),
            "bbox": np.array(self.bbox, dtype=np.float64),
            "resolution_m": np.array(self.resolution_m, dtype=np.float64),
            "terrain_z": self.terrain_z,
            "surface_z": self.surface_z,
            "height_above_ground": self.height_above_ground,
            "building_mask": self.building_mask,
            "building_height": self.building_height,
            "building_roof_z": self.building_roof_z,
            "road_mask": self.road_mask,
            "sidewalk_mask": self.sidewalk_mask,
            "green_mask": self.green_mask,
            "canopy_mask": self.canopy_mask,
            "canopy_height": self.canopy_height,
            "canopy_top_z": self.canopy_top_z,
            "canopy_density": self.canopy_density,
            "sensitive_mask": self.sensitive_mask,
        }
        for optional in OPTIONAL_LAYER_KEYS:
            arr = getattr(self, optional)
            if arr is not None:
                d[optional] = arr
        if self.meta:
            d["meta_json"] = np.array(json.dumps(self.meta, ensure_ascii=False, sort_keys=True))
        return d

    def save_npz(self, path: str | Path) -> str:
        """Write the world NPZ; returns the file's path (NumPy adds ``.npz`` if it is missing)."""
        path = Path(path)
        if not path.name.endswith(".npz"):
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_npz_dict())
        return str(path)

    @staticmethod
    def load_npz(path: str | Path) -> "UrbanWorldLayers":
        """Load a world NPZ.

        Files are opened with ``allow_pickle=False``, so a world file cannot execute
        code when it is loaded. Use ``ruscuav validate`` for a full report on a file
        that fails to load.
        """
        with np.load(path, allow_pickle=False) as data:
            missing = [k for k in REQUIRED_SCALAR_KEYS + REQUIRED_LAYER_KEYS if k not in data.files]
            if missing:
                raise ValueError(
                    f"{path}: not a RUSCUAV world file; missing keys: {', '.join(missing)}. "
                    "Run `ruscuav validate` for details."
                )
            arrays: Dict[str, Any] = {}
            for key in data.files:
                try:
                    arrays[key] = data[key]
                except ValueError as exc:
                    if "allow_pickle" in str(exc):  # object arrays need pickle, which is refused
                        raise ValueError(
                            f"{path}: {key!r} contains pickled Python objects, which are not allowed "
                            "in world files. Store strings and metadata as plain NumPy arrays."
                        ) from exc
                    raise ValueError(f"{path}: {key!r} could not be read ({exc})") from exc
                if not isinstance(arrays[key], np.ndarray):
                    raise ValueError(f"{path}: {key!r} is not a NumPy array")

        def get_str(key: str) -> str:
            value = arrays[key]
            return str(value.item()) if value.shape == () else str(value)

        if np.shape(arrays["bbox"]) != (4,):
            raise ValueError(f"{path}: bbox must have shape (4,); got {np.shape(arrays['bbox'])}")
        kwargs: Dict[str, Any] = {
            "aoi_id": get_str("aoi_id"),
            "crs": get_str("crs"),
            "bbox": tuple(float(x) for x in arrays["bbox"]),
            "resolution_m": float(np.asarray(arrays["resolution_m"]).reshape(())),
        }
        for key in REQUIRED_LAYER_KEYS:
            kwargs[key] = arrays[key]
        for optional in OPTIONAL_LAYER_KEYS:
            if optional in arrays:
                kwargs[optional] = arrays[optional]
        if "schema_version" in arrays:
            kwargs["schema_version"] = get_str("schema_version")
        if "meta_json" in arrays:
            try:
                kwargs["meta"] = json.loads(get_str("meta_json"))
            except ValueError as exc:
                raise ValueError(f"{path}: meta_json is not valid JSON ({exc})") from exc
        return UrbanWorldLayers(**kwargs)


def _mask_stats(mask: np.ndarray) -> Dict[str, float]:
    m = np.asarray(mask) > 0.5
    return {"pixels": int(m.sum()), "coverage": float(m.mean())}


def _value_stats(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float32)
    vals = a[np.asarray(mask) > 0.5] if mask is not None else a[np.isfinite(a)]
    if vals.size == 0:
        return {"count": 0, "min": float("nan"), "mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {
        "count": int(vals.size),
        "min": float(np.nanmin(vals)),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "max": float(np.nanmax(vals)),
    }


BASE_RISK_PROFILES: Dict[str, Dict[str, float]] = {
    "efficiency_first": {"green": 0.05, "canopy": 0.10, "building": 1.50, "road": 3.00, "sidewalk": 6.00, "sensitive": 30.00},
    "balanced": {"green": 0.10, "canopy": 0.20, "building": 3.00, "road": 6.00, "sidewalk": 12.00, "sensitive": 50.00},
    "safety_first": {"green": 0.05, "canopy": 0.10, "building": 5.00, "road": 10.00, "sidewalk": 20.00, "sensitive": 100.00},
}


def compute_base_semantic_risk_map(world: UrbanWorldLayers, risk_profile: str = "balanced") -> np.ndarray:
    if risk_profile not in BASE_RISK_PROFILES:
        raise ValueError(f"Unknown risk_profile={risk_profile!r}")
    w = BASE_RISK_PROFILES[risk_profile]
    risk = np.zeros(world.shape, dtype=np.float32)
    risk += w["green"] * world.green_mask
    risk += w["canopy"] * world.canopy_density
    risk += w["building"] * world.building_mask
    risk += w["road"] * world.road_mask
    risk += w["sidewalk"] * world.sidewalk_mask
    risk += w["sensitive"] * world.sensitive_mask
    risk[world.sensitive_mask > 0.5] = np.maximum(risk[world.sensitive_mask > 0.5], w["sensitive"])
    return risk.astype(np.float32)


def diagnose_world_layer_overlaps(world: UrbanWorldLayers) -> Dict[str, Any]:
    b = world.building_mask > 0.5
    r = world.road_mask > 0.5
    s = world.sidewalk_mask > 0.5
    g = world.green_mask > 0.5
    c = world.canopy_mask > 0.5
    z = world.sensitive_mask > 0.5
    checks = {
        "building__road": b & r,
        "building__sidewalk": b & s,
        "building__green": b & g,
        "road__green": r & g,
        "road__sidewalk": r & s,
        "sidewalk__green": s & g,
        "canopy_outside_green": c & (~g),
        "sensitive__building": z & b,
        "sensitive__road": z & r,
        "sensitive__sidewalk": z & s,
        "sensitive__green": z & g,
    }
    hard_zero = {
        "building__road", "building__sidewalk", "building__green",
        "road__green", "road__sidewalk", "sidewalk__green", "canopy_outside_green",
    }
    total = float(world.terrain_z.size)
    rows = {}
    hard_fail = False
    for name, mask in checks.items():
        count = int(mask.sum())
        rows[name] = {"pixels": count, "coverage": count / total, "hard_zero": name in hard_zero}
        if name in hard_zero and count > 0:
            hard_fail = True
    return {"passed": not hard_fail, "checks": rows}


def diagnose_world_height_sanity(world: UrbanWorldLayers) -> Dict[str, Any]:
    checks = {
        "surface_below_terrain_pixels": int(np.sum(world.surface_z < world.terrain_z - 1e-3)),
        "negative_height_above_ground_pixels": int(np.sum(world.height_above_ground < -1e-3)),
        "building_height_negative_pixels": int(np.sum(world.building_height < -1e-3)),
        "canopy_height_negative_pixels": int(np.sum(world.canopy_height < -1e-3)),
        "building_mask_without_height_pixels": int(np.sum((world.building_mask > 0.5) & (world.building_height <= 0.1))),
        "canopy_mask_without_height_pixels": int(np.sum((world.canopy_mask > 0.5) & (world.canopy_height <= 0.1))),
    }
    return {"passed": all(v == 0 for v in checks.values()), "checks": checks}


def summarize_world_layers(world: UrbanWorldLayers) -> Dict[str, Any]:
    return {
        "aoi_id": world.aoi_id,
        "crs": world.crs,
        "bbox": list(world.bbox),
        "resolution_m": world.resolution_m,
        "shape": list(world.shape),
        "terrain_z": _value_stats(world.terrain_z),
        "surface_z": _value_stats(world.surface_z),
        "height_above_ground": _value_stats(world.height_above_ground),
        "building_mask": _mask_stats(world.building_mask),
        "road_mask": _mask_stats(world.road_mask),
        "sidewalk_mask": _mask_stats(world.sidewalk_mask),
        "green_mask": _mask_stats(world.green_mask),
        "canopy_mask": _mask_stats(world.canopy_mask),
        "sensitive_mask": _mask_stats(world.sensitive_mask),
        "building_height_over_buildings": _value_stats(world.building_height, world.building_mask),
        "canopy_height_over_canopy": _value_stats(world.canopy_height, world.canopy_mask),
    }


def run_world_gate_checks(world: UrbanWorldLayers, risk_profile: str = "balanced") -> Dict[str, Any]:
    """The notebook's world gate (M1), which its synthetic worlds pass: mutually
    exclusive classes (no cell two of building, road, sidewalk, green; canopy inside
    green areas) and height sanity. Worlds built from OpenStreetMap, the paper's
    Vienna world among them, have overlapping classes and fail the first part
    (``overlap["passed"]``); ``ruscuav validate`` accepts them, as the environment
    counts a cell towards each of its classes."""
    overlap = diagnose_world_layer_overlaps(world)
    height = diagnose_world_height_sanity(world)
    risk = compute_base_semantic_risk_map(world, risk_profile)
    return {
        "passed": bool(overlap["passed"] and height["passed"]),
        "summary": summarize_world_layers(world),
        "overlap": overlap,
        "height": height,
        "risk_profile": risk_profile,
        "risk_summary": {
            "min": float(np.min(risk)),
            "mean": float(np.mean(risk)),
            "median": float(np.median(risk)),
            "max": float(np.max(risk)),
        },
    }
