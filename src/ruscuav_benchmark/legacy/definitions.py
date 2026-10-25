# -*- coding: utf-8 -*-
"""Definitions-only extraction from TOP_UAV_colab_export.py.

This module intentionally contains imports, constants, class definitions, and
function definitions only. Notebook execution cells, Colab Drive side effects,
plots, and training/evaluation calls were removed so the exact benchmark source
can be imported by CLI scripts.

Some functions require optional extras such as stable-baselines3, sb3-contrib,
rasterio, geopandas, shapely, pyproj, and osmnx. Install the reproducibility
requirements before running training, evaluation, real-GIS, or planner scripts.
"""

import os

import json

from pathlib import Path

from dataclasses import dataclass, asdict

from typing import Optional, Dict, Any, Tuple

import numpy as np

def setup_ruscuav_project(root="/content/ruscuav"):
    """
    Creates the canonical RUSCUAV project directory layout.
    """
    root = Path(root)

    dirs = [
        root,
        root / "data",
        root / "data" / "raw",
        root / "data" / "fake_gis",
        root / "data" / "processed",
        root / "manifests",
        root / "results",
        root / "figures",
        root / "models",
        root / "logs",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RUSCUAV project directories ready")
    print("=" * 80)
    for d in dirs:
        print(d)

    return root

@dataclass
class AOIManifestEntry:
    """
    Canonical description of one Area of Interest.

    bbox is assumed to be in a metric CRS:
        (minx, miny, maxx, maxy)
    """
    aoi_id: str
    name: str
    crs: str
    bbox: Tuple[float, float, float, float]
    resolution_m: float

    dem_path: Optional[str] = None
    dsm_path: Optional[str] = None
    buildings_path: Optional[str] = None
    roads_path: Optional[str] = None
    sidewalks_path: Optional[str] = None
    green_path: Optional[str] = None
    sensitive_path: Optional[str] = None
    water_path: Optional[str] = None
    railway_path: Optional[str] = None
    parking_path: Optional[str] = None

    notes: str = ""

    def to_dict(self):
        d = asdict(self)
        d["bbox"] = list(self.bbox)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["bbox"] = tuple(float(x) for x in d["bbox"])
        return AOIManifestEntry(**d)

def save_aoi_manifest(entry: AOIManifestEntry, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(entry.to_dict(), f, indent=2)

    print("Saved AOI manifest:", path)
    return str(path)

def load_aoi_manifest(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    return AOIManifestEntry.from_dict(d)

@dataclass
class UrbanWorldLayers:
    """
    Canonical RUSCUAV world representation.

    This is the object that all future environments, planners,
    risk models and visualizers should consume.

    Important distinction:
        semantic class != physical obstacle != risk field
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

    # Optional layers for future expansion
    water_mask: Optional[np.ndarray] = None
    railway_mask: Optional[np.ndarray] = None
    parking_mask: Optional[np.ndarray] = None

    def __post_init__(self):
        self.bbox = tuple(float(x) for x in self.bbox)
        self.resolution_m = float(self.resolution_m)

        # Force numeric layers to float32 / binary float32.
        float_layers = [
            "terrain_z",
            "surface_z",
            "height_above_ground",
            "building_height",
            "building_roof_z",
            "canopy_height",
            "canopy_top_z",
            "canopy_density",
        ]

        mask_layers = [
            "building_mask",
            "road_mask",
            "sidewalk_mask",
            "green_mask",
            "canopy_mask",
            "sensitive_mask",
        ]

        for name in float_layers:
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float32))

        for name in mask_layers:
            arr = np.asarray(getattr(self, name), dtype=np.float32)
            arr = (arr > 0.5).astype(np.float32)
            setattr(self, name, arr)

        for name in ["water_mask", "railway_mask", "parking_mask"]:
            arr = getattr(self, name)
            if arr is not None:
                setattr(self, name, (np.asarray(arr, dtype=np.float32) > 0.5).astype(np.float32))

        self.validate_shapes()

    @property
    def shape(self):
        return self.terrain_z.shape

    @property
    def width(self):
        return int(self.terrain_z.shape[1])

    @property
    def height(self):
        return int(self.terrain_z.shape[0])

    def validate_shapes(self):
        expected = self.terrain_z.shape

        required_layers = [
            "surface_z",
            "height_above_ground",
            "building_mask",
            "building_height",
            "building_roof_z",
            "road_mask",
            "sidewalk_mask",
            "green_mask",
            "canopy_mask",
            "canopy_height",
            "canopy_top_z",
            "canopy_density",
            "sensitive_mask",
        ]

        for name in required_layers:
            arr = getattr(self, name)
            if arr.shape != expected:
                raise ValueError(f"Layer '{name}' has shape {arr.shape}, expected {expected}.")

        for name in ["water_mask", "railway_mask", "parking_mask"]:
            arr = getattr(self, name)
            if arr is not None and arr.shape != expected:
                raise ValueError(f"Optional layer '{name}' has shape {arr.shape}, expected {expected}.")

        return True

    def to_npz_dict(self):
        d = {
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

        if self.water_mask is not None:
            d["water_mask"] = self.water_mask
        if self.railway_mask is not None:
            d["railway_mask"] = self.railway_mask
        if self.parking_mask is not None:
            d["parking_mask"] = self.parking_mask

        return d

    def save_npz(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.to_npz_dict())
        print("Saved UrbanWorldLayers:", path)
        return str(path)

    @staticmethod
    def load_npz(path):
        data = np.load(path, allow_pickle=True)

        def get_str(key):
            return str(data[key].item()) if data[key].shape == () else str(data[key])

        kwargs = {
            "aoi_id": get_str("aoi_id"),
            "crs": get_str("crs"),
            "bbox": tuple(float(x) for x in data["bbox"]),
            "resolution_m": float(data["resolution_m"]),

            "terrain_z": data["terrain_z"],
            "surface_z": data["surface_z"],
            "height_above_ground": data["height_above_ground"],

            "building_mask": data["building_mask"],
            "building_height": data["building_height"],
            "building_roof_z": data["building_roof_z"],

            "road_mask": data["road_mask"],
            "sidewalk_mask": data["sidewalk_mask"],

            "green_mask": data["green_mask"],

            "canopy_mask": data["canopy_mask"],
            "canopy_height": data["canopy_height"],
            "canopy_top_z": data["canopy_top_z"],
            "canopy_density": data["canopy_density"],

            "sensitive_mask": data["sensitive_mask"],
        }

        for optional in ["water_mask", "railway_mask", "parking_mask"]:
            if optional in data.files:
                kwargs[optional] = data[optional]

        return UrbanWorldLayers(**kwargs)

    @staticmethod
    def from_adapter_layers(
        layers: Dict[str, np.ndarray],
        aoi_id: str,
        crs: str,
        bbox,
        resolution_m: float,
    ):
        """
        Converts UrbanGISDataAdapter output into UrbanWorldLayers.
        Expected adapter keys:
            dem, dsm, height_above_ground,
            building_mask, building_height,
            road_mask, sidewalk_mask, green_mask,
            canopy_height, canopy_density, sensitive_mask
        """
        terrain_z = np.asarray(layers["dem"], dtype=np.float32)
        surface_z = np.asarray(layers.get("dsm", layers["dem"]), dtype=np.float32)

        height_above_ground = np.asarray(
            layers.get("height_above_ground", np.maximum(surface_z - terrain_z, 0.0)),
            dtype=np.float32,
        )

        building_mask = np.asarray(layers["building_mask"], dtype=np.float32)
        building_height = np.asarray(layers["building_height"], dtype=np.float32)
        building_roof_z = terrain_z + building_height

        road_mask = np.asarray(layers["road_mask"], dtype=np.float32)
        sidewalk_mask = np.asarray(layers["sidewalk_mask"], dtype=np.float32)

        green_mask = np.asarray(layers["green_mask"], dtype=np.float32)

        canopy_height = np.asarray(layers["canopy_height"], dtype=np.float32)
        canopy_density = np.asarray(layers["canopy_density"], dtype=np.float32)
        canopy_mask = (canopy_height > 0.1).astype(np.float32)
        canopy_top_z = terrain_z + canopy_height

        sensitive_mask = np.asarray(layers["sensitive_mask"], dtype=np.float32)

        return UrbanWorldLayers(
            aoi_id=aoi_id,
            crs=crs,
            bbox=tuple(float(x) for x in bbox),
            resolution_m=float(resolution_m),

            terrain_z=terrain_z,
            surface_z=surface_z,
            height_above_ground=height_above_ground,

            building_mask=building_mask,
            building_height=building_height,
            building_roof_z=building_roof_z,

            road_mask=road_mask,
            sidewalk_mask=sidewalk_mask,

            green_mask=green_mask,

            canopy_mask=canopy_mask,
            canopy_height=canopy_height,
            canopy_top_z=canopy_top_z,
            canopy_density=canopy_density,

            sensitive_mask=sensitive_mask,
        )

import numpy as np

def _mask_stats(mask):
    mask = np.asarray(mask) > 0.5
    return int(mask.sum()), float(mask.mean())

def _value_stats(arr, mask=None):
    arr = np.asarray(arr, dtype=np.float32)

    if mask is not None:
        mask = np.asarray(mask) > 0.5
        vals = arr[mask]
    else:
        vals = arr[np.isfinite(arr)]

    if vals.size == 0:
        return dict(count=0, min=np.nan, mean=np.nan, median=np.nan, max=np.nan)

    return dict(
        count=int(vals.size),
        min=float(np.nanmin(vals)),
        mean=float(np.nanmean(vals)),
        median=float(np.nanmedian(vals)),
        max=float(np.nanmax(vals)),
    )

def summarize_world_layers(world: UrbanWorldLayers):
    print("=" * 80)
    print("RUSCUAV UrbanWorldLayers summary")
    print("=" * 80)
    print("AOI:", world.aoi_id)
    print("CRS:", world.crs)
    print("bbox:", world.bbox)
    print("resolution_m:", world.resolution_m)
    print("shape:", world.shape)
    print("-" * 80)

    terrain = _value_stats(world.terrain_z)
    surface = _value_stats(world.surface_z)
    hag = _value_stats(world.height_above_ground)

    print(f"terrain_z            | min={terrain['min']:.2f} | mean={terrain['mean']:.2f} | max={terrain['max']:.2f}")
    print(f"surface_z            | min={surface['min']:.2f} | mean={surface['mean']:.2f} | max={surface['max']:.2f}")
    print(f"height_above_ground  | min={hag['min']:.2f} | mean={hag['mean']:.2f} | max={hag['max']:.2f}")
    print("-" * 80)

    mask_layers = [
        ("building_mask", world.building_mask),
        ("road_mask", world.road_mask),
        ("sidewalk_mask", world.sidewalk_mask),
        ("green_mask", world.green_mask),
        ("canopy_mask", world.canopy_mask),
        ("sensitive_mask", world.sensitive_mask),
    ]

    for name, mask in mask_layers:
        count, frac = _mask_stats(mask)
        print(f"{name:18s} | pixels={count:8d} | coverage={100 * frac:7.3f}%")

    print("-" * 80)

    b = _value_stats(world.building_height, world.building_mask)
    c = _value_stats(world.canopy_height, world.canopy_mask)

    print(
        f"building_height over buildings | count={b['count']} | "
        f"min={b['min']:.2f} | mean={b['mean']:.2f} | max={b['max']:.2f}"
    )
    print(
        f"canopy_height over canopy      | count={c['count']} | "
        f"min={c['min']:.2f} | mean={c['mean']:.2f} | max={c['max']:.2f}"
    )

    print("=" * 80)

def diagnose_world_layer_overlaps(world: UrbanWorldLayers):
    """
    Critical overlap diagnostics.

    Sensitive zones are allowed to overlap with other masks, because they are
    semantic overlays. Other overlaps should usually be zero or near-zero.
    """
    b = world.building_mask > 0.5
    r = world.road_mask > 0.5
    s = world.sidewalk_mask > 0.5
    g = world.green_mask > 0.5
    c = world.canopy_mask > 0.5
    z = world.sensitive_mask > 0.5

    checks = {
        "building ∩ road": b & r,
        "building ∩ sidewalk": b & s,
        "building ∩ green": b & g,
        "road ∩ green": r & g,
        "road ∩ sidewalk": r & s,
        "sidewalk ∩ green": s & g,
        "canopy outside green": c & (~g),
        "sensitive ∩ building": z & b,
        "sensitive ∩ road": z & r,
        "sensitive ∩ sidewalk": z & s,
        "sensitive ∩ green": z & g,
    }

    total = world.terrain_z.size

    print("=" * 80)
    print("RUSCUAV overlap diagnostics")
    print("=" * 80)

    hard_fail = False

    hard_zero = [
        "building ∩ road",
        "building ∩ sidewalk",
        "building ∩ green",
        "road ∩ green",
        "road ∩ sidewalk",
        "sidewalk ∩ green",
        "canopy outside green",
    ]

    for name, mask in checks.items():
        count = int(mask.sum())
        frac = 100.0 * count / total
        status = "OK"

        if name in hard_zero and count > 0:
            status = "CHECK"
            hard_fail = True

        print(f"{name:24s} | pixels={count:8d} | coverage={frac:8.4f}% | {status}")

    print("-" * 80)
    print("Hard-overlap status:", "CHECK REQUIRED" if hard_fail else "PASS")
    print("Note: sensitive overlaps are allowed as semantic overlays.")
    print("=" * 80)

    return not hard_fail

def diagnose_world_height_sanity(world: UrbanWorldLayers):
    """
    Checks numerical sanity of terrain/surface/building/canopy heights.
    """
    print("=" * 80)
    print("RUSCUAV height sanity diagnostics")
    print("=" * 80)

    checks = {}

    checks["surface_below_terrain_pixels"] = int(np.sum(world.surface_z < world.terrain_z - 1e-3))
    checks["negative_height_above_ground_pixels"] = int(np.sum(world.height_above_ground < -1e-3))
    checks["building_height_negative_pixels"] = int(np.sum(world.building_height < -1e-3))
    checks["canopy_height_negative_pixels"] = int(np.sum(world.canopy_height < -1e-3))

    checks["building_mask_without_height_pixels"] = int(
        np.sum((world.building_mask > 0.5) & (world.building_height <= 0.1))
    )
    checks["canopy_mask_without_height_pixels"] = int(
        np.sum((world.canopy_mask > 0.5) & (world.canopy_height <= 0.1))
    )

    for k, v in checks.items():
        status = "OK" if v == 0 else "CHECK"
        print(f"{k:40s} | {v:8d} | {status}")

    print("=" * 80)

    return all(v == 0 for v in checks.values())

RUSCUAV_RISK_PROFILES = {
    "efficiency_first": {
        "green": 0.05,
        "canopy": 0.10,
        "building": 1.50,
        "road": 3.00,
        "sidewalk": 6.00,
        "sensitive": 30.00,
    },
    "balanced": {
        "green": 0.10,
        "canopy": 0.20,
        "building": 3.00,
        "road": 6.00,
        "sidewalk": 12.00,
        "sensitive": 50.00,
    },
    "safety_first": {
        "green": 0.05,
        "canopy": 0.10,
        "building": 5.00,
        "road": 10.00,
        "sidewalk": 20.00,
        "sensitive": 100.00,
    },
}

def compute_base_semantic_risk_map(world: UrbanWorldLayers, risk_profile="balanced"):
    """
    Base semantic risk map without altitude correction.

    Later this will become:
        risk(x, y, z) = base_risk(x, y) * altitude_decay(z_agl)
    """
    if risk_profile not in RUSCUAV_RISK_PROFILES:
        raise ValueError(f"Unknown risk_profile={risk_profile}")

    w = RUSCUAV_RISK_PROFILES[risk_profile]

    risk = np.zeros(world.shape, dtype=np.float32)

    risk += float(w["green"]) * world.green_mask
    risk += float(w["canopy"]) * world.canopy_density
    risk += float(w["building"]) * world.building_mask
    risk += float(w["road"]) * world.road_mask
    risk += float(w["sidewalk"]) * world.sidewalk_mask
    risk += float(w["sensitive"]) * world.sensitive_mask

    # Sensitive zones dominate.
    risk[world.sensitive_mask > 0.5] = np.maximum(
        risk[world.sensitive_mask > 0.5],
        float(w["sensitive"]),
    )

    return risk.astype(np.float32)

def run_world_gate_checks(world: UrbanWorldLayers, risk_profile="balanced"):
    """
    Runs all M1 gate checks.
    """
    summarize_world_layers(world)

    overlap_ok = diagnose_world_layer_overlaps(world)
    height_ok = diagnose_world_height_sanity(world)

    risk = compute_base_semantic_risk_map(world, risk_profile=risk_profile)

    print("=" * 80)
    print("Risk map summary")
    print("=" * 80)
    print(f"risk_profile: {risk_profile}")
    print(f"risk min:     {float(np.min(risk)):.3f}")
    print(f"risk mean:    {float(np.mean(risk)):.3f}")
    print(f"risk median:  {float(np.median(risk)):.3f}")
    print(f"risk max:     {float(np.max(risk)):.3f}")
    print("=" * 80)

    passed = bool(overlap_ok and height_ok)
    print("M1 WORLD GATE:", "PASS" if passed else "CHECK REQUIRED")

    return {
        "passed": passed,
        "overlap_ok": overlap_ok,
        "height_ok": height_ok,
        "risk": risk,
    }

import numpy as np

import matplotlib.pyplot as plt

def plot_world_layers_2d(world: UrbanWorldLayers, risk_profile="balanced", figsize=(12, 12)):
    """
    2D diagnostic visualization of UrbanWorldLayers.
    """
    risk = compute_base_semantic_risk_map(world, risk_profile=risk_profile)

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(
        world.terrain_z,
        origin="lower",
        cmap="terrain",
        alpha=0.80,
    )

    def overlay(mask, cmap, alpha, name):
        arr = np.ma.masked_where(mask <= 0.5, mask)
        ax.imshow(arr, origin="lower", cmap=cmap, alpha=alpha)

    overlay(world.green_mask, "Greens", 0.30, "green")
    overlay(world.road_mask, "Oranges", 0.45, "roads")
    overlay(world.sidewalk_mask, "Wistia", 0.45, "sidewalks")
    overlay(world.building_mask, "Greys", 0.55, "buildings")
    overlay(world.sensitive_mask, "Reds", 0.50, "sensitive")

    ax.set_title(f"RUSCUAV UrbanWorldLayers 2D | AOI={world.aoi_id}")
    ax.set_xlabel("X grid")
    ax.set_ylabel("Y grid")
    ax.grid(alpha=0.2)

    plt.show()

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        risk,
        origin="lower",
        cmap="inferno",
    )

    ax.set_title(f"Base semantic risk map | profile={risk_profile}")
    ax.set_xlabel("X grid")
    ax.set_ylabel("Y grid")
    plt.colorbar(im, ax=ax, label="Base risk")
    plt.show()

def _add_masked_surface(fig, X, Y, Z, mask, name, color, opacity):
    import plotly.graph_objects as go

    Zm = np.where(mask > 0.5, Z, np.nan)

    if not np.isfinite(Zm).any():
        return

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Zm,
            surfacecolor=np.ones_like(Zm, dtype=np.float32),
            colorscale=[[0.0, color], [1.0, color]],
            opacity=opacity,
            showscale=False,
            name=name,
            hoverinfo="skip",
        )
    )

def _add_building_cuboids_world(fig, world: UrbanWorldLayers, max_buildings=120, min_area_px=25):
    """
    Approximate building cuboids from raster components.
    """
    import plotly.graph_objects as go
    from scipy.ndimage import label, find_objects

    mask = world.building_mask > 0.5
    labeled, n = label(mask)
    slices = find_objects(labeled)

    added = 0

    for comp_id, slc in enumerate(slices, start=1):
        if slc is None:
            continue

        yy, xx = slc
        comp = labeled[yy, xx] == comp_id
        area = int(comp.sum())

        if area < min_area_px:
            continue

        y0, y1 = yy.start, yy.stop
        x0, x1 = xx.start, xx.stop

        terrain_vals = world.terrain_z[yy, xx][comp]
        roof_vals = world.building_roof_z[yy, xx][comp]

        if terrain_vals.size == 0 or roof_vals.size == 0:
            continue

        z0 = float(np.nanmedian(terrain_vals))
        z1 = float(np.nanmedian(roof_vals))

        if z1 <= z0 + 1.0:
            continue

        vertices = np.array([
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ], dtype=np.float32)

        i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
        j = [1, 2, 6, 7, 4, 5, 5, 6, 6, 7, 7, 4]
        k = [2, 3, 5, 6, 5, 1, 6, 2, 7, 3, 4, 0]

        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=i,
                j=j,
                k=k,
                color="lightgray",
                opacity=0.58,
                name="buildings",
                showlegend=(added == 0),
            )
        )

        added += 1

        if added >= max_buildings:
            break

def plot_world_layers_3d(
    world: UrbanWorldLayers,
    risk_profile="balanced",
    stride=8,
    color_by="risk",
    show_buildings=True,
    show_canopy=True,
    show_roads=True,
    show_sidewalks=True,
    show_green=True,
    show_sensitive=True,
    max_buildings=120,
    title=None,
):
    """
    3D diagnostic visualization of UrbanWorldLayers.
    """
    import plotly.graph_objects as go

    stride = max(1, int(stride))

    ys = np.arange(0, world.height, stride)
    xs = np.arange(0, world.width, stride)

    X, Y = np.meshgrid(xs, ys)

    Z = world.terrain_z[np.ix_(ys, xs)]

    risk = compute_base_semantic_risk_map(world, risk_profile=risk_profile)
    risk_ds = risk[np.ix_(ys, xs)]

    if color_by == "risk":
        surface_color = risk_ds
        colorscale = "Inferno"
        colorbar_title = "Base semantic risk"
    elif color_by == "terrain":
        surface_color = Z
        colorscale = "Earth"
        colorbar_title = "Terrain elevation"
    else:
        raise ValueError("color_by must be 'risk' or 'terrain'.")

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=surface_color,
            colorscale=colorscale,
            opacity=0.82,
            colorbar=dict(title=colorbar_title),
            name="terrain",
            showscale=True,
        )
    )

    green_ds = world.green_mask[np.ix_(ys, xs)]
    road_ds = world.road_mask[np.ix_(ys, xs)]
    sidewalk_ds = world.sidewalk_mask[np.ix_(ys, xs)]
    sensitive_ds = world.sensitive_mask[np.ix_(ys, xs)]
    canopy_ds = world.canopy_mask[np.ix_(ys, xs)]
    canopy_top_ds = world.canopy_top_z[np.ix_(ys, xs)]

    if show_green:
        _add_masked_surface(fig, X, Y, Z + 0.7, green_ds, "green", "green", 0.30)

    if show_roads:
        _add_masked_surface(fig, X, Y, Z + 1.0, road_ds, "roads", "orange", 0.42)

    if show_sidewalks:
        _add_masked_surface(fig, X, Y, Z + 1.3, sidewalk_ds, "sidewalks", "yellow", 0.38)

    if show_sensitive:
        _add_masked_surface(fig, X, Y, Z + 2.0, sensitive_ds, "sensitive", "red", 0.58)

    if show_canopy:
        _add_masked_surface(fig, X, Y, canopy_top_ds, canopy_ds, "canopy top", "darkgreen", 0.42)

    if show_buildings:
        _add_building_cuboids_world(fig, world, max_buildings=max_buildings)

    if title is None:
        title = f"RUSCUAV UrbanWorldLayers 3D | AOI={world.aoi_id}"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X grid",
            yaxis_title="Y grid",
            zaxis_title="Z",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.25),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )

    fig.show()
    return fig

import os

from pathlib import Path

import numpy as np

class UrbanGISDataAdapter:
    """
    Adapter for converting real GIS data into EnvV15-compatible raster layers.

    Coordinate assumptions:
        - bbox is given in target CRS coordinates, usually meters;
        - target CRS should be metric, e.g. EPSG:2180 for Poland;
        - resolution_m defines grid cell size;
        - generated rasters share the same shape and transform.

    Expected input:
        dem_path        raster terrain model, e.g. NMT / DEM GeoTIFF
        dsm_path        raster surface model, e.g. NMPT / DSM GeoTIFF
        buildings_path  building footprints, vector
        roads_path      road geometries, vector
        sidewalks_path  sidewalk / footway polygons or lines, optional
        green_path      green areas / parks / forests, vector
        sensitive_path  sensitive zones, optional vector

    Output:
        dictionary containing:
            dem
            dsm
            height_above_ground
            building_mask
            building_height
            road_mask
            sidewalk_mask
            green_mask
            canopy_height
            canopy_density
            sensitive_mask

    These can be used as:
        layers = adapter.build_layers()
        dem_data = layers["dem"]
        semantic_layers = {k: v for k, v in layers.items() if k not in ["dem", "dsm", "height_above_ground"]}
        env = UrbanSemanticUAVEnvV15Clean(dem_data=dem_data, semantic_layers=semantic_layers)
    """

    def __init__(
        self,
        bbox,
        resolution_m=1.0,
        target_crs="EPSG:2180",
        dem_path=None,
        dsm_path=None,
        buildings_path=None,
        roads_path=None,
        sidewalks_path=None,
        green_path=None,
        sensitive_path=None,
        water_path=None,
        railway_path=None,
        road_buffer_m=6.0,
        sidewalk_buffer_m=5.0,
        canopy_min_height_m=3.0,
        canopy_max_height_m=45.0,
        building_min_height_m=2.5,
        fill_nodata=True,
    ):
        self.bbox = tuple(float(v) for v in bbox)  # minx, miny, maxx, maxy
        self.resolution_m = float(resolution_m)
        self.target_crs = target_crs

        self.dem_path = dem_path
        self.dsm_path = dsm_path
        self.buildings_path = buildings_path
        self.roads_path = roads_path
        self.sidewalks_path = sidewalks_path
        self.green_path = green_path
        self.sensitive_path = sensitive_path
        self.water_path = water_path
        self.railway_path = railway_path

        self.road_buffer_m = float(road_buffer_m)
        self.sidewalk_buffer_m = float(sidewalk_buffer_m)
        self.canopy_min_height_m = float(canopy_min_height_m)
        self.canopy_max_height_m = float(canopy_max_height_m)
        self.building_min_height_m = float(building_min_height_m)
        self.fill_nodata = bool(fill_nodata)

        minx, miny, maxx, maxy = self.bbox

        self.width = int(np.ceil((maxx - minx) / self.resolution_m))
        self.height = int(np.ceil((maxy - miny) / self.resolution_m))

        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid bbox/resolution. width={self.width}, height={self.height}")

        self.transform = self._make_transform()

    def _make_transform(self):
        from rasterio.transform import from_origin

        minx, miny, maxx, maxy = self.bbox

        return from_origin(
            west=minx,
            north=maxy,
            xsize=self.resolution_m,
            ysize=self.resolution_m,
        )

    def _empty(self, dtype=np.float32):
        return np.zeros((self.height, self.width), dtype=dtype)

    def _read_raster_to_grid(self, raster_path, resampling="bilinear"):
        """
        Reprojects/resamples a raster to the adapter grid.
        """
        if raster_path is None:
            raise ValueError("Raster path is None.")

        import rasterio
        from rasterio.warp import reproject, Resampling

        raster_path = str(raster_path)

        if resampling == "nearest":
            resampling_method = Resampling.nearest
        elif resampling == "bilinear":
            resampling_method = Resampling.bilinear
        else:
            raise ValueError("resampling must be 'nearest' or 'bilinear'.")

        dst = np.full((self.height, self.width), np.nan, dtype=np.float32)

        with rasterio.open(raster_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=self.transform,
                dst_crs=self.target_crs,
                dst_nodata=np.nan,
                resampling=resampling_method,
            )

        if self.fill_nodata:
            dst = self._fill_nan_with_nearest_or_zero(dst)

        return dst.astype(np.float32)

    def _fill_nan_with_nearest_or_zero(self, arr):
        """
        Simple nodata filler. For production, replace with stronger interpolation.
        """
        arr = np.asarray(arr, dtype=np.float32)

        if np.isfinite(arr).all():
            return arr

        if not np.isfinite(arr).any():
            return np.zeros_like(arr, dtype=np.float32)

        try:
            from scipy.ndimage import distance_transform_edt

            mask = ~np.isfinite(arr)
            indices = distance_transform_edt(
                mask,
                return_distances=False,
                return_indices=True,
            )

            filled = arr[tuple(indices)]
            filled[~np.isfinite(filled)] = 0.0
            return filled.astype(np.float32)

        except Exception:
            out = arr.copy()
            median = float(np.nanmedian(out))
            out[~np.isfinite(out)] = median
            return out.astype(np.float32)

    def _read_vector(self, vector_path):
        if vector_path is None:
            return None

        import geopandas as gpd
        from shapely.geometry import box

        vector_path = str(vector_path)

        if not os.path.exists(vector_path):
            raise FileNotFoundError(vector_path)

        gdf = gpd.read_file(vector_path)

        if gdf.empty:
            return gdf

        if gdf.crs is None:
            raise ValueError(f"Vector file has no CRS: {vector_path}")

        gdf = gdf.to_crs(self.target_crs)

        minx, miny, maxx, maxy = self.bbox
        bbox_geom = box(minx, miny, maxx, maxy)

        try:
            gdf = gdf[gdf.intersects(bbox_geom)].copy()
            gdf["geometry"] = gdf.geometry.intersection(bbox_geom)
            gdf = gdf[~gdf.geometry.is_empty].copy()
        except Exception:
            pass

        return gdf

    def _rasterize_gdf(self, gdf, burn_value=1.0, dtype=np.float32):
        """
        Rasterizes geometries to the adapter grid.
        """
        if gdf is None or gdf.empty:
            return self._empty(dtype=dtype)

        from rasterio.features import rasterize

        shapes = [
            (geom, burn_value)
            for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        ]

        if len(shapes) == 0:
            return self._empty(dtype=dtype)

        out = rasterize(
            shapes=shapes,
            out_shape=(self.height, self.width),
            transform=self.transform,
            fill=0.0,
            dtype=dtype,
            all_touched=True,
        )

        return out.astype(dtype)

    def _buffer_gdf(self, gdf, buffer_m):
        if gdf is None or gdf.empty:
            return gdf

        gdf = gdf.copy()
        gdf["geometry"] = gdf.geometry.buffer(float(buffer_m))
        gdf = gdf[~gdf.geometry.is_empty].copy()
        return gdf

    def _rasterize_vector_path(self, vector_path, buffer_m=0.0):
        gdf = self._read_vector(vector_path)

        if gdf is None or gdf.empty:
            return self._empty(dtype=np.float32)

        if buffer_m and abs(float(buffer_m)) > 1e-9:
            gdf = self._buffer_gdf(gdf, buffer_m=buffer_m)

        return self._rasterize_gdf(gdf, burn_value=1.0, dtype=np.float32)

    def _derive_sidewalk_from_roads(self, roads_path, road_mask):
        """
        Fallback sidewalk proxy:
            sidewalk = buffered road corridor - road_mask.

        This is not a substitute for real sidewalk data, but it gives a usable
        proxy when only road centerlines are available.
        """
        if roads_path is None:
            return self._empty(dtype=np.float32)

        outer = self._rasterize_vector_path(
            roads_path,
            buffer_m=self.road_buffer_m + self.sidewalk_buffer_m,
        )

        sidewalk = np.maximum(outer - road_mask, 0.0)
        sidewalk = (sidewalk > 0.5).astype(np.float32)

        return sidewalk

    def _clean_layers(
        self,
        building_mask,
        building_height,
        road_mask,
        sidewalk_mask,
        green_mask,
        canopy_height,
        canopy_density,
        sensitive_mask,
    ):
        """
        Final layer priority cleanup.
        """
        building_mask = (building_mask > 0.5).astype(np.float32)
        road_mask = (road_mask > 0.5).astype(np.float32)
        sidewalk_mask = (sidewalk_mask > 0.5).astype(np.float32)
        green_mask = (green_mask > 0.5).astype(np.float32)
        sensitive_mask = (sensitive_mask > 0.5).astype(np.float32)

        # Buildings dominate ground semantics.
        road_mask[building_mask > 0.5] = 0.0
        sidewalk_mask[building_mask > 0.5] = 0.0
        green_mask[building_mask > 0.5] = 0.0
        canopy_height[building_mask > 0.5] = 0.0
        canopy_density[building_mask > 0.5] = 0.0

        # Green areas are not roads.
        road_mask[green_mask > 0.5] = 0.0

        # Sidewalks are not roads or green areas.
        sidewalk_mask[road_mask > 0.5] = 0.0
        sidewalk_mask[green_mask > 0.5] = 0.0

        # Canopy only over green.
        canopy_height *= green_mask
        canopy_density *= green_mask

        # Building height only over buildings.
        building_height *= building_mask

        return {
            "building_mask": building_mask.astype(np.float32),
            "building_height": np.maximum(building_height, 0.0).astype(np.float32),
            "road_mask": road_mask.astype(np.float32),
            "sidewalk_mask": sidewalk_mask.astype(np.float32),
            "green_mask": green_mask.astype(np.float32),
            "canopy_height": np.maximum(canopy_height, 0.0).astype(np.float32),
            "canopy_density": np.clip(canopy_density, 0.0, 1.0).astype(np.float32),
            "sensitive_mask": sensitive_mask.astype(np.float32),
        }

    def build_layers(self):
        """
        Builds EnvV15-compatible layers from real GIS data.
        """
        if self.dem_path is None:
            raise ValueError("dem_path is required for real GIS adapter.")

        dem = self._read_raster_to_grid(self.dem_path, resampling="bilinear")

        if self.dsm_path is not None:
            dsm = self._read_raster_to_grid(self.dsm_path, resampling="bilinear")
        else:
            dsm = dem.copy()

        height_above_ground = np.maximum(dsm - dem, 0.0).astype(np.float32)

        # Buildings
        building_mask = self._rasterize_vector_path(
            self.buildings_path,
            buffer_m=0.0,
        )

        # Roads
        road_mask = self._rasterize_vector_path(
            self.roads_path,
            buffer_m=self.road_buffer_m,
        )

        # Sidewalks
        if self.sidewalks_path is not None:
            sidewalk_mask = self._rasterize_vector_path(
                self.sidewalks_path,
                buffer_m=1.0,
            )
        else:
            sidewalk_mask = self._derive_sidewalk_from_roads(
                self.roads_path,
                road_mask=road_mask,
            )

        # Green
        green_mask = self._rasterize_vector_path(
            self.green_path,
            buffer_m=0.0,
        )

        # Sensitive zones
        sensitive_mask = self._rasterize_vector_path(
            self.sensitive_path,
            buffer_m=0.0,
        )

        # Building height from DSM - DEM over building footprints.
        building_height = height_above_ground * building_mask
        building_height[building_height < self.building_min_height_m] = 0.0
        building_mask[building_height <= 0.0] = 0.0

        # Canopy height from DSM - DEM over green areas.
        canopy_height = height_above_ground * green_mask

        canopy_valid = (
            (canopy_height >= self.canopy_min_height_m)
            & (canopy_height <= self.canopy_max_height_m)
            & (building_mask <= 0.5)
        )

        canopy_density = canopy_valid.astype(np.float32)
        canopy_height = canopy_height * canopy_valid.astype(np.float32)

        cleaned = self._clean_layers(
            building_mask=building_mask,
            building_height=building_height,
            road_mask=road_mask,
            sidewalk_mask=sidewalk_mask,
            green_mask=green_mask,
            canopy_height=canopy_height,
            canopy_density=canopy_density,
            sensitive_mask=sensitive_mask,
        )

        layers = {
            "dem": dem.astype(np.float32),
            "dsm": dsm.astype(np.float32),
            "height_above_ground": height_above_ground.astype(np.float32),
            **cleaned,
        }

        return layers

    def save_layers_npz(self, output_path, layers):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_path, **layers)
        print("Saved GIS semantic layers:", output_path)

    @staticmethod
    def load_layers_npz(path):
        data = np.load(path)
        return {k: data[k].astype(np.float32) for k in data.files}

def make_env_v15clean_from_gis_layers(
    layers,
    risk_profile="balanced",
    include_semantic_vector=True,
    terminate_on_critical_semantic_violation=False,
    semantic_lambda=0.10,
    n_obstacles=0,
):
    """
    Creates UrbanSemanticUAVEnvV15Clean from layers generated by UrbanGISDataAdapter.

    For real-GIS mode, procedural v14 obstacles should usually start at n_obstacles=0,
    because buildings/canopy are already physical/semantic structures.
    """
    dem_data = layers["dem"]

    semantic_layers = {
        "building_mask": layers["building_mask"],
        "building_height": layers["building_height"],
        "road_mask": layers["road_mask"],
        "sidewalk_mask": layers["sidewalk_mask"],
        "green_mask": layers["green_mask"],
        "canopy_height": layers["canopy_height"],
        "canopy_density": layers["canopy_density"],
        "sensitive_mask": layers["sensitive_mask"],
    }

    env = UrbanSemanticUAVEnvV15Clean(
        dem_data=dem_data,
        semantic_layers=semantic_layers,
        risk_profile=risk_profile,
        include_semantic_vector=include_semantic_vector,
        use_semantic_costs=True,
        use_semantic_altitude=True,
        terminate_on_critical_semantic_violation=terminate_on_critical_semantic_violation,
        semantic_curriculum_enabled=False,
        semantic_lambda=semantic_lambda,
        enforce_semantic_safe_start_target=True,
        start_target_max_risk=1.0,
        max_semantic_reset_attempts=300,
    )

    env.n_obstacles = int(n_obstacles)

    return env

import os

from pathlib import Path

import numpy as np

def create_synthetic_gis_files_for_v15(
    output_dir="/content/data",
    width=1000,
    height=1000,
    resolution_m=1.0,
    crs="EPSG:2180",
    seed=42,
):
    """
    Creates synthetic GIS files:
        dem.tif
        dsm.tif
        buildings.gpkg
        roads.gpkg
        green.gpkg
        sensitive_zones.gpkg

    These files are not real GIS data, but they are valid geospatial files.
    They are designed to test UrbanGISDataAdapter end-to-end.
    """

    try:
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.features import rasterize
        import geopandas as gpd
        from shapely.geometry import box, LineString, Point
    except ImportError as e:
        raise ImportError(
            "Missing GIS dependencies. In Colab run:\n"
            "!pip install rasterio geopandas shapely pyproj fiona rtree"
        ) from e

    rng = np.random.default_rng(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Coordinate frame: fake but metric.
    minx, miny = 0.0, 0.0
    maxx, maxy = width * resolution_m, height * resolution_m

    transform = from_origin(
        west=minx,
        north=maxy,
        xsize=resolution_m,
        ysize=resolution_m,
    )

    # ------------------------------------------------------------
    # 1. Synthetic DEM
    # ------------------------------------------------------------
    y, x = np.mgrid[0:height, 0:width]

    dem = (
        15.0
        + 3.0 * np.sin(x / 90.0)
        + 2.0 * np.cos(y / 120.0)
        + 1.5 * np.sin((x + y) / 150.0)
    ).astype(np.float32)

    # ------------------------------------------------------------
    # 2. Vector geometries
    # ------------------------------------------------------------

    # Roads as centerlines. Adapter will buffer them.
    road_lines = [
        LineString([(200, 0), (200, 1000)]),
        LineString([(500, 0), (500, 1000)]),
        LineString([(780, 0), (780, 1000)]),
        LineString([(0, 180), (1000, 180)]),
        LineString([(0, 470), (1000, 470)]),
        LineString([(0, 720), (1000, 720)]),
        LineString([(100, 100), (900, 900)]),
    ]

    roads_gdf = gpd.GeoDataFrame(
        {
            "road_id": list(range(len(road_lines))),
            "class": ["primary", "primary", "secondary", "primary", "primary", "secondary", "local"],
        },
        geometry=road_lines,
        crs=crs,
    )

    # Green areas: parks and median-like corridors.
    green_polys = [
        box(60, 480, 170, 690),
        box(270, 735, 390, 910),
        box(610, 780, 720, 980),
        box(820, 520, 970, 760),
        box(620, 190, 760, 330),
        box(120, 190, 220, 310),
        box(490, 0, 510, 1000),   # vertical green corridor / median
        box(0, 460, 1000, 480),   # horizontal green corridor / median
    ]

    green_gdf = gpd.GeoDataFrame(
        {
            "green_id": list(range(len(green_polys))),
            "type": ["park", "park", "park", "park", "park", "park", "median", "median"],
        },
        geometry=green_polys,
        crs=crs,
    )

    # Buildings: rectangular footprints, avoid obvious roads/parks roughly.
    building_polys = []
    attempts = 0

    road_forbidden = roads_gdf.buffer(35).unary_union
    green_forbidden = green_gdf.geometry.unary_union
    forbidden = road_forbidden.union(green_forbidden)

    while len(building_polys) < 75 and attempts < 500:
        attempts += 1

        bw = float(rng.integers(22, 80))
        bh = float(rng.integers(22, 90))
        x0 = float(rng.integers(30, int(width - bw - 30)))
        y0 = float(rng.integers(30, int(height - bh - 30)))

        poly = box(x0, y0, x0 + bw, y0 + bh)

        if poly.intersects(forbidden):
            continue

        if any(poly.intersects(existing) for existing in building_polys):
            continue

        building_polys.append(poly)

    buildings_gdf = gpd.GeoDataFrame(
        {
            "building_id": list(range(len(building_polys))),
            "height_m": rng.uniform(8.0, 55.0, size=len(building_polys)),
        },
        geometry=building_polys,
        crs=crs,
    )

    # Sensitive zones as circular polygons.
    sensitive_centers = [
        (340, 840),
        (520, 600),
        (760, 620),
        (680, 270),
        (490, 370),
    ]

    sensitive_polys = [
        Point(cx, cy).buffer(radius)
        for (cx, cy), radius in zip(sensitive_centers, [52, 36, 48, 46, 40])
    ]

    sensitive_gdf = gpd.GeoDataFrame(
        {
            "sensitive_id": list(range(len(sensitive_polys))),
            "type": ["school", "hospital", "playground", "school", "hospital"],
        },
        geometry=sensitive_polys,
        crs=crs,
    )

    # ------------------------------------------------------------
    # 3. Rasterize for DSM creation
    # ------------------------------------------------------------
    building_shapes = [
        (geom, float(h))
        for geom, h in zip(buildings_gdf.geometry, buildings_gdf["height_m"])
    ]

    building_height = rasterize(
        building_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype="float32",
        all_touched=True,
    )

    green_shapes = [(geom, 1.0) for geom in green_gdf.geometry]

    green_mask = rasterize(
        green_shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0.0,
        dtype="float32",
        all_touched=True,
    )

    # Synthetic canopy: only parts of green areas have trees.
    canopy_noise = rng.random((height, width)).astype(np.float32)
    canopy_mask = (green_mask > 0.5) & (canopy_noise > 0.35)

    canopy_height = np.zeros((height, width), dtype=np.float32)
    canopy_height[canopy_mask] = rng.uniform(7.0, 24.0, size=int(canopy_mask.sum()))

    # DSM = DEM + buildings + canopy
    dsm = dem + np.maximum(building_height, canopy_height)
    dsm = dsm.astype(np.float32)

    # ------------------------------------------------------------
    # 4. Write rasters
    # ------------------------------------------------------------
    dem_path = output_dir / "dem.tif"
    dsm_path = output_dir / "dsm.tif"

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "compress": "lzw",
    }

    with rasterio.open(dem_path, "w", **profile) as dst:
        dst.write(dem, 1)

    with rasterio.open(dsm_path, "w", **profile) as dst:
        dst.write(dsm, 1)

    # ------------------------------------------------------------
    # 5. Write vectors
    # ------------------------------------------------------------
    buildings_path = output_dir / "buildings.gpkg"
    roads_path = output_dir / "roads.gpkg"
    green_path = output_dir / "green.gpkg"
    sensitive_path = output_dir / "sensitive_zones.gpkg"

    buildings_gdf.to_file(buildings_path, driver="GPKG", layer="buildings")
    roads_gdf.to_file(roads_path, driver="GPKG", layer="roads")
    green_gdf.to_file(green_path, driver="GPKG", layer="green")
    sensitive_gdf.to_file(sensitive_path, driver="GPKG", layer="sensitive_zones")

    paths = {
        "dem_path": str(dem_path),
        "dsm_path": str(dsm_path),
        "buildings_path": str(buildings_path),
        "roads_path": str(roads_path),
        "green_path": str(green_path),
        "sensitive_path": str(sensitive_path),
        "bbox": (minx, miny, maxx, maxy),
        "crs": crs,
        "resolution_m": resolution_m,
    }

    print("=" * 80)
    print("Synthetic GIS files created")
    print("=" * 80)

    for k, v in paths.items():
        print(f"{k}: {v}")

    print("=" * 80)
    print(f"Buildings: {len(buildings_gdf)}")
    print(f"Roads:     {len(roads_gdf)}")
    print(f"Green:     {len(green_gdf)}")
    print(f"Sensitive: {len(sensitive_gdf)}")
    print("=" * 80)

    return paths

def build_m1_fake_gis_world(
    root="/content/ruscuav",
    fake_data_dir="/content/ruscuav/data/fake_gis/AOI_FAKE_001",
    aoi_id="AOI_FAKE_001",
    seed=42,
):
    """
    End-to-end M1 test:
        synthetic GIS files → UrbanGISDataAdapter → UrbanWorldLayers → diagnostics → save.
    """
    root = setup_ruscuav_project(root)
    fake_data_dir = Path(fake_data_dir)
    fake_data_dir.mkdir(parents=True, exist_ok=True)

    if "create_synthetic_gis_files_for_v15" not in globals():
        raise NameError(
            "create_synthetic_gis_files_for_v15 is not defined. "
            "Run the synthetic GIS generator block first."
        )

    if "UrbanGISDataAdapter" not in globals():
        raise NameError(
            "UrbanGISDataAdapter is not defined. "
            "Run the UrbanGISDataAdapter block first."
        )

    # 1. Create fake but valid GIS files.
    synthetic_paths = create_synthetic_gis_files_for_v15(
        output_dir=str(fake_data_dir),
        width=1000,
        height=1000,
        resolution_m=1.0,
        crs="EPSG:2180",
        seed=seed,
    )

    # 2. Create manifest.
    manifest = AOIManifestEntry(
        aoi_id=aoi_id,
        name="synthetic_fake_gis_urban_1km",
        crs=synthetic_paths["crs"],
        bbox=tuple(synthetic_paths["bbox"]),
        resolution_m=float(synthetic_paths["resolution_m"]),
        dem_path=synthetic_paths["dem_path"],
        dsm_path=synthetic_paths["dsm_path"],
        buildings_path=synthetic_paths["buildings_path"],
        roads_path=synthetic_paths["roads_path"],
        sidewalks_path=None,
        green_path=synthetic_paths["green_path"],
        sensitive_path=synthetic_paths["sensitive_path"],
        notes="Synthetic but valid GIS files for RUSCUAV pipeline validation.",
    )

    manifest_path = root / "manifests" / f"{aoi_id}.json"
    save_aoi_manifest(manifest, manifest_path)

    # 3. Build raster layers through adapter.
    adapter = UrbanGISDataAdapter(
        bbox=manifest.bbox,
        resolution_m=manifest.resolution_m,
        target_crs=manifest.crs,
        dem_path=manifest.dem_path,
        dsm_path=manifest.dsm_path,
        buildings_path=manifest.buildings_path,
        roads_path=manifest.roads_path,
        sidewalks_path=manifest.sidewalks_path,
        green_path=manifest.green_path,
        sensitive_path=manifest.sensitive_path,
        road_buffer_m=7.0,
        sidewalk_buffer_m=6.0,
    )

    adapter_layers = adapter.build_layers()

    # 4. Convert to canonical UrbanWorldLayers.
    world = UrbanWorldLayers.from_adapter_layers(
        layers=adapter_layers,
        aoi_id=manifest.aoi_id,
        crs=manifest.crs,
        bbox=manifest.bbox,
        resolution_m=manifest.resolution_m,
    )

    # 5. Save.
    processed_dir = root / "data" / "processed" / aoi_id
    processed_dir.mkdir(parents=True, exist_ok=True)

    world_path = processed_dir / "urban_world_layers.npz"
    world.save_npz(world_path)

    # 6. Gate checks.
    gate = run_world_gate_checks(world, risk_profile="balanced")

    return {
        "root": str(root),
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "synthetic_paths": synthetic_paths,
        "world": world,
        "world_path": str(world_path),
        "gate": gate,
    }

from dataclasses import dataclass

import numpy as np

@dataclass
class UrbanRiskProfile:
    """
    Risk profile for semantic urban UAV navigation.

    We distinguish:
        - semantic exposure costs,
        - altitude decay scales,
        - physical-clearance thresholds,
        - hard constraint thresholds.

    All costs are dimensionless per-step risk components.
    """
    name: str = "balanced"

    # Semantic class weights
    green_weight: float = 0.10
    canopy_weight: float = 0.20
    building_weight: float = 3.00
    road_weight: float = 6.00
    sidewalk_weight: float = 12.00
    sensitive_weight: float = 50.00

    # Altitude decay scales [m]
    # Larger value means risk decays more slowly with altitude.
    green_h_scale_m: float = 35.0
    road_h_scale_m: float = 45.0
    sidewalk_h_scale_m: float = 55.0
    sensitive_h_scale_m: float = 80.0
    building_clearance_h_scale_m: float = 35.0
    canopy_clearance_h_scale_m: float = 20.0

    # Clearance thresholds [m]
    terrain_min_clearance_m: float = 2.0
    building_min_clearance_m: float = 5.0
    canopy_min_clearance_m: float = 3.0

    # Sensitive zones may be overflown only above this altitude if hard mode is active
    sensitive_min_overflight_agl_m: float = 80.0

    # Preferred altitude band [m AGL]
    preferred_agl_low_m: float = 20.0
    preferred_agl_high_m: float = 80.0

    # Penalties for globally too low / too high flight
    low_altitude_weight: float = 2.0
    high_altitude_weight: float = 0.50

    # Multipliers for hard physical intrusions
    building_collision_multiplier: float = 8.0
    canopy_collision_multiplier: float = 4.0
    terrain_collision_multiplier: float = 10.0
    sensitive_low_altitude_multiplier: float = 5.0

RUSCUAV_RISK_PROFILES_M2 = {
    "efficiency_first": UrbanRiskProfile(
        name="efficiency_first",
        green_weight=0.05,
        canopy_weight=0.10,
        building_weight=1.50,
        road_weight=3.00,
        sidewalk_weight=6.00,
        sensitive_weight=30.00,
        preferred_agl_low_m=15.0,
        preferred_agl_high_m=70.0,
        high_altitude_weight=0.35,
    ),
    "balanced": UrbanRiskProfile(
        name="balanced",
        green_weight=0.10,
        canopy_weight=0.20,
        building_weight=3.00,
        road_weight=6.00,
        sidewalk_weight=12.00,
        sensitive_weight=50.00,
        preferred_agl_low_m=20.0,
        preferred_agl_high_m=80.0,
        high_altitude_weight=0.50,
    ),
    "safety_first": UrbanRiskProfile(
        name="safety_first",
        green_weight=0.05,
        canopy_weight=0.10,
        building_weight=5.00,
        road_weight=10.00,
        sidewalk_weight=20.00,
        sensitive_weight=100.00,
        road_h_scale_m=60.0,
        sidewalk_h_scale_m=75.0,
        sensitive_h_scale_m=100.0,
        sensitive_min_overflight_agl_m=100.0,
        preferred_agl_low_m=30.0,
        preferred_agl_high_m=100.0,
        high_altitude_weight=0.70,
    ),
}

class UrbanRiskModel:
    """
    Altitude-aware semantic risk model for RUSCUAV.

    Coordinate convention:
        x, y are local AOI coordinates in meters, starting at (0, 0).
        For resolution_m=1.0, x/y correspond directly to raster column/row.

    Main distinction:
        semantic class != physical obstacle != risk field

    Examples:
        - road: high social/traffic exposure, not a physical obstacle;
        - sidewalk: critical pedestrian exposure, not a physical obstacle;
        - building: semantic risk + physical obstacle;
        - canopy: low/moderate semantic risk + clearance constraint;
        - sensitive: soft or hard no-fly depending on altitude and experiment mode.
    """

    def __init__(
        self,
        world,
        profile="balanced",
        hard_sensitive_zones: bool = True,
        include_global_altitude_band_cost: bool = True,
    ):
        self.world = world

        if isinstance(profile, str):
            if profile not in RUSCUAV_RISK_PROFILES_M2:
                raise ValueError(
                    f"Unknown risk profile '{profile}'. "
                    f"Available: {list(RUSCUAV_RISK_PROFILES_M2.keys())}"
                )
            self.profile = RUSCUAV_RISK_PROFILES_M2[profile]
        elif isinstance(profile, UrbanRiskProfile):
            self.profile = profile
        else:
            raise TypeError("profile must be str or UrbanRiskProfile")

        self.hard_sensitive_zones = bool(hard_sensitive_zones)
        self.include_global_altitude_band_cost = bool(include_global_altitude_band_cost)

    # ------------------------------------------------------------
    # Coordinate and layer helpers
    # ------------------------------------------------------------
    def _ij_from_xy(self, x, y):
        res = float(self.world.resolution_m)

        xi = int(np.clip(np.floor(float(x) / res), 0, self.world.width - 1))
        yi = int(np.clip(np.floor(float(y) / res), 0, self.world.height - 1))

        return yi, xi

    def terrain_at(self, x, y):
        yi, xi = self._ij_from_xy(x, y)
        return float(self.world.terrain_z[yi, xi])

    def z_agl_at(self, x, y, z_abs):
        return float(z_abs - self.terrain_at(x, y))

    def flags_at(self, x, y):
        yi, xi = self._ij_from_xy(x, y)

        return {
            "building": bool(self.world.building_mask[yi, xi] > 0.5),
            "road": bool(self.world.road_mask[yi, xi] > 0.5),
            "sidewalk": bool(self.world.sidewalk_mask[yi, xi] > 0.5),
            "green": bool(self.world.green_mask[yi, xi] > 0.5),
            "canopy": bool(self.world.canopy_mask[yi, xi] > 0.5),
            "sensitive": bool(self.world.sensitive_mask[yi, xi] > 0.5),
        }

    def heights_at(self, x, y):
        yi, xi = self._ij_from_xy(x, y)

        return {
            "terrain_z": float(self.world.terrain_z[yi, xi]),
            "surface_z": float(self.world.surface_z[yi, xi]),
            "height_above_ground": float(self.world.height_above_ground[yi, xi]),
            "building_height": float(self.world.building_height[yi, xi]),
            "building_roof_z": float(self.world.building_roof_z[yi, xi]),
            "canopy_height": float(self.world.canopy_height[yi, xi]),
            "canopy_top_z": float(self.world.canopy_top_z[yi, xi]),
            "canopy_density": float(self.world.canopy_density[yi, xi]),
        }

    # ------------------------------------------------------------
    # Altitude and clearance helpers
    # ------------------------------------------------------------
    @staticmethod
    def _decay(value_m, scale_m):
        value_m = np.maximum(value_m, 0.0)
        scale_m = max(float(scale_m), 1e-6)
        return np.exp(-value_m / scale_m)

    def clearances_at(self, x, y, z_abs):
        h = self.heights_at(x, y)
        flags = self.flags_at(x, y)

        terrain_clearance = float(z_abs - h["terrain_z"])

        if flags["building"]:
            building_clearance = float(z_abs - h["building_roof_z"])
        else:
            building_clearance = 1e9

        if flags["canopy"]:
            canopy_clearance = float(z_abs - h["canopy_top_z"])
        else:
            canopy_clearance = 1e9

        return {
            "terrain_clearance_m": terrain_clearance,
            "building_clearance_m": building_clearance,
            "canopy_clearance_m": canopy_clearance,
        }

    def constraints_at(self, x, y, z_abs):
        """
        Hard and semi-hard constraints at a point.
        """
        p = self.profile
        flags = self.flags_at(x, y)
        clear = self.clearances_at(x, y, z_abs)
        z_agl = self.z_agl_at(x, y, z_abs)

        terrain_collision = clear["terrain_clearance_m"] <= p.terrain_min_clearance_m

        building_collision = (
            flags["building"]
            and clear["building_clearance_m"] <= 0.0
        )

        building_clearance_violation = (
            flags["building"]
            and clear["building_clearance_m"] < p.building_min_clearance_m
        )

        canopy_collision = (
            flags["canopy"]
            and clear["canopy_clearance_m"] <= 0.0
        )

        canopy_clearance_violation = (
            flags["canopy"]
            and clear["canopy_clearance_m"] < p.canopy_min_clearance_m
        )

        sensitive_low_altitude_violation = (
            self.hard_sensitive_zones
            and flags["sensitive"]
            and z_agl < p.sensitive_min_overflight_agl_m
        )

        return {
            "terrain_collision": bool(terrain_collision),
            "building_collision": bool(building_collision),
            "building_clearance_violation": bool(building_clearance_violation),
            "canopy_collision": bool(canopy_collision),
            "canopy_clearance_violation": bool(canopy_clearance_violation),
            "sensitive_low_altitude_violation": bool(sensitive_low_altitude_violation),
            "any_hard_violation": bool(
                terrain_collision
                or building_collision
                or canopy_collision
                or sensitive_low_altitude_violation
            ),
        }

    # ------------------------------------------------------------
    # Risk components at one point
    # ------------------------------------------------------------
    def cost_components_at(self, x, y, z_abs):
        p = self.profile
        flags = self.flags_at(x, y)
        heights = self.heights_at(x, y)
        clear = self.clearances_at(x, y, z_abs)
        constraints = self.constraints_at(x, y, z_abs)

        z_agl = max(0.0, self.z_agl_at(x, y, z_abs))

        # Road and sidewalk: social exposure decreasing with altitude.
        road_cost = (
            p.road_weight * self._decay(z_agl, p.road_h_scale_m)
            if flags["road"] else 0.0
        )

        sidewalk_cost = (
            p.sidewalk_weight * self._decay(z_agl, p.sidewalk_h_scale_m)
            if flags["sidewalk"] else 0.0
        )

        sensitive_cost = (
            p.sensitive_weight * self._decay(z_agl, p.sensitive_h_scale_m)
            if flags["sensitive"] else 0.0
        )

        if constraints["sensitive_low_altitude_violation"]:
            sensitive_cost += p.sensitive_weight * p.sensitive_low_altitude_multiplier

        # Building: not just altitude AGL, but clearance above roof.
        if flags["building"]:
            b_clear = clear["building_clearance_m"]

            if b_clear <= 0.0:
                building_cost = p.building_weight * p.building_collision_multiplier
            else:
                building_cost = p.building_weight * self._decay(
                    b_clear,
                    p.building_clearance_h_scale_m,
                )

            if constraints["building_clearance_violation"]:
                building_cost += p.building_weight
        else:
            building_cost = 0.0

        # Canopy: risk depends on clearance above canopy top and density.
        if flags["canopy"]:
            c_clear = clear["canopy_clearance_m"]
            density = heights["canopy_density"]

            if c_clear <= 0.0:
                canopy_cost = p.canopy_weight * p.canopy_collision_multiplier * max(0.1, density)
            else:
                canopy_cost = p.canopy_weight * max(0.1, density) * self._decay(
                    c_clear,
                    p.canopy_clearance_h_scale_m,
                )

            if constraints["canopy_clearance_violation"]:
                canopy_cost += p.canopy_weight * max(0.1, density)
        else:
            canopy_cost = 0.0

        # Green: low residual risk / operational preference corridor.
        # This is a small positive cost, not a reward.
        # Rewards/preference can be handled separately in the environment.
        green_clean = (
            flags["green"]
            and not flags["road"]
            and not flags["sidewalk"]
            and not flags["building"]
            and not flags["sensitive"]
        )

        green_cost = (
            p.green_weight * self._decay(z_agl, p.green_h_scale_m)
            if green_clean else 0.0
        )

        # Terrain / altitude band cost
        if constraints["terrain_collision"]:
            terrain_cost = p.low_altitude_weight * p.terrain_collision_multiplier
        else:
            terrain_cost = 0.0

        if self.include_global_altitude_band_cost:
            if z_agl < p.preferred_agl_low_m:
                low_altitude_cost = p.low_altitude_weight * (
                    (p.preferred_agl_low_m - z_agl) / max(1e-6, p.preferred_agl_low_m)
                ) ** 2
            else:
                low_altitude_cost = 0.0

            if z_agl > p.preferred_agl_high_m:
                high_altitude_cost = p.high_altitude_weight * (
                    (z_agl - p.preferred_agl_high_m) / max(1e-6, p.preferred_agl_high_m)
                ) ** 2
            else:
                high_altitude_cost = 0.0
        else:
            low_altitude_cost = 0.0
            high_altitude_cost = 0.0

        components = {
            "road_cost": float(road_cost),
            "sidewalk_cost": float(sidewalk_cost),
            "building_cost": float(building_cost),
            "green_cost": float(green_cost),
            "canopy_cost": float(canopy_cost),
            "sensitive_cost": float(sensitive_cost),
            "terrain_cost": float(terrain_cost),
            "low_altitude_cost": float(low_altitude_cost),
            "high_altitude_cost": float(high_altitude_cost),
        }

        components["total_risk"] = float(sum(components.values()))

        return components

    def risk_at(self, x, y, z_abs):
        return float(self.cost_components_at(x, y, z_abs)["total_risk"])

    # ------------------------------------------------------------
    # Vectorized risk maps at constant AGL
    # ------------------------------------------------------------
    def component_maps_at_agl(self, agl_m):
        """
        Risk component maps at a constant altitude above terrain.

        This is useful for diagnostics and planner cost maps.
        """
        p = self.profile
        agl = float(agl_m)

        z_abs = self.world.terrain_z + agl

        road = self.world.road_mask
        sidewalk = self.world.sidewalk_mask
        building = self.world.building_mask
        green = self.world.green_mask
        canopy = self.world.canopy_mask
        sensitive = self.world.sensitive_mask

        # Road/sidewalk/sensitive
        road_cost = p.road_weight * road * self._decay(agl, p.road_h_scale_m)
        sidewalk_cost = p.sidewalk_weight * sidewalk * self._decay(agl, p.sidewalk_h_scale_m)
        sensitive_cost = p.sensitive_weight * sensitive * self._decay(agl, p.sensitive_h_scale_m)

        if self.hard_sensitive_zones and agl < p.sensitive_min_overflight_agl_m:
            sensitive_cost = sensitive_cost + sensitive * p.sensitive_weight * p.sensitive_low_altitude_multiplier

        # Building clearance relative to roof
        building_clearance = agl - self.world.building_height

        building_cost = np.zeros(self.world.shape, dtype=np.float32)
        b_active = building > 0.5

        b_collision = b_active & (building_clearance <= 0.0)
        b_clearance_bad = b_active & (building_clearance > 0.0) & (building_clearance < p.building_min_clearance_m)
        b_ok = b_active & (building_clearance >= p.building_min_clearance_m)

        building_cost[b_collision] = p.building_weight * p.building_collision_multiplier
        building_cost[b_clearance_bad] = (
            p.building_weight
            + p.building_weight * self._decay(
                building_clearance[b_clearance_bad],
                p.building_clearance_h_scale_m,
            )
        )
        building_cost[b_ok] = p.building_weight * self._decay(
            building_clearance[b_ok],
            p.building_clearance_h_scale_m,
        )

        # Canopy clearance relative to canopy top
        canopy_clearance = agl - self.world.canopy_height
        density = np.maximum(0.1, self.world.canopy_density)

        canopy_cost = np.zeros(self.world.shape, dtype=np.float32)
        c_active = canopy > 0.5

        c_collision = c_active & (canopy_clearance <= 0.0)
        c_clearance_bad = c_active & (canopy_clearance > 0.0) & (canopy_clearance < p.canopy_min_clearance_m)
        c_ok = c_active & (canopy_clearance >= p.canopy_min_clearance_m)

        canopy_cost[c_collision] = (
            p.canopy_weight
            * p.canopy_collision_multiplier
            * density[c_collision]
        )
        canopy_cost[c_clearance_bad] = (
            p.canopy_weight * density[c_clearance_bad]
            + p.canopy_weight
            * density[c_clearance_bad]
            * self._decay(canopy_clearance[c_clearance_bad], p.canopy_clearance_h_scale_m)
        )
        canopy_cost[c_ok] = (
            p.canopy_weight
            * density[c_ok]
            * self._decay(canopy_clearance[c_ok], p.canopy_clearance_h_scale_m)
        )

        # Green clean area
        green_clean = (
            (green > 0.5)
            & (road <= 0.5)
            & (sidewalk <= 0.5)
            & (building <= 0.5)
            & (sensitive <= 0.5)
        )

        green_cost = np.zeros(self.world.shape, dtype=np.float32)
        green_cost[green_clean] = p.green_weight * self._decay(agl, p.green_h_scale_m)

        # Global altitude band
        low_altitude_cost = np.zeros(self.world.shape, dtype=np.float32)
        high_altitude_cost = np.zeros(self.world.shape, dtype=np.float32)

        if self.include_global_altitude_band_cost:
            if agl < p.preferred_agl_low_m:
                low = p.low_altitude_weight * (
                    (p.preferred_agl_low_m - agl) / max(1e-6, p.preferred_agl_low_m)
                ) ** 2
                low_altitude_cost[:] = low

            if agl > p.preferred_agl_high_m:
                high = p.high_altitude_weight * (
                    (agl - p.preferred_agl_high_m) / max(1e-6, p.preferred_agl_high_m)
                ) ** 2
                high_altitude_cost[:] = high

        maps = {
            "road_cost": road_cost.astype(np.float32),
            "sidewalk_cost": sidewalk_cost.astype(np.float32),
            "building_cost": building_cost.astype(np.float32),
            "green_cost": green_cost.astype(np.float32),
            "canopy_cost": canopy_cost.astype(np.float32),
            "sensitive_cost": sensitive_cost.astype(np.float32),
            "low_altitude_cost": low_altitude_cost.astype(np.float32),
            "high_altitude_cost": high_altitude_cost.astype(np.float32),
        }

        total = np.zeros(self.world.shape, dtype=np.float32)
        for v in maps.values():
            total += v.astype(np.float32)

        maps["total_risk"] = total.astype(np.float32)

        return maps

    def risk_map_at_agl(self, agl_m):
        return self.component_maps_at_agl(agl_m)["total_risk"]

    def normalized_risk_map_at_agl(self, agl_m):
        risk = self.risk_map_at_agl(agl_m)
        maxv = float(np.nanmax(risk))

        if maxv <= 1e-9:
            return risk

        return (risk / maxv).astype(np.float32)

import numpy as np

import matplotlib.pyplot as plt

def _sample_pixel_from_mask(mask, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)

    ys, xs = np.where(mask > 0.5)

    if len(xs) == 0:
        return None

    idx = int(rng.integers(0, len(xs)))
    return int(xs[idx]), int(ys[idx])

def print_risk_samples_m2(world, risk_model, seed=123):
    """
    Prints point risk samples for representative classes at multiple AGL levels.
    """
    rng = np.random.default_rng(seed)

    samples = {
        "road": _sample_pixel_from_mask(world.road_mask, rng),
        "sidewalk": _sample_pixel_from_mask(world.sidewalk_mask, rng),
        "building": _sample_pixel_from_mask(world.building_mask, rng),
        "green": _sample_pixel_from_mask(world.green_mask, rng),
        "canopy": _sample_pixel_from_mask(world.canopy_mask, rng),
        "sensitive": _sample_pixel_from_mask(world.sensitive_mask, rng),
    }

    agls = [10.0, 20.0, 40.0, 80.0, 120.0]

    print("=" * 100)
    print("M2 POINT RISK SAMPLES")
    print("=" * 100)

    for label, pt in samples.items():
        if pt is None:
            print(f"{label:10s}: no active pixels")
            continue

        x, y = pt
        terrain = risk_model.terrain_at(x, y)

        print("-" * 100)
        print(f"{label.upper()} sample at x={x}, y={y}, terrain_z={terrain:.2f}")
        print("flags:", risk_model.flags_at(x, y))
        print("heights:", risk_model.heights_at(x, y))

        for agl in agls:
            z_abs = terrain + agl

            comps = risk_model.cost_components_at(x, y, z_abs)
            cons = risk_model.constraints_at(x, y, z_abs)

            print(
                f"AGL={agl:6.1f} | risk={comps['total_risk']:9.3f} | "
                f"road={comps['road_cost']:7.3f} | "
                f"sidewalk={comps['sidewalk_cost']:7.3f} | "
                f"building={comps['building_cost']:7.3f} | "
                f"canopy={comps['canopy_cost']:7.3f} | "
                f"sensitive={comps['sensitive_cost']:7.3f} | "
                f"low_alt={comps['low_altitude_cost']:6.3f} | "
                f"high_alt={comps['high_altitude_cost']:6.3f} | "
                f"hard={cons['any_hard_violation']}"
            )

    print("=" * 100)

def summarize_risk_maps_m2(world, risk_model, agls=(10, 20, 40, 80, 120)):
    """
    Summarizes total risk maps at several altitudes.
    """
    print("=" * 100)
    print("M2 RISK MAP SUMMARY")
    print("=" * 100)
    print(f"profile: {risk_model.profile.name}")
    print("-" * 100)

    for agl in agls:
        risk = risk_model.risk_map_at_agl(float(agl))

        print(
            f"AGL={float(agl):6.1f} | "
            f"min={np.nanmin(risk):9.3f} | "
            f"mean={np.nanmean(risk):9.3f} | "
            f"median={np.nanmedian(risk):9.3f} | "
            f"p90={np.nanquantile(risk, 0.90):9.3f} | "
            f"max={np.nanmax(risk):9.3f}"
        )

    print("=" * 100)

def _mean_on_mask(arr, mask):
    vals = arr[mask > 0.5]
    if vals.size == 0:
        return np.nan
    return float(np.nanmean(vals))

def m2_gate_checks(world, risk_model):
    """
    Gate conditions:
        1. risk maps finite and non-negative;
        2. road/sidewalk/sensitive risk decreases with altitude;
        3. building risk high at low altitude over buildings;
        4. canopy risk high below/near canopy and lower above canopy;
        5. constraints detect obvious building/canopy intrusions where applicable.
    """
    print("=" * 100)
    print("M2 GATE CHECKS")
    print("=" * 100)

    passed = True
    messages = []

    agl_low = 20.0
    agl_high = 100.0

    risk_low = risk_model.risk_map_at_agl(agl_low)
    risk_high = risk_model.risk_map_at_agl(agl_high)

    # 1. finite and nonnegative
    finite_ok = np.isfinite(risk_low).all() and np.isfinite(risk_high).all()
    nonneg_ok = (np.nanmin(risk_low) >= -1e-9) and (np.nanmin(risk_high) >= -1e-9)

    if not finite_ok:
        passed = False
        messages.append("Risk maps contain non-finite values.")
    if not nonneg_ok:
        passed = False
        messages.append("Risk maps contain negative values.")

    print(f"[1] finite risk maps:     {'PASS' if finite_ok else 'FAIL'}")
    print(f"[2] nonnegative risk:     {'PASS' if nonneg_ok else 'FAIL'}")

    # 2. altitude decay on social exposure masks
    for label, mask in [
        ("road", world.road_mask),
        ("sidewalk", world.sidewalk_mask),
        ("sensitive", world.sensitive_mask),
    ]:
        if np.sum(mask > 0.5) == 0:
            print(f"[altitude decay] {label:10s}: SKIP no active pixels")
            continue

        mean_low = _mean_on_mask(risk_low, mask)
        mean_high = _mean_on_mask(risk_high, mask)

        ok = mean_high < mean_low

        print(
            f"[altitude decay] {label:10s}: "
            f"low={mean_low:9.3f}, high={mean_high:9.3f} | "
            f"{'PASS' if ok else 'CHECK'}"
        )

        if not ok:
            passed = False
            messages.append(f"{label} risk does not decrease with altitude.")

    # 3. Building risk and constraints
    building_pt = _sample_pixel_from_mask(world.building_mask)

    if building_pt is not None:
        x, y = building_pt
        terrain = risk_model.terrain_at(x, y)
        heights = risk_model.heights_at(x, y)

        # Put UAV inside the building if building_height allows it.
        z_inside = terrain + max(1.0, 0.5 * heights["building_height"])
        cons_inside = risk_model.constraints_at(x, y, z_inside)
        risk_inside = risk_model.risk_at(x, y, z_inside)

        ok = cons_inside["building_collision"] and risk_inside > 0.0

        print(
            f"[building collision] risk={risk_inside:9.3f} | "
            f"building_collision={cons_inside['building_collision']} | "
            f"{'PASS' if ok else 'CHECK'}"
        )

        if not ok:
            passed = False
            messages.append("Building collision not detected or no risk assigned.")
    else:
        print("[building collision] SKIP no buildings")

    # 4. Canopy risk and constraints
    canopy_pt = _sample_pixel_from_mask(world.canopy_mask)

    if canopy_pt is not None:
        x, y = canopy_pt
        terrain = risk_model.terrain_at(x, y)
        heights = risk_model.heights_at(x, y)

        z_inside = terrain + max(0.5, 0.5 * heights["canopy_height"])
        cons_inside = risk_model.constraints_at(x, y, z_inside)
        risk_inside = risk_model.risk_at(x, y, z_inside)

        z_above = terrain + heights["canopy_height"] + 20.0
        risk_above = risk_model.risk_at(x, y, z_above)

        ok = cons_inside["canopy_collision"] and risk_above < risk_inside

        print(
            f"[canopy collision] inside_risk={risk_inside:9.3f}, above_risk={risk_above:9.3f} | "
            f"canopy_collision={cons_inside['canopy_collision']} | "
            f"{'PASS' if ok else 'CHECK'}"
        )

        if not ok:
            passed = False
            messages.append("Canopy collision/risk behavior failed.")
    else:
        print("[canopy collision] SKIP no canopy")

    print("-" * 100)
    print("M2 GATE:", "PASS" if passed else "CHECK REQUIRED")

    if messages:
        print("Messages:")
        for m in messages:
            print(" -", m)

    print("=" * 100)

    return {
        "passed": bool(passed),
        "messages": messages,
        "risk_low": risk_low,
        "risk_high": risk_high,
    }

def plot_risk_maps_by_altitude_m2(
    world,
    risk_model,
    agls=(10, 20, 40, 80, 120),
    figsize=(6, 5),
):
    """
    One figure per altitude. Keeps charts separate and readable.
    """
    for agl in agls:
        risk = risk_model.risk_map_at_agl(float(agl))

        plt.figure(figsize=figsize)
        plt.imshow(risk, origin="lower", cmap="inferno")
        plt.colorbar(label="Total risk")
        plt.title(f"RUSCUAV risk map | AGL={float(agl):.1f} m | profile={risk_model.profile.name}")
        plt.xlabel("X grid")
        plt.ylabel("Y grid")
        plt.tight_layout()
        plt.show()

def plot_component_maps_m2(world, risk_model, agl_m=40.0, figsize=(6, 5)):
    """
    Plots each component map separately.
    """
    maps = risk_model.component_maps_at_agl(float(agl_m))

    for name, arr in maps.items():
        plt.figure(figsize=figsize)
        plt.imshow(arr, origin="lower", cmap="inferno")
        plt.colorbar(label=name)
        plt.title(f"{name} | AGL={float(agl_m):.1f} m")
        plt.xlabel("X grid")
        plt.ylabel("Y grid")
        plt.tight_layout()
        plt.show()

import os

import glob

import csv

import time

import random

import warnings

import heapq

import gymnasium as gym

from gymnasium import spaces

import numpy as np

import torch

from scipy.ndimage import gaussian_filter

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback

class RandomizedCurriculumEnvV14(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dem_data, include_shield_features: bool = False):
        super().__init__()
        self.dem_data = dem_data

        self.bounds_low = np.array([MIN_X, MIN_Y, MIN_Z], dtype=np.float32)
        self.bounds_high = np.array([MAX_X, MAX_Y, MAX_Z], dtype=np.float32)

        self.obstacles = []
        self.n_obstacles = 0

        self.ideal_vec_2d = np.array([1, 0], dtype=np.float32)
        self.ideal_len_2d = 1.0

        self.CORRIDOR_LIMIT = 100.0
        self.SOFT_LIMIT = 20.0

        # --- LIDAR ---
        self.lidar_range = 250.0
        self.lidar_angles = np.linspace(np.deg2rad(-135), np.deg2rad(135), 64).astype(np.float32)
        self.cached_lidar = np.zeros(64, dtype=np.float32)

        # --- Robustness toggles (default OFF) ---
        self.wind_vel_xy = np.zeros(2, dtype=np.float32)
        self.lidar_noise_std = 0.0

        self.safety_margin_xy = 1.5
        self.safety_margin_z = 1.0

        # Backoff w metrach dla safety shield
        self.shield_backoff_m = 0.10

        self.dt = 0.5
        self.v_max_xy = 18.0
        self.v_max_z = 6.0
        self.a_max_xy = 6.0
        self.a_max_z = 3.0
        self.jerk_max = 30.0

        self.alpha_xy = 0.2
        self.alpha_z = 0.2

        self.acc_xy_f = np.zeros(2, dtype=np.float32)
        self.prev_acc_xy = np.zeros(2, dtype=np.float32)
        self.acc_z_f = np.float32(0.0)
        self.prev_acc_z = np.float32(0.0)
        self.vel = np.zeros(3, dtype=np.float32)
        self.current_pos = None

        self.rel_h_cruise_m = 30.0
        self.rel_h_min_m = 2.0
        self.rel_h_max_m = 80.0
        self.kp_z = 0.35
        self.kd_z = 0.90

        self.g0 = 9.81
        self.overload_soft_g = 0.30
        self.overload_hard_g = 1.20
        self.hard_over_steps_to_fail = 6
        self.cargo_health = 1.0
        self.hard_over_counter = 0

        # --- REWARD BALANCING ---
        self.progress_scale = 12.0
        self.time_penalty = 0.1

        self.dist_k = 0.0
        self.dist_cap = 0.0

        self.cte_quad_k = 0.002
        self.cte_lin_k = 0.03
        self.cte_cap = 14.0
        self.lidar_k = 150.0
        self.lidar_trigger = 0.25
        self.lidar_cap = 30.0
        self.shield_penalty = 5.0
        self.alt_k = 0.04
        self.vz_k = 0.06

        self.acc_k = 0.001
        self.jerk_k = 0.001
        self.over_k = 80.0
        self.health_decay_k = 0.005

        self.align_k = 0.8

        self.speed_min = 2.0
        self.low_speed_penalty = 0.5

        self.success_radius_xy = 30.0
        self.success_bonus = 3000.0

        self.max_steps = 1000

        self.improve_eps_m = 0.5
        self.stuck_patience_steps = 300
        self.shield_stuck_patience_steps = 40
        self.shield_stuck_penalty = 300.0
        self.shielded_steps = 0
        self.shield_no_progress_steps = 0
        self.shield_progress_eps_m = 0.20
        self.shield_min_step_xy_m = 0.05

        self.collision_penalty = 500.0
        self.penetration_eps_m = 0.02
        self.penetration_patience_steps = 3
        self.penetration_counter = 0

        # near_success jako osobny terminal state
        self.near_success_radius_xy = 45.0
        self.near_success_bonus = 0.5 * self.success_bonus

        self.ep_over_g_sum = 0.0
        self.ep_rel_h_sum = 0.0
        self.ep_step_count = 0
        self.best_dist_xy = None
        self.steps_since_improve = 0

        # --- OBS: optional shield features ---
        self.include_shield_features = bool(include_shield_features)
        self.last_shielded = 0.0
        self.shield_clearance_progress_eps_m = 0.10

        # shaping: bonus for increasing clearance when shield active
        self.shield_clearance_bonus_k = 2.0
        self.shield_clearance_bonus_cap = 2.0

        # --- SHIELD SLIDE (deterministic contact exit) ---
        self.shield_slide_enabled = True
        self.shield_slide_max_m = 2.5
        self.shield_slide_min_m = 0.10
        self.shield_slide_backoff_iters = 8
        self.shield_slide_clearance_eps = 1e-3
        self.last_eff_acc = np.zeros(3, dtype=np.float32)

        # --- SCENARIO FEASIBILITY FILTER ---
        self.enable_scenario_feasibility_check = True
        self.scenario_grid_res_m = 10.0
        self.scenario_clearance_m = 6.0
        self.scenario_max_detour_ratio = 2.75
        self.scenario_nominal_xy_speed_mps = 12.0
        self.scenario_max_required_vz_ratio = 0.90
        self.scenario_max_required_az_ratio = 0.90
        self.scenario_profile_sample_step_m = 5.0

        # target_point[2] pozostaje tylko jako pole pomocnicze/kompatybilność,
        # ale logika zadania jest XY-only + autopilot wysokości
        start_ground = self._ground_at(50, 50)
        target_ground = self._ground_at(950, 950)

        self.start_point = np.array(
            [50, 50, min(MAX_Z, max(start_ground + self.rel_h_min_m, start_ground + self.rel_h_cruise_m))],
            dtype=np.float32,
        )
        self.target_point = np.array(
            [950, 950, min(MAX_Z, max(target_ground + self.rel_h_min_m, target_ground + self.rel_h_cruise_m))],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Wariant A: obserwacja celu tylko w XY
        # base_obs:
        # pos_norm(3) + rel_target_xy(2) + dist_xy_norm(1) + dir_to_target(2) +
        # speed_xy_norm(1) + heading_cos(1) + rel_h_norm(1) +
        # lidar(64) + vel_norm(3) + over_g_proxy(1) + cargo_health(1)
        # = 80
        obs_dim = 80 + (2 if self.include_shield_features else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def set_n_obstacles(self, n: int):
        self.n_obstacles = int(n)

    def _ground_at(self, x, y) -> float:
        xi = int(np.clip(x, 0, 999))
        yi = int(np.clip(y, 0, 999))
        return float(self.dem_data[yi, xi])

    def _reference_z_from_ground(self, ground_val: float):
        floor = ground_val + self.rel_h_min_m
        ceiling = min(MAX_Z, ground_val + self.rel_h_max_m)
        if floor > ceiling:
            return None
        target = ground_val + self.rel_h_cruise_m
        return float(np.clip(target, floor, ceiling))

    def _world_to_grid(self, xy):
        res = float(self.scenario_grid_res_m)
        nx = int(MAX_X / res)
        ny = int(MAX_Y / res)
        gx = int(np.clip(xy[0] / res, 0, nx - 1))
        gy = int(np.clip(xy[1] / res, 0, ny - 1))
        return gx, gy

    def _grid_to_world(self, ij):
        res = float(self.scenario_grid_res_m)
        gx, gy = ij
        x = min(MAX_X - 1e-3, (gx + 0.5) * res)
        y = min(MAX_Y - 1e-3, (gy + 0.5) * res)
        return np.array([x, y], dtype=np.float32)

    def _build_feasibility_occupancy(self, obstacles):
        res = float(self.scenario_grid_res_m)
        nx = int(MAX_X / res)
        ny = int(MAX_Y / res)

        blocked = np.zeros((ny, nx), dtype=bool)

        for obs in obstacles:
            center = np.array(obs[0], dtype=float)
            obs_r = float(obs[2])
            fatal_zone = float(obs[3])
            extra_margin = float(obs[4])

            r_block = obs_r + fatal_zone + self.safety_margin_xy + extra_margin + self.scenario_clearance_m

            min_x = max(0, int((center[0] - r_block) / res))
            max_x = min(nx - 1, int((center[0] + r_block) / res))
            min_y = max(0, int((center[1] - r_block) / res))
            max_y = min(ny - 1, int((center[1] + r_block) / res))

            for gy in range(min_y, max_y + 1):
                cy = (gy + 0.5) * res
                dy2 = (cy - center[1]) ** 2
                for gx in range(min_x, max_x + 1):
                    cx = (gx + 0.5) * res
                    dx2 = (cx - center[0]) ** 2
                    if dx2 + dy2 <= r_block ** 2:
                        blocked[gy, gx] = True

        return blocked

    def _astar_xy_path(self, start_xy, target_xy, obstacles):
        blocked = self._build_feasibility_occupancy(obstacles)

        start = self._world_to_grid(start_xy)
        goal = self._world_to_grid(target_xy)

        if blocked[start[1], start[0]] or blocked[goal[1], goal[0]]:
            return None

        neighbors = [
            (-1, -1), (0, -1), (1, -1),
            (-1,  0),          (1,  0),
            (-1,  1), (0,  1), (1,  1),
        ]
        res = float(self.scenario_grid_res_m)
        ny, nx = blocked.shape

        def heuristic(a, b):
            dx = (a[0] - b[0]) * res
            dy = (a[1] - b[1]) * res
            return float(np.hypot(dx, dy))

        open_heap = []
        heapq.heappush(open_heap, (heuristic(start, goal), 0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)

            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                grid_path = [current]
                while current in came_from:
                    current = came_from[current]
                    grid_path.append(current)
                grid_path.reverse()

                world_path = [start_xy.astype(np.float32)]
                for cell in grid_path[1:-1]:
                    world_path.append(self._grid_to_world(cell))
                world_path.append(target_xy.astype(np.float32))
                return np.array(world_path, dtype=np.float32)

            cx, cy = current

            for dx, dy in neighbors:
                nx_i, ny_i = cx + dx, cy + dy
                if nx_i < 0 or nx_i >= nx or ny_i < 0 or ny_i >= ny:
                    continue
                if blocked[ny_i, nx_i]:
                    continue

                # Dodatkowa ochrona przed przeciskaniem się po przekątnej przez narożniki
                if dx != 0 and dy != 0:
                    if blocked[cy, nx_i] or blocked[ny_i, cx]:
                        continue

                step_cost = float(np.hypot(dx * res, dy * res))
                neighbor = (nx_i, ny_i)
                tentative_g = g_curr + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))

        return None

    def _path_length_xy(self, path_xy):
        if path_xy is None or len(path_xy) < 2:
            return 0.0
        segs = np.diff(path_xy, axis=0)
        return float(np.sum(np.linalg.norm(segs, axis=1)))

    def _densify_xy_path(self, path_xy, step_m: float):
        if len(path_xy) <= 1:
            return path_xy.copy()

        dense = [path_xy[0].astype(np.float32)]
        step_m = float(max(1e-3, step_m))

        for i in range(len(path_xy) - 1):
            p0 = path_xy[i].astype(np.float32)
            p1 = path_xy[i + 1].astype(np.float32)
            d = float(np.linalg.norm(p1 - p0))
            if d < 1e-9:
                continue

            n_steps = max(1, int(np.ceil(d / step_m)))
            for k in range(1, n_steps + 1):
                a = k / n_steps
                pt = (1.0 - a) * p0 + a * p1
                dense.append(pt.astype(np.float32))

        return np.array(dense, dtype=np.float32)

    def _terrain_profile_is_feasible(self, path_xy):
        if path_xy is None or len(path_xy) < 2:
            return False

        dense_xy = self._densify_xy_path(path_xy, self.scenario_profile_sample_step_m)

        grounds = np.array([self._ground_at(p[0], p[1]) for p in dense_xy], dtype=np.float32)

        z_refs = []
        for g in grounds:
            z_ref = self._reference_z_from_ground(float(g))
            if z_ref is None:
                return False
            z_refs.append(z_ref)
        z_refs = np.array(z_refs, dtype=np.float32)

        # jeśli gdziekolwiek nie da się utrzymać minimalnego AGL przez limit MAX_Z,
        # _reference_z_from_ground zwróciłby None, ale zostawiamy też jawny test
        if np.any(grounds + self.rel_h_min_m > MAX_Z):
            return False

        segs = np.diff(dense_xy, axis=0)
        ds = np.linalg.norm(segs, axis=1).astype(np.float32)
        nominal_vxy = float(max(1.0, self.scenario_nominal_xy_speed_mps))
        dts = np.maximum(ds / nominal_vxy, 1e-3).astype(np.float32)

        dz = np.diff(z_refs).astype(np.float32)
        req_vz = dz / dts

        max_req_vz = float(self.scenario_max_required_vz_ratio * self.v_max_z)
        if np.any(np.abs(req_vz) > max_req_vz):
            return False

        if len(req_vz) >= 2:
            dt_mid = 0.5 * (dts[:-1] + dts[1:])
            dt_mid = np.maximum(dt_mid, 1e-3).astype(np.float32)
            req_az = np.diff(req_vz) / dt_mid

            max_req_az = float(self.scenario_max_required_az_ratio * self.a_max_z)
            if np.any(np.abs(req_az) > max_req_az):
                return False

        return True

    def _scenario_is_feasible(self, start_point, target_point, obstacles):
        if not self.enable_scenario_feasibility_check:
            return True

        start_xy = start_point[:2].astype(np.float32)
        target_xy = target_point[:2].astype(np.float32)

        straight_dist = float(np.linalg.norm(target_xy - start_xy))
        if straight_dist < 1e-6:
            return False

        path_xy = self._astar_xy_path(start_xy, target_xy, obstacles)
        if path_xy is None:
            return False

        path_len = self._path_length_xy(path_xy)
        detour_ratio = path_len / max(straight_dist, 1e-6)
        if detour_ratio > self.scenario_max_detour_ratio:
            return False

        if not self._terrain_profile_is_feasible(path_xy):
            return False

        return True

    def _generate_random_scenario(self):
        max_outer_attempts = 1000

        for _ in range(max_outer_attempts):
            s_x, s_y = self.np_random.integers(50, 950), self.np_random.integers(50, 950)
            t_x, t_y = self.np_random.integers(50, 950), self.np_random.integers(50, 950)

            ground_s = self._ground_at(s_x, s_y)
            ground_t = self._ground_at(t_x, t_y)

            min_start_z = ground_s + self.rel_h_min_m
            min_target_z = ground_t + self.rel_h_min_m

            if min_start_z > MAX_Z or min_target_z > MAX_Z:
                continue

            start_z = min(MAX_Z, ground_s + self.rel_h_cruise_m)
            target_z = min(MAX_Z, ground_t + self.rel_h_cruise_m)

            start_z = max(start_z, min_start_z)
            target_z = max(target_z, min_target_z)

            p1 = np.array([s_x, s_y, start_z], dtype=np.float32)
            p2 = np.array([t_x, t_y, target_z], dtype=np.float32)

            dist = np.linalg.norm(p1[:2] - p2[:2])
            if dist <= 500.0:
                continue

            candidate_obstacles = []
            attempts = 0
            max_obstacle_attempts = max(400, self.n_obstacles * 60)

            while len(candidate_obstacles) < self.n_obstacles and attempts < max_obstacle_attempts:
                attempts += 1

                ox, oy = self.np_random.integers(100, 900), self.np_random.integers(100, 900)
                d_start = np.linalg.norm(np.array([ox, oy]) - p1[:2])
                d_target = np.linalg.norm(np.array([ox, oy]) - p2[:2])

                if d_start < 80.0 or d_target < 80.0:
                    continue

                h = self.np_random.integers(80, 140)
                r = self.np_random.integers(30, 80)
                fatal = 5
                margin = self.np_random.integers(3, 8)
                otype = "cylinder" if self.np_random.random() > 0.5 else "cone"

                candidate_obstacles.append([[ox, oy], h, r, fatal, margin, otype])

            if len(candidate_obstacles) != self.n_obstacles:
                continue

            if not self._scenario_is_feasible(p1, p2, candidate_obstacles):
                continue

            self.start_point = p1
            self.target_point = p2
            self.obstacles = candidate_obstacles
            self.ideal_vec_2d = (self.target_point[:2] - self.start_point[:2]).astype(np.float32)
            self.ideal_len_2d = float(np.linalg.norm(self.ideal_vec_2d) + 1e-9)
            return

        raise RuntimeError("Could not generate a valid feasible random scenario with the requested number of obstacles.")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._generate_random_scenario()

        self.current_pos = self.start_point.copy()
        self.vel[:] = 0.0
        self.acc_xy_f[:] = 0.0
        self.prev_acc_xy[:] = 0.0
        self.acc_z_f = np.float32(0.0)
        self.prev_acc_z = np.float32(0.0)
        self.cargo_health = 1.0
        self.hard_over_counter = 0
        self.current_step = 0
        self.ep_over_g_sum = 0.0
        self.ep_rel_h_sum = 0.0
        self.ep_step_count = 0
        self.shielded_steps = 0
        self.shield_no_progress_steps = 0
        self.penetration_counter = 0
        self.best_dist_xy = self._dist_xy(self.current_pos)
        self.steps_since_improve = 0
        self.last_shielded = 0.0
        self.last_eff_acc[:] = 0.0

        self.cached_lidar = self._compute_lidar_rays()

        return self._get_obs(), {}

    def _rel_h_m(self, pos) -> float:
        return float(pos[2] - self._ground_at(pos[0], pos[1]))

    def _dist_xy(self, pos) -> float:
        d = self.target_point[:2] - pos[:2]
        return float(np.linalg.norm(d))

    def _cross_track_error(self) -> float:
        p = self.current_pos[:2] - self.start_point[:2]
        cross = float(p[0] * self.ideal_vec_2d[1] - p[1] * self.ideal_vec_2d[0])
        return abs(cross) / self.ideal_len_2d

    def _effective_radius(self, obstacle, z_abs: float) -> float:
        center = np.array(obstacle[0], dtype=float)
        obs_h = float(obstacle[1])
        obs_r = float(obstacle[2])
        fatal_zone = float(obstacle[3])
        obs_type = obstacle[5]
        ground = self._ground_at(center[0], center[1])
        z_rel = z_abs - ground

        if z_rel > obs_h + self.safety_margin_z:
            return 0.0

        if obs_type == "cylinder":
            return obs_r + fatal_zone + self.safety_margin_xy

        if z_rel <= 0.0:
            r_at_z = obs_r
        elif z_rel >= obs_h:
            r_at_z = 0.0
        else:
            r_at_z = obs_r * (1.0 - z_rel / obs_h)

        return r_at_z + fatal_zone + self.safety_margin_xy

    def _ray_circle_hit_distance(self, origin, direction, center, radius, max_dist):
        f = origin - center
        b = 2.0 * np.dot(f, direction)
        c = np.dot(f, f) - radius * radius
        disc = b * b - 4.0 * c
        if disc < 0.0:
            return None

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / 2.0
        t2 = (-b + sqrt_disc) / 2.0
        candidates = [t for t in (t1, t2) if 0.0 <= t <= max_dist]
        if not candidates:
            return None
        return float(min(candidates))

    def _compute_lidar_rays(self):
        scan_dist = self.lidar_range
        origin = self.current_pos[:2].astype(float)

        to_target = (self.target_point[:2] - origin)
        norm = np.linalg.norm(to_target)
        if norm > 1e-9:
            to_target /= norm
        else:
            to_target = np.array([1.0, 0.0], dtype=float)

        readings = []
        z_abs = float(self.current_pos[2])

        for ang in self.lidar_angles:
            c, s = float(np.cos(ang)), float(np.sin(ang))
            ray_dir = np.array(
                [c * to_target[0] - s * to_target[1], s * to_target[0] + c * to_target[1]],
                dtype=float,
            )
            rn = np.linalg.norm(ray_dir)
            if rn > 1e-9:
                ray_dir /= rn

            hit = scan_dist
            for obs in self.obstacles:
                center = np.array(obs[0], dtype=float)
                r_eff = self._effective_radius(obs, z_abs)
                if r_eff <= 0.0:
                    continue
                d = self._ray_circle_hit_distance(origin, ray_dir, center, r_eff, scan_dist)
                if d is not None and d < hit:
                    hit = d
            readings.append(hit / scan_dist)

        rays = np.array(readings, dtype=np.float32)

        sigma = float(getattr(self, "lidar_noise_std", 0.0))
        if sigma > 0.0:
            noise = self.np_random.normal(0.0, sigma, size=rays.shape).astype(np.float32)
            rays = np.clip(rays + noise, 0.0, 1.0)

        return rays

    def _segment_circle_first_hit_t(self, A2, B2, C2, r):
        d = B2 - A2
        f = A2 - C2
        a = np.dot(d, d)
        if a < 1e-12:
            return None
        b = 2.0 * np.dot(f, d)
        c = np.dot(f, f) - r * r
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)
        candidates = [t for t in (t1, t2) if 0.0 <= t <= 1.0]
        if not candidates:
            return None
        return float(min(candidates))

    def _apply_safety_shield(self, prev_pos, desired_pos):
        A = prev_pos.astype(float)
        B = desired_pos.astype(float)
        A2, B2 = A[:2], B[:2]

        if np.linalg.norm(B2 - A2) < 1e-9:
            return desired_pos, False

        z_abs_min = float(min(A[2], B[2]))

        first_t = None
        hit_obs = None

        for obs in self.obstacles:
            center = np.array(obs[0], dtype=float)
            r_eff = self._effective_radius(obs, z_abs_min)
            if r_eff <= 0.0:
                continue

            t_hit = self._segment_circle_first_hit_t(A2, B2, center, r_eff)
            if t_hit is None:
                continue

            Qz = float(A[2] + t_hit * (B[2] - A[2]))
            ground = self._ground_at(center[0], center[1])
            z_rel = Qz - ground
            obs_h = float(obs[1])

            if z_rel <= obs_h + self.safety_margin_z:
                if first_t is None or t_hit < first_t:
                    first_t = t_hit
                    hit_obs = obs

        if first_t is None:
            return desired_pos, False

        # Backoff liczony w metrach, a nie w "ułamku odcinka"
        seg_len_xy = float(np.linalg.norm(B2 - A2) + 1e-9)
        backoff_t = min(first_t, self.shield_backoff_m / seg_len_xy)
        t_safe = max(0.0, first_t - backoff_t)
        safe_pos = A + t_safe * (B - A)

        if (not getattr(self, "shield_slide_enabled", True)) or (hit_obs is None):
            return safe_pos.astype(np.float32), True

        center = np.array(hit_obs[0], dtype=float)

        r_vec = safe_pos[:2] - center
        r_norm = float(np.linalg.norm(r_vec))
        if r_norm < 1e-6:
            r_vec = A2 - center
            r_norm = float(np.linalg.norm(r_vec))
            if r_norm < 1e-6:
                return safe_pos.astype(np.float32), True

        r_u = r_vec / r_norm
        t1 = np.array([-r_u[1], r_u[0]], dtype=float)
        t2 = -t1

        d_xy = (B - A)[:2]
        step_len = float(np.linalg.norm(d_xy) + 1e-9)
        blocked_frac = float(1.0 - t_safe)
        slide_len = blocked_frac * step_len

        slide_min = float(getattr(self, "shield_slide_min_m", 0.10))
        slide_max = float(getattr(self, "shield_slide_max_m", 2.50))
        slide_len = float(np.clip(slide_len, slide_min, slide_max))

        backoff_iters = int(getattr(self, "shield_slide_backoff_iters", 8))
        clear_eps = float(getattr(self, "shield_slide_clearance_eps", 1e-3))

        prev_dist = self._dist_xy(prev_pos)
        prev_clear = self._min_clearance_to_safety_zone_m(prev_pos)

        def try_dir(dir_vec):
            for k in range(backoff_iters):
                scale = 0.5 ** k
                cand = safe_pos.copy()
                cand[:2] = cand[:2] + dir_vec * (slide_len * scale)
                cand = np.clip(cand, self.bounds_low.astype(float), self.bounds_high.astype(float))

                if self._min_clearance_to_safety_zone_m(cand.astype(np.float32)) >= clear_eps:
                    return cand
            return None

        cands = []
        c1 = try_dir(t1)
        if c1 is not None:
            cands.append(c1)
        c2 = try_dir(t2)
        if c2 is not None:
            cands.append(c2)
        c3 = try_dir(r_u)
        if c3 is not None:
            cands.append(c3)

        if not cands:
            return safe_pos.astype(np.float32), True

        def score(cand):
            cand32 = cand.astype(np.float32)
            dist_gain = float(prev_dist - self._dist_xy(cand32))
            clear_gain = float(self._min_clearance_to_safety_zone_m(cand32) - prev_clear)
            return 1.0 * dist_gain + 0.25 * clear_gain

        best = max(cands, key=score)
        return best.astype(np.float32), True

    def _z_reference(self) -> float:
        ground_here = self._ground_at(self.current_pos[0], self.current_pos[1])

        floor = ground_here + self.rel_h_min_m
        ceiling = min(MAX_Z, ground_here + self.rel_h_max_m)

        if floor > ceiling:
            return float(ceiling)

        target = ground_here + self.rel_h_cruise_m
        return float(np.clip(target, floor, ceiling))

    def _min_clearance_to_obstacles_m(self, pos) -> float:
        x, y, z_abs = float(pos[0]), float(pos[1]), float(pos[2])
        best = float("inf")
        for obs in self.obstacles:
            center = np.array(obs[0], dtype=float)
            obs_h = float(obs[1])
            obs_r = float(obs[2])
            obs_type = obs[5]
            ground = self._ground_at(center[0], center[1])
            z_rel = z_abs - ground
            if z_rel > obs_h:
                continue
            if obs_type == "cylinder":
                r_real = obs_r
            else:
                if z_rel <= 0.0:
                    r_real = obs_r
                elif z_rel >= obs_h:
                    r_real = 0.0
                else:
                    r_real = obs_r * (1.0 - z_rel / obs_h)
            d_xy = float(np.linalg.norm(np.array([x, y], dtype=float) - center))
            clearance = d_xy - r_real
            if clearance < best:
                best = clearance
        if best == float("inf"):
            return 1e9
        return best

    def _min_clearance_to_safety_zone_m(self, pos) -> float:
        x, y, z_abs = float(pos[0]), float(pos[1]), float(pos[2])
        best = float("inf")

        p_xy = np.array([x, y], dtype=float)

        for obs in self.obstacles:
            center = np.array(obs[0], dtype=float)
            r_eff = self._effective_radius(obs, z_abs)
            if r_eff <= 0.0:
                continue

            d_xy = float(np.linalg.norm(p_xy - center))
            clearance = d_xy - float(r_eff)
            if clearance < best:
                best = clearance

        if best == float("inf"):
            return 1e9
        return best

    def _predict_next_pos_xy(self, pos, vel, acc_xy):
        pos = pos.astype(np.float32)
        vel = vel.astype(np.float32)
        acc_xy = acc_xy.astype(np.float32)

        v_next_xy = vel[:2] + acc_xy * self.dt
        v_mag = float(np.linalg.norm(v_next_xy))
        if v_mag > self.v_max_xy:
            v_next_xy = v_next_xy * (self.v_max_xy / (v_mag + 1e-9))

        p_next_xy = pos[:2] + v_next_xy * self.dt
        p_next_xy = np.clip(p_next_xy, self.bounds_low[:2], self.bounds_high[:2])
        return p_next_xy.astype(np.float32)

    def _clearance_safety_at_xy(self, pos_xy, z_abs):
        tmp = np.array([pos_xy[0], pos_xy[1], z_abs], dtype=np.float32)
        return float(self._min_clearance_to_safety_zone_m(tmp))

    def _apply_action_shield_xy(self, prev_pos, prev_vel, desired_acc_xy):
        acc0 = desired_acc_xy.astype(np.float32)

        if not self.obstacles:
            return acc0

        z_abs = float(prev_pos[2])
        eps = 1e-3

        p0_xy = self._predict_next_pos_xy(prev_pos, prev_vel, acc0)
        c0 = self._clearance_safety_at_xy(p0_xy, z_abs)
        if c0 >= eps:
            return acc0

        ax, ay = float(acc0[0]), float(acc0[1])
        a_left = np.array([-ay, ax], dtype=np.float32)
        a_right = np.array([ay, -ax], dtype=np.float32)

        best = None
        best_center = None
        p_xy = prev_pos[:2].astype(float)

        for obs in self.obstacles:
            center = np.array(obs[0], dtype=float)
            r_eff = self._effective_radius(obs, z_abs)
            if r_eff <= 0.0:
                continue
            d = float(np.linalg.norm(p_xy - center) - r_eff)
            if (best is None) or (d < best):
                best = d
                best_center = center

        if best_center is not None:
            away = (p_xy - best_center)
            n = float(np.linalg.norm(away) + 1e-9)
            away_u = (away / n).astype(np.float32)
            a_away = away_u * float(np.linalg.norm(acc0) + 1e-6)
        else:
            a_away = np.array([0.0, 0.0], dtype=np.float32)

        candidates = [acc0, a_left, a_right, a_away]

        prev_dist = float(self._dist_xy(prev_pos))
        prev_clear = float(self._min_clearance_to_safety_zone_m(prev_pos))

        def score(acc_xy):
            acc_xy = np.clip(acc_xy, -self.a_max_xy, self.a_max_xy).astype(np.float32)
            p_xy_next = self._predict_next_pos_xy(prev_pos, prev_vel, acc_xy)
            clear_next = self._clearance_safety_at_xy(p_xy_next, z_abs)

            if clear_next < eps:
                return -1e9

            tmp_next = np.array([p_xy_next[0], p_xy_next[1], prev_pos[2]], dtype=np.float32)
            dist_gain = prev_dist - float(self._dist_xy(tmp_next))
            clear_gain = clear_next - prev_clear

            return 1.0 * dist_gain + 0.25 * clear_gain

        best_acc = max(candidates, key=score).astype(np.float32)

        if score(best_acc) < -1e8:
            fallback_candidates = [
                np.clip(a_away, -self.a_max_xy, self.a_max_xy).astype(np.float32),
                np.zeros(2, dtype=np.float32),
            ]
            fallback_best = max(fallback_candidates, key=score).astype(np.float32)

            if score(fallback_best) < -1e8:
                return np.zeros(2, dtype=np.float32)

            return fallback_best

        return np.clip(best_acc, -self.a_max_xy, self.a_max_xy).astype(np.float32)

    def _boundary_penalty_xy(self, pos_xy: np.ndarray) -> float:
        x, y = float(pos_xy[0]), float(pos_xy[1])
        margin = 35.0
        k = 0.015
        cap = 25.0
        pen = 0.0
        if x > (MAX_X - margin):
            pen += k * (x - (MAX_X - margin)) ** 2
        elif x < (MIN_X + margin):
            pen += k * ((MIN_X + margin) - x) ** 2
        if y > (MAX_Y - margin):
            pen += k * (y - (MAX_Y - margin)) ** 2
        elif y < (MIN_Y + margin):
            pen += k * ((MIN_Y + margin) - y) ** 2
        return float(min(cap, pen))

    def _get_obs(self):
        pos_norm = self.current_pos / np.array([MAX_X, MAX_Y, MAX_Z], dtype=np.float32)

        # Wariant A: cel tylko w XY
        rel_target_xy = (self.target_point[:2] - self.current_pos[:2]) / 1000.0

        dist_xy = self._dist_xy(self.current_pos)
        dist_xy_norm = np.float32(np.clip(dist_xy / 1500.0, 0.0, 1.0))

        to_t = (self.target_point[:2] - self.current_pos[:2]).astype(np.float32)
        to_t_norm = float(np.linalg.norm(to_t) + 1e-9)
        dir_to_target = (to_t / to_t_norm).astype(np.float32)

        vxy = self.vel[:2].astype(np.float32)
        vxy_norm = float(np.linalg.norm(vxy) + 1e-9)
        speed_xy_norm = np.float32(np.clip(vxy_norm / self.v_max_xy, 0.0, 1.0))

        if vxy_norm > 1e-6:
            heading_cos = np.float32(np.dot((vxy / vxy_norm).astype(np.float32), dir_to_target))
        else:
            heading_cos = np.float32(0.0)

        rel_h_m = self._rel_h_m(self.current_pos)
        rel_h_norm = np.float32(rel_h_m / 100.0)

        vel_norm = self.vel / np.array([self.v_max_xy, self.v_max_xy, self.v_max_z], dtype=np.float32)
        over_g_proxy = np.float32(np.linalg.norm(self.last_eff_acc) / self.g0)

        base_obs = np.concatenate([
            pos_norm.astype(np.float32),
            rel_target_xy.astype(np.float32),
            np.array([dist_xy_norm], dtype=np.float32),
            dir_to_target.astype(np.float32),
            np.array([speed_xy_norm], dtype=np.float32),
            np.array([heading_cos], dtype=np.float32),
            np.array([rel_h_norm], dtype=np.float32),
            self.cached_lidar.astype(np.float32),
            vel_norm.astype(np.float32),
            np.array([over_g_proxy], dtype=np.float32),
            np.array([np.float32(self.cargo_health)], dtype=np.float32),
        ], axis=0).astype(np.float32)

        if not self.include_shield_features:
            return base_obs

        shielded_f = np.array([np.float32(self.last_shielded)], dtype=np.float32)
        denom = float(max(1, self.shield_stuck_patience_steps))
        shield_np_norm = np.array(
            [np.float32(np.clip(self.shield_no_progress_steps / denom, 0.0, 1.0))],
            dtype=np.float32,
        )

        obs = np.concatenate([base_obs, shielded_f, shield_np_norm], axis=0).astype(np.float32)
        return obs

    def step(self, action):
        self.current_step += 1
        prev_pos = self.current_pos.copy()
        prev_vel = self.vel.copy()

        prev_clearance_safety_m = self._min_clearance_to_safety_zone_m(prev_pos)

        action = np.array(action, dtype=np.float32).reshape(2)
        desired_acc_xy = np.clip(action, -1.0, 1.0) * self.a_max_xy

        desired_acc_xy = self._apply_action_shield_xy(prev_pos, prev_vel, desired_acc_xy)

        self.acc_xy_f = self.alpha_xy * self.acc_xy_f + (1.0 - self.alpha_xy) * desired_acc_xy

        jerk_xy = (self.acc_xy_f - self.prev_acc_xy) / self.dt
        jerk_xy_mag = float(np.linalg.norm(jerk_xy))
        if jerk_xy_mag > self.jerk_max:
            jerk_xy = jerk_xy * (self.jerk_max / (jerk_xy_mag + 1e-9))
            self.acc_xy_f = self.prev_acc_xy + jerk_xy * self.dt
        self.prev_acc_xy = self.acc_xy_f.copy()

        # Autopilot wysokości względem lokalnego terenu
        z_ref = self._z_reference()
        z_err = float(z_ref - float(self.current_pos[2]))
        vz = float(self.vel[2])
        az_cmd = self.kp_z * z_err - self.kd_z * vz
        az_cmd = float(np.clip(az_cmd, -self.a_max_z, self.a_max_z))
        self.acc_z_f = np.float32(self.alpha_z * float(self.acc_z_f) + (1.0 - self.alpha_z) * az_cmd)

        jerk_z = float((float(self.acc_z_f) - float(self.prev_acc_z)) / self.dt)
        if abs(jerk_z) > self.jerk_max:
            jerk_z = float(np.sign(jerk_z) * self.jerk_max)
            self.acc_z_f = np.float32(float(self.prev_acc_z) + jerk_z * self.dt)
        self.prev_acc_z = np.float32(self.acc_z_f)

        acc_cmd = np.array([self.acc_xy_f[0], self.acc_xy_f[1], float(self.acc_z_f)], dtype=np.float32)

        self.vel = self.vel + acc_cmd * self.dt
        vxy = self.vel[:2]
        vxy_mag = float(np.linalg.norm(vxy))
        if vxy_mag > self.v_max_xy:
            self.vel[:2] = vxy * (self.v_max_xy / (vxy_mag + 1e-9))
        if abs(float(self.vel[2])) > self.v_max_z:
            self.vel[2] = np.float32(np.sign(self.vel[2]) * self.v_max_z)

        desired_pos = self.current_pos + self.vel * self.dt

        w = getattr(self, "wind_vel_xy", None)
        if w is not None:
            desired_pos[:2] = desired_pos[:2] + np.array(w, dtype=np.float32) * self.dt

        desired_pos = np.clip(desired_pos, self.bounds_low, self.bounds_high)

        safe_pos, shielded = self._apply_safety_shield(prev_pos, desired_pos)
        self.current_pos = safe_pos

        if shielded:
            self.vel = (self.current_pos - prev_pos) / self.dt

        eff_acc = (self.vel - prev_vel) / self.dt

        self.cached_lidar = self._compute_lidar_rays()

        over_g_eff = float(np.linalg.norm(eff_acc) / self.g0)
        excess = max(0.0, over_g_eff - self.overload_soft_g)
        if excess > 0.0:
            self.cargo_health = max(0.0, self.cargo_health - self.health_decay_k * (excess ** 2))

        rel_h_m = self._rel_h_m(self.current_pos)
        self.ep_over_g_sum += over_g_eff
        self.ep_rel_h_sum += rel_h_m
        self.ep_step_count += 1

        dist_xy = self._dist_xy(self.current_pos)
        prev_dist_xy = self._dist_xy(prev_pos)

        reward = 0.0
        terminated, truncated = False, False
        termination = None

        reward += self.progress_scale * (prev_dist_xy - dist_xy)

        to_t = (self.target_point[:2] - self.current_pos[:2]).astype(np.float32)
        to_t_norm = float(np.linalg.norm(to_t) + 1e-9)
        vxy_u = self.vel[:2] / (float(np.linalg.norm(self.vel[:2])) + 1e-9)
        align = float(np.dot(to_t / to_t_norm, vxy_u))
        reward += self.align_k * align

        if float(np.linalg.norm(self.vel[:2])) < self.speed_min:
            reward -= self.low_speed_penalty

        reward -= self.time_penalty

        cte = self._cross_track_error()
        cte_pen = self.cte_quad_k * (max(0, cte - self.SOFT_LIMIT) ** 2) + self.cte_lin_k * max(0, cte - self.CORRIDOR_LIMIT)
        reward -= float(np.clip(cte_pen, 0.0, self.cte_cap))

        min_l_norm = float(np.min(self.cached_lidar))
        prox = max(0.0, self.lidar_trigger - min_l_norm)
        reward -= float(np.clip(self.lidar_k * (prox ** 2), 0.0, self.lidar_cap))

        clearance_safety_m = self._min_clearance_to_safety_zone_m(self.current_pos)

        if clearance_safety_m <= -self.penetration_eps_m:
            self.penetration_counter += 1
        else:
            self.penetration_counter = max(0, self.penetration_counter - 1)

        if (not terminated) and (self.penetration_counter >= self.penetration_patience_steps):
            reward -= self.collision_penalty
            terminated = True
            termination = "collision"

        # Kolizja z terenem
        ground_here = self._ground_at(self.current_pos[0], self.current_pos[1])
        if (not terminated) and (self.current_pos[2] <= ground_here + self.safety_margin_z):
            reward -= self.collision_penalty
            terminated = True
            termination = "terrain_collision"

        step_xy_m = float(np.linalg.norm((self.current_pos[:2] - prev_pos[:2])))

        if shielded:
            reward -= self.shield_penalty
            if step_xy_m < 0.1:
                reward -= 10.0

            delta_c = float(clearance_safety_m - prev_clearance_safety_m)
            if delta_c > 0.0:
                reward += float(np.clip(self.shield_clearance_bonus_k * delta_c, 0.0, self.shield_clearance_bonus_cap))

            progressed_dist = (dist_xy < (prev_dist_xy - self.shield_progress_eps_m))
            progressed_clear = (clearance_safety_m > (prev_clearance_safety_m + self.shield_clearance_progress_eps_m))
            progressed = progressed_dist or progressed_clear

            if (not progressed) or ((step_xy_m < self.shield_min_step_xy_m) and (not progressed_clear)):
                self.shield_no_progress_steps += 1
            else:
                self.shield_no_progress_steps = 0
        else:
            self.shield_no_progress_steps = 0

        if (not terminated) and (self.shield_no_progress_steps >= self.shield_stuck_patience_steps):
            reward -= 1500.0
            terminated = True
            termination = "shield_stuck"

        reward -= self.alt_k * (abs(rel_h_m - 30.0) / 30.0)
        reward -= self.acc_k * float(np.dot(eff_acc, eff_acc))
        reward -= self.over_k * (excess ** 2)
        reward -= self._boundary_penalty_xy(self.current_pos[:2])

        if self.best_dist_xy is None:
            self.best_dist_xy = dist_xy

        if dist_xy < (self.best_dist_xy - self.improve_eps_m):
            self.best_dist_xy = dist_xy
            self.steps_since_improve = 0
        else:
            self.steps_since_improve += 1

        # Wariant A: success / near_success tylko po XY
        if (not terminated) and (dist_xy < self.success_radius_xy):
            reward += self.success_bonus
            terminated = True
            termination = "success"

        if (not terminated) and (self.steps_since_improve >= self.stuck_patience_steps):
            if self.success_radius_xy < dist_xy <= self.near_success_radius_xy:
                reward += self.near_success_bonus
                terminated = True
                termination = "near_success"
            else:
                reward -= 200.0
                terminated = True
                termination = "stuck"

        if (not terminated) and (self.cargo_health <= 0.0):
            reward -= 200.0
            terminated = True
            termination = "cargo"

        if (not terminated) and (self.current_step >= self.max_steps):
            truncated = True
            termination = "timeout"

        self.last_shielded = 1.0 if shielded else 0.0

        info = {
            "termination": termination,
            "is_success": bool(termination == "success"),
            "dist_xy": float(dist_xy),
            "cargo_health": float(self.cargo_health),
            "shielded": bool(shielded),
            "shield_no_progress_steps": int(self.shield_no_progress_steps),
            "min_clearance_safety_m": float(clearance_safety_m),
            "rel_h_m": float(rel_h_m),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

from dataclasses import dataclass

import numpy as np

from gymnasium import spaces

@dataclass
class RealisticUrbanEnvConfig:
    """
    Configuration for RealisticUrbanCargoUAVEnv.

    Stage logic:
        Stage A/B: terminate_on_urban_constraint=False
        Stage C/evaluation: terminate_on_urban_constraint=True
    """
    risk_profile: str = "balanced"

    include_urban_vector_obs: bool = True
    urban_vector_dim: int = 20

    use_urban_risk_costs: bool = True
    urban_risk_lambda: float = 0.10

    use_urban_altitude_reference: bool = True

    terminate_on_urban_constraint: bool = False
    hard_sensitive_zones: bool = True

    enforce_safe_start_target: bool = True
    start_target_max_risk: float = 1.0
    max_reset_attempts: int = 300

    # Exposure budgets for mission acceptability
    mission_accept_cargo_min: float = 0.80
    mission_accept_road_budget_s: float = 20.0
    mission_accept_sidewalk_budget_s: float = 5.0
    mission_accept_building_budget_s: float = 15.0
    mission_accept_sensitive_budget_s: float = 0.0

    # Terminal budgets if terminate_on_urban_constraint=True
    terminal_road_budget_s: float = 30.0
    terminal_sidewalk_budget_s: float = 10.0
    terminal_building_budget_s: float = 20.0
    terminal_sensitive_budget_s: float = 0.0

    # Reward shaping
    green_bonus_lambda: float = 0.03
    acceptable_success_bonus: float = 500.0

    # For initial M3 use: v14 procedural obstacles can be zero with real GIS.
    default_n_obstacles: int = 0

class RealisticUrbanCargoUAVEnv(RandomizedCurriculumEnvV14):
    """
    First RUSCUAV Gymnasium environment.

    This class reuses v14 dynamics and safety stack, but augments the environment
    with:
        - UrbanWorldLayers;
        - UrbanRiskModel;
        - semantic risk/cost;
        - safe urban endpoints;
        - mission acceptability metrics.

    Important:
        M3 does not yet integrate buildings/canopy directly into v14 LiDAR and shield.
        That is M4. In M3, buildings/canopy are handled by altitude reference,
        risk penalties, constraints, and terminations.
    """

    def __init__(
        self,
        world: UrbanWorldLayers,
        config: RealisticUrbanEnvConfig = None,
        include_shield_features: bool = False,
    ):
        self.world = world
        self.config = config if config is not None else RealisticUrbanEnvConfig()

        # v14 expects dem_data. We use the canonical terrain layer.
        super().__init__(
            dem_data=world.terrain_z.astype(np.float32),
            include_shield_features=include_shield_features,
        )

        self.risk_model = UrbanRiskModel(
            world=world,
            profile=self.config.risk_profile,
            hard_sensitive_zones=self.config.hard_sensitive_zones,
            include_global_altitude_band_cost=True,
        )

        self.n_obstacles = int(self.config.default_n_obstacles)

        self.include_urban_vector_obs = bool(self.config.include_urban_vector_obs)
        self.urban_vector_dim = int(self.config.urban_vector_dim if self.include_urban_vector_obs else 0)

        self.base_obs_dim_ruscuav = 80 + (2 if self.include_shield_features else 0)
        self.total_obs_dim_ruscuav = self.base_obs_dim_ruscuav + self.urban_vector_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_obs_dim_ruscuav,),
            dtype=np.float32,
        )

        self._reset_urban_episode_metrics()

    # ============================================================
    # Episode metrics
    # ============================================================
    def _reset_urban_episode_metrics(self):
        self.ep_urban_risk_sum = 0.0
        self.ep_urban_risk_max = 0.0
        self.ep_urban_steps = 0

        self.ep_road_time = 0.0
        self.ep_sidewalk_time = 0.0
        self.ep_building_time = 0.0
        self.ep_green_time = 0.0
        self.ep_canopy_time = 0.0
        self.ep_sensitive_time = 0.0

        self.ep_building_collision_steps = 0
        self.ep_building_clearance_violation_steps = 0
        self.ep_canopy_collision_steps = 0
        self.ep_canopy_clearance_violation_steps = 0
        self.ep_sensitive_low_altitude_violation_steps = 0
        self.ep_urban_hard_violation_steps = 0

    # ============================================================
    # Safe endpoints
    # ============================================================
    def _endpoint_report(self, point):
        x, y, z = float(point[0]), float(point[1]), float(point[2])
        risk = float(self.risk_model.risk_at(x, y, z))
        flags = self.risk_model.flags_at(x, y)
        constraints = self.risk_model.constraints_at(x, y, z)

        disallowed_flags = []
        for key in ["building", "road", "sidewalk", "sensitive"]:
            if flags.get(key, False):
                disallowed_flags.append(key)

        safe = (
            risk <= self.config.start_target_max_risk
            and len(disallowed_flags) == 0
            and not constraints["any_hard_violation"]
        )

        return {
            "safe": bool(safe),
            "risk": risk,
            "flags": flags,
            "constraints": constraints,
            "disallowed_flags": disallowed_flags,
            "reason": "ok" if safe else f"risk={risk:.2f}, disallowed={disallowed_flags}, constraints={constraints}",
        }

    def _start_target_safe(self):
        start_report = self._endpoint_report(self.start_point)
        target_report = self._endpoint_report(self.target_point)

        return bool(start_report["safe"] and target_report["safe"]), start_report, target_report

    def reset(self, seed=None, options=None):
        attempts = self.config.max_reset_attempts if self.config.enforce_safe_start_target else 1

        info = {}
        obs = None
        safe = False
        start_report = None
        target_report = None

        for attempt in range(int(attempts)):
            attempt_seed = None if seed is None else int(seed) + 9973 * int(attempt)

            obs, info = super().reset(seed=attempt_seed, options=options)

            self._reset_urban_episode_metrics()

            safe, start_report, target_report = self._start_target_safe()

            if (not self.config.enforce_safe_start_target) or safe:
                break

        obs = self._get_obs()

        info = dict(info) if info is not None else {}
        info.update({
            "urban_safe_start_target": bool(safe),
            "urban_reset_attempts": int(attempt + 1),
            "start_urban_risk": float(start_report["risk"]) if start_report else np.nan,
            "target_urban_risk": float(target_report["risk"]) if target_report else np.nan,
            "start_urban_reason": start_report["reason"] if start_report else "unknown",
            "target_urban_reason": target_report["reason"] if target_report else "unknown",
        })

        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                f"Observation mismatch: obs.shape={obs.shape}, "
                f"observation_space.shape={self.observation_space.shape}"
            )

        return obs, info

    # ============================================================
    # Urban altitude reference
    # ============================================================
    def _z_reference(self) -> float:
        if not self.config.use_urban_altitude_reference:
            return super()._z_reference()

        x, y = float(self.current_pos[0]), float(self.current_pos[1])
        ground = self._ground_at(x, y)

        floor = ground + self.rel_h_min_m
        ceiling = min(MAX_Z, ground + self.rel_h_max_m)

        if floor > ceiling:
            return float(ceiling)

        z_ref = ground + self.rel_h_cruise_m

        flags = self.risk_model.flags_at(x, y)
        heights = self.risk_model.heights_at(x, y)
        p = self.risk_model.profile

        if flags["building"]:
            z_ref = max(z_ref, heights["building_roof_z"] + p.building_min_clearance_m)

        if flags["canopy"]:
            z_ref = max(z_ref, heights["canopy_top_z"] + p.canopy_min_clearance_m)

        return float(np.clip(z_ref, floor, ceiling))

    # ============================================================
    # Urban observation vector
    # ============================================================
    def _sample_risk_along_direction(self, direction_xy, distances_m):
        pos = self.current_pos[:2].astype(np.float32)
        d = np.asarray(direction_xy, dtype=np.float32)
        n = float(np.linalg.norm(d) + 1e-9)
        d = d / n

        z_abs = float(self.current_pos[2])

        risks = []
        roads = []
        sidewalks = []
        buildings = []
        greens = []
        canopies = []
        sensitives = []

        for dist in distances_m:
            p = pos + d * float(dist)
            x = float(np.clip(p[0], MIN_X, MAX_X - 1e-3))
            y = float(np.clip(p[1], MIN_Y, MAX_Y - 1e-3))

            risks.append(self.risk_model.risk_at(x, y, z_abs))

            flags = self.risk_model.flags_at(x, y)
            roads.append(float(flags["road"]))
            sidewalks.append(float(flags["sidewalk"]))
            buildings.append(float(flags["building"]))
            greens.append(float(flags["green"]))
            canopies.append(float(flags["canopy"]))
            sensitives.append(float(flags["sensitive"]))

        return {
            "risk_mean": float(np.mean(risks)) if risks else 0.0,
            "risk_max": float(np.max(risks)) if risks else 0.0,
            "road_mean": float(np.mean(roads)) if roads else 0.0,
            "sidewalk_mean": float(np.mean(sidewalks)) if sidewalks else 0.0,
            "building_mean": float(np.mean(buildings)) if buildings else 0.0,
            "green_mean": float(np.mean(greens)) if greens else 0.0,
            "canopy_mean": float(np.mean(canopies)) if canopies else 0.0,
            "sensitive_mean": float(np.mean(sensitives)) if sensitives else 0.0,
        }

    def _get_urban_vector_obs(self):
        x, y, z = float(self.current_pos[0]), float(self.current_pos[1]), float(self.current_pos[2])

        flags = self.risk_model.flags_at(x, y)
        clear = self.risk_model.clearances_at(x, y, z)
        constraints = self.risk_model.constraints_at(x, y, z)
        components = self.risk_model.cost_components_at(x, y, z)

        # Risk normalization: sensitive low-alt can be very large.
        # Use a robust denominator.
        risk_norm = float(np.clip(components["total_risk"] / 300.0, 0.0, 1.0))

        b_clear = clear["building_clearance_m"]
        c_clear = clear["canopy_clearance_m"]

        b_clear_norm = 1.0 if b_clear > 1e8 else float(np.clip(b_clear / 80.0, -1.0, 1.0))
        c_clear_norm = 1.0 if c_clear > 1e8 else float(np.clip(c_clear / 40.0, -1.0, 1.0))

        to_target = (self.target_point[:2] - self.current_pos[:2]).astype(np.float32)
        to_target_norm = float(np.linalg.norm(to_target) + 1e-9)
        forward = to_target / to_target_norm
        left = np.array([-forward[1], forward[0]], dtype=np.float32)
        right = -left

        distances = np.array([20.0, 50.0, 100.0, 150.0, 220.0], dtype=np.float32)

        fwd = self._sample_risk_along_direction(forward, distances)
        lft = self._sample_risk_along_direction(left, distances[:4])
        rgt = self._sample_risk_along_direction(right, distances[:4])

        road_budget = max(1e-6, self.config.mission_accept_road_budget_s)
        sidewalk_budget = max(1e-6, self.config.mission_accept_sidewalk_budget_s)
        building_budget = max(1e-6, self.config.mission_accept_building_budget_s)
        sensitive_budget = max(1.0, self.config.mission_accept_sensitive_budget_s + 1.0)

        obs = np.array([
            risk_norm,
            float(flags["building"]),
            float(flags["road"]),
            float(flags["sidewalk"]),
            float(flags["green"]),
            float(flags["canopy"]),
            float(flags["sensitive"]),
            b_clear_norm,
            c_clear_norm,
            float(constraints["any_hard_violation"]),
            float(np.clip(fwd["risk_mean"] / 300.0, 0.0, 1.0)),
            float(np.clip(fwd["risk_max"] / 300.0, 0.0, 1.0)),
            fwd["road_mean"],
            fwd["sidewalk_mean"],
            fwd["building_mean"],
            fwd["green_mean"],
            float(np.clip(lft["risk_mean"] / 300.0, 0.0, 1.0)),
            float(np.clip(rgt["risk_mean"] / 300.0, 0.0, 1.0)),
            float(np.clip(self.ep_road_time / road_budget, 0.0, 1.0)),
            float(np.clip(
                self.ep_sidewalk_time / sidewalk_budget
                + self.ep_building_time / building_budget
                + self.ep_sensitive_time / sensitive_budget,
                0.0,
                1.0,
            )),
        ], dtype=np.float32)

        assert obs.shape[0] == self.config.urban_vector_dim
        return obs

    def _get_obs(self):
        base_obs = super()._get_obs().astype(np.float32)

        if not self.include_urban_vector_obs:
            return base_obs

        urban_obs = self._get_urban_vector_obs().astype(np.float32)
        return np.concatenate([base_obs, urban_obs], axis=0).astype(np.float32)

    # ============================================================
    # Urban metrics and step
    # ============================================================
    def _update_urban_episode_metrics(self):
        x, y, z = float(self.current_pos[0]), float(self.current_pos[1]), float(self.current_pos[2])

        flags = self.risk_model.flags_at(x, y)
        constraints = self.risk_model.constraints_at(x, y, z)
        components = self.risk_model.cost_components_at(x, y, z)

        dt = float(self.dt)

        if flags["road"]:
            self.ep_road_time += dt
        if flags["sidewalk"]:
            self.ep_sidewalk_time += dt
        if flags["building"]:
            self.ep_building_time += dt
        if flags["green"]:
            self.ep_green_time += dt
        if flags["canopy"]:
            self.ep_canopy_time += dt
        if flags["sensitive"]:
            self.ep_sensitive_time += dt

        if constraints["building_collision"]:
            self.ep_building_collision_steps += 1
        if constraints["building_clearance_violation"]:
            self.ep_building_clearance_violation_steps += 1
        if constraints["canopy_collision"]:
            self.ep_canopy_collision_steps += 1
        if constraints["canopy_clearance_violation"]:
            self.ep_canopy_clearance_violation_steps += 1
        if constraints["sensitive_low_altitude_violation"]:
            self.ep_sensitive_low_altitude_violation_steps += 1
        if constraints["any_hard_violation"]:
            self.ep_urban_hard_violation_steps += 1

        total_risk = float(components["total_risk"])

        self.ep_urban_risk_sum += total_risk * dt
        self.ep_urban_risk_max = max(self.ep_urban_risk_max, total_risk)
        self.ep_urban_steps += 1

        return {
            "flags": flags,
            "constraints": constraints,
            "components": components,
            "total_risk": total_risk,
        }

    def _mission_acceptable(self, termination):
        if termination != "success":
            return False

        if float(self.cargo_health) < self.config.mission_accept_cargo_min:
            return False

        if self.ep_road_time > self.config.mission_accept_road_budget_s:
            return False
        if self.ep_sidewalk_time > self.config.mission_accept_sidewalk_budget_s:
            return False
        if self.ep_building_time > self.config.mission_accept_building_budget_s:
            return False
        if self.ep_sensitive_time > self.config.mission_accept_sensitive_budget_s:
            return False

        if self.ep_urban_hard_violation_steps > 0:
            return False

        return True

    def _urban_terminal_check(self, urban_state):
        constraints = urban_state["constraints"]

        if constraints["terrain_collision"]:
            return "urban_terrain_constraint"
        if constraints["building_collision"]:
            return "urban_building_collision"
        if constraints["canopy_collision"]:
            return "urban_canopy_collision"
        if constraints["sensitive_low_altitude_violation"]:
            return "urban_sensitive_low_altitude_violation"

        if self.ep_sidewalk_time > self.config.terminal_sidewalk_budget_s:
            return "urban_sidewalk_budget_exceeded"
        if self.ep_road_time > self.config.terminal_road_budget_s:
            return "urban_road_budget_exceeded"
        if self.ep_building_time > self.config.terminal_building_budget_s:
            return "urban_building_budget_exceeded"
        if self.ep_sensitive_time > self.config.terminal_sensitive_budget_s:
            return "urban_sensitive_budget_exceeded"

        return None

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        urban_state = self._update_urban_episode_metrics()
        total_risk = urban_state["total_risk"]
        flags = urban_state["flags"]
        constraints = urban_state["constraints"]
        components = urban_state["components"]

        if self.config.use_urban_risk_costs:
            reward -= float(self.config.urban_risk_lambda) * total_risk

            # Small preference for clean green corridors.
            clean_green = (
                flags["green"]
                and not flags["road"]
                and not flags["sidewalk"]
                and not flags["building"]
                and not flags["sensitive"]
            )

            if clean_green:
                reward += float(self.config.green_bonus_lambda)

        urban_termination = None

        if self.config.terminate_on_urban_constraint and (not terminated) and (not truncated):
            urban_termination = self._urban_terminal_check(urban_state)

            if urban_termination is not None:
                terminated = True
                info["termination"] = urban_termination
                info["is_success"] = False

                # Strong penalty for hard urban failure.
                reward -= 1000.0

        termination = info.get("termination", None)

        # Add small extra bonus for acceptable success.
        mission_acceptable = self._mission_acceptable(termination)

        if mission_acceptable and termination == "success":
            reward += float(self.config.acceptable_success_bonus)

        mean_risk = self.ep_urban_risk_sum / max(1e-6, self.ep_urban_steps * self.dt)

        info.update({
            "urban_risk": float(total_risk),
            "urban_mean_risk": float(mean_risk),
            "urban_max_risk": float(self.ep_urban_risk_max),
            "urban_cumulative_risk": float(self.ep_urban_risk_sum),

            "urban_road_cost": float(components["road_cost"]),
            "urban_sidewalk_cost": float(components["sidewalk_cost"]),
            "urban_building_cost": float(components["building_cost"]),
            "urban_green_cost": float(components["green_cost"]),
            "urban_canopy_cost": float(components["canopy_cost"]),
            "urban_sensitive_cost": float(components["sensitive_cost"]),

            "urban_road_time": float(self.ep_road_time),
            "urban_sidewalk_time": float(self.ep_sidewalk_time),
            "urban_building_time": float(self.ep_building_time),
            "urban_green_time": float(self.ep_green_time),
            "urban_canopy_time": float(self.ep_canopy_time),
            "urban_sensitive_time": float(self.ep_sensitive_time),

            "urban_building_collision_steps": int(self.ep_building_collision_steps),
            "urban_building_clearance_violation_steps": int(self.ep_building_clearance_violation_steps),
            "urban_canopy_collision_steps": int(self.ep_canopy_collision_steps),
            "urban_canopy_clearance_violation_steps": int(self.ep_canopy_clearance_violation_steps),
            "urban_sensitive_low_altitude_violation_steps": int(self.ep_sensitive_low_altitude_violation_steps),
            "urban_hard_violation_steps": int(self.ep_urban_hard_violation_steps),

            "urban_building": bool(flags["building"]),
            "urban_road": bool(flags["road"]),
            "urban_sidewalk": bool(flags["sidewalk"]),
            "urban_green": bool(flags["green"]),
            "urban_canopy": bool(flags["canopy"]),
            "urban_sensitive": bool(flags["sensitive"]),

            "mission_acceptable": bool(mission_acceptable),
        })

        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

def make_ruscuav_env_m3(
    world,
    stage="A",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
):
    """
    Creates RealisticUrbanCargoUAVEnv for a training/evaluation stage.

    Stage A:
        soft urban risk, no urban terminal, lambda=0.05

    Stage B:
        stronger urban risk, no urban terminal, lambda=0.10

    Stage C:
        stronger urban risk + urban constraints, lambda=0.10
    """
    stage = str(stage).upper()

    if stage == "A":
        cfg = RealisticUrbanEnvConfig(
            risk_profile=risk_profile,
            urban_risk_lambda=0.05,
            terminate_on_urban_constraint=False,
            hard_sensitive_zones=False,
            default_n_obstacles=int(n_obstacles),
        )

    elif stage == "B":
        cfg = RealisticUrbanEnvConfig(
            risk_profile=risk_profile,
            urban_risk_lambda=0.10,
            terminate_on_urban_constraint=False,
            hard_sensitive_zones=True,
            default_n_obstacles=int(n_obstacles),
        )

    elif stage == "C":
        cfg = RealisticUrbanEnvConfig(
            risk_profile=risk_profile,
            urban_risk_lambda=0.10,
            terminate_on_urban_constraint=True,
            hard_sensitive_zones=True,
            default_n_obstacles=int(n_obstacles),
            terminal_road_budget_s=30.0,
            terminal_sidewalk_budget_s=10.0,
            terminal_building_budget_s=20.0,
            terminal_sensitive_budget_s=0.0,
        )

    else:
        raise ValueError("stage must be A, B, or C")

    env = RealisticUrbanCargoUAVEnv(
        world=world,
        config=cfg,
        include_shield_features=False,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        print(f"[make_ruscuav_env_m3] stage={stage}, obs.shape={obs.shape}, info={info}")

    return env

def smoke_test_ruscuav_env_m3(
    world,
    stage="A",
    seed=123,
    n_steps=50,
    n_obstacles=0,
):
    env = make_ruscuav_env_m3(
        world=world,
        stage=stage,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    print("=" * 100)
    print("RUSCUAV M3 SMOKE TEST")
    print("=" * 100)
    print("stage:", stage)
    print("obs.shape:", obs.shape)
    print("observation_space:", env.observation_space)
    print("reset info:", info)

    assert obs.shape == env.observation_space.shape, (obs.shape, env.observation_space.shape)

    for i in range(int(n_steps)):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={i+1:03d} | r={reward:9.2f} | "
            f"term={info.get('termination')} | "
            f"risk={info.get('urban_risk', 0):8.3f} | "
            f"road_t={info.get('urban_road_time', 0):6.1f} | "
            f"sidewalk_t={info.get('urban_sidewalk_time', 0):6.1f} | "
            f"building_t={info.get('urban_building_time', 0):6.1f} | "
            f"sensitive_t={info.get('urban_sensitive_time', 0):6.1f} | "
            f"acceptable={info.get('mission_acceptable', False)}"
        )

        if terminated or truncated:
            break

    print("=" * 100)
    env.close()

    return obs, info

MIN_X, MAX_X = 0, 1000

MIN_Y, MAX_Y = 0, 1000

MIN_Z, MAX_Z = 0, 150

def check_world_matches_v14_bounds(world):
    assert world.shape == (1000, 1000), f"Expected grid (1000,1000), got {world.shape}"
    assert abs(world.resolution_m - 1.0) < 1e-6, f"Expected resolution 1.0 m, got {world.resolution_m}"

    z_required = max(
        float(np.nanmax(world.building_roof_z)) + 10.0,
        float(np.nanmax(world.canopy_top_z)) + 5.0,
        float(np.nanmax(world.terrain_z)) + 80.0,
    )

    print("=" * 80)
    print("V14 bounds compatibility check")
    print("=" * 80)
    print("world.shape:", world.shape)
    print("world.resolution_m:", world.resolution_m)
    print("required safe z estimate:", z_required)
    print("MAX_Z:", MAX_Z)

    if z_required > MAX_Z:
        print("CHECK REQUIRED: MAX_Z may be too low for this world.")
    else:
        print("PASS: current MAX_Z is sufficient for this world.")

    print("=" * 80)

import numpy as np

def _sample_xy_from_world_mask(world, mask_name, seed=123):
    rng = np.random.default_rng(seed)
    mask = getattr(world, mask_name)

    ys, xs = np.where(mask > 0.5)

    if len(xs) == 0:
        return None

    idx = int(rng.integers(0, len(xs)))
    return float(xs[idx] * world.resolution_m), float(ys[idx] * world.resolution_m)

def _force_env_position_agl(env, x, y, agl_m):
    """
    Force UAV state to a selected XY and altitude above terrain.
    """
    ground = env._ground_at(x, y)
    z = float(ground + agl_m)

    env.current_pos = np.array([x, y, z], dtype=np.float32)
    env.vel[:] = 0.0

    env.acc_xy_f[:] = 0.0
    env.prev_acc_xy[:] = 0.0
    env.acc_z_f = np.float32(0.0)
    env.prev_acc_z = np.float32(0.0)

    env.best_dist_xy = env._dist_xy(env.current_pos)
    env.steps_since_improve = 0
    env.cached_lidar = env._compute_lidar_rays()

    return env.current_pos.copy()

def probe_single_forced_step_m3(
    world,
    stage,
    mask_name,
    agl_m,
    seed=123,
    n_obstacles=0,
):
    """
    Creates env, places UAV over selected mask, performs one zero-action step.
    """
    pt = _sample_xy_from_world_mask(world, mask_name, seed=seed)

    if pt is None:
        print(f"[{mask_name}] no active pixels, skipping.")
        return None

    x, y = pt

    env = make_ruscuav_env_m3(
        world=world,
        stage=stage,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)
    pos = _force_env_position_agl(env, x, y, agl_m)

    flags_before = env.risk_model.flags_at(x, y)
    constraints_before = env.risk_model.constraints_at(x, y, pos[2])
    risk_before = env.risk_model.risk_at(x, y, pos[2])

    action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)

    result = {
        "stage": stage,
        "mask_name": mask_name,
        "x": x,
        "y": y,
        "agl_m": agl_m,
        "flags_before": flags_before,
        "constraints_before": constraints_before,
        "risk_before": risk_before,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination": info.get("termination"),
        "urban_risk": info.get("urban_risk"),
        "urban_road_time": info.get("urban_road_time"),
        "urban_sidewalk_time": info.get("urban_sidewalk_time"),
        "urban_building_time": info.get("urban_building_time"),
        "urban_green_time": info.get("urban_green_time"),
        "urban_canopy_time": info.get("urban_canopy_time"),
        "urban_sensitive_time": info.get("urban_sensitive_time"),
        "urban_building_collision_steps": info.get("urban_building_collision_steps"),
        "urban_canopy_collision_steps": info.get("urban_canopy_collision_steps"),
        "urban_sensitive_low_altitude_violation_steps": info.get("urban_sensitive_low_altitude_violation_steps"),
        "mission_acceptable": info.get("mission_acceptable"),
    }

    env.close()

    return result

def print_forced_probe_result_m3(result):
    if result is None:
        return

    print("=" * 100)
    print(f"M3 FORCED SINGLE-STEP PROBE | stage={result['stage']} | mask={result['mask_name']}")
    print("=" * 100)
    print(f"x={result['x']:.1f}, y={result['y']:.1f}, AGL={result['agl_m']:.1f}")
    print("flags_before:", result["flags_before"])
    print("constraints_before:", result["constraints_before"])
    print(f"risk_before: {result['risk_before']:.3f}")
    print("-" * 100)
    print(f"reward:      {result['reward']:.3f}")
    print(f"terminated:  {result['terminated']}")
    print(f"truncated:   {result['truncated']}")
    print(f"termination: {result['termination']}")
    print(f"urban_risk:  {result['urban_risk']}")
    print("-" * 100)
    print(f"road_t:      {result['urban_road_time']}")
    print(f"sidewalk_t:  {result['urban_sidewalk_time']}")
    print(f"building_t:  {result['urban_building_time']}")
    print(f"green_t:     {result['urban_green_time']}")
    print(f"canopy_t:    {result['urban_canopy_time']}")
    print(f"sensitive_t: {result['urban_sensitive_time']}")
    print("-" * 100)
    print(f"building_collision_steps:         {result['urban_building_collision_steps']}")
    print(f"canopy_collision_steps:           {result['urban_canopy_collision_steps']}")
    print(f"sensitive_low_alt_violation_steps:{result['urban_sensitive_low_altitude_violation_steps']}")
    print(f"mission_acceptable:               {result['mission_acceptable']}")
    print("=" * 100)

def run_forced_single_step_probes_m3(world, stage="C", seed=123):
    """
    Runs one-step forced probes over key urban classes.
    """
    probes = [
        # social exposure classes
        ("road_mask", 40.0),
        ("sidewalk_mask", 40.0),

        # physical classes
        # building at AGL 10 is often inside building if building is tall enough
        ("building_mask", 10.0),

        # canopy at AGL 5 is often inside canopy
        ("canopy_mask", 5.0),

        # sensitive at AGL 20 should trigger sensitive low-altitude violation in Stage C
        ("sensitive_mask", 20.0),

        # green should have low risk if not also canopy/sensitive
        ("green_mask", 40.0),
    ]

    results = []

    for mask_name, agl_m in probes:
        result = probe_single_forced_step_m3(
            world=world,
            stage=stage,
            mask_name=mask_name,
            agl_m=agl_m,
            seed=seed,
            n_obstacles=0,
        )
        print_forced_probe_result_m3(result)
        results.append(result)

    return results

def forced_budget_terminal_probe_m3(
    world,
    stage="C",
    mask_name="sidewalk_mask",
    agl_m=40.0,
    seed=123,
    max_steps=50,
):
    """
    Forces UAV to remain over selected semantic mask and checks budget terminal.
    """
    pt = _sample_xy_from_world_mask(world, mask_name, seed=seed)

    if pt is None:
        print(f"[{mask_name}] no active pixels, skipping.")
        return None

    x, y = pt

    env = make_ruscuav_env_m3(
        world=world,
        stage=stage,
        risk_profile="balanced",
        n_obstacles=0,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    history = []

    for step in range(int(max_steps)):
        _force_env_position_agl(env, x, y, agl_m)

        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        row = {
            "step": step + 1,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination": info.get("termination"),
            "urban_risk": float(info.get("urban_risk", np.nan)),
            "road_time": float(info.get("urban_road_time", 0.0)),
            "sidewalk_time": float(info.get("urban_sidewalk_time", 0.0)),
            "building_time": float(info.get("urban_building_time", 0.0)),
            "sensitive_time": float(info.get("urban_sensitive_time", 0.0)),
            "mission_acceptable": bool(info.get("mission_acceptable", False)),
        }

        history.append(row)

        print(
            f"step={row['step']:03d} | r={row['reward']:9.2f} | "
            f"risk={row['urban_risk']:8.3f} | "
            f"road_t={row['road_time']:6.1f} | "
            f"sidewalk_t={row['sidewalk_time']:6.1f} | "
            f"building_t={row['building_time']:6.1f} | "
            f"sensitive_t={row['sensitive_time']:6.1f} | "
            f"term={row['termination']} | "
            f"terminated={row['terminated']}"
        )

        if terminated or truncated:
            break

    env.close()

    return history

import numpy as np

from scipy.ndimage import distance_transform_edt

class RealisticUrbanCargoUAVEnvM4(RealisticUrbanCargoUAVEnv):
    """
    M4 extension of RealisticUrbanCargoUAVEnv.

    M3 behavior:
        Urban hazards are detected after the state update by risk/constraint checks.

    M4 behavior:
        Urban hazards are also used preventively by:
            - LiDAR,
            - action shield,
            - geometric post-integration shield.

    This is still a computationally light 2.5D approximation:
        - buildings are raster footprints with roof height;
        - canopy is raster footprint with top height;
        - sensitive zones are no-fly volumes below a configured altitude.
    """

    def __init__(
        self,
        world: UrbanWorldLayers,
        config: RealisticUrbanEnvConfig = None,
        include_shield_features: bool = False,
        urban_lidar_enabled: bool = True,
        urban_action_shield_enabled: bool = True,
        urban_geometric_shield_enabled: bool = True,
        urban_lidar_step_m: float = 3.0,
        urban_segment_check_step_m: float = 1.0,
        urban_shield_backoff_m: float = 0.50,
        urban_horizontal_safety_margin_m: float = 1.0,
    ):
        super().__init__(
            world=world,
            config=config,
            include_shield_features=include_shield_features,
        )

        self.urban_lidar_enabled = bool(urban_lidar_enabled)
        self.urban_action_shield_enabled = bool(urban_action_shield_enabled)
        self.urban_geometric_shield_enabled = bool(urban_geometric_shield_enabled)

        self.urban_lidar_step_m = float(max(0.5, urban_lidar_step_m))
        self.urban_segment_check_step_m = float(max(0.25, urban_segment_check_step_m))
        self.urban_shield_backoff_m = float(max(0.0, urban_shield_backoff_m))
        self.urban_horizontal_safety_margin_m = float(max(0.0, urban_horizontal_safety_margin_m))

        self._precompute_urban_sdf_maps()

        self.last_urban_action_shielded = False
        self.last_urban_geom_shielded = False
        self.last_urban_lidar_min_norm = 1.0
        self.last_urban_shield_reason = None

    # ============================================================
    # SDF maps and basic queries
    # ============================================================
    def _precompute_urban_sdf_maps(self):
        """
        Signed distance fields for raster footprints.

        Convention:
            positive outside footprint,
            negative inside footprint,
            units: meters.
        """
        res = float(self.world.resolution_m)

        def sdf_from_mask(mask):
            mask = np.asarray(mask > 0.5, dtype=bool)

            # outside distance to nearest obstacle pixel
            outside_dist = distance_transform_edt(~mask) * res

            # inside distance to nearest free pixel
            inside_dist = distance_transform_edt(mask) * res

            return (outside_dist - inside_dist).astype(np.float32)

        self._building_sdf_m = sdf_from_mask(self.world.building_mask)
        self._canopy_sdf_m = sdf_from_mask(self.world.canopy_mask)
        self._sensitive_sdf_m = sdf_from_mask(self.world.sensitive_mask)

    def _sdf_at_xy(self, sdf, x, y):
        yi, xi = self.risk_model._ij_from_xy(x, y)
        return float(sdf[yi, xi])

    def _urban_constraints_at_pos(self, pos):
        return self.risk_model.constraints_at(
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
        )

    def _urban_flags_at_pos(self, pos):
        return self.risk_model.flags_at(
            float(pos[0]),
            float(pos[1]),
        )

    def _urban_is_unsafe_point(self, pos, include_clearance_violation=True):
        """
        Returns True if a point is inside an urban hard/semi-hard hazard.

        include_clearance_violation=True means that not only collisions,
        but also low clearance over building/canopy are treated as unsafe
        for shield/LiDAR purposes.
        """
        c = self._urban_constraints_at_pos(pos)

        if c["terrain_collision"]:
            return True
        if c["building_collision"]:
            return True
        if c["canopy_collision"]:
            return True
        if c["sensitive_low_altitude_violation"]:
            return True

        if include_clearance_violation:
            if c["building_clearance_violation"]:
                return True
            if c["canopy_clearance_violation"]:
                return True

        return False

    def _urban_signed_clearance_proxy_m(self, pos):
        """
        Lightweight signed-clearance proxy.

        This combines:
            - terrain clearance;
            - vertical building/canopy clearance when over their footprint;
            - horizontal footprint SDF otherwise;
            - sensitive-zone SDF if below sensitive overflight altitude.

        It is not an exact 3D distance, but it is stable and fast enough
        for scoring action-shield candidates.
        """
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])

        p = self.risk_model.profile
        flags = self.risk_model.flags_at(x, y)
        clear = self.risk_model.clearances_at(x, y, z)
        z_agl = self.risk_model.z_agl_at(x, y, z)

        vals = []

        # Terrain clearance
        vals.append(clear["terrain_clearance_m"] - p.terrain_min_clearance_m)

        # Building
        if flags["building"]:
            vals.append(clear["building_clearance_m"] - p.building_min_clearance_m)
        else:
            vals.append(
                self._sdf_at_xy(self._building_sdf_m, x, y)
                - self.urban_horizontal_safety_margin_m
            )

        # Canopy
        if flags["canopy"]:
            vals.append(clear["canopy_clearance_m"] - p.canopy_min_clearance_m)
        else:
            vals.append(
                self._sdf_at_xy(self._canopy_sdf_m, x, y)
                - self.urban_horizontal_safety_margin_m
            )

        # Sensitive no-fly volume below threshold
        if z_agl < p.sensitive_min_overflight_agl_m:
            vals.append(
                self._sdf_at_xy(self._sensitive_sdf_m, x, y)
                - self.urban_horizontal_safety_margin_m
            )

        return float(np.min(vals))

    def _urban_clearance_gradient_xy(self, pos, eps_m=2.0):
        """
        Finite-difference gradient of the signed-clearance proxy.
        Points toward increasing urban clearance.
        """
        pos = np.asarray(pos, dtype=np.float32)

        px = pos.copy()
        mx = pos.copy()
        py = pos.copy()
        my = pos.copy()

        px[0] += eps_m
        mx[0] -= eps_m
        py[1] += eps_m
        my[1] -= eps_m

        px[:2] = np.clip(px[:2], self.bounds_low[:2], self.bounds_high[:2])
        mx[:2] = np.clip(mx[:2], self.bounds_low[:2], self.bounds_high[:2])
        py[:2] = np.clip(py[:2], self.bounds_low[:2], self.bounds_high[:2])
        my[:2] = np.clip(my[:2], self.bounds_low[:2], self.bounds_high[:2])

        gx = self._urban_signed_clearance_proxy_m(px) - self._urban_signed_clearance_proxy_m(mx)
        gy = self._urban_signed_clearance_proxy_m(py) - self._urban_signed_clearance_proxy_m(my)

        g = np.array([gx, gy], dtype=np.float32)
        n = float(np.linalg.norm(g) + 1e-9)

        if n < 1e-6:
            return np.zeros(2, dtype=np.float32)

        return (g / n).astype(np.float32)

    # ============================================================
    # Urban-aware LiDAR
    # ============================================================
    def _urban_ray_hit_distance(self, origin_xy, ray_dir_xy, z_abs, max_dist):
        """
        Samples along one ray and returns first urban hazard distance.
        """
        step = float(self.urban_lidar_step_m)
        n_steps = int(np.ceil(float(max_dist) / step))

        for i in range(1, n_steps + 1):
            d = min(float(max_dist), i * step)

            p_xy = origin_xy + ray_dir_xy * d

            if (
                p_xy[0] < MIN_X
                or p_xy[0] > MAX_X
                or p_xy[1] < MIN_Y
                or p_xy[1] > MAX_Y
            ):
                continue

            pos = np.array([p_xy[0], p_xy[1], z_abs], dtype=np.float32)

            if self._urban_is_unsafe_point(pos, include_clearance_violation=True):
                return float(d)

        return None

    def _compute_lidar_rays(self):
        """
        Extends v14 LiDAR with urban physical hazards.

        v14 LiDAR sees procedural obstacles.
        M4 LiDAR additionally sees:
            - building low-clearance / collision volumes,
            - canopy low-clearance / collision volumes,
            - sensitive no-fly volume below threshold.
        """
        base_rays = super()._compute_lidar_rays().astype(np.float32)

        if not self.urban_lidar_enabled:
            self.last_urban_lidar_min_norm = 1.0
            return base_rays

        scan_dist = float(self.lidar_range)
        origin = self.current_pos[:2].astype(np.float32)
        z_abs = float(self.current_pos[2])

        to_target = (self.target_point[:2] - origin).astype(np.float32)
        norm = float(np.linalg.norm(to_target) + 1e-9)

        if norm > 1e-9:
            to_target = to_target / norm
        else:
            to_target = np.array([1.0, 0.0], dtype=np.float32)

        urban_rays = np.ones_like(base_rays, dtype=np.float32)

        for idx, ang in enumerate(self.lidar_angles):
            c, s = float(np.cos(ang)), float(np.sin(ang))

            ray_dir = np.array(
                [
                    c * to_target[0] - s * to_target[1],
                    s * to_target[0] + c * to_target[1],
                ],
                dtype=np.float32,
            )

            rn = float(np.linalg.norm(ray_dir) + 1e-9)
            ray_dir = ray_dir / rn

            hit = self._urban_ray_hit_distance(
                origin_xy=origin,
                ray_dir_xy=ray_dir,
                z_abs=z_abs,
                max_dist=scan_dist,
            )

            if hit is not None:
                urban_rays[idx] = float(np.clip(hit / scan_dist, 0.0, 1.0))

        self.last_urban_lidar_min_norm = float(np.min(urban_rays))

        return np.minimum(base_rays, urban_rays).astype(np.float32)

    # ============================================================
    # Urban-aware action shield
    # ============================================================
    def _predict_pos_for_acc_xy_m4(self, prev_pos, prev_vel, acc_xy):
        """
        Uses the same one-step prediction logic as v14 action shield.
        """
        p_xy = self._predict_next_pos_xy(
            prev_pos,
            prev_vel,
            acc_xy.astype(np.float32),
        )

        return np.array([p_xy[0], p_xy[1], prev_pos[2]], dtype=np.float32)

    def _apply_action_shield_xy(self, prev_pos, prev_vel, desired_acc_xy):
        """
        v14 action shield + urban candidate filter.
        """
        self.last_urban_action_shielded = False

        # First let v14 handle procedural obstacles.
        acc0 = super()._apply_action_shield_xy(
            prev_pos,
            prev_vel,
            desired_acc_xy,
        ).astype(np.float32)

        if not self.urban_action_shield_enabled:
            return acc0

        pred0 = self._predict_pos_for_acc_xy_m4(prev_pos, prev_vel, acc0)

        if not self._urban_is_unsafe_point(pred0, include_clearance_violation=True):
            return acc0

        ax, ay = float(acc0[0]), float(acc0[1])

        a_left = np.array([-ay, ax], dtype=np.float32)
        a_right = np.array([ay, -ax], dtype=np.float32)

        grad = self._urban_clearance_gradient_xy(prev_pos)
        acc_mag = float(np.linalg.norm(acc0) + 1e-6)
        a_away = grad * acc_mag

        candidates = [
            acc0,
            a_left,
            a_right,
            a_away,
            np.zeros(2, dtype=np.float32),
        ]

        prev_dist = float(self._dist_xy(prev_pos))
        prev_clear = float(self._urban_signed_clearance_proxy_m(prev_pos))

        def score(acc_xy):
            acc_xy = np.clip(acc_xy, -self.a_max_xy, self.a_max_xy).astype(np.float32)
            pred = self._predict_pos_for_acc_xy_m4(prev_pos, prev_vel, acc_xy)

            if self._urban_is_unsafe_point(pred, include_clearance_violation=True):
                return -1e12

            dist_gain = prev_dist - float(self._dist_xy(pred))
            clear_gain = float(self._urban_signed_clearance_proxy_m(pred) - prev_clear)
            risk = float(self.risk_model.risk_at(pred[0], pred[1], pred[2]))

            return 1.00 * dist_gain + 0.35 * clear_gain - 0.02 * risk

        best = max(candidates, key=score)
        best_score = score(best)

        if best_score < -1e11:
            self.last_urban_action_shielded = True
            self.last_urban_shield_reason = "urban_action_no_feasible_candidate"
            return np.zeros(2, dtype=np.float32)

        if not np.allclose(best, acc0, atol=1e-6):
            self.last_urban_action_shielded = True
            self.last_urban_shield_reason = "urban_action_candidate_replaced"

        return np.clip(best, -self.a_max_xy, self.a_max_xy).astype(np.float32)

    # ============================================================
    # Urban-aware geometric post-integration shield
    # ============================================================
    def _first_urban_hit_on_segment(self, A, B):
        """
        Samples a segment and returns first hit fraction t in [0, 1].
        """
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)

        d = B - A
        seg_len = float(np.linalg.norm(d) + 1e-9)

        if seg_len < 1e-9:
            if self._urban_is_unsafe_point(B, include_clearance_violation=True):
                return 0.0
            return None

        n = max(2, int(np.ceil(seg_len / self.urban_segment_check_step_m)))

        for k in range(1, n + 1):
            t = float(k / n)
            P = A + t * d

            if self._urban_is_unsafe_point(P, include_clearance_violation=True):
                return t

        return None

    def _apply_urban_geometric_shield(self, prev_pos, desired_pos):
        """
        Prevents final accepted position from entering urban hazard volume.
        """
        if not self.urban_geometric_shield_enabled:
            return desired_pos.astype(np.float32), False

        A = np.asarray(prev_pos, dtype=np.float32)
        B = np.asarray(desired_pos, dtype=np.float32)

        # If previous point is already unsafe, do not try to project; let terminal logic handle it.
        if self._urban_is_unsafe_point(A, include_clearance_violation=True):
            return B.astype(np.float32), False

        hit_t = self._first_urban_hit_on_segment(A, B)

        if hit_t is None:
            return B.astype(np.float32), False

        seg_len = float(np.linalg.norm(B - A) + 1e-9)
        backoff_t = min(hit_t, self.urban_shield_backoff_m / seg_len)

        safe_t = max(0.0, hit_t - backoff_t)
        safe_pos = A + safe_t * (B - A)
        safe_pos = np.clip(safe_pos, self.bounds_low, self.bounds_high).astype(np.float32)

        # If numerical sampling still leaves us unsafe, fall back to prev_pos.
        if self._urban_is_unsafe_point(safe_pos, include_clearance_violation=True):
            safe_pos = A.copy().astype(np.float32)

        self.last_urban_geom_shielded = True
        self.last_urban_shield_reason = "urban_geometric_segment_hit"

        return safe_pos.astype(np.float32), True

    def _apply_safety_shield(self, prev_pos, desired_pos):
        """
        v14 geometric shield + urban geometric shield.
        """
        self.last_urban_geom_shielded = False

        # First handle v14 procedural obstacles.
        safe_pos_v14, shielded_v14 = super()._apply_safety_shield(
            prev_pos,
            desired_pos,
        )

        safe_pos_urban, shielded_urban = self._apply_urban_geometric_shield(
            prev_pos,
            safe_pos_v14,
        )

        return safe_pos_urban.astype(np.float32), bool(shielded_v14 or shielded_urban)

    # ============================================================
    # Step info augmentation
    # ============================================================
    def step(self, action):
        self.last_urban_action_shielded = False
        self.last_urban_geom_shielded = False
        self.last_urban_shield_reason = None

        obs, reward, terminated, truncated, info = super().step(action)

        info.update({
            "urban_action_shielded": bool(self.last_urban_action_shielded),
            "urban_geom_shielded": bool(self.last_urban_geom_shielded),
            "urban_shielded_any": bool(self.last_urban_action_shielded or self.last_urban_geom_shielded),
            "urban_lidar_min_norm": float(self.last_urban_lidar_min_norm),
            "urban_shield_reason": self.last_urban_shield_reason,
            "urban_signed_clearance_proxy_m": float(self._urban_signed_clearance_proxy_m(self.current_pos)),
        })

        return obs, reward, terminated, truncated, info

def make_ruscuav_env_m4(
    world,
    stage="A",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
    urban_lidar_enabled=True,
    urban_action_shield_enabled=True,
    urban_geometric_shield_enabled=True,
):
    """
    Creates M4 environment.

    Stage A:
        soft risk, no urban terminal, hard_sensitive_zones=False.

    Stage B:
        stronger risk, no urban terminal, hard_sensitive_zones=True.

    Stage C:
        urban constraints active.
    """
    stage = str(stage).upper()

    if stage == "A":
        cfg = RealisticUrbanEnvConfig(
            risk_profile=risk_profile,
            urban_risk_lambda=0.05,
            terminate_on_urban_constraint=False,
            hard_sensitive_zones=False,
            default_n_obstacles=int(n_obstacles),
        )

    elif stage == "B":
        cfg = RealisticUrbanEnvConfig(
            risk_profile=risk_profile,
            urban_risk_lambda=0.10,
            terminate_on_urban_constraint=False,
            hard_sensitive_zones=True,
            default_n_obstacles=int(n_obstacles),
        )

    elif stage == "C":
        cfg = RealisticUrbanEnvConfig(
            risk_profile=risk_profile,
            urban_risk_lambda=0.10,
            terminate_on_urban_constraint=True,
            hard_sensitive_zones=True,
            default_n_obstacles=int(n_obstacles),
            terminal_road_budget_s=30.0,
            terminal_sidewalk_budget_s=10.0,
            terminal_building_budget_s=20.0,
            terminal_sensitive_budget_s=0.0,
        )

    else:
        raise ValueError("stage must be A, B, or C")

    env = RealisticUrbanCargoUAVEnvM4(
        world=world,
        config=cfg,
        include_shield_features=False,
        urban_lidar_enabled=urban_lidar_enabled,
        urban_action_shield_enabled=urban_action_shield_enabled,
        urban_geometric_shield_enabled=urban_geometric_shield_enabled,
        urban_lidar_step_m=3.0,
        urban_segment_check_step_m=1.0,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        print(f"[make_ruscuav_env_m4] stage={stage}, obs.shape={obs.shape}, info={info}")

    return env

def smoke_test_ruscuav_env_m4(
    world,
    stage="A",
    seed=123,
    n_steps=50,
    n_obstacles=0,
):
    env = make_ruscuav_env_m4(
        world=world,
        stage=stage,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    print("=" * 100)
    print("RUSCUAV M4 SMOKE TEST")
    print("=" * 100)
    print("stage:", stage)
    print("obs.shape:", obs.shape)
    print("observation_space:", env.observation_space)
    print("reset info:", info)

    assert obs.shape == env.observation_space.shape, (obs.shape, env.observation_space.shape)

    for i in range(int(n_steps)):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={i+1:03d} | r={reward:9.2f} | "
            f"term={info.get('termination')} | "
            f"risk={info.get('urban_risk', 0):8.3f} | "
            f"road_t={info.get('urban_road_time', 0):6.1f} | "
            f"sidewalk_t={info.get('urban_sidewalk_time', 0):6.1f} | "
            f"shield_any={info.get('urban_shielded_any')} | "
            f"lidar_min={info.get('urban_lidar_min_norm', 1.0):6.3f} | "
            f"clear={info.get('urban_signed_clearance_proxy_m', np.nan):8.3f} | "
            f"acceptable={info.get('mission_acceptable', False)}"
        )

        if terminated or truncated:
            break

    print("=" * 100)

    env.close()
    return obs, info

import numpy as np

def _sample_xy_from_world_mask_m4(world, mask_name, seed=123):
    rng = np.random.default_rng(seed)
    mask = getattr(world, mask_name)

    ys, xs = np.where(mask > 0.5)

    if len(xs) == 0:
        return None

    idx = int(rng.integers(0, len(xs)))
    return float(xs[idx] * world.resolution_m), float(ys[idx] * world.resolution_m)

def _force_env_position_agl_m4(env, x, y, agl_m):
    ground = env._ground_at(x, y)
    z = float(ground + agl_m)

    env.current_pos = np.array([x, y, z], dtype=np.float32)
    env.vel[:] = 0.0

    env.acc_xy_f[:] = 0.0
    env.prev_acc_xy[:] = 0.0
    env.acc_z_f = np.float32(0.0)
    env.prev_acc_z = np.float32(0.0)

    env.best_dist_xy = env._dist_xy(env.current_pos)
    env.steps_since_improve = 0

    env.cached_lidar = env._compute_lidar_rays()

    return env.current_pos.copy()

def find_nearby_free_point_toward_mask_m4(
    world,
    mask_name,
    seed=123,
    offset_m=40.0,
):
    """
    Finds one target mask point and one free point offset from it.
    Returns:
        free_xy, target_xy
    """
    target = _sample_xy_from_world_mask_m4(world, mask_name, seed=seed)

    if target is None:
        return None, None

    tx, ty = target

    candidates = [
        (tx - offset_m, ty),
        (tx + offset_m, ty),
        (tx, ty - offset_m),
        (tx, ty + offset_m),
        (tx - offset_m, ty - offset_m),
        (tx + offset_m, ty + offset_m),
    ]

    for x, y in candidates:
        if x < MIN_X + 5 or x > MAX_X - 5 or y < MIN_Y + 5 or y > MAX_Y - 5:
            continue

        # free means not already in that mask
        yi, xi = world.resolution_m, world.resolution_m
        flags_ok = True

        # Use a temporary conservative check directly on masks.
        gx = int(np.clip(np.floor(x / world.resolution_m), 0, world.width - 1))
        gy = int(np.clip(np.floor(y / world.resolution_m), 0, world.height - 1))

        if getattr(world, mask_name)[gy, gx] > 0.5:
            flags_ok = False

        if flags_ok:
            return (float(x), float(y)), (float(tx), float(ty))

    return None, target

def forced_lidar_probe_m4(
    world,
    mask_name="building_mask",
    stage="C",
    seed=123,
    agl_m=20.0,
    offset_m=50.0,
):
    """
    Places UAV near a hazard and checks whether urban LiDAR sees it.
    """
    free_xy, target_xy = find_nearby_free_point_toward_mask_m4(
        world=world,
        mask_name=mask_name,
        seed=seed,
        offset_m=offset_m,
    )

    if free_xy is None or target_xy is None:
        print(f"[{mask_name}] could not find probe points.")
        return None

    env = make_ruscuav_env_m4(
        world=world,
        stage=stage,
        risk_profile="balanced",
        n_obstacles=0,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    # Put target_point in the direction of the hazard so goal-centric LiDAR points toward it.
    env.target_point[:2] = np.array(target_xy, dtype=np.float32)

    _force_env_position_agl_m4(env, free_xy[0], free_xy[1], agl_m=agl_m)

    lidar = env._compute_lidar_rays()
    min_lidar = float(np.min(lidar))
    min_urban = float(env.last_urban_lidar_min_norm)

    print("=" * 100)
    print(f"M4 FORCED LIDAR PROBE | mask={mask_name}")
    print("=" * 100)
    print("free_xy:", free_xy)
    print("target_xy:", target_xy)
    print("AGL:", agl_m)
    print("flags at UAV:", env.risk_model.flags_at(free_xy[0], free_xy[1]))
    print("flags at target:", env.risk_model.flags_at(target_xy[0], target_xy[1]))
    print("min combined lidar:", min_lidar)
    print("min urban lidar:", min_urban)
    print("expected: min urban lidar < 1.0 if hazard is visible")
    print("=" * 100)

    env.close()

    return {
        "mask_name": mask_name,
        "free_xy": free_xy,
        "target_xy": target_xy,
        "agl_m": agl_m,
        "min_lidar": min_lidar,
        "min_urban_lidar": min_urban,
    }

def forced_geometric_shield_probe_m4(
    world,
    mask_name="building_mask",
    stage="C",
    seed=123,
    agl_m=10.0,
    offset_m=40.0,
):
    """
    Builds a segment from a free point into a hazard footprint and checks
    whether M4 geometric shield stops before the hazard.
    """
    free_xy, target_xy = find_nearby_free_point_toward_mask_m4(
        world=world,
        mask_name=mask_name,
        seed=seed,
        offset_m=offset_m,
    )

    if free_xy is None or target_xy is None:
        print(f"[{mask_name}] could not find probe points.")
        return None

    env = make_ruscuav_env_m4(
        world=world,
        stage=stage,
        risk_profile="balanced",
        n_obstacles=0,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    prev = _force_env_position_agl_m4(env, free_xy[0], free_xy[1], agl_m=agl_m)

    ground_target = env._ground_at(target_xy[0], target_xy[1])
    desired = np.array(
        [target_xy[0], target_xy[1], ground_target + agl_m],
        dtype=np.float32,
    )

    unsafe_desired = env._urban_is_unsafe_point(desired, include_clearance_violation=True)

    safe, shielded = env._apply_safety_shield(prev, desired)
    unsafe_safe = env._urban_is_unsafe_point(safe, include_clearance_violation=True)

    print("=" * 100)
    print(f"M4 FORCED GEOMETRIC SHIELD PROBE | mask={mask_name}")
    print("=" * 100)
    print("prev:", prev)
    print("desired:", desired)
    print("safe:", safe)
    print("AGL:", agl_m)
    print("unsafe desired:", unsafe_desired)
    print("shielded:", shielded)
    print("unsafe safe:", unsafe_safe)
    print("urban geom shielded:", env.last_urban_geom_shielded)
    print("reason:", env.last_urban_shield_reason)
    print("expected: desired unsafe=True, shielded=True, safe unsafe=False")
    print("=" * 100)

    env.close()

    return {
        "mask_name": mask_name,
        "prev": prev,
        "desired": desired,
        "safe": safe,
        "unsafe_desired": bool(unsafe_desired),
        "shielded": bool(shielded),
        "unsafe_safe": bool(unsafe_safe),
        "urban_geom_shielded": bool(env.last_urban_geom_shielded),
        "reason": env.last_urban_shield_reason,
    }

def run_m4_forced_tests(world, seed=123):
    """
    Runs M4 forced tests.
    """
    results = {}

    print("\n" + "#" * 100)
    print("M4 LIDAR PROBES")
    print("#" * 100)

    for mask_name, agl in [
        ("building_mask", 20.0),
        ("canopy_mask", 5.0),
        ("sensitive_mask", 20.0),
    ]:
        results[f"lidar_{mask_name}"] = forced_lidar_probe_m4(
            world=world,
            mask_name=mask_name,
            stage="C",
            seed=seed,
            agl_m=agl,
            offset_m=50.0,
        )

    print("\n" + "#" * 100)
    print("M4 GEOMETRIC SHIELD PROBES")
    print("#" * 100)

    for mask_name, agl in [
        ("building_mask", 10.0),
        ("canopy_mask", 5.0),
        ("sensitive_mask", 20.0),
    ]:
        results[f"shield_{mask_name}"] = forced_geometric_shield_probe_m4(
            world=world,
            mask_name=mask_name,
            stage="C",
            seed=seed,
            agl_m=agl,
            offset_m=40.0,
        )

    return results

def m4_gate_check(results):
    """
    Checks whether key M4 forced tests passed.

    Expected:
        - LiDAR probes: min urban lidar < 1.0
        - Geometric shield probes: desired unsafe, shielded, final safe
    """
    print("=" * 100)
    print("RUSCUAV M4 GATE CHECK")
    print("=" * 100)

    passed = True

    for key, r in results.items():
        if r is None:
            print(f"{key:30s} | SKIP/FAIL: result None")
            passed = False
            continue

        if key.startswith("lidar_"):
            ok = r["min_urban_lidar"] < 1.0
            print(
                f"{key:30s} | min_urban_lidar={r['min_urban_lidar']:.3f} | "
                f"{'PASS' if ok else 'CHECK'}"
            )
            if not ok:
                passed = False

        elif key.startswith("shield_"):
            ok = (
                r["unsafe_desired"]
                and r["shielded"]
                and not r["unsafe_safe"]
            )
            print(
                f"{key:30s} | unsafe_desired={r['unsafe_desired']} | "
                f"shielded={r['shielded']} | unsafe_safe={r['unsafe_safe']} | "
                f"{'PASS' if ok else 'CHECK'}"
            )
            if not ok:
                passed = False

    print("-" * 100)
    print("M4 GATE:", "PASS" if passed else "CHECK REQUIRED")
    print("=" * 100)

    return {
        "passed": bool(passed),
    }

import os

import json

from pathlib import Path

import numpy as np

import torch

from stable_baselines3 import PPO

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

RUSCUAV_M4_INFO_KEYWORDS = (
    "termination",
    "is_success",
    "mission_acceptable",

    "urban_risk",
    "urban_mean_risk",
    "urban_max_risk",
    "urban_cumulative_risk",

    "urban_road_time",
    "urban_sidewalk_time",
    "urban_building_time",
    "urban_green_time",
    "urban_canopy_time",
    "urban_sensitive_time",

    "urban_building_collision_steps",
    "urban_building_clearance_violation_steps",
    "urban_canopy_collision_steps",
    "urban_canopy_clearance_violation_steps",
    "urban_sensitive_low_altitude_violation_steps",
    "urban_hard_violation_steps",

    "urban_action_shielded",
    "urban_geom_shielded",
    "urban_shielded_any",
    "urban_lidar_min_norm",
    "urban_signed_clearance_proxy_m",

    "cargo_health",
    "dist_xy",
)

class RUSCUAVTrainStatsCallback(BaseCallback):
    """
    Lightweight training callback for M5 debug.
    Tracks terminal causes, success and mission acceptability.
    """

    def __init__(self, print_every_steps=10_000, window=100, verbose=0):
        super().__init__(verbose)
        self.print_every_steps = int(print_every_steps)
        self.window = int(window)

        self.ep_success = []
        self.ep_acceptable = []
        self.ep_rewards = []
        self.term_counts = {}

        self.last_print_timestep = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for done, info in zip(dones, infos):
            if not done:
                continue

            term = info.get("termination", "unknown")
            self.term_counts[term] = self.term_counts.get(term, 0) + 1

            self.ep_success.append(int(bool(info.get("is_success", False))))
            self.ep_acceptable.append(int(bool(info.get("mission_acceptable", False))))

            if "episode" in info:
                self.ep_rewards.append(float(info["episode"]["r"]))

        if (self.num_timesteps - self.last_print_timestep) >= self.print_every_steps:
            self.last_print_timestep = self.num_timesteps

            w = min(self.window, len(self.ep_success))

            sr = float(np.mean(self.ep_success[-w:])) if w > 0 else 0.0
            ar = float(np.mean(self.ep_acceptable[-w:])) if w > 0 else 0.0
            rmean = float(np.mean(self.ep_rewards[-w:])) if w > 0 and len(self.ep_rewards) > 0 else 0.0

            print("=" * 90)
            print(
                f"[M5 train step={self.num_timesteps}] "
                f"SR({w})={sr:.1%} | MAR({w})={ar:.1%} | R={rmean:.1f}"
            )
            print("Term counts:", self.term_counts)
            print("=" * 90)

            self.term_counts = {}

        return True

def make_ruscuav_m4_stage_env_for_training(
    world,
    stage,
    seed=1001,
    n_obstacles=0,
    log_dir=None,
):
    """
    Creates Monitor-wrapped M4 env for PPO training.
    """
    raw_env = make_ruscuav_env_m4(
        world=world,
        stage=stage,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,
    )

    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        monitor_file = str(Path(log_dir) / f"monitor_stage_{stage}.csv")
    else:
        monitor_file = None

    env = Monitor(
        raw_env,
        filename=monitor_file,
        info_keywords=RUSCUAV_M4_INFO_KEYWORDS,
    )

    return env

def create_ruscuav_m4_ppo_model(env, seed=1001):
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=int(seed),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=DEVICE,
    )

    return model

def train_ruscuav_m4_three_stage_debug(
    world,
    output_dir="/content/ruscuav/models/m5_debug_ppo_m4",
    seed=1001,
    n_obstacles=0,
    steps_A=50_000,
    steps_B=50_000,
    steps_C=50_000,
    checkpoint_freq=25_000,
):
    """
    M5 debug training protocol.

    Stage A:
        soft urban risk, hard sensitive zones disabled, no urban terminal.

    Stage B:
        stronger/normal urban risk, hard sensitive zones enabled, no urban terminal.

    Stage C:
        urban risk + urban terminal budgets.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "seed": int(seed),
        "n_obstacles": int(n_obstacles),
        "steps_A": int(steps_A),
        "steps_B": int(steps_B),
        "steps_C": int(steps_C),
        "env": "RealisticUrbanCargoUAVEnvM4",
        "protocol": {
            "A": "soft urban risk, no urban terminal, hard_sensitive_zones=False",
            "B": "urban risk, no urban terminal, hard_sensitive_zones=True",
            "C": "urban risk + urban terminal constraints",
        },
    }

    with open(output_dir / "m5_training_protocol.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # -----------------------------
    # Stage A
    # -----------------------------
    stage_A_dir = output_dir / "stage_A"
    env_A = make_ruscuav_m4_stage_env_for_training(
        world=world,
        stage="A",
        seed=seed,
        n_obstacles=n_obstacles,
        log_dir=stage_A_dir,
    )

    model = create_ruscuav_m4_ppo_model(env_A, seed=seed)

    callback_A = [
        CheckpointCallback(
            save_freq=int(checkpoint_freq),
            save_path=str(stage_A_dir),
            name_prefix="ppo_ruscuav_m4_stage_A",
        ),
        RUSCUAVTrainStatsCallback(print_every_steps=10_000),
    ]

    print("\n" + "=" * 100)
    print("M5 TRAINING STAGE A")
    print("=" * 100)

    model.learn(
        total_timesteps=int(steps_A),
        callback=callback_A,
        reset_num_timesteps=True,
    )

    model.save(str(output_dir / "ppo_ruscuav_m4_after_stage_A.zip"))
    env_A.close()

    # -----------------------------
    # Stage B
    # -----------------------------
    stage_B_dir = output_dir / "stage_B"
    env_B = make_ruscuav_m4_stage_env_for_training(
        world=world,
        stage="B",
        seed=seed + 100,
        n_obstacles=n_obstacles,
        log_dir=stage_B_dir,
    )

    model.set_env(env_B)

    callback_B = [
        CheckpointCallback(
            save_freq=int(checkpoint_freq),
            save_path=str(stage_B_dir),
            name_prefix="ppo_ruscuav_m4_stage_B",
        ),
        RUSCUAVTrainStatsCallback(print_every_steps=10_000),
    ]

    print("\n" + "=" * 100)
    print("M5 TRAINING STAGE B")
    print("=" * 100)

    model.learn(
        total_timesteps=int(steps_B),
        callback=callback_B,
        reset_num_timesteps=False,
    )

    model.save(str(output_dir / "ppo_ruscuav_m4_after_stage_B.zip"))
    env_B.close()

    # -----------------------------
    # Stage C
    # -----------------------------
    stage_C_dir = output_dir / "stage_C"
    env_C = make_ruscuav_m4_stage_env_for_training(
        world=world,
        stage="C",
        seed=seed + 200,
        n_obstacles=n_obstacles,
        log_dir=stage_C_dir,
    )

    model.set_env(env_C)

    callback_C = [
        CheckpointCallback(
            save_freq=int(checkpoint_freq),
            save_path=str(stage_C_dir),
            name_prefix="ppo_ruscuav_m4_stage_C",
        ),
        RUSCUAVTrainStatsCallback(print_every_steps=10_000),
    ]

    print("\n" + "=" * 100)
    print("M5 TRAINING STAGE C")
    print("=" * 100)

    model.learn(
        total_timesteps=int(steps_C),
        callback=callback_C,
        reset_num_timesteps=False,
    )

    model.save(str(output_dir / "ppo_ruscuav_m4_after_stage_C.zip"))
    env_C.close()

    print("=" * 100)
    print("M5 debug training complete.")
    print("Saved to:", output_dir)
    print("=" * 100)

    return model, str(output_dir)

import pandas as pd

import numpy as np

from pathlib import Path

import json

def rollout_ruscuav_m4_model(
    model,
    world,
    stage="C",
    seed=5000,
    n_obstacles=0,
    deterministic=True,
    max_steps=1000,
):
    env = make_ruscuav_env_m4(
        world=world,
        stage=stage,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,
    )

    obs, reset_info = env.reset(seed=seed)

    path = [env.current_pos.copy()]
    infos = []
    rewards = []

    done = False
    step_count = 0

    while not done and step_count < int(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        done = bool(terminated or truncated)
        step_count += 1

        path.append(env.current_pos.copy())
        rewards.append(float(reward))
        infos.append(dict(info))

    final_info = infos[-1] if infos else {}

    result = {
        "path": np.asarray(path, dtype=np.float32),
        "rewards": rewards,
        "infos": infos,
        "reset_info": reset_info,
        "final_info": final_info,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(step_count),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),
    }

    env.close()

    return result

def evaluate_ruscuav_m4_model(
    model,
    world,
    stage="C",
    n_episodes=100,
    seed0=5000,
    n_obstacles=0,
    deterministic=True,
):
    rows = []

    for ep in range(int(n_episodes)):
        seed = int(seed0 + ep)

        roll = rollout_ruscuav_m4_model(
            model=model,
            world=world,
            stage=stage,
            seed=seed,
            n_obstacles=n_obstacles,
            deterministic=deterministic,
        )

        info = roll["final_info"]

        row = {
            "episode": ep,
            "seed": seed,
            "stage": stage,
            "total_reward": float(roll["total_reward"]),
            "steps": int(roll["steps"]),
            "termination": info.get("termination", None),
            "is_success": bool(info.get("is_success", False)),
            "mission_acceptable": bool(info.get("mission_acceptable", False)),

            "dist_xy": float(info.get("dist_xy", np.nan)),
            "cargo_health": float(info.get("cargo_health", np.nan)),

            "urban_mean_risk": float(info.get("urban_mean_risk", np.nan)),
            "urban_max_risk": float(info.get("urban_max_risk", np.nan)),
            "urban_cumulative_risk": float(info.get("urban_cumulative_risk", np.nan)),

            "road_time": float(info.get("urban_road_time", 0.0)),
            "sidewalk_time": float(info.get("urban_sidewalk_time", 0.0)),
            "building_time": float(info.get("urban_building_time", 0.0)),
            "green_time": float(info.get("urban_green_time", 0.0)),
            "canopy_time": float(info.get("urban_canopy_time", 0.0)),
            "sensitive_time": float(info.get("urban_sensitive_time", 0.0)),

            "building_collision_steps": int(info.get("urban_building_collision_steps", 0)),
            "canopy_collision_steps": int(info.get("urban_canopy_collision_steps", 0)),
            "sensitive_low_altitude_violation_steps": int(info.get("urban_sensitive_low_altitude_violation_steps", 0)),
            "urban_hard_violation_steps": int(info.get("urban_hard_violation_steps", 0)),

            "urban_action_shielded_last": bool(info.get("urban_action_shielded", False)),
            "urban_geom_shielded_last": bool(info.get("urban_geom_shielded", False)),
            "urban_lidar_min_norm_last": float(info.get("urban_lidar_min_norm", np.nan)),
            "urban_signed_clearance_proxy_m_last": float(info.get("urban_signed_clearance_proxy_m", np.nan)),
        }

        rows.append(row)

    return pd.DataFrame(rows)

def summarize_ruscuav_m4_eval(df):
    summary = {
        "n_episodes": int(len(df)),
        "success_rate": float(df["is_success"].mean()),
        "mission_acceptability_rate": float(df["mission_acceptable"].mean()),
        "mean_reward": float(df["total_reward"].mean()),
        "mean_steps": float(df["steps"].mean()),
        "mean_dist_xy": float(df["dist_xy"].mean()),
        "mean_cargo_health": float(df["cargo_health"].mean()),
        "mean_urban_risk": float(df["urban_mean_risk"].mean()),
        "mean_cumulative_risk": float(df["urban_cumulative_risk"].mean()),
        "mean_road_time": float(df["road_time"].mean()),
        "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
        "mean_building_time": float(df["building_time"].mean()),
        "mean_green_time": float(df["green_time"].mean()),
        "mean_sensitive_time": float(df["sensitive_time"].mean()),
        "mean_hard_violation_steps": float(df["urban_hard_violation_steps"].mean()),
        "termination_counts": {
            str(k): int(v)
            for k, v in df["termination"].fillna("None").value_counts().to_dict().items()
        },
    }

    print("=" * 100)
    print("RUSCUAV M4 EVALUATION SUMMARY")
    print("=" * 100)
    for k, v in summary.items():
        if k == "termination_counts":
            print("termination_counts:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        elif isinstance(v, float):
            print(f"{k:35s}: {v:.4f}")
        else:
            print(f"{k:35s}: {v}")
    print("=" * 100)

    return summary

def save_ruscuav_m4_eval(df, summary, output_dir, prefix="m5_debug_eval"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{prefix}_episodes.csv"
    json_path = output_dir / f"{prefix}_summary.json"

    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved:", csv_path)
    print("Saved:", json_path)

    return str(csv_path), str(json_path)

from pathlib import Path

import os

import json

import shutil

import time

from datetime import datetime

IN_COLAB = False

def mount_drive_if_colab(force_remount=False):
    if IN_COLAB:
        drive.mount("/content/drive", force_remount=force_remount)
        print("Google Drive mounted.")
    else:
        print("Not running in Colab or google.colab unavailable.")

def setup_ruscuav_drive_project(
    drive_root="/content/drive/MyDrive/RUSCUAV",
    experiment_name="m5_m4_ppo_debug",
):
    drive_root = Path(drive_root)
    exp_dir = drive_root / "experiments" / experiment_name

    dirs = {
        "drive_root": drive_root,
        "exp_dir": exp_dir,
        "models": exp_dir / "models",
        "checkpoints": exp_dir / "checkpoints",
        "logs": exp_dir / "logs",
        "eval": exp_dir / "eval",
        "worlds": drive_root / "worlds",
        "manifests": drive_root / "manifests",
        "figures": exp_dir / "figures",
        "metadata": exp_dir / "metadata",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("RUSCUAV Drive project ready")
    print("=" * 90)
    for k, v in dirs.items():
        print(f"{k:12s}: {v}")
    print("=" * 90)

    return dirs

def save_world_to_drive(
    world,
    drive_dirs,
    name=None,
):
    if name is None:
        name = f"{world.aoi_id}_urban_world_layers.npz"

    world_path_drive = Path(drive_dirs["worlds"]) / name
    world.save_npz(world_path_drive)

    meta = {
        "aoi_id": world.aoi_id,
        "crs": world.crs,
        "bbox": list(world.bbox),
        "resolution_m": float(world.resolution_m),
        "shape": list(world.shape),
        "saved_at": datetime.now().isoformat(),
        "world_path_drive": str(world_path_drive),
    }

    meta_path = Path(drive_dirs["worlds"]) / name.replace(".npz", "_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved world to Drive:", world_path_drive)
    print("Saved metadata:", meta_path)

    return str(world_path_drive)

def load_world_from_drive(world_path_drive):
    world_path_drive = Path(world_path_drive)
    if not world_path_drive.exists():
        raise FileNotFoundError(world_path_drive)

    world = UrbanWorldLayers.load_npz(world_path_drive)
    print("Loaded world:", world_path_drive)
    print("AOI:", world.aoi_id, "| shape:", world.shape, "| resolution:", world.resolution_m)
    return world

WORLD_PATH_DRIVE = "/content/drive/MyDrive/RUSCUAV/worlds/AOI_FAKE_001_urban_world_layers.npz"

from stable_baselines3.common.callbacks import BaseCallback

import numpy as np

def atomic_write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

class RUSCUAVDriveAutosaveCallback(BaseCallback):
    """
    Robust autosave callback for Colab.

    Saves:
        - stage_latest.zip
        - numbered checkpoints
        - progress JSON
        - heartbeat JSON

    Works with single-env PPO.
    """

    def __init__(
        self,
        stage_dir,
        stage_name,
        target_stage_steps,
        stage_start_timesteps,
        save_freq_steps=5_000,
        keep_numbered_checkpoints=True,
        verbose=0,
    ):
        super().__init__(verbose)

        self.stage_dir = Path(stage_dir)
        self.stage_dir.mkdir(parents=True, exist_ok=True)

        self.stage_name = str(stage_name)
        self.target_stage_steps = int(target_stage_steps)
        self.stage_start_timesteps = int(stage_start_timesteps)

        self.save_freq_steps = int(max(1, save_freq_steps))
        self.keep_numbered_checkpoints = bool(keep_numbered_checkpoints)

        self.latest_model_path = self.stage_dir / f"stage_{self.stage_name}_latest.zip"
        self.progress_path = self.stage_dir / f"stage_{self.stage_name}_progress.json"
        self.heartbeat_path = self.stage_dir / f"stage_{self.stage_name}_heartbeat.json"

        self.last_saved_completed = -1
        self.training_started_at = time.time()

    def _completed_stage_steps(self):
        return int(max(0, self.num_timesteps - self.stage_start_timesteps))

    def _save_all(self, reason="periodic"):
        completed = self._completed_stage_steps()

        # Save latest model
        self.model.save(str(self.latest_model_path))

        numbered_path = None
        if self.keep_numbered_checkpoints:
            numbered_path = self.stage_dir / (
                f"stage_{self.stage_name}_ckpt_"
                f"completed_{completed:09d}_global_{self.num_timesteps:09d}.zip"
            )
            self.model.save(str(numbered_path))

        progress = {
            "stage": self.stage_name,
            "reason": reason,
            "target_stage_steps": int(self.target_stage_steps),
            "stage_start_timesteps": int(self.stage_start_timesteps),
            "last_model_timesteps": int(self.num_timesteps),
            "completed_stage_steps": int(completed),
            "remaining_stage_steps": int(max(0, self.target_stage_steps - completed)),
            "latest_model_path": str(self.latest_model_path),
            "numbered_checkpoint_path": str(numbered_path) if numbered_path is not None else None,
            "saved_at": datetime.now().isoformat(),
            "elapsed_minutes_this_callback": float((time.time() - self.training_started_at) / 60.0),
        }

        atomic_write_json(self.progress_path, progress)

        heartbeat = {
            "stage": self.stage_name,
            "num_timesteps": int(self.num_timesteps),
            "completed_stage_steps": int(completed),
            "saved_at": datetime.now().isoformat(),
            "reason": reason,
        }
        atomic_write_json(self.heartbeat_path, heartbeat)

        self.last_saved_completed = completed

        if self.verbose:
            print(f"[DriveAutosave] stage={self.stage_name} completed={completed} reason={reason}")

    def _on_training_start(self) -> None:
        self._save_all(reason="training_start")

    def _on_step(self) -> bool:
        completed = self._completed_stage_steps()

        if completed - self.last_saved_completed >= self.save_freq_steps:
            self._save_all(reason="periodic")

        return True

    def _on_training_end(self) -> None:
        self._save_all(reason="training_end")

from stable_baselines3 import PPO

def read_progress_json(progress_path):
    progress_path = Path(progress_path)
    if not progress_path.exists():
        return None
    with open(progress_path, "r") as f:
        return json.load(f)

def stage_paths_on_drive(drive_dirs, stage_name):
    stage_dir = Path(drive_dirs["checkpoints"]) / f"stage_{stage_name}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    return {
        "stage_dir": stage_dir,
        "latest": stage_dir / f"stage_{stage_name}_latest.zip",
        "progress": stage_dir / f"stage_{stage_name}_progress.json",
        "final": Path(drive_dirs["models"]) / f"ppo_ruscuav_m4_after_stage_{stage_name}.zip",
        "done": stage_dir / f"stage_{stage_name}_DONE.json",
    }

def load_stage_model_if_available(stage_paths, env=None, device="cpu"):
    latest = Path(stage_paths["latest"])
    final = Path(stage_paths["final"])

    if latest.exists():
        print("Loading latest checkpoint:", latest)
        return PPO.load(str(latest), env=env, device=device), str(latest)

    if final.exists():
        print("Loading final model:", final)
        return PPO.load(str(final), env=env, device=device), str(final)

    return None, None

def mark_stage_done(stage_paths, stage_name, model, completed_stage_steps, target_stage_steps):
    final_path = Path(stage_paths["final"])
    final_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(final_path))

    done = {
        "stage": stage_name,
        "completed_stage_steps": int(completed_stage_steps),
        "target_stage_steps": int(target_stage_steps),
        "model_timesteps": int(model.num_timesteps),
        "final_model_path": str(final_path),
        "done_at": datetime.now().isoformat(),
    }

    atomic_write_json(stage_paths["done"], done)

    print(f"Stage {stage_name} DONE. Saved final:", final_path)
    return str(final_path)

def is_stage_done(stage_paths, target_stage_steps):
    done_path = Path(stage_paths["done"])
    if not done_path.exists():
        return False

    with open(done_path, "r") as f:
        d = json.load(f)

    return int(d.get("completed_stage_steps", 0)) >= int(target_stage_steps)

from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback

def make_ruscuav_m4_stage_env_for_training_drive(
    world,
    stage,
    seed,
    n_obstacles,
    drive_dirs,
):
    raw_env = make_ruscuav_env_m4(
        world=world,
        stage=stage,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,
    )

    monitor_dir = Path(drive_dirs["logs"]) / f"stage_{stage}"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    monitor_file = str(monitor_dir / f"monitor_stage_{stage}.csv")

    env = Monitor(
        raw_env,
        filename=monitor_file,
        info_keywords=RUSCUAV_M4_INFO_KEYWORDS,
    )

    return env

def create_or_load_model_for_stage(
    stage,
    env,
    drive_dirs,
    previous_model_path=None,
    seed=1001,
    device="cpu",
):
    paths = stage_paths_on_drive(drive_dirs, stage)

    model, loaded_from = load_stage_model_if_available(paths, env=env, device=device)

    if model is not None:
        print(f"Resuming stage {stage} from:", loaded_from)
        return model, loaded_from

    if previous_model_path is not None and Path(previous_model_path).exists():
        print(f"Starting stage {stage} from previous stage model:", previous_model_path)
        model = PPO.load(str(previous_model_path), env=env, device=device)
        return model, str(previous_model_path)

    print(f"Starting stage {stage} from scratch.")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=int(seed),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
    )

    return model, None

def train_one_stage_resume_safe(
    world,
    stage,
    target_stage_steps,
    drive_dirs,
    seed=1001,
    n_obstacles=0,
    previous_model_path=None,
    save_freq_steps=5_000,
    device="cpu",
):
    stage = str(stage).upper()
    paths = stage_paths_on_drive(drive_dirs, stage)

    if is_stage_done(paths, target_stage_steps):
        print(f"Stage {stage} already marked DONE. Skipping.")
        return str(paths["final"])

    env = make_ruscuav_m4_stage_env_for_training_drive(
        world=world,
        stage=stage,
        seed=seed,
        n_obstacles=n_obstacles,
        drive_dirs=drive_dirs,
    )

    model, loaded_from = create_or_load_model_for_stage(
        stage=stage,
        env=env,
        drive_dirs=drive_dirs,
        previous_model_path=previous_model_path,
        seed=seed,
        device=device,
    )

    progress = read_progress_json(paths["progress"])

    if progress is not None and Path(paths["latest"]).exists():
        stage_start_timesteps = int(progress["stage_start_timesteps"])
        completed_stage_steps = int(progress["completed_stage_steps"])
        print(f"Read progress for stage {stage}: completed={completed_stage_steps}/{target_stage_steps}")
    else:
        # New stage starts at current model timesteps.
        stage_start_timesteps = int(model.num_timesteps)
        completed_stage_steps = 0
        print(f"New progress for stage {stage}: stage_start_timesteps={stage_start_timesteps}")

    remaining = int(max(0, target_stage_steps - completed_stage_steps))

    print("=" * 100)
    print(f"TRAIN STAGE {stage}")
    print("=" * 100)
    print("target_stage_steps:", target_stage_steps)
    print("completed_stage_steps:", completed_stage_steps)
    print("remaining:", remaining)
    print("model.num_timesteps:", model.num_timesteps)
    print("stage_start_timesteps:", stage_start_timesteps)
    print("stage dir:", paths["stage_dir"])
    print("=" * 100)

    if remaining <= 0:
        final_path = mark_stage_done(
            paths,
            stage_name=stage,
            model=model,
            completed_stage_steps=completed_stage_steps,
            target_stage_steps=target_stage_steps,
        )
        env.close()
        return final_path

    autosave_cb = RUSCUAVDriveAutosaveCallback(
        stage_dir=paths["stage_dir"],
        stage_name=stage,
        target_stage_steps=target_stage_steps,
        stage_start_timesteps=stage_start_timesteps,
        save_freq_steps=save_freq_steps,
        keep_numbered_checkpoints=True,
        verbose=1,
    )

    stats_cb = RUSCUAVTrainStatsCallback(
        print_every_steps=10_000,
        window=100,
    )

    callback = CallbackList([autosave_cb, stats_cb])

    # If model is freshly created and num_timesteps == 0, reset_num_timesteps=True is fine.
    # If loaded or continuing from previous stage, keep global timestep continuity.
    reset_num_timesteps = bool(model.num_timesteps == 0)

    try:
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        # Always attempt final emergency save, even if interrupted manually.
        model.save(str(paths["latest"]))
        progress_now = {
            "stage": stage,
            "reason": "finally_block",
            "target_stage_steps": int(target_stage_steps),
            "stage_start_timesteps": int(stage_start_timesteps),
            "last_model_timesteps": int(model.num_timesteps),
            "completed_stage_steps": int(max(0, model.num_timesteps - stage_start_timesteps)),
            "saved_at": datetime.now().isoformat(),
            "latest_model_path": str(paths["latest"]),
        }
        atomic_write_json(paths["progress"], progress_now)
        print("Emergency/finally save completed:", paths["latest"])

    completed_final = int(max(0, model.num_timesteps - stage_start_timesteps))

    final_path = mark_stage_done(
        paths,
        stage_name=stage,
        model=model,
        completed_stage_steps=completed_final,
        target_stage_steps=target_stage_steps,
    )

    env.close()

    return final_path

def train_ruscuav_m4_three_stage_drive_resume(
    world,
    drive_dirs,
    seed=1001,
    n_obstacles=0,
    steps_A=50_000,
    steps_B=50_000,
    steps_C=50_000,
    save_freq_steps=5_000,
    device="cpu",
):
    """
    Resume-safe A/B/C training.

    Can be re-run after Colab disconnect.
    Already completed stages are skipped.
    Interrupted stage resumes from stage_latest.zip.
    """
    protocol = {
        "experiment": "RUSCUAV M4 PPO resume-safe",
        "seed": int(seed),
        "n_obstacles": int(n_obstacles),
        "steps_A": int(steps_A),
        "steps_B": int(steps_B),
        "steps_C": int(steps_C),
        "device": device,
        "started_or_resumed_at": datetime.now().isoformat(),
    }

    protocol_path = Path(drive_dirs["metadata"]) / "training_protocol_resume_safe.json"
    atomic_write_json(protocol_path, protocol)

    final_A = train_one_stage_resume_safe(
        world=world,
        stage="A",
        target_stage_steps=int(steps_A),
        drive_dirs=drive_dirs,
        seed=seed,
        n_obstacles=n_obstacles,
        previous_model_path=None,
        save_freq_steps=save_freq_steps,
        device=device,
    )

    final_B = train_one_stage_resume_safe(
        world=world,
        stage="B",
        target_stage_steps=int(steps_B),
        drive_dirs=drive_dirs,
        seed=seed + 100,
        n_obstacles=n_obstacles,
        previous_model_path=final_A,
        save_freq_steps=save_freq_steps,
        device=device,
    )

    final_C = train_one_stage_resume_safe(
        world=world,
        stage="C",
        target_stage_steps=int(steps_C),
        drive_dirs=drive_dirs,
        seed=seed + 200,
        n_obstacles=n_obstacles,
        previous_model_path=final_B,
        save_freq_steps=save_freq_steps,
        device=device,
    )

    print("=" * 100)
    print("ALL STAGES COMPLETE")
    print("=" * 100)
    print("final_A:", final_A)
    print("final_B:", final_B)
    print("final_C:", final_C)
    print("=" * 100)

    return {
        "final_A": final_A,
        "final_B": final_B,
        "final_C": final_C,
    }

def make_ruscuav_env_m4_mode(
    world,
    stage="A",
    mode="fast_train",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
):
    """
    mode:
        'fast_train'  -> szybszy trening
        'strict_eval' -> dokładna ewaluacja / paper results
    """
    mode = str(mode).lower()

    if mode == "fast_train":
        urban_lidar_step_m = 5.0
        urban_segment_check_step_m = 2.0

    elif mode == "strict_eval":
        urban_lidar_step_m = 3.0
        urban_segment_check_step_m = 1.0

    else:
        raise ValueError("mode must be 'fast_train' or 'strict_eval'.")

    stage = str(stage).upper()

    if stage == "A":
        cfg = RealisticUrbanEnvConfig(
            risk_profile=risk_profile,
            urban_risk_lambda=0.05,
            terminate_on_urban_constraint=False,
            hard_sensitive_zones=False,
            default_n_obstacles=int(n_obstacles),
        )

    elif stage == "B":
        cfg = RealisticUrbanEnvConfig(
            risk_profile=risk_profile,
            urban_risk_lambda=0.10,
            terminate_on_urban_constraint=False,
            hard_sensitive_zones=True,
            default_n_obstacles=int(n_obstacles),
        )

    elif stage == "C":
        cfg = RealisticUrbanEnvConfig(
            risk_profile=risk_profile,
            urban_risk_lambda=0.10,
            terminate_on_urban_constraint=True,
            hard_sensitive_zones=True,
            default_n_obstacles=int(n_obstacles),
            terminal_road_budget_s=30.0,
            terminal_sidewalk_budget_s=10.0,
            terminal_building_budget_s=20.0,
            terminal_sensitive_budget_s=0.0,
        )

    else:
        raise ValueError("stage must be A, B, or C.")

    env = RealisticUrbanCargoUAVEnvM4(
        world=world,
        config=cfg,
        include_shield_features=False,
        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,
        urban_lidar_step_m=urban_lidar_step_m,
        urban_segment_check_step_m=urban_segment_check_step_m,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        print(
            f"[make_ruscuav_env_m4_mode] "
            f"mode={mode}, stage={stage}, obs.shape={obs.shape}, info={info}"
        )

    return env

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

def make_ruscuav_m4_vec_env_from_world_path(
    world_path,
    stage,
    mode,
    rank,
    base_seed=1001,
    n_obstacles=0,
):
    def _init():
        w = UrbanWorldLayers.load_npz(world_path)

        env = make_ruscuav_env_m4_mode(
            world=w,
            stage=stage,
            mode=mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=base_seed + rank,
        )

        env.reset(seed=base_seed + rank)
        return env

    return _init

def make_vec_ruscuav_m4_env(
    world_path,
    stage="A",
    mode="fast_train",
    n_envs=2,
    base_seed=1001,
    n_obstacles=0,
    use_subproc=True,
):
    env_fns = [
        make_ruscuav_m4_vec_env_from_world_path(
            world_path=world_path,
            stage=stage,
            mode=mode,
            rank=i,
            base_seed=base_seed,
            n_obstacles=n_obstacles,
        )
        for i in range(int(n_envs))
    ]

    if use_subproc and int(n_envs) > 1:
        vec_env = SubprocVecEnv(env_fns, start_method="fork")
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(
        vec_env,
        info_keywords=RUSCUAV_M4_INFO_KEYWORDS,
    )

    return vec_env

import time

from stable_baselines3 import PPO

def benchmark_ruscuav_m4_speed(
    world_path,
    n_envs_list=(1, 2, 4),
    stage="A",
    mode="fast_train",
    timesteps=10_000,
    base_seed=1001,
):
    results = []

    for n_envs in n_envs_list:
        print("=" * 100)
        print(f"Benchmark n_envs={n_envs}, mode={mode}, timesteps={timesteps}")
        print("=" * 100)

        vec_env = make_vec_ruscuav_m4_env(
            world_path=world_path,
            stage=stage,
            mode=mode,
            n_envs=n_envs,
            base_seed=base_seed,
            n_obstacles=0,
            use_subproc=(n_envs > 1),
        )

        n_steps = 1024 if n_envs <= 2 else 512

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=base_seed,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=256,
            ent_coef=0.01,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            policy_kwargs=dict(net_arch=[256, 256]),
            device="cpu",
        )

        t0 = time.time()
        model.learn(total_timesteps=int(timesteps))
        elapsed = time.time() - t0

        fps = float(timesteps / max(elapsed, 1e-9))

        result = {
            "n_envs": int(n_envs),
            "mode": mode,
            "timesteps": int(timesteps),
            "elapsed_s": float(elapsed),
            "fps": fps,
            "n_steps": int(n_steps),
        }

        print(result)

        results.append(result)
        vec_env.close()

    return results

from stable_baselines3 import PPO

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

from stable_baselines3.common.callbacks import CallbackList

from pathlib import Path

import json

from datetime import datetime

def make_ruscuav_m4_vec_env_from_world_path(
    world_path,
    stage,
    mode,
    rank,
    base_seed=1001,
    n_obstacles=0,
):
    def _init():
        w = UrbanWorldLayers.load_npz(world_path)

        env = make_ruscuav_env_m4_mode(
            world=w,
            stage=stage,
            mode=mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=base_seed + rank,
        )

        env.reset(seed=base_seed + rank)
        return env

    return _init

def make_vec_ruscuav_m4_env(
    world_path,
    stage="A",
    mode="fast_train",
    n_envs=4,
    base_seed=1001,
    n_obstacles=0,
    use_subproc=True,
    start_method="fork",
):
    env_fns = [
        make_ruscuav_m4_vec_env_from_world_path(
            world_path=world_path,
            stage=stage,
            mode=mode,
            rank=i,
            base_seed=base_seed,
            n_obstacles=n_obstacles,
        )
        for i in range(int(n_envs))
    ]

    if use_subproc and int(n_envs) > 1:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(
        vec_env,
        info_keywords=RUSCUAV_M4_INFO_KEYWORDS,
    )

    return vec_env

def create_or_load_vec_model_for_stage(
    stage,
    vec_env,
    drive_dirs,
    previous_model_path=None,
    seed=1001,
    device="cpu",
    n_envs=4,
):
    paths = stage_paths_on_drive(drive_dirs, stage)

    model, loaded_from = load_stage_model_if_available(paths, env=vec_env, device=device)

    if model is not None:
        print(f"Resuming stage {stage} from:", loaded_from)
        return model, loaded_from

    if previous_model_path is not None and Path(previous_model_path).exists():
        print(f"Starting stage {stage} from previous stage model:", previous_model_path)
        model = PPO.load(str(previous_model_path), env=vec_env, device=device)
        return model, str(previous_model_path)

    print(f"Starting stage {stage} from scratch.")

    n_steps = 512 if int(n_envs) >= 4 else 1024

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=int(seed),
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
    )

    return model, None

def train_one_stage_vec_resume_safe(
    world_path,
    stage,
    target_stage_steps,
    drive_dirs,
    seed=1001,
    n_obstacles=0,
    previous_model_path=None,
    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    stage = str(stage).upper()
    paths = stage_paths_on_drive(drive_dirs, stage)

    if is_stage_done(paths, target_stage_steps):
        print(f"Stage {stage} already marked DONE. Skipping.")
        return str(paths["final"])

    vec_env = make_vec_ruscuav_m4_env(
        world_path=world_path,
        stage=stage,
        mode=mode,
        n_envs=n_envs,
        base_seed=seed,
        n_obstacles=n_obstacles,
        use_subproc=(n_envs > 1),
        start_method=start_method,
    )

    model, loaded_from = create_or_load_vec_model_for_stage(
        stage=stage,
        vec_env=vec_env,
        drive_dirs=drive_dirs,
        previous_model_path=previous_model_path,
        seed=seed,
        device=device,
        n_envs=n_envs,
    )

    progress = read_progress_json(paths["progress"])

    if progress is not None and Path(paths["latest"]).exists():
        stage_start_timesteps = int(progress["stage_start_timesteps"])
        completed_stage_steps = int(progress["completed_stage_steps"])
        print(f"Read progress for stage {stage}: completed={completed_stage_steps}/{target_stage_steps}")
    else:
        stage_start_timesteps = int(model.num_timesteps)
        completed_stage_steps = 0
        print(f"New progress for stage {stage}: stage_start_timesteps={stage_start_timesteps}")

    remaining = int(max(0, target_stage_steps - completed_stage_steps))

    print("=" * 100)
    print(f"TRAIN VEC STAGE {stage}")
    print("=" * 100)
    print("mode:", mode)
    print("n_envs:", n_envs)
    print("target_stage_steps:", target_stage_steps)
    print("completed_stage_steps:", completed_stage_steps)
    print("remaining:", remaining)
    print("model.num_timesteps:", model.num_timesteps)
    print("stage_start_timesteps:", stage_start_timesteps)
    print("stage dir:", paths["stage_dir"])
    print("=" * 100)

    if remaining <= 0:
        final_path = mark_stage_done(
            paths,
            stage_name=stage,
            model=model,
            completed_stage_steps=completed_stage_steps,
            target_stage_steps=target_stage_steps,
        )
        vec_env.close()
        return final_path

    autosave_cb = RUSCUAVDriveAutosaveCallback(
        stage_dir=paths["stage_dir"],
        stage_name=stage,
        target_stage_steps=target_stage_steps,
        stage_start_timesteps=stage_start_timesteps,
        save_freq_steps=save_freq_steps,
        keep_numbered_checkpoints=True,
        verbose=1,
    )

    stats_cb = RUSCUAVTrainStatsCallback(
        print_every_steps=10_000,
        window=100,
    )

    callback = CallbackList([autosave_cb, stats_cb])

    reset_num_timesteps = bool(model.num_timesteps == 0)

    try:
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        model.save(str(paths["latest"]))

        progress_now = {
            "stage": stage,
            "reason": "finally_block",
            "target_stage_steps": int(target_stage_steps),
            "stage_start_timesteps": int(stage_start_timesteps),
            "last_model_timesteps": int(model.num_timesteps),
            "completed_stage_steps": int(max(0, model.num_timesteps - stage_start_timesteps)),
            "saved_at": datetime.now().isoformat(),
            "latest_model_path": str(paths["latest"]),
            "mode": mode,
            "n_envs": int(n_envs),
        }

        atomic_write_json(paths["progress"], progress_now)
        print("Emergency/finally save completed:", paths["latest"])

    completed_final = int(max(0, model.num_timesteps - stage_start_timesteps))

    final_path = mark_stage_done(
        paths,
        stage_name=stage,
        model=model,
        completed_stage_steps=completed_final,
        target_stage_steps=target_stage_steps,
    )

    vec_env.close()

    return final_path

def train_ruscuav_m6_vec_drive_resume(
    world_path,
    drive_dirs,
    seed=1001,
    n_obstacles=0,
    steps_A=150_000,
    steps_B=250_000,
    steps_C=400_000,
    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    protocol = {
        "experiment": "RUSCUAV M6 vectorized M4 PPO",
        "world_path": str(world_path),
        "seed": int(seed),
        "n_obstacles": int(n_obstacles),
        "steps_A": int(steps_A),
        "steps_B": int(steps_B),
        "steps_C": int(steps_C),
        "device": device,
        "n_envs": int(n_envs),
        "mode": mode,
        "start_method": start_method,
        "started_or_resumed_at": datetime.now().isoformat(),
    }

    protocol_path = Path(drive_dirs["metadata"]) / "m6_training_protocol_vec_resume_safe.json"
    atomic_write_json(protocol_path, protocol)

    final_A = train_one_stage_vec_resume_safe(
        world_path=world_path,
        stage="A",
        target_stage_steps=int(steps_A),
        drive_dirs=drive_dirs,
        seed=seed,
        n_obstacles=n_obstacles,
        previous_model_path=None,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_B = train_one_stage_vec_resume_safe(
        world_path=world_path,
        stage="B",
        target_stage_steps=int(steps_B),
        drive_dirs=drive_dirs,
        seed=seed + 100,
        n_obstacles=n_obstacles,
        previous_model_path=final_A,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_C = train_one_stage_vec_resume_safe(
        world_path=world_path,
        stage="C",
        target_stage_steps=int(steps_C),
        drive_dirs=drive_dirs,
        seed=seed + 200,
        n_obstacles=n_obstacles,
        previous_model_path=final_B,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    print("=" * 100)
    print("M6 VECTOR TRAINING COMPLETE")
    print("=" * 100)
    print("final_A:", final_A)
    print("final_B:", final_B)
    print("final_C:", final_C)
    print("=" * 100)

    return {
        "final_A": final_A,
        "final_B": final_B,
        "final_C": final_C,
    }

WORLD_PATH_DRIVE = "/content/drive/MyDrive/RUSCUAV/worlds/AOI_FAKE_001_urban_world_layers.npz"

from pathlib import Path

from datetime import datetime

import json

import numpy as np

import pandas as pd

from stable_baselines3 import PPO

def _json_safe(obj):
    """
    Convert numpy / pandas objects to JSON-safe Python objects.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_json_safe(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if pd.isna(obj) if not isinstance(obj, (dict, list, tuple)) else False:
        return None
    return obj

def _atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_suffix(path.suffix + ".tmp")

    with open(tmp_path, "w") as f:
        json.dump(_json_safe(data), f, indent=2)

    tmp_path.replace(path)

def rollout_one_ruscuav_m4_eval_episode(
    model,
    world,
    seed,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    deterministic=True,
    max_steps=1000,
):
    """
    One deterministic evaluation rollout in strict_eval mode.
    """
    env = make_ruscuav_env_m4_mode(
        world=world,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, reset_info = env.reset(seed=seed)

    rewards = []
    infos = []
    path = [env.current_pos.copy()]

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(dict(info))
        path.append(env.current_pos.copy())

        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}

    if not done:
        final_info = dict(final_info)
        final_info["termination"] = "manual_max_steps"

    # Fallback distance if not present in info.
    if "dist_xy" not in final_info or final_info.get("dist_xy") is None:
        try:
            final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))
        except Exception:
            final_info["dist_xy"] = np.nan

    row = {
        "episode": None,  # filled outside
        "seed": int(seed),
        "stage": str(stage),
        "mode": str(mode),
        "n_obstacles": int(n_obstacles),

        "total_reward": float(np.sum(rewards)) if len(rewards) > 0 else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),

        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),

        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(final_info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),
        "urban_risk_last": float(final_info.get("urban_risk", np.nan)),

        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),

        "building_collision_steps": int(final_info.get("urban_building_collision_steps", 0)),
        "building_clearance_violation_steps": int(final_info.get("urban_building_clearance_violation_steps", 0)),
        "canopy_collision_steps": int(final_info.get("urban_canopy_collision_steps", 0)),
        "canopy_clearance_violation_steps": int(final_info.get("urban_canopy_clearance_violation_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(final_info.get("urban_sensitive_low_altitude_violation_steps", 0)),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),

        "urban_lidar_min_norm_last": float(final_info.get("urban_lidar_min_norm", np.nan)),
        "urban_signed_clearance_proxy_m_last": float(final_info.get("urban_signed_clearance_proxy_m", np.nan)),
        "urban_action_shielded_last": bool(final_info.get("urban_action_shielded", False)),
        "urban_geom_shielded_last": bool(final_info.get("urban_geom_shielded", False)),
        "urban_shielded_any_last": bool(final_info.get("urban_shielded_any", False)),

        "reset_urban_safe_start_target": bool(reset_info.get("urban_safe_start_target", False)),
        "reset_urban_attempts": int(reset_info.get("urban_reset_attempts", -1)),
        "start_urban_risk": float(reset_info.get("start_urban_risk", np.nan)),
        "target_urban_risk": float(reset_info.get("target_urban_risk", np.nan)),
    }

    path_arr = np.asarray(path, dtype=np.float32)

    env.close()

    return row, path_arr

def summarize_ruscuav_m4_eval_df(df):
    """
    Summary dictionary for RUSCUAV eval dataframe.
    """
    if len(df) == 0:
        return {"n_episodes": 0}

    df = df.copy()

    # Ensure boolean columns.
    for col in ["is_success", "mission_acceptable"]:
        if col in df.columns:
            df[col] = df[col].astype(bool)

    summary = {
        "n_episodes": int(len(df)),
        "success_rate": float(df["is_success"].mean()),
        "mission_acceptability_rate": float(df["mission_acceptable"].mean()),

        "mean_reward": float(df["total_reward"].mean()),
        "std_reward": float(df["total_reward"].std(ddof=0)),

        "mean_steps": float(df["steps"].mean()),
        "mean_dist_xy": float(df["dist_xy"].mean()),
        "median_dist_xy": float(df["dist_xy"].median()),

        "mean_cargo_health": float(df["cargo_health"].mean()),

        "mean_urban_risk": float(df["urban_mean_risk"].mean()),
        "mean_urban_max_risk": float(df["urban_max_risk"].mean()),
        "mean_cumulative_risk": float(df["urban_cumulative_risk"].mean()),

        "mean_road_time": float(df["road_time"].mean()),
        "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
        "mean_building_time": float(df["building_time"].mean()),
        "mean_green_time": float(df["green_time"].mean()),
        "mean_canopy_time": float(df["canopy_time"].mean()),
        "mean_sensitive_time": float(df["sensitive_time"].mean()),

        "mean_building_collision_steps": float(df["building_collision_steps"].mean()),
        "mean_canopy_collision_steps": float(df["canopy_collision_steps"].mean()),
        "mean_sensitive_low_altitude_violation_steps": float(df["sensitive_low_altitude_violation_steps"].mean()),
        "mean_hard_violation_steps": float(df["urban_hard_violation_steps"].mean()),

        "termination_counts": {
            str(k): int(v)
            for k, v in df["termination"].fillna("None").value_counts().to_dict().items()
        },
    }

    # Among successful missions, how many are acceptable?
    success_df = df[df["is_success"] == True]
    if len(success_df) > 0:
        summary["acceptable_given_success_rate"] = float(success_df["mission_acceptable"].mean())
    else:
        summary["acceptable_given_success_rate"] = 0.0

    return summary

def print_ruscuav_m4_eval_summary(summary):
    print("=" * 100)
    print("RUSCUAV M4 STRICT EVALUATION SUMMARY")
    print("=" * 100)

    ordered = [
        "n_episodes",
        "success_rate",
        "mission_acceptability_rate",
        "acceptable_given_success_rate",
        "mean_reward",
        "std_reward",
        "mean_steps",
        "mean_dist_xy",
        "median_dist_xy",
        "mean_cargo_health",
        "mean_urban_risk",
        "mean_urban_max_risk",
        "mean_cumulative_risk",
        "mean_road_time",
        "mean_sidewalk_time",
        "mean_building_time",
        "mean_green_time",
        "mean_canopy_time",
        "mean_sensitive_time",
        "mean_building_collision_steps",
        "mean_canopy_collision_steps",
        "mean_sensitive_low_altitude_violation_steps",
        "mean_hard_violation_steps",
    ]

    for k in ordered:
        if k not in summary:
            continue
        v = summary[k]
        if isinstance(v, float):
            print(f"{k:45s}: {v:.6f}")
        else:
            print(f"{k:45s}: {v}")

    print("-" * 100)
    print("termination_counts:")
    for k, v in summary.get("termination_counts", {}).items():
        print(f"  {k}: {v}")

    print("=" * 100)

def evaluate_and_save_ruscuav_m4_drive(
    model_path,
    world,
    drive_dirs,
    stage="C",
    n_episodes=200,
    seed0=5000,
    n_obstacles=0,
    deterministic=True,
    prefix=None,
    mode="strict_eval",
    max_steps=1000,
    save_paths=True,
):
    """
    Resume-safe strict evaluation.

    If CSV already exists, the function loads it and skips completed episodes.
    Saves:
      - episodes CSV
      - summary JSON
      - optional per-episode path NPY files
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if prefix is None:
        prefix = (
            f"eval_{model_path.stem}_"
            f"stage_{stage}_mode_{mode}_"
            f"n{n_episodes}_seed{seed0}"
        )

    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths_dir = eval_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"
    progress_path = eval_dir / f"{prefix}_progress.json"

    # Load existing partial eval if present.
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        completed_episodes = set(int(e) for e in df_existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming evaluation from existing CSV: {csv_path}")
        print(f"Completed episodes: {len(completed_episodes)} / {n_episodes}")
    else:
        rows = []
        completed_episodes = set()

    model = PPO.load(str(model_path), device="cpu")

    t0 = datetime.now()

    for ep in range(int(n_episodes)):
        if ep in completed_episodes:
            continue

        seed = int(seed0 + ep)

        row, path_arr = rollout_one_ruscuav_m4_eval_episode(
            model=model,
            world=world,
            seed=seed,
            stage=stage,
            mode=mode,
            n_obstacles=n_obstacles,
            deterministic=deterministic,
            max_steps=max_steps,
        )

        row["episode"] = int(ep)
        rows.append(row)

        if save_paths:
            np.save(paths_dir / f"episode_{ep:05d}_seed_{seed}_path.npy", path_arr)

        # Save after every episode.
        df_now = pd.DataFrame(rows).sort_values("episode")
        df_now.to_csv(csv_path, index=False)

        partial_summary = summarize_ruscuav_m4_eval_df(df_now)
        partial_summary["status"] = "partial" if len(df_now) < int(n_episodes) else "complete"
        partial_summary["last_completed_episode"] = int(ep)
        partial_summary["updated_at"] = datetime.now().isoformat()
        partial_summary["model_path"] = str(model_path)
        partial_summary["mode"] = mode
        partial_summary["stage"] = stage

        _atomic_write_json(summary_path, partial_summary)

        progress = {
            "status": partial_summary["status"],
            "completed_episodes": int(len(df_now)),
            "target_episodes": int(n_episodes),
            "last_completed_episode": int(ep),
            "last_seed": int(seed),
            "csv_path": str(csv_path),
            "summary_path": str(summary_path),
            "updated_at": datetime.now().isoformat(),
        }
        _atomic_write_json(progress_path, progress)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[eval] ep={ep+1}/{n_episodes} | "
                f"SR={partial_summary['success_rate']:.1%} | "
                f"MAR={partial_summary['mission_acceptability_rate']:.1%} | "
                f"term={row['termination']}"
            )

    df_final = pd.DataFrame(rows).sort_values("episode")
    final_summary = summarize_ruscuav_m4_eval_df(df_final)
    final_summary["status"] = "complete"
    final_summary["model_path"] = str(model_path)
    final_summary["mode"] = mode
    final_summary["stage"] = stage
    final_summary["n_episodes_requested"] = int(n_episodes)
    final_summary["seed0"] = int(seed0)
    final_summary["deterministic"] = bool(deterministic)
    final_summary["started_at"] = t0.isoformat()
    final_summary["finished_at"] = datetime.now().isoformat()
    final_summary["csv_path"] = str(csv_path)
    final_summary["summary_path"] = str(summary_path)

    df_final.to_csv(csv_path, index=False)
    _atomic_write_json(summary_path, final_summary)

    print_ruscuav_m4_eval_summary(final_summary)

    print("Saved eval CSV:", csv_path)
    print("Saved eval summary:", summary_path)
    if save_paths:
        print("Saved paths dir:", paths_dir)

    return df_final, final_summary, str(csv_path), str(summary_path)

import numpy as np

import pandas as pd

from pathlib import Path

from datetime import datetime

import matplotlib.pyplot as plt

def rollout_ruscuav_m4_for_visualization(
    model,
    world,
    seed=5000,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    deterministic=True,
    max_steps=1000,
):
    """
    Executes one rollout and records rich per-step telemetry.

    Recommended:
        mode="strict_eval" for all reported/visualized trajectories.
    """
    env = make_ruscuav_env_m4_mode(
        world=world,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, reset_info = env.reset(seed=seed)

    rows = []
    path = [env.current_pos.copy()]
    rewards = []

    done = False
    step_idx = 0

    while not done and step_idx < int(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        pos = env.current_pos.copy()
        path.append(pos)
        rewards.append(float(reward))

        flags = env.risk_model.flags_at(float(pos[0]), float(pos[1]))
        clear = env.risk_model.clearances_at(float(pos[0]), float(pos[1]), float(pos[2]))
        constraints = env.risk_model.constraints_at(float(pos[0]), float(pos[1]), float(pos[2]))

        row = {
            "step": int(step_idx + 1),
            "t_s": float((step_idx + 1) * env.dt),

            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "ground_z": float(env._ground_at(pos[0], pos[1])),
            "agl_m": float(pos[2] - env._ground_at(pos[0], pos[1])),

            "vx": float(env.vel[0]),
            "vy": float(env.vel[1]),
            "vz": float(env.vel[2]),

            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination": info.get("termination"),

            "is_success": bool(info.get("is_success", False)),
            "mission_acceptable": bool(info.get("mission_acceptable", False)),
            "cargo_health": float(info.get("cargo_health", np.nan)),
            "dist_xy": float(info.get("dist_xy", np.nan)),

            "urban_risk": float(info.get("urban_risk", np.nan)),
            "urban_mean_risk": float(info.get("urban_mean_risk", np.nan)),
            "urban_cumulative_risk": float(info.get("urban_cumulative_risk", np.nan)),

            "road_time": float(info.get("urban_road_time", 0.0)),
            "sidewalk_time": float(info.get("urban_sidewalk_time", 0.0)),
            "building_time": float(info.get("urban_building_time", 0.0)),
            "green_time": float(info.get("urban_green_time", 0.0)),
            "canopy_time": float(info.get("urban_canopy_time", 0.0)),
            "sensitive_time": float(info.get("urban_sensitive_time", 0.0)),

            "urban_action_shielded": bool(info.get("urban_action_shielded", False)),
            "urban_geom_shielded": bool(info.get("urban_geom_shielded", False)),
            "urban_shielded_any": bool(info.get("urban_shielded_any", False)),
            "urban_lidar_min_norm": float(info.get("urban_lidar_min_norm", np.nan)),
            "urban_signed_clearance_proxy_m": float(info.get("urban_signed_clearance_proxy_m", np.nan)),

            "flag_building": bool(flags["building"]),
            "flag_road": bool(flags["road"]),
            "flag_sidewalk": bool(flags["sidewalk"]),
            "flag_green": bool(flags["green"]),
            "flag_canopy": bool(flags["canopy"]),
            "flag_sensitive": bool(flags["sensitive"]),

            "terrain_clearance_m": float(clear["terrain_clearance_m"]),
            "building_clearance_m": float(clear["building_clearance_m"]),
            "canopy_clearance_m": float(clear["canopy_clearance_m"]),

            "building_collision": bool(constraints["building_collision"]),
            "canopy_collision": bool(constraints["canopy_collision"]),
            "sensitive_low_altitude_violation": bool(constraints["sensitive_low_altitude_violation"]),
            "any_hard_violation": bool(constraints["any_hard_violation"]),
        }

        rows.append(row)

        done = bool(terminated or truncated)
        step_idx += 1

    df = pd.DataFrame(rows)
    path = np.asarray(path, dtype=np.float32)

    final_info = rows[-1] if rows else {}

    result = {
        "seed": int(seed),
        "stage": stage,
        "mode": mode,
        "path": path,
        "df": df,
        "rewards": rewards,
        "reset_info": reset_info,
        "final_info": final_info,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(step_idx),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),
    }

    env.close()

    return result

def plot_ruscuav_rollout_2d(
    world,
    rollout,
    risk_profile="balanced",
    background="semantic",
    figsize=(12, 12),
    title=None,
):
    """
    background:
        'semantic' -> DEM + semantic overlays
        'risk'     -> base semantic risk map
        'terrain'  -> terrain only
    """
    path = rollout["path"]
    df = rollout["df"]

    fig, ax = plt.subplots(figsize=figsize)

    if background == "risk":
        risk = compute_base_semantic_risk_map(world, risk_profile=risk_profile)
        im = ax.imshow(risk, origin="lower", cmap="inferno")
        plt.colorbar(im, ax=ax, label="Base semantic risk")

    else:
        ax.imshow(world.terrain_z, origin="lower", cmap="terrain", alpha=0.80)

        def overlay(mask, cmap, alpha):
            arr = np.ma.masked_where(mask <= 0.5, mask)
            ax.imshow(arr, origin="lower", cmap=cmap, alpha=alpha)

        if background == "semantic":
            overlay(world.green_mask, "Greens", 0.30)
            overlay(world.road_mask, "Oranges", 0.45)
            overlay(world.sidewalk_mask, "Wistia", 0.45)
            overlay(world.building_mask, "Greys", 0.55)
            overlay(world.sensitive_mask, "Reds", 0.50)

    # Trajectory
    ax.plot(path[:, 0], path[:, 1], linewidth=2.5, label="UAV path")
    ax.scatter(path[0, 0], path[0, 1], s=100, marker="o", edgecolors="black", label="START")
    ax.scatter(path[-1, 0], path[-1, 1], s=120, marker="s", edgecolors="black", label="END")

    if title is None:
        title = (
            f"RUSCUAV rollout | seed={rollout['seed']} | "
            f"term={rollout['termination']} | "
            f"success={rollout['is_success']} | "
            f"acceptable={rollout['mission_acceptable']}"
        )

    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    return fig, ax

def plot_ruscuav_rollout_3d(
    world,
    rollout,
    risk_profile="balanced",
    stride=8,
    color_by="risk",
    show_buildings=True,
    show_canopy=True,
    show_roads=True,
    show_sidewalks=True,
    show_green=True,
    show_sensitive=True,
    max_buildings=120,
    title=None,
    html_path=None,
):
    """
    Interactive 3D Plotly visualization of one rollout.
    Saves to html_path if provided.
    """
    import plotly.graph_objects as go

    path = rollout["path"]
    df = rollout["df"]

    ys = np.arange(0, world.height, stride)
    xs = np.arange(0, world.width, stride)

    X, Y = np.meshgrid(xs, ys)
    Z = world.terrain_z[np.ix_(ys, xs)]

    risk = compute_base_semantic_risk_map(world, risk_profile=risk_profile)
    risk_ds = risk[np.ix_(ys, xs)]

    if color_by == "risk":
        surface_color = risk_ds
        colorscale = "Inferno"
        colorbar_title = "Base semantic risk"
    elif color_by == "terrain":
        surface_color = Z
        colorscale = "Earth"
        colorbar_title = "Terrain elevation"
    else:
        raise ValueError("color_by must be 'risk' or 'terrain'.")

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            surfacecolor=surface_color,
            colorscale=colorscale,
            opacity=0.78,
            colorbar=dict(title=colorbar_title),
            name="terrain",
            showscale=True,
        )
    )

    def add_masked_surface(Zsurf, mask, name, color, opacity):
        Zm = np.where(mask > 0.5, Zsurf, np.nan)
        if not np.isfinite(Zm).any():
            return
        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Zm,
                surfacecolor=np.ones_like(Zm, dtype=np.float32),
                colorscale=[[0.0, color], [1.0, color]],
                opacity=opacity,
                showscale=False,
                name=name,
                hoverinfo="skip",
            )
        )

    green_ds = world.green_mask[np.ix_(ys, xs)]
    road_ds = world.road_mask[np.ix_(ys, xs)]
    sidewalk_ds = world.sidewalk_mask[np.ix_(ys, xs)]
    sensitive_ds = world.sensitive_mask[np.ix_(ys, xs)]
    canopy_ds = world.canopy_mask[np.ix_(ys, xs)]
    canopy_top_ds = world.canopy_top_z[np.ix_(ys, xs)]

    if show_green:
        add_masked_surface(Z + 0.7, green_ds, "green", "green", 0.28)

    if show_roads:
        add_masked_surface(Z + 1.0, road_ds, "roads", "orange", 0.42)

    if show_sidewalks:
        add_masked_surface(Z + 1.3, sidewalk_ds, "sidewalks", "yellow", 0.38)

    if show_sensitive:
        add_masked_surface(Z + 2.0, sensitive_ds, "sensitive zones", "red", 0.55)

    if show_canopy:
        add_masked_surface(canopy_top_ds, canopy_ds, "canopy top", "darkgreen", 0.42)

    # Buildings as cuboids if helper exists
    if show_buildings and "_add_building_cuboids_world" in globals():
        _add_building_cuboids_world(fig, world, max_buildings=max_buildings)

    # UAV path line
    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines",
            line=dict(width=7, color="cyan"),
            name="UAV path",
        )
    )

    # Optional risk-colored path markers
    if len(df) > 0 and "urban_risk" in df.columns:
        fig.add_trace(
            go.Scatter3d(
                x=df["x"],
                y=df["y"],
                z=df["z"],
                mode="markers",
                marker=dict(
                    size=3,
                    color=df["urban_risk"],
                    colorscale="Turbo",
                    colorbar=dict(title="Path risk"),
                    opacity=0.85,
                ),
                name="path risk markers",
            )
        )

    # Start and end
    fig.add_trace(
        go.Scatter3d(
            x=[path[0, 0]],
            y=[path[0, 1]],
            z=[path[0, 2]],
            mode="markers+text",
            marker=dict(size=7, color="blue"),
            text=["START"],
            textposition="top center",
            name="START",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[path[-1, 0]],
            y=[path[-1, 1]],
            z=[path[-1, 2]],
            mode="markers+text",
            marker=dict(size=8, color="cyan"),
            text=["END"],
            textposition="top center",
            name="END",
        )
    )

    if title is None:
        title = (
            f"RUSCUAV 3D rollout | seed={rollout['seed']} | "
            f"term={rollout['termination']} | "
            f"success={rollout['is_success']} | "
            f"acceptable={rollout['mission_acceptable']}"
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X [m]",
            yaxis_title="Y [m]",
            zaxis_title="Z [m]",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.25),
        ),
        margin=dict(l=0, r=0, t=45, b=0),
    )

    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path))
        print("Saved 3D rollout HTML:", html_path)

    fig.show()
    return fig

def plot_ruscuav_rollout_timeseries(rollout, figsize=(10, 4)):
    df = rollout["df"]

    if len(df) == 0:
        print("Empty rollout dataframe.")
        return

    series_list = [
        ("urban_risk", "Urban risk"),
        ("agl_m", "Altitude AGL [m]"),
        ("cargo_health", "Cargo Health"),
        ("dist_xy", "Distance to target [m]"),
        ("urban_signed_clearance_proxy_m", "Urban clearance proxy [m]"),
        ("urban_lidar_min_norm", "Urban LiDAR min norm"),
    ]

    for col, title in series_list:
        if col not in df.columns:
            continue

        plt.figure(figsize=figsize)
        plt.plot(df["t_s"], df[col])
        plt.xlabel("Time [s]")
        plt.ylabel(col)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    exposure_cols = ["road_time", "sidewalk_time", "building_time", "green_time", "canopy_time", "sensitive_time"]
    available = [c for c in exposure_cols if c in df.columns]

    if available:
        plt.figure(figsize=(10, 4))
        for c in available:
            plt.plot(df["t_s"], df[c], label=c)
        plt.xlabel("Time [s]")
        plt.ylabel("Cumulative exposure time [s]")
        plt.title("Semantic exposure accumulation")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

def save_ruscuav_rollout_to_drive(
    rollout,
    drive_dirs,
    prefix=None,
):
    if prefix is None:
        prefix = (
            f"rollout_seed_{rollout['seed']}_"
            f"term_{rollout['termination']}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    out_dir = Path(drive_dirs["figures"]) / "rollouts"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{prefix}_trajectory.csv"
    npy_path = out_dir / f"{prefix}_path.npy"
    json_path = out_dir / f"{prefix}_summary.json"
    html_path = out_dir / f"{prefix}_3d.html"

    rollout["df"].to_csv(csv_path, index=False)
    np.save(npy_path, rollout["path"])

    summary = {
        "seed": int(rollout["seed"]),
        "stage": rollout["stage"],
        "mode": rollout["mode"],
        "termination": rollout["termination"],
        "is_success": bool(rollout["is_success"]),
        "mission_acceptable": bool(rollout["mission_acceptable"]),
        "total_reward": float(rollout["total_reward"]),
        "steps": int(rollout["steps"]),
        "final_info": {
            k: (v.item() if hasattr(v, "item") else v)
            for k, v in rollout["final_info"].items()
            if isinstance(v, (int, float, str, bool, type(None), np.integer, np.floating, np.bool_))
        },
    }

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved rollout CSV:", csv_path)
    print("Saved path NPY:", npy_path)
    print("Saved summary JSON:", json_path)

    return {
        "csv_path": str(csv_path),
        "npy_path": str(npy_path),
        "json_path": str(json_path),
        "html_path": str(html_path),
    }

from stable_baselines3 import PPO

from pathlib import Path

import json

from stable_baselines3 import PPO

import json

from pathlib import Path

def make_ruscuav_geo_baseline_env_m4_mode(
    world,
    mode="fast_train",
    stage="GEO",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
):
    """
    Geometry-only baseline.

    Training mode:
        - 80D observation only.
        - no urban semantic vector.
        - no urban risk reward.
        - no urban terminal constraints.
        - physical urban safety still active for buildings/canopy.

    Strict evaluation mode:
        - still 80D observation.
        - no urban risk reward.
        - urban terminal constraints active.
        - mission_acceptability and exposure metrics are computed.
    """
    mode = str(mode).lower()

    if mode == "fast_train":
        urban_lidar_step_m = 5.0
        urban_segment_check_step_m = 2.0
        terminate_on_urban_constraint = False
        hard_sensitive_zones = False

    elif mode == "strict_eval":
        urban_lidar_step_m = 3.0
        urban_segment_check_step_m = 1.0
        terminate_on_urban_constraint = True
        hard_sensitive_zones = True

    else:
        raise ValueError("mode must be 'fast_train' or 'strict_eval'.")

    cfg = RealisticUrbanEnvConfig(
        risk_profile=risk_profile,

        # Key M7 geometry-only settings
        include_urban_vector_obs=False,
        use_urban_risk_costs=False,
        urban_risk_lambda=0.0,
        green_bonus_lambda=0.0,

        # Still use urban altitude reference for physical buildings/canopy.
        use_urban_altitude_reference=True,

        # In training: off. In strict eval: on.
        terminate_on_urban_constraint=terminate_on_urban_constraint,
        hard_sensitive_zones=hard_sensitive_zones,

        default_n_obstacles=int(n_obstacles),

        # Same mission acceptability budgets as full M4
        mission_accept_cargo_min=0.80,
        mission_accept_road_budget_s=20.0,
        mission_accept_sidewalk_budget_s=5.0,
        mission_accept_building_budget_s=15.0,
        mission_accept_sensitive_budget_s=0.0,

        # Same terminal budgets for strict eval
        terminal_road_budget_s=30.0,
        terminal_sidewalk_budget_s=10.0,
        terminal_building_budget_s=20.0,
        terminal_sensitive_budget_s=0.0,
    )

    env = RealisticUrbanCargoUAVEnvM4(
        world=world,
        config=cfg,
        include_shield_features=False,

        # Keep physical safety stack active.
        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,

        urban_lidar_step_m=urban_lidar_step_m,
        urban_segment_check_step_m=urban_segment_check_step_m,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        print(
            f"[make_geo_baseline_m4] mode={mode}, obs.shape={obs.shape}, "
            f"observation_space={env.observation_space}, info={info}"
        )

    return env

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

def make_geo_baseline_vec_env_from_world_path(
    world_path,
    mode,
    rank,
    base_seed=1001,
    n_obstacles=0,
):
    def _init():
        w = UrbanWorldLayers.load_npz(world_path)

        env = make_ruscuav_geo_baseline_env_m4_mode(
            world=w,
            mode=mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=base_seed + rank,
        )

        env.reset(seed=base_seed + rank)
        return env

    return _init

def make_vec_geo_baseline_env_m4(
    world_path,
    mode="fast_train",
    n_envs=4,
    base_seed=1001,
    n_obstacles=0,
    use_subproc=True,
    start_method="fork",
):
    env_fns = [
        make_geo_baseline_vec_env_from_world_path(
            world_path=world_path,
            mode=mode,
            rank=i,
            base_seed=base_seed,
            n_obstacles=n_obstacles,
        )
        for i in range(int(n_envs))
    ]

    if use_subproc and int(n_envs) > 1:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(
        vec_env,
        info_keywords=RUSCUAV_M4_INFO_KEYWORDS,
    )

    return vec_env

from pathlib import Path

from datetime import datetime

import json

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList

def create_or_load_geo_baseline_model(
    vec_env,
    drive_dirs,
    stage_name="GEO",
    seed=1001,
    device="cpu",
    n_envs=4,
):
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    model, loaded_from = load_stage_model_if_available(paths, env=vec_env, device=device)

    if model is not None:
        print("Resuming geometry-only baseline from:", loaded_from)
        return model, loaded_from

    print("Starting geometry-only baseline from scratch.")

    n_steps = 512 if int(n_envs) >= 4 else 1024

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=int(seed),
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
    )

    return model, None

def train_geo_baseline_m7_drive_resume(
    world_path,
    drive_dirs,
    total_steps=800_000,
    seed=2001,
    n_obstacles=0,
    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    """
    Geometry-only baseline training.

    It uses:
        - 80D observation;
        - no semantic vector;
        - no semantic risk reward;
        - physical urban safety stack active;
        - fast_train mode for speed.

    Evaluation must be done in strict_eval.
    """
    stage_name = "GEO"
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    protocol = {
        "experiment": "RUSCUAV M7 geometry-only baseline",
        "world_path": str(world_path),
        "stage_name": stage_name,
        "seed": int(seed),
        "n_obstacles": int(n_obstacles),
        "total_steps": int(total_steps),
        "device": device,
        "n_envs": int(n_envs),
        "mode": mode,
        "start_method": start_method,
        "key_settings": {
            "include_urban_vector_obs": False,
            "use_urban_risk_costs": False,
            "use_physical_urban_safety": True,
            "train_fast_eval_strict": True,
        },
        "started_or_resumed_at": datetime.now().isoformat(),
    }

    protocol_path = Path(drive_dirs["metadata"]) / "m7_geometry_only_training_protocol.json"
    atomic_write_json(protocol_path, protocol)

    if is_stage_done(paths, total_steps):
        print("M7 geometry-only baseline already DONE. Skipping.")
        return str(paths["final"])

    vec_env = make_vec_geo_baseline_env_m4(
        world_path=world_path,
        mode=mode,
        n_envs=n_envs,
        base_seed=seed,
        n_obstacles=n_obstacles,
        use_subproc=(n_envs > 1),
        start_method=start_method,
    )

    model, loaded_from = create_or_load_geo_baseline_model(
        vec_env=vec_env,
        drive_dirs=drive_dirs,
        stage_name=stage_name,
        seed=seed,
        device=device,
        n_envs=n_envs,
    )

    progress = read_progress_json(paths["progress"])

    if progress is not None and Path(paths["latest"]).exists():
        stage_start_timesteps = int(progress["stage_start_timesteps"])
        completed_steps = int(progress["completed_stage_steps"])
        print(f"Read GEO progress: completed={completed_steps}/{total_steps}")
    else:
        stage_start_timesteps = int(model.num_timesteps)
        completed_steps = 0
        print(f"New GEO progress: stage_start_timesteps={stage_start_timesteps}")

    remaining = int(max(0, total_steps - completed_steps))

    print("=" * 100)
    print("TRAIN M7 GEOMETRY-ONLY BASELINE")
    print("=" * 100)
    print("total_steps:", total_steps)
    print("completed_steps:", completed_steps)
    print("remaining:", remaining)
    print("n_envs:", n_envs)
    print("mode:", mode)
    print("model.num_timesteps:", model.num_timesteps)
    print("stage_start_timesteps:", stage_start_timesteps)
    print("stage dir:", paths["stage_dir"])
    print("=" * 100)

    if remaining <= 0:
        final_path = mark_stage_done(
            paths,
            stage_name=stage_name,
            model=model,
            completed_stage_steps=completed_steps,
            target_stage_steps=total_steps,
        )
        vec_env.close()
        return final_path

    autosave_cb = RUSCUAVDriveAutosaveCallback(
        stage_dir=paths["stage_dir"],
        stage_name=stage_name,
        target_stage_steps=total_steps,
        stage_start_timesteps=stage_start_timesteps,
        save_freq_steps=save_freq_steps,
        keep_numbered_checkpoints=True,
        verbose=1,
    )

    stats_cb = RUSCUAVTrainStatsCallback(
        print_every_steps=10_000,
        window=100,
    )

    callback = CallbackList([autosave_cb, stats_cb])

    reset_num_timesteps = bool(model.num_timesteps == 0)

    try:
        model.learn(
            total_timesteps=remaining,
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        model.save(str(paths["latest"]))

        progress_now = {
            "stage": stage_name,
            "reason": "finally_block",
            "target_stage_steps": int(total_steps),
            "stage_start_timesteps": int(stage_start_timesteps),
            "last_model_timesteps": int(model.num_timesteps),
            "completed_stage_steps": int(max(0, model.num_timesteps - stage_start_timesteps)),
            "saved_at": datetime.now().isoformat(),
            "latest_model_path": str(paths["latest"]),
            "mode": mode,
            "n_envs": int(n_envs),
        }

        atomic_write_json(paths["progress"], progress_now)
        print("Emergency/finally save completed:", paths["latest"])

    completed_final = int(max(0, model.num_timesteps - stage_start_timesteps))

    final_path = mark_stage_done(
        paths,
        stage_name=stage_name,
        model=model,
        completed_stage_steps=completed_final,
        target_stage_steps=total_steps,
    )

    vec_env.close()

    return final_path

import numpy as np

import pandas as pd

from pathlib import Path

from stable_baselines3 import PPO

def rollout_geo_baseline_eval_episode(
    model,
    world,
    seed,
    mode="strict_eval",
    n_obstacles=0,
    deterministic=True,
    max_steps=1000,
):
    env = make_ruscuav_geo_baseline_env_m4_mode(
        world=world,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, reset_info = env.reset(seed=seed)

    rewards = []
    infos = []
    path = [env.current_pos.copy()]

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(dict(info))
        path.append(env.current_pos.copy())

        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}

    if "dist_xy" not in final_info or final_info.get("dist_xy") is None:
        try:
            final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))
        except Exception:
            final_info["dist_xy"] = np.nan

    row = {
        "episode": None,
        "seed": int(seed),
        "mode": str(mode),
        "n_obstacles": int(n_obstacles),

        "total_reward": float(np.sum(rewards)) if len(rewards) > 0 else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),

        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),

        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(final_info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),
        "urban_risk_last": float(final_info.get("urban_risk", np.nan)),

        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),

        "building_collision_steps": int(final_info.get("urban_building_collision_steps", 0)),
        "building_clearance_violation_steps": int(final_info.get("urban_building_clearance_violation_steps", 0)),
        "canopy_collision_steps": int(final_info.get("urban_canopy_collision_steps", 0)),
        "canopy_clearance_violation_steps": int(final_info.get("urban_canopy_clearance_violation_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(final_info.get("urban_sensitive_low_altitude_violation_steps", 0)),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),

        "reset_urban_safe_start_target": bool(reset_info.get("urban_safe_start_target", False)),
        "reset_urban_attempts": int(reset_info.get("urban_reset_attempts", -1)),
        "start_urban_risk": float(reset_info.get("start_urban_risk", np.nan)),
        "target_urban_risk": float(reset_info.get("target_urban_risk", np.nan)),
    }

    path_arr = np.asarray(path, dtype=np.float32)

    env.close()

    return row, path_arr

def evaluate_geo_baseline_m7_drive(
    model_path,
    world,
    drive_dirs,
    n_episodes=200,
    seed0=5000,
    n_obstacles=0,
    deterministic=True,
    prefix="m7_geometry_only_strict_eval",
    mode="strict_eval",
    max_steps=1000,
    save_paths=True,
):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths_dir = eval_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        completed = set(int(e) for e in df_existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming M7 eval from {csv_path}: {len(completed)}/{n_episodes}")
    else:
        rows = []
        completed = set()

    model = PPO.load(str(model_path), device="cpu")

    for ep in range(int(n_episodes)):
        if ep in completed:
            continue

        seed = int(seed0 + ep)

        row, path_arr = rollout_geo_baseline_eval_episode(
            model=model,
            world=world,
            seed=seed,
            mode=mode,
            n_obstacles=n_obstacles,
            deterministic=deterministic,
            max_steps=max_steps,
        )

        row["episode"] = int(ep)
        rows.append(row)

        if save_paths:
            np.save(paths_dir / f"episode_{ep:05d}_seed_{seed}_path.npy", path_arr)

        df_now = pd.DataFrame(rows).sort_values("episode")
        df_now.to_csv(csv_path, index=False)

        summary_now = summarize_ruscuav_m4_eval_df(df_now)
        summary_now["status"] = "partial" if len(df_now) < int(n_episodes) else "complete"
        summary_now["model_path"] = str(model_path)
        summary_now["mode"] = mode
        summary_now["baseline"] = "geometry_only"
        summary_now["updated_at"] = datetime.now().isoformat()

        _atomic_write_json(summary_path, summary_now)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[M7 eval] ep={ep+1}/{n_episodes} | "
                f"SR={summary_now['success_rate']:.1%} | "
                f"MAR={summary_now['mission_acceptability_rate']:.1%} | "
                f"term={row['termination']}"
            )

    df_final = pd.DataFrame(rows).sort_values("episode")
    summary = summarize_ruscuav_m4_eval_df(df_final)
    summary["status"] = "complete"
    summary["model_path"] = str(model_path)
    summary["mode"] = mode
    summary["baseline"] = "geometry_only"
    summary["n_episodes_requested"] = int(n_episodes)
    summary["seed0"] = int(seed0)
    summary["deterministic"] = bool(deterministic)
    summary["csv_path"] = str(csv_path)
    summary["summary_path"] = str(summary_path)
    summary["finished_at"] = datetime.now().isoformat()

    df_final.to_csv(csv_path, index=False)
    _atomic_write_json(summary_path, summary)

    print_ruscuav_m4_eval_summary(summary)

    print("Saved M7 eval CSV:", csv_path)
    print("Saved M7 summary:", summary_path)

    return df_final, summary, str(csv_path), str(summary_path)

WORLD_PATH_DRIVE = "/content/drive/MyDrive/RUSCUAV/worlds/AOI_FAKE_001_urban_world_layers.npz"

def posthoc_acceptability(df, road_budget, sidewalk_budget, building_budget, sensitive_budget=0.0, cargo_min=0.8):
    return (
        (df["is_success"] == True)
        & (df["cargo_health"] >= cargo_min)
        & (df["road_time"] <= road_budget)
        & (df["sidewalk_time"] <= sidewalk_budget)
        & (df["building_time"] <= building_budget)
        & (df["sensitive_time"] <= sensitive_budget)
        & (df["urban_hard_violation_steps"] == 0)
    )

from scipy.stats import binomtest

import json

from pathlib import Path

M8_VARIANTS = {
    "reward_only": {
        "include_urban_vector_obs": False,
        "use_urban_risk_costs": True,
        "green_bonus_lambda": 0.03,
        "urban_lidar_enabled": True,
        "urban_action_shield_enabled": True,
        "urban_geometric_shield_enabled": True,
        "description": "No semantic vector obs; semantic risk reward active; full urban safety.",
    },

    "obs_only": {
        "include_urban_vector_obs": True,
        "use_urban_risk_costs": False,
        "green_bonus_lambda": 0.0,
        "urban_lidar_enabled": True,
        "urban_action_shield_enabled": True,
        "urban_geometric_shield_enabled": True,
        "description": "Semantic vector obs active; no semantic risk reward; full urban safety.",
    },

    "no_urban_lidar": {
        "include_urban_vector_obs": True,
        "use_urban_risk_costs": True,
        "green_bonus_lambda": 0.03,
        "urban_lidar_enabled": False,
        "urban_action_shield_enabled": True,
        "urban_geometric_shield_enabled": True,
        "description": "Full semantic learning; urban LiDAR disabled; shields active.",
    },

    "no_action_shield": {
        "include_urban_vector_obs": True,
        "use_urban_risk_costs": True,
        "green_bonus_lambda": 0.03,
        "urban_lidar_enabled": True,
        "urban_action_shield_enabled": False,
        "urban_geometric_shield_enabled": True,
        "description": "Full semantic learning; action shield disabled; geometric shield active.",
    },

    "no_geom_shield": {
        "include_urban_vector_obs": True,
        "use_urban_risk_costs": True,
        "green_bonus_lambda": 0.03,
        "urban_lidar_enabled": True,
        "urban_action_shield_enabled": True,
        "urban_geometric_shield_enabled": False,
        "description": "Full semantic learning; geometric shield disabled; action shield active.",
    },
}

def make_ruscuav_m8_ablation_env_m4_mode(
    world,
    variant_name,
    stage="A",
    mode="fast_train",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
):
    """
    Creates one M8 ablation environment.

    stage:
        A/B/C

    mode:
        fast_train or strict_eval

    variant_name:
        one of M8_VARIANTS
    """
    if variant_name not in M8_VARIANTS:
        raise ValueError(f"Unknown variant_name={variant_name}. Available: {list(M8_VARIANTS.keys())}")

    variant = M8_VARIANTS[variant_name]
    stage = str(stage).upper()
    mode = str(mode).lower()

    if mode == "fast_train":
        urban_lidar_step_m = 5.0
        urban_segment_check_step_m = 2.0
    elif mode == "strict_eval":
        urban_lidar_step_m = 3.0
        urban_segment_check_step_m = 1.0
    else:
        raise ValueError("mode must be 'fast_train' or 'strict_eval'.")

    if stage == "A":
        urban_risk_lambda = 0.05
        terminate_on_urban_constraint = False
        hard_sensitive_zones = False

    elif stage == "B":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = False
        hard_sensitive_zones = True

    elif stage == "C":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = True
        hard_sensitive_zones = True

    else:
        raise ValueError("stage must be A, B, or C.")

    cfg = RealisticUrbanEnvConfig(
        risk_profile=risk_profile,

        include_urban_vector_obs=bool(variant["include_urban_vector_obs"]),
        use_urban_risk_costs=bool(variant["use_urban_risk_costs"]),
        urban_risk_lambda=float(urban_risk_lambda),
        green_bonus_lambda=float(variant["green_bonus_lambda"]),

        use_urban_altitude_reference=True,

        terminate_on_urban_constraint=bool(terminate_on_urban_constraint),
        hard_sensitive_zones=bool(hard_sensitive_zones),

        default_n_obstacles=int(n_obstacles),

        mission_accept_cargo_min=0.80,
        mission_accept_road_budget_s=20.0,
        mission_accept_sidewalk_budget_s=5.0,
        mission_accept_building_budget_s=15.0,
        mission_accept_sensitive_budget_s=0.0,

        terminal_road_budget_s=30.0,
        terminal_sidewalk_budget_s=10.0,
        terminal_building_budget_s=20.0,
        terminal_sensitive_budget_s=0.0,
    )

    env = RealisticUrbanCargoUAVEnvM4(
        world=world,
        config=cfg,
        include_shield_features=False,

        urban_lidar_enabled=bool(variant["urban_lidar_enabled"]),
        urban_action_shield_enabled=bool(variant["urban_action_shield_enabled"]),
        urban_geometric_shield_enabled=bool(variant["urban_geometric_shield_enabled"]),

        urban_lidar_step_m=urban_lidar_step_m,
        urban_segment_check_step_m=urban_segment_check_step_m,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        print(
            f"[M8 env] variant={variant_name}, stage={stage}, mode={mode}, "
            f"obs.shape={obs.shape}, observation_space={env.observation_space}, info={info}"
        )

    return env

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

def make_m8_ablation_vec_env_from_world_path(
    world_path,
    variant_name,
    stage,
    mode,
    rank,
    base_seed=1001,
    n_obstacles=0,
):
    def _init():
        w = UrbanWorldLayers.load_npz(world_path)

        env = make_ruscuav_m8_ablation_env_m4_mode(
            world=w,
            variant_name=variant_name,
            stage=stage,
            mode=mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=base_seed + rank,
        )

        env.reset(seed=base_seed + rank)
        return env

    return _init

def make_vec_m8_ablation_env(
    world_path,
    variant_name,
    stage="A",
    mode="fast_train",
    n_envs=4,
    base_seed=1001,
    n_obstacles=0,
    use_subproc=True,
    start_method="fork",
):
    env_fns = [
        make_m8_ablation_vec_env_from_world_path(
            world_path=world_path,
            variant_name=variant_name,
            stage=stage,
            mode=mode,
            rank=i,
            base_seed=base_seed,
            n_obstacles=n_obstacles,
        )
        for i in range(int(n_envs))
    ]

    if use_subproc and int(n_envs) > 1:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(
        vec_env,
        info_keywords=RUSCUAV_M4_INFO_KEYWORDS,
    )

    return vec_env

from pathlib import Path

from datetime import datetime

import json

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList

def _m8_stage_name(variant_name, stage):
    return f"M8_{variant_name}_{stage}"

def create_or_load_m8_model_for_stage(
    variant_name,
    stage,
    vec_env,
    drive_dirs,
    previous_model_path=None,
    seed=1001,
    device="cpu",
    n_envs=4,
):
    stage_name = _m8_stage_name(variant_name, stage)
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    model, loaded_from = load_stage_model_if_available(paths, env=vec_env, device=device)

    if model is not None:
        print(f"Resuming M8 {variant_name} stage {stage} from:", loaded_from)
        return model, loaded_from

    if previous_model_path is not None and Path(previous_model_path).exists():
        print(f"Starting M8 {variant_name} stage {stage} from previous stage:", previous_model_path)
        model = PPO.load(str(previous_model_path), env=vec_env, device=device)
        return model, str(previous_model_path)

    print(f"Starting M8 {variant_name} stage {stage} from scratch.")

    n_steps = 512 if int(n_envs) >= 4 else 1024

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=int(seed),
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
    )

    return model, None

def train_one_m8_stage_resume_safe(
    world_path,
    variant_name,
    stage,
    target_stage_steps,
    drive_dirs,
    seed=1001,
    n_obstacles=0,
    previous_model_path=None,
    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    stage = str(stage).upper()
    stage_name = _m8_stage_name(variant_name, stage)
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    if is_stage_done(paths, target_stage_steps):
        print(f"M8 {variant_name} stage {stage} already DONE. Skipping.")
        return str(paths["final"])

    vec_env = make_vec_m8_ablation_env(
        world_path=world_path,
        variant_name=variant_name,
        stage=stage,
        mode=mode,
        n_envs=n_envs,
        base_seed=seed,
        n_obstacles=n_obstacles,
        use_subproc=(n_envs > 1),
        start_method=start_method,
    )

    model, loaded_from = create_or_load_m8_model_for_stage(
        variant_name=variant_name,
        stage=stage,
        vec_env=vec_env,
        drive_dirs=drive_dirs,
        previous_model_path=previous_model_path,
        seed=seed,
        device=device,
        n_envs=n_envs,
    )

    progress = read_progress_json(paths["progress"])

    if progress is not None and Path(paths["latest"]).exists():
        stage_start_timesteps = int(progress["stage_start_timesteps"])
        completed_stage_steps = int(progress["completed_stage_steps"])
        print(f"Read progress {stage_name}: completed={completed_stage_steps}/{target_stage_steps}")
    else:
        stage_start_timesteps = int(model.num_timesteps)
        completed_stage_steps = 0
        print(f"New progress {stage_name}: stage_start_timesteps={stage_start_timesteps}")

    remaining = int(max(0, target_stage_steps - completed_stage_steps))

    print("=" * 100)
    print(f"TRAIN M8 VARIANT={variant_name} STAGE={stage}")
    print("=" * 100)
    print("target_stage_steps:", target_stage_steps)
    print("completed_stage_steps:", completed_stage_steps)
    print("remaining:", remaining)
    print("model.num_timesteps:", model.num_timesteps)
    print("stage_start_timesteps:", stage_start_timesteps)
    print("stage dir:", paths["stage_dir"])
    print("=" * 100)

    if remaining <= 0:
        final_path = mark_stage_done(
            paths,
            stage_name=stage_name,
            model=model,
            completed_stage_steps=completed_stage_steps,
            target_stage_steps=target_stage_steps,
        )
        vec_env.close()
        return final_path

    autosave_cb = RUSCUAVDriveAutosaveCallback(
        stage_dir=paths["stage_dir"],
        stage_name=stage_name,
        target_stage_steps=target_stage_steps,
        stage_start_timesteps=stage_start_timesteps,
        save_freq_steps=save_freq_steps,
        keep_numbered_checkpoints=True,
        verbose=1,
    )

    stats_cb = RUSCUAVTrainStatsCallback(
        print_every_steps=10_000,
        window=100,
    )

    callback = CallbackList([autosave_cb, stats_cb])
    reset_num_timesteps = bool(model.num_timesteps == 0)

    try:
        model.learn(
            total_timesteps=int(remaining),
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        model.save(str(paths["latest"]))

        progress_now = {
            "stage": stage_name,
            "variant_name": variant_name,
            "reason": "finally_block",
            "target_stage_steps": int(target_stage_steps),
            "stage_start_timesteps": int(stage_start_timesteps),
            "last_model_timesteps": int(model.num_timesteps),
            "completed_stage_steps": int(max(0, model.num_timesteps - stage_start_timesteps)),
            "saved_at": datetime.now().isoformat(),
            "latest_model_path": str(paths["latest"]),
            "mode": mode,
            "n_envs": int(n_envs),
        }

        atomic_write_json(paths["progress"], progress_now)
        print("Emergency/finally save completed:", paths["latest"])

    completed_final = int(max(0, model.num_timesteps - stage_start_timesteps))

    final_path = mark_stage_done(
        paths,
        stage_name=stage_name,
        model=model,
        completed_stage_steps=completed_final,
        target_stage_steps=target_stage_steps,
    )

    vec_env.close()

    return final_path

def train_m8_ablation_variant_drive_resume(
    world_path,
    variant_name,
    drive_dirs,
    seed=3001,
    n_obstacles=0,
    steps_A=50_000,
    steps_B=75_000,
    steps_C=75_000,
    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    if variant_name not in M8_VARIANTS:
        raise ValueError(f"Unknown variant_name={variant_name}")

    protocol = {
        "experiment": "RUSCUAV M8 ablation",
        "variant_name": variant_name,
        "variant_description": M8_VARIANTS[variant_name]["description"],
        "world_path": str(world_path),
        "seed": int(seed),
        "n_obstacles": int(n_obstacles),
        "steps_A": int(steps_A),
        "steps_B": int(steps_B),
        "steps_C": int(steps_C),
        "device": device,
        "n_envs": int(n_envs),
        "mode": mode,
        "start_method": start_method,
        "started_or_resumed_at": datetime.now().isoformat(),
    }

    protocol_path = Path(drive_dirs["metadata"]) / f"m8_protocol_{variant_name}.json"
    atomic_write_json(protocol_path, protocol)

    final_A = train_one_m8_stage_resume_safe(
        world_path=world_path,
        variant_name=variant_name,
        stage="A",
        target_stage_steps=int(steps_A),
        drive_dirs=drive_dirs,
        seed=seed,
        n_obstacles=n_obstacles,
        previous_model_path=None,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_B = train_one_m8_stage_resume_safe(
        world_path=world_path,
        variant_name=variant_name,
        stage="B",
        target_stage_steps=int(steps_B),
        drive_dirs=drive_dirs,
        seed=seed + 100,
        n_obstacles=n_obstacles,
        previous_model_path=final_A,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_C = train_one_m8_stage_resume_safe(
        world_path=world_path,
        variant_name=variant_name,
        stage="C",
        target_stage_steps=int(steps_C),
        drive_dirs=drive_dirs,
        seed=seed + 200,
        n_obstacles=n_obstacles,
        previous_model_path=final_B,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    return {
        "final_A": final_A,
        "final_B": final_B,
        "final_C": final_C,
    }

import pandas as pd

import numpy as np

from pathlib import Path

from stable_baselines3 import PPO

def rollout_m8_ablation_eval_episode(
    model,
    world,
    variant_name,
    seed,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    deterministic=True,
    max_steps=1000,
):
    env = make_ruscuav_m8_ablation_env_m4_mode(
        world=world,
        variant_name=variant_name,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, reset_info = env.reset(seed=seed)

    rewards = []
    infos = []
    path = [env.current_pos.copy()]

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(dict(info))
        path.append(env.current_pos.copy())

        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}

    if "dist_xy" not in final_info or final_info.get("dist_xy") is None:
        try:
            final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))
        except Exception:
            final_info["dist_xy"] = np.nan

    row = {
        "episode": None,
        "seed": int(seed),
        "variant_name": variant_name,
        "stage": stage,
        "mode": mode,

        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),

        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),

        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(final_info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),

        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),

        "building_collision_steps": int(final_info.get("urban_building_collision_steps", 0)),
        "canopy_collision_steps": int(final_info.get("urban_canopy_collision_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(final_info.get("urban_sensitive_low_altitude_violation_steps", 0)),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),
    }

    path_arr = np.asarray(path, dtype=np.float32)

    env.close()

    return row, path_arr

def evaluate_m8_ablation_variant_drive(
    model_path,
    world,
    variant_name,
    drive_dirs,
    n_episodes=100,
    seed0=5000,
    n_obstacles=0,
    deterministic=True,
    prefix=None,
    mode="strict_eval",
    max_steps=1000,
    save_paths=False,
):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if prefix is None:
        prefix = f"m8_{variant_name}_strict_eval"

    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths_dir = eval_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        completed = set(int(e) for e in df_existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming M8 eval {variant_name}: {len(completed)}/{n_episodes}")
    else:
        rows = []
        completed = set()

    model = PPO.load(str(model_path), device="cpu")

    for ep in range(int(n_episodes)):
        if ep in completed:
            continue

        seed = int(seed0 + ep)

        row, path_arr = rollout_m8_ablation_eval_episode(
            model=model,
            world=world,
            variant_name=variant_name,
            seed=seed,
            stage="C",
            mode=mode,
            n_obstacles=n_obstacles,
            deterministic=deterministic,
            max_steps=max_steps,
        )

        row["episode"] = int(ep)
        rows.append(row)

        if save_paths:
            np.save(paths_dir / f"episode_{ep:05d}_seed_{seed}_path.npy", path_arr)

        df_now = pd.DataFrame(rows).sort_values("episode")
        df_now.to_csv(csv_path, index=False)

        summary_now = summarize_ruscuav_m4_eval_df(df_now)
        summary_now["variant_name"] = variant_name
        summary_now["status"] = "partial" if len(df_now) < int(n_episodes) else "complete"
        summary_now["model_path"] = str(model_path)
        summary_now["mode"] = mode
        summary_now["updated_at"] = datetime.now().isoformat()

        _atomic_write_json(summary_path, summary_now)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[M8 eval {variant_name}] ep={ep+1}/{n_episodes} | "
                f"SR={summary_now['success_rate']:.1%} | "
                f"MAR={summary_now['mission_acceptability_rate']:.1%} | "
                f"term={row['termination']}"
            )

    df_final = pd.DataFrame(rows).sort_values("episode")
    summary = summarize_ruscuav_m4_eval_df(df_final)
    summary["variant_name"] = variant_name
    summary["status"] = "complete"
    summary["model_path"] = str(model_path)
    summary["mode"] = mode
    summary["n_episodes_requested"] = int(n_episodes)
    summary["seed0"] = int(seed0)
    summary["deterministic"] = bool(deterministic)
    summary["csv_path"] = str(csv_path)
    summary["summary_path"] = str(summary_path)
    summary["finished_at"] = datetime.now().isoformat()

    df_final.to_csv(csv_path, index=False)
    _atomic_write_json(summary_path, summary)

    print_ruscuav_m4_eval_summary(summary)

    return df_final, summary, str(csv_path), str(summary_path)

WORLD_PATH_DRIVE = "/content/drive/MyDrive/RUSCUAV/worlds/AOI_FAKE_001_urban_world_layers.npz"

import heapq

import math

from dataclasses import dataclass

from pathlib import Path

import numpy as np

import pandas as pd

@dataclass
class M9PlannerConfig:
    plan_res_m: float = 5.0
    planner_agl_m: float = 40.0

    # Movement objective:
    # edge cost = distance * (1 + risk_lambda * normalized_risk)
    risk_lambda: float = 0.35

    # If True, use A* heuristic. If False, Dijkstra.
    use_astar: bool = True

    # Safety blocking at planner altitude.
    block_buildings: bool = True
    block_canopy: bool = True
    block_sensitive_low_altitude: bool = True

    # Clearance margins.
    building_clearance_m: float = 5.0
    canopy_clearance_m: float = 3.0

    # For sensitive zones: if planner_agl < sensitive_min_agl, block sensitive cells.
    sensitive_min_agl_m: float = 80.0

    # Waypoint follower.
    waypoint_radius_m: float = 8.0
    waypoint_lookahead_m: float = 12.0
    target_speed_mps: float = 12.0
    max_steps: int = 1000

def _block_reduce_mean(arr, factor):
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor

    cropped = arr[:h2, :w2]
    reduced = cropped.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))
    return reduced.astype(np.float32)

def _block_reduce_max(arr, factor):
    arr = np.asarray(arr, dtype=np.float32)
    h, w = arr.shape
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor

    cropped = arr[:h2, :w2]
    reduced = cropped.reshape(h2 // factor, factor, w2 // factor, factor).max(axis=(1, 3))
    return reduced.astype(np.float32)

class M9StaticRiskCostPlanner:
    """
    Static full-map risk-cost planner.

    Planning state:
        grid cell (row, col)

    Cost:
        distance * (1 + risk_lambda * normalized risk)

    This planner intentionally ignores UAV dynamics and Cargo Health.
    That is the point of B0: strong static map baseline, but position-only.
    """

    def __init__(
        self,
        world: UrbanWorldLayers,
        config: M9PlannerConfig = None,
        risk_profile: str = "balanced",
    ):
        self.world = world
        self.config = config if config is not None else M9PlannerConfig()
        self.risk_profile = risk_profile

        self.factor = int(round(self.config.plan_res_m / float(world.resolution_m)))
        self.factor = max(1, self.factor)

        self.plan_res_m = self.factor * float(world.resolution_m)

        self.risk_model = UrbanRiskModel(
            world=world,
            profile=risk_profile,
            hard_sensitive_zones=True,
            include_global_altitude_band_cost=True,
        )

        self._build_cost_and_block_maps()

    def _build_cost_and_block_maps(self):
        cfg = self.config

        # Full-resolution risk at selected AGL.
        full_risk = self.risk_model.risk_map_at_agl(cfg.planner_agl_m).astype(np.float32)

        # Reduce to planning grid.
        risk_low = _block_reduce_mean(full_risk, self.factor)

        # Normalize robustly.
        p95 = float(np.nanpercentile(risk_low, 95))
        denom = max(1e-6, p95)
        risk_norm = np.clip(risk_low / denom, 0.0, 10.0).astype(np.float32)

        # Hard blocking.
        block = np.zeros_like(risk_low, dtype=bool)

        if cfg.block_buildings:
            building_block_full = (
                (self.world.building_mask > 0.5)
                & ((self.world.building_height + cfg.building_clearance_m) >= cfg.planner_agl_m)
            )
            block |= _block_reduce_max(building_block_full.astype(np.float32), self.factor) > 0.5

        if cfg.block_canopy:
            canopy_block_full = (
                (self.world.canopy_mask > 0.5)
                & ((self.world.canopy_height + cfg.canopy_clearance_m) >= cfg.planner_agl_m)
            )
            block |= _block_reduce_max(canopy_block_full.astype(np.float32), self.factor) > 0.5

        if cfg.block_sensitive_low_altitude and cfg.planner_agl_m < cfg.sensitive_min_agl_m:
            block |= _block_reduce_max(self.world.sensitive_mask.astype(np.float32), self.factor) > 0.5

        self.risk_low = risk_low
        self.risk_norm = risk_norm
        self.block = block

        # Cell traversal cost multiplier.
        self.cell_cost = (1.0 + float(cfg.risk_lambda) * risk_norm).astype(np.float32)
        self.cell_cost[block] = np.inf

        self.grid_h, self.grid_w = self.cell_cost.shape

    def xy_to_cell(self, xy):
        x, y = float(xy[0]), float(xy[1])

        col = int(np.clip(np.floor(x / self.plan_res_m), 0, self.grid_w - 1))
        row = int(np.clip(np.floor(y / self.plan_res_m), 0, self.grid_h - 1))

        return row, col

    def cell_to_xy(self, cell):
        row, col = cell
        x = (col + 0.5) * self.plan_res_m
        y = (row + 0.5) * self.plan_res_m

        x = float(np.clip(x, MIN_X, MAX_X - 1e-3))
        y = float(np.clip(y, MIN_Y, MAX_Y - 1e-3))

        return np.array([x, y], dtype=np.float32)

    def _nearest_free_cell(self, cell, max_radius_cells=20):
        row, col = cell

        if (
            0 <= row < self.grid_h
            and 0 <= col < self.grid_w
            and not self.block[row, col]
            and np.isfinite(self.cell_cost[row, col])
        ):
            return cell

        best = None
        best_d2 = float("inf")

        for r in range(1, int(max_radius_cells) + 1):
            r0 = max(0, row - r)
            r1 = min(self.grid_h - 1, row + r)
            c0 = max(0, col - r)
            c1 = min(self.grid_w - 1, col + r)

            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if self.block[rr, cc] or not np.isfinite(self.cell_cost[rr, cc]):
                        continue

                    d2 = (rr - row) ** 2 + (cc - col) ** 2
                    if d2 < best_d2:
                        best_d2 = d2
                        best = (rr, cc)

            if best is not None:
                return best

        return None

    def plan(self, start_xy, target_xy):
        start_cell_raw = self.xy_to_cell(start_xy)
        goal_cell_raw = self.xy_to_cell(target_xy)

        start = self._nearest_free_cell(start_cell_raw)
        goal = self._nearest_free_cell(goal_cell_raw)

        if start is None or goal is None:
            return {
                "success": False,
                "reason": "start_or_goal_blocked",
                "path_xy": None,
                "path_cells": None,
                "cost": np.inf,
                "expanded": 0,
            }

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        def heuristic(cell):
            if not self.config.use_astar:
                return 0.0

            dr = (cell[0] - goal[0]) * self.plan_res_m
            dc = (cell[1] - goal[1]) * self.plan_res_m
            return float(np.hypot(dr, dc))

        open_heap = []
        heapq.heappush(open_heap, (heuristic(start), 0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        expanded = 0

        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)

            if current in visited:
                continue

            visited.add(current)
            expanded += 1

            if current == goal:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()

                path_xy = [np.asarray(start_xy, dtype=np.float32)]

                for cell in cells[1:-1]:
                    path_xy.append(self.cell_to_xy(cell))

                path_xy.append(np.asarray(target_xy, dtype=np.float32))
                path_xy = np.asarray(path_xy, dtype=np.float32)

                return {
                    "success": True,
                    "reason": "ok",
                    "path_xy": path_xy,
                    "path_cells": cells,
                    "cost": float(g_curr),
                    "expanded": int(expanded),
                    "start_cell": start,
                    "goal_cell": goal,
                    "start_cell_raw": start_cell_raw,
                    "goal_cell_raw": goal_cell_raw,
                }

            cr, cc = current

            for dr, dc in neighbors:
                nr = cr + dr
                nc = cc + dc

                if nr < 0 or nr >= self.grid_h or nc < 0 or nc >= self.grid_w:
                    continue

                if self.block[nr, nc] or not np.isfinite(self.cell_cost[nr, nc]):
                    continue

                # Avoid diagonal corner cutting through blocked cells.
                if dr != 0 and dc != 0:
                    if self.block[cr, nc] or self.block[nr, cc]:
                        continue

                step_len = float(np.hypot(dr, dc) * self.plan_res_m)

                # Average cost of current and neighbor.
                c_curr = float(self.cell_cost[cr, cc])
                c_next = float(self.cell_cost[nr, nc])

                step_cost = step_len * 0.5 * (c_curr + c_next)

                neighbor = (nr, nc)
                tentative_g = g_curr + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))

        return {
            "success": False,
            "reason": "no_path",
            "path_xy": None,
            "path_cells": None,
            "cost": np.inf,
            "expanded": int(expanded),
            "start_cell": start,
            "goal_cell": goal,
            "start_cell_raw": start_cell_raw,
            "goal_cell_raw": goal_cell_raw,
        }

def m9_path_length_xy(path_xy):
    if path_xy is None or len(path_xy) < 2:
        return np.nan
    return float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1)))

def m9_waypoint_controller_action(env, path_xy, wp_idx, cfg: M9PlannerConfig):
    """
    Simple acceleration controller for following planner waypoints.

    Returns:
        action, new_wp_idx
    """
    pos_xy = env.current_pos[:2].astype(np.float32)

    if path_xy is None or len(path_xy) == 0:
        return np.zeros(env.action_space.shape, dtype=np.float32), wp_idx

    wp_idx = int(np.clip(wp_idx, 0, len(path_xy) - 1))

    # Advance waypoints if close.
    while wp_idx < len(path_xy) - 1:
        d = float(np.linalg.norm(path_xy[wp_idx] - pos_xy))
        if d <= cfg.waypoint_radius_m:
            wp_idx += 1
        else:
            break

    target_xy = path_xy[wp_idx]
    delta = target_xy - pos_xy
    dist = float(np.linalg.norm(delta) + 1e-9)

    if dist < 1e-6:
        desired_vel_xy = np.zeros(2, dtype=np.float32)
    else:
        direction = delta / dist

        # Slow down near final target.
        final_dist = float(np.linalg.norm(path_xy[-1] - pos_xy))
        speed = min(float(cfg.target_speed_mps), max(2.0, 0.8 * final_dist))

        desired_vel_xy = direction.astype(np.float32) * speed

    # Acceleration command to reach desired velocity.
    acc_xy = (desired_vel_xy - env.vel[:2].astype(np.float32)) / float(env.dt)

    action = np.clip(acc_xy / float(env.a_max_xy), -1.0, 1.0).astype(np.float32)

    return action, wp_idx

def rollout_m9_static_planner_episode(
    world,
    planner: M9StaticRiskCostPlanner,
    seed,
    cfg: M9PlannerConfig,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    max_steps=None,
    save_path=False,
):
    """
    One episode:
        env reset -> plan from start to target -> waypoint-following controller -> metrics.
    """
    if max_steps is None:
        max_steps = int(cfg.max_steps)

    env = make_ruscuav_env_m4_mode(
        world=world,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, reset_info = env.reset(seed=seed)

    start_xy = env.start_point[:2].copy()
    target_xy = env.target_point[:2].copy()

    plan_result = planner.plan(start_xy, target_xy)

    if not plan_result["success"]:
        row = {
            "episode": None,
            "seed": int(seed),
            "planner_success": False,
            "planner_reason": plan_result["reason"],
            "planner_cost": float(plan_result["cost"]),
            "planner_expanded": int(plan_result["expanded"]),
            "planned_path_length": np.nan,
            "termination": "planner_no_path",
            "is_success": False,
            "mission_acceptable": False,
            "total_reward": 0.0,
            "steps": 0,
            "dist_xy": float(np.linalg.norm(target_xy - start_xy)),
            "cargo_health": np.nan,
            "urban_mean_risk": np.nan,
            "urban_max_risk": np.nan,
            "urban_cumulative_risk": np.nan,
            "road_time": 0.0,
            "sidewalk_time": 0.0,
            "building_time": 0.0,
            "green_time": 0.0,
            "canopy_time": 0.0,
            "sensitive_time": 0.0,
            "urban_hard_violation_steps": 0,
        }

        env.close()
        return row, None, plan_result

    path_xy = plan_result["path_xy"]
    planned_len = m9_path_length_xy(path_xy)

    path_actual = [env.current_pos.copy()]
    rewards = []
    infos = []

    wp_idx = 1 if len(path_xy) > 1 else 0
    done = False
    steps = 0

    while not done and steps < int(max_steps):
        action, wp_idx = m9_waypoint_controller_action(env, path_xy, wp_idx, cfg)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(dict(info))
        path_actual.append(env.current_pos.copy())

        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}

    if not done:
        final_info = dict(final_info)
        final_info["termination"] = "manual_max_steps"

    if "dist_xy" not in final_info or final_info.get("dist_xy") is None:
        final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))

    row = {
        "episode": None,
        "seed": int(seed),

        "planner_success": True,
        "planner_reason": plan_result["reason"],
        "planner_cost": float(plan_result["cost"]),
        "planner_expanded": int(plan_result["expanded"]),
        "planned_path_length": float(planned_len),

        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),

        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),

        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(final_info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),

        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),

        "building_collision_steps": int(final_info.get("urban_building_collision_steps", 0)),
        "canopy_collision_steps": int(final_info.get("urban_canopy_collision_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(final_info.get("urban_sensitive_low_altitude_violation_steps", 0)),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),
    }

    env.close()

    return row, np.asarray(path_actual, dtype=np.float32), plan_result

import json

from pathlib import Path

from datetime import datetime

def summarize_m9_planner_eval_df(df):
    if len(df) == 0:
        return {"n_episodes": 0}

    summary = {
        "n_episodes": int(len(df)),

        "planner_path_found_rate": float(df["planner_success"].mean()),

        "success_rate": float(df["is_success"].mean()),
        "mission_acceptability_rate": float(df["mission_acceptable"].mean()),

        "mean_reward": float(df["total_reward"].mean()),
        "std_reward": float(df["total_reward"].std(ddof=0)),

        "mean_steps": float(df["steps"].mean()),
        "mean_dist_xy": float(df["dist_xy"].mean()),
        "median_dist_xy": float(df["dist_xy"].median()),

        "mean_cargo_health": float(df["cargo_health"].mean()),

        "mean_urban_risk": float(df["urban_mean_risk"].mean()),
        "mean_urban_max_risk": float(df["urban_max_risk"].mean()),
        "mean_cumulative_risk": float(df["urban_cumulative_risk"].mean()),

        "mean_road_time": float(df["road_time"].mean()),
        "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
        "mean_building_time": float(df["building_time"].mean()),
        "mean_green_time": float(df["green_time"].mean()),
        "mean_canopy_time": float(df["canopy_time"].mean()),
        "mean_sensitive_time": float(df["sensitive_time"].mean()),

        "mean_hard_violation_steps": float(df["urban_hard_violation_steps"].mean()),

        "mean_planner_cost": float(df["planner_cost"].replace([np.inf, -np.inf], np.nan).mean()),
        "mean_planner_expanded": float(df["planner_expanded"].mean()),
        "mean_planned_path_length": float(df["planned_path_length"].mean()),

        "termination_counts": {
            str(k): int(v)
            for k, v in df["termination"].fillna("None").value_counts().to_dict().items()
        },
    }

    success_df = df[df["is_success"] == True]
    if len(success_df) > 0:
        summary["acceptable_given_success_rate"] = float(success_df["mission_acceptable"].mean())
    else:
        summary["acceptable_given_success_rate"] = 0.0

    return summary

def print_m9_summary(summary):
    print("=" * 100)
    print("RUSCUAV M9-B0 STATIC PLANNER EVALUATION SUMMARY")
    print("=" * 100)

    for k, v in summary.items():
        if k == "termination_counts":
            print("termination_counts:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        elif isinstance(v, float):
            print(f"{k:45s}: {v:.6f}")
        else:
            print(f"{k:45s}: {v}")

    print("=" * 100)

def evaluate_m9_static_planner_drive(
    world,
    drive_dirs,
    planner_config: M9PlannerConfig = None,
    risk_profile="balanced",
    n_episodes=200,
    seed0=5000,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    prefix=None,
    save_paths=True,
):
    if planner_config is None:
        planner_config = M9PlannerConfig()

    if prefix is None:
        prefix = (
            f"m9_b0_static_astar_"
            f"res{planner_config.plan_res_m}_"
            f"agl{planner_config.planner_agl_m}_"
            f"lambda{planner_config.risk_lambda}_"
            f"n{n_episodes}_seed{seed0}"
        )

    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths_dir = eval_dir / "paths"
    plans_dir = eval_dir / "planned_paths"

    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)
        plans_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"
    config_path = eval_dir / f"{prefix}_planner_config.json"

    with open(config_path, "w") as f:
        json.dump(planner_config.__dict__, f, indent=2)

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        completed = set(int(e) for e in df_existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming M9 eval: {len(completed)}/{n_episodes}")
    else:
        rows = []
        completed = set()

    planner = M9StaticRiskCostPlanner(
        world=world,
        config=planner_config,
        risk_profile=risk_profile,
    )

    for ep in range(int(n_episodes)):
        if ep in completed:
            continue

        seed = int(seed0 + ep)

        row, actual_path, plan_result = rollout_m9_static_planner_episode(
            world=world,
            planner=planner,
            seed=seed,
            cfg=planner_config,
            stage=stage,
            mode=mode,
            n_obstacles=n_obstacles,
            max_steps=planner_config.max_steps,
        )

        row["episode"] = int(ep)
        rows.append(row)

        if save_paths:
            if actual_path is not None:
                np.save(paths_dir / f"episode_{ep:05d}_seed_{seed}_actual_path.npy", actual_path)

            if plan_result is not None and plan_result.get("path_xy", None) is not None:
                np.save(plans_dir / f"episode_{ep:05d}_seed_{seed}_planned_path.npy", plan_result["path_xy"])

        df_now = pd.DataFrame(rows).sort_values("episode")
        df_now.to_csv(csv_path, index=False)

        summary_now = summarize_m9_planner_eval_df(df_now)
        summary_now["status"] = "partial" if len(df_now) < int(n_episodes) else "complete"
        summary_now["updated_at"] = datetime.now().isoformat()
        summary_now["prefix"] = prefix
        summary_now["risk_profile"] = risk_profile
        summary_now["planner_config"] = planner_config.__dict__

        _atomic_write_json(summary_path, summary_now)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[M9 B0 eval] ep={ep+1}/{n_episodes} | "
                f"path={row['planner_success']} | "
                f"SR={summary_now['success_rate']:.1%} | "
                f"MAR={summary_now['mission_acceptability_rate']:.1%} | "
                f"term={row['termination']}"
            )

    df_final = pd.DataFrame(rows).sort_values("episode")
    summary = summarize_m9_planner_eval_df(df_final)

    summary["status"] = "complete"
    summary["finished_at"] = datetime.now().isoformat()
    summary["prefix"] = prefix
    summary["risk_profile"] = risk_profile
    summary["planner_config"] = planner_config.__dict__
    summary["csv_path"] = str(csv_path)
    summary["summary_path"] = str(summary_path)

    df_final.to_csv(csv_path, index=False)
    _atomic_write_json(summary_path, summary)

    print_m9_summary(summary)

    print("Saved M9 CSV:", csv_path)
    print("Saved M9 summary:", summary_path)

    return df_final, summary, str(csv_path), str(summary_path)

def run_m9_lambda_sweep(
    world,
    drive_dirs,
    lambdas=(0.0, 0.10, 0.25, 0.35, 0.50, 0.75, 1.00),
    plan_res_m=5.0,
    planner_agl_m=40.0,
    n_episodes=100,
    seed0=5000,
):
    rows = []

    for lam in lambdas:
        cfg = M9PlannerConfig(
            plan_res_m=plan_res_m,
            planner_agl_m=planner_agl_m,
            risk_lambda=float(lam),
            use_astar=True,
            waypoint_radius_m=8.0,
            target_speed_mps=12.0,
            max_steps=1000,
        )

        prefix = f"m9_b0_sweep_res{plan_res_m}_agl{planner_agl_m}_lambda{lam}_n{n_episodes}"

        df_lam, summary_lam, csv_lam, json_lam = evaluate_m9_static_planner_drive(
            world=world,
            drive_dirs=drive_dirs,
            planner_config=cfg,
            risk_profile="balanced",
            n_episodes=n_episodes,
            seed0=seed0,
            stage="C",
            mode="strict_eval",
            n_obstacles=0,
            prefix=prefix,
            save_paths=False,
        )

        row = {
            "risk_lambda": float(lam),
            "n_episodes": n_episodes,
            "success_rate": summary_lam["success_rate"],
            "mission_acceptability_rate": summary_lam["mission_acceptability_rate"],
            "acceptable_given_success_rate": summary_lam["acceptable_given_success_rate"],
            "mean_steps": summary_lam["mean_steps"],
            "mean_dist_xy": summary_lam["mean_dist_xy"],
            "mean_cumulative_risk": summary_lam["mean_cumulative_risk"],
            "mean_road_time": summary_lam["mean_road_time"],
            "mean_sidewalk_time": summary_lam["mean_sidewalk_time"],
            "mean_green_time": summary_lam["mean_green_time"],
            "mean_sensitive_time": summary_lam["mean_sensitive_time"],
            "mean_hard_violation_steps": summary_lam["mean_hard_violation_steps"],
            "termination_counts": json.dumps(summary_lam["termination_counts"]),
            "csv_path": csv_lam,
            "summary_path": json_lam,
        }

        rows.append(row)

    sweep_df = pd.DataFrame(rows)

    out_path = Path(drive_dirs["eval"]) / "m9_b0_lambda_sweep_summary.csv"
    sweep_df.to_csv(out_path, index=False)

    print("Saved M9 lambda sweep:", out_path)

    return sweep_df

import matplotlib.pyplot as plt

from scipy.stats import binomtest

import json

from pathlib import Path

import time

import json

import heapq

from dataclasses import dataclass

from pathlib import Path

import numpy as np

import pandas as pd

@dataclass
class M10RecedingHorizonPlannerConfig:
    # Base planning grid
    plan_res_m: float = 5.0
    planner_agl_m: float = 40.0
    risk_lambda: float = 0.35
    use_astar: bool = True

    # Partial observability
    sensing_radius_m: float = 75.0
    replan_every_steps: int = 5

    # Prior over unknown risk:
    #   "zero"     -> unknown cells assumed low risk
    #   "mean"     -> unknown cells set to mean true risk
    #   "blurred"  -> coarse blurred version of true risk
    #   "perfect"  -> same as full map, diagnostic only
    prior_risk_mode: str = "mean"

    # For hard blockers:
    #   True  -> planner knows static hard blockers from full map
    #   False -> hard blockers are also only revealed locally
    # Recommended first baseline: True
    hard_blockers_known: bool = True

    # Blocking rules at planner altitude
    block_buildings: bool = True
    block_canopy: bool = True
    block_sensitive_low_altitude: bool = True
    building_clearance_m: float = 5.0
    canopy_clearance_m: float = 3.0
    sensitive_min_agl_m: float = 80.0

    # Waypoint following
    waypoint_radius_m: float = 8.0
    waypoint_lookahead_m: float = 12.0
    target_speed_mps: float = 12.0
    max_steps: int = 1000

def _m10_to_m9_config(cfg: M10RecedingHorizonPlannerConfig):
    return M9PlannerConfig(
        plan_res_m=cfg.plan_res_m,
        planner_agl_m=cfg.planner_agl_m,
        risk_lambda=cfg.risk_lambda,
        use_astar=cfg.use_astar,
        block_buildings=cfg.block_buildings,
        block_canopy=cfg.block_canopy,
        block_sensitive_low_altitude=cfg.block_sensitive_low_altitude,
        building_clearance_m=cfg.building_clearance_m,
        canopy_clearance_m=cfg.canopy_clearance_m,
        sensitive_min_agl_m=cfg.sensitive_min_agl_m,
        waypoint_radius_m=cfg.waypoint_radius_m,
        waypoint_lookahead_m=cfg.waypoint_lookahead_m,
        target_speed_mps=cfg.target_speed_mps,
        max_steps=cfg.max_steps,
    )

class M10RecedingHorizonRiskPlanner:
    """
    Receding-horizon partial-observability risk-cost A*.

    It maintains:
        - true risk map, hidden from planner except sensed cells;
        - prior risk map for unknown cells;
        - known mask updated by local sensing;
        - hard blockers either fully known or locally revealed.

    Planning is repeated from current UAV position to goal every replan interval.
    """

    def __init__(
        self,
        world: UrbanWorldLayers,
        config: M10RecedingHorizonPlannerConfig = None,
        risk_profile: str = "balanced",
    ):
        self.world = world
        self.config = config if config is not None else M10RecedingHorizonPlannerConfig()
        self.risk_profile = risk_profile

        self.base_m9_cfg = _m10_to_m9_config(self.config)

        # Use M9 builder to ensure identical risk/block construction.
        self.full_planner = M9StaticRiskCostPlanner(
            world=world,
            config=self.base_m9_cfg,
            risk_profile=risk_profile,
        )

        self.plan_res_m = self.full_planner.plan_res_m
        self.grid_h = self.full_planner.grid_h
        self.grid_w = self.full_planner.grid_w

        self.true_risk_norm = self.full_planner.risk_norm.astype(np.float32)
        self.true_block = self.full_planner.block.astype(bool)

        self.known_mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)

        if self.config.hard_blockers_known:
            self.known_block = self.true_block.copy()
        else:
            self.known_block = np.zeros_like(self.true_block, dtype=bool)

        self.prior_risk_norm = self._build_prior_risk_norm(self.config.prior_risk_mode)

        self.current_cell_cost = None
        self.current_block = None
        self.replan_count = 0
        self.total_expanded = 0
        self.total_plan_time_s = 0.0

        self._update_current_maps()

    def reset_belief(self):
        self.known_mask[:] = False

        if self.config.hard_blockers_known:
            self.known_block = self.true_block.copy()
        else:
            self.known_block[:] = False

        self.replan_count = 0
        self.total_expanded = 0
        self.total_plan_time_s = 0.0

        self._update_current_maps()

    def _build_prior_risk_norm(self, mode):
        mode = str(mode).lower()

        if mode == "zero":
            return np.zeros_like(self.true_risk_norm, dtype=np.float32)

        if mode == "mean":
            mean_val = float(np.nanmean(self.true_risk_norm))
            return np.full_like(self.true_risk_norm, mean_val, dtype=np.float32)

        if mode == "blurred":
            if gaussian_filter is None:
                mean_val = float(np.nanmean(self.true_risk_norm))
                return np.full_like(self.true_risk_norm, mean_val, dtype=np.float32)

            blurred = gaussian_filter(self.true_risk_norm, sigma=5.0).astype(np.float32)
            return blurred

        if mode == "perfect":
            return self.true_risk_norm.copy().astype(np.float32)

        raise ValueError("prior_risk_mode must be 'zero', 'mean', 'blurred', or 'perfect'.")

    def xy_to_cell(self, xy):
        return self.full_planner.xy_to_cell(xy)

    def cell_to_xy(self, cell):
        return self.full_planner.cell_to_xy(cell)

    def _nearest_free_cell(self, cell, max_radius_cells=20):
        row, col = cell

        if (
            0 <= row < self.grid_h
            and 0 <= col < self.grid_w
            and not self.current_block[row, col]
            and np.isfinite(self.current_cell_cost[row, col])
        ):
            return cell

        best = None
        best_d2 = float("inf")

        for r in range(1, int(max_radius_cells) + 1):
            r0 = max(0, row - r)
            r1 = min(self.grid_h - 1, row + r)
            c0 = max(0, col - r)
            c1 = min(self.grid_w - 1, col + r)

            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if self.current_block[rr, cc] or not np.isfinite(self.current_cell_cost[rr, cc]):
                        continue

                    d2 = (rr - row) ** 2 + (cc - col) ** 2

                    if d2 < best_d2:
                        best_d2 = d2
                        best = (rr, cc)

            if best is not None:
                return best

        return None

    def reveal_around_xy(self, xy):
        """
        Reveal true risk and optionally true blockers around current position.
        """
        center = self.xy_to_cell(xy)
        cr, cc = center

        radius_cells = int(np.ceil(float(self.config.sensing_radius_m) / self.plan_res_m))
        radius_cells = max(1, radius_cells)

        r0 = max(0, cr - radius_cells)
        r1 = min(self.grid_h - 1, cr + radius_cells)
        c0 = max(0, cc - radius_cells)
        c1 = min(self.grid_w - 1, cc + radius_cells)

        rr, cc_grid = np.ogrid[r0:r1 + 1, c0:c1 + 1]
        dist2 = (rr - cr) ** 2 + (cc_grid - cc) ** 2
        mask = dist2 <= radius_cells ** 2

        local_known = self.known_mask[r0:r1 + 1, c0:c1 + 1]
        local_known[mask] = True
        self.known_mask[r0:r1 + 1, c0:c1 + 1] = local_known

        if not self.config.hard_blockers_known:
            local_block = self.known_block[r0:r1 + 1, c0:c1 + 1]
            local_true_block = self.true_block[r0:r1 + 1, c0:c1 + 1]
            local_block[mask] = local_true_block[mask]
            self.known_block[r0:r1 + 1, c0:c1 + 1] = local_block

        self._update_current_maps()

    def _update_current_maps(self):
        estimated_risk = np.where(
            self.known_mask,
            self.true_risk_norm,
            self.prior_risk_norm,
        ).astype(np.float32)

        if self.config.hard_blockers_known:
            current_block = self.true_block.copy()
        else:
            current_block = self.known_block.copy()

        cell_cost = (1.0 + float(self.config.risk_lambda) * estimated_risk).astype(np.float32)
        cell_cost[current_block] = np.inf

        self.current_cell_cost = cell_cost
        self.current_block = current_block

    def known_fraction(self):
        return float(np.mean(self.known_mask))

    def plan(self, start_xy, target_xy):
        """
        A* on current belief map.
        """
        t0 = time.time()

        start_cell_raw = self.xy_to_cell(start_xy)
        goal_cell_raw = self.xy_to_cell(target_xy)

        start = self._nearest_free_cell(start_cell_raw)
        goal = self._nearest_free_cell(goal_cell_raw)

        if start is None or goal is None:
            return {
                "success": False,
                "reason": "start_or_goal_blocked",
                "path_xy": None,
                "path_cells": None,
                "cost": np.inf,
                "expanded": 0,
                "plan_time_s": time.time() - t0,
                "known_fraction": self.known_fraction(),
            }

        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        def heuristic(cell):
            if not self.config.use_astar:
                return 0.0

            dr = (cell[0] - goal[0]) * self.plan_res_m
            dc = (cell[1] - goal[1]) * self.plan_res_m
            return float(np.hypot(dr, dc))

        open_heap = []
        heapq.heappush(open_heap, (heuristic(start), 0.0, start))

        came_from = {}
        g_score = {start: 0.0}
        visited = set()

        expanded = 0

        while open_heap:
            _, g_curr, current = heapq.heappop(open_heap)

            if current in visited:
                continue

            visited.add(current)
            expanded += 1

            if current == goal:
                cells = [current]
                while current in came_from:
                    current = came_from[current]
                    cells.append(current)
                cells.reverse()

                path_xy = [np.asarray(start_xy, dtype=np.float32)]

                for cell in cells[1:-1]:
                    path_xy.append(self.cell_to_xy(cell))

                path_xy.append(np.asarray(target_xy, dtype=np.float32))
                path_xy = np.asarray(path_xy, dtype=np.float32)

                plan_time_s = time.time() - t0

                self.replan_count += 1
                self.total_expanded += int(expanded)
                self.total_plan_time_s += float(plan_time_s)

                return {
                    "success": True,
                    "reason": "ok",
                    "path_xy": path_xy,
                    "path_cells": cells,
                    "cost": float(g_curr),
                    "expanded": int(expanded),
                    "plan_time_s": float(plan_time_s),
                    "known_fraction": self.known_fraction(),
                    "start_cell": start,
                    "goal_cell": goal,
                    "start_cell_raw": start_cell_raw,
                    "goal_cell_raw": goal_cell_raw,
                }

            cr, cc = current

            for dr, dc in neighbors:
                nr = cr + dr
                nc = cc + dc

                if nr < 0 or nr >= self.grid_h or nc < 0 or nc >= self.grid_w:
                    continue

                if self.current_block[nr, nc] or not np.isfinite(self.current_cell_cost[nr, nc]):
                    continue

                if dr != 0 and dc != 0:
                    if self.current_block[cr, nc] or self.current_block[nr, cc]:
                        continue

                step_len = float(np.hypot(dr, dc) * self.plan_res_m)

                c_curr = float(self.current_cell_cost[cr, cc])
                c_next = float(self.current_cell_cost[nr, nc])

                step_cost = step_len * 0.5 * (c_curr + c_next)

                neighbor = (nr, nc)
                tentative_g = g_curr + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))

        plan_time_s = time.time() - t0

        self.replan_count += 1
        self.total_expanded += int(expanded)
        self.total_plan_time_s += float(plan_time_s)

        return {
            "success": False,
            "reason": "no_path",
            "path_xy": None,
            "path_cells": None,
            "cost": np.inf,
            "expanded": int(expanded),
            "plan_time_s": float(plan_time_s),
            "known_fraction": self.known_fraction(),
            "start_cell": start,
            "goal_cell": goal,
            "start_cell_raw": start_cell_raw,
            "goal_cell_raw": goal_cell_raw,
        }

def rollout_m10_receding_planner_episode(
    world,
    planner: M10RecedingHorizonRiskPlanner,
    seed,
    cfg: M10RecedingHorizonPlannerConfig,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    max_steps=None,
):
    if max_steps is None:
        max_steps = int(cfg.max_steps)

    env = make_ruscuav_env_m4_mode(
        world=world,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, reset_info = env.reset(seed=seed)

    planner.reset_belief()
    planner.reveal_around_xy(env.current_pos[:2])

    target_xy = env.target_point[:2].copy()

    current_plan = None
    current_plan_result = None
    wp_idx = 0

    path_actual = [env.current_pos.copy()]
    rewards = []
    infos = []

    replan_success_count = 0
    replan_fail_count = 0
    last_plan_expanded = 0
    last_plan_time_s = 0.0
    last_known_fraction = planner.known_fraction()

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        need_replan = False

        if current_plan is None:
            need_replan = True

        if steps % int(max(1, cfg.replan_every_steps)) == 0:
            need_replan = True

        if need_replan:
            planner.reveal_around_xy(env.current_pos[:2])

            current_plan_result = planner.plan(
                start_xy=env.current_pos[:2].copy(),
                target_xy=target_xy,
            )

            last_plan_expanded = int(current_plan_result.get("expanded", 0))
            last_plan_time_s = float(current_plan_result.get("plan_time_s", 0.0))
            last_known_fraction = float(current_plan_result.get("known_fraction", planner.known_fraction()))

            if current_plan_result["success"]:
                current_plan = current_plan_result["path_xy"]
                wp_idx = 1 if len(current_plan) > 1 else 0
                replan_success_count += 1
            else:
                replan_fail_count += 1

                # If we had no valid previous plan, stop.
                if current_plan is None:
                    final_info = {
                        "termination": "planner_no_path",
                        "is_success": False,
                        "mission_acceptable": False,
                        "dist_xy": float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2])),
                    }
                    break

        action, wp_idx = m9_waypoint_controller_action(
            env=env,
            path_xy=current_plan,
            wp_idx=wp_idx,
            cfg=_m10_to_m9_config(cfg),
        )

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(dict(info))
        path_actual.append(env.current_pos.copy())

        done = bool(terminated or truncated)
        steps += 1

    if infos:
        final_info = infos[-1]
    else:
        final_info = locals().get("final_info", {})

    if not done and final_info.get("termination", None) is None:
        final_info = dict(final_info)
        final_info["termination"] = "manual_max_steps"
        final_info["is_success"] = False
        final_info["mission_acceptable"] = False
        final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))

    if "dist_xy" not in final_info or final_info.get("dist_xy") is None:
        final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))

    total_plan_time_s = float(planner.total_plan_time_s)
    replan_count = int(planner.replan_count)

    row = {
        "episode": None,
        "seed": int(seed),

        "planner_type": "M10_receding_horizon_Astar",
        "prior_risk_mode": cfg.prior_risk_mode,
        "sensing_radius_m": float(cfg.sensing_radius_m),
        "replan_every_steps": int(cfg.replan_every_steps),
        "risk_lambda": float(cfg.risk_lambda),

        "replan_count": int(replan_count),
        "replan_success_count": int(replan_success_count),
        "replan_fail_count": int(replan_fail_count),
        "total_expanded": int(planner.total_expanded),
        "mean_expanded_per_replan": float(planner.total_expanded / max(1, replan_count)),
        "total_plan_time_s": total_plan_time_s,
        "mean_plan_time_s": float(total_plan_time_s / max(1, replan_count)),
        "last_plan_expanded": int(last_plan_expanded),
        "last_plan_time_s": float(last_plan_time_s),
        "final_known_fraction": float(planner.known_fraction()),
        "last_known_fraction": float(last_known_fraction),

        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),

        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),

        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(final_info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),

        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),

        "building_collision_steps": int(final_info.get("urban_building_collision_steps", 0)),
        "canopy_collision_steps": int(final_info.get("urban_canopy_collision_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(final_info.get("urban_sensitive_low_altitude_violation_steps", 0)),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),
    }

    env.close()

    return row, np.asarray(path_actual, dtype=np.float32)

def summarize_m10_receding_eval_df(df):
    if len(df) == 0:
        return {"n_episodes": 0}

    summary = {
        "n_episodes": int(len(df)),

        "success_rate": float(df["is_success"].mean()),
        "mission_acceptability_rate": float(df["mission_acceptable"].mean()),

        "mean_reward": float(df["total_reward"].mean()),
        "std_reward": float(df["total_reward"].std(ddof=0)),

        "mean_steps": float(df["steps"].mean()),
        "mean_dist_xy": float(df["dist_xy"].mean()),
        "median_dist_xy": float(df["dist_xy"].median()),

        "mean_cargo_health": float(df["cargo_health"].mean()),

        "mean_urban_risk": float(df["urban_mean_risk"].mean()),
        "mean_urban_max_risk": float(df["urban_max_risk"].mean()),
        "mean_cumulative_risk": float(df["urban_cumulative_risk"].mean()),

        "mean_road_time": float(df["road_time"].mean()),
        "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
        "mean_building_time": float(df["building_time"].mean()),
        "mean_green_time": float(df["green_time"].mean()),
        "mean_canopy_time": float(df["canopy_time"].mean()),
        "mean_sensitive_time": float(df["sensitive_time"].mean()),

        "mean_hard_violation_steps": float(df["urban_hard_violation_steps"].mean()),

        "mean_replan_count": float(df["replan_count"].mean()),
        "mean_replan_success_count": float(df["replan_success_count"].mean()),
        "mean_replan_fail_count": float(df["replan_fail_count"].mean()),
        "mean_total_expanded": float(df["total_expanded"].mean()),
        "mean_expanded_per_replan": float(df["mean_expanded_per_replan"].mean()),
        "mean_total_plan_time_s": float(df["total_plan_time_s"].mean()),
        "mean_plan_time_s": float(df["mean_plan_time_s"].mean()),
        "mean_final_known_fraction": float(df["final_known_fraction"].mean()),

        "termination_counts": {
            str(k): int(v)
            for k, v in df["termination"].fillna("None").value_counts().to_dict().items()
        },
    }

    success_df = df[df["is_success"] == True]
    if len(success_df) > 0:
        summary["acceptable_given_success_rate"] = float(success_df["mission_acceptable"].mean())
    else:
        summary["acceptable_given_success_rate"] = 0.0

    return summary

def print_m10_summary(summary):
    print("=" * 100)
    print("RUSCUAV M10-B1 RECEDING-HORIZON PLANNER SUMMARY")
    print("=" * 100)

    for k, v in summary.items():
        if k == "termination_counts":
            print("termination_counts:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        elif isinstance(v, float):
            print(f"{k:45s}: {v:.6f}")
        else:
            print(f"{k:45s}: {v}")

    print("=" * 100)

def evaluate_m10_receding_planner_drive(
    world,
    drive_dirs,
    cfg: M10RecedingHorizonPlannerConfig,
    risk_profile="balanced",
    n_episodes=200,
    seed0=5000,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    prefix=None,
    save_paths=True,
):
    if prefix is None:
        prefix = (
            f"m10_b1_receding_"
            f"prior{cfg.prior_risk_mode}_"
            f"r{cfg.sensing_radius_m}_"
            f"replan{cfg.replan_every_steps}_"
            f"lambda{cfg.risk_lambda}_"
            f"n{n_episodes}_seed{seed0}"
        )

    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths_dir = eval_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"
    config_path = eval_dir / f"{prefix}_config.json"

    with open(config_path, "w") as f:
        json.dump(cfg.__dict__, f, indent=2)

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        completed = set(int(e) for e in df_existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming M10 eval: {len(completed)}/{n_episodes}")
    else:
        rows = []
        completed = set()

    planner = M10RecedingHorizonRiskPlanner(
        world=world,
        config=cfg,
        risk_profile=risk_profile,
    )

    for ep in range(int(n_episodes)):
        if ep in completed:
            continue

        seed = int(seed0 + ep)

        row, actual_path = rollout_m10_receding_planner_episode(
            world=world,
            planner=planner,
            seed=seed,
            cfg=cfg,
            stage=stage,
            mode=mode,
            n_obstacles=n_obstacles,
            max_steps=cfg.max_steps,
        )

        row["episode"] = int(ep)
        rows.append(row)

        if save_paths and actual_path is not None:
            np.save(paths_dir / f"episode_{ep:05d}_seed_{seed}_actual_path.npy", actual_path)

        df_now = pd.DataFrame(rows).sort_values("episode")
        df_now.to_csv(csv_path, index=False)

        summary_now = summarize_m10_receding_eval_df(df_now)
        summary_now["status"] = "partial" if len(df_now) < int(n_episodes) else "complete"
        summary_now["updated_at"] = datetime.now().isoformat()
        summary_now["prefix"] = prefix
        summary_now["risk_profile"] = risk_profile
        summary_now["config"] = cfg.__dict__

        _atomic_write_json(summary_path, summary_now)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[M10 B1 eval] ep={ep+1}/{n_episodes} | "
                f"SR={summary_now['success_rate']:.1%} | "
                f"MAR={summary_now['mission_acceptability_rate']:.1%} | "
                f"known={summary_now['mean_final_known_fraction']:.2%} | "
                f"term={row['termination']}"
            )

    df_final = pd.DataFrame(rows).sort_values("episode")
    summary = summarize_m10_receding_eval_df(df_final)

    summary["status"] = "complete"
    summary["finished_at"] = datetime.now().isoformat()
    summary["prefix"] = prefix
    summary["risk_profile"] = risk_profile
    summary["config"] = cfg.__dict__
    summary["csv_path"] = str(csv_path)
    summary["summary_path"] = str(summary_path)

    df_final.to_csv(csv_path, index=False)
    _atomic_write_json(summary_path, summary)

    print_m10_summary(summary)

    print("Saved M10 CSV:", csv_path)
    print("Saved M10 summary:", summary_path)

    return df_final, summary, str(csv_path), str(summary_path)

def run_m10_sensing_radius_sweep(
    world,
    drive_dirs,
    radii=(40.0, 60.0, 75.0, 100.0, 150.0),
    prior_modes=("mean",),
    replan_every_steps=5,
    risk_lambda=0.35,
    n_episodes=100,
    seed0=5000,
):
    rows = []

    for prior in prior_modes:
        for r in radii:
            cfg = M10RecedingHorizonPlannerConfig(
                plan_res_m=5.0,
                planner_agl_m=40.0,
                risk_lambda=float(risk_lambda),
                use_astar=True,

                sensing_radius_m=float(r),
                replan_every_steps=int(replan_every_steps),
                prior_risk_mode=str(prior),
                hard_blockers_known=True,

                waypoint_radius_m=8.0,
                target_speed_mps=12.0,
                max_steps=1000,
            )

            prefix = f"m10_b1_sweep_prior{prior}_r{r}_replan{replan_every_steps}_lambda{risk_lambda}_n{n_episodes}"

            df_r, summary_r, csv_r, json_r = evaluate_m10_receding_planner_drive(
                world=world,
                drive_dirs=drive_dirs,
                cfg=cfg,
                risk_profile="balanced",
                n_episodes=n_episodes,
                seed0=seed0,
                stage="C",
                mode="strict_eval",
                n_obstacles=0,
                prefix=prefix,
                save_paths=False,
            )

            rows.append({
                "prior_risk_mode": prior,
                "sensing_radius_m": float(r),
                "replan_every_steps": int(replan_every_steps),
                "risk_lambda": float(risk_lambda),

                "success_rate": summary_r["success_rate"],
                "mission_acceptability_rate": summary_r["mission_acceptability_rate"],
                "acceptable_given_success_rate": summary_r["acceptable_given_success_rate"],
                "mean_steps": summary_r["mean_steps"],
                "mean_dist_xy": summary_r["mean_dist_xy"],
                "mean_cumulative_risk": summary_r["mean_cumulative_risk"],
                "mean_road_time": summary_r["mean_road_time"],
                "mean_sidewalk_time": summary_r["mean_sidewalk_time"],
                "mean_green_time": summary_r["mean_green_time"],
                "mean_sensitive_time": summary_r["mean_sensitive_time"],
                "mean_hard_violation_steps": summary_r["mean_hard_violation_steps"],

                "mean_replan_count": summary_r["mean_replan_count"],
                "mean_total_expanded": summary_r["mean_total_expanded"],
                "mean_total_plan_time_s": summary_r["mean_total_plan_time_s"],
                "mean_final_known_fraction": summary_r["mean_final_known_fraction"],

                "termination_counts": json.dumps(summary_r["termination_counts"]),
                "csv_path": csv_r,
                "summary_path": json_r,
            })

    sweep_df = pd.DataFrame(rows)

    out_path = Path(drive_dirs["eval"]) / "m10_b1_sensing_radius_sweep_summary.csv"
    sweep_df.to_csv(out_path, index=False)

    print("Saved M10 sensing sweep:", out_path)

    return sweep_df

import json

from pathlib import Path

import pandas as pd

import numpy as np

def _safe_summary_value(summary, key, default=np.nan):
    try:
        return summary.get(key, default)
    except Exception:
        return default

def _termination_count(summary, name):
    try:
        return int(summary.get("termination_counts", {}).get(name, 0))
    except Exception:
        return 0

def make_ruscuav_static_summary_table(
    save_path="/content/drive/MyDrive/RUSCUAV/experiments/ruscuav_static_regime_summary.csv",
):
    """
    Builds a single comparison table for static RUSCUAV-v0 results.

    Expected variables if available:
        summary_geo_eval          -> M7 geometry-only
        summary_m8_obs           -> M8 obs-only
        summary_eval             -> M6 full semantic
        summary_m8_reward_200    -> M8 reward-only
        summary_m9_b0            -> M9 one-shot static A*
        summary_m10_b1           -> M10 partial-risk receding A*
        summary_m10_b0r          -> M10 full-map receding A*
    """
    rows = []

    candidates = [
        {
            "var_name": "summary_geo_eval",
            "model": "M7_geometry_only_RL",
            "method_type": "learned_policy",
            "semantic_obs": False,
            "risk_reward": False,
            "risk_cost_objective": False,
            "planner_full_map": False,
            "receding_replan": False,
            "urban_lidar": True,
            "action_shield": True,
            "geometric_shield": True,
        },
        {
            "var_name": "summary_m8_obs",
            "model": "M8_obs_only_RL",
            "method_type": "learned_policy",
            "semantic_obs": True,
            "risk_reward": False,
            "risk_cost_objective": False,
            "planner_full_map": False,
            "receding_replan": False,
            "urban_lidar": True,
            "action_shield": True,
            "geometric_shield": True,
        },
        {
            "var_name": "summary_eval",
            "model": "M6_full_semantic_RL",
            "method_type": "learned_policy",
            "semantic_obs": True,
            "risk_reward": True,
            "risk_cost_objective": True,
            "planner_full_map": False,
            "receding_replan": False,
            "urban_lidar": True,
            "action_shield": True,
            "geometric_shield": True,
        },
        {
            "var_name": "summary_m8_reward_200",
            "model": "M8_reward_only_RL",
            "method_type": "learned_policy",
            "semantic_obs": False,
            "risk_reward": True,
            "risk_cost_objective": True,
            "planner_full_map": False,
            "receding_replan": False,
            "urban_lidar": True,
            "action_shield": True,
            "geometric_shield": True,
        },
        {
            "var_name": "summary_m9_b0",
            "model": "M9_static_risk_Astar_one_shot",
            "method_type": "planner",
            "semantic_obs": False,
            "risk_reward": False,
            "risk_cost_objective": True,
            "planner_full_map": True,
            "receding_replan": False,
            "urban_lidar": None,
            "action_shield": True,
            "geometric_shield": True,
        },
        {
            "var_name": "summary_m10_b1",
            "model": "M10_partial_risk_receding_Astar",
            "method_type": "planner",
            "semantic_obs": False,
            "risk_reward": False,
            "risk_cost_objective": True,
            "planner_full_map": False,
            "receding_replan": True,
            "urban_lidar": None,
            "action_shield": True,
            "geometric_shield": True,
        },
        {
            "var_name": "summary_m10_b0r",
            "model": "M10_fullmap_receding_Astar",
            "method_type": "planner",
            "semantic_obs": False,
            "risk_reward": False,
            "risk_cost_objective": True,
            "planner_full_map": True,
            "receding_replan": True,
            "urban_lidar": None,
            "action_shield": True,
            "geometric_shield": True,
        },
    ]

    g = globals()

    for cfg in candidates:
        var_name = cfg["var_name"]

        if var_name not in g:
            print(f"[skip] {var_name} not found")
            continue

        summary = g[var_name]

        row = dict(cfg)
        row.pop("var_name", None)

        row.update({
            "success_rate": _safe_summary_value(summary, "success_rate"),
            "mission_acceptability_rate": _safe_summary_value(summary, "mission_acceptability_rate"),
            "acceptable_given_success_rate": _safe_summary_value(summary, "acceptable_given_success_rate"),
            "mean_reward": _safe_summary_value(summary, "mean_reward"),
            "std_reward": _safe_summary_value(summary, "std_reward"),
            "mean_steps": _safe_summary_value(summary, "mean_steps"),
            "mean_dist_xy": _safe_summary_value(summary, "mean_dist_xy"),
            "mean_cargo_health": _safe_summary_value(summary, "mean_cargo_health"),
            "mean_urban_risk": _safe_summary_value(summary, "mean_urban_risk"),
            "mean_cumulative_risk": _safe_summary_value(summary, "mean_cumulative_risk"),
            "mean_road_time": _safe_summary_value(summary, "mean_road_time"),
            "mean_sidewalk_time": _safe_summary_value(summary, "mean_sidewalk_time"),
            "mean_building_time": _safe_summary_value(summary, "mean_building_time"),
            "mean_green_time": _safe_summary_value(summary, "mean_green_time"),
            "mean_canopy_time": _safe_summary_value(summary, "mean_canopy_time"),
            "mean_sensitive_time": _safe_summary_value(summary, "mean_sensitive_time"),
            "mean_hard_violation_steps": _safe_summary_value(summary, "mean_hard_violation_steps"),

            "term_success": _termination_count(summary, "success"),
            "term_stuck": _termination_count(summary, "stuck"),
            "term_shield_stuck": _termination_count(summary, "shield_stuck"),
            "term_near_success": _termination_count(summary, "near_success"),
            "term_urban_sidewalk_budget_exceeded": _termination_count(summary, "urban_sidewalk_budget_exceeded"),
            "term_urban_road_budget_exceeded": _termination_count(summary, "urban_road_budget_exceeded"),
            "term_urban_building_collision": _termination_count(summary, "urban_building_collision"),
            "term_urban_canopy_collision": _termination_count(summary, "urban_canopy_collision"),
            "term_urban_sensitive_low_altitude_violation": _termination_count(summary, "urban_sensitive_low_altitude_violation"),

            "mean_replan_count": _safe_summary_value(summary, "mean_replan_count"),
            "mean_total_expanded": _safe_summary_value(summary, "mean_total_expanded"),
            "mean_total_plan_time_s": _safe_summary_value(summary, "mean_total_plan_time_s"),
            "mean_plan_time_s": _safe_summary_value(summary, "mean_plan_time_s"),
            "mean_final_known_fraction": _safe_summary_value(summary, "mean_final_known_fraction"),
        })

        rows.append(row)

    df = pd.DataFrame(rows)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)

    print("Saved static summary table:", save_path)

    return df

from dataclasses import dataclass

import numpy as np

import pandas as pd

from pathlib import Path

import json

from datetime import datetime

import time

@dataclass
class M11DynamicRiskConfig:
    n_hotspots: int = 8

    # Where to place hotspots:
    # "road_sidewalk" -> sample from road/sidewalk masks
    # "sidewalk"      -> sidewalk only
    # "road"          -> road only
    # "mixed"         -> road/sidewalk/green
    hotspot_mode: str = "road_sidewalk"

    radius_m_min: float = 35.0
    radius_m_max: float = 90.0

    amplitude_min: float = 4.0
    amplitude_max: float = 14.0

    # Time activation in env steps
    start_step_min: int = 0
    start_step_max: int = 250
    duration_min_steps: int = 60
    duration_max_steps: int = 220

    # Temporal shape:
    # "box"  -> active constant during [start, end]
    # "hann" -> smooth temporal pulse
    temporal_mode: str = "hann"

    # Altitude decay
    altitude_h_scale_m: float = 55.0

    # For mission acceptability in dynamic setting
    dynamic_cumulative_risk_budget: float = 35.0

    # Reward penalty multiplier when used in env
    dynamic_risk_lambda: float = 0.10

    # Planner normalization
    dynamic_norm_denom: float = 10.0

class M11DynamicRiskField:
    """
    Episode-specific dynamic soft-risk field.

    It creates temporal Gaussian hotspots over road/sidewalk/green masks.
    Dynamic risk is not a hard obstacle; it contributes to reward/evaluation risk.
    """

    def __init__(self, world: UrbanWorldLayers, config: M11DynamicRiskConfig, seed: int):
        self.world = world
        self.config = config
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.hotspots = self._sample_hotspots()

    def _sample_xy_from_mask(self, mask):
        ys, xs = np.where(mask > 0.5)

        if len(xs) == 0:
            return None

        idx = int(self.rng.integers(0, len(xs)))

        x = float(xs[idx] * self.world.resolution_m)
        y = float(ys[idx] * self.world.resolution_m)

        return x, y

    def _candidate_mask(self):
        mode = str(self.config.hotspot_mode).lower()

        if mode == "road_sidewalk":
            mask = ((self.world.road_mask > 0.5) | (self.world.sidewalk_mask > 0.5)).astype(np.float32)

        elif mode == "sidewalk":
            mask = (self.world.sidewalk_mask > 0.5).astype(np.float32)

        elif mode == "road":
            mask = (self.world.road_mask > 0.5).astype(np.float32)

        elif mode == "mixed":
            mask = (
                (self.world.road_mask > 0.5)
                | (self.world.sidewalk_mask > 0.5)
                | (self.world.green_mask > 0.5)
            ).astype(np.float32)

        else:
            raise ValueError("hotspot_mode must be road_sidewalk, sidewalk, road, or mixed")

        if np.sum(mask > 0.5) == 0:
            mask = np.ones_like(self.world.terrain_z, dtype=np.float32)

        return mask

    def _sample_hotspots(self):
        mask = self._candidate_mask()
        hotspots = []

        for _ in range(int(self.config.n_hotspots)):
            xy = self._sample_xy_from_mask(mask)

            if xy is None:
                x = float(self.rng.uniform(MIN_X, MAX_X))
                y = float(self.rng.uniform(MIN_Y, MAX_Y))
            else:
                x, y = xy

            radius = float(self.rng.uniform(self.config.radius_m_min, self.config.radius_m_max))
            amplitude = float(self.rng.uniform(self.config.amplitude_min, self.config.amplitude_max))

            start = int(self.rng.integers(self.config.start_step_min, self.config.start_step_max + 1))
            duration = int(self.rng.integers(self.config.duration_min_steps, self.config.duration_max_steps + 1))
            end = start + duration

            hotspots.append({
                "x": float(x),
                "y": float(y),
                "radius_m": float(radius),
                "sigma_m": float(max(1e-6, radius / 2.0)),
                "amplitude": float(amplitude),
                "start_step": int(start),
                "end_step": int(end),
                "duration_steps": int(duration),
            })

        return hotspots

    def temporal_factor(self, step, h):
        step = int(step)

        if step < h["start_step"] or step > h["end_step"]:
            return 0.0

        if self.config.temporal_mode == "box":
            return 1.0

        if self.config.temporal_mode == "hann":
            u = (step - h["start_step"]) / max(1, h["duration_steps"])
            u = float(np.clip(u, 0.0, 1.0))
            return float(0.5 - 0.5 * np.cos(2.0 * np.pi * u))

        raise ValueError("temporal_mode must be box or hann")

    def risk_at_xy_agl(self, x, y, agl_m, step):
        x = float(x)
        y = float(y)
        agl_m = float(max(0.0, agl_m))
        step = int(step)

        alt_factor = float(np.exp(-agl_m / max(1e-6, self.config.altitude_h_scale_m)))

        total = 0.0

        for h in self.hotspots:
            tf = self.temporal_factor(step, h)
            if tf <= 0.0:
                continue

            dx = x - h["x"]
            dy = y - h["y"]
            d2 = dx * dx + dy * dy
            sigma2 = h["sigma_m"] ** 2

            spatial = float(np.exp(-0.5 * d2 / max(1e-6, sigma2)))

            total += h["amplitude"] * tf * spatial * alt_factor

        return float(total)

    def risk_at_pos(self, pos, step):
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        ground = float(self.world.terrain_z[
            int(np.clip(np.floor(y / self.world.resolution_m), 0, self.world.height - 1)),
            int(np.clip(np.floor(x / self.world.resolution_m), 0, self.world.width - 1)),
        ])
        agl_m = z - ground
        return self.risk_at_xy_agl(x, y, agl_m, step)

    def risk_lowres_map_at_step(self, step, plan_res_m, grid_h, grid_w, planner_agl_m):
        """
        Dynamic risk map on planner grid.
        """
        xs = (np.arange(grid_w, dtype=np.float32) + 0.5) * float(plan_res_m)
        ys = (np.arange(grid_h, dtype=np.float32) + 0.5) * float(plan_res_m)

        X, Y = np.meshgrid(xs, ys)

        out = np.zeros((grid_h, grid_w), dtype=np.float32)

        alt_factor = float(np.exp(-float(planner_agl_m) / max(1e-6, self.config.altitude_h_scale_m)))

        for h in self.hotspots:
            tf = self.temporal_factor(step, h)
            if tf <= 0.0:
                continue

            dx = X - h["x"]
            dy = Y - h["y"]
            d2 = dx * dx + dy * dy
            sigma2 = h["sigma_m"] ** 2

            out += (h["amplitude"] * tf * alt_factor * np.exp(-0.5 * d2 / max(1e-6, sigma2))).astype(np.float32)

        return out.astype(np.float32)

    def to_dict(self):
        return {
            "seed": int(self.seed),
            "config": self.config.__dict__,
            "hotspots": self.hotspots,
        }

class RealisticUrbanCargoUAVEnvM11Dynamic(RealisticUrbanCargoUAVEnvM4):
    """
    M4 environment with episode-specific dynamic soft-risk overlay.

    It keeps all static urban safety logic from M4 and adds:
        - dynamic_risk at current position/time,
        - dynamic cumulative risk,
        - dynamic-aware mission acceptability.
    """

    def __init__(
        self,
        world: UrbanWorldLayers,
        config: RealisticUrbanEnvConfig = None,
        dynamic_config: M11DynamicRiskConfig = None,
        include_shield_features: bool = False,
        urban_lidar_enabled: bool = True,
        urban_action_shield_enabled: bool = True,
        urban_geometric_shield_enabled: bool = True,
        urban_lidar_step_m: float = 3.0,
        urban_segment_check_step_m: float = 1.0,
        urban_shield_backoff_m: float = 0.50,
        urban_horizontal_safety_margin_m: float = 1.0,
    ):
        self.dynamic_config = dynamic_config if dynamic_config is not None else M11DynamicRiskConfig()
        self.dynamic_field = None
        self.ep_dynamic_risk_sum = 0.0
        self.ep_dynamic_risk_max = 0.0
        self.ep_dynamic_steps = 0

        super().__init__(
            world=world,
            config=config,
            include_shield_features=include_shield_features,
            urban_lidar_enabled=urban_lidar_enabled,
            urban_action_shield_enabled=urban_action_shield_enabled,
            urban_geometric_shield_enabled=urban_geometric_shield_enabled,
            urban_lidar_step_m=urban_lidar_step_m,
            urban_segment_check_step_m=urban_segment_check_step_m,
            urban_shield_backoff_m=urban_shield_backoff_m,
            urban_horizontal_safety_margin_m=urban_horizontal_safety_margin_m,
        )

    def _reset_dynamic_metrics(self):
        self.ep_dynamic_risk_sum = 0.0
        self.ep_dynamic_risk_max = 0.0
        self.ep_dynamic_steps = 0

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        dyn_seed = 0 if seed is None else int(seed)
        self.dynamic_field = M11DynamicRiskField(
            world=self.world,
            config=self.dynamic_config,
            seed=dyn_seed,
        )

        self._reset_dynamic_metrics()

        info = dict(info)
        info["dynamic_risk_enabled"] = True
        info["dynamic_hotspots"] = len(self.dynamic_field.hotspots)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if self.dynamic_field is None:
            dyn_risk = 0.0
        else:
            dyn_risk = float(self.dynamic_field.risk_at_pos(self.current_pos, self.current_step))

        dt = float(self.dt)

        self.ep_dynamic_risk_sum += dyn_risk * dt
        self.ep_dynamic_risk_max = max(self.ep_dynamic_risk_max, dyn_risk)
        self.ep_dynamic_steps += 1

        # Penalize dynamic risk during training/eval.
        reward -= float(self.dynamic_config.dynamic_risk_lambda) * dyn_risk

        base_acceptable = bool(info.get("mission_acceptable", False))
        dynamic_budget_ok = self.ep_dynamic_risk_sum <= float(self.dynamic_config.dynamic_cumulative_risk_budget)

        dynamic_mission_acceptable = bool(base_acceptable and dynamic_budget_ok)

        # Override mission_acceptable to include dynamic risk.
        info["mission_acceptable_static_only"] = base_acceptable
        info["mission_acceptable"] = dynamic_mission_acceptable

        info.update({
            "dynamic_risk": float(dyn_risk),
            "dynamic_risk_mean": float(self.ep_dynamic_risk_sum / max(1e-6, self.ep_dynamic_steps * dt)),
            "dynamic_risk_max": float(self.ep_dynamic_risk_max),
            "dynamic_cumulative_risk": float(self.ep_dynamic_risk_sum),
            "dynamic_budget_ok": bool(dynamic_budget_ok),
            "dynamic_mission_acceptable": bool(dynamic_mission_acceptable),
        })

        return obs, float(reward), bool(terminated), bool(truncated), info

def make_ruscuav_env_m11_dynamic_mode(
    world,
    stage="C",
    mode="strict_eval",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
    dynamic_config: M11DynamicRiskConfig = None,
    reward_only_obs: bool = True,
):
    """
    Dynamic-risk env.

    reward_only_obs=True:
        80D observation, risk reward active.
        This matches the current best learned-policy variant.
    """
    mode = str(mode).lower()
    stage = str(stage).upper()

    if mode == "fast_train":
        urban_lidar_step_m = 5.0
        urban_segment_check_step_m = 2.0
    elif mode == "strict_eval":
        urban_lidar_step_m = 3.0
        urban_segment_check_step_m = 1.0
    else:
        raise ValueError("mode must be fast_train or strict_eval")

    if stage == "A":
        urban_risk_lambda = 0.05
        terminate_on_urban_constraint = False
        hard_sensitive_zones = False
    elif stage == "B":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = False
        hard_sensitive_zones = True
    elif stage == "C":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = True
        hard_sensitive_zones = True
    else:
        raise ValueError("stage must be A, B, or C")

    cfg = RealisticUrbanEnvConfig(
        risk_profile=risk_profile,

        include_urban_vector_obs=not bool(reward_only_obs),
        use_urban_risk_costs=True,
        urban_risk_lambda=urban_risk_lambda,
        green_bonus_lambda=0.03,

        use_urban_altitude_reference=True,

        terminate_on_urban_constraint=terminate_on_urban_constraint,
        hard_sensitive_zones=hard_sensitive_zones,

        default_n_obstacles=int(n_obstacles),

        mission_accept_cargo_min=0.80,
        mission_accept_road_budget_s=20.0,
        mission_accept_sidewalk_budget_s=5.0,
        mission_accept_building_budget_s=15.0,
        mission_accept_sensitive_budget_s=0.0,

        terminal_road_budget_s=30.0,
        terminal_sidewalk_budget_s=10.0,
        terminal_building_budget_s=20.0,
        terminal_sensitive_budget_s=0.0,
    )

    env = RealisticUrbanCargoUAVEnvM11Dynamic(
        world=world,
        config=cfg,
        dynamic_config=dynamic_config,
        include_shield_features=False,

        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,

        urban_lidar_step_m=urban_lidar_step_m,
        urban_segment_check_step_m=urban_segment_check_step_m,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        print(
            f"[M11 dynamic env] mode={mode}, stage={stage}, obs.shape={obs.shape}, info={info}"
        )

    return env

from stable_baselines3 import PPO

def rollout_m11_dynamic_rl_episode(
    model,
    world,
    seed,
    dynamic_config,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    deterministic=True,
    max_steps=1000,
    reward_only_obs=True,
):
    env = make_ruscuav_env_m11_dynamic_mode(
        world=world,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
        dynamic_config=dynamic_config,
        reward_only_obs=reward_only_obs,
    )

    obs, reset_info = env.reset(seed=seed)

    rewards = []
    infos = []
    path = [env.current_pos.copy()]

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(dict(info))
        path.append(env.current_pos.copy())

        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}

    if "dist_xy" not in final_info or final_info.get("dist_xy") is None:
        try:
            final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))
        except Exception:
            final_info["dist_xy"] = np.nan

    row = {
        "episode": None,
        "seed": int(seed),
        "method": "RL_reward_only_dynamic_eval",

        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),
        "mission_acceptable_static_only": bool(final_info.get("mission_acceptable_static_only", False)),

        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),

        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),

        "dynamic_risk_mean": float(final_info.get("dynamic_risk_mean", np.nan)),
        "dynamic_risk_max": float(final_info.get("dynamic_risk_max", np.nan)),
        "dynamic_cumulative_risk": float(final_info.get("dynamic_cumulative_risk", np.nan)),
        "dynamic_budget_ok": bool(final_info.get("dynamic_budget_ok", False)),

        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),

        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),
    }

    env.close()

    return row, np.asarray(path, dtype=np.float32)

def summarize_m11_dynamic_eval_df(df):
    if len(df) == 0:
        return {"n_episodes": 0}

    summary = {
        "n_episodes": int(len(df)),
        "success_rate": float(df["is_success"].mean()),
        "mission_acceptability_rate": float(df["mission_acceptable"].mean()),
        "static_only_acceptability_rate": float(df["mission_acceptable_static_only"].mean()) if "mission_acceptable_static_only" in df else np.nan,

        "mean_reward": float(df["total_reward"].mean()),
        "std_reward": float(df["total_reward"].std(ddof=0)),

        "mean_steps": float(df["steps"].mean()),
        "mean_dist_xy": float(df["dist_xy"].mean()),
        "median_dist_xy": float(df["dist_xy"].median()),
        "mean_cargo_health": float(df["cargo_health"].mean()),

        "mean_urban_risk": float(df["urban_mean_risk"].mean()),
        "mean_urban_cumulative_risk": float(df["urban_cumulative_risk"].mean()),

        "mean_dynamic_risk": float(df["dynamic_risk_mean"].mean()),
        "mean_dynamic_risk_max": float(df["dynamic_risk_max"].mean()),
        "mean_dynamic_cumulative_risk": float(df["dynamic_cumulative_risk"].mean()),
        "dynamic_budget_ok_rate": float(df["dynamic_budget_ok"].mean()),

        "mean_road_time": float(df["road_time"].mean()),
        "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
        "mean_green_time": float(df["green_time"].mean()),
        "mean_sensitive_time": float(df["sensitive_time"].mean()),
        "mean_hard_violation_steps": float(df["urban_hard_violation_steps"].mean()),

        "termination_counts": {
            str(k): int(v)
            for k, v in df["termination"].fillna("None").value_counts().to_dict().items()
        },
    }

    success_df = df[df["is_success"] == True]
    if len(success_df) > 0:
        summary["acceptable_given_success_rate"] = float(success_df["mission_acceptable"].mean())
    else:
        summary["acceptable_given_success_rate"] = 0.0

    return summary

def evaluate_m11_dynamic_rl_drive(
    model_path,
    world,
    drive_dirs,
    dynamic_config,
    n_episodes=200,
    seed0=5000,
    prefix="m11_dynamic_rl_reward_only_eval",
    stage="C",
    mode="strict_eval",
    reward_only_obs=True,
    save_paths=False,
):
    model = PPO.load(str(model_path), device="cpu")

    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths_dir = eval_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"
    cfg_path = eval_dir / f"{prefix}_dynamic_config.json"

    with open(cfg_path, "w") as f:
        json.dump(dynamic_config.__dict__, f, indent=2)

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        completed = set(int(e) for e in df_existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming M11 RL eval: {len(completed)}/{n_episodes}")
    else:
        rows = []
        completed = set()

    for ep in range(int(n_episodes)):
        if ep in completed:
            continue

        seed = int(seed0 + ep)

        row, path = rollout_m11_dynamic_rl_episode(
            model=model,
            world=world,
            seed=seed,
            dynamic_config=dynamic_config,
            stage=stage,
            mode=mode,
            n_obstacles=0,
            deterministic=True,
            max_steps=1000,
            reward_only_obs=reward_only_obs,
        )

        row["episode"] = int(ep)
        rows.append(row)

        if save_paths:
            np.save(paths_dir / f"episode_{ep:05d}_seed_{seed}_path.npy", path)

        df_now = pd.DataFrame(rows).sort_values("episode")
        df_now.to_csv(csv_path, index=False)

        summary_now = summarize_m11_dynamic_eval_df(df_now)
        summary_now["status"] = "partial" if len(df_now) < int(n_episodes) else "complete"
        summary_now["updated_at"] = datetime.now().isoformat()
        summary_now["dynamic_config"] = dynamic_config.__dict__

        _atomic_write_json(summary_path, summary_now)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[M11 RL eval] ep={ep+1}/{n_episodes} | "
                f"SR={summary_now['success_rate']:.1%} | "
                f"MAR={summary_now['mission_acceptability_rate']:.1%} | "
                f"DynBudget={summary_now['dynamic_budget_ok_rate']:.1%} | "
                f"term={row['termination']}"
            )

    df_final = pd.DataFrame(rows).sort_values("episode")
    summary = summarize_m11_dynamic_eval_df(df_final)
    summary["status"] = "complete"
    summary["finished_at"] = datetime.now().isoformat()
    summary["dynamic_config"] = dynamic_config.__dict__
    summary["csv_path"] = str(csv_path)
    summary["summary_path"] = str(summary_path)

    df_final.to_csv(csv_path, index=False)
    _atomic_write_json(summary_path, summary)

    print("=" * 100)
    print("M11 DYNAMIC RL EVALUATION SUMMARY")
    print("=" * 100)
    for k, v in summary.items():
        if k == "termination_counts":
            print("termination_counts:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        elif isinstance(v, float):
            print(f"{k:45s}: {v:.6f}")
        else:
            print(f"{k:45s}: {v}")
    print("=" * 100)

    return df_final, summary, str(csv_path), str(summary_path)

class M11DynamicRecedingRiskPlanner(M10RecedingHorizonRiskPlanner):
    """
    Receding A* with dynamic soft-risk overlay.

    Static risk is known through prior as in M10.
    Dynamic risk is:
        - true locally within sensing radius,
        - prior elsewhere.
    """

    def __init__(
        self,
        world: UrbanWorldLayers,
        config: M10RecedingHorizonPlannerConfig,
        dynamic_config: M11DynamicRiskConfig,
        episode_seed: int,
        risk_profile: str = "balanced",
        dynamic_prior_mode: str = "zero",
    ):
        self.dynamic_config = dynamic_config
        self.dynamic_prior_mode = str(dynamic_prior_mode)
        self.dynamic_field = M11DynamicRiskField(world, dynamic_config, seed=int(episode_seed))
        self.current_dynamic_true_norm = None
        self.dynamic_known_mask = None

        super().__init__(
            world=world,
            config=config,
            risk_profile=risk_profile,
        )

        self.dynamic_known_mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        self.current_dynamic_true_norm = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def reset_belief(self):
        super().reset_belief()

        self.dynamic_known_mask = np.zeros((self.grid_h, self.grid_w), dtype=bool)
        self.current_dynamic_true_norm = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        self._update_current_maps()

    def _dynamic_prior_norm_map(self):
        if self.dynamic_prior_mode == "zero":
            return np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        if self.dynamic_prior_mode == "mean":
            mean_val = float(np.nanmean(self.current_dynamic_true_norm))
            return np.full((self.grid_h, self.grid_w), mean_val, dtype=np.float32)

        raise ValueError("dynamic_prior_mode must be zero or mean")

    def update_dynamic_true_map(self, step):
        dyn_raw = self.dynamic_field.risk_lowres_map_at_step(
            step=int(step),
            plan_res_m=self.plan_res_m,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            planner_agl_m=self.config.planner_agl_m,
        )

        self.current_dynamic_true_norm = (
            dyn_raw / max(1e-6, float(self.dynamic_config.dynamic_norm_denom))
        ).astype(np.float32)

    def reveal_around_xy(self, xy, step=0):
        """
        Reveal static and dynamic risk around current position.
        """
        self.update_dynamic_true_map(step)

        center = self.xy_to_cell(xy)
        cr, cc = center

        radius_cells = int(np.ceil(float(self.config.sensing_radius_m) / self.plan_res_m))
        radius_cells = max(1, radius_cells)

        r0 = max(0, cr - radius_cells)
        r1 = min(self.grid_h - 1, cr + radius_cells)
        c0 = max(0, cc - radius_cells)
        c1 = min(self.grid_w - 1, cc + radius_cells)

        rr, cc_grid = np.ogrid[r0:r1 + 1, c0:c1 + 1]
        dist2 = (rr - cr) ** 2 + (cc_grid - cc) ** 2
        mask = dist2 <= radius_cells ** 2

        local_known = self.known_mask[r0:r1 + 1, c0:c1 + 1]
        local_known[mask] = True
        self.known_mask[r0:r1 + 1, c0:c1 + 1] = local_known

        local_dyn_known = self.dynamic_known_mask[r0:r1 + 1, c0:c1 + 1]
        local_dyn_known[mask] = True
        self.dynamic_known_mask[r0:r1 + 1, c0:c1 + 1] = local_dyn_known

        if not self.config.hard_blockers_known:
            local_block = self.known_block[r0:r1 + 1, c0:c1 + 1]
            local_true_block = self.true_block[r0:r1 + 1, c0:c1 + 1]
            local_block[mask] = local_true_block[mask]
            self.known_block[r0:r1 + 1, c0:c1 + 1] = local_block

        self._update_current_maps()

    def _update_current_maps(self):
        # Static estimated risk from M10.
        estimated_static_risk = np.where(
            self.known_mask,
            self.true_risk_norm,
            self.prior_risk_norm,
        ).astype(np.float32)

        if self.current_dynamic_true_norm is None:
            estimated_dynamic_risk = np.zeros_like(estimated_static_risk, dtype=np.float32)
        else:
            dynamic_prior = self._dynamic_prior_norm_map()
            estimated_dynamic_risk = np.where(
                self.dynamic_known_mask,
                self.current_dynamic_true_norm,
                dynamic_prior,
            ).astype(np.float32)

        estimated_risk = estimated_static_risk + estimated_dynamic_risk

        if self.config.hard_blockers_known:
            current_block = self.true_block.copy()
        else:
            current_block = self.known_block.copy()

        cell_cost = (1.0 + float(self.config.risk_lambda) * estimated_risk).astype(np.float32)
        cell_cost[current_block] = np.inf

        self.current_cell_cost = cell_cost
        self.current_block = current_block

    def known_fraction(self):
        # Report static known fraction for compatibility.
        return float(np.mean(self.known_mask))

    def dynamic_known_fraction(self):
        return float(np.mean(self.dynamic_known_mask))

def rollout_m11_dynamic_receding_planner_episode(
    world,
    seed,
    cfg: M10RecedingHorizonPlannerConfig,
    dynamic_config: M11DynamicRiskConfig,
    dynamic_prior_mode="zero",
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    max_steps=1000,
):
    env = make_ruscuav_env_m11_dynamic_mode(
        world=world,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
        dynamic_config=dynamic_config,
        reward_only_obs=True,
    )

    obs, reset_info = env.reset(seed=seed)

    planner = M11DynamicRecedingRiskPlanner(
        world=world,
        config=cfg,
        dynamic_config=dynamic_config,
        episode_seed=int(seed),
        risk_profile="balanced",
        dynamic_prior_mode=dynamic_prior_mode,
    )

    planner.reset_belief()
    planner.reveal_around_xy(env.current_pos[:2], step=env.current_step)

    target_xy = env.target_point[:2].copy()

    current_plan = None
    wp_idx = 0

    rewards = []
    infos = []
    path_actual = [env.current_pos.copy()]

    replan_success_count = 0
    replan_fail_count = 0

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        need_replan = False

        if current_plan is None:
            need_replan = True

        if steps % int(max(1, cfg.replan_every_steps)) == 0:
            need_replan = True

        if need_replan:
            planner.reveal_around_xy(env.current_pos[:2], step=env.current_step)

            plan_result = planner.plan(
                start_xy=env.current_pos[:2].copy(),
                target_xy=target_xy,
            )

            if plan_result["success"]:
                current_plan = plan_result["path_xy"]
                wp_idx = 1 if len(current_plan) > 1 else 0
                replan_success_count += 1
            else:
                replan_fail_count += 1

                if current_plan is None:
                    break

        action, wp_idx = m9_waypoint_controller_action(
            env=env,
            path_xy=current_plan,
            wp_idx=wp_idx,
            cfg=_m10_to_m9_config(cfg),
        )

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(dict(info))
        path_actual.append(env.current_pos.copy())

        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}

    if not done and final_info.get("termination", None) is None:
        final_info = dict(final_info)
        final_info["termination"] = "manual_max_steps"
        final_info["is_success"] = False
        final_info["mission_acceptable"] = False
        final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))

    if "dist_xy" not in final_info or final_info.get("dist_xy") is None:
        final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))

    row = {
        "episode": None,
        "seed": int(seed),
        "method": "M11_dynamic_receding_Astar",
        "dynamic_prior_mode": dynamic_prior_mode,

        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),
        "mission_acceptable_static_only": bool(final_info.get("mission_acceptable_static_only", False)),

        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),

        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),

        "dynamic_risk_mean": float(final_info.get("dynamic_risk_mean", np.nan)),
        "dynamic_risk_max": float(final_info.get("dynamic_risk_max", np.nan)),
        "dynamic_cumulative_risk": float(final_info.get("dynamic_cumulative_risk", np.nan)),
        "dynamic_budget_ok": bool(final_info.get("dynamic_budget_ok", False)),

        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),

        "replan_count": int(planner.replan_count),
        "replan_success_count": int(replan_success_count),
        "replan_fail_count": int(replan_fail_count),
        "total_expanded": int(planner.total_expanded),
        "mean_expanded_per_replan": float(planner.total_expanded / max(1, planner.replan_count)),
        "total_plan_time_s": float(planner.total_plan_time_s),
        "mean_plan_time_s": float(planner.total_plan_time_s / max(1, planner.replan_count)),
        "final_known_fraction": float(planner.known_fraction()),
        "final_dynamic_known_fraction": float(planner.dynamic_known_fraction()),
    }

    env.close()

    return row, np.asarray(path_actual, dtype=np.float32)

def evaluate_m11_dynamic_receding_planner_drive(
    world,
    drive_dirs,
    cfg: M10RecedingHorizonPlannerConfig,
    dynamic_config: M11DynamicRiskConfig,
    dynamic_prior_mode="zero",
    n_episodes=200,
    seed0=5000,
    prefix="m11_dynamic_receding_astar_eval",
    save_paths=False,
):
    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths_dir = eval_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        completed = set(int(e) for e in df_existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming M11 planner eval: {len(completed)}/{n_episodes}")
    else:
        rows = []
        completed = set()

    for ep in range(int(n_episodes)):
        if ep in completed:
            continue

        seed = int(seed0 + ep)

        row, path = rollout_m11_dynamic_receding_planner_episode(
            world=world,
            seed=seed,
            cfg=cfg,
            dynamic_config=dynamic_config,
            dynamic_prior_mode=dynamic_prior_mode,
            stage="C",
            mode="strict_eval",
            n_obstacles=0,
            max_steps=cfg.max_steps,
        )

        row["episode"] = int(ep)
        rows.append(row)

        if save_paths:
            np.save(paths_dir / f"episode_{ep:05d}_seed_{seed}_path.npy", path)

        df_now = pd.DataFrame(rows).sort_values("episode")
        df_now.to_csv(csv_path, index=False)

        summary_now = summarize_m11_dynamic_eval_df(df_now)
        summary_now["status"] = "partial" if len(df_now) < int(n_episodes) else "complete"
        summary_now["updated_at"] = datetime.now().isoformat()
        summary_now["dynamic_config"] = dynamic_config.__dict__
        summary_now["planner_config"] = cfg.__dict__
        summary_now["dynamic_prior_mode"] = dynamic_prior_mode

        _atomic_write_json(summary_path, summary_now)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[M11 planner eval] ep={ep+1}/{n_episodes} | "
                f"SR={summary_now['success_rate']:.1%} | "
                f"MAR={summary_now['mission_acceptability_rate']:.1%} | "
                f"DynBudget={summary_now['dynamic_budget_ok_rate']:.1%} | "
                f"term={row['termination']}"
            )

    df_final = pd.DataFrame(rows).sort_values("episode")
    summary = summarize_m11_dynamic_eval_df(df_final)
    summary["status"] = "complete"
    summary["finished_at"] = datetime.now().isoformat()
    summary["dynamic_config"] = dynamic_config.__dict__
    summary["planner_config"] = cfg.__dict__
    summary["dynamic_prior_mode"] = dynamic_prior_mode
    summary["csv_path"] = str(csv_path)
    summary["summary_path"] = str(summary_path)

    df_final.to_csv(csv_path, index=False)
    _atomic_write_json(summary_path, summary)

    print("=" * 100)
    print("M11 DYNAMIC RECEDING PLANNER SUMMARY")
    print("=" * 100)
    for k, v in summary.items():
        if k == "termination_counts":
            print("termination_counts:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        elif isinstance(v, float):
            print(f"{k:45s}: {v:.6f}")
        else:
            print(f"{k:45s}: {v}")
    print("=" * 100)

    return df_final, summary, str(csv_path), str(summary_path)

def posthoc_dynamic_acceptability(df, dynamic_budget, static_required=True):
    if static_required and "mission_acceptable_static_only" in df.columns:
        base = df["mission_acceptable_static_only"] == True
    else:
        base = df["is_success"] == True

    return (
        base
        & (df["dynamic_cumulative_risk"] <= dynamic_budget)
        & (df["urban_hard_violation_steps"] == 0)
    )

def dynamic_budget_only_rate(df, dynamic_budget):
    return (df["dynamic_cumulative_risk"] <= dynamic_budget).mean()

def static_accept_rate(df):
    if "mission_acceptable_static_only" in df.columns:
        return df["mission_acceptable_static_only"].mean()
    return df["mission_acceptable"].mean()

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

from scipy.stats import binomtest

def paired_binom_counts(a_only, b_only):
    n = a_only + b_only
    if n == 0:
        return 1.0
    return binomtest(
        k=min(a_only, b_only),
        n=n,
        p=0.5,
        alternative="two-sided",
    ).pvalue

def run_m11_v1_planner_calibration(
    world,
    drive_dirs,
    dynamic_config,
    risk_lambdas=(0.35, 0.75, 1.5),
    dynamic_prior_modes=("zero", "mean"),
    replan_steps=(1, 3, 5),
    sensing_radius_m=75.0,
    n_episodes=100,
    seed0=5000,
):
    rows = []

    for dyn_prior in dynamic_prior_modes:
        for lam in risk_lambdas:
            for repl in replan_steps:
                cfg = M10RecedingHorizonPlannerConfig(
                    plan_res_m=5.0,
                    planner_agl_m=40.0,
                    risk_lambda=float(lam),
                    use_astar=True,

                    sensing_radius_m=float(sensing_radius_m),
                    replan_every_steps=int(repl),
                    prior_risk_mode="perfect",  # static risk full-map
                    hard_blockers_known=True,

                    waypoint_radius_m=8.0,
                    target_speed_mps=12.0,
                    max_steps=1000,
                )

                prefix = f"m11_v1_planner_dynprior{dyn_prior}_lambda{lam}_replan{repl}_r{sensing_radius_m}_n{n_episodes}"

                df_p, summary_p, csv_p, json_p = evaluate_m11_dynamic_receding_planner_drive(
                    world=world,
                    drive_dirs=drive_dirs,
                    cfg=cfg,
                    dynamic_config=dynamic_config,
                    dynamic_prior_mode=dyn_prior,
                    n_episodes=n_episodes,
                    seed0=seed0,
                    prefix=prefix,
                    save_paths=False,
                )

                rows.append({
                    "dynamic_prior_mode": dyn_prior,
                    "risk_lambda": float(lam),
                    "replan_every_steps": int(repl),
                    "sensing_radius_m": float(sensing_radius_m),

                    "success_rate": summary_p["success_rate"],
                    "mission_acceptability_rate": summary_p["mission_acceptability_rate"],
                    "static_only_acceptability_rate": summary_p["static_only_acceptability_rate"],
                    "dynamic_budget_ok_rate": summary_p["dynamic_budget_ok_rate"],

                    "mean_steps": summary_p["mean_steps"],
                    "mean_dist_xy": summary_p["mean_dist_xy"],
                    "mean_urban_cumulative_risk": summary_p["mean_urban_cumulative_risk"],
                    "mean_dynamic_cumulative_risk": summary_p["mean_dynamic_cumulative_risk"],

                    "mean_replan_count": float(df_p["replan_count"].mean()) if "replan_count" in df_p.columns else np.nan,
                    "mean_total_expanded": float(df_p["total_expanded"].mean()) if "total_expanded" in df_p.columns else np.nan,
                    "mean_total_plan_time_s": float(df_p["total_plan_time_s"].mean()) if "total_plan_time_s" in df_p.columns else np.nan,
                    "mean_plan_time_s": float(df_p["mean_plan_time_s"].mean()) if "mean_plan_time_s" in df_p.columns else np.nan,

                    "termination_counts": json.dumps(summary_p["termination_counts"]),
                    "csv_path": csv_p,
                    "summary_path": json_p,
                })

    out = pd.DataFrame(rows)

    out_path = Path(drive_dirs["eval"]) / "m11_v1_planner_calibration_summary.csv"
    out.to_csv(out_path, index=False)

    print("Saved M11-v1 planner calibration:", out_path)

    return out

from scipy.stats import binomtest

def paired_binom_p(a_only, b_only):
    n = int(a_only + b_only)
    if n == 0:
        return 1.0
    return binomtest(
        k=min(int(a_only), int(b_only)),
        n=n,
        p=0.5,
        alternative="two-sided",
    ).pvalue

def dynamic_tail_summary(df, label):
    s = df["dynamic_cumulative_risk"]
    return {
        "method": label,
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "budget_ok_rate": float((s <= 10.0).mean()),
    }

class RealisticUrbanCargoUAVEnvM11DynamicV2(RealisticUrbanCargoUAVEnvM11Dynamic):
    """
    Same as RealisticUrbanCargoUAVEnvM11Dynamic, but fixes dynamic field seeding.

    If reset(seed=None), we sample an episode dynamic seed from env.np_random.
    This is essential for SB3 training, where most resets happen without explicit seed.
    """

    def reset(self, seed=None, options=None):
        # Call M4/urban reset directly, not old M11 reset.
        obs, info = RealisticUrbanCargoUAVEnvM4.reset(self, seed=seed, options=options)

        if seed is not None:
            dyn_seed = int(seed)
        else:
            dyn_seed = int(self.np_random.integers(0, 2**31 - 1))

        self.dynamic_field = M11DynamicRiskField(
            world=self.world,
            config=self.dynamic_config,
            seed=dyn_seed,
        )

        self._reset_dynamic_metrics()

        info = dict(info)
        info["dynamic_risk_enabled"] = True
        info["dynamic_hotspots"] = len(self.dynamic_field.hotspots)
        info["dynamic_seed"] = int(dyn_seed)

        return obs, info

def make_ruscuav_env_m11_dynamic_mode(
    world,
    stage="C",
    mode="strict_eval",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
    dynamic_config: M11DynamicRiskConfig = None,
    reward_only_obs: bool = True,
):
    """
    Dynamic-risk env factory.

    reward_only_obs=True:
        80D observation, dynamic + static risk reward active.
        This matches our current best learned-policy family.
    """
    mode = str(mode).lower()
    stage = str(stage).upper()

    if mode == "fast_train":
        urban_lidar_step_m = 5.0
        urban_segment_check_step_m = 2.0
    elif mode == "strict_eval":
        urban_lidar_step_m = 3.0
        urban_segment_check_step_m = 1.0
    else:
        raise ValueError("mode must be 'fast_train' or 'strict_eval'.")

    if stage == "A":
        urban_risk_lambda = 0.05
        terminate_on_urban_constraint = False
        hard_sensitive_zones = False

    elif stage == "B":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = False
        hard_sensitive_zones = True

    elif stage == "C":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = True
        hard_sensitive_zones = True

    else:
        raise ValueError("stage must be A, B, or C.")

    cfg = RealisticUrbanEnvConfig(
        risk_profile=risk_profile,

        # reward-only variant: no explicit 20D semantic vector
        include_urban_vector_obs=not bool(reward_only_obs),

        use_urban_risk_costs=True,
        urban_risk_lambda=float(urban_risk_lambda),
        green_bonus_lambda=0.03,

        use_urban_altitude_reference=True,

        terminate_on_urban_constraint=bool(terminate_on_urban_constraint),
        hard_sensitive_zones=bool(hard_sensitive_zones),

        default_n_obstacles=int(n_obstacles),

        mission_accept_cargo_min=0.80,
        mission_accept_road_budget_s=20.0,
        mission_accept_sidewalk_budget_s=5.0,
        mission_accept_building_budget_s=15.0,
        mission_accept_sensitive_budget_s=0.0,

        terminal_road_budget_s=30.0,
        terminal_sidewalk_budget_s=10.0,
        terminal_building_budget_s=20.0,
        terminal_sensitive_budget_s=0.0,
    )

    env = RealisticUrbanCargoUAVEnvM11DynamicV2(
        world=world,
        config=cfg,
        dynamic_config=dynamic_config,
        include_shield_features=False,

        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,

        urban_lidar_step_m=urban_lidar_step_m,
        urban_segment_check_step_m=urban_segment_check_step_m,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        print(
            f"[M11 dynamic env] mode={mode}, stage={stage}, "
            f"obs.shape={obs.shape}, observation_space={env.observation_space}, info={info}"
        )

    return env

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

def make_m11_dynamic_vec_env_from_world_path(
    world_path,
    dynamic_config,
    stage,
    mode,
    rank,
    base_seed=4001,
    n_obstacles=0,
    reward_only_obs=True,
):
    def _init():
        w = UrbanWorldLayers.load_npz(world_path)

        env = make_ruscuav_env_m11_dynamic_mode(
            world=w,
            stage=stage,
            mode=mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=base_seed + rank,
            dynamic_config=dynamic_config,
            reward_only_obs=reward_only_obs,
        )

        env.reset(seed=base_seed + rank)
        return env

    return _init

def make_vec_m11_dynamic_env(
    world_path,
    dynamic_config,
    stage="A",
    mode="fast_train",
    n_envs=4,
    base_seed=4001,
    n_obstacles=0,
    reward_only_obs=True,
    use_subproc=True,
    start_method="fork",
):
    env_fns = [
        make_m11_dynamic_vec_env_from_world_path(
            world_path=world_path,
            dynamic_config=dynamic_config,
            stage=stage,
            mode=mode,
            rank=i,
            base_seed=base_seed,
            n_obstacles=n_obstacles,
            reward_only_obs=reward_only_obs,
        )
        for i in range(int(n_envs))
    ]

    if use_subproc and int(n_envs) > 1:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(
        vec_env,
        info_keywords=RUSCUAV_M11_INFO_KEYWORDS,
    )

    return vec_env

from pathlib import Path

from datetime import datetime

import json

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList

def _m11b_stage_name(stage):
    return f"M11B_dynamic_reward_only_{str(stage).upper()}"

def create_or_load_m11b_model_for_stage(
    stage,
    vec_env,
    drive_dirs,
    previous_model_path=None,
    seed=4001,
    device="cpu",
    n_envs=4,
):
    stage = str(stage).upper()
    stage_name = _m11b_stage_name(stage)
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    model, loaded_from = load_stage_model_if_available(paths, env=vec_env, device=device)

    if model is not None:
        print(f"Resuming M11B stage {stage} from:", loaded_from)
        return model, loaded_from

    if previous_model_path is not None and Path(previous_model_path).exists():
        print(f"Starting M11B stage {stage} from previous stage:", previous_model_path)
        model = PPO.load(str(previous_model_path), env=vec_env, device=device)
        return model, str(previous_model_path)

    print(f"Starting M11B stage {stage} from scratch.")

    n_steps = 512 if int(n_envs) >= 4 else 1024

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=int(seed),
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
    )

    return model, None

def train_one_m11b_stage_resume_safe(
    world_path,
    dynamic_config,
    stage,
    target_stage_steps,
    drive_dirs,
    seed=4001,
    n_obstacles=0,
    previous_model_path=None,
    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    stage = str(stage).upper()
    stage_name = _m11b_stage_name(stage)
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    if is_stage_done(paths, target_stage_steps):
        print(f"M11B stage {stage} already DONE. Skipping.")
        return str(paths["final"])

    vec_env = make_vec_m11_dynamic_env(
        world_path=world_path,
        dynamic_config=dynamic_config,
        stage=stage,
        mode=mode,
        n_envs=n_envs,
        base_seed=seed,
        n_obstacles=n_obstacles,
        reward_only_obs=True,
        use_subproc=(n_envs > 1),
        start_method=start_method,
    )

    model, loaded_from = create_or_load_m11b_model_for_stage(
        stage=stage,
        vec_env=vec_env,
        drive_dirs=drive_dirs,
        previous_model_path=previous_model_path,
        seed=seed,
        device=device,
        n_envs=n_envs,
    )

    progress = read_progress_json(paths["progress"])

    if progress is not None and Path(paths["latest"]).exists():
        stage_start_timesteps = int(progress["stage_start_timesteps"])
        completed_stage_steps = int(progress["completed_stage_steps"])
        print(f"Read M11B progress {stage_name}: completed={completed_stage_steps}/{target_stage_steps}")
    else:
        stage_start_timesteps = int(model.num_timesteps)
        completed_stage_steps = 0
        print(f"New M11B progress {stage_name}: stage_start_timesteps={stage_start_timesteps}")

    remaining = int(max(0, target_stage_steps - completed_stage_steps))

    print("=" * 100)
    print(f"TRAIN M11B DYNAMIC-TRAINED REWARD-ONLY RL | STAGE={stage}")
    print("=" * 100)
    print("target_stage_steps:", target_stage_steps)
    print("completed_stage_steps:", completed_stage_steps)
    print("remaining:", remaining)
    print("model.num_timesteps:", model.num_timesteps)
    print("stage_start_timesteps:", stage_start_timesteps)
    print("dynamic_config:", dynamic_config.__dict__)
    print("stage dir:", paths["stage_dir"])
    print("=" * 100)

    if remaining <= 0:
        final_path = mark_stage_done(
            paths,
            stage_name=stage_name,
            model=model,
            completed_stage_steps=completed_stage_steps,
            target_stage_steps=target_stage_steps,
        )
        vec_env.close()
        return final_path

    autosave_cb = RUSCUAVDriveAutosaveCallback(
        stage_dir=paths["stage_dir"],
        stage_name=stage_name,
        target_stage_steps=target_stage_steps,
        stage_start_timesteps=stage_start_timesteps,
        save_freq_steps=save_freq_steps,
        keep_numbered_checkpoints=True,
        verbose=1,
    )

    stats_cb = RUSCUAVTrainStatsCallback(
        print_every_steps=10_000,
        window=100,
    )

    callback = CallbackList([autosave_cb, stats_cb])
    reset_num_timesteps = bool(model.num_timesteps == 0)

    try:
        model.learn(
            total_timesteps=int(remaining),
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        model.save(str(paths["latest"]))

        progress_now = {
            "stage": stage_name,
            "reason": "finally_block",
            "target_stage_steps": int(target_stage_steps),
            "stage_start_timesteps": int(stage_start_timesteps),
            "last_model_timesteps": int(model.num_timesteps),
            "completed_stage_steps": int(max(0, model.num_timesteps - stage_start_timesteps)),
            "saved_at": datetime.now().isoformat(),
            "latest_model_path": str(paths["latest"]),
            "mode": mode,
            "n_envs": int(n_envs),
            "dynamic_config": dynamic_config.__dict__,
        }

        atomic_write_json(paths["progress"], progress_now)
        print("Emergency/finally save completed:", paths["latest"])

    completed_final = int(max(0, model.num_timesteps - stage_start_timesteps))

    final_path = mark_stage_done(
        paths,
        stage_name=stage_name,
        model=model,
        completed_stage_steps=completed_final,
        target_stage_steps=target_stage_steps,
    )

    vec_env.close()

    return final_path

def train_m11b_dynamic_reward_only_drive_resume(
    world_path,
    dynamic_config,
    drive_dirs,
    seed=4001,
    n_obstacles=0,

    # pilot schedule
    steps_A=50_000,
    steps_B=75_000,
    steps_C=75_000,

    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    """
    Dynamic-trained reward-only RL.

    Suggested pilot:
        A=50k, B=75k, C=75k

    If promising:
        A=150k, B=250k, C=400k
    """
    protocol = {
        "experiment": "RUSCUAV M11B dynamic-trained reward-only RL",
        "world_path": str(world_path),
        "seed": int(seed),
        "n_obstacles": int(n_obstacles),
        "steps_A": int(steps_A),
        "steps_B": int(steps_B),
        "steps_C": int(steps_C),
        "device": device,
        "n_envs": int(n_envs),
        "mode": mode,
        "start_method": start_method,
        "dynamic_config": dynamic_config.__dict__,
        "started_or_resumed_at": datetime.now().isoformat(),
    }

    protocol_path = Path(drive_dirs["metadata"]) / "m11b_dynamic_reward_only_protocol.json"
    atomic_write_json(protocol_path, protocol)

    final_A = train_one_m11b_stage_resume_safe(
        world_path=world_path,
        dynamic_config=dynamic_config,
        stage="A",
        target_stage_steps=int(steps_A),
        drive_dirs=drive_dirs,
        seed=seed,
        n_obstacles=n_obstacles,
        previous_model_path=None,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_B = train_one_m11b_stage_resume_safe(
        world_path=world_path,
        dynamic_config=dynamic_config,
        stage="B",
        target_stage_steps=int(steps_B),
        drive_dirs=drive_dirs,
        seed=seed + 100,
        n_obstacles=n_obstacles,
        previous_model_path=final_A,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_C = train_one_m11b_stage_resume_safe(
        world_path=world_path,
        dynamic_config=dynamic_config,
        stage="C",
        target_stage_steps=int(steps_C),
        drive_dirs=drive_dirs,
        seed=seed + 200,
        n_obstacles=n_obstacles,
        previous_model_path=final_B,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    print("=" * 100)
    print("M11B DYNAMIC-TRAINED RL COMPLETE")
    print("=" * 100)
    print("final_A:", final_A)
    print("final_B:", final_B)
    print("final_C:", final_C)
    print("=" * 100)

    return {
        "final_A": final_A,
        "final_B": final_B,
        "final_C": final_C,
    }

WORLD_PATH_DRIVE = "/content/drive/MyDrive/RUSCUAV/worlds/AOI_FAKE_001_urban_world_layers.npz"

def m11_row_from_summary(name, summary):
    return {
        "model": name,
        "success_rate": summary["success_rate"],
        "mission_acceptability_rate": summary["mission_acceptability_rate"],
        "static_only_acceptability_rate": summary.get("static_only_acceptability_rate", np.nan),
        "acceptable_given_success_rate": summary.get("acceptable_given_success_rate", np.nan),
        "dynamic_budget_ok_rate": summary["dynamic_budget_ok_rate"],
        "mean_steps": summary["mean_steps"],
        "mean_dist_xy": summary["mean_dist_xy"],
        "mean_cargo_health": summary["mean_cargo_health"],
        "mean_urban_cumulative_risk": summary["mean_urban_cumulative_risk"],
        "mean_dynamic_cumulative_risk": summary["mean_dynamic_cumulative_risk"],
        "mean_dynamic_risk": summary["mean_dynamic_risk"],
        "mean_dynamic_risk_max": summary["mean_dynamic_risk_max"],
        "mean_road_time": summary["mean_road_time"],
        "mean_sidewalk_time": summary["mean_sidewalk_time"],
        "mean_green_time": summary["mean_green_time"],
        "mean_hard_violation_steps": summary["mean_hard_violation_steps"],
    }

from scipy.stats import binomtest

def paired_binom_p(a_only, b_only):
    n = int(a_only + b_only)
    if n == 0:
        return 1.0
    return binomtest(
        k=min(int(a_only), int(b_only)),
        n=n,
        p=0.5,
        alternative="two-sided",
    ).pvalue

def paired_compare_m11(df_a, df_b, label_a="A", label_b="B"):
    a = df_a.copy().add_prefix("a_")
    b = df_b.copy().add_prefix("b_")

    paired = a.merge(
        b,
        left_on="a_seed",
        right_on="b_seed",
        how="inner",
    )

    summary = {
        "n_paired": len(paired),

        f"{label_a}_success_{label_b}_fail": int(
            ((paired["a_is_success"] == True) & (paired["b_is_success"] == False)).sum()
        ),
        f"{label_b}_success_{label_a}_fail": int(
            ((paired["b_is_success"] == True) & (paired["a_is_success"] == False)).sum()
        ),

        f"{label_a}_accept_{label_b}_not": int(
            ((paired["a_mission_acceptable"] == True) & (paired["b_mission_acceptable"] == False)).sum()
        ),
        f"{label_b}_accept_{label_a}_not": int(
            ((paired["b_mission_acceptable"] == True) & (paired["a_mission_acceptable"] == False)).sum()
        ),

        f"{label_a}_dynbudget_{label_b}_not": int(
            ((paired["a_dynamic_budget_ok"] == True) & (paired["b_dynamic_budget_ok"] == False)).sum()
        ),
        f"{label_b}_dynbudget_{label_a}_not": int(
            ((paired["b_dynamic_budget_ok"] == True) & (paired["a_dynamic_budget_ok"] == False)).sum()
        ),

        f"mean_delta_steps_{label_a}_minus_{label_b}": float(
            (paired["a_steps"] - paired["b_steps"]).mean()
        ),
        f"mean_delta_dynamic_cumrisk_{label_a}_minus_{label_b}": float(
            (paired["a_dynamic_cumulative_risk"] - paired["b_dynamic_cumulative_risk"]).mean()
        ),
        f"median_delta_dynamic_cumrisk_{label_a}_minus_{label_b}": float(
            (paired["a_dynamic_cumulative_risk"] - paired["b_dynamic_cumulative_risk"]).median()
        ),
        f"mean_delta_static_cumrisk_{label_a}_minus_{label_b}": float(
            (paired["a_urban_cumulative_risk"] - paired["b_urban_cumulative_risk"]).mean()
        ),
    }

    # p-values
    a_succ = summary[f"{label_a}_success_{label_b}_fail"]
    b_succ = summary[f"{label_b}_success_{label_a}_fail"]

    a_acc = summary[f"{label_a}_accept_{label_b}_not"]
    b_acc = summary[f"{label_b}_accept_{label_a}_not"]

    a_dyn = summary[f"{label_a}_dynbudget_{label_b}_not"]
    b_dyn = summary[f"{label_b}_dynbudget_{label_a}_not"]

    pvals = {
        "success_p": paired_binom_p(a_succ, b_succ),
        "acceptability_p": paired_binom_p(a_acc, b_acc),
        "dynamic_budget_ok_p": paired_binom_p(a_dyn, b_dyn),
    }

    return paired, summary, pvals

def dynamic_tail_summary(df, label, budget=10.0):
    s = df["dynamic_cumulative_risk"]

    return {
        "model": label,
        "mean": float(s.mean()),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "budget": float(budget),
        "budget_ok_rate": float((s <= budget).mean()),
    }

import numpy as np

from gymnasium import spaces

M11C_DYNAMIC_OBS_DIM = 6

class RealisticUrbanCargoUAVEnvM11CDynamicObs(RealisticUrbanCargoUAVEnvM11DynamicV2):
    """
    Dynamic-risk-aware RL environment.

    This variant keeps the reward-only semantic formulation:
        - no 20D static semantic vector,
        - static semantic risk reward active,
        - dynamic risk reward active,
        - full urban safety stack active.

    It adds a compact local dynamic-risk sensor to the observation.
    """

    def __init__(
        self,
        world: UrbanWorldLayers,
        config: RealisticUrbanEnvConfig = None,
        dynamic_config: M11DynamicRiskConfig = None,
        include_shield_features: bool = False,
        urban_lidar_enabled: bool = True,
        urban_action_shield_enabled: bool = True,
        urban_geometric_shield_enabled: bool = True,
        urban_lidar_step_m: float = 3.0,
        urban_segment_check_step_m: float = 1.0,
        urban_shield_backoff_m: float = 0.50,
        urban_horizontal_safety_margin_m: float = 1.0,
    ):
        super().__init__(
            world=world,
            config=config,
            dynamic_config=dynamic_config,
            include_shield_features=include_shield_features,
            urban_lidar_enabled=urban_lidar_enabled,
            urban_action_shield_enabled=urban_action_shield_enabled,
            urban_geometric_shield_enabled=urban_geometric_shield_enabled,
            urban_lidar_step_m=urban_lidar_step_m,
            urban_segment_check_step_m=urban_segment_check_step_m,
            urban_shield_backoff_m=urban_shield_backoff_m,
            urban_horizontal_safety_margin_m=urban_horizontal_safety_margin_m,
        )

        self.base_obs_dim_m11c = int(self.observation_space.shape[0])
        self.dynamic_obs_dim_m11c = int(M11C_DYNAMIC_OBS_DIM)
        self.total_obs_dim_m11c = self.base_obs_dim_m11c + self.dynamic_obs_dim_m11c

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.total_obs_dim_m11c,),
            dtype=np.float32,
        )

    def _risk_norm_m11c(self, risk_value):
        """
        Normalize dynamic risk robustly.
        dynamic_norm_denom=10 and clip=5 maps roughly 0..50 risk into 0..1.
        """
        denom = max(1e-6, float(self.dynamic_config.dynamic_norm_denom))
        x = float(risk_value) / denom
        x = np.clip(x, 0.0, 5.0) / 5.0
        return float(x)

    def _dynamic_risk_at_xy_current_z(self, x, y):
        if self.dynamic_field is None:
            return 0.0

        x = float(np.clip(x, MIN_X, MAX_X - 1e-3))
        y = float(np.clip(y, MIN_Y, MAX_Y - 1e-3))

        z_abs = float(self.current_pos[2])
        ground = float(self._ground_at(x, y))
        agl_m = float(z_abs - ground)

        return float(
            self.dynamic_field.risk_at_xy_agl(
                x=x,
                y=y,
                agl_m=agl_m,
                step=int(self.current_step),
            )
        )

    def _sample_dynamic_risk_along_direction(self, direction_xy, distances_m):
        if self.dynamic_field is None:
            return {
                "mean": 0.0,
                "max": 0.0,
            }

        pos_xy = self.current_pos[:2].astype(np.float32)

        d = np.asarray(direction_xy, dtype=np.float32)
        n = float(np.linalg.norm(d) + 1e-9)

        if n < 1e-8:
            d = np.array([1.0, 0.0], dtype=np.float32)
        else:
            d = d / n

        vals = []

        for dist in distances_m:
            p = pos_xy + d * float(dist)
            x = float(np.clip(p[0], MIN_X, MAX_X - 1e-3))
            y = float(np.clip(p[1], MIN_Y, MAX_Y - 1e-3))
            vals.append(self._dynamic_risk_at_xy_current_z(x, y))

        if len(vals) == 0:
            return {"mean": 0.0, "max": 0.0}

        vals = np.asarray(vals, dtype=np.float32)

        return {
            "mean": float(np.mean(vals)),
            "max": float(np.max(vals)),
        }

    def _get_dynamic_vector_obs_m11c(self):
        """
        6D compact dynamic risk observation:
            current risk,
            forward mean,
            forward max,
            left mean,
            right mean,
            dynamic budget used.
        """
        if self.current_pos is None:
            return np.zeros(M11C_DYNAMIC_OBS_DIM, dtype=np.float32)

        current_dyn_risk = self._dynamic_risk_at_xy_current_z(
            float(self.current_pos[0]),
            float(self.current_pos[1]),
        )

        to_target = (self.target_point[:2] - self.current_pos[:2]).astype(np.float32)
        to_target_norm = float(np.linalg.norm(to_target) + 1e-9)

        if to_target_norm < 1e-8:
            forward = np.array([1.0, 0.0], dtype=np.float32)
        else:
            forward = to_target / to_target_norm

        left = np.array([-forward[1], forward[0]], dtype=np.float32)
        right = -left

        distances_fwd = np.array([20.0, 50.0, 100.0, 150.0, 220.0], dtype=np.float32)
        distances_side = np.array([20.0, 50.0, 100.0, 150.0], dtype=np.float32)

        fwd = self._sample_dynamic_risk_along_direction(forward, distances_fwd)
        lft = self._sample_dynamic_risk_along_direction(left, distances_side)
        rgt = self._sample_dynamic_risk_along_direction(right, distances_side)

        budget = max(1e-6, float(self.dynamic_config.dynamic_cumulative_risk_budget))
        budget_used = float(np.clip(self.ep_dynamic_risk_sum / budget, 0.0, 1.0))

        obs = np.array([
            self._risk_norm_m11c(current_dyn_risk),
            self._risk_norm_m11c(fwd["mean"]),
            self._risk_norm_m11c(fwd["max"]),
            self._risk_norm_m11c(lft["mean"]),
            self._risk_norm_m11c(rgt["mean"]),
            budget_used,
        ], dtype=np.float32)

        assert obs.shape[0] == M11C_DYNAMIC_OBS_DIM
        return obs

    def _get_obs(self):
        base_obs = super()._get_obs().astype(np.float32)
        dyn_obs = self._get_dynamic_vector_obs_m11c().astype(np.float32)
        return np.concatenate([base_obs, dyn_obs], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        # Directly call M4 reset, then create dynamic field, then recompute obs.
        obs, info = RealisticUrbanCargoUAVEnvM4.reset(self, seed=seed, options=options)

        if seed is not None:
            dyn_seed = int(seed)
        else:
            dyn_seed = int(self.np_random.integers(0, 2**31 - 1))

        self.dynamic_field = M11DynamicRiskField(
            world=self.world,
            config=self.dynamic_config,
            seed=dyn_seed,
        )

        self._reset_dynamic_metrics()

        info = dict(info)
        info["dynamic_risk_enabled"] = True
        info["dynamic_hotspots"] = len(self.dynamic_field.hotspots)
        info["dynamic_seed"] = int(dyn_seed)

        obs = self._get_obs()

        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                f"M11C observation mismatch: obs.shape={obs.shape}, "
                f"observation_space.shape={self.observation_space.shape}"
            )

        return obs, info

    def step(self, action):
        # Parent step updates dynamic reward/metrics but returns obs computed before dynamic update.
        obs, reward, terminated, truncated, info = super().step(action)

        # Recompute obs after dynamic metrics/risk update.
        obs = self._get_obs()

        return obs, float(reward), bool(terminated), bool(truncated), info

def make_ruscuav_env_m11c_dynamic_obs_mode(
    world,
    stage="C",
    mode="strict_eval",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
    dynamic_config: M11DynamicRiskConfig = None,
):
    """
    M11C factory.

    Observation:
        80D reward-only base + 6D dynamic risk sensor = 86D.

    It does NOT include the old 20D static semantic vector.
    """
    mode = str(mode).lower()
    stage = str(stage).upper()

    if mode == "fast_train":
        urban_lidar_step_m = 5.0
        urban_segment_check_step_m = 2.0

    elif mode == "strict_eval":
        urban_lidar_step_m = 3.0
        urban_segment_check_step_m = 1.0

    else:
        raise ValueError("mode must be 'fast_train' or 'strict_eval'.")

    if stage == "A":
        urban_risk_lambda = 0.05
        terminate_on_urban_constraint = False
        hard_sensitive_zones = False

    elif stage == "B":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = False
        hard_sensitive_zones = True

    elif stage == "C":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = True
        hard_sensitive_zones = True

    else:
        raise ValueError("stage must be A, B, or C.")

    cfg = RealisticUrbanEnvConfig(
        risk_profile=risk_profile,

        # reward-only base: no 20D static semantic vector
        include_urban_vector_obs=False,

        use_urban_risk_costs=True,
        urban_risk_lambda=float(urban_risk_lambda),
        green_bonus_lambda=0.03,

        use_urban_altitude_reference=True,

        terminate_on_urban_constraint=bool(terminate_on_urban_constraint),
        hard_sensitive_zones=bool(hard_sensitive_zones),

        default_n_obstacles=int(n_obstacles),

        mission_accept_cargo_min=0.80,
        mission_accept_road_budget_s=20.0,
        mission_accept_sidewalk_budget_s=5.0,
        mission_accept_building_budget_s=15.0,
        mission_accept_sensitive_budget_s=0.0,

        terminal_road_budget_s=30.0,
        terminal_sidewalk_budget_s=10.0,
        terminal_building_budget_s=20.0,
        terminal_sensitive_budget_s=0.0,
    )

    env = RealisticUrbanCargoUAVEnvM11CDynamicObs(
        world=world,
        config=cfg,
        dynamic_config=dynamic_config,
        include_shield_features=False,

        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,

        urban_lidar_step_m=urban_lidar_step_m,
        urban_segment_check_step_m=urban_segment_check_step_m,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        print(
            f"[M11C env] mode={mode}, stage={stage}, "
            f"obs.shape={obs.shape}, observation_space={env.observation_space}, info={info}"
        )

    return env

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

def make_m11c_vec_env_from_world_path(
    world_path,
    dynamic_config,
    stage,
    mode,
    rank,
    base_seed=5001,
    n_obstacles=0,
):
    def _init():
        w = UrbanWorldLayers.load_npz(world_path)

        env = make_ruscuav_env_m11c_dynamic_obs_mode(
            world=w,
            stage=stage,
            mode=mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=base_seed + rank,
            dynamic_config=dynamic_config,
        )

        env.reset(seed=base_seed + rank)
        return env

    return _init

def make_vec_m11c_dynamic_obs_env(
    world_path,
    dynamic_config,
    stage="A",
    mode="fast_train",
    n_envs=4,
    base_seed=5001,
    n_obstacles=0,
    use_subproc=True,
    start_method="fork",
):
    env_fns = [
        make_m11c_vec_env_from_world_path(
            world_path=world_path,
            dynamic_config=dynamic_config,
            stage=stage,
            mode=mode,
            rank=i,
            base_seed=base_seed,
            n_obstacles=n_obstacles,
        )
        for i in range(int(n_envs))
    ]

    if use_subproc and int(n_envs) > 1:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(
        vec_env,
        info_keywords=RUSCUAV_M11_INFO_KEYWORDS,
    )

    return vec_env

from pathlib import Path

from datetime import datetime

import json

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList

def _m11c_stage_name(stage):
    return f"M11C_dynamic_obs_reward_only_{str(stage).upper()}"

def create_or_load_m11c_model_for_stage(
    stage,
    vec_env,
    drive_dirs,
    previous_model_path=None,
    seed=5001,
    device="cpu",
    n_envs=4,
):
    stage = str(stage).upper()
    stage_name = _m11c_stage_name(stage)
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    model, loaded_from = load_stage_model_if_available(paths, env=vec_env, device=device)

    if model is not None:
        print(f"Resuming M11C stage {stage} from:", loaded_from)
        return model, loaded_from

    if previous_model_path is not None and Path(previous_model_path).exists():
        print(f"Starting M11C stage {stage} from previous stage:", previous_model_path)
        model = PPO.load(str(previous_model_path), env=vec_env, device=device)
        return model, str(previous_model_path)

    print(f"Starting M11C stage {stage} from scratch.")

    n_steps = 512 if int(n_envs) >= 4 else 1024

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=int(seed),
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
    )

    return model, None

def train_one_m11c_stage_resume_safe(
    world_path,
    dynamic_config,
    stage,
    target_stage_steps,
    drive_dirs,
    seed=5001,
    n_obstacles=0,
    previous_model_path=None,
    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    stage = str(stage).upper()
    stage_name = _m11c_stage_name(stage)
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    if is_stage_done(paths, target_stage_steps):
        print(f"M11C stage {stage} already DONE. Skipping.")
        return str(paths["final"])

    vec_env = make_vec_m11c_dynamic_obs_env(
        world_path=world_path,
        dynamic_config=dynamic_config,
        stage=stage,
        mode=mode,
        n_envs=n_envs,
        base_seed=seed,
        n_obstacles=n_obstacles,
        use_subproc=(n_envs > 1),
        start_method=start_method,
    )

    model, loaded_from = create_or_load_m11c_model_for_stage(
        stage=stage,
        vec_env=vec_env,
        drive_dirs=drive_dirs,
        previous_model_path=previous_model_path,
        seed=seed,
        device=device,
        n_envs=n_envs,
    )

    progress = read_progress_json(paths["progress"])

    if progress is not None and Path(paths["latest"]).exists():
        stage_start_timesteps = int(progress["stage_start_timesteps"])
        completed_stage_steps = int(progress["completed_stage_steps"])
        print(f"Read M11C progress {stage_name}: completed={completed_stage_steps}/{target_stage_steps}")
    else:
        stage_start_timesteps = int(model.num_timesteps)
        completed_stage_steps = 0
        print(f"New M11C progress {stage_name}: stage_start_timesteps={stage_start_timesteps}")

    remaining = int(max(0, target_stage_steps - completed_stage_steps))

    print("=" * 100)
    print(f"TRAIN M11C DYNAMIC-RISK-OBS RL | STAGE={stage}")
    print("=" * 100)
    print("target_stage_steps:", target_stage_steps)
    print("completed_stage_steps:", completed_stage_steps)
    print("remaining:", remaining)
    print("model.num_timesteps:", model.num_timesteps)
    print("stage_start_timesteps:", stage_start_timesteps)
    print("dynamic_config:", dynamic_config.__dict__)
    print("stage dir:", paths["stage_dir"])
    print("=" * 100)

    if remaining <= 0:
        final_path = mark_stage_done(
            paths,
            stage_name=stage_name,
            model=model,
            completed_stage_steps=completed_stage_steps,
            target_stage_steps=target_stage_steps,
        )
        vec_env.close()
        return final_path

    autosave_cb = RUSCUAVDriveAutosaveCallback(
        stage_dir=paths["stage_dir"],
        stage_name=stage_name,
        target_stage_steps=target_stage_steps,
        stage_start_timesteps=stage_start_timesteps,
        save_freq_steps=save_freq_steps,
        keep_numbered_checkpoints=True,
        verbose=1,
    )

    stats_cb = RUSCUAVTrainStatsCallback(
        print_every_steps=10_000,
        window=100,
    )

    callback = CallbackList([autosave_cb, stats_cb])
    reset_num_timesteps = bool(model.num_timesteps == 0)

    try:
        model.learn(
            total_timesteps=int(remaining),
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        model.save(str(paths["latest"]))

        progress_now = {
            "stage": stage_name,
            "reason": "finally_block",
            "target_stage_steps": int(target_stage_steps),
            "stage_start_timesteps": int(stage_start_timesteps),
            "last_model_timesteps": int(model.num_timesteps),
            "completed_stage_steps": int(max(0, model.num_timesteps - stage_start_timesteps)),
            "saved_at": datetime.now().isoformat(),
            "latest_model_path": str(paths["latest"]),
            "mode": mode,
            "n_envs": int(n_envs),
            "dynamic_config": dynamic_config.__dict__,
            "obs_dim": 86,
        }

        atomic_write_json(paths["progress"], progress_now)
        print("Emergency/finally save completed:", paths["latest"])

    completed_final = int(max(0, model.num_timesteps - stage_start_timesteps))

    final_path = mark_stage_done(
        paths,
        stage_name=stage_name,
        model=model,
        completed_stage_steps=completed_final,
        target_stage_steps=target_stage_steps,
    )

    vec_env.close()

    return final_path

def train_m11c_dynamic_obs_reward_only_drive_resume(
    world_path,
    dynamic_config,
    drive_dirs,
    seed=5001,
    n_obstacles=0,

    # pilot schedule
    steps_A=50_000,
    steps_B=75_000,
    steps_C=75_000,

    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    """
    Dynamic-risk-aware reward-only RL.

    Observation:
        80D base reward-only obs + 6D dynamic sensor = 86D.
    """
    protocol = {
        "experiment": "RUSCUAV M11C dynamic-risk-observation reward-only RL",
        "world_path": str(world_path),
        "seed": int(seed),
        "n_obstacles": int(n_obstacles),
        "steps_A": int(steps_A),
        "steps_B": int(steps_B),
        "steps_C": int(steps_C),
        "device": device,
        "n_envs": int(n_envs),
        "mode": mode,
        "start_method": start_method,
        "dynamic_config": dynamic_config.__dict__,
        "obs_dim": 86,
        "dynamic_obs_dim": 6,
        "started_or_resumed_at": datetime.now().isoformat(),
    }

    protocol_path = Path(drive_dirs["metadata"]) / "m11c_dynamic_obs_reward_only_protocol.json"
    atomic_write_json(protocol_path, protocol)

    final_A = train_one_m11c_stage_resume_safe(
        world_path=world_path,
        dynamic_config=dynamic_config,
        stage="A",
        target_stage_steps=int(steps_A),
        drive_dirs=drive_dirs,
        seed=seed,
        n_obstacles=n_obstacles,
        previous_model_path=None,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_B = train_one_m11c_stage_resume_safe(
        world_path=world_path,
        dynamic_config=dynamic_config,
        stage="B",
        target_stage_steps=int(steps_B),
        drive_dirs=drive_dirs,
        seed=seed + 100,
        n_obstacles=n_obstacles,
        previous_model_path=final_A,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_C = train_one_m11c_stage_resume_safe(
        world_path=world_path,
        dynamic_config=dynamic_config,
        stage="C",
        target_stage_steps=int(steps_C),
        drive_dirs=drive_dirs,
        seed=seed + 200,
        n_obstacles=n_obstacles,
        previous_model_path=final_B,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    print("=" * 100)
    print("M11C DYNAMIC-RISK-OBS RL COMPLETE")
    print("=" * 100)
    print("final_A:", final_A)
    print("final_B:", final_B)
    print("final_C:", final_C)
    print("=" * 100)

    return {
        "final_A": final_A,
        "final_B": final_B,
        "final_C": final_C,
    }

WORLD_PATH_DRIVE = "/content/drive/MyDrive/RUSCUAV/worlds/AOI_FAKE_001_urban_world_layers.npz"

from stable_baselines3 import PPO

def rollout_m11c_dynamic_obs_rl_episode(
    model,
    world,
    seed,
    dynamic_config,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    deterministic=True,
    max_steps=1000,
):
    env = make_ruscuav_env_m11c_dynamic_obs_mode(
        world=world,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
        dynamic_config=dynamic_config,
    )

    obs, reset_info = env.reset(seed=seed)

    rewards = []
    infos = []
    path = [env.current_pos.copy()]

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(dict(info))
        path.append(env.current_pos.copy())

        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}

    if "dist_xy" not in final_info or final_info.get("dist_xy") is None:
        try:
            final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))
        except Exception:
            final_info["dist_xy"] = np.nan

    row = {
        "episode": None,
        "seed": int(seed),
        "method": "RL_dynamic_obs_reward_only",

        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),
        "mission_acceptable_static_only": bool(final_info.get("mission_acceptable_static_only", False)),

        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),

        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),

        "dynamic_risk_mean": float(final_info.get("dynamic_risk_mean", np.nan)),
        "dynamic_risk_max": float(final_info.get("dynamic_risk_max", np.nan)),
        "dynamic_cumulative_risk": float(final_info.get("dynamic_cumulative_risk", np.nan)),
        "dynamic_budget_ok": bool(final_info.get("dynamic_budget_ok", False)),

        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),

        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),
    }

    env.close()

    return row, np.asarray(path, dtype=np.float32)

def evaluate_m11c_dynamic_obs_rl_drive(
    model_path,
    world,
    drive_dirs,
    dynamic_config,
    n_episodes=200,
    seed0=5000,
    prefix="m11c_dynamic_obs_reward_only_eval_200",
    stage="C",
    mode="strict_eval",
    save_paths=False,
):
    model = PPO.load(str(model_path), device="cpu")

    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths_dir = eval_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        completed = set(int(e) for e in df_existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming M11C RL eval: {len(completed)}/{n_episodes}")
    else:
        rows = []
        completed = set()

    for ep in range(int(n_episodes)):
        if ep in completed:
            continue

        seed = int(seed0 + ep)

        row, path = rollout_m11c_dynamic_obs_rl_episode(
            model=model,
            world=world,
            seed=seed,
            dynamic_config=dynamic_config,
            stage=stage,
            mode=mode,
            n_obstacles=0,
            deterministic=True,
            max_steps=1000,
        )

        row["episode"] = int(ep)
        rows.append(row)

        if save_paths:
            np.save(paths_dir / f"episode_{ep:05d}_seed_{seed}_path.npy", path)

        df_now = pd.DataFrame(rows).sort_values("episode")
        df_now.to_csv(csv_path, index=False)

        summary_now = summarize_m11_dynamic_eval_df(df_now)
        summary_now["status"] = "partial" if len(df_now) < int(n_episodes) else "complete"
        summary_now["updated_at"] = datetime.now().isoformat()
        summary_now["dynamic_config"] = dynamic_config.__dict__
        summary_now["obs_dim"] = 86
        summary_now["dynamic_obs_dim"] = 6

        _atomic_write_json(summary_path, summary_now)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[M11C RL eval] ep={ep+1}/{n_episodes} | "
                f"SR={summary_now['success_rate']:.1%} | "
                f"MAR={summary_now['mission_acceptability_rate']:.1%} | "
                f"DynBudget={summary_now['dynamic_budget_ok_rate']:.1%} | "
                f"term={row['termination']}"
            )

    df_final = pd.DataFrame(rows).sort_values("episode")
    summary = summarize_m11_dynamic_eval_df(df_final)
    summary["status"] = "complete"
    summary["finished_at"] = datetime.now().isoformat()
    summary["dynamic_config"] = dynamic_config.__dict__
    summary["obs_dim"] = 86
    summary["dynamic_obs_dim"] = 6
    summary["csv_path"] = str(csv_path)
    summary["summary_path"] = str(summary_path)

    df_final.to_csv(csv_path, index=False)
    _atomic_write_json(summary_path, summary)

    print("=" * 100)
    print("M11C DYNAMIC-OBS RL EVALUATION SUMMARY")
    print("=" * 100)
    for k, v in summary.items():
        if k == "termination_counts":
            print("termination_counts:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        elif isinstance(v, float):
            print(f"{k:45s}: {v:.6f}")
        else:
            print(f"{k:45s}: {v}")
    print("=" * 100)

    return df_final, summary, str(csv_path), str(summary_path)

def m11_row_from_summary(name, summary):
    return {
        "model": name,
        "success_rate": summary["success_rate"],
        "mission_acceptability_rate": summary["mission_acceptability_rate"],
        "static_only_acceptability_rate": summary.get("static_only_acceptability_rate", np.nan),
        "acceptable_given_success_rate": summary.get("acceptable_given_success_rate", np.nan),
        "dynamic_budget_ok_rate": summary["dynamic_budget_ok_rate"],
        "mean_steps": summary["mean_steps"],
        "mean_dist_xy": summary["mean_dist_xy"],
        "mean_cargo_health": summary["mean_cargo_health"],
        "mean_urban_cumulative_risk": summary["mean_urban_cumulative_risk"],
        "mean_dynamic_cumulative_risk": summary["mean_dynamic_cumulative_risk"],
        "mean_dynamic_risk": summary["mean_dynamic_risk"],
        "mean_dynamic_risk_max": summary["mean_dynamic_risk_max"],
        "mean_road_time": summary["mean_road_time"],
        "mean_sidewalk_time": summary["mean_sidewalk_time"],
        "mean_green_time": summary["mean_green_time"],
        "mean_hard_violation_steps": summary["mean_hard_violation_steps"],
    }

import os

import json

import shutil

from pathlib import Path

from datetime import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import binomtest

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, pd.Series):
        return json_safe(obj.to_dict())

    if isinstance(obj, pd.DataFrame):
        return json_safe(obj.to_dict(orient="records"))

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")

    with open(tmp, "w") as f:
        json.dump(json_safe(obj), f, indent=2)

    tmp.replace(path)
    print("Saved JSON:", path)

def safe_get_global(name, default=None):
    return globals().get(name, default)

def safe_summary_value(summary, key, default=np.nan):
    if summary is None:
        return default
    try:
        return summary.get(key, default)
    except Exception:
        return default

def termination_count(summary, term):
    if summary is None:
        return 0
    try:
        return int(summary.get("termination_counts", {}).get(term, 0))
    except Exception:
        return 0

MODEL_REGISTRY = [
    # ---------------- Static / learned policies ----------------
    {
        "model": "M7_geometry_only_RL",
        "regime": "static",
        "method_type": "learned_policy",
        "summary_var": "summary_geo_eval",
        "df_var": "df_geo_eval",
        "semantic_obs": False,
        "risk_reward": False,
        "risk_cost_objective": False,
        "dynamic_obs": False,
        "full_map": False,
        "receding_replan": False,
        "planner": False,
    },
    {
        "model": "M8_obs_only_RL",
        "regime": "static",
        "method_type": "learned_policy",
        "summary_var": "summary_m8_obs",
        "df_var": "df_m8_obs",
        "semantic_obs": True,
        "risk_reward": False,
        "risk_cost_objective": False,
        "dynamic_obs": False,
        "full_map": False,
        "receding_replan": False,
        "planner": False,
    },
    {
        "model": "M6_full_semantic_RL",
        "regime": "static",
        "method_type": "learned_policy",
        "summary_var": "summary_eval",
        "df_var": "df_eval",
        "semantic_obs": True,
        "risk_reward": True,
        "risk_cost_objective": True,
        "dynamic_obs": False,
        "full_map": False,
        "receding_replan": False,
        "planner": False,
    },
    {
        "model": "M8_reward_only_RL",
        "regime": "static",
        "method_type": "learned_policy",
        "summary_var": "summary_m8_reward_200",
        "df_var": "df_m8_reward_200",
        "semantic_obs": False,
        "risk_reward": True,
        "risk_cost_objective": True,
        "dynamic_obs": False,
        "full_map": False,
        "receding_replan": False,
        "planner": False,
    },

    # ---------------- Static / planners ----------------
    {
        "model": "M9_static_risk_Astar_one_shot",
        "regime": "static",
        "method_type": "planner",
        "summary_var": "summary_m9_b0",
        "df_var": "df_m9_b0",
        "semantic_obs": False,
        "risk_reward": False,
        "risk_cost_objective": True,
        "dynamic_obs": False,
        "full_map": True,
        "receding_replan": False,
        "planner": True,
    },
    {
        "model": "M10_partial_risk_receding_Astar",
        "regime": "static",
        "method_type": "planner",
        "summary_var": "summary_m10_b1",
        "df_var": "df_m10_b1",
        "semantic_obs": False,
        "risk_reward": False,
        "risk_cost_objective": True,
        "dynamic_obs": False,
        "full_map": False,
        "receding_replan": True,
        "planner": True,
    },
    {
        "model": "M10_fullmap_receding_Astar",
        "regime": "static",
        "method_type": "planner",
        "summary_var": "summary_m10_b0r",
        "df_var": "df_m10_b0r",
        "semantic_obs": False,
        "risk_reward": False,
        "risk_cost_objective": True,
        "dynamic_obs": False,
        "full_map": True,
        "receding_replan": True,
        "planner": True,
    },

    # ---------------- Dynamic regime ----------------
    {
        "model": "M11_RL_static_trained_reward_only",
        "regime": "dynamic",
        "method_type": "learned_policy",
        "summary_var": "summary_m11_rl_v1_200",
        "df_var": "df_m11_rl_v1_200",
        "semantic_obs": False,
        "risk_reward": True,
        "risk_cost_objective": True,
        "dynamic_obs": False,
        "full_map": False,
        "receding_replan": False,
        "planner": False,
    },
    {
        "model": "M11_Planner_receding_Astar_best",
        "regime": "dynamic",
        "method_type": "planner",
        "summary_var": "summary_m11_planner_v1_best_200",
        "df_var": "df_m11_planner_v1_best_200",
        "semantic_obs": False,
        "risk_reward": False,
        "risk_cost_objective": True,
        "dynamic_obs": True,
        "full_map": True,
        "receding_replan": True,
        "planner": True,
    },
    {
        "model": "M11B_RL_dynamic_trained_no_dynamic_obs",
        "regime": "dynamic",
        "method_type": "learned_policy",
        "summary_var": "summary_m11b_rl_200",
        "df_var": "df_m11b_rl_200",
        "semantic_obs": False,
        "risk_reward": True,
        "risk_cost_objective": True,
        "dynamic_obs": False,
        "full_map": False,
        "receding_replan": False,
        "planner": False,
    },
    {
        "model": "M11C_RL_dynamic_trained_dynamic_obs",
        "regime": "dynamic",
        "method_type": "learned_policy",
        "summary_var": "summary_m11c_rl_200",
        "df_var": "df_m11c_rl_200",
        "semantic_obs": False,
        "risk_reward": True,
        "risk_cost_objective": True,
        "dynamic_obs": True,
        "full_map": False,
        "receding_replan": False,
        "planner": False,
    },
]

def make_model_row(reg):
    summary = safe_get_global(reg["summary_var"], None)

    if summary is None:
        return None

    row = {
        "model": reg["model"],
        "regime": reg["regime"],
        "method_type": reg["method_type"],

        "semantic_obs": reg["semantic_obs"],
        "risk_reward": reg["risk_reward"],
        "risk_cost_objective": reg["risk_cost_objective"],
        "dynamic_obs": reg["dynamic_obs"],
        "full_map": reg["full_map"],
        "receding_replan": reg["receding_replan"],
        "planner": reg["planner"],

        "success_rate": safe_summary_value(summary, "success_rate"),
        "mission_acceptability_rate": safe_summary_value(summary, "mission_acceptability_rate"),
        "static_only_acceptability_rate": safe_summary_value(summary, "static_only_acceptability_rate"),
        "acceptable_given_success_rate": safe_summary_value(summary, "acceptable_given_success_rate"),
        "dynamic_budget_ok_rate": safe_summary_value(summary, "dynamic_budget_ok_rate"),

        "mean_reward": safe_summary_value(summary, "mean_reward"),
        "std_reward": safe_summary_value(summary, "std_reward"),

        "mean_steps": safe_summary_value(summary, "mean_steps"),
        "mean_dist_xy": safe_summary_value(summary, "mean_dist_xy"),
        "median_dist_xy": safe_summary_value(summary, "median_dist_xy"),
        "mean_cargo_health": safe_summary_value(summary, "mean_cargo_health"),

        "mean_urban_risk": safe_summary_value(summary, "mean_urban_risk"),
        "mean_cumulative_risk": safe_summary_value(summary, "mean_cumulative_risk"),
        "mean_urban_cumulative_risk": safe_summary_value(summary, "mean_urban_cumulative_risk"),

        "mean_dynamic_risk": safe_summary_value(summary, "mean_dynamic_risk"),
        "mean_dynamic_risk_max": safe_summary_value(summary, "mean_dynamic_risk_max"),
        "mean_dynamic_cumulative_risk": safe_summary_value(summary, "mean_dynamic_cumulative_risk"),

        "mean_road_time": safe_summary_value(summary, "mean_road_time"),
        "mean_sidewalk_time": safe_summary_value(summary, "mean_sidewalk_time"),
        "mean_building_time": safe_summary_value(summary, "mean_building_time"),
        "mean_green_time": safe_summary_value(summary, "mean_green_time"),
        "mean_canopy_time": safe_summary_value(summary, "mean_canopy_time"),
        "mean_sensitive_time": safe_summary_value(summary, "mean_sensitive_time"),
        "mean_hard_violation_steps": safe_summary_value(summary, "mean_hard_violation_steps"),

        "mean_replan_count": safe_summary_value(summary, "mean_replan_count"),
        "mean_total_expanded": safe_summary_value(summary, "mean_total_expanded"),
        "mean_total_plan_time_s": safe_summary_value(summary, "mean_total_plan_time_s"),
        "mean_plan_time_s": safe_summary_value(summary, "mean_plan_time_s"),
        "mean_final_known_fraction": safe_summary_value(summary, "mean_final_known_fraction"),

        "term_success": termination_count(summary, "success"),
        "term_stuck": termination_count(summary, "stuck"),
        "term_shield_stuck": termination_count(summary, "shield_stuck"),
        "term_near_success": termination_count(summary, "near_success"),
        "term_urban_sidewalk_budget_exceeded": termination_count(summary, "urban_sidewalk_budget_exceeded"),
        "term_urban_road_budget_exceeded": termination_count(summary, "urban_road_budget_exceeded"),
        "term_urban_building_collision": termination_count(summary, "urban_building_collision"),
        "term_urban_canopy_collision": termination_count(summary, "urban_canopy_collision"),
        "term_urban_sensitive_low_altitude_violation": termination_count(summary, "urban_sensitive_low_altitude_violation"),
    }

    # Harmonize risk naming:
    if pd.isna(row["mean_cumulative_risk"]) and not pd.isna(row["mean_urban_cumulative_risk"]):
        row["mean_cumulative_risk"] = row["mean_urban_cumulative_risk"]

    if pd.isna(row["mean_urban_cumulative_risk"]) and not pd.isna(row["mean_cumulative_risk"]):
        row["mean_urban_cumulative_risk"] = row["mean_cumulative_risk"]

    return row

def build_final_summary_tables():
    rows = []

    for reg in MODEL_REGISTRY:
        row = make_model_row(reg)
        if row is None:
            print(f"[skip] missing summary variable: {reg['summary_var']} for {reg['model']}")
            continue
        rows.append(row)

    all_df = pd.DataFrame(rows)

    static_df = all_df[all_df["regime"] == "static"].copy()
    dynamic_df = all_df[all_df["regime"] == "dynamic"].copy()

    all_path = FINAL_TABLES_DIR / "final_all_methods_summary.csv"
    static_path = FINAL_TABLES_DIR / "final_static_regime_summary.csv"
    dynamic_path = FINAL_TABLES_DIR / "final_dynamic_regime_summary.csv"

    all_df.to_csv(all_path, index=False)
    static_df.to_csv(static_path, index=False)
    dynamic_df.to_csv(dynamic_path, index=False)

    print("Saved:", all_path)
    print("Saved:", static_path)
    print("Saved:", dynamic_path)

    return all_df, static_df, dynamic_df

def paired_binom_p(a_only, b_only):
    n = int(a_only + b_only)
    if n == 0:
        return 1.0

    return float(
        binomtest(
            k=min(int(a_only), int(b_only)),
            n=n,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )

def paired_compare_binary_and_continuous(df_a, df_b, label_a, label_b):
    """
    Paired comparison by seed.

    Required columns:
        seed
        is_success
        mission_acceptable
        steps
        dist_xy

    Optional columns:
        dynamic_budget_ok
        dynamic_cumulative_risk
        urban_cumulative_risk or urban_mean_risk
        road_time / sidewalk_time
    """
    a = df_a.copy().add_prefix("a_")
    b = df_b.copy().add_prefix("b_")

    paired = a.merge(
        b,
        left_on="a_seed",
        right_on="b_seed",
        how="inner",
    )

    def bool_col(name):
        return paired[name].astype(bool)

    summary = {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": int(len(paired)),
    }

    if len(paired) == 0:
        return paired, summary, {}

    # Binary outcomes
    binary_pairs = [
        ("success", "is_success"),
        ("mission_acceptability", "mission_acceptable"),
    ]

    if "a_dynamic_budget_ok" in paired.columns and "b_dynamic_budget_ok" in paired.columns:
        binary_pairs.append(("dynamic_budget_ok", "dynamic_budget_ok"))

    pvals = {}

    for out_name, col in binary_pairs:
        a_col = f"a_{col}"
        b_col = f"b_{col}"

        a_only = int(((bool_col(a_col) == True) & (bool_col(b_col) == False)).sum())
        b_only = int(((bool_col(b_col) == True) & (bool_col(a_col) == False)).sum())

        summary[f"{label_a}_{out_name}_{label_b}_not"] = a_only
        summary[f"{label_b}_{out_name}_{label_a}_not"] = b_only
        summary[f"{out_name}_discordant"] = a_only + b_only

        pvals[f"{out_name}_p"] = paired_binom_p(a_only, b_only)

    # Continuous deltas
    continuous_cols = [
        "steps",
        "dist_xy",
        "cargo_health",
        "urban_cumulative_risk",
        "mean_urban_risk",
        "urban_mean_risk",
        "dynamic_cumulative_risk",
        "road_time",
        "sidewalk_time",
        "green_time",
    ]

    for col in continuous_cols:
        a_col = f"a_{col}"
        b_col = f"b_{col}"

        if a_col in paired.columns and b_col in paired.columns:
            delta = paired[a_col] - paired[b_col]
            summary[f"mean_delta_{col}_{label_a}_minus_{label_b}"] = float(delta.mean())
            summary[f"median_delta_{col}_{label_a}_minus_{label_b}"] = float(delta.median())

    return paired, summary, pvals

def get_df_by_var(name):
    df = safe_get_global(name, None)
    if df is None:
        print(f"[missing df] {name}")
    return df

PAIRED_COMPARISONS = [
    # Static learned policy comparisons
    ("df_m8_reward_200", "df_geo_eval", "M8_reward_only", "M7_geometry_only"),
    ("df_eval", "df_geo_eval", "M6_full_semantic", "M7_geometry_only"),
    ("df_m8_reward_200", "df_eval", "M8_reward_only", "M6_full_semantic"),

    # Planner comparisons
    ("df_m8_reward_200", "df_m9_b0", "M8_reward_only", "M9_static_Astar"),
    ("df_m8_reward_200", "df_m10_b0r", "M8_reward_only", "M10_fullmap_receding_Astar"),

    # Dynamic comparisons
    ("df_m11_rl_v1_200", "df_m11_planner_v1_best_200", "M11_staticRL", "M11_planner"),
    ("df_m11b_rl_200", "df_m11_planner_v1_best_200", "M11B_dynamicRL_noDynObs", "M11_planner"),
    ("df_m11c_rl_200", "df_m11_planner_v1_best_200", "M11C_dynamicObsRL", "M11_planner"),
    ("df_m11c_rl_200", "df_m11b_rl_200", "M11C_dynamicObsRL", "M11B_noDynObsRL"),
    ("df_m11c_rl_200", "df_m11_rl_v1_200", "M11C_dynamicObsRL", "M11_staticRL"),
]

def run_all_paired_comparisons():
    summaries = []
    pvals_rows = []

    for df_a_name, df_b_name, label_a, label_b in PAIRED_COMPARISONS:
        df_a = get_df_by_var(df_a_name)
        df_b = get_df_by_var(df_b_name)

        if df_a is None or df_b is None:
            print(f"[skip paired] {label_a} vs {label_b}")
            continue

        paired, summary, pvals = paired_compare_binary_and_continuous(
            df_a=df_a,
            df_b=df_b,
            label_a=label_a,
            label_b=label_b,
        )

        paired_path = FINAL_TABLES_DIR / f"paired_{label_a}_vs_{label_b}.csv"
        paired.to_csv(paired_path, index=False)

        summary["paired_csv"] = str(paired_path)
        summaries.append(summary)

        pvals_row = {
            "comparison": f"{label_a}_vs_{label_b}",
            **pvals,
        }
        pvals_rows.append(pvals_row)

        write_json(
            FINAL_JSON_DIR / f"paired_{label_a}_vs_{label_b}_summary.json",
            {
                "summary": summary,
                "p_values": pvals,
            },
        )

    summaries_df = pd.DataFrame(summaries)
    pvals_df = pd.DataFrame(pvals_rows)

    summaries_path = FINAL_TABLES_DIR / "final_paired_comparison_summaries.csv"
    pvals_path = FINAL_TABLES_DIR / "final_paired_p_values.csv"

    summaries_df.to_csv(summaries_path, index=False)
    pvals_df.to_csv(pvals_path, index=False)

    print("Saved:", summaries_path)
    print("Saved:", pvals_path)

    return summaries_df, pvals_df

def dynamic_tail_summary(df, label, budget=10.0):
    if df is None or "dynamic_cumulative_risk" not in df.columns:
        return None

    s = df["dynamic_cumulative_risk"].astype(float)

    return {
        "model": label,
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "budget": float(budget),
        "budget_ok_rate": float((s <= budget).mean()),
    }

def build_dynamic_tail_tables(budget=10.0):
    models = [
        ("df_m11_rl_v1_200", "M11_RL_static_trained_reward_only"),
        ("df_m11_planner_v1_best_200", "M11_Planner_receding_Astar_best"),
        ("df_m11b_rl_200", "M11B_RL_dynamic_trained_no_dynamic_obs"),
        ("df_m11c_rl_200", "M11C_RL_dynamic_trained_dynamic_obs"),
    ]

    rows = []

    for df_name, label in models:
        df = get_df_by_var(df_name)
        row = dynamic_tail_summary(df, label, budget=budget)
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)

    out_path = FINAL_TABLES_DIR / "final_dynamic_tail_risk_summary.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)

    return out

def posthoc_dynamic_acceptability(df, dynamic_budget, static_required=True):
    if df is None:
        return pd.Series(dtype=bool)

    if static_required and "mission_acceptable_static_only" in df.columns:
        base = df["mission_acceptable_static_only"].astype(bool)
    else:
        base = df["is_success"].astype(bool)

    return (
        base
        & (df["dynamic_cumulative_risk"].astype(float) <= float(dynamic_budget))
        & (df["urban_hard_violation_steps"].astype(float) == 0)
    )

def dynamic_budget_only(df, dynamic_budget):
    if df is None:
        return pd.Series(dtype=bool)
    return df["dynamic_cumulative_risk"].astype(float) <= float(dynamic_budget)

def build_dynamic_budget_sensitivity(budgets=(1.0, 2.5, 5.0, 10.0, 20.0, 35.0)):
    models = [
        ("df_m11_rl_v1_200", "M11_staticRL"),
        ("df_m11_planner_v1_best_200", "M11_planner"),
        ("df_m11b_rl_200", "M11B_dynamicRL_noDynObs"),
        ("df_m11c_rl_200", "M11C_dynamicObsRL"),
    ]

    rows = []

    for b in budgets:
        for df_name, label in models:
            df = get_df_by_var(df_name)
            if df is None or "dynamic_cumulative_risk" not in df.columns:
                continue

            rows.append({
                "model": label,
                "dynamic_budget": float(b),
                "dynamic_budget_only_rate": float(dynamic_budget_only(df, b).mean()),
                "full_dynamic_acceptability_rate": float(posthoc_dynamic_acceptability(df, b).mean()),
                "success_rate": float(df["is_success"].astype(bool).mean()),
                "static_only_acceptability_rate": (
                    float(df["mission_acceptable_static_only"].astype(bool).mean())
                    if "mission_acceptable_static_only" in df.columns else np.nan
                ),
            })

    out = pd.DataFrame(rows)

    out_path = FINAL_TABLES_DIR / "final_dynamic_budget_sensitivity.csv"
    out.to_csv(out_path, index=False)

    print("Saved:", out_path)

    return out

def select_diagnostic_seeds(df, label, n=5):
    if df is None or len(df) == 0:
        return {}

    out = {}

    if "seed" not in df.columns:
        return out

    # Accepted successes with lowest risk
    if "mission_acceptable" in df.columns:
        acc = df[(df["is_success"] == True) & (df["mission_acceptable"] == True)].copy()
        if len(acc) > 0:
            sort_col = "dynamic_cumulative_risk" if "dynamic_cumulative_risk" in acc.columns else (
                "urban_cumulative_risk" if "urban_cumulative_risk" in acc.columns else "total_reward"
            )
            out["accepted_success_low_risk"] = acc.sort_values(sort_col)["seed"].head(n).astype(int).tolist()

    # Non-accepted successes
    bad_succ = df[(df["is_success"] == True) & (df["mission_acceptable"] == False)].copy()
    if len(bad_succ) > 0:
        sort_col = "dynamic_cumulative_risk" if "dynamic_cumulative_risk" in bad_succ.columns else (
            "urban_cumulative_risk" if "urban_cumulative_risk" in bad_succ.columns else "total_reward"
        )
        out["nonaccepted_success_high_risk"] = (
            bad_succ.sort_values(sort_col, ascending=False)["seed"].head(n).astype(int).tolist()
        )

    # Stuck / shield stuck
    stuck = df[df["termination"].isin(["stuck", "shield_stuck"])].copy()
    if len(stuck) > 0:
        out["stuck_or_shield_stuck"] = stuck.sort_values("dist_xy")["seed"].head(n).astype(int).tolist()

    # Dynamic outliers
    if "dynamic_cumulative_risk" in df.columns:
        dyn = df.copy()
        out["dynamic_risk_outliers"] = (
            dyn.sort_values("dynamic_cumulative_risk", ascending=False)["seed"].head(n).astype(int).tolist()
        )

        dyn_ok_fail = df[(df["dynamic_budget_ok"] == False)].copy() if "dynamic_budget_ok" in df.columns else pd.DataFrame()
        if len(dyn_ok_fail) > 0:
            out["dynamic_budget_failures"] = (
                dyn_ok_fail.sort_values("dynamic_cumulative_risk", ascending=False)["seed"].head(n).astype(int).tolist()
            )

    return out

def build_all_diagnostic_seeds():
    seed_dict = {}

    for reg in MODEL_REGISTRY:
        df = safe_get_global(reg["df_var"], None)
        if df is None:
            continue

        seed_dict[reg["model"]] = select_diagnostic_seeds(df, reg["model"], n=5)

    path = FINAL_SEEDS_DIR / "diagnostic_seeds_all_models.json"
    write_json(path, seed_dict)

    return seed_dict

def save_current_fig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print("Saved figure:", path)
    plt.show()

def plot_static_sr_mar(static_df):
    if len(static_df) == 0:
        return

    df = static_df.copy()
    df = df.sort_values("mission_acceptability_rate", ascending=True)

    y = np.arange(len(df))

    plt.figure(figsize=(10, max(4, 0.45 * len(df))))
    plt.barh(y - 0.18, df["success_rate"], height=0.35, label="Success rate")
    plt.barh(y + 0.18, df["mission_acceptability_rate"], height=0.35, label="Mission acceptability")
    plt.yticks(y, df["model"])
    plt.xlabel("Rate")
    plt.title("Static regime: success vs mission acceptability")
    plt.xlim(0.0, 1.05)
    plt.grid(axis="x", alpha=0.3)
    plt.legend()

    save_current_fig(FINAL_FIGURES_DIR / "static_success_vs_acceptability.png")

def plot_static_service_risk(static_df):
    if len(static_df) == 0:
        return

    df = static_df.copy()

    plt.figure(figsize=(8, 6))

    for _, row in df.iterrows():
        x = row["mean_steps"]
        y = row["mean_cumulative_risk"]

        if pd.isna(x) or pd.isna(y):
            continue

        plt.scatter(x, y, s=80)
        plt.annotate(row["model"], (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)

    plt.xlabel("Mean episode steps")
    plt.ylabel("Mean cumulative urban risk")
    plt.title("Static regime: service–risk trade-off")
    plt.grid(alpha=0.3)

    save_current_fig(FINAL_FIGURES_DIR / "static_service_risk_tradeoff.png")

def plot_dynamic_key_metrics(dynamic_df):
    if len(dynamic_df) == 0:
        return

    df = dynamic_df.copy()
    df = df.sort_values("mission_acceptability_rate", ascending=True)

    y = np.arange(len(df))

    plt.figure(figsize=(10, max(4, 0.45 * len(df))))
    plt.barh(y - 0.25, df["success_rate"], height=0.25, label="Success")
    plt.barh(y, df["mission_acceptability_rate"], height=0.25, label="Mission acceptable")
    plt.barh(y + 0.25, df["dynamic_budget_ok_rate"], height=0.25, label="Dynamic budget OK")

    plt.yticks(y, df["model"])
    plt.xlabel("Rate")
    plt.title("Dynamic regime: success, mission acceptability, dynamic budget")
    plt.xlim(0.0, 1.05)
    plt.grid(axis="x", alpha=0.3)
    plt.legend()

    save_current_fig(FINAL_FIGURES_DIR / "dynamic_success_accept_dynbudget.png")

def plot_dynamic_tail_box():
    models = [
        ("df_m11_rl_v1_200", "Static-trained RL"),
        ("df_m11_planner_v1_best_200", "Receding A*"),
        ("df_m11b_rl_200", "Dynamic-trained RL\nno dyn obs"),
        ("df_m11c_rl_200", "Dynamic-trained RL\n6D dyn obs"),
    ]

    data = []
    labels = []

    for df_name, label in models:
        df = safe_get_global(df_name, None)
        if df is None or "dynamic_cumulative_risk" not in df.columns:
            continue

        data.append(df["dynamic_cumulative_risk"].astype(float).values)
        labels.append(label)

    if not data:
        return

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.axhline(10.0, linestyle="--", label="Dynamic budget = 10")
    plt.ylabel("Dynamic cumulative risk")
    plt.title("Dynamic regime: dynamic risk distribution without extreme outliers")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    save_current_fig(FINAL_FIGURES_DIR / "dynamic_tail_risk_boxplot_no_outliers.png")

def plot_dynamic_ecdf():
    models = [
        ("df_m11_rl_v1_200", "Static-trained RL"),
        ("df_m11_planner_v1_best_200", "Receding A*"),
        ("df_m11b_rl_200", "Dynamic-trained RL no dyn obs"),
        ("df_m11c_rl_200", "Dynamic-trained RL 6D dyn obs"),
    ]

    plt.figure(figsize=(8, 6))

    any_data = False

    for df_name, label in models:
        df = safe_get_global(df_name, None)
        if df is None or "dynamic_cumulative_risk" not in df.columns:
            continue

        x = np.sort(df["dynamic_cumulative_risk"].astype(float).values)
        y = np.arange(1, len(x) + 1) / max(1, len(x))

        plt.plot(x, y, label=label)
        any_data = True

    if not any_data:
        plt.close()
        return

    plt.axvline(10.0, linestyle="--", label="Dynamic budget = 10")
    plt.xlabel("Dynamic cumulative risk")
    plt.ylabel("ECDF")
    plt.title("Dynamic cumulative risk ECDF")
    plt.grid(alpha=0.3)
    plt.legend()

    save_current_fig(FINAL_FIGURES_DIR / "dynamic_cumulative_risk_ecdf.png")

def build_final_manifest():
    manifest = {
        "package_name": "RUSCUAV-v0 synthetic benchmark final package",
        "created_at": datetime.now().isoformat(),
        "final_root": str(FINAL_ROOT),
        "description": (
            "Consolidated static and dynamic synthetic benchmark results for RUSCUAV. "
            "Includes learned-policy baselines, semantic-risk ablations, risk-cost planners, "
            "dynamic risk evaluations, paired tests, tail-risk summaries, figures, and diagnostic seeds."
        ),
        "available_variables": {
            reg["model"]: {
                "summary_var": reg["summary_var"],
                "summary_available": reg["summary_var"] in globals(),
                "df_var": reg["df_var"],
                "df_available": reg["df_var"] in globals(),
            }
            for reg in MODEL_REGISTRY
        },
        "key_tables": {
            "all_methods_summary": str(FINAL_TABLES_DIR / "final_all_methods_summary.csv"),
            "static_regime_summary": str(FINAL_TABLES_DIR / "final_static_regime_summary.csv"),
            "dynamic_regime_summary": str(FINAL_TABLES_DIR / "final_dynamic_regime_summary.csv"),
            "paired_comparison_summaries": str(FINAL_TABLES_DIR / "final_paired_comparison_summaries.csv"),
            "paired_p_values": str(FINAL_TABLES_DIR / "final_paired_p_values.csv"),
            "dynamic_tail_risk_summary": str(FINAL_TABLES_DIR / "final_dynamic_tail_risk_summary.csv"),
            "dynamic_budget_sensitivity": str(FINAL_TABLES_DIR / "final_dynamic_budget_sensitivity.csv"),
        },
        "figures": {
            "static_success_vs_acceptability": str(FINAL_FIGURES_DIR / "static_success_vs_acceptability.png"),
            "static_service_risk_tradeoff": str(FINAL_FIGURES_DIR / "static_service_risk_tradeoff.png"),
            "dynamic_success_accept_dynbudget": str(FINAL_FIGURES_DIR / "dynamic_success_accept_dynbudget.png"),
            "dynamic_tail_risk_boxplot_no_outliers": str(FINAL_FIGURES_DIR / "dynamic_tail_risk_boxplot_no_outliers.png"),
            "dynamic_cumulative_risk_ecdf": str(FINAL_FIGURES_DIR / "dynamic_cumulative_risk_ecdf.png"),
        },
        "diagnostic_seeds": str(FINAL_SEEDS_DIR / "diagnostic_seeds_all_models.json"),
        "notes": {
            "static_regime_interpretation": (
                "Risk-cost replanning is the strongest method in the static/full-map regime."
            ),
            "dynamic_regime_interpretation": (
                "Dynamic risk reveals a trade-off: planner remains best for success and tail-risk control, "
                "while dynamic-observation RL improves threshold-based dynamic-budget satisfaction."
            ),
        },
    }

    manifest_path = FINAL_JSON_DIR / "final_manifest.json"
    write_json(manifest_path, manifest)

    return manifest

def make_text_summary():
    lines = []
    lines.append("RUSCUAV-v0 Synthetic Benchmark — Consolidated Findings")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Static regime:")
    lines.append("- Geometry-only RL establishes the baseline learned policy.")
    lines.append("- Reward-only RL improves learned mission acceptability over geometry-only.")
    lines.append("- Obs-only provides limited gain, suggesting objective > compact semantic observation.")
    lines.append("- Static/receding risk-cost A* dominates the static full-map regime.")
    lines.append("")
    lines.append("Dynamic regime:")
    lines.append("- Static-trained RL remains competitive in dynamic-budget threshold satisfaction but is less reliable.")
    lines.append("- Dynamic-trained reward-only RL without dynamic observation improves threshold behavior slightly but hurts success.")
    lines.append("- M11C dynamic-observation RL improves dynamic-budget satisfaction significantly versus planner, but planner dominates success and tail-risk control.")
    lines.append("")
    lines.append("Core interpretation:")
    lines.append("- Mission acceptability is a distinct metric from goal-reaching success.")
    lines.append("- Semantic risk objectives improve learned UAV policies.")
    lines.append("- Classical risk-cost replanning is strongest in known/static regimes.")
    lines.append("- Dynamic risk exposes a trade-off between success reliability, threshold-budget satisfaction, and tail-risk robustness.")
    lines.append("")
    lines.append("Next recommended step:")
    lines.append("- Stop synthetic variant exploration and move to result consolidation + real GIS AOI validation + multi-seed final learned policies.")
    lines.append("")

    path = FINAL_ROOT / "RUSCUAV_v0_synthetic_summary.txt"
    path.write_text("\n".join(lines))

    print("Saved text summary:", path)

    return path

import os

import sys

import importlib

import os

import json

import math

import warnings

from pathlib import Path

from datetime import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import geopandas as gpd

import osmnx as ox

from shapely.geometry import box, Point, LineString, Polygon, MultiPolygon, MultiLineString

from shapely.ops import transform as shapely_transform

from shapely.errors import GEOSException

import pyproj

from affine import Affine

import rasterio

from rasterio.features import rasterize

from rasterio.enums import Resampling

from rasterio.warp import reproject

from rasterio.transform import from_origin

REAL_AOI_ID = "warsaw_centrum_osm_1km_v1"

CENTER_LAT = 52.2297

CENTER_LON = 21.0122

AOI_SIZE_M = 1000.0

RESOLUTION_M = 1.0

DEM_TIF_PATH = None

DSM_TIF_PATH = None

def utm_epsg_from_latlon(lat, lon):
    zone = int((lon + 180.0) // 6.0) + 1
    if lat >= 0:
        return 32600 + zone
    return 32700 + zone

def make_aoi_crs_and_bounds(center_lat, center_lon, size_m):
    epsg = utm_epsg_from_latlon(center_lat, center_lon)
    utm_crs = f"EPSG:{epsg}"

    to_utm = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    to_wgs = pyproj.Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True)

    cx, cy = to_utm.transform(center_lon, center_lat)
    half = float(size_m) / 2.0

    minx = cx - half
    maxx = cx + half
    miny = cy - half
    maxy = cy + half

    # WGS bbox for OSM query.
    corners_utm = [
        (minx, miny),
        (minx, maxy),
        (maxx, miny),
        (maxx, maxy),
    ]
    corners_wgs = [to_wgs.transform(x, y) for x, y in corners_utm]
    lons = [p[0] for p in corners_wgs]
    lats = [p[1] for p in corners_wgs]

    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    return {
        "utm_crs": utm_crs,
        "epsg": epsg,
        "metric_bounds": (float(minx), float(miny), float(maxx), float(maxy)),
        "wgs_bbox": {
            "north": float(north),
            "south": float(south),
            "east": float(east),
            "west": float(west),
        },
        "transformers": {
            "to_utm": to_utm,
            "to_wgs": to_wgs,
        }
    }

def ox_features_from_bbox_compat(north, south, east, west, tags):
    """
    Compatible wrapper for OSMnx 1.x / 2.x.
    OSMnx 2.x uses bbox=(west, south, east, north), tags=...
    Older versions used north, south, east, west, tags.
    """
    try:
        return ox.features_from_bbox(
            bbox=(west, south, east, north),
            tags=tags,
        )
    except TypeError:
        return ox.features_from_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            tags=tags,
        )

def fetch_osm_features(tags, aoi_geo, label):
    bbox_wgs = aoi_geo["wgs_bbox"]

    try:
        gdf = ox_features_from_bbox_compat(
            north=bbox_wgs["north"],
            south=bbox_wgs["south"],
            east=bbox_wgs["east"],
            west=bbox_wgs["west"],
            tags=tags,
        )

        if gdf is None or len(gdf) == 0:
            print(f"[OSM] {label}: empty")
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")

        print(f"[OSM] {label}: {len(gdf)} raw features")
        return gdf

    except Exception as e:
        print(f"[OSM] {label}: FAILED -> {repr(e)}")
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

def project_clip_to_local(gdf, aoi_geo, size_m):
    """
    Project WGS/OSM geometry to UTM, clip to AOI, then convert to local RUSCUAV coordinates:
        local_x = utm_x - minx
        local_y = maxy - utm_y
    """
    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="LOCAL_RUSCUAV")

    minx, miny, maxx, maxy = aoi_geo["metric_bounds"]
    utm_crs = aoi_geo["utm_crs"]

    try:
        gdf_utm = gdf.to_crs(utm_crs)
    except Exception:
        gdf_utm = gdf.copy()
        if gdf_utm.crs is None:
            gdf_utm = gdf_utm.set_crs("EPSG:4326")
        gdf_utm = gdf_utm.to_crs(utm_crs)

    aoi_poly_utm = box(minx, miny, maxx, maxy)
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_poly_utm], crs=utm_crs)

    try:
        clipped = gpd.clip(gdf_utm, aoi_gdf)
    except Exception:
        clipped = gdf_utm[gdf_utm.intersects(aoi_poly_utm)].copy()
        clipped["geometry"] = clipped.geometry.intersection(aoi_poly_utm)

    clipped = clipped[~clipped.geometry.is_empty & clipped.geometry.notnull()].copy()

    if len(clipped) == 0:
        return gpd.GeoDataFrame(geometry=[], crs="LOCAL_RUSCUAV")

    def to_local_geom(geom):
        def f(x, y, z=None):
            lx = x - minx
            ly = maxy - y
            if z is None:
                return (lx, ly)
            return (lx, ly, z)
        try:
            return shapely_transform(f, geom)
        except Exception:
            return geom

    local = clipped.copy()
    local["geometry"] = local.geometry.apply(to_local_geom)
    local = local[~local.geometry.is_empty & local.geometry.notnull()].copy()
    local = local.set_crs(None, allow_override=True)

    # Final clipping in local coordinates.
    local_box = box(0.0, 0.0, float(size_m), float(size_m))
    try:
        local["geometry"] = local.geometry.intersection(local_box)
        local = local[~local.geometry.is_empty & local.geometry.notnull()].copy()
    except Exception:
        pass

    return local

def fetch_real_aoi_osm_layers(aoi_geo):
    # Buildings
    tags_buildings = {
        "building": True,
    }

    # Roads and paths
    tags_highways = {
        "highway": True,
    }

    # Green / natural areas
    tags_green = {
        "leisure": ["park", "garden", "recreation_ground", "playground"],
        "landuse": ["grass", "forest", "meadow", "recreation_ground", "village_green", "cemetery"],
        "natural": ["wood", "grassland", "scrub", "heath"],
    }

    # Sensitive facilities / no-fly-like proxy
    tags_sensitive = {
        "amenity": [
            "school", "kindergarten", "hospital", "clinic", "doctors",
            "university", "college", "social_facility", "nursing_home",
        ],
        "healthcare": True,
        "leisure": ["playground"],
    }

    layers_raw = {
        "buildings": fetch_osm_features(tags_buildings, aoi_geo, "buildings"),
        "highways": fetch_osm_features(tags_highways, aoi_geo, "highways"),
        "green": fetch_osm_features(tags_green, aoi_geo, "green"),
        "sensitive": fetch_osm_features(tags_sensitive, aoi_geo, "sensitive"),
    }

    layers_local = {}
    for k, gdf in layers_raw.items():
        layers_local[k] = project_clip_to_local(gdf, aoi_geo, AOI_SIZE_M)
        print(f"[LOCAL] {k}: {len(layers_local[k])} features")

    return layers_local

from rasterio.features import rasterize

from rasterio.enums import MergeAlg

def raster_shape(size_m=AOI_SIZE_M, resolution_m=RESOLUTION_M):
    n = int(round(float(size_m) / float(resolution_m)))
    return (n, n)

def local_affine_transform(resolution_m=RESOLUTION_M):
    # Pixel column -> x, row -> y. y increases downward, matching RUSCUAV row indexing.
    return Affine(float(resolution_m), 0.0, 0.0, 0.0, float(resolution_m), 0.0)

def valid_geoms(gdf):
    if gdf is None or len(gdf) == 0:
        return []

    geoms = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:
            pass
        if geom is not None and not geom.is_empty:
            geoms.append(geom)

    return geoms

def rasterize_mask(gdf_or_geoms, out_shape=OUT_SHAPE, transform=LOCAL_TRANSFORM, all_touched=True):
    if isinstance(gdf_or_geoms, gpd.GeoDataFrame) or isinstance(gdf_or_geoms, gpd.GeoSeries):
        geoms = valid_geoms(gdf_or_geoms)
    else:
        geoms = [g for g in gdf_or_geoms if g is not None and not g.is_empty]

    if len(geoms) == 0:
        return np.zeros(out_shape, dtype=np.float32)

    arr = rasterize(
        [(g, 1.0) for g in geoms],
        out_shape=out_shape,
        fill=0.0,
        transform=transform,
        all_touched=all_touched,
        dtype="float32",
    )

    return np.clip(arr, 0.0, 1.0).astype(np.float32)

def rasterize_values(gdf, value_col, out_shape=OUT_SHAPE, transform=LOCAL_TRANSFORM, all_touched=True, default_value=0.0):
    if gdf is None or len(gdf) == 0 or value_col not in gdf.columns:
        return np.zeros(out_shape, dtype=np.float32)

    shapes = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        try:
            if not geom.is_valid:
                geom = geom.buffer(0)
        except Exception:
            pass
        if geom is None or geom.is_empty:
            continue

        val = row.get(value_col, default_value)
        try:
            val = float(val)
        except Exception:
            val = default_value

        shapes.append((geom, val))

    if len(shapes) == 0:
        return np.zeros(out_shape, dtype=np.float32)

    arr = rasterize(
        shapes,
        out_shape=out_shape,
        fill=0.0,
        transform=transform,
        all_touched=all_touched,
        dtype="float32",
        merge_alg=MergeAlg.replace,
    )

    return np.maximum(arr, 0.0).astype(np.float32)

def parse_osm_numeric(value):
    if value is None:
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip().lower()

    if s in ["", "nan", "none", "null"]:
        return np.nan

    # Remove common units and separators.
    s = s.replace("meters", "").replace("metres", "").replace("meter", "").replace("metre", "")
    s = s.replace("m", "")
    s = s.replace(",", ".")
    s = s.strip()

    # If tag has multiple values, take first numeric-ish token.
    for token in s.replace(";", " ").split():
        try:
            return float(token)
        except Exception:
            continue

    try:
        return float(s)
    except Exception:
        return np.nan

def estimate_building_height_row(row):
    # Prefer height tag.
    for col in ["height", "building:height"]:
        if col in row:
            h = parse_osm_numeric(row.get(col))
            if np.isfinite(h) and h > 0:
                return float(np.clip(h, 3.0, 150.0))

    # Then levels.
    for col in ["building:levels", "levels"]:
        if col in row:
            levels = parse_osm_numeric(row.get(col))
            if np.isfinite(levels) and levels > 0:
                return float(np.clip(levels * 3.2, 3.0, 150.0))

    # Fallback by building type.
    b = str(row.get("building", "")).lower()
    if b in ["apartments", "residential", "house", "detached", "terrace"]:
        return 12.0
    if b in ["commercial", "retail", "office"]:
        return 18.0
    if b in ["industrial", "warehouse"]:
        return 10.0
    if b in ["church", "cathedral"]:
        return 25.0
    if b in ["school", "hospital", "university"]:
        return 16.0

    return 12.0

def prepare_buildings_layer(buildings_gdf):
    if buildings_gdf is None or len(buildings_gdf) == 0:
        gdf = gpd.GeoDataFrame(geometry=[], crs=None)
        gdf["height_m"] = []
        return gdf

    gdf = buildings_gdf.copy()
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    if len(gdf) == 0:
        gdf["height_m"] = []
        return gdf

    gdf["height_m"] = gdf.apply(estimate_building_height_row, axis=1).astype(float)
    return gdf

ROAD_WIDTH_BY_HIGHWAY = {
    "motorway": 28.0,
    "trunk": 24.0,
    "primary": 18.0,
    "secondary": 15.0,
    "tertiary": 12.0,
    "unclassified": 9.0,
    "residential": 8.0,
    "living_street": 6.0,
    "service": 5.0,
    "road": 7.0,
    "track": 4.0,
}

FOOTWAY_WIDTH_M = 3.0

SIDEWALK_PROXY_WIDTH_M = 4.0

def highway_value(row):
    h = row.get("highway", "")
    if isinstance(h, list):
        h = h[0] if h else ""
    return str(h).lower()

def is_drive_road(row):
    h = highway_value(row)
    if h in ["footway", "path", "pedestrian", "steps", "cycleway", "bridleway", "corridor", "platform"]:
        return False
    return h in ROAD_WIDTH_BY_HIGHWAY

def is_pedestrian_way(row):
    h = highway_value(row)
    if h in ["footway", "path", "pedestrian", "steps", "crossing"]:
        return True

    footway = str(row.get("footway", "")).lower()
    if footway in ["sidewalk", "crossing"]:
        return True

    return False

def road_width_row(row):
    h = highway_value(row)
    return float(ROAD_WIDTH_BY_HIGHWAY.get(h, 7.0))

def buffer_geometry_safe(geom, radius):
    if geom is None or geom.is_empty:
        return None

    try:
        if geom.geom_type in ["Polygon", "MultiPolygon"]:
            return geom
        return geom.buffer(float(radius), cap_style=2, join_style=2)
    except Exception:
        try:
            return geom.buffer(float(radius))
        except Exception:
            return None

def prepare_roads_and_sidewalks(highways_gdf):
    if highways_gdf is None or len(highways_gdf) == 0:
        return (
            gpd.GeoDataFrame(geometry=[], crs=None),
            gpd.GeoDataFrame(geometry=[], crs=None),
            gpd.GeoDataFrame(geometry=[], crs=None),
        )

    gdf = highways_gdf.copy()
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    drive_rows = []
    road_buffers = []
    sidewalk_proxy_buffers = []

    footway_buffers = []

    for _, row in gdf.iterrows():
        geom = row.geometry

        if is_drive_road(row):
            w = road_width_row(row)
            road_poly = buffer_geometry_safe(geom, w / 2.0)
            outer_poly = buffer_geometry_safe(geom, w / 2.0 + SIDEWALK_PROXY_WIDTH_M)

            if road_poly is not None and not road_poly.is_empty:
                drive_rows.append(row)
                road_buffers.append(road_poly)

            if outer_poly is not None and road_poly is not None and not outer_poly.is_empty:
                try:
                    sidewalk_ring = outer_poly.difference(road_poly)
                except Exception:
                    sidewalk_ring = outer_poly
                if sidewalk_ring is not None and not sidewalk_ring.is_empty:
                    sidewalk_proxy_buffers.append(sidewalk_ring)

        if is_pedestrian_way(row):
            foot_poly = buffer_geometry_safe(geom, FOOTWAY_WIDTH_M / 2.0)
            if foot_poly is not None and not foot_poly.is_empty:
                footway_buffers.append(foot_poly)

    roads_gdf = gpd.GeoDataFrame(geometry=road_buffers, crs=None)
    sidewalk_proxy_gdf = gpd.GeoDataFrame(geometry=sidewalk_proxy_buffers, crs=None)
    footway_gdf = gpd.GeoDataFrame(geometry=footway_buffers, crs=None)

    return roads_gdf, sidewalk_proxy_gdf, footway_gdf

def prepare_green_layer(green_gdf):
    if green_gdf is None or len(green_gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=None)

    gdf = green_gdf.copy()
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    # Keep polygons only.
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    return gdf

def is_canopy_green_row(row):
    landuse = str(row.get("landuse", "")).lower()
    natural = str(row.get("natural", "")).lower()
    leisure = str(row.get("leisure", "")).lower()

    if landuse in ["forest"]:
        return True
    if natural in ["wood", "scrub"]:
        return True
    if leisure in ["park", "garden"]:
        return True

    return False

def canopy_height_row(row):
    landuse = str(row.get("landuse", "")).lower()
    natural = str(row.get("natural", "")).lower()
    leisure = str(row.get("leisure", "")).lower()

    if landuse == "forest" or natural == "wood":
        return 18.0
    if natural == "scrub":
        return 6.0
    if leisure in ["park", "garden"]:
        return 12.0

    return 10.0

def prepare_sensitive_layer(sens_gdf):
    if sens_gdf is None or len(sens_gdf) == 0:
        return gpd.GeoDataFrame(geometry=[], crs=None)

    gdf = sens_gdf.copy()
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty].copy()

    buffers = []

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        amenity = str(row.get("amenity", "")).lower()
        healthcare = str(row.get("healthcare", "")).lower()
        leisure = str(row.get("leisure", "")).lower()

        if amenity in ["hospital", "clinic", "doctors"] or healthcare not in ["", "nan", "none"]:
            buf = 55.0
        elif amenity in ["school", "kindergarten", "university", "college"]:
            buf = 45.0
        elif leisure in ["playground"]:
            buf = 35.0
        else:
            buf = 35.0

        try:
            if geom.geom_type in ["Point", "MultiPoint", "LineString", "MultiLineString"]:
                buffers.append(geom.buffer(buf))
            else:
                buffers.append(geom.buffer(10.0))
        except Exception:
            pass

    return gpd.GeoDataFrame(geometry=buffers, crs=None)

def load_raster_to_aoi_grid(tif_path, aoi_geo, out_shape=OUT_SHAPE, resolution_m=RESOLUTION_M, fill_value=20.0):
    if tif_path is None:
        return None

    tif_path = Path(tif_path)

    if not tif_path.exists():
        print("[Raster] Missing:", tif_path)
        return None

    minx, miny, maxx, maxy = aoi_geo["metric_bounds"]
    dst_crs = aoi_geo["utm_crs"]
    dst_transform = from_origin(minx, maxy, float(resolution_m), float(resolution_m))

    dst = np.full(out_shape, np.nan, dtype=np.float32)

    with rasterio.open(tif_path) as src:
        src_nodata = src.nodata

        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

    if np.all(~np.isfinite(dst)):
        print("[Raster] Reprojected raster has no finite values:", tif_path)
        return None

    med = float(np.nanmedian(dst))
    dst = np.where(np.isfinite(dst), dst, med).astype(np.float32)

    # Shift terrain so local min is not negative; RUSCUAV supports any z but this keeps it clean.
    return dst.astype(np.float32)

def make_terrain_fallback(out_shape=OUT_SHAPE):
    """
    Controlled fallback if real DEM is not available.
    This keeps terrain smooth but does NOT claim to be real DEM.
    """
    h, w = out_shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    terrain = 18.0 + 1.5 * np.sin(xx / 180.0) + 1.0 * np.cos(yy / 220.0)
    terrain = terrain.astype(np.float32)

    return terrain

import inspect

import numpy as np

def _safe_set_attr(obj, key, value, verbose=False):
    """
    Set attribute unless it is a read-only @property or otherwise blocked.
    """
    cls_attr = getattr(type(obj), key, None)

    if isinstance(cls_attr, property) and cls_attr.fset is None:
        if verbose:
            print(f"[safe_set_attr] skip read-only property: {key}")
        return False

    try:
        setattr(obj, key, value)
        return True
    except Exception as e:
        if verbose:
            print(f"[safe_set_attr] failed for {key}: {repr(e)}")
        return False

def _construct_urban_world_layers_compatible(attrs, verbose=True):
    """
    Construct UrbanWorldLayers robustly across different class definitions.

    Strategy:
    1. Inspect UrbanWorldLayers.__init__ signature.
    2. Pass only accepted constructor arguments.
    3. Add remaining non-read-only attributes after construction.
    4. Never force-set read-only properties such as height/width/shape.
    """
    # These are usually derived properties and should not be forced into constructor.
    derived_keys = {"height", "width", "shape"}

    clean_attrs = {
        k: v for k, v in attrs.items()
        if k not in derived_keys
    }

    # First try: constructor with accepted keyword args only.
    try:
        sig = inspect.signature(UrbanWorldLayers.__init__)
        params = sig.parameters

        accepts_var_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )

        if accepts_var_kwargs:
            ctor_kwargs = clean_attrs.copy()
        else:
            accepted = {
                name for name, p in params.items()
                if name != "self"
                and p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            ctor_kwargs = {
                k: v for k, v in clean_attrs.items()
                if k in accepted
            }

        obj = UrbanWorldLayers(**ctor_kwargs)

        if verbose:
            print("[UrbanWorldLayers] constructed with filtered keyword args:")
            print("  ctor keys:", sorted(list(ctor_kwargs.keys())))

        # Add any missing compatible attributes after construction.
        for k, v in clean_attrs.items():
            if not hasattr(obj, k):
                _safe_set_attr(obj, k, v, verbose=verbose)

        return obj

    except Exception as e:
        if verbose:
            print("[UrbanWorldLayers] filtered constructor failed, using __new__ fallback:")
            print(" ", repr(e))

    # Fallback: raw object + safe setattr.
    obj = UrbanWorldLayers.__new__(UrbanWorldLayers)

    for k, v in clean_attrs.items():
        _safe_set_attr(obj, k, v, verbose=verbose)

    return obj

def make_urban_world_layers_object(
    aoi_id,
    crs,
    bbox,
    resolution_m,
    terrain_z,
    surface_z,
    building_mask,
    building_height,
    road_mask,
    sidewalk_mask,
    green_mask,
    canopy_mask,
    canopy_height,
    canopy_density,
    sensitive_mask,
):
    terrain_z = terrain_z.astype(np.float32)
    surface_z = surface_z.astype(np.float32)

    building_mask = building_mask.astype(np.float32)
    building_height = building_height.astype(np.float32)

    road_mask = road_mask.astype(np.float32)
    sidewalk_mask = sidewalk_mask.astype(np.float32)
    green_mask = green_mask.astype(np.float32)

    canopy_mask = canopy_mask.astype(np.float32)
    canopy_height = canopy_height.astype(np.float32)
    canopy_density = canopy_density.astype(np.float32)

    sensitive_mask = sensitive_mask.astype(np.float32)

    h, w = terrain_z.shape

    attrs = {
        "aoi_id": str(aoi_id),
        "crs": str(crs),
        "bbox": tuple(float(x) for x in bbox),
        "resolution_m": float(resolution_m),

        "terrain_z": terrain_z,
        "surface_z": surface_z,

        "building_mask": building_mask,
        "building_height": building_height,
        "building_roof_z": (terrain_z + building_height).astype(np.float32),

        "road_mask": road_mask,
        "sidewalk_mask": sidewalk_mask,
        "green_mask": green_mask,

        "canopy_mask": canopy_mask,
        "canopy_height": canopy_height,
        "canopy_top_z": (terrain_z + canopy_height).astype(np.float32),
        "canopy_density": canopy_density,

        "sensitive_mask": sensitive_mask,

        # Do NOT force these into constructor/setattr if they are read-only properties.
        # They are included here only for fallback compatibility and safely skipped if needed.
        "height": int(h),
        "width": int(w),
        "shape": (int(h), int(w)),
    }

    obj = _construct_urban_world_layers_compatible(attrs, verbose=True)

    # Diagnostics.
    required = [
        "terrain_z",
        "surface_z",
        "building_mask",
        "building_height",
        "road_mask",
        "sidewalk_mask",
        "green_mask",
        "canopy_height",
        "canopy_density",
        "sensitive_mask",
    ]

    missing = [k for k in required if not hasattr(obj, k)]
    if missing:
        raise RuntimeError(
            f"UrbanWorldLayers object is missing required attributes: {missing}"
        )

    print("=" * 90)
    print("UrbanWorldLayers real AOI object created")
    print("=" * 90)
    print("aoi_id:", getattr(obj, "aoi_id", None))
    print("terrain_z shape:", obj.terrain_z.shape)

    try:
        print("height:", obj.height)
    except Exception:
        print("height: unavailable")

    try:
        print("width:", obj.width)
    except Exception:
        print("width: unavailable")

    try:
        print("shape:", obj.shape)
    except Exception:
        print("shape: unavailable")

    print("=" * 90)

    return obj

def save_real_world_robust(world, npz_path, meta_path=None):
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        world.save_npz(npz_path)
        print("[save] world.save_npz:", npz_path)
    except Exception as e:
        print("[save] world.save_npz failed, using manual np.savez_compressed:", repr(e))

        np.savez_compressed(
            npz_path,
            aoi_id=world.aoi_id,
            crs=world.crs,
            bbox=np.array(world.bbox, dtype=np.float32),
            resolution_m=np.float32(world.resolution_m),

            terrain_z=world.terrain_z,
            surface_z=world.surface_z,

            building_mask=world.building_mask,
            building_height=world.building_height,
            building_roof_z=world.building_roof_z,

            road_mask=world.road_mask,
            sidewalk_mask=world.sidewalk_mask,
            green_mask=world.green_mask,

            canopy_mask=world.canopy_mask,
            canopy_height=world.canopy_height,
            canopy_top_z=world.canopy_top_z,
            canopy_density=world.canopy_density,

            sensitive_mask=world.sensitive_mask,
        )
        print("[save] manual npz:", npz_path)

    if meta_path is not None:
        meta = {
            "aoi_id": world.aoi_id,
            "created_at": datetime.now().isoformat(),
            "center_lat": CENTER_LAT,
            "center_lon": CENTER_LON,
            "aoi_size_m": AOI_SIZE_M,
            "resolution_m": RESOLUTION_M,
            "shape": list(world.shape),
            "crs": world.crs,
            "source": {
                "osm": True,
                "dem_tif_path": str(DEM_TIF_PATH) if DEM_TIF_PATH is not None else None,
                "dsm_tif_path": str(DSM_TIF_PATH) if DSM_TIF_PATH is not None else None,
                "terrain_fallback_used": DEM_TIF_PATH is None,
            },
            "coverage": {
                "building": float(world.building_mask.mean()),
                "road": float(world.road_mask.mean()),
                "sidewalk": float(world.sidewalk_mask.mean()),
                "green": float(world.green_mask.mean()),
                "canopy": float(world.canopy_mask.mean()),
                "sensitive": float(world.sensitive_mask.mean()),
            },
            "metric_bounds_utm": list(AOI_GEO["metric_bounds"]),
            "wgs_bbox": AOI_GEO["wgs_bbox"],
            "utm_crs": AOI_GEO["utm_crs"],
            "notes": [
                "OSM-derived vectors are real GIS data.",
                "Sidewalk mask includes explicit footways plus road-buffer sidewalk proxy.",
                "Canopy is proxy from OSM green/wood/park geometries unless DSM/canopy data are supplied.",
                "If DEM_TIF_PATH is None, terrain_z is a controlled fallback and should not be interpreted as real DEM.",
            ],
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print("[save] metadata:", meta_path)

    return str(npz_path)

def print_real_world_layer_summary(world):
    print("=" * 100)
    print("REAL GIS URBAN WORLD LAYER SUMMARY")
    print("=" * 100)
    print("AOI:", world.aoi_id)
    print("shape:", world.shape)
    print("resolution_m:", world.resolution_m)
    print("-" * 100)

    layers = [
        ("terrain_z", world.terrain_z),
        ("surface_z", world.surface_z),
        ("building_mask", world.building_mask),
        ("building_height", world.building_height),
        ("road_mask", world.road_mask),
        ("sidewalk_mask", world.sidewalk_mask),
        ("green_mask", world.green_mask),
        ("canopy_mask", world.canopy_mask),
        ("canopy_height", world.canopy_height),
        ("canopy_density", world.canopy_density),
        ("sensitive_mask", world.sensitive_mask),
    ]

    for name, arr in layers:
        arr = np.asarray(arr)
        active = int(np.sum(arr > 0))
        coverage = active / arr.size
        print(
            f"{name:18s} | active={active:8d} | coverage={coverage:8.3%} | "
            f"min={float(np.nanmin(arr)):8.3f} | mean={float(np.nanmean(arr)):8.3f} | max={float(np.nanmax(arr)):8.3f}"
        )

    print("=" * 100)

def m12_smoke_test_risk_model(world):
    risk_model = UrbanRiskModel(
        world=world,
        profile="balanced",
        hard_sensitive_zones=True,
        include_global_altitude_band_cost=True,
    )

    for agl in [20.0, 40.0, 80.0]:
        risk = risk_model.risk_map_at_agl(agl)
        print(
            f"AGL={agl:5.1f} | risk min={float(np.nanmin(risk)):.3f} "
            f"mean={float(np.nanmean(risk)):.3f} p95={float(np.nanpercentile(risk,95)):.3f} "
            f"max={float(np.nanmax(risk)):.3f}"
        )

    print("Risk model smoke test: OK")

def plot_real_world_layers_2d(world, title=None):
    if title is None:
        title = f"Real GIS AOI: {world.aoi_id}"

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(world.terrain_z, origin="upper", cmap="terrain", alpha=0.85)

    def overlay(mask, color_cmap, alpha, label):
        arr = np.ma.masked_where(mask <= 0.5, mask)
        ax.imshow(arr, origin="upper", cmap=color_cmap, alpha=alpha)

    overlay(world.green_mask, "Greens", 0.30, "green")
    overlay(world.canopy_mask, "Greens", 0.45, "canopy")
    overlay(world.road_mask, "Oranges", 0.50, "roads")
    overlay(world.sidewalk_mask, "Wistia", 0.45, "sidewalks")
    overlay(world.building_mask, "Greys", 0.60, "buildings")
    overlay(world.sensitive_mask, "Reds", 0.55, "sensitive")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    return fig, ax

import numpy as np

from pathlib import Path

import json

from datetime import datetime

def patch_urban_world_layers_for_ruscuav(world, verbose=True):
    """
    Adds derived/compatibility fields expected by UrbanRiskModel and M4 env.

    Required by current RUSCUAV stack:
        terrain_z
        surface_z
        height_above_ground
        building_mask
        building_height
        building_roof_z
        road_mask
        sidewalk_mask
        green_mask
        canopy_mask
        canopy_height
        canopy_top_z
        canopy_density
        sensitive_mask
    """

    terrain_z = np.asarray(world.terrain_z, dtype=np.float32)

    # ------------------------------------------------------------------
    # Ensure surface_z
    # ------------------------------------------------------------------
    if not hasattr(world, "surface_z") or world.surface_z is None:
        if verbose:
            print("[patch] surface_z missing -> reconstructing from terrain/building/canopy")

        building_height = np.asarray(getattr(world, "building_height", np.zeros_like(terrain_z)), dtype=np.float32)
        canopy_height = np.asarray(getattr(world, "canopy_height", np.zeros_like(terrain_z)), dtype=np.float32)

        surface_z = np.maximum.reduce([
            terrain_z,
            terrain_z + building_height,
            terrain_z + canopy_height,
        ]).astype(np.float32)

        setattr(world, "surface_z", surface_z)

    else:
        world.surface_z = np.asarray(world.surface_z, dtype=np.float32)

    # ------------------------------------------------------------------
    # Ensure basic masks/heights exist
    # ------------------------------------------------------------------
    defaults = {
        "building_mask": np.zeros_like(terrain_z, dtype=np.float32),
        "building_height": np.zeros_like(terrain_z, dtype=np.float32),
        "road_mask": np.zeros_like(terrain_z, dtype=np.float32),
        "sidewalk_mask": np.zeros_like(terrain_z, dtype=np.float32),
        "green_mask": np.zeros_like(terrain_z, dtype=np.float32),
        "canopy_height": np.zeros_like(terrain_z, dtype=np.float32),
        "canopy_density": np.zeros_like(terrain_z, dtype=np.float32),
        "sensitive_mask": np.zeros_like(terrain_z, dtype=np.float32),
    }

    for name, default in defaults.items():
        if not hasattr(world, name) or getattr(world, name) is None:
            if verbose:
                print(f"[patch] {name} missing -> zeros")
            setattr(world, name, default.copy())
        else:
            setattr(world, name, np.asarray(getattr(world, name), dtype=np.float32))

    # ------------------------------------------------------------------
    # Ensure canopy_mask
    # ------------------------------------------------------------------
    if not hasattr(world, "canopy_mask") or world.canopy_mask is None:
        canopy_mask = (
            (np.asarray(world.canopy_height, dtype=np.float32) > 0.1)
            | (np.asarray(world.canopy_density, dtype=np.float32) > 0.05)
        ).astype(np.float32)

        if verbose:
            print("[patch] canopy_mask missing -> derived from canopy_height/canopy_density")

        setattr(world, "canopy_mask", canopy_mask)

    else:
        world.canopy_mask = np.asarray(world.canopy_mask, dtype=np.float32)

    # ------------------------------------------------------------------
    # Derived fields
    # ------------------------------------------------------------------
    world.building_height = np.asarray(world.building_height, dtype=np.float32) * np.asarray(world.building_mask, dtype=np.float32)
    world.canopy_height = np.asarray(world.canopy_height, dtype=np.float32) * np.asarray(world.canopy_mask, dtype=np.float32)
    world.canopy_density = np.asarray(world.canopy_density, dtype=np.float32) * np.asarray(world.canopy_mask, dtype=np.float32)

    building_roof_z = (terrain_z + world.building_height).astype(np.float32)
    canopy_top_z = (terrain_z + world.canopy_height).astype(np.float32)

    setattr(world, "building_roof_z", building_roof_z)
    setattr(world, "canopy_top_z", canopy_top_z)

    # This is the missing field from your traceback.
    height_above_ground = np.maximum(
        np.asarray(world.surface_z, dtype=np.float32) - terrain_z,
        0.0,
    ).astype(np.float32)

    setattr(world, "height_above_ground", height_above_ground)

    # ------------------------------------------------------------------
    # Normalize masks to 0/1-ish float32
    # ------------------------------------------------------------------
    for name in [
        "building_mask",
        "road_mask",
        "sidewalk_mask",
        "green_mask",
        "canopy_mask",
        "sensitive_mask",
    ]:
        arr = np.asarray(getattr(world, name), dtype=np.float32)
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
        setattr(world, name, arr)

    # ------------------------------------------------------------------
    # Metadata compatibility
    # ------------------------------------------------------------------
    if not hasattr(world, "resolution_m"):
        setattr(world, "resolution_m", 1.0)

    if not hasattr(world, "bbox"):
        setattr(world, "bbox", (0.0, 0.0, float(terrain_z.shape[1]), float(terrain_z.shape[0])))

    if not hasattr(world, "crs"):
        setattr(world, "crs", "LOCAL_RUSCUAV_0_1000")

    if not hasattr(world, "aoi_id"):
        setattr(world, "aoi_id", "real_gis_aoi")

    if verbose:
        print("=" * 90)
        print("UrbanWorldLayers compatibility patch complete")
        print("=" * 90)
        print("aoi_id:", getattr(world, "aoi_id", None))
        print("terrain_z:", world.terrain_z.shape, world.terrain_z.dtype)
        print("surface_z:", world.surface_z.shape, world.surface_z.dtype)
        print("height_above_ground:", world.height_above_ground.shape, world.height_above_ground.dtype)
        print("height_above_ground min/mean/max:",
              float(world.height_above_ground.min()),
              float(world.height_above_ground.mean()),
              float(world.height_above_ground.max()))
        print("building coverage:", float(world.building_mask.mean()))
        print("road coverage:", float(world.road_mask.mean()))
        print("sidewalk coverage:", float(world.sidewalk_mask.mean()))
        print("green coverage:", float(world.green_mask.mean()))
        print("canopy coverage:", float(world.canopy_mask.mean()))
        print("sensitive coverage:", float(world.sensitive_mask.mean()))
        print("=" * 90)

    return world

def save_real_world_manual_npz_v2(world, npz_path, meta_path=None):
    npz_path = Path(npz_path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure compatibility fields exist before saving.
    world = patch_urban_world_layers_for_ruscuav(world, verbose=False)

    np.savez_compressed(
        npz_path,
        aoi_id=str(getattr(world, "aoi_id", "real_aoi")),
        crs=str(getattr(world, "crs", "LOCAL_RUSCUAV")),
        bbox=np.array(getattr(world, "bbox", (0.0, 0.0, 1000.0, 1000.0)), dtype=np.float32),
        resolution_m=np.float32(getattr(world, "resolution_m", 1.0)),

        terrain_z=np.asarray(world.terrain_z, dtype=np.float32),
        surface_z=np.asarray(world.surface_z, dtype=np.float32),
        height_above_ground=np.asarray(world.height_above_ground, dtype=np.float32),

        building_mask=np.asarray(world.building_mask, dtype=np.float32),
        building_height=np.asarray(world.building_height, dtype=np.float32),
        building_roof_z=np.asarray(world.building_roof_z, dtype=np.float32),

        road_mask=np.asarray(world.road_mask, dtype=np.float32),
        sidewalk_mask=np.asarray(world.sidewalk_mask, dtype=np.float32),
        green_mask=np.asarray(world.green_mask, dtype=np.float32),

        canopy_mask=np.asarray(world.canopy_mask, dtype=np.float32),
        canopy_height=np.asarray(world.canopy_height, dtype=np.float32),
        canopy_top_z=np.asarray(world.canopy_top_z, dtype=np.float32),
        canopy_density=np.asarray(world.canopy_density, dtype=np.float32),

        sensitive_mask=np.asarray(world.sensitive_mask, dtype=np.float32),
    )

    print("[save] manual npz v2:", npz_path)

    if meta_path is not None:
        meta = {
            "aoi_id": str(getattr(world, "aoi_id", "real_aoi")),
            "created_at": datetime.now().isoformat(),
            "shape": list(np.asarray(world.terrain_z).shape),
            "resolution_m": float(getattr(world, "resolution_m", 1.0)),
            "crs": str(getattr(world, "crs", "LOCAL_RUSCUAV")),
            "coverage": {
                "building": float(np.asarray(world.building_mask).mean()),
                "road": float(np.asarray(world.road_mask).mean()),
                "sidewalk": float(np.asarray(world.sidewalk_mask).mean()),
                "green": float(np.asarray(world.green_mask).mean()),
                "canopy": float(np.asarray(world.canopy_mask).mean()),
                "sensitive": float(np.asarray(world.sensitive_mask).mean()),
            },
            "height_above_ground": {
                "min": float(np.asarray(world.height_above_ground).min()),
                "mean": float(np.asarray(world.height_above_ground).mean()),
                "max": float(np.asarray(world.height_above_ground).max()),
            },
            "notes": [
                "Manual NPZ v2 save used for compatibility with current UrbanWorldLayers class.",
                "height_above_ground included.",
                "height/width/shape are treated as derived properties.",
            ],
        }

        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        print("[save] metadata:", meta_path)

    return str(npz_path)

def smoke_test_real_gis_world_in_m4(world, seed=123, n_steps=20):
    env = make_ruscuav_env_m4_mode(
        world=world,
        stage="C",
        mode="strict_eval",
        risk_profile="balanced",
        n_obstacles=0,
        seed=seed,
    )

    obs, info = env.reset(seed=seed)

    print("=" * 100)
    print("REAL GIS AOI M4 SMOKE TEST")
    print("=" * 100)
    print("obs.shape:", obs.shape)
    print("observation_space:", env.observation_space)
    print("reset info:", info)

    for i in range(int(n_steps)):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={i+1:03d} | r={reward:9.2f} | "
            f"term={info.get('termination')} | "
            f"risk={info.get('urban_risk', np.nan):8.3f} | "
            f"road_t={info.get('urban_road_time', 0.0):6.1f} | "
            f"sidewalk_t={info.get('urban_sidewalk_time', 0.0):6.1f} | "
            f"building_t={info.get('urban_building_time', 0.0):6.1f} | "
            f"acceptable={info.get('mission_acceptable', False)}"
        )

        if terminated or truncated:
            break

    env.close()

    print("=" * 100)
    return obs, info

import numpy as np

import pandas as pd

from pathlib import Path

import json

def m12_layer_diagnostics(world, save_dir=None):
    masks = {
        "building": np.asarray(world.building_mask, dtype=np.float32),
        "road": np.asarray(world.road_mask, dtype=np.float32),
        "sidewalk": np.asarray(world.sidewalk_mask, dtype=np.float32),
        "green": np.asarray(world.green_mask, dtype=np.float32),
        "canopy": np.asarray(world.canopy_mask, dtype=np.float32) if hasattr(world, "canopy_mask") else (np.asarray(world.canopy_height) > 0).astype(np.float32),
        "sensitive": np.asarray(world.sensitive_mask, dtype=np.float32),
    }

    rows = []

    n = float(next(iter(masks.values())).size)

    for name, mask in masks.items():
        active = mask > 0.5
        rows.append({
            "layer": name,
            "active_pixels": int(active.sum()),
            "coverage": float(active.sum() / n),
            "mean": float(mask.mean()),
            "max": float(mask.max()),
        })

    coverage_df = pd.DataFrame(rows)

    overlap_rows = []
    names = list(masks.keys())

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ma = masks[a] > 0.5
            mb = masks[b] > 0.5
            inter = ma & mb
            union = ma | mb

            overlap_rows.append({
                "layer_a": a,
                "layer_b": b,
                "intersection_pixels": int(inter.sum()),
                "intersection_coverage": float(inter.sum() / n),
                "jaccard": float(inter.sum() / max(1, union.sum())),
                "a_fraction_overlapping_b": float(inter.sum() / max(1, ma.sum())),
                "b_fraction_overlapping_a": float(inter.sum() / max(1, mb.sum())),
            })

    overlap_df = pd.DataFrame(overlap_rows)

    print("=" * 100)
    print("M12 REAL AOI LAYER COVERAGE")
    print("=" * 100)
    display(coverage_df)

    print("=" * 100)
    print("M12 REAL AOI LAYER OVERLAPS")
    print("=" * 100)
    display(overlap_df.sort_values("intersection_coverage", ascending=False))

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        coverage_path = save_dir / "m12_real_layer_coverage.csv"
        overlap_path = save_dir / "m12_real_layer_overlaps.csv"

        coverage_df.to_csv(coverage_path, index=False)
        overlap_df.to_csv(overlap_path, index=False)

        print("Saved:", coverage_path)
        print("Saved:", overlap_path)

    return coverage_df, overlap_df

def m12_sidewalk_proxy_diagnostics():
    if "footway_mask" not in globals() or "sidewalk_proxy_mask" not in globals():
        print("footway_mask or sidewalk_proxy_mask not found in runtime.")
        return None

    fw = np.asarray(footway_mask, dtype=np.float32)
    proxy = np.asarray(sidewalk_proxy_mask, dtype=np.float32)
    final = np.asarray(real_world.sidewalk_mask, dtype=np.float32)
    road = np.asarray(real_world.road_mask, dtype=np.float32)

    n = fw.size

    rows = [
        {
            "layer": "explicit_footway_mask",
            "coverage": float((fw > 0.5).sum() / n),
            "active_pixels": int((fw > 0.5).sum()),
        },
        {
            "layer": "sidewalk_proxy_mask",
            "coverage": float((proxy > 0.5).sum() / n),
            "active_pixels": int((proxy > 0.5).sum()),
        },
        {
            "layer": "final_sidewalk_mask",
            "coverage": float((final > 0.5).sum() / n),
            "active_pixels": int((final > 0.5).sum()),
        },
        {
            "layer": "road_mask",
            "coverage": float((road > 0.5).sum() / n),
            "active_pixels": int((road > 0.5).sum()),
        },
    ]

    df = pd.DataFrame(rows)

    print("=" * 100)
    print("M12 SIDEWALK PROXY DIAGNOSTICS")
    print("=" * 100)
    display(df)

    path = m12_diag_dir / "m12_sidewalk_proxy_diagnostics.csv"
    df.to_csv(path, index=False)
    print("Saved:", path)

    return df

def m12_posthoc_acceptability(
    df,
    road_budget_s,
    sidewalk_budget_s,
    building_budget_s,
    sensitive_budget_s=0.0,
    cargo_min=0.80,
):
    return (
        (df["is_success"].astype(bool) == True)
        & (df["cargo_health"].astype(float) >= float(cargo_min))
        & (df["road_time"].astype(float) <= float(road_budget_s))
        & (df["sidewalk_time"].astype(float) <= float(sidewalk_budget_s))
        & (df["building_time"].astype(float) <= float(building_budget_s))
        & (df["sensitive_time"].astype(float) <= float(sensitive_budget_s))
        & (df["urban_hard_violation_steps"].astype(float) == 0.0)
    )

def m12_budget_sensitivity(df, label="m12_no_terminal"):
    budget_sets = [
        # synthetic nominal
        {"name": "synthetic_nominal", "road": 20, "sidewalk": 5, "building": 15},

        # current terminal-like
        {"name": "terminal_like", "road": 30, "sidewalk": 10, "building": 20},

        # moderate dense-city
        {"name": "dense_city_moderate", "road": 45, "sidewalk": 20, "building": 30},

        # relaxed dense-city
        {"name": "dense_city_relaxed", "road": 60, "sidewalk": 30, "building": 45},

        # very relaxed diagnostic
        {"name": "diagnostic_relaxed", "road": 90, "sidewalk": 45, "building": 60},
    ]

    rows = []

    for b in budget_sets:
        acc = m12_posthoc_acceptability(
            df,
            road_budget_s=b["road"],
            sidewalk_budget_s=b["sidewalk"],
            building_budget_s=b["building"],
        )

        rows.append({
            "label": label,
            "budget_name": b["name"],
            "road_budget_s": b["road"],
            "sidewalk_budget_s": b["sidewalk"],
            "building_budget_s": b["building"],
            "acceptability_rate": float(acc.mean()),
            "success_rate": float(df["is_success"].astype(bool).mean()),
            "mean_road_time": float(df["road_time"].mean()),
            "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
            "mean_building_time": float(df["building_time"].mean()),
            "mean_green_time": float(df["green_time"].mean()),
            "mean_steps": float(df["steps"].mean()),
            "mean_dist_xy": float(df["dist_xy"].mean()),
        })

    out = pd.DataFrame(rows)

    path = m12_diag_dir / f"{label}_budget_sensitivity.csv"
    out.to_csv(path, index=False)
    print("Saved:", path)

    display(out)

    return out

def run_m12_real_planner_lambda_agl_sweep(
    world,
    drive_dirs,
    lambdas=(0.35, 1.0, 3.0, 8.0),
    agls=(30.0, 40.0, 60.0, 80.0),
    n_episodes=50,
    seed0=7000,
    stage="B",
):
    rows = []

    for agl in agls:
        for lam in lambdas:
            cfg = M10RecedingHorizonPlannerConfig(
                plan_res_m=5.0,
                planner_agl_m=float(agl),
                risk_lambda=float(lam),
                use_astar=True,

                sensing_radius_m=75.0,
                replan_every_steps=5,
                prior_risk_mode="perfect",
                hard_blockers_known=True,

                waypoint_radius_m=8.0,
                target_speed_mps=12.0,
                max_steps=1000,
            )

            prefix = (
                f"m12_real_{REAL_AOI_ID}_m10_sweep_"
                f"agl{agl}_lambda{lam}_stage{stage}_n{n_episodes}"
            )

            df_s, summary_s, csv_s, json_s = evaluate_m10_receding_planner_drive(
                world=world,
                drive_dirs=drive_dirs,
                cfg=cfg,
                risk_profile="balanced",
                n_episodes=n_episodes,
                seed0=seed0,
                stage=stage,
                mode="strict_eval",
                n_obstacles=0,
                prefix=prefix,
                save_paths=False,
            )

            rows.append({
                "planner_agl_m": float(agl),
                "risk_lambda": float(lam),
                "stage": stage,
                "n_episodes": int(n_episodes),

                "success_rate": summary_s["success_rate"],
                "mission_acceptability_rate": summary_s["mission_acceptability_rate"],
                "acceptable_given_success_rate": summary_s["acceptable_given_success_rate"],

                "mean_steps": summary_s["mean_steps"],
                "mean_dist_xy": summary_s["mean_dist_xy"],
                "mean_urban_risk": summary_s["mean_urban_risk"],
                "mean_cumulative_risk": summary_s["mean_cumulative_risk"],

                "mean_road_time": summary_s["mean_road_time"],
                "mean_sidewalk_time": summary_s["mean_sidewalk_time"],
                "mean_building_time": summary_s["mean_building_time"],
                "mean_green_time": summary_s["mean_green_time"],
                "mean_canopy_time": summary_s["mean_canopy_time"],
                "mean_sensitive_time": summary_s["mean_sensitive_time"],
                "mean_hard_violation_steps": summary_s["mean_hard_violation_steps"],

                "term_success": summary_s["termination_counts"].get("success", 0),
                "term_stuck": summary_s["termination_counts"].get("stuck", 0),
                "term_shield_stuck": summary_s["termination_counts"].get("shield_stuck", 0),
                "term_sidewalk_budget": summary_s["termination_counts"].get("urban_sidewalk_budget_exceeded", 0),
                "term_building_budget": summary_s["termination_counts"].get("urban_building_budget_exceeded", 0),

                "csv_path": csv_s,
                "summary_path": json_s,
            })

    sweep_df = pd.DataFrame(rows)

    out_path = m12_diag_dir / "m12_real_m10_lambda_agl_sweep.csv"
    sweep_df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    display(sweep_df.sort_values(["mission_acceptability_rate", "success_rate"], ascending=False))

    return sweep_df

def clone_world_with_replaced_sidewalk(world, new_sidewalk_mask, new_aoi_id_suffix):
    # Shallow clone object.
    obj = UrbanWorldLayers.__new__(UrbanWorldLayers)

    attrs = [
        "aoi_id", "crs", "bbox", "resolution_m",
        "terrain_z", "surface_z", "height_above_ground",
        "building_mask", "building_height", "building_roof_z",
        "road_mask", "sidewalk_mask", "green_mask",
        "canopy_mask", "canopy_height", "canopy_top_z", "canopy_density",
        "sensitive_mask",
    ]

    for a in attrs:
        if hasattr(world, a):
            try:
                setattr(obj, a, getattr(world, a))
            except Exception:
                pass

    try:
        setattr(obj, "aoi_id", str(getattr(world, "aoi_id", "real_world")) + "_" + str(new_aoi_id_suffix))
    except Exception:
        pass

    setattr(obj, "sidewalk_mask", np.asarray(new_sidewalk_mask, dtype=np.float32))

    obj = patch_urban_world_layers_for_ruscuav(obj, verbose=False)

    return obj

def classify_real_aoi_from_coverage(coverage_df):
    cov = {
        row["layer"]: float(row["coverage"])
        for _, row in coverage_df.iterrows()
    }

    building = cov.get("building", 0.0)
    road = cov.get("road", 0.0)
    sidewalk = cov.get("sidewalk", 0.0)
    green = cov.get("green", 0.0)
    canopy = cov.get("canopy", 0.0)
    sensitive = cov.get("sensitive", 0.0)

    hard_urban = building + road + sidewalk

    if building > 0.30 or hard_urban > 0.65 or sensitive > 0.30:
        label = "dense_center_stress_test"
    elif green >= 0.20 and building <= 0.25 and sidewalk <= 0.20:
        label = "mixed_urban_green_validation"
    else:
        label = "intermediate_urban"

    diagnosis = {
        "label": label,
        "building_coverage": building,
        "road_coverage": road,
        "sidewalk_coverage": sidewalk,
        "green_coverage": green,
        "canopy_coverage": canopy,
        "sensitive_coverage": sensitive,
        "hard_urban_sum_building_road_sidewalk": hard_urban,
        "interpretation": None,
    }

    if label == "dense_center_stress_test":
        diagnosis["interpretation"] = (
            "AOI is extremely dense for low-altitude cargo UAV operations. "
            "Use as dense-city stress test, not as normal validation AOI."
        )
    elif label == "mixed_urban_green_validation":
        diagnosis["interpretation"] = (
            "AOI has a useful mixture of urban fabric and green corridors. "
            "Suitable as real validation AOI."
        )
    else:
        diagnosis["interpretation"] = (
            "AOI is intermediate; inspect maps and feasibility before selecting as validation AOI."
        )

    return diagnosis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

import json

def _m12_mask_value_at(world, mask_name, x, y):
    arr = np.asarray(getattr(world, mask_name), dtype=np.float32)
    h, w = arr.shape
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))
    return float(arr[yi, xi])

def _m12_ground_at_world(world, x, y):
    arr = np.asarray(world.terrain_z, dtype=np.float32)
    h, w = arr.shape
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))
    return float(arr[yi, xi])

def pick_m12_episode_seed(df, termination=None, success=None, sort_by="dist_xy", ascending=True):
    """
    Pick one seed from an evaluation dataframe.

    Examples:
        seed = pick_m12_episode_seed(df_m12_m10_no_terminal, success=True)
        seed = pick_m12_episode_seed(df_m12_m10, termination="urban_sidewalk_budget_exceeded")
    """
    d = df.copy()

    if termination is not None:
        d = d[d["termination"] == termination]

    if success is not None:
        d = d[d["is_success"].astype(bool) == bool(success)]

    if len(d) == 0:
        raise ValueError("No episode matching the requested filters.")

    if sort_by in d.columns:
        d = d.sort_values(sort_by, ascending=ascending)

    row = d.iloc[0]
    return int(row["seed"]), int(row["episode"])

def rollout_m12_receding_planner_with_telemetry(
    world,
    seed,
    cfg,
    stage="B",
    mode="strict_eval",
    n_obstacles=0,
    max_steps=None,
):
    """
    Re-run a receding A* planner episode and record rich per-step telemetry.
    Use stage='B' for no-terminal budget diagnostic, stage='C' for strict terminal budgets.
    """
    if max_steps is None:
        max_steps = int(cfg.max_steps)

    env = make_ruscuav_env_m4_mode(
        world=world,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
    )

    obs, reset_info = env.reset(seed=seed)

    planner = M10RecedingHorizonRiskPlanner(
        world=world,
        config=cfg,
        risk_profile="balanced",
    )

    planner.reset_belief()
    planner.reveal_around_xy(env.current_pos[:2])

    target_xy = env.target_point[:2].copy()

    current_plan = None
    current_plan_result = None
    wp_idx = 0

    rows = []
    rewards = []
    path_actual = [env.current_pos.copy()]
    planned_paths = []

    replan_success_count = 0
    replan_fail_count = 0

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        need_replan = False

        if current_plan is None:
            need_replan = True

        if steps % int(max(1, cfg.replan_every_steps)) == 0:
            need_replan = True

        if need_replan:
            planner.reveal_around_xy(env.current_pos[:2])

            current_plan_result = planner.plan(
                start_xy=env.current_pos[:2].copy(),
                target_xy=target_xy,
            )

            if current_plan_result["success"]:
                current_plan = current_plan_result["path_xy"]
                planned_paths.append(current_plan.copy())
                wp_idx = 1 if len(current_plan) > 1 else 0
                replan_success_count += 1
            else:
                replan_fail_count += 1

                if current_plan is None:
                    break

        action, wp_idx = m9_waypoint_controller_action(
            env=env,
            path_xy=current_plan,
            wp_idx=wp_idx,
            cfg=_m10_to_m9_config(cfg),
        )

        obs, reward, terminated, truncated, info = env.step(action)

        pos = env.current_pos.copy()
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        ground = _m12_ground_at_world(world, x, y)

        row = {
            "step": int(steps + 1),
            "t_s": float((steps + 1) * env.dt),

            "x": x,
            "y": y,
            "z": z,
            "ground_z": ground,
            "agl_m": float(z - ground),

            "reward": float(reward),
            "termination": info.get("termination"),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "is_success": bool(info.get("is_success", False)),
            "mission_acceptable": bool(info.get("mission_acceptable", False)),

            "dist_xy": float(info.get("dist_xy", np.nan)),
            "cargo_health": float(info.get("cargo_health", np.nan)),

            "urban_risk": float(info.get("urban_risk", info.get("urban_mean_risk", np.nan))),
            "urban_mean_risk": float(info.get("urban_mean_risk", np.nan)),
            "urban_cumulative_risk": float(info.get("urban_cumulative_risk", info.get("urban_risk", np.nan))),

            "road_time": float(info.get("urban_road_time", 0.0)),
            "sidewalk_time": float(info.get("urban_sidewalk_time", 0.0)),
            "building_time": float(info.get("urban_building_time", 0.0)),
            "green_time": float(info.get("urban_green_time", 0.0)),
            "canopy_time": float(info.get("urban_canopy_time", 0.0)),
            "sensitive_time": float(info.get("urban_sensitive_time", 0.0)),

            "building_mask": _m12_mask_value_at(world, "building_mask", x, y),
            "road_mask": _m12_mask_value_at(world, "road_mask", x, y),
            "sidewalk_mask": _m12_mask_value_at(world, "sidewalk_mask", x, y),
            "green_mask": _m12_mask_value_at(world, "green_mask", x, y),
            "canopy_mask": _m12_mask_value_at(world, "canopy_mask", x, y),
            "sensitive_mask": _m12_mask_value_at(world, "sensitive_mask", x, y),

            "wp_idx": int(wp_idx),
            "replan_count": int(planner.replan_count),
            "total_expanded": int(planner.total_expanded),
            "total_plan_time_s": float(planner.total_plan_time_s),
        }

        rows.append(row)
        rewards.append(float(reward))
        path_actual.append(pos)

        done = bool(terminated or truncated)
        steps += 1

    df = pd.DataFrame(rows)
    path_actual = np.asarray(path_actual, dtype=np.float32)

    final = rows[-1] if rows else {
        "termination": "planner_no_path",
        "is_success": False,
        "mission_acceptable": False,
    }

    result = {
        "seed": int(seed),
        "stage": stage,
        "mode": mode,
        "path": path_actual,
        "planned_paths": planned_paths,
        "df": df,
        "reset_info": reset_info,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": final.get("termination"),
        "is_success": bool(final.get("is_success", False)),
        "mission_acceptable": bool(final.get("mission_acceptable", False)),
        "planner_replan_count": int(planner.replan_count),
        "planner_total_expanded": int(planner.total_expanded),
        "planner_total_plan_time_s": float(planner.total_plan_time_s),
        "replan_success_count": int(replan_success_count),
        "replan_fail_count": int(replan_fail_count),
    }

    env.close()

    return result

def m12_plot_real_aoi_semantic_2d(
    world,
    rollout=None,
    title=None,
    save_path=None,
    figsize=(10, 10),
):
    if title is None:
        title = f"Real GIS AOI semantic map: {getattr(world, 'aoi_id', '')}"

    h, w = world.terrain_z.shape

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(
        world.terrain_z,
        origin="upper",
        extent=[0, w, h, 0],
        cmap="terrain",
        alpha=0.85,
    )

    def overlay(mask, cmap, alpha):
        arr = np.ma.masked_where(np.asarray(mask) <= 0.5, mask)
        ax.imshow(
            arr,
            origin="upper",
            extent=[0, w, h, 0],
            cmap=cmap,
            alpha=alpha,
        )

    overlay(world.green_mask, "Greens", 0.30)
    overlay(world.canopy_mask, "Greens", 0.48)
    overlay(world.road_mask, "Oranges", 0.48)
    overlay(world.sidewalk_mask, "Wistia", 0.48)
    overlay(world.building_mask, "Greys", 0.60)
    overlay(world.sensitive_mask, "Reds", 0.38)

    if rollout is not None:
        path = rollout["path"]
        ax.plot(path[:, 0], path[:, 1], linewidth=2.5, label="actual UAV path")
        ax.scatter(path[0, 0], path[0, 1], s=80, marker="o", edgecolors="black", label="START")
        ax.scatter(path[-1, 0], path[-1, 1], s=90, marker="s", edgecolors="black", label="END")

        if rollout.get("planned_paths"):
            last_plan = rollout["planned_paths"][-1]
            ax.plot(last_plan[:, 0], last_plan[:, 1], linestyle="--", linewidth=1.5, label="last planned path")

        title = (
            f"{title}\n"
            f"seed={rollout['seed']} | stage={rollout['stage']} | "
            f"term={rollout['termination']} | success={rollout['is_success']} | "
            f"acceptable={rollout['mission_acceptable']}"
        )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.grid(alpha=0.2)

    if rollout is not None:
        ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220)
        print("Saved:", save_path)

    plt.show()

    return fig, ax

def m12_plot_real_aoi_risk_2d(
    world,
    rollout=None,
    agl_m=30.0,
    title=None,
    save_path=None,
    figsize=(10, 10),
):
    risk_model = UrbanRiskModel(
        world=world,
        profile="balanced",
        hard_sensitive_zones=True,
        include_global_altitude_band_cost=True,
    )

    risk = risk_model.risk_map_at_agl(float(agl_m))

    h, w = risk.shape

    if title is None:
        title = f"Real GIS AOI risk map at AGL={agl_m:.0f} m"

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        risk,
        origin="upper",
        extent=[0, w, h, 0],
        cmap="inferno",
    )
    plt.colorbar(im, ax=ax, label="urban risk")

    if rollout is not None:
        path = rollout["path"]
        ax.plot(path[:, 0], path[:, 1], linewidth=2.5, label="actual UAV path")
        ax.scatter(path[0, 0], path[0, 1], s=80, marker="o", edgecolors="black", label="START")
        ax.scatter(path[-1, 0], path[-1, 1], s=90, marker="s", edgecolors="black", label="END")

        if rollout.get("planned_paths"):
            last_plan = rollout["planned_paths"][-1]
            ax.plot(last_plan[:, 0], last_plan[:, 1], linestyle="--", linewidth=1.5, label="last planned path")

        title = (
            f"{title}\n"
            f"seed={rollout['seed']} | stage={rollout['stage']} | "
            f"term={rollout['termination']} | success={rollout['is_success']} | "
            f"acceptable={rollout['mission_acceptable']}"
        )

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.grid(alpha=0.2)

    if rollout is not None:
        ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220)
        print("Saved:", save_path)

    plt.show()

    return fig, ax

def m12_plot_real_aoi_rollout_3d(
    world,
    rollout,
    stride=8,
    title=None,
    html_path=None,
    show_roads=True,
    show_sidewalks=True,
    show_green=True,
    show_canopy=True,
    show_buildings=True,
    show_sensitive=True,
):
    import plotly.graph_objects as go

    terrain = np.asarray(world.terrain_z, dtype=np.float32)
    h, w = terrain.shape

    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    X, Y = np.meshgrid(xs, ys)
    Z = terrain[np.ix_(ys, xs)]

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Earth",
            opacity=0.78,
            showscale=False,
            name="terrain",
        )
    )

    def add_mask_surface(mask, zsurf, name, color, opacity):
        mask_ds = np.asarray(mask, dtype=np.float32)[np.ix_(ys, xs)]
        z_ds = np.asarray(zsurf, dtype=np.float32)[np.ix_(ys, xs)]
        Zm = np.where(mask_ds > 0.5, z_ds, np.nan)

        if not np.isfinite(Zm).any():
            return

        fig.add_trace(
            go.Surface(
                x=X,
                y=Y,
                z=Zm,
                surfacecolor=np.ones_like(Zm),
                colorscale=[[0.0, color], [1.0, color]],
                opacity=opacity,
                showscale=False,
                name=name,
                hoverinfo="skip",
            )
        )

    base = terrain

    if show_green:
        add_mask_surface(world.green_mask, base + 0.7, "green", "green", 0.25)

    if show_roads:
        add_mask_surface(world.road_mask, base + 1.0, "roads", "orange", 0.42)

    if show_sidewalks:
        add_mask_surface(world.sidewalk_mask, base + 1.4, "sidewalks", "yellow", 0.38)

    if show_sensitive:
        add_mask_surface(world.sensitive_mask, base + 2.0, "sensitive", "red", 0.28)

    if show_canopy:
        canopy_top = np.asarray(world.canopy_top_z, dtype=np.float32)
        add_mask_surface(world.canopy_mask, canopy_top, "canopy top", "darkgreen", 0.45)

    if show_buildings:
        roof = np.asarray(world.building_roof_z, dtype=np.float32)
        add_mask_surface(world.building_mask, roof, "building roofs", "lightgrey", 0.75)

    path = rollout["path"]

    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines",
            line=dict(width=7, color="cyan"),
            name="actual UAV path",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[path[0, 0]],
            y=[path[0, 1]],
            z=[path[0, 2]],
            mode="markers+text",
            marker=dict(size=7, color="blue"),
            text=["START"],
            textposition="top center",
            name="START",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[path[-1, 0]],
            y=[path[-1, 1]],
            z=[path[-1, 2]],
            mode="markers+text",
            marker=dict(size=8, color="cyan"),
            text=["END"],
            textposition="top center",
            name="END",
        )
    )

    if rollout.get("planned_paths"):
        last_plan = rollout["planned_paths"][-1]
        # approximate z of planned path over terrain + 30 m
        z_plan = []
        for p in last_plan:
            z_plan.append(_m12_ground_at_world(world, p[0], p[1]) + 30.0)
        z_plan = np.asarray(z_plan, dtype=np.float32)

        fig.add_trace(
            go.Scatter3d(
                x=last_plan[:, 0],
                y=last_plan[:, 1],
                z=z_plan,
                mode="lines",
                line=dict(width=4, color="white", dash="dash"),
                name="last planned path approx.",
            )
        )

    if title is None:
        title = (
            f"Real AOI 3D rollout | seed={rollout['seed']} | "
            f"stage={rollout['stage']} | term={rollout['termination']} | "
            f"success={rollout['is_success']} | acceptable={rollout['mission_acceptable']}"
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.25),
        ),
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.65)"),
    )

    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path))
        print("Saved HTML:", html_path)

    fig.show()

    return fig

def m12_plot_rollout_timeseries(rollout, save_dir=None):
    df = rollout["df"]

    if len(df) == 0:
        print("Empty rollout telemetry.")
        return

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    series = [
        ("dist_xy", "Distance to target [m]"),
        ("agl_m", "AGL altitude [m]"),
        ("urban_risk", "Instant urban risk"),
        ("cargo_health", "Cargo Health"),
        ("road_time", "Cumulative road time [s]"),
        ("sidewalk_time", "Cumulative sidewalk time [s]"),
        ("building_time", "Cumulative building time [s]"),
        ("green_time", "Cumulative green time [s]"),
    ]

    for col, title in series:
        if col not in df.columns:
            continue

        plt.figure(figsize=(9, 4))
        plt.plot(df["t_s"], df[col])
        plt.xlabel("Time [s]")
        plt.ylabel(col)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_dir is not None:
            path = save_dir / f"timeseries_{col}_seed_{rollout['seed']}.png"
            plt.savefig(path, dpi=220)
            print("Saved:", path)

        plt.show()

    # Categorical exposure along path
    plt.figure(figsize=(10, 4))
    for col in ["road_mask", "sidewalk_mask", "building_mask", "green_mask", "canopy_mask", "sensitive_mask"]:
        if col in df.columns:
            plt.plot(df["t_s"], df[col], label=col)

    plt.xlabel("Time [s]")
    plt.ylabel("Layer active")
    plt.title("Semantic layer under UAV over time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if save_dir is not None:
        path = save_dir / f"timeseries_layers_seed_{rollout['seed']}.png"
        plt.savefig(path, dpi=220)
        print("Saved:", path)

    plt.show()

def run_m12_stepA_planner_eval(world, world_label, stage, n_episodes=100, seed0=7000):
    prefix = (
        f"m12_stepA_{REAL_AOI_ID}_{world_label}_"
        f"m10_agl30_lambda035_stage{stage}_n{n_episodes}"
    )

    df, summary, csv_path, json_path = evaluate_m10_receding_planner_drive(
        world=world,
        drive_dirs=M12_REAL_DRIVE_DIRS,
        cfg=m12_stepA_cfg_agl30,
        risk_profile="balanced",
        n_episodes=n_episodes,
        seed0=seed0,
        stage=stage,
        mode="strict_eval",
        n_obstacles=0,
        prefix=prefix,
        save_paths=True,
    )

    return df, summary, csv_path, json_path

def m12_posthoc_acceptability(
    df,
    road_budget_s,
    sidewalk_budget_s,
    building_budget_s,
    sensitive_budget_s=0.0,
    cargo_min=0.80,
):
    return (
        (df["is_success"].astype(bool) == True)
        & (df["cargo_health"].astype(float) >= float(cargo_min))
        & (df["road_time"].astype(float) <= float(road_budget_s))
        & (df["sidewalk_time"].astype(float) <= float(sidewalk_budget_s))
        & (df["building_time"].astype(float) <= float(building_budget_s))
        & (df["sensitive_time"].astype(float) <= float(sensitive_budget_s))
        & (df["urban_hard_violation_steps"].astype(float) == 0.0)
    )

def m12_budget_sensitivity_table(df, label):
    budget_sets = [
        {"name": "synthetic_nominal", "road": 20, "sidewalk": 5, "building": 15},
        {"name": "terminal_like", "road": 30, "sidewalk": 10, "building": 20},
        {"name": "dense_city_moderate", "road": 45, "sidewalk": 20, "building": 30},
        {"name": "dense_city_relaxed", "road": 60, "sidewalk": 30, "building": 45},
        {"name": "diagnostic_relaxed", "road": 90, "sidewalk": 45, "building": 60},
    ]

    rows = []

    for b in budget_sets:
        acc = m12_posthoc_acceptability(
            df,
            road_budget_s=b["road"],
            sidewalk_budget_s=b["sidewalk"],
            building_budget_s=b["building"],
        )

        rows.append({
            "label": label,
            "budget_name": b["name"],
            "road_budget_s": b["road"],
            "sidewalk_budget_s": b["sidewalk"],
            "building_budget_s": b["building"],

            "acceptability_rate": float(acc.mean()),
            "success_rate": float(df["is_success"].astype(bool).mean()),

            "mean_steps": float(df["steps"].mean()),
            "mean_dist_xy": float(df["dist_xy"].mean()),
            "mean_road_time": float(df["road_time"].mean()),
            "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
            "mean_building_time": float(df["building_time"].mean()),
            "mean_green_time": float(df["green_time"].mean()),
            "mean_cumulative_risk": float(df["urban_cumulative_risk"].mean() if "urban_cumulative_risk" in df.columns else df["urban_mean_risk"].mean()),
        })

    return pd.DataFrame(rows)

def _m12_summary_row(label, summary, csv_path=None, json_path=None):
    return {
        "label": label,

        "success_rate": summary.get("success_rate", np.nan),
        "mission_acceptability_rate": summary.get("mission_acceptability_rate", np.nan),
        "acceptable_given_success_rate": summary.get("acceptable_given_success_rate", np.nan),

        "mean_steps": summary.get("mean_steps", np.nan),
        "mean_dist_xy": summary.get("mean_dist_xy", np.nan),
        "mean_cargo_health": summary.get("mean_cargo_health", np.nan),
        "mean_urban_risk": summary.get("mean_urban_risk", np.nan),
        "mean_cumulative_risk": summary.get("mean_cumulative_risk", np.nan),

        "mean_road_time": summary.get("mean_road_time", np.nan),
        "mean_sidewalk_time": summary.get("mean_sidewalk_time", np.nan),
        "mean_building_time": summary.get("mean_building_time", np.nan),
        "mean_green_time": summary.get("mean_green_time", np.nan),
        "mean_canopy_time": summary.get("mean_canopy_time", np.nan),
        "mean_sensitive_time": summary.get("mean_sensitive_time", np.nan),
        "mean_hard_violation_steps": summary.get("mean_hard_violation_steps", np.nan),

        "term_success": summary.get("termination_counts", {}).get("success", 0),
        "term_stuck": summary.get("termination_counts", {}).get("stuck", 0),
        "term_shield_stuck": summary.get("termination_counts", {}).get("shield_stuck", 0),
        "term_planner_no_path": summary.get("termination_counts", {}).get("planner_no_path", 0),
        "term_sidewalk_budget": summary.get("termination_counts", {}).get("urban_sidewalk_budget_exceeded", 0),
        "term_building_budget": summary.get("termination_counts", {}).get("urban_building_budget_exceeded", 0),

        "csv_path": csv_path,
        "summary_path": json_path,
    }

from shapely.ops import unary_union

import geopandas as gpd

import numpy as np

import pandas as pd

def fetch_building_parts_for_current_aoi(aoi_geo):
    tags_building_parts = {
        "building:part": True,
    }

    parts_raw = fetch_osm_features(
        tags_building_parts,
        aoi_geo,
        "building_parts",
    )

    parts_local = project_clip_to_local(
        parts_raw,
        aoi_geo,
        AOI_SIZE_M,
    )

    if parts_local is None or len(parts_local) == 0:
        print("[building:part] none found")
        return gpd.GeoDataFrame(geometry=[], crs=None)

    parts_local = parts_local.copy()
    parts_local = parts_local[
        parts_local.geometry.notnull()
        & ~parts_local.geometry.is_empty
        & parts_local.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if len(parts_local) == 0:
        print("[building:part] no polygonal parts after filtering")
        return gpd.GeoDataFrame(geometry=[], crs=None)

    parts_local["height_m"] = parts_local.apply(
        estimate_building_height_row,
        axis=1,
    ).astype(float)

    # Optional min_height if available, useful for parts on top of podiums.
    def parse_min_height(row):
        for col in ["min_height", "building:min_height"]:
            if col in row:
                v = parse_osm_numeric(row.get(col))
                if np.isfinite(v) and v >= 0:
                    return float(v)
        return 0.0

    parts_local["min_height_m"] = parts_local.apply(
        parse_min_height,
        axis=1,
    ).astype(float)

    print("[building:part] local polygon parts:", len(parts_local))
    print(
        "[building:part] height min/mean/max:",
        float(parts_local["height_m"].min()),
        float(parts_local["height_m"].mean()),
        float(parts_local["height_m"].max()),
    )

    return parts_local

def prepare_visual_buildings_gdf(buildings_gdf, building_parts_gdf=None, overlap_threshold=0.20):
    """
    Creates a visualization building layer.

    If building:part geometries are available, they are preferred.
    Original building footprints strongly overlapped by parts are removed
    to avoid duplicate z-fighting.
    """
    if buildings_gdf is None:
        buildings_gdf = gpd.GeoDataFrame(geometry=[], crs=None)

    base = buildings_gdf.copy()

    if len(base) > 0:
        base = base[
            base.geometry.notnull()
            & ~base.geometry.is_empty
            & base.geometry.type.isin(["Polygon", "MultiPolygon"])
        ].copy()

        if "height_m" not in base.columns:
            base["height_m"] = base.apply(
                estimate_building_height_row,
                axis=1,
            ).astype(float)

        if "min_height_m" not in base.columns:
            base["min_height_m"] = 0.0

    if building_parts_gdf is None or len(building_parts_gdf) == 0:
        print("[visual buildings] using building footprints only:", len(base))
        return base

    parts = building_parts_gdf.copy()

    if len(parts) > 0:
        parts = parts[
            parts.geometry.notnull()
            & ~parts.geometry.is_empty
            & parts.geometry.type.isin(["Polygon", "MultiPolygon"])
        ].copy()

        if "height_m" not in parts.columns:
            parts["height_m"] = parts.apply(
                estimate_building_height_row,
                axis=1,
            ).astype(float)

        if "min_height_m" not in parts.columns:
            parts["min_height_m"] = 0.0

    if len(parts) == 0:
        print("[visual buildings] no valid parts, using footprints only:", len(base))
        return base

    try:
        parts_union = unary_union(list(parts.geometry))
    except Exception:
        parts_union = None

    keep_base = []

    if parts_union is not None and len(base) > 0:
        for idx, row in base.iterrows():
            geom = row.geometry
            area = float(max(1e-6, geom.area))

            try:
                inter_area = float(geom.intersection(parts_union).area)
                frac = inter_area / area
            except Exception:
                frac = 0.0

            if frac < float(overlap_threshold):
                keep_base.append(idx)

        base_kept = base.loc[keep_base].copy()
    else:
        base_kept = base

    combined = pd.concat(
        [base_kept, parts],
        ignore_index=True,
    )

    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=None)

    print("[visual buildings] base kept:", len(base_kept))
    print("[visual buildings] parts:", len(parts))
    print("[visual buildings] combined:", len(combined))

    return combined

from shapely.geometry import Polygon, MultiPolygon

from shapely.ops import triangulate

import plotly.graph_objects as go

def _terrain_at_world_xy(world, x, y):
    arr = np.asarray(world.terrain_z, dtype=np.float32)
    h, w = arr.shape
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))
    return float(arr[yi, xi])

def _iter_polygons_from_geom(geom):
    if geom is None or geom.is_empty:
        return

    if isinstance(geom, Polygon):
        yield geom

    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            if p is not None and not p.is_empty:
                yield p

def _safe_polygon_for_mesh(poly):
    if poly is None or poly.is_empty:
        return None

    try:
        if not poly.is_valid:
            poly = poly.buffer(0)
    except Exception:
        return None

    if poly is None or poly.is_empty:
        return None

    if not isinstance(poly, Polygon):
        return None

    if poly.area < 1.0:
        return None

    return poly

def make_building_prism_mesh3d(
    world,
    buildings_gdf,
    min_area_m2=20.0,
    max_buildings=1200,
    simplify_tolerance_m=0.35,
    default_height_m=12.0,
    color="lightgrey",
    opacity=0.82,
    name="solid buildings",
):
    """
    Create a single Plotly Mesh3d trace containing extruded building solids.

    Geometry:
        - vertical walls from terrain to roof;
        - triangulated flat roof;
        - optional min_height_m for building:part.

    Limitations:
        - no true LOD2/LOD3 roof geometry;
        - no detailed PKiN tower/roof unless OSM building:part contains it;
        - holes are not explicitly walled, but roof triangulation respects polygon body approximately.
    """
    if buildings_gdf is None or len(buildings_gdf) == 0:
        print("[building mesh] no buildings")
        return None

    gdf = buildings_gdf.copy()
    gdf = gdf[
        gdf.geometry.notnull()
        & ~gdf.geometry.is_empty
        & gdf.geometry.type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if len(gdf) == 0:
        print("[building mesh] no valid polygon buildings")
        return None

    gdf["area_m2_for_sort"] = gdf.geometry.area.astype(float)
    gdf = gdf[gdf["area_m2_for_sort"] >= float(min_area_m2)].copy()
    gdf = gdf.sort_values("area_m2_for_sort", ascending=False)

    if max_buildings is not None:
        gdf = gdf.head(int(max_buildings)).copy()

    xs = []
    ys = []
    zs = []
    ii = []
    jj = []
    kk = []

    def add_vertex(x, y, z):
        idx = len(xs)
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
        return idx

    def add_tri(a, b, c):
        ii.append(int(a))
        jj.append(int(b))
        kk.append(int(c))

    buildings_added = 0
    roof_triangles = 0
    wall_triangles = 0

    for _, row in gdf.iterrows():
        h_val = row.get("height_m", default_height_m)
        try:
            h_val = float(h_val)
        except Exception:
            h_val = float(default_height_m)

        if not np.isfinite(h_val) or h_val <= 0:
            h_val = float(default_height_m)

        min_h = row.get("min_height_m", 0.0)
        try:
            min_h = float(min_h)
        except Exception:
            min_h = 0.0

        min_h = max(0.0, min_h)

        for poly in _iter_polygons_from_geom(row.geometry):
            poly = _safe_polygon_for_mesh(poly)
            if poly is None or poly.area < float(min_area_m2):
                continue

            if simplify_tolerance_m is not None and simplify_tolerance_m > 0:
                try:
                    poly_s = poly.simplify(float(simplify_tolerance_m), preserve_topology=True)
                    poly_s = _safe_polygon_for_mesh(poly_s)
                    if poly_s is not None and poly_s.area > 0:
                        poly = poly_s
                except Exception:
                    pass

            # Use centroid ground as base reference for flat roof.
            c = poly.representative_point()
            base_ground = _terrain_at_world_xy(world, c.x, c.y)

            bottom_z_flat = base_ground + min_h
            roof_z_flat = base_ground + h_val

            if roof_z_flat <= bottom_z_flat + 0.5:
                roof_z_flat = bottom_z_flat + 3.0

            # Walls from exterior ring.
            coords = list(poly.exterior.coords)
            if len(coords) < 4:
                continue

            for p0, p1 in zip(coords[:-1], coords[1:]):
                x0, y0 = float(p0[0]), float(p0[1])
                x1, y1 = float(p1[0]), float(p1[1])

                # Use terrain at each wall vertex for natural base, but keep part min height.
                z0b = _terrain_at_world_xy(world, x0, y0) + min_h
                z1b = _terrain_at_world_xy(world, x1, y1) + min_h

                z0t = roof_z_flat
                z1t = roof_z_flat

                v0 = add_vertex(x0, y0, z0b)
                v1 = add_vertex(x1, y1, z1b)
                v2 = add_vertex(x1, y1, z1t)
                v3 = add_vertex(x0, y0, z0t)

                add_tri(v0, v1, v2)
                add_tri(v0, v2, v3)
                wall_triangles += 2

            # Roof triangulation.
            try:
                tris = triangulate(poly)
            except Exception:
                tris = []

            for tri in tris:
                if tri is None or tri.is_empty:
                    continue

                # Filter triangles outside polygon.
                try:
                    if not poly.contains(tri.representative_point()):
                        continue
                except Exception:
                    continue

                tcoords = list(tri.exterior.coords)[:3]
                if len(tcoords) != 3:
                    continue

                v = []
                for x, y in tcoords:
                    v.append(add_vertex(float(x), float(y), roof_z_flat))

                add_tri(v[0], v[1], v[2])
                roof_triangles += 1

            buildings_added += 1

    print("=" * 90)
    print("Building solid mesh summary")
    print("=" * 90)
    print("buildings_added:", buildings_added)
    print("vertices:", len(xs))
    print("triangles:", len(ii))
    print("wall_triangles:", wall_triangles)
    print("roof_triangles:", roof_triangles)
    print("=" * 90)

    if len(xs) == 0 or len(ii) == 0:
        return None

    mesh = go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=ii,
        j=jj,
        k=kk,
        color=color,
        opacity=float(opacity),
        name=name,
        flatshading=True,
        lighting=dict(
            ambient=0.55,
            diffuse=0.75,
            specular=0.15,
            roughness=0.85,
        ),
        hoverinfo="skip",
    )

    return mesh

def add_semantic_ground_points(
    fig,
    world,
    mask_name,
    color,
    name,
    stride=8,
    z_offset=0.8,
    marker_size=2,
    opacity=0.45,
):
    arr = np.asarray(getattr(world, mask_name), dtype=np.float32)
    terrain = np.asarray(world.terrain_z, dtype=np.float32)

    h, w = arr.shape

    ys = np.arange(0, h, int(stride))
    xs = np.arange(0, w, int(stride))

    sub = arr[np.ix_(ys, xs)]
    yy, xx = np.where(sub > 0.5)

    if len(xx) == 0:
        return

    xpts = xs[xx]
    ypts = ys[yy]
    zpts = terrain[ypts, xpts] + float(z_offset)

    fig.add_trace(
        go.Scatter3d(
            x=xpts,
            y=ypts,
            z=zpts,
            mode="markers",
            marker=dict(
                size=marker_size,
                color=color,
                opacity=opacity,
            ),
            name=name,
            hoverinfo="skip",
        )
    )

def plot_m12_rollout_3d_solid_buildings(
    world,
    rollout,
    buildings_gdf,
    stride=8,
    semantic_points=True,
    semantic_stride=10,
    max_buildings=1200,
    min_building_area_m2=20.0,
    title=None,
    html_path=None,
):
    import plotly.graph_objects as go

    terrain = np.asarray(world.terrain_z, dtype=np.float32)
    h, w = terrain.shape

    ys = np.arange(0, h, int(stride))
    xs = np.arange(0, w, int(stride))
    X, Y = np.meshgrid(xs, ys)
    Z = terrain[np.ix_(ys, xs)]

    fig = go.Figure()

    # Terrain
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Earth",
            opacity=0.88,
            showscale=False,
            name="terrain",
        )
    )

    # Ground semantic points to avoid fake vertical surfaces.
    if semantic_points:
        add_semantic_ground_points(fig, world, "green_mask", "green", "green", stride=semantic_stride, z_offset=0.5, opacity=0.35)
        add_semantic_ground_points(fig, world, "road_mask", "orange", "roads", stride=semantic_stride, z_offset=0.8, opacity=0.45)
        add_semantic_ground_points(fig, world, "sidewalk_mask", "yellow", "sidewalks", stride=semantic_stride, z_offset=1.1, opacity=0.45)
        add_semantic_ground_points(fig, world, "sensitive_mask", "red", "sensitive", stride=semantic_stride, z_offset=1.4, opacity=0.30)

    # Solid buildings.
    bmesh = make_building_prism_mesh3d(
        world=world,
        buildings_gdf=buildings_gdf,
        min_area_m2=min_building_area_m2,
        max_buildings=max_buildings,
        simplify_tolerance_m=0.35,
        default_height_m=12.0,
        color="lightgrey",
        opacity=0.82,
        name="solid buildings",
    )

    if bmesh is not None:
        fig.add_trace(bmesh)

    # Canopy as semi-transparent top surface if desired.
    if hasattr(world, "canopy_mask") and hasattr(world, "canopy_top_z"):
        canopy = np.asarray(world.canopy_mask, dtype=np.float32)
        canopy_top = np.asarray(world.canopy_top_z, dtype=np.float32)

        canopy_ds = canopy[np.ix_(ys, xs)]
        zc = canopy_top[np.ix_(ys, xs)]
        zc = np.where(canopy_ds > 0.5, zc, np.nan)

        if np.isfinite(zc).any():
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=zc,
                    colorscale=[[0, "darkgreen"], [1, "darkgreen"]],
                    opacity=0.35,
                    showscale=False,
                    name="canopy top",
                    hoverinfo="skip",
                )
            )

    # Actual rollout path.
    path = rollout["path"]

    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines",
            line=dict(width=8, color="cyan"),
            name="actual UAV path",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[path[0, 0]],
            y=[path[0, 1]],
            z=[path[0, 2]],
            mode="markers+text",
            marker=dict(size=7, color="blue"),
            text=["START"],
            textposition="top center",
            name="START",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[path[-1, 0]],
            y=[path[-1, 1]],
            z=[path[-1, 2]],
            mode="markers+text",
            marker=dict(size=8, color="cyan"),
            text=["END"],
            textposition="top center",
            name="END",
        )
    )

    # Last planned path.
    if rollout.get("planned_paths"):
        last_plan = rollout["planned_paths"][-1]
        z_plan = np.array([
            _terrain_at_world_xy(world, p[0], p[1]) + 30.0
            for p in last_plan
        ], dtype=np.float32)

        fig.add_trace(
            go.Scatter3d(
                x=last_plan[:, 0],
                y=last_plan[:, 1],
                z=z_plan,
                mode="lines",
                line=dict(width=4, color="white", dash="dash"),
                name="last planned path approx.",
            )
        )

    if title is None:
        title = (
            f"Real AOI solid 3D rollout | seed={rollout['seed']} | "
            f"stage={rollout['stage']} | term={rollout['termination']} | "
            f"success={rollout['is_success']} | acceptable={rollout['mission_acceptable']}"
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.28),
        ),
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.65)"),
    )

    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path))
        print("Saved HTML:", html_path)

    fig.show()

    return fig

import numpy as np

import pandas as pd

import json

import geopandas as gpd

from pathlib import Path

from datetime import datetime

VIENNA_AOI_ID = "vienna_prater_leopoldstadt_osm_1km_v1"

VIENNA_CENTER_LAT = 48.2152

VIENNA_CENTER_LON = 16.3975

VIENNA_AOI_SIZE_M = 1000.0

VIENNA_RESOLUTION_M = 1.0

VIENNA_DEM_TIF_PATH = None

VIENNA_DSM_TIF_PATH = None

def build_real_world_for_aoi(
    aoi_id,
    center_lat,
    center_lon,
    size_m=1000.0,
    resolution_m=1.0,
    dem_tif_path=None,
    dsm_tif_path=None,
    verbose=True,
):
    """
    Pełny pipeline: OSM -> warstwy lokalne -> rastry -> UrbanWorldLayers
    (z patchem RUSCUAV). Zwraca (world, aoi_geo, layers_local, masks).
    `masks` zawiera maski pośrednie (m.in. footway/sidewalk proxy)
    potrzebne do diagnostyki i wariantu footway-only.
    """
    # --- 1. CRS + bbox -------------------------------------------------
    aoi_geo = make_aoi_crs_and_bounds(center_lat, center_lon, size_m)
    if verbose:
        print("UTM CRS:", aoi_geo["utm_crs"])
        print("Metric bounds:", aoi_geo["metric_bounds"])
        print("WGS bbox:", aoi_geo["wgs_bbox"])

    # --- 2. Pobranie warstw OSM ----------------------------------------
    tags_buildings = {"building": True}
    tags_highways = {"highway": True}
    tags_green = {
        "leisure": ["park", "garden", "recreation_ground", "playground"],
        "landuse": ["grass", "forest", "meadow", "recreation_ground",
                    "village_green", "cemetery"],
        "natural": ["wood", "grassland", "scrub", "heath"],
    }
    tags_sensitive = {
        "amenity": [
            "school", "kindergarten", "hospital", "clinic", "doctors",
            "university", "college", "social_facility", "nursing_home",
        ],
        "healthcare": True,
        "leisure": ["playground"],
    }

    layers_raw = {
        "buildings": fetch_osm_features(tags_buildings, aoi_geo, f"{aoi_id}/buildings"),
        "highways": fetch_osm_features(tags_highways, aoi_geo, f"{aoi_id}/highways"),
        "green": fetch_osm_features(tags_green, aoi_geo, f"{aoi_id}/green"),
        "sensitive": fetch_osm_features(tags_sensitive, aoi_geo, f"{aoi_id}/sensitive"),
    }

    layers_local = {}
    for k, gdf in layers_raw.items():
        layers_local[k] = project_clip_to_local(gdf, aoi_geo, size_m)
        if verbose:
            print(f"[LOCAL] {k}: {len(layers_local[k])} features")

    # --- 3. Budynki + wysokości ----------------------------------------
    buildings_local = prepare_buildings_layer(layers_local["buildings"])
    building_mask = rasterize_mask(buildings_local)
    building_height = rasterize_values(buildings_local, "height_m")
    building_height = (building_height * building_mask).astype(np.float32)

    if verbose:
        print("buildings:", len(buildings_local),
              "| coverage:", float(building_mask.mean()))

    # --- 4. Drogi / chodniki -------------------------------------------
    roads_poly_gdf, sidewalk_proxy_gdf, footway_gdf = prepare_roads_and_sidewalks(
        layers_local["highways"]
    )

    road_mask = rasterize_mask(roads_poly_gdf)
    sidewalk_proxy_mask = rasterize_mask(sidewalk_proxy_gdf)
    footway_mask = rasterize_mask(footway_gdf)

    sidewalk_mask = np.clip(
        sidewalk_proxy_mask + footway_mask, 0.0, 1.0
    ).astype(np.float32)
    sidewalk_mask[road_mask > 0.5] = 0.0

    if verbose:
        print("road coverage:", float(road_mask.mean()),
              "| sidewalk coverage:", float(sidewalk_mask.mean()))

    # --- 5. Green / canopy / sensitive ---------------------------------
    green_local = prepare_green_layer(layers_local["green"])
    green_mask = rasterize_mask(green_local)

    if len(green_local) > 0:
        green_local = green_local.copy()
        green_local["is_canopy"] = green_local.apply(is_canopy_green_row, axis=1)
        canopy_gdf = green_local[green_local["is_canopy"] == True].copy()
        canopy_gdf["canopy_height_m"] = canopy_gdf.apply(canopy_height_row, axis=1)
    else:
        canopy_gdf = gpd.GeoDataFrame(geometry=[], crs=None)
        canopy_gdf["canopy_height_m"] = []

    canopy_mask = rasterize_mask(canopy_gdf)
    canopy_height = rasterize_values(canopy_gdf, "canopy_height_m")
    canopy_height = (canopy_height * canopy_mask).astype(np.float32)

    canopy_density = np.zeros(OUT_SHAPE, dtype=np.float32)
    canopy_density[canopy_mask > 0.5] = 0.75

    sensitive_gdf = prepare_sensitive_layer(layers_local["sensitive"])
    sensitive_mask = rasterize_mask(sensitive_gdf)

    if verbose:
        print("green coverage:", float(green_mask.mean()),
              "| canopy coverage:", float(canopy_mask.mean()),
              "| sensitive coverage:", float(sensitive_mask.mean()))

    # --- 6. Teren / powierzchnia ---------------------------------------
    terrain_z = load_raster_to_aoi_grid(
        dem_tif_path, aoi_geo,
        out_shape=OUT_SHAPE, resolution_m=resolution_m,
    )
    terrain_fallback_used = terrain_z is None
    if terrain_fallback_used:
        if verbose:
            print("[DEM] Brak DEM — controlled terrain fallback.")
        terrain_z = make_terrain_fallback(OUT_SHAPE)

    dsm_z = load_raster_to_aoi_grid(
        dsm_tif_path, aoi_geo,
        out_shape=OUT_SHAPE, resolution_m=resolution_m,
    )

    building_roof_z = terrain_z + building_height
    canopy_top_z = terrain_z + canopy_height
    surface_proxy_z = np.maximum.reduce(
        [terrain_z, building_roof_z, canopy_top_z]
    ).astype(np.float32)

    surface_z = (
        np.maximum(surface_proxy_z, dsm_z).astype(np.float32)
        if dsm_z is not None else surface_proxy_z
    )

    # --- 7. Ekskluzywność masek (jak Block 10 Warszawa) ----------------
    green_mask = green_mask.astype(np.float32)
    green_mask[building_mask > 0.5] = 0.0

    canopy_mask = canopy_mask.astype(np.float32)
    canopy_mask[building_mask > 0.5] = 0.0
    canopy_height = canopy_height * canopy_mask
    canopy_density = canopy_density * canopy_mask

    sidewalk_mask[building_mask > 0.5] = 0.0

    # --- 8. Obiekt UrbanWorldLayers + patch RUSCUAV --------------------
    world = make_urban_world_layers_object(
        aoi_id=aoi_id,
        crs="LOCAL_RUSCUAV_0_1000",
        bbox=(0.0, 0.0, size_m, size_m),
        resolution_m=resolution_m,
        terrain_z=terrain_z,
        surface_z=surface_z,
        building_mask=building_mask,
        building_height=building_height,
        road_mask=road_mask,
        sidewalk_mask=sidewalk_mask,
        green_mask=green_mask,
        canopy_mask=canopy_mask,
        canopy_height=canopy_height,
        canopy_density=canopy_density,
        sensitive_mask=sensitive_mask,
    )
    world = patch_urban_world_layers_for_ruscuav(world, verbose=verbose)

    masks = {
        "footway_mask": footway_mask,
        "sidewalk_proxy_mask": sidewalk_proxy_mask,
        "terrain_fallback_used": terrain_fallback_used,
    }

    return world, aoi_geo, layers_local, masks

def m12_sidewalk_proxy_diagnostics_v2(world, footway_mask, sidewalk_proxy_mask,
                                      save_dir, label="sidewalk_proxy"):
    fw = np.asarray(footway_mask, dtype=np.float32)
    proxy = np.asarray(sidewalk_proxy_mask, dtype=np.float32)
    final = np.asarray(world.sidewalk_mask, dtype=np.float32)
    road = np.asarray(world.road_mask, dtype=np.float32)
    n = fw.size

    df = pd.DataFrame([
        {"layer": "explicit_footway_mask",
         "coverage": float((fw > 0.5).sum() / n),
         "active_pixels": int((fw > 0.5).sum())},
        {"layer": "sidewalk_proxy_mask",
         "coverage": float((proxy > 0.5).sum() / n),
         "active_pixels": int((proxy > 0.5).sum())},
        {"layer": "final_sidewalk_mask",
         "coverage": float((final > 0.5).sum() / n),
         "active_pixels": int((final > 0.5).sum())},
        {"layer": "road_mask",
         "coverage": float((road > 0.5).sum() / n),
         "active_pixels": int((road > 0.5).sum())},
    ])

    path = Path(save_dir) / f"{label}_diagnostics.csv"
    df.to_csv(path, index=False)
    print("Saved:", path)
    display(df)
    return df

import matplotlib.pyplot as plt

def plot_world_layers_quicklook(world, title):
    layers = [
        ("building_mask", world.building_mask, "Reds"),
        ("road_mask", world.road_mask, "Greys"),
        ("sidewalk_mask", world.sidewalk_mask, "Purples"),
        ("green_mask", world.green_mask, "Greens"),
        ("canopy_height", world.canopy_height, "YlGn"),
        ("sensitive_mask", world.sensitive_mask, "Oranges"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, (name, arr, cmap) in zip(axes.ravel(), layers):
        im = ax.imshow(np.asarray(arr), cmap=cmap)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def m12_budget_sensitivity_v2(df, save_dir, label):
    """Sparametryzowana wersja m12_budget_sensitivity (bez globali WAW)."""
    budget_sets = [
        {"name": "synthetic_nominal", "road": 20, "sidewalk": 5, "building": 15},
        {"name": "terminal_like", "road": 30, "sidewalk": 10, "building": 20},
        {"name": "dense_city_moderate", "road": 45, "sidewalk": 20, "building": 30},
        {"name": "dense_city_relaxed", "road": 60, "sidewalk": 30, "building": 45},
        {"name": "diagnostic_relaxed", "road": 90, "sidewalk": 45, "building": 60},
    ]

    rows = []
    for b in budget_sets:
        acc = m12_posthoc_acceptability(
            df,
            road_budget_s=b["road"],
            sidewalk_budget_s=b["sidewalk"],
            building_budget_s=b["building"],
        )
        rows.append({
            "label": label,
            "budget_name": b["name"],
            "road_budget_s": b["road"],
            "sidewalk_budget_s": b["sidewalk"],
            "building_budget_s": b["building"],
            "acceptability_rate": float(acc.mean()),
            "success_rate": float(df["is_success"].astype(bool).mean()),
            "mean_road_time": float(df["road_time"].mean()),
            "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
            "mean_building_time": float(df["building_time"].mean()),
            "mean_green_time": float(df["green_time"].mean()),
            "mean_steps": float(df["steps"].mean()),
            "mean_dist_xy": float(df["dist_xy"].mean()),
        })

    out = pd.DataFrame(rows)
    path = Path(save_dir) / f"{label}_budget_sensitivity.csv"
    out.to_csv(path, index=False)
    print("Saved:", path)
    display(out)
    return out

def run_m12_planner_lambda_agl_sweep_v2(
    world,
    drive_dirs,
    aoi_id,
    save_dir,
    lambdas=(0.35, 1.0, 3.0, 8.0),
    agls=(30.0, 40.0, 60.0, 80.0),
    n_episodes=50,
    seed0=7000,
    stage="B",
):
    rows = []

    for agl in agls:
        for lam in lambdas:
            cfg = M10RecedingHorizonPlannerConfig(
                plan_res_m=5.0,
                planner_agl_m=float(agl),
                risk_lambda=float(lam),
                use_astar=True,

                sensing_radius_m=75.0,
                replan_every_steps=5,
                prior_risk_mode="perfect",
                hard_blockers_known=True,

                waypoint_radius_m=8.0,
                target_speed_mps=12.0,
                max_steps=1000,
            )

            prefix = (
                f"m12_real_{aoi_id}_m10_sweep_"
                f"agl{agl}_lambda{lam}_stage{stage}_n{n_episodes}"
            )

            df_s, summary_s, csv_s, json_s = evaluate_m10_receding_planner_drive(
                world=world,
                drive_dirs=drive_dirs,
                cfg=cfg,
                risk_profile="balanced",
                n_episodes=n_episodes,
                seed0=seed0,
                stage=stage,
                mode="strict_eval",
                n_obstacles=0,
                prefix=prefix,
                save_paths=False,
            )

            rows.append({
                "planner_agl_m": float(agl),
                "risk_lambda": float(lam),
                "stage": stage,
                "n_episodes": int(n_episodes),

                "success_rate": summary_s["success_rate"],
                "mission_acceptability_rate": summary_s["mission_acceptability_rate"],
                "acceptable_given_success_rate": summary_s["acceptable_given_success_rate"],

                "mean_steps": summary_s["mean_steps"],
                "mean_dist_xy": summary_s["mean_dist_xy"],
                "mean_urban_risk": summary_s["mean_urban_risk"],
                "mean_cumulative_risk": summary_s["mean_cumulative_risk"],

                "mean_road_time": summary_s["mean_road_time"],
                "mean_sidewalk_time": summary_s["mean_sidewalk_time"],
                "mean_building_time": summary_s["mean_building_time"],
                "mean_green_time": summary_s["mean_green_time"],
                "mean_canopy_time": summary_s["mean_canopy_time"],
                "mean_sensitive_time": summary_s["mean_sensitive_time"],
                "mean_hard_violation_steps": summary_s["mean_hard_violation_steps"],

                "term_success": summary_s["termination_counts"].get("success", 0),
                "term_stuck": summary_s["termination_counts"].get("stuck", 0),
                "term_shield_stuck": summary_s["termination_counts"].get("shield_stuck", 0),
                "term_sidewalk_budget": summary_s["termination_counts"].get("urban_sidewalk_budget_exceeded", 0),
                "term_building_budget": summary_s["termination_counts"].get("urban_building_budget_exceeded", 0),

                "csv_path": csv_s,
                "summary_path": json_s,
            })

    sweep_df = pd.DataFrame(rows)
    out_path = Path(save_dir) / f"m12_{aoi_id}_m10_lambda_agl_sweep.csv"
    sweep_df.to_csv(out_path, index=False)
    print("Saved:", out_path)
    display(sweep_df.sort_values(
        ["mission_acceptability_rate", "success_rate"], ascending=False
    ))
    return sweep_df

def _get_summary(global_name, fallback_json_path=None):
    if global_name in globals():
        return globals()[global_name]
    if fallback_json_path is not None and Path(fallback_json_path).exists():
        with open(fallback_json_path, "r") as f:
            return json.load(f)
    return None

def _coverage_row(world, label):
    return {
        "aoi": label,
        "building": float(np.asarray(world.building_mask).mean()),
        "road": float(np.asarray(world.road_mask).mean()),
        "sidewalk": float(np.asarray(world.sidewalk_mask).mean()),
        "green": float(np.asarray(world.green_mask).mean()),
        "canopy": float(np.asarray(world.canopy_mask).mean()),
        "sensitive": float(np.asarray(world.sensitive_mask).mean()),
    }

def _summary_row(summary, label):
    if summary is None:
        return {"aoi": label, "note": "summary unavailable"}
    keys = [
        "success_rate", "mission_acceptability_rate",
        "acceptable_given_success_rate",
        "mean_steps", "mean_dist_xy",
        "mean_urban_risk", "mean_cumulative_risk",
        "mean_road_time", "mean_sidewalk_time", "mean_building_time",
        "mean_green_time", "mean_canopy_time", "mean_sensitive_time",
        "mean_hard_violation_steps",
    ]
    row = {"aoi": label}
    for k in keys:
        row[k] = summary.get(k, None)
    return row

def _get_first_available_summary(*names):
    for name in names:
        if name in globals():
            return globals()[name]
    return None

from pathlib import Path

from datetime import datetime

import json

import numpy as np

import pandas as pd

M12C_AOI_ID = "vienna_prater_leopoldstadt_footway_only"

M12C_EXPERIMENT_NAME = "m12C_real_vienna_footway_learning"

def _safe_set_attr_m12c(obj, key, value):
    cls_attr = getattr(type(obj), key, None)
    if isinstance(cls_attr, property) and cls_attr.fset is None:
        return False
    try:
        setattr(obj, key, value)
        return True
    except Exception:
        return False

def load_urban_world_layers_compat_m12c(npz_path, verbose=False):
    """
    Robust loader for manually saved UrbanWorldLayers NPZ.

    First tries UrbanWorldLayers.load_npz().
    If that fails or misses compatibility fields, uses __new__ and patches.
    """
    npz_path = Path(npz_path)

    try:
        world = UrbanWorldLayers.load_npz(npz_path)
        world = patch_urban_world_layers_for_ruscuav(world, verbose=verbose)
        return world
    except Exception as e:
        if verbose:
            print("[load compat] UrbanWorldLayers.load_npz failed:", repr(e))

    data = np.load(npz_path, allow_pickle=True)

    obj = UrbanWorldLayers.__new__(UrbanWorldLayers)

    for key in data.files:
        val = data[key]

        if key in ["aoi_id", "crs"]:
            try:
                val = str(val.item())
            except Exception:
                val = str(val)

        elif key == "bbox":
            val = tuple(float(x) for x in np.asarray(val).ravel())

        elif key == "resolution_m":
            val = float(np.asarray(val).item())

        else:
            if isinstance(val, np.ndarray):
                val = val.astype(np.float32) if val.dtype.kind in ["f", "i", "u", "b"] else val

        _safe_set_attr_m12c(obj, key, val)

    obj = patch_urban_world_layers_for_ruscuav(obj, verbose=verbose)

    return obj

def make_m12c_real_vienna_env_mode(
    world,
    policy_kind="reward_only",
    stage="C",
    mode="strict_eval",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
    verbose=False,
):
    """
    policy_kind:
        reward_only   -> no static semantic vector, static risk reward active
        geometry_only -> no static semantic vector, no static risk reward

    Both keep:
        urban LiDAR,
        action shield,
        geometric shield,
        altitude reference,
        strict evaluation constraints when stage='C'.
    """
    policy_kind = str(policy_kind).lower()
    stage = str(stage).upper()
    mode = str(mode).lower()

    if policy_kind not in ["reward_only", "geometry_only"]:
        raise ValueError("policy_kind must be 'reward_only' or 'geometry_only'.")

    if mode == "fast_train":
        urban_lidar_step_m = 5.0
        urban_segment_check_step_m = 2.0
    elif mode == "strict_eval":
        urban_lidar_step_m = 3.0
        urban_segment_check_step_m = 1.0
    else:
        raise ValueError("mode must be 'fast_train' or 'strict_eval'.")

    if stage == "A":
        urban_risk_lambda = 0.05
        terminate_on_urban_constraint = False
        hard_sensitive_zones = False

    elif stage == "B":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = False
        hard_sensitive_zones = True

    elif stage == "C":
        urban_risk_lambda = 0.10
        terminate_on_urban_constraint = True
        hard_sensitive_zones = True

    else:
        raise ValueError("stage must be A, B, or C.")

    use_risk_costs = bool(policy_kind == "reward_only")
    green_bonus_lambda = 0.03 if use_risk_costs else 0.0

    cfg = RealisticUrbanEnvConfig(
        risk_profile=risk_profile,

        include_urban_vector_obs=False,
        use_urban_risk_costs=use_risk_costs,
        urban_risk_lambda=float(urban_risk_lambda),
        green_bonus_lambda=float(green_bonus_lambda),

        use_urban_altitude_reference=True,

        terminate_on_urban_constraint=bool(terminate_on_urban_constraint),
        hard_sensitive_zones=bool(hard_sensitive_zones),

        default_n_obstacles=int(n_obstacles),

        mission_accept_cargo_min=0.80,
        mission_accept_road_budget_s=20.0,
        mission_accept_sidewalk_budget_s=5.0,
        mission_accept_building_budget_s=15.0,
        mission_accept_sensitive_budget_s=0.0,

        terminal_road_budget_s=30.0,
        terminal_sidewalk_budget_s=10.0,
        terminal_building_budget_s=20.0,
        terminal_sensitive_budget_s=0.0,
    )

    env = RealisticUrbanCargoUAVEnvM4(
        world=world,
        config=cfg,
        include_shield_features=False,

        urban_lidar_enabled=True,
        urban_action_shield_enabled=True,
        urban_geometric_shield_enabled=True,

        urban_lidar_step_m=urban_lidar_step_m,
        urban_segment_check_step_m=urban_segment_check_step_m,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        if verbose:
            print(
                f"[M12C env] policy={policy_kind}, stage={stage}, mode={mode}, "
                f"obs.shape={obs.shape}, observation_space={env.observation_space}, info={info}"
            )

    return env

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

def make_m12c_vec_env_from_world_path(
    world_path,
    policy_kind,
    stage,
    mode,
    rank,
    base_seed=6001,
    n_obstacles=0,
):
    def _init():
        w = load_urban_world_layers_compat_m12c(world_path, verbose=False)

        env = make_m12c_real_vienna_env_mode(
            world=w,
            policy_kind=policy_kind,
            stage=stage,
            mode=mode,
            risk_profile="balanced",
            n_obstacles=n_obstacles,
            seed=base_seed + rank,
            verbose=False,
        )

        env.reset(seed=base_seed + rank)

        return env

    return _init

def make_vec_m12c_real_vienna_env(
    world_path,
    policy_kind="reward_only",
    stage="A",
    mode="fast_train",
    n_envs=4,
    base_seed=6001,
    n_obstacles=0,
    use_subproc=True,
    start_method="fork",
):
    env_fns = [
        make_m12c_vec_env_from_world_path(
            world_path=world_path,
            policy_kind=policy_kind,
            stage=stage,
            mode=mode,
            rank=i,
            base_seed=base_seed,
            n_obstacles=n_obstacles,
        )
        for i in range(int(n_envs))
    ]

    if use_subproc and int(n_envs) > 1:
        vec_env = SubprocVecEnv(env_fns, start_method=start_method)
    else:
        vec_env = DummyVecEnv(env_fns)

    vec_env = VecMonitor(
        vec_env,
        info_keywords=RUSCUAV_M4_INFO_KEYWORDS,
    )

    return vec_env

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CallbackList

def _m12c_stage_name(policy_kind, stage):
    return f"M12C_vienna_{policy_kind}_{str(stage).upper()}"

def create_or_load_m12c_model_for_stage(
    policy_kind,
    stage,
    vec_env,
    drive_dirs,
    previous_model_path=None,
    seed=6001,
    device="cpu",
    n_envs=4,
):
    stage = str(stage).upper()
    stage_name = _m12c_stage_name(policy_kind, stage)
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    model, loaded_from = load_stage_model_if_available(paths, env=vec_env, device=device)

    if model is not None:
        print(f"Resuming M12C {policy_kind} stage {stage} from:", loaded_from)
        return model, loaded_from

    if previous_model_path is not None and Path(previous_model_path).exists():
        print(f"Starting M12C {policy_kind} stage {stage} from previous model:", previous_model_path)
        model = PPO.load(str(previous_model_path), env=vec_env, device=device)
        return model, str(previous_model_path)

    print(f"Starting M12C {policy_kind} stage {stage} from scratch.")

    n_steps = 512 if int(n_envs) >= 4 else 1024

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        seed=int(seed),
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
    )

    return model, None

def train_one_m12c_stage_resume_safe(
    world_path,
    policy_kind,
    stage,
    target_stage_steps,
    drive_dirs,
    seed=6001,
    n_obstacles=0,
    previous_model_path=None,
    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    stage = str(stage).upper()
    stage_name = _m12c_stage_name(policy_kind, stage)
    paths = stage_paths_on_drive(drive_dirs, stage_name)

    if is_stage_done(paths, target_stage_steps):
        print(f"M12C {policy_kind} stage {stage} already DONE. Skipping.")
        return str(paths["final"])

    vec_env = make_vec_m12c_real_vienna_env(
        world_path=world_path,
        policy_kind=policy_kind,
        stage=stage,
        mode=mode,
        n_envs=n_envs,
        base_seed=seed,
        n_obstacles=n_obstacles,
        use_subproc=(n_envs > 1),
        start_method=start_method,
    )

    model, loaded_from = create_or_load_m12c_model_for_stage(
        policy_kind=policy_kind,
        stage=stage,
        vec_env=vec_env,
        drive_dirs=drive_dirs,
        previous_model_path=previous_model_path,
        seed=seed,
        device=device,
        n_envs=n_envs,
    )

    progress = read_progress_json(paths["progress"])

    if progress is not None and Path(paths["latest"]).exists():
        stage_start_timesteps = int(progress["stage_start_timesteps"])
        completed_stage_steps = int(progress["completed_stage_steps"])
        print(f"Read M12C progress {stage_name}: completed={completed_stage_steps}/{target_stage_steps}")
    else:
        stage_start_timesteps = int(model.num_timesteps)
        completed_stage_steps = 0
        print(f"New M12C progress {stage_name}: stage_start_timesteps={stage_start_timesteps}")

    remaining = int(max(0, target_stage_steps - completed_stage_steps))

    print("=" * 100)
    print(f"TRAIN M12C REAL VIENNA | policy={policy_kind} | STAGE={stage}")
    print("=" * 100)
    print("target_stage_steps:", target_stage_steps)
    print("completed_stage_steps:", completed_stage_steps)
    print("remaining:", remaining)
    print("model.num_timesteps:", model.num_timesteps)
    print("stage_start_timesteps:", stage_start_timesteps)
    print("world_path:", world_path)
    print("stage dir:", paths["stage_dir"])
    print("=" * 100)

    if remaining <= 0:
        final_path = mark_stage_done(
            paths,
            stage_name=stage_name,
            model=model,
            completed_stage_steps=completed_stage_steps,
            target_stage_steps=target_stage_steps,
        )
        vec_env.close()
        return final_path

    autosave_cb = RUSCUAVDriveAutosaveCallback(
        stage_dir=paths["stage_dir"],
        stage_name=stage_name,
        target_stage_steps=target_stage_steps,
        stage_start_timesteps=stage_start_timesteps,
        save_freq_steps=save_freq_steps,
        keep_numbered_checkpoints=True,
        verbose=1,
    )

    stats_cb = RUSCUAVTrainStatsCallback(
        print_every_steps=10_000,
        window=100,
    )

    callback = CallbackList([autosave_cb, stats_cb])
    reset_num_timesteps = bool(model.num_timesteps == 0)

    try:
        model.learn(
            total_timesteps=int(remaining),
            callback=callback,
            reset_num_timesteps=reset_num_timesteps,
        )
    finally:
        model.save(str(paths["latest"]))

        progress_now = {
            "stage": stage_name,
            "policy_kind": policy_kind,
            "reason": "finally_block",
            "target_stage_steps": int(target_stage_steps),
            "stage_start_timesteps": int(stage_start_timesteps),
            "last_model_timesteps": int(model.num_timesteps),
            "completed_stage_steps": int(max(0, model.num_timesteps - stage_start_timesteps)),
            "saved_at": datetime.now().isoformat(),
            "latest_model_path": str(paths["latest"]),
            "mode": mode,
            "n_envs": int(n_envs),
            "world_path": str(world_path),
        }

        atomic_write_json(paths["progress"], progress_now)
        print("Emergency/finally save completed:", paths["latest"])

    completed_final = int(max(0, model.num_timesteps - stage_start_timesteps))

    final_path = mark_stage_done(
        paths,
        stage_name=stage_name,
        model=model,
        completed_stage_steps=completed_final,
        target_stage_steps=target_stage_steps,
    )

    vec_env.close()

    return final_path

def train_m12c_real_vienna_policy_drive_resume(
    world_path,
    policy_kind,
    drive_dirs,
    seed=6001,
    n_obstacles=0,

    steps_A=50_000,
    steps_B=75_000,
    steps_C=75_000,

    initial_model_path=None,
    save_freq_steps=10_000,
    device="cpu",
    n_envs=4,
    mode="fast_train",
    start_method="fork",
):
    """
    Fine-tune or train a policy on Vienna footway-only real AOI.

    For reward_only:
        recommended initial_model_path = m8_reward_only_model_path

    For geometry_only:
        recommended initial_model_path = m7_geometry_only_model_path if available,
        otherwise None -> from scratch.
    """
    policy_kind = str(policy_kind).lower()

    if policy_kind not in ["reward_only", "geometry_only"]:
        raise ValueError("policy_kind must be 'reward_only' or 'geometry_only'.")

    protocol = {
        "experiment": "M12C real Vienna footway-only learning",
        "policy_kind": policy_kind,
        "world_path": str(world_path),
        "seed": int(seed),
        "n_obstacles": int(n_obstacles),
        "steps_A": int(steps_A),
        "steps_B": int(steps_B),
        "steps_C": int(steps_C),
        "initial_model_path": str(initial_model_path) if initial_model_path is not None else None,
        "device": device,
        "n_envs": int(n_envs),
        "mode": mode,
        "start_method": start_method,
        "started_or_resumed_at": datetime.now().isoformat(),
    }

    protocol_path = Path(drive_dirs["metadata"]) / f"m12c_{policy_kind}_protocol.json"
    atomic_write_json(protocol_path, protocol)

    final_A = train_one_m12c_stage_resume_safe(
        world_path=world_path,
        policy_kind=policy_kind,
        stage="A",
        target_stage_steps=int(steps_A),
        drive_dirs=drive_dirs,
        seed=seed,
        n_obstacles=n_obstacles,
        previous_model_path=initial_model_path,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_B = train_one_m12c_stage_resume_safe(
        world_path=world_path,
        policy_kind=policy_kind,
        stage="B",
        target_stage_steps=int(steps_B),
        drive_dirs=drive_dirs,
        seed=seed + 100,
        n_obstacles=n_obstacles,
        previous_model_path=final_A,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    final_C = train_one_m12c_stage_resume_safe(
        world_path=world_path,
        policy_kind=policy_kind,
        stage="C",
        target_stage_steps=int(steps_C),
        drive_dirs=drive_dirs,
        seed=seed + 200,
        n_obstacles=n_obstacles,
        previous_model_path=final_B,
        save_freq_steps=save_freq_steps,
        device=device,
        n_envs=n_envs,
        mode=mode,
        start_method=start_method,
    )

    return {
        "final_A": final_A,
        "final_B": final_B,
        "final_C": final_C,
    }

from stable_baselines3 import PPO

def rollout_m12c_real_policy_episode(
    model,
    world,
    policy_kind,
    seed,
    stage="C",
    mode="strict_eval",
    n_obstacles=0,
    deterministic=True,
    max_steps=1000,
):
    env = make_m12c_real_vienna_env_mode(
        world=world,
        policy_kind=policy_kind,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=n_obstacles,
        seed=seed,
        verbose=False,
    )

    obs, reset_info = env.reset(seed=seed)

    rewards = []
    infos = []
    path = [env.current_pos.copy()]

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)

        obs, reward, terminated, truncated, info = env.step(action)

        rewards.append(float(reward))
        infos.append(dict(info))
        path.append(env.current_pos.copy())

        done = bool(terminated or truncated)
        steps += 1

    final_info = infos[-1] if infos else {}

    if "dist_xy" not in final_info or final_info.get("dist_xy") is None:
        try:
            final_info["dist_xy"] = float(np.linalg.norm(env.target_point[:2] - env.current_pos[:2]))
        except Exception:
            final_info["dist_xy"] = np.nan

    row = {
        "episode": None,
        "seed": int(seed),
        "policy_kind": policy_kind,

        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": final_info.get("termination", None),
        "is_success": bool(final_info.get("is_success", False)),
        "mission_acceptable": bool(final_info.get("mission_acceptable", False)),

        "dist_xy": float(final_info.get("dist_xy", np.nan)),
        "cargo_health": float(final_info.get("cargo_health", np.nan)),

        "urban_mean_risk": float(final_info.get("urban_mean_risk", np.nan)),
        "urban_max_risk": float(final_info.get("urban_max_risk", np.nan)),
        "urban_cumulative_risk": float(final_info.get("urban_cumulative_risk", np.nan)),

        "road_time": float(final_info.get("urban_road_time", 0.0)),
        "sidewalk_time": float(final_info.get("urban_sidewalk_time", 0.0)),
        "building_time": float(final_info.get("urban_building_time", 0.0)),
        "green_time": float(final_info.get("urban_green_time", 0.0)),
        "canopy_time": float(final_info.get("urban_canopy_time", 0.0)),
        "sensitive_time": float(final_info.get("urban_sensitive_time", 0.0)),

        "building_collision_steps": int(final_info.get("urban_building_collision_steps", 0)),
        "canopy_collision_steps": int(final_info.get("urban_canopy_collision_steps", 0)),
        "sensitive_low_altitude_violation_steps": int(final_info.get("urban_sensitive_low_altitude_violation_steps", 0)),
        "urban_hard_violation_steps": int(final_info.get("urban_hard_violation_steps", 0)),
    }

    env.close()

    return row, np.asarray(path, dtype=np.float32)

def summarize_m12c_real_policy_eval_df(df):
    if len(df) == 0:
        return {"n_episodes": 0}

    summary = {
        "n_episodes": int(len(df)),
        "success_rate": float(df["is_success"].mean()),
        "mission_acceptability_rate": float(df["mission_acceptable"].mean()),

        "mean_reward": float(df["total_reward"].mean()),
        "std_reward": float(df["total_reward"].std(ddof=0)),

        "mean_steps": float(df["steps"].mean()),
        "mean_dist_xy": float(df["dist_xy"].mean()),
        "median_dist_xy": float(df["dist_xy"].median()),

        "mean_cargo_health": float(df["cargo_health"].mean()),

        "mean_urban_risk": float(df["urban_mean_risk"].mean()),
        "mean_urban_max_risk": float(df["urban_max_risk"].mean()),
        "mean_cumulative_risk": float(df["urban_cumulative_risk"].mean()),

        "mean_road_time": float(df["road_time"].mean()),
        "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
        "mean_building_time": float(df["building_time"].mean()),
        "mean_green_time": float(df["green_time"].mean()),
        "mean_canopy_time": float(df["canopy_time"].mean()),
        "mean_sensitive_time": float(df["sensitive_time"].mean()),

        "mean_hard_violation_steps": float(df["urban_hard_violation_steps"].mean()),

        "termination_counts": {
            str(k): int(v)
            for k, v in df["termination"].fillna("None").value_counts().to_dict().items()
        },
    }

    success_df = df[df["is_success"] == True]

    if len(success_df) > 0:
        summary["acceptable_given_success_rate"] = float(success_df["mission_acceptable"].mean())
    else:
        summary["acceptable_given_success_rate"] = 0.0

    return summary

def print_m12c_real_policy_summary(summary):
    print("=" * 100)
    print("M12C REAL-AOI RL POLICY EVALUATION SUMMARY")
    print("=" * 100)

    for k, v in summary.items():
        if k == "termination_counts":
            print("termination_counts:")
            for kk, vv in v.items():
                print(f"  {kk}: {vv}")
        elif isinstance(v, float):
            print(f"{k:45s}: {v:.6f}")
        else:
            print(f"{k:45s}: {v}")

    print("=" * 100)

def evaluate_m12c_real_policy_drive(
    model_path,
    world,
    policy_kind,
    drive_dirs,
    n_episodes=200,
    seed0=7000,
    prefix=None,
    stage="C",
    mode="strict_eval",
    deterministic=True,
    save_paths=False,
):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if prefix is None:
        prefix = f"m12c_vienna_{policy_kind}_strict_eval_{n_episodes}"

    eval_dir = Path(drive_dirs["eval"]) / prefix
    eval_dir.mkdir(parents=True, exist_ok=True)

    paths_dir = eval_dir / "paths"
    if save_paths:
        paths_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eval_dir / f"{prefix}_episodes.csv"
    summary_path = eval_dir / f"{prefix}_summary.json"

    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        rows = df_existing.to_dict("records")
        completed = set(int(e) for e in df_existing["episode"].dropna().astype(int).tolist())
        print(f"Resuming M12C eval {policy_kind}: {len(completed)}/{n_episodes}")
    else:
        rows = []
        completed = set()

    model = PPO.load(str(model_path), device="cpu")

    for ep in range(int(n_episodes)):
        if ep in completed:
            continue

        seed = int(seed0 + ep)

        row, path = rollout_m12c_real_policy_episode(
            model=model,
            world=world,
            policy_kind=policy_kind,
            seed=seed,
            stage=stage,
            mode=mode,
            n_obstacles=0,
            deterministic=deterministic,
            max_steps=1000,
        )

        row["episode"] = int(ep)
        rows.append(row)

        if save_paths:
            np.save(paths_dir / f"episode_{ep:05d}_seed_{seed}_path.npy", path)

        df_now = pd.DataFrame(rows).sort_values("episode")
        df_now.to_csv(csv_path, index=False)

        summary_now = summarize_m12c_real_policy_eval_df(df_now)
        summary_now["status"] = "partial" if len(df_now) < int(n_episodes) else "complete"
        summary_now["updated_at"] = datetime.now().isoformat()
        summary_now["policy_kind"] = policy_kind
        summary_now["model_path"] = str(model_path)

        atomic_write_json(summary_path, summary_now)

        if (ep + 1) % 10 == 0 or ep == 0:
            print(
                f"[M12C eval {policy_kind}] ep={ep+1}/{n_episodes} | "
                f"SR={summary_now['success_rate']:.1%} | "
                f"MAR={summary_now['mission_acceptability_rate']:.1%} | "
                f"term={row['termination']}"
            )

    df_final = pd.DataFrame(rows).sort_values("episode")
    summary = summarize_m12c_real_policy_eval_df(df_final)

    summary["status"] = "complete"
    summary["finished_at"] = datetime.now().isoformat()
    summary["policy_kind"] = policy_kind
    summary["model_path"] = str(model_path)
    summary["csv_path"] = str(csv_path)
    summary["summary_path"] = str(summary_path)

    df_final.to_csv(csv_path, index=False)
    atomic_write_json(summary_path, summary)

    print_m12c_real_policy_summary(summary)

    return df_final, summary, str(csv_path), str(summary_path)

def m12c_summary_row(label, summary, method_type, notes=""):
    return {
        "label": label,
        "method_type": method_type,
        "notes": notes,

        "success_rate": summary.get("success_rate", np.nan),
        "mission_acceptability_rate": summary.get("mission_acceptability_rate", np.nan),
        "acceptable_given_success_rate": summary.get("acceptable_given_success_rate", np.nan),

        "mean_reward": summary.get("mean_reward", np.nan),
        "std_reward": summary.get("std_reward", np.nan),

        "mean_steps": summary.get("mean_steps", np.nan),
        "mean_dist_xy": summary.get("mean_dist_xy", np.nan),
        "median_dist_xy": summary.get("median_dist_xy", np.nan),
        "mean_cargo_health": summary.get("mean_cargo_health", np.nan),

        "mean_urban_risk": summary.get("mean_urban_risk", np.nan),
        "mean_cumulative_risk": summary.get("mean_cumulative_risk", np.nan),

        "mean_road_time": summary.get("mean_road_time", np.nan),
        "mean_sidewalk_time": summary.get("mean_sidewalk_time", np.nan),
        "mean_building_time": summary.get("mean_building_time", np.nan),
        "mean_green_time": summary.get("mean_green_time", np.nan),
        "mean_canopy_time": summary.get("mean_canopy_time", np.nan),
        "mean_sensitive_time": summary.get("mean_sensitive_time", np.nan),
        "mean_hard_violation_steps": summary.get("mean_hard_violation_steps", np.nan),

        "term_success": summary.get("termination_counts", {}).get("success", 0),
        "term_stuck": summary.get("termination_counts", {}).get("stuck", 0),
        "term_shield_stuck": summary.get("termination_counts", {}).get("shield_stuck", 0),
        "term_planner_no_path": summary.get("termination_counts", {}).get("planner_no_path", 0),
        "term_sidewalk_budget": summary.get("termination_counts", {}).get("urban_sidewalk_budget_exceeded", 0),
        "term_road_budget": summary.get("termination_counts", {}).get("urban_road_budget_exceeded", 0),
        "term_building_budget": summary.get("termination_counts", {}).get("urban_building_budget_exceeded", 0),
    }

from scipy.stats import binomtest

def paired_binom_p(a_only, b_only):
    n = int(a_only + b_only)
    if n == 0:
        return 1.0
    return float(
        binomtest(
            k=min(int(a_only), int(b_only)),
            n=n,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )

def paired_compare_m12c(df_a, df_b, label_a, label_b):
    a = df_a.copy().add_prefix("a_")
    b = df_b.copy().add_prefix("b_")

    paired = a.merge(
        b,
        left_on="a_seed",
        right_on="b_seed",
        how="inner",
    )

    summary = {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": int(len(paired)),

        f"{label_a}_success_{label_b}_fail": int(
            ((paired["a_is_success"] == True) &
             (paired["b_is_success"] == False)).sum()
        ),
        f"{label_b}_success_{label_a}_fail": int(
            ((paired["b_is_success"] == True) &
             (paired["a_is_success"] == False)).sum()
        ),

        f"{label_a}_accept_{label_b}_not": int(
            ((paired["a_mission_acceptable"] == True) &
             (paired["b_mission_acceptable"] == False)).sum()
        ),
        f"{label_b}_accept_{label_a}_not": int(
            ((paired["b_mission_acceptable"] == True) &
             (paired["a_mission_acceptable"] == False)).sum()
        ),

        f"mean_delta_steps_{label_a}_minus_{label_b}": float(
            (paired["a_steps"] - paired["b_steps"]).mean()
        ),
        f"mean_delta_cumulative_risk_{label_a}_minus_{label_b}": float(
            (paired["a_urban_cumulative_risk"] - paired["b_urban_cumulative_risk"]).mean()
        ),
        f"mean_delta_road_time_{label_a}_minus_{label_b}": float(
            (paired["a_road_time"] - paired["b_road_time"]).mean()
        ),
        f"mean_delta_sidewalk_time_{label_a}_minus_{label_b}": float(
            (paired["a_sidewalk_time"] - paired["b_sidewalk_time"]).mean()
        ),
        f"mean_delta_green_time_{label_a}_minus_{label_b}": float(
            (paired["a_green_time"] - paired["b_green_time"]).mean()
        ),
    }

    a_succ = summary[f"{label_a}_success_{label_b}_fail"]
    b_succ = summary[f"{label_b}_success_{label_a}_fail"]

    a_acc = summary[f"{label_a}_accept_{label_b}_not"]
    b_acc = summary[f"{label_b}_accept_{label_a}_not"]

    pvals = {
        "success_p": paired_binom_p(a_succ, b_succ),
        "acceptability_p": paired_binom_p(a_acc, b_acc),
    }

    return paired, summary, pvals

def m12c_policy_budget_sensitivity(df, label):
    budget_sets = [
        {"name": "synthetic_nominal", "road": 20, "sidewalk": 5, "building": 15},
        {"name": "terminal_like", "road": 30, "sidewalk": 10, "building": 20},
        {"name": "dense_city_moderate", "road": 45, "sidewalk": 20, "building": 30},
        {"name": "dense_city_relaxed", "road": 60, "sidewalk": 30, "building": 45},
    ]

    rows = []

    for b in budget_sets:
        acc = (
            (df["is_success"].astype(bool) == True)
            & (df["cargo_health"].astype(float) >= 0.80)
            & (df["road_time"].astype(float) <= b["road"])
            & (df["sidewalk_time"].astype(float) <= b["sidewalk"])
            & (df["building_time"].astype(float) <= b["building"])
            & (df["sensitive_time"].astype(float) <= 0.0)
            & (df["urban_hard_violation_steps"].astype(float) == 0.0)
        )

        rows.append({
            "label": label,
            "budget_name": b["name"],
            "road_budget_s": b["road"],
            "sidewalk_budget_s": b["sidewalk"],
            "building_budget_s": b["building"],
            "posthoc_acceptability_rate": float(acc.mean()),
            "success_rate": float(df["is_success"].mean()),
            "mean_road_time": float(df["road_time"].mean()),
            "mean_sidewalk_time": float(df["sidewalk_time"].mean()),
            "mean_building_time": float(df["building_time"].mean()),
            "mean_green_time": float(df["green_time"].mean()),
            "mean_cumulative_risk": float(df["urban_cumulative_risk"].mean()),
        })

    return pd.DataFrame(rows)

def termination_table_from_dfs(items):
    rows = []

    for label, df in items:
        vc = df["termination"].fillna("None").value_counts().to_dict()
        row = {"label": label, "n": len(df)}
        for k, v in vc.items():
            row[f"term_{k}"] = int(v)
        rows.append(row)

    out = pd.DataFrame(rows).fillna(0)

    return out

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

import json

def get_vienna_planner_eval_df():
    if "df_m12c_planner_footway_200" in globals():
        return df_m12c_planner_footway_200

    if "df_m12c_planner_200" in globals():
        return df_m12c_planner_200

    if "df_vienna_m10_footway_only" in globals():
        return df_vienna_m10_footway_only

    raise RuntimeError(
        "No Vienna planner evaluation dataframe found. "
        "Expected df_m12c_planner_footway_200, df_m12c_planner_200, or df_vienna_m10_footway_only."
    )

def pick_vienna_visualization_seed(
    df,
    prefer="accepted_success",
    sort_by="dist_xy",
    ascending=True,
):
    d = df.copy()

    if prefer == "accepted_success":
        if "mission_acceptable" in d.columns:
            d = d[(d["is_success"] == True) & (d["mission_acceptable"] == True)]
        else:
            d = d[d["is_success"] == True]

    elif prefer == "success":
        d = d[d["is_success"] == True]

    elif prefer == "nonaccepted_success":
        d = d[(d["is_success"] == True) & (d["mission_acceptable"] == False)]

    elif prefer == "failure":
        d = d[d["is_success"] == False]

    else:
        raise ValueError("prefer must be accepted_success, success, nonaccepted_success, or failure")

    if len(d) == 0:
        raise ValueError(f"No episode found for prefer={prefer}")

    if sort_by in d.columns:
        d = d.sort_values(sort_by, ascending=ascending)

    row = d.iloc[0]

    return int(row["seed"]), int(row["episode"])

def _mask_value_at(world, mask_name, x, y):
    arr = np.asarray(getattr(world, mask_name), dtype=np.float32)
    h, w = arr.shape
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))
    return float(arr[yi, xi])

def _ground_at_world(world, x, y):
    arr = np.asarray(world.terrain_z, dtype=np.float32)
    h, w = arr.shape
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))
    return float(arr[yi, xi])

def rollout_vienna_receding_planner_with_telemetry(
    world,
    seed,
    stage="C",
    mode="strict_eval",
    max_steps=1000,
):
    cfg = M10RecedingHorizonPlannerConfig(
        plan_res_m=5.0,
        planner_agl_m=30.0,
        risk_lambda=0.35,
        use_astar=True,

        sensing_radius_m=75.0,
        replan_every_steps=5,
        prior_risk_mode="perfect",
        hard_blockers_known=True,

        waypoint_radius_m=8.0,
        target_speed_mps=12.0,
        max_steps=max_steps,
    )

    env = make_ruscuav_env_m4_mode(
        world=world,
        stage=stage,
        mode=mode,
        risk_profile="balanced",
        n_obstacles=0,
        seed=seed,
    )

    obs, reset_info = env.reset(seed=seed)

    planner = M10RecedingHorizonRiskPlanner(
        world=world,
        config=cfg,
        risk_profile="balanced",
    )

    planner.reset_belief()
    planner.reveal_around_xy(env.current_pos[:2])

    target_xy = env.target_point[:2].copy()

    current_plan = None
    planned_paths = []
    wp_idx = 0

    path = [env.current_pos.copy()]
    rows = []
    rewards = []

    done = False
    steps = 0
    replan_success_count = 0
    replan_fail_count = 0

    while not done and steps < int(max_steps):
        need_replan = False

        if current_plan is None:
            need_replan = True

        if steps % int(max(1, cfg.replan_every_steps)) == 0:
            need_replan = True

        if need_replan:
            planner.reveal_around_xy(env.current_pos[:2])

            plan_result = planner.plan(
                start_xy=env.current_pos[:2].copy(),
                target_xy=target_xy,
            )

            if plan_result["success"]:
                current_plan = plan_result["path_xy"]
                planned_paths.append(current_plan.copy())
                wp_idx = 1 if len(current_plan) > 1 else 0
                replan_success_count += 1
            else:
                replan_fail_count += 1
                if current_plan is None:
                    break

        action, wp_idx = m9_waypoint_controller_action(
            env=env,
            path_xy=current_plan,
            wp_idx=wp_idx,
            cfg=_m10_to_m9_config(cfg),
        )

        obs, reward, terminated, truncated, info = env.step(action)

        pos = env.current_pos.copy()
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        ground_z = _ground_at_world(world, x, y)

        row = {
            "step": int(steps + 1),
            "t_s": float((steps + 1) * env.dt),

            "x": x,
            "y": y,
            "z": z,
            "ground_z": ground_z,
            "agl_m": float(z - ground_z),

            "reward": float(reward),
            "termination": info.get("termination"),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "is_success": bool(info.get("is_success", False)),
            "mission_acceptable": bool(info.get("mission_acceptable", False)),

            "dist_xy": float(info.get("dist_xy", np.nan)),
            "cargo_health": float(info.get("cargo_health", np.nan)),

            "urban_risk": float(info.get("urban_risk", info.get("urban_mean_risk", np.nan))),
            "urban_mean_risk": float(info.get("urban_mean_risk", np.nan)),
            "urban_cumulative_risk": float(info.get("urban_cumulative_risk", np.nan)),

            "road_time": float(info.get("urban_road_time", 0.0)),
            "sidewalk_time": float(info.get("urban_sidewalk_time", 0.0)),
            "building_time": float(info.get("urban_building_time", 0.0)),
            "green_time": float(info.get("urban_green_time", 0.0)),
            "canopy_time": float(info.get("urban_canopy_time", 0.0)),
            "sensitive_time": float(info.get("urban_sensitive_time", 0.0)),

            "building_mask": _mask_value_at(world, "building_mask", x, y),
            "road_mask": _mask_value_at(world, "road_mask", x, y),
            "sidewalk_mask": _mask_value_at(world, "sidewalk_mask", x, y),
            "green_mask": _mask_value_at(world, "green_mask", x, y),
            "canopy_mask": _mask_value_at(world, "canopy_mask", x, y),
            "sensitive_mask": _mask_value_at(world, "sensitive_mask", x, y),

            "wp_idx": int(wp_idx),
            "replan_count": int(planner.replan_count),
            "total_expanded": int(planner.total_expanded),
            "total_plan_time_s": float(planner.total_plan_time_s),
        }

        rows.append(row)
        rewards.append(float(reward))
        path.append(pos)

        done = bool(terminated or truncated)
        steps += 1

    env.close()

    df = pd.DataFrame(rows)
    path = np.asarray(path, dtype=np.float32)

    final = rows[-1] if rows else {
        "termination": "planner_no_path",
        "is_success": False,
        "mission_acceptable": False,
    }

    rollout = {
        "seed": int(seed),
        "stage": stage,
        "mode": mode,
        "path": path,
        "planned_paths": planned_paths,
        "df": df,
        "reset_info": reset_info,
        "steps": int(steps),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "termination": final.get("termination"),
        "is_success": bool(final.get("is_success", False)),
        "mission_acceptable": bool(final.get("mission_acceptable", False)),
        "replan_success_count": int(replan_success_count),
        "replan_fail_count": int(replan_fail_count),
        "planner_replan_count": int(planner.replan_count),
        "planner_total_expanded": int(planner.total_expanded),
        "planner_total_plan_time_s": float(planner.total_plan_time_s),
    }

    return rollout

def plot_vienna_semantic_2d(world, rollout, save_path=None):
    h, w = world.terrain_z.shape

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(
        world.terrain_z,
        origin="upper",
        extent=[0, w, h, 0],
        cmap="terrain",
        alpha=0.85,
    )

    def overlay(mask, cmap, alpha):
        arr = np.ma.masked_where(np.asarray(mask) <= 0.5, mask)
        ax.imshow(
            arr,
            origin="upper",
            extent=[0, w, h, 0],
            cmap=cmap,
            alpha=alpha,
        )

    overlay(world.green_mask, "Greens", 0.30)
    overlay(world.canopy_mask, "Greens", 0.50)
    overlay(world.road_mask, "Oranges", 0.48)
    overlay(world.sidewalk_mask, "Wistia", 0.48)
    overlay(world.building_mask, "Greys", 0.60)
    overlay(world.sensitive_mask, "Reds", 0.35)

    path = rollout["path"]
    ax.plot(path[:, 0], path[:, 1], linewidth=2.5, label="actual UAV path")
    ax.scatter(path[0, 0], path[0, 1], s=80, marker="o", edgecolors="black", label="START")
    ax.scatter(path[-1, 0], path[-1, 1], s=90, marker="s", edgecolors="black", label="END")

    if rollout.get("planned_paths"):
        last_plan = rollout["planned_paths"][-1]
        ax.plot(last_plan[:, 0], last_plan[:, 1], linestyle="--", linewidth=1.5, label="last planned path")

    ax.set_title(
        f"Vienna Prater footway-only — semantic 2D\n"
        f"seed={rollout['seed']} | term={rollout['termination']} | "
        f"success={rollout['is_success']} | acceptable={rollout['mission_acceptable']}"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220)
        print("Saved:", save_path)

    plt.show()

def plot_vienna_risk_2d(world, rollout, agl_m=30.0, save_path=None):
    risk_model = UrbanRiskModel(
        world=world,
        profile="balanced",
        hard_sensitive_zones=True,
        include_global_altitude_band_cost=True,
    )

    risk = risk_model.risk_map_at_agl(float(agl_m))
    h, w = risk.shape

    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(
        risk,
        origin="upper",
        extent=[0, w, h, 0],
        cmap="inferno",
    )
    plt.colorbar(im, ax=ax, label="urban risk")

    path = rollout["path"]
    ax.plot(path[:, 0], path[:, 1], linewidth=2.5, label="actual UAV path")
    ax.scatter(path[0, 0], path[0, 1], s=80, marker="o", edgecolors="black", label="START")
    ax.scatter(path[-1, 0], path[-1, 1], s=90, marker="s", edgecolors="black", label="END")

    if rollout.get("planned_paths"):
        last_plan = rollout["planned_paths"][-1]
        ax.plot(last_plan[:, 0], last_plan[:, 1], linestyle="--", linewidth=1.5, label="last planned path")

    ax.set_title(
        f"Vienna Prater footway-only — risk map at AGL={agl_m:.0f} m\n"
        f"seed={rollout['seed']} | term={rollout['termination']} | "
        f"success={rollout['is_success']} | acceptable={rollout['mission_acceptable']}"
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220)
        print("Saved:", save_path)

    plt.show()

def plot_vienna_rollout_timeseries(rollout, save_dir):
    df = rollout["df"].copy()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    series = [
        ("dist_xy", "Distance to target [m]"),
        ("agl_m", "AGL altitude [m]"),
        ("urban_risk", "Instant urban risk"),
        ("cargo_health", "Cargo Health"),
        ("road_time", "Cumulative road time [s]"),
        ("sidewalk_time", "Cumulative sidewalk time [s]"),
        ("building_time", "Cumulative building time [s]"),
        ("green_time", "Cumulative green time [s]"),
        ("canopy_time", "Cumulative canopy time [s]"),
    ]

    for col, title in series:
        if col not in df.columns:
            continue

        plt.figure(figsize=(9, 4))
        plt.plot(df["t_s"], df[col])
        plt.xlabel("Time [s]")
        plt.ylabel(col)
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        out = save_dir / f"timeseries_{col}_seed_{rollout['seed']}.png"
        plt.savefig(out, dpi=220)
        print("Saved:", out)
        plt.show()

    plt.figure(figsize=(10, 4))
    for col in ["road_mask", "sidewalk_mask", "building_mask", "green_mask", "canopy_mask", "sensitive_mask"]:
        if col in df.columns:
            plt.plot(df["t_s"], df[col], label=col)

    plt.xlabel("Time [s]")
    plt.ylabel("Layer active")
    plt.title("Semantic layer under UAV over time")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out = save_dir / f"timeseries_layers_seed_{rollout['seed']}.png"
    plt.savefig(out, dpi=220)
    print("Saved:", out)
    plt.show()

def plot_vienna_3d_basic(world, rollout, stride=8, html_path=None):
    import plotly.graph_objects as go

    terrain = np.asarray(world.terrain_z, dtype=np.float32)
    h, w = terrain.shape

    ys = np.arange(0, h, int(stride))
    xs = np.arange(0, w, int(stride))
    X, Y = np.meshgrid(xs, ys)
    Z = terrain[np.ix_(ys, xs)]

    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Earth",
            opacity=0.82,
            showscale=False,
            name="terrain",
        )
    )

    def add_ground_points(mask_name, color, name, z_offset, marker_size=2, opacity=0.45, pstride=10):
        arr = np.asarray(getattr(world, mask_name), dtype=np.float32)
        terrain_local = np.asarray(world.terrain_z, dtype=np.float32)

        pys = np.arange(0, h, int(pstride))
        pxs = np.arange(0, w, int(pstride))
        sub = arr[np.ix_(pys, pxs)]
        yy, xx = np.where(sub > 0.5)

        if len(xx) == 0:
            return

        xpts = pxs[xx]
        ypts = pys[yy]
        zpts = terrain_local[ypts, xpts] + float(z_offset)

        fig.add_trace(
            go.Scatter3d(
                x=xpts,
                y=ypts,
                z=zpts,
                mode="markers",
                marker=dict(size=marker_size, color=color, opacity=opacity),
                name=name,
                hoverinfo="skip",
            )
        )

    add_ground_points("green_mask", "green", "green", 0.5, opacity=0.30)
    add_ground_points("road_mask", "orange", "roads", 0.8, opacity=0.45)
    add_ground_points("sidewalk_mask", "yellow", "footways/sidewalks", 1.1, opacity=0.45)
    add_ground_points("sensitive_mask", "red", "sensitive", 1.4, opacity=0.25)

    # Building roofs as fallback surface
    if hasattr(world, "building_roof_z"):
        bmask = np.asarray(world.building_mask, dtype=np.float32)[np.ix_(ys, xs)]
        roof = np.asarray(world.building_roof_z, dtype=np.float32)[np.ix_(ys, xs)]
        Zb = np.where(bmask > 0.5, roof, np.nan)

        if np.isfinite(Zb).any():
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Zb,
                    colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
                    opacity=0.75,
                    showscale=False,
                    name="building roofs",
                    hoverinfo="skip",
                )
            )

    if hasattr(world, "canopy_top_z"):
        cmask = np.asarray(world.canopy_mask, dtype=np.float32)[np.ix_(ys, xs)]
        ctop = np.asarray(world.canopy_top_z, dtype=np.float32)[np.ix_(ys, xs)]
        Zc = np.where(cmask > 0.5, ctop, np.nan)

        if np.isfinite(Zc).any():
            fig.add_trace(
                go.Surface(
                    x=X,
                    y=Y,
                    z=Zc,
                    colorscale=[[0, "darkgreen"], [1, "darkgreen"]],
                    opacity=0.35,
                    showscale=False,
                    name="canopy top",
                    hoverinfo="skip",
                )
            )

    path = rollout["path"]

    fig.add_trace(
        go.Scatter3d(
            x=path[:, 0],
            y=path[:, 1],
            z=path[:, 2],
            mode="lines",
            line=dict(width=8, color="cyan"),
            name="actual UAV path",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[path[0, 0]],
            y=[path[0, 1]],
            z=[path[0, 2]],
            mode="markers+text",
            marker=dict(size=7, color="blue"),
            text=["START"],
            textposition="top center",
            name="START",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[path[-1, 0]],
            y=[path[-1, 1]],
            z=[path[-1, 2]],
            mode="markers+text",
            marker=dict(size=8, color="cyan"),
            text=["END"],
            textposition="top center",
            name="END",
        )
    )

    if rollout.get("planned_paths"):
        last_plan = rollout["planned_paths"][-1]
        z_plan = np.array([
            _ground_at_world(world, p[0], p[1]) + 30.0
            for p in last_plan
        ], dtype=np.float32)

        fig.add_trace(
            go.Scatter3d(
                x=last_plan[:, 0],
                y=last_plan[:, 1],
                z=z_plan,
                mode="lines",
                line=dict(width=4, color="white", dash="dash"),
                name="last planned path approx.",
            )
        )

    fig.update_layout(
        title=(
            f"Vienna Prater footway-only — UAV rollout 3D | "
            f"seed={rollout['seed']} | term={rollout['termination']} | "
            f"success={rollout['is_success']} | acceptable={rollout['mission_acceptable']}"
        ),
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=1.0, z=0.28),
        ),
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(bgcolor="rgba(255,255,255,0.65)"),
    )

    if html_path is not None:
        html_path = Path(html_path)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(html_path))
        print("Saved HTML:", html_path)

    fig.show()
    return fig

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from pathlib import Path

from datetime import datetime

import json

M13_EXPERIMENT_NAME = "m13_controlled_urban_uav_gates_warsaw"

def _m13_ground_at(world, x, y):
    arr = np.asarray(world.terrain_z, dtype=np.float32)
    h, w = arr.shape
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))
    return float(arr[yi, xi])

def _m13_mask_at(world, mask_name, x, y):
    arr = np.asarray(getattr(world, mask_name), dtype=np.float32)
    h, w = arr.shape
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))
    return float(arr[yi, xi])

def rollout_m13_no_terminal_with_telemetry(world, seed, cfg=m13_planner_cfg, max_steps=1000):
    """
    Re-run one Stage B no-terminal planner episode and store per-step semantic exposure.
    """
    env = make_ruscuav_env_m4_mode(
        world=world,
        stage="B",
        mode="strict_eval",
        risk_profile="balanced",
        n_obstacles=0,
        seed=seed,
    )

    obs, reset_info = env.reset(seed=seed)

    planner = M10RecedingHorizonRiskPlanner(
        world=world,
        config=cfg,
        risk_profile="balanced",
    )

    planner.reset_belief()
    planner.reveal_around_xy(env.current_pos[:2])

    target_xy = env.target_point[:2].copy()

    current_plan = None
    wp_idx = 0
    planned_paths = []

    rows = []
    rewards = []
    path = [env.current_pos.copy()]

    done = False
    steps = 0

    while not done and steps < int(max_steps):
        if current_plan is None or steps % int(max(1, cfg.replan_every_steps)) == 0:
            planner.reveal_around_xy(env.current_pos[:2])
            plan_result = planner.plan(
                start_xy=env.current_pos[:2].copy(),
                target_xy=target_xy,
            )

            if plan_result["success"]:
                current_plan = plan_result["path_xy"]
                planned_paths.append(current_plan.copy())
                wp_idx = 1 if len(current_plan) > 1 else 0
            elif current_plan is None:
                break

        action, wp_idx = m9_waypoint_controller_action(
            env=env,
            path_xy=current_plan,
            wp_idx=wp_idx,
            cfg=_m10_to_m9_config(cfg),
        )

        obs, reward, terminated, truncated, info = env.step(action)

        pos = env.current_pos.copy()
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        ground = _m13_ground_at(world, x, y)

        rows.append({
            "seed": int(seed),
            "step": int(steps + 1),
            "t_s": float((steps + 1) * env.dt),

            "x": x,
            "y": y,
            "z": z,
            "agl_m": float(z - ground),

            "reward": float(reward),
            "termination": info.get("termination"),
            "is_success": bool(info.get("is_success", False)),
            "mission_acceptable": bool(info.get("mission_acceptable", False)),

            "dist_xy": float(info.get("dist_xy", np.nan)),
            "cargo_health": float(info.get("cargo_health", np.nan)),

            "road_mask": _m13_mask_at(world, "road_mask", x, y),
            "sidewalk_mask": _m13_mask_at(world, "sidewalk_mask", x, y),
            "building_mask": _m13_mask_at(world, "building_mask", x, y),
            "green_mask": _m13_mask_at(world, "green_mask", x, y),
            "canopy_mask": _m13_mask_at(world, "canopy_mask", x, y),
            "sensitive_mask": _m13_mask_at(world, "sensitive_mask", x, y),

            "road_time": float(info.get("urban_road_time", 0.0)),
            "sidewalk_time": float(info.get("urban_sidewalk_time", 0.0)),
            "building_time": float(info.get("urban_building_time", 0.0)),
            "green_time": float(info.get("urban_green_time", 0.0)),
            "canopy_time": float(info.get("urban_canopy_time", 0.0)),
            "sensitive_time": float(info.get("urban_sensitive_time", 0.0)),

            "urban_cumulative_risk": float(info.get("urban_cumulative_risk", np.nan)),
            "urban_hard_violation_steps": int(info.get("urban_hard_violation_steps", 0)),
        })

        rewards.append(float(reward))
        path.append(pos)

        done = bool(terminated or truncated)
        steps += 1

    env.close()

    df = pd.DataFrame(rows)

    result = {
        "seed": int(seed),
        "df": df,
        "path": np.asarray(path, dtype=np.float32),
        "planned_paths": planned_paths,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "steps": int(steps),
        "termination": rows[-1]["termination"] if rows else "planner_no_path",
        "is_success": bool(rows[-1]["is_success"]) if rows else False,
        "mission_acceptable": bool(rows[-1]["mission_acceptable"]) if rows else False,
    }

    return result

def collect_m13_telemetry(world, seed_df, n_episodes=100, seed_col="seed"):
    """
    Collect telemetry for selected seeds.
    """
    if seed_col not in seed_df.columns:
        raise ValueError(f"seed_df missing column {seed_col}")

    seeds = seed_df[seed_col].astype(int).head(int(n_episodes)).tolist()

    results = {}
    rows_all = []

    for i, seed in enumerate(seeds):
        res = rollout_m13_no_terminal_with_telemetry(world, seed)
        results[int(seed)] = res

        if len(res["df"]) > 0:
            rows_all.append(res["df"])

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[M13 telemetry] {i+1}/{len(seeds)} seed={seed} term={res['termination']} success={res['is_success']}")

    telemetry_df = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()

    out_path = M13_DIR / f"m13_{M13_WORLD_LABEL}_telemetry.csv"
    telemetry_df.to_csv(out_path, index=False)

    print("Saved telemetry:", out_path)

    return results, telemetry_df

def build_m13_exposure_heatmap(world, telemetry_df, use_building=False):
    h, w = np.asarray(world.terrain_z).shape
    heat = np.zeros((h, w), dtype=np.float32)

    for _, r in telemetry_df.iterrows():
        x = float(r["x"])
        y = float(r["y"])

        xi = int(np.clip(x, 0, w - 1))
        yi = int(np.clip(y, 0, h - 1))

        # Weighted exposure. Sidewalk is usually the binding budget.
        val = 0.0
        val += 3.0 * float(r.get("sidewalk_mask", 0.0) > 0.5)
        val += 1.5 * float(r.get("road_mask", 0.0) > 0.5)

        if use_building:
            val += 1.0 * float(r.get("building_mask", 0.0) > 0.5)

        # Prefer non-green crossing conflicts.
        if float(r.get("green_mask", 0.0) > 0.5):
            val *= 0.25

        if float(r.get("sensitive_mask", 0.0) > 0.5):
            # We do not want to place gates in sensitive zones.
            val = 0.0

        heat[yi, xi] += val

    if gaussian_filter is not None:
        heat_smooth = gaussian_filter(heat, sigma=10.0)
    else:
        heat_smooth = heat

    return heat.astype(np.float32), heat_smooth.astype(np.float32)

def select_gate_candidates_from_heatmap(
    heat,
    world,
    n_candidates=20,
    min_distance_m=60.0,
    avoid_sensitive=True,
):
    heat_work = heat.copy()

    h, w = heat_work.shape
    candidates = []

    yy_grid, xx_grid = np.mgrid[0:h, 0:w]

    for _ in range(int(n_candidates)):
        idx = int(np.nanargmax(heat_work))
        y, x = np.unravel_index(idx, heat_work.shape)
        score = float(heat_work[y, x])

        if score <= 0:
            break

        if avoid_sensitive and _m13_mask_at(world, "sensitive_mask", x, y) > 0.5:
            heat_work[y, x] = 0.0
            continue

        candidates.append({
            "gate_id": len(candidates),
            "x": float(x),
            "y": float(y),
            "score": score,
            "road": _m13_mask_at(world, "road_mask", x, y),
            "sidewalk": _m13_mask_at(world, "sidewalk_mask", x, y),
            "building": _m13_mask_at(world, "building_mask", x, y),
            "green": _m13_mask_at(world, "green_mask", x, y),
            "sensitive": _m13_mask_at(world, "sensitive_mask", x, y),
        })

        # Suppress neighborhood.
        d2 = (xx_grid - x) ** 2 + (yy_grid - y) ** 2
        heat_work[d2 <= float(min_distance_m) ** 2] = 0.0

    return pd.DataFrame(candidates)

def plot_m13_gate_candidates(world, heat, candidates_df, save_path=None):
    h, w = heat.shape

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(
        world.terrain_z,
        origin="upper",
        cmap="terrain",
        alpha=0.65,
        extent=[0, w, h, 0],
    )

    hm = ax.imshow(
        heat,
        origin="upper",
        cmap="magma",
        alpha=0.65,
        extent=[0, w, h, 0],
    )

    plt.colorbar(hm, ax=ax, label="exposure hotspot score")

    if len(candidates_df) > 0:
        ax.scatter(
            candidates_df["x"],
            candidates_df["y"],
            s=110,
            facecolors="none",
            edgecolors="cyan",
            linewidths=2,
            label="candidate gates",
        )

        for _, r in candidates_df.head(15).iterrows():
            ax.text(
                r["x"] + 6,
                r["y"] + 6,
                str(int(r["gate_id"])),
                color="cyan",
                fontsize=9,
                weight="bold",
            )

    ax.set_title(f"M13 gate candidate heatmap — {M13_WORLD_LABEL}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220)
        print("Saved:", save_path)

    plt.show()

def make_circular_gate_mask(world, gates_df, n_gates, radius_m=30.0):
    h, w = np.asarray(world.terrain_z).shape
    mask = np.zeros((h, w), dtype=np.float32)

    if gates_df is None or len(gates_df) == 0 or int(n_gates) <= 0:
        return mask

    use = gates_df.head(int(n_gates)).copy()

    yy, xx = np.mgrid[0:h, 0:w]

    for _, g in use.iterrows():
        x0 = float(g["x"])
        y0 = float(g["y"])
        d2 = (xx - x0) ** 2 + (yy - y0) ** 2
        mask[d2 <= float(radius_m) ** 2] = 1.0

    # Never allow gates in sensitive zones.
    mask[np.asarray(world.sensitive_mask) > 0.5] = 0.0

    return mask.astype(np.float32)

def plot_m13_gate_mask(world, gate_mask, candidates_df=None, save_path=None, title=None):
    h, w = gate_mask.shape

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(
        world.terrain_z,
        origin="upper",
        cmap="terrain",
        alpha=0.75,
        extent=[0, w, h, 0],
    )

    # underlying sidewalks/roads
    road = np.ma.masked_where(np.asarray(world.road_mask) <= 0.5, world.road_mask)
    side = np.ma.masked_where(np.asarray(world.sidewalk_mask) <= 0.5, world.sidewalk_mask)
    gates = np.ma.masked_where(gate_mask <= 0.5, gate_mask)

    ax.imshow(road, origin="upper", cmap="Oranges", alpha=0.35, extent=[0, w, h, 0])
    ax.imshow(side, origin="upper", cmap="Wistia", alpha=0.35, extent=[0, w, h, 0])
    ax.imshow(gates, origin="upper", cmap="cool", alpha=0.70, extent=[0, w, h, 0])

    if candidates_df is not None and len(candidates_df) > 0:
        ax.scatter(candidates_df["x"], candidates_df["y"], s=50, facecolors="none", edgecolors="cyan")

    if title is None:
        title = f"M13 controlled gate mask — {M13_WORLD_LABEL}"

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.grid(alpha=0.2)

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220)
        print("Saved:", save_path)

    plt.show()

def _point_in_gate_mask(gate_mask, x, y):
    h, w = gate_mask.shape
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))
    return bool(gate_mask[yi, xi] > 0.5)

def compute_gate_discounted_metrics_for_episode(
    df,
    gate_mask,
    road_factor=0.25,
    sidewalk_factor=0.25,
    building_factor=1.0,
):
    if len(df) == 0:
        return {
            "is_success": False,
            "mission_acceptable_gate": False,
            "road_time_gate": np.nan,
            "sidewalk_time_gate": np.nan,
            "building_time_gate": np.nan,
        }

    if "t_s" in df.columns and len(df) > 1:
        dt = float(np.nanmedian(np.diff(df["t_s"].values)))
        if not np.isfinite(dt) or dt <= 0:
            dt = 0.5
    else:
        dt = 0.5

    road_t = 0.0
    sidewalk_t = 0.0
    building_t = 0.0
    green_t = 0.0
    sensitive_t = 0.0

    for _, r in df.iterrows():
        in_gate = _point_in_gate_mask(gate_mask, float(r["x"]), float(r["y"]))

        rf = road_factor if in_gate else 1.0
        sf = sidewalk_factor if in_gate else 1.0
        bf = building_factor if in_gate else 1.0

        road_t += dt * float(r.get("road_mask", 0.0) > 0.5) * rf
        sidewalk_t += dt * float(r.get("sidewalk_mask", 0.0) > 0.5) * sf
        building_t += dt * float(r.get("building_mask", 0.0) > 0.5) * bf
        green_t += dt * float(r.get("green_mask", 0.0) > 0.5)
        sensitive_t += dt * float(r.get("sensitive_mask", 0.0) > 0.5)

    last = df.iloc[-1]

    is_success = bool(last.get("is_success", False))
    cargo_health = float(last.get("cargo_health", np.nan))
    hard = int(last.get("urban_hard_violation_steps", 0))

    accept = (
        is_success
        and cargo_health >= 0.80
        and road_t <= 20.0
        and sidewalk_t <= 5.0
        and building_t <= 15.0
        and sensitive_t <= 0.0
        and hard == 0
    )

    return {
        "is_success": is_success,
        "mission_acceptable_gate": bool(accept),
        "cargo_health": cargo_health,
        "road_time_gate": float(road_t),
        "sidewalk_time_gate": float(sidewalk_t),
        "building_time_gate": float(building_t),
        "green_time": float(green_t),
        "sensitive_time": float(sensitive_t),
        "urban_hard_violation_steps": int(hard),
    }

def evaluate_gate_set_posthoc(
    m13_results,
    gate_mask,
    label,
    road_factor=0.25,
    sidewalk_factor=0.25,
    building_factor=1.0,
):
    rows = []

    for seed, res in m13_results.items():
        met = compute_gate_discounted_metrics_for_episode(
            res["df"],
            gate_mask=gate_mask,
            road_factor=road_factor,
            sidewalk_factor=sidewalk_factor,
            building_factor=building_factor,
        )

        met["seed"] = int(seed)
        met["label"] = label
        rows.append(met)

    out = pd.DataFrame(rows)

    summary = {
        "label": label,
        "n": int(len(out)),
        "success_rate": float(out["is_success"].mean()),
        "gate_acceptability_rate": float(out["mission_acceptable_gate"].mean()),
        "mean_road_time_gate": float(out["road_time_gate"].mean()),
        "mean_sidewalk_time_gate": float(out["sidewalk_time_gate"].mean()),
        "mean_building_time_gate": float(out["building_time_gate"].mean()),
        "mean_green_time": float(out["green_time"].mean()),
        "mean_sensitive_time": float(out["sensitive_time"].mean()),
    }

    return out, summary

def run_m13_posthoc_gate_sweep(
    world,
    gates_df,
    m13_results,
    n_gates_list=(0, 1, 3, 5, 10, 15, 20),
    radius_m=30.0,
    road_factor=0.25,
    sidewalk_factor=0.25,
    building_factor=1.0,
):
    all_rows = []
    summaries = []

    for n_gates in n_gates_list:
        gate_mask = make_circular_gate_mask(
            world,
            gates_df,
            n_gates=int(n_gates),
            radius_m=float(radius_m),
        )

        label = f"N{n_gates}_r{radius_m}_rf{road_factor}_sf{sidewalk_factor}_bf{building_factor}"

        df_eval, summary = evaluate_gate_set_posthoc(
            m13_results=m13_results,
            gate_mask=gate_mask,
            label=label,
            road_factor=road_factor,
            sidewalk_factor=sidewalk_factor,
            building_factor=building_factor,
        )

        summary.update({
            "n_gates": int(n_gates),
            "radius_m": float(radius_m),
            "road_factor": float(road_factor),
            "sidewalk_factor": float(sidewalk_factor),
            "building_factor": float(building_factor),
            "gate_mask_coverage": float(gate_mask.mean()),
        })

        all_rows.append(df_eval)
        summaries.append(summary)

    all_df = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    all_path = M13_DIR / f"m13_{M13_WORLD_LABEL}_posthoc_gate_episode_results.csv"
    summary_path = M13_DIR / f"m13_{M13_WORLD_LABEL}_posthoc_gate_sweep_summary.csv"

    all_df.to_csv(all_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("Saved:", all_path)
    print("Saved:", summary_path)

    return all_df, summary_df

def clone_world_with_gate_mask(
    world,
    gate_mask,
    label_suffix,
    road_factor=0.25,
    sidewalk_factor=0.25,
    building_factor=1.0,
):
    obj = UrbanWorldLayers.__new__(UrbanWorldLayers)

    attrs = [
        "aoi_id", "crs", "bbox", "resolution_m",
        "terrain_z", "surface_z", "height_above_ground",
        "building_mask", "building_height", "building_roof_z",
        "road_mask", "sidewalk_mask", "green_mask",
        "canopy_mask", "canopy_height", "canopy_top_z", "canopy_density",
        "sensitive_mask",
    ]

    for a in attrs:
        if hasattr(world, a):
            try:
                setattr(obj, a, np.asarray(getattr(world, a)).copy() if isinstance(getattr(world, a), np.ndarray) else getattr(world, a))
            except Exception:
                pass

    gate_mask = np.asarray(gate_mask, dtype=np.float32)

    # Store gate mask if possible.
    try:
        setattr(obj, "managed_gate_mask", gate_mask)
    except Exception:
        pass

    # Reduce road/sidewalk/building values inside gates.
    road = np.asarray(obj.road_mask, dtype=np.float32).copy()
    side = np.asarray(obj.sidewalk_mask, dtype=np.float32).copy()
    build = np.asarray(obj.building_mask, dtype=np.float32).copy()

    road[gate_mask > 0.5] *= float(road_factor)
    side[gate_mask > 0.5] *= float(sidewalk_factor)
    build[gate_mask > 0.5] *= float(building_factor)

    obj.road_mask = road.astype(np.float32)
    obj.sidewalk_mask = side.astype(np.float32)
    obj.building_mask = build.astype(np.float32)

    # Never alter sensitive mask.
    obj = patch_urban_world_layers_for_ruscuav(obj, verbose=False)

    try:
        obj.aoi_id = str(getattr(world, "aoi_id", "world")) + "_" + str(label_suffix)
    except Exception:
        pass

    return obj

def evaluate_gate_aware_planner_variants(
    base_world,
    gates_df,
    n_gates_list=(0, 3, 5, 10, 15),
    radius_m=30.0,
    road_factor=0.25,
    sidewalk_factor=0.25,
    building_factor=1.0,
    n_episodes=100,
    seed0=7000,
):
    rows = []

    worlds = {}

    for n_gates in n_gates_list:
        gate_mask = make_circular_gate_mask(
            base_world,
            gates_df,
            n_gates=int(n_gates),
            radius_m=float(radius_m),
        )

        gate_world = clone_world_with_gate_mask(
            base_world,
            gate_mask=gate_mask,
            label_suffix=f"gates_N{n_gates}_r{radius_m}",
            road_factor=road_factor,
            sidewalk_factor=sidewalk_factor,
            building_factor=building_factor,
        )

        worlds[int(n_gates)] = gate_world

        prefix = (
            f"m13_{M13_WORLD_LABEL}_gateaware_"
            f"N{n_gates}_r{radius_m}_rf{road_factor}_sf{sidewalk_factor}_bf{building_factor}_stageC_{n_episodes}"
        )

        df_eval, summary, csv_path, json_path = evaluate_m10_receding_planner_drive(
            world=gate_world,
            drive_dirs=M13_DRIVE_DIRS,
            cfg=m13_planner_cfg,
            risk_profile="balanced",
            n_episodes=int(n_episodes),
            seed0=int(seed0),
            stage="C",
            mode="strict_eval",
            n_obstacles=0,
            prefix=prefix,
            save_paths=True,
        )

        rows.append({
            "n_gates": int(n_gates),
            "radius_m": float(radius_m),
            "road_factor": float(road_factor),
            "sidewalk_factor": float(sidewalk_factor),
            "building_factor": float(building_factor),
            "gate_mask_coverage": float(gate_mask.mean()),

            "success_rate": summary["success_rate"],
            "mission_acceptability_rate": summary["mission_acceptability_rate"],
            "acceptable_given_success_rate": summary["acceptable_given_success_rate"],

            "mean_steps": summary["mean_steps"],
            "mean_dist_xy": summary["mean_dist_xy"],
            "mean_cumulative_risk": summary["mean_cumulative_risk"],
            "mean_road_time": summary["mean_road_time"],
            "mean_sidewalk_time": summary["mean_sidewalk_time"],
            "mean_building_time": summary["mean_building_time"],
            "mean_green_time": summary["mean_green_time"],
            "mean_canopy_time": summary["mean_canopy_time"],
            "mean_sensitive_time": summary["mean_sensitive_time"],
            "mean_hard_violation_steps": summary["mean_hard_violation_steps"],

            "termination_counts": json.dumps(summary["termination_counts"]),
            "csv_path": csv_path,
            "summary_path": json_path,
        })

    summary_df = pd.DataFrame(rows)

    out_path = M13_DIR / f"m13_{M13_WORLD_LABEL}_gateaware_planner_summary.csv"
    summary_df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    display(summary_df)

    return summary_df, worlds

def plot_m13_sweep(summary_df, y_col, title, save_path=None):
    plt.figure(figsize=(8, 5))

    plt.plot(
        summary_df["n_gates"],
        summary_df[y_col],
        marker="o",
        linewidth=2,
    )

    plt.xlabel("Number of controlled gates")
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220)
        print("Saved:", save_path)

    plt.show()

def clean_m13_gate_candidates(candidates_df):
    d = candidates_df.copy()

    d = d[
        ((d["road"] > 0.5) | (d["sidewalk"] > 0.5))
        & (d["building"] <= 0.5)
        & (d["sensitive"] <= 0.5)
    ].copy()

    d = d.sort_values("score", ascending=False).reset_index(drop=True)
    d["gate_id_clean"] = np.arange(len(d))

    return d

def run_m13_radius_sweep(
    base_world,
    gates_df,
    n_gates=15,
    radii=(20.0, 30.0, 45.0),
    road_factor=0.25,
    sidewalk_factor=0.25,
    building_factor=1.0,
    n_episodes=100,
    seed0=7000,
):
    rows = []

    for r in radii:
        summary_df, _worlds = evaluate_gate_aware_planner_variants(
            base_world=base_world,
            gates_df=gates_df,
            n_gates_list=(int(n_gates),),
            radius_m=float(r),
            road_factor=road_factor,
            sidewalk_factor=sidewalk_factor,
            building_factor=building_factor,
            n_episodes=n_episodes,
            seed0=seed0,
        )

        rows.append(summary_df.iloc[0].to_dict())

    out = pd.DataFrame(rows)

    path = M13_DIR / f"m13_{M13_WORLD_LABEL}_radius_sweep_N{n_gates}.csv"
    out.to_csv(path, index=False)

    display(out)
    print("Saved:", path)

    return out

def plot_m13_final_gate_curve(summary_df, save_path=None):
    d = summary_df.copy().sort_values("n_gates")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        d["n_gates"],
        d["mission_acceptability_rate"],
        marker="o",
        linewidth=2,
        label="Mission acceptability",
    )

    ax1.plot(
        d["n_gates"],
        d["success_rate"],
        marker="s",
        linewidth=2,
        label="Success rate",
    )

    ax1.set_xlabel("Number of controlled gates")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        d["n_gates"],
        d["gate_mask_coverage"],
        marker="^",
        linestyle="--",
        linewidth=1.5,
        label="Gate area coverage",
    )
    ax2.set_ylabel("Gate mask coverage")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("M13: controlled gates improve dense-city mission acceptability")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=220)
        print("Saved:", save_path)

    plt.show()

from scipy.stats import binomtest

def paired_m13_gate_test(summary_df, n0=0, n_best=15):
    row0 = summary_df[summary_df["n_gates"] == n0].iloc[0]
    rowB = summary_df[summary_df["n_gates"] == n_best].iloc[0]

    df0 = pd.read_csv(row0["csv_path"])
    dfB = pd.read_csv(rowB["csv_path"])

    a = df0.copy().add_prefix("n0_")
    b = dfB.copy().add_prefix("best_")

    paired = a.merge(
        b,
        left_on="n0_seed",
        right_on="best_seed",
        how="inner",
    )

    n0_success_best_fail = int(
        ((paired["n0_is_success"] == True) & (paired["best_is_success"] == False)).sum()
    )
    best_success_n0_fail = int(
        ((paired["best_is_success"] == True) & (paired["n0_is_success"] == False)).sum()
    )

    n0_accept_best_not = int(
        ((paired["n0_mission_acceptable"] == True) & (paired["best_mission_acceptable"] == False)).sum()
    )
    best_accept_n0_not = int(
        ((paired["best_mission_acceptable"] == True) & (paired["n0_mission_acceptable"] == False)).sum()
    )

    def pval(a_only, b_only):
        n = int(a_only + b_only)
        if n == 0:
            return 1.0
        return float(
            binomtest(
                k=min(int(a_only), int(b_only)),
                n=n,
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )

    out = {
        "n_paired": int(len(paired)),
        "n0": int(n0),
        "n_best": int(n_best),

        "n0_success_best_fail": n0_success_best_fail,
        "best_success_n0_fail": best_success_n0_fail,
        "success_p": pval(n0_success_best_fail, best_success_n0_fail),

        "n0_accept_best_not": n0_accept_best_not,
        "best_accept_n0_not": best_accept_n0_not,
        "acceptability_p": pval(n0_accept_best_not, best_accept_n0_not),

        "mean_delta_steps_best_minus_n0": float(
            (paired["best_steps"] - paired["n0_steps"]).mean()
        ),
        "mean_delta_road_time_best_minus_n0": float(
            (paired["best_road_time"] - paired["n0_road_time"]).mean()
        ),
        "mean_delta_sidewalk_time_best_minus_n0": float(
            (paired["best_sidewalk_time"] - paired["n0_sidewalk_time"]).mean()
        ),
        "mean_delta_cumulative_risk_best_minus_n0": float(
            (paired["best_urban_cumulative_risk"] - paired["n0_urban_cumulative_risk"]).mean()
        ),
    }

    path_paired = M13_DIR / f"m13_paired_N{n0}_vs_N{n_best}.csv"
    paired.to_csv(path_paired, index=False)

    path_json = M13_DIR / f"m13_paired_N{n0}_vs_N{n_best}_summary.json"
    with open(path_json, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print("Saved:", path_paired)
    print("Saved:", path_json)

    return paired, out

def make_m13_effect_table(summary_df):
    d = summary_df.copy().sort_values("n_gates")
    base = d[d["n_gates"] == 0].iloc[0]

    rows = []

    for _, r in d.iterrows():
        rows.append({
            "n_gates": int(r["n_gates"]),
            "radius_m": float(r["radius_m"]),
            "gate_mask_coverage": float(r["gate_mask_coverage"]),

            "success_rate": float(r["success_rate"]),
            "mission_acceptability_rate": float(r["mission_acceptability_rate"]),

            "delta_success_vs_N0": float(r["success_rate"] - base["success_rate"]),
            "delta_acceptability_vs_N0": float(r["mission_acceptability_rate"] - base["mission_acceptability_rate"]),

            "mean_road_time": float(r["mean_road_time"]),
            "mean_sidewalk_time": float(r["mean_sidewalk_time"]),
            "mean_building_time": float(r["mean_building_time"]),
            "mean_cumulative_risk": float(r["mean_cumulative_risk"]),

            "delta_road_time_vs_N0": float(r["mean_road_time"] - base["mean_road_time"]),
            "delta_sidewalk_time_vs_N0": float(r["mean_sidewalk_time"] - base["mean_sidewalk_time"]),
            "delta_cumulative_risk_vs_N0": float(r["mean_cumulative_risk"] - base["mean_cumulative_risk"]),
        })

    out = pd.DataFrame(rows)

    return out

def plot_m13_final_confirm(summary_df, save_path=None):
    d = summary_df.copy().sort_values("n_gates")

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(
        d["n_gates"],
        d["mission_acceptability_rate"],
        marker="o",
        linewidth=2.5,
        label="Mission acceptability",
    )

    ax1.plot(
        d["n_gates"],
        d["success_rate"],
        marker="s",
        linewidth=2.0,
        label="Success rate",
    )

    ax1.set_xlabel("Number of controlled UAV gates")
    ax1.set_ylabel("Rate")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()

    ax2.plot(
        d["n_gates"],
        d["gate_mask_coverage"],
        marker="^",
        linestyle="--",
        linewidth=1.8,
        label="Gate area coverage",
    )

    ax2.set_ylabel("Gate mask coverage")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    plt.title("Controlled UAV gates improve dense-city mission acceptability")
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=240)
        print("Saved:", save_path)

    plt.show()

import os

import json

import shutil

from pathlib import Path

from datetime import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import binomtest

def _exists(name):
    return name in globals() and globals()[name] is not None

def _g(name, default=None):
    return globals().get(name, default)

def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(x) for x in obj]
    if isinstance(obj, tuple):
        return [json_safe(x) for x in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(json_safe(obj), f, indent=2)
    tmp.replace(path)
    print("Saved JSON:", path)

def safe_val(summary, key, default=np.nan):
    if summary is None:
        return default
    try:
        return summary.get(key, default)
    except Exception:
        return default

def term_count(summary, term):
    if summary is None:
        return 0
    try:
        return int(summary.get("termination_counts", {}).get(term, 0))
    except Exception:
        return 0

def first_available(*names):
    for name in names:
        if _exists(name):
            return _g(name)
    return None

def first_available_name(*names):
    for name in names:
        if _exists(name):
            return name
    return None

def maybe_read_csv(path):
    if path is None:
        return None
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None

def summary_to_row(label, summary, section, method_type, notes="", extra=None):
    if summary is None:
        return None

    row = {
        "section": section,
        "label": label,
        "method_type": method_type,
        "notes": notes,

        "n_episodes": safe_val(summary, "n_episodes"),
        "success_rate": safe_val(summary, "success_rate"),
        "mission_acceptability_rate": safe_val(summary, "mission_acceptability_rate"),
        "acceptable_given_success_rate": safe_val(summary, "acceptable_given_success_rate"),
        "static_only_acceptability_rate": safe_val(summary, "static_only_acceptability_rate"),
        "dynamic_budget_ok_rate": safe_val(summary, "dynamic_budget_ok_rate"),

        "mean_reward": safe_val(summary, "mean_reward"),
        "std_reward": safe_val(summary, "std_reward"),
        "mean_steps": safe_val(summary, "mean_steps"),
        "mean_dist_xy": safe_val(summary, "mean_dist_xy"),
        "median_dist_xy": safe_val(summary, "median_dist_xy"),
        "mean_cargo_health": safe_val(summary, "mean_cargo_health"),

        "mean_urban_risk": safe_val(summary, "mean_urban_risk"),
        "mean_cumulative_risk": safe_val(summary, "mean_cumulative_risk"),
        "mean_urban_cumulative_risk": safe_val(summary, "mean_urban_cumulative_risk"),

        "mean_dynamic_risk": safe_val(summary, "mean_dynamic_risk"),
        "mean_dynamic_risk_max": safe_val(summary, "mean_dynamic_risk_max"),
        "mean_dynamic_cumulative_risk": safe_val(summary, "mean_dynamic_cumulative_risk"),

        "mean_road_time": safe_val(summary, "mean_road_time"),
        "mean_sidewalk_time": safe_val(summary, "mean_sidewalk_time"),
        "mean_building_time": safe_val(summary, "mean_building_time"),
        "mean_green_time": safe_val(summary, "mean_green_time"),
        "mean_canopy_time": safe_val(summary, "mean_canopy_time"),
        "mean_sensitive_time": safe_val(summary, "mean_sensitive_time"),
        "mean_hard_violation_steps": safe_val(summary, "mean_hard_violation_steps"),

        "mean_replan_count": safe_val(summary, "mean_replan_count"),
        "mean_total_expanded": safe_val(summary, "mean_total_expanded"),
        "mean_total_plan_time_s": safe_val(summary, "mean_total_plan_time_s"),
        "mean_plan_time_s": safe_val(summary, "mean_plan_time_s"),

        "term_success": term_count(summary, "success"),
        "term_stuck": term_count(summary, "stuck"),
        "term_shield_stuck": term_count(summary, "shield_stuck"),
        "term_planner_no_path": term_count(summary, "planner_no_path"),
        "term_near_success": term_count(summary, "near_success"),
        "term_sidewalk_budget": term_count(summary, "urban_sidewalk_budget_exceeded"),
        "term_road_budget": term_count(summary, "urban_road_budget_exceeded"),
        "term_building_budget": term_count(summary, "urban_building_budget_exceeded"),
        "term_sensitive_violation": term_count(summary, "urban_sensitive_low_altitude_violation"),

        "csv_path": safe_val(summary, "csv_path", None),
        "summary_path": safe_val(summary, "summary_path", None),
    }

    if pd.isna(row["mean_cumulative_risk"]) and not pd.isna(row["mean_urban_cumulative_risk"]):
        row["mean_cumulative_risk"] = row["mean_urban_cumulative_risk"]

    if pd.isna(row["mean_urban_cumulative_risk"]) and not pd.isna(row["mean_cumulative_risk"]):
        row["mean_urban_cumulative_risk"] = row["mean_cumulative_risk"]

    if extra:
        row.update(extra)

    return row

RESULT_SPECS = [
    # Synthetic static learned policies
    {
        "label": "M7_geometry_only_RL",
        "section": "synthetic_static",
        "method_type": "learned_policy",
        "summary_names": ["summary_geo_eval"],
        "df_names": ["df_geo_eval"],
        "notes": "geometry-only learned policy baseline",
    },
    {
        "label": "M8_obs_only_RL",
        "section": "synthetic_static",
        "method_type": "learned_policy",
        "summary_names": ["summary_m8_obs"],
        "df_names": ["df_m8_obs"],
        "notes": "semantic observation only, no semantic risk reward",
    },
    {
        "label": "M6_full_semantic_RL",
        "section": "synthetic_static",
        "method_type": "learned_policy",
        "summary_names": ["summary_eval"],
        "df_names": ["df_eval"],
        "notes": "semantic observation + semantic risk reward",
    },
    {
        "label": "M8_reward_only_RL",
        "section": "synthetic_static",
        "method_type": "learned_policy",
        "summary_names": ["summary_m8_reward_200", "summary_m8_reward"],
        "df_names": ["df_m8_reward_200", "df_m8_reward"],
        "notes": "semantic risk reward, no explicit semantic vector",
    },

    # Synthetic planners
    {
        "label": "M9_static_risk_Astar",
        "section": "synthetic_static",
        "method_type": "planner",
        "summary_names": ["summary_m9_b0"],
        "df_names": ["df_m9_b0"],
        "notes": "one-shot full-map static risk-cost A*",
    },
    {
        "label": "M10_partial_risk_receding_Astar",
        "section": "synthetic_static",
        "method_type": "planner",
        "summary_names": ["summary_m10_b1"],
        "df_names": ["df_m10_b1"],
        "notes": "partial-risk receding-horizon A*",
    },
    {
        "label": "M10_fullmap_receding_Astar",
        "section": "synthetic_static",
        "method_type": "planner",
        "summary_names": ["summary_m10_b0r"],
        "df_names": ["df_m10_b0r"],
        "notes": "full-map receding-horizon A*",
    },

    # Dynamic synthetic
    {
        "label": "M11_static_trained_RL_dynamic_eval",
        "section": "synthetic_dynamic",
        "method_type": "learned_policy",
        "summary_names": ["summary_m11_rl_v1_200"],
        "df_names": ["df_m11_rl_v1_200"],
        "notes": "static-trained reward-only RL evaluated under calibrated dynamic risk",
    },
    {
        "label": "M11_dynamic_receding_Astar",
        "section": "synthetic_dynamic",
        "method_type": "planner",
        "summary_names": ["summary_m11_planner_v1_best_200"],
        "df_names": ["df_m11_planner_v1_best_200"],
        "notes": "best calibrated dynamic receding A*",
    },
    {
        "label": "M11B_dynamic_trained_RL_no_dynamic_obs",
        "section": "synthetic_dynamic",
        "method_type": "learned_policy",
        "summary_names": ["summary_m11b_rl_200"],
        "df_names": ["df_m11b_rl_200"],
        "notes": "dynamic-trained reward-only RL without dynamic-risk observation",
    },
    {
        "label": "M11C_dynamic_trained_RL_dynamic_obs",
        "section": "synthetic_dynamic",
        "method_type": "learned_policy",
        "summary_names": ["summary_m11c_rl_200"],
        "df_names": ["df_m11c_rl_200"],
        "notes": "dynamic-trained RL with compact dynamic-risk observation",
    },

    # Real GIS Warsaw
    {
        "label": "Warsaw_proxy_stageC_strict",
        "section": "real_gis_warsaw",
        "method_type": "planner",
        "summary_names": ["summary_m12_stepA_proxy_C", "summary_m12_m10"],
        "df_names": ["df_m12_stepA_proxy_C", "df_m12_m10"],
        "notes": "dense-city stress test, proxy sidewalk, strict Stage C",
    },
    {
        "label": "Warsaw_proxy_stageB_no_terminal",
        "section": "real_gis_warsaw",
        "method_type": "planner",
        "summary_names": ["summary_m12_stepA_proxy_B", "summary_m12_m10_no_terminal"],
        "df_names": ["df_m12_stepA_proxy_B", "df_m12_m10_no_terminal"],
        "notes": "dense-city diagnostic, proxy sidewalk, no terminal budgets",
    },
    {
        "label": "Warsaw_footway_only_stageC_strict",
        "section": "real_gis_warsaw",
        "method_type": "planner",
        "summary_names": ["summary_m12_stepA_foot_C", "summary_m12_m10_footway_only"],
        "df_names": ["df_m12_stepA_foot_C", "df_m12_m10_footway_only"],
        "notes": "dense-city stress test, explicit footways only, strict Stage C",
    },
    {
        "label": "Warsaw_footway_only_stageB_no_terminal",
        "section": "real_gis_warsaw",
        "method_type": "planner",
        "summary_names": ["summary_m12_stepA_foot_B"],
        "df_names": ["df_m12_stepA_foot_B"],
        "notes": "dense-city diagnostic, explicit footways only, no terminal budgets",
    },

    # Real GIS Vienna
    {
        "label": "Vienna_footway_only_M10_planner_200",
        "section": "real_gis_vienna",
        "method_type": "planner",
        "summary_names": ["summary_m12c_planner_footway_200", "summary_m12c_planner_200"],
        "df_names": ["df_m12c_planner_footway_200", "df_m12c_planner_200"],
        "notes": "mixed urban-green validation AOI, footway-only, M10 planner",
    },
    {
        "label": "Vienna_M8_reward_transfer",
        "section": "real_gis_vienna",
        "method_type": "learned_policy",
        "summary_names": ["summary_m12c_reward_transfer_footway_200", "summary_m12c_reward_transfer_200"],
        "df_names": ["df_m12c_reward_transfer_footway_200", "df_m12c_reward_transfer_200"],
        "notes": "synthetic-trained reward-only transfer to Vienna",
    },
    {
        "label": "Vienna_M8_reward_finetuned",
        "section": "real_gis_vienna",
        "method_type": "learned_policy",
        "summary_names": ["summary_m12c_reward_ft_200"],
        "df_names": ["df_m12c_reward_ft_200"],
        "notes": "short fine-tune reward-only on Vienna",
    },
    {
        "label": "Vienna_M8_reward_extended",
        "section": "real_gis_vienna",
        "method_type": "learned_policy",
        "summary_names": ["summary_m12c_reward_ext_200"],
        "df_names": ["df_m12c_reward_ext_200"],
        "notes": "extended fine-tune reward-only on Vienna",
    },
    {
        "label": "Vienna_M7_geometry_only",
        "section": "real_gis_vienna",
        "method_type": "learned_policy",
        "summary_names": ["summary_m12c_geometry_200"],
        "df_names": ["df_m12c_geometry_200"],
        "notes": "geometry-only learned baseline on Vienna",
    },
]

def build_master_results_table():
    rows = []
    missing = []

    for spec in RESULT_SPECS:
        summary_name = first_available_name(*spec["summary_names"])
        df_name = first_available_name(*spec["df_names"])
        summary = _g(summary_name) if summary_name else None

        if summary is None:
            missing.append({
                "label": spec["label"],
                "missing_summary_candidates": spec["summary_names"],
                "missing_df_candidates": spec["df_names"],
            })
            continue

        row = summary_to_row(
            label=spec["label"],
            summary=summary,
            section=spec["section"],
            method_type=spec["method_type"],
            notes=spec["notes"],
            extra={
                "summary_var": summary_name,
                "df_var": df_name,
            },
        )
        rows.append(row)

    master = pd.DataFrame(rows)

    path = PAPER_TABLES / "paper_master_results_table.csv"
    master.to_csv(path, index=False)

    missing_path = PAPER_CHECKS / "missing_result_objects.json"
    write_json(missing_path, missing)

    print("Saved:", path)
    print("Missing objects:", len(missing))

    return master, missing

def save_section_table(master, section, filename):
    d = master[master["section"] == section].copy()
    path = PAPER_TABLES / filename
    d.to_csv(path, index=False)
    print("Saved:", path)
    return d

def get_m13_confirm_table():
    if _exists("m13_gateaware_confirm_200"):
        return _g("m13_gateaware_confirm_200").copy()

    if _exists("m13_gateaware_planner_summary_ext"):
        d = _g("m13_gateaware_planner_summary_ext").copy()
        if set([0, 15, 30]).issubset(set(d["n_gates"].astype(int))):
            return d[d["n_gates"].isin([0, 15, 30])].copy()

    return None

def make_m13_effect_table(summary_df):
    if summary_df is None or len(summary_df) == 0:
        return pd.DataFrame()

    d = summary_df.copy().sort_values("n_gates")
    base = d[d["n_gates"].astype(int) == 0].iloc[0]

    rows = []

    for _, r in d.iterrows():
        rows.append({
            "n_gates": int(r["n_gates"]),
            "radius_m": float(r["radius_m"]),
            "gate_mask_coverage": float(r["gate_mask_coverage"]),

            "success_rate": float(r["success_rate"]),
            "mission_acceptability_rate": float(r["mission_acceptability_rate"]),
            "acceptable_given_success_rate": float(r["acceptable_given_success_rate"]),

            "delta_success_vs_N0": float(r["success_rate"] - base["success_rate"]),
            "delta_acceptability_vs_N0": float(r["mission_acceptability_rate"] - base["mission_acceptability_rate"]),

            "mean_road_time": float(r["mean_road_time"]),
            "mean_sidewalk_time": float(r["mean_sidewalk_time"]),
            "mean_building_time": float(r["mean_building_time"]),
            "mean_green_time": float(r["mean_green_time"]),
            "mean_cumulative_risk": float(r["mean_cumulative_risk"]),

            "delta_road_time_vs_N0": float(r["mean_road_time"] - base["mean_road_time"]),
            "delta_sidewalk_time_vs_N0": float(r["mean_sidewalk_time"] - base["mean_sidewalk_time"]),
            "delta_cumulative_risk_vs_N0": float(r["mean_cumulative_risk"] - base["mean_cumulative_risk"]),
        })

    return pd.DataFrame(rows)

def paired_binom_p(a_only, b_only):
    n = int(a_only + b_only)
    if n == 0:
        return 1.0
    return float(
        binomtest(
            k=min(int(a_only), int(b_only)),
            n=n,
            p=0.5,
            alternative="two-sided",
        ).pvalue
    )

def paired_compare(df_a, df_b, label_a, label_b):
    if df_a is None or df_b is None:
        return None, None, None

    if "seed" not in df_a.columns or "seed" not in df_b.columns:
        return None, None, None

    a = df_a.copy().add_prefix("a_")
    b = df_b.copy().add_prefix("b_")

    paired = a.merge(
        b,
        left_on="a_seed",
        right_on="b_seed",
        how="inner",
    )

    if len(paired) == 0:
        return paired, {}, {}

    summary = {
        "label_a": label_a,
        "label_b": label_b,
        "n_paired": int(len(paired)),
    }

    outcomes = [
        ("success", "is_success"),
        ("acceptability", "mission_acceptable"),
    ]

    if "a_dynamic_budget_ok" in paired.columns and "b_dynamic_budget_ok" in paired.columns:
        outcomes.append(("dynamic_budget_ok", "dynamic_budget_ok"))

    pvals = {}

    for name, col in outcomes:
        a_col = f"a_{col}"
        b_col = f"b_{col}"

        a_only = int(((paired[a_col].astype(bool) == True) & (paired[b_col].astype(bool) == False)).sum())
        b_only = int(((paired[b_col].astype(bool) == True) & (paired[a_col].astype(bool) == False)).sum())

        summary[f"{label_a}_{name}_{label_b}_not"] = a_only
        summary[f"{label_b}_{name}_{label_a}_not"] = b_only
        summary[f"{name}_discordant"] = a_only + b_only

        pvals[f"{name}_p"] = paired_binom_p(a_only, b_only)

    continuous = [
        "steps",
        "dist_xy",
        "cargo_health",
        "urban_cumulative_risk",
        "dynamic_cumulative_risk",
        "road_time",
        "sidewalk_time",
        "building_time",
        "green_time",
        "canopy_time",
    ]

    for col in continuous:
        a_col = f"a_{col}"
        b_col = f"b_{col}"
        if a_col in paired.columns and b_col in paired.columns:
            delta = paired[a_col].astype(float) - paired[b_col].astype(float)
            summary[f"mean_delta_{col}_{label_a}_minus_{label_b}"] = float(delta.mean())
            summary[f"median_delta_{col}_{label_a}_minus_{label_b}"] = float(delta.median())

    return paired, summary, pvals

def get_df_for_label(label):
    # Map important labels to likely dataframe variables.
    mapping = {
        "M8_reward_only_RL": ["df_m8_reward_200", "df_m8_reward"],
        "M7_geometry_only_RL": ["df_geo_eval"],
        "M6_full_semantic_RL": ["df_eval"],
        "M10_fullmap_receding_Astar": ["df_m10_b0r"],
        "M11_static_trained_RL_dynamic_eval": ["df_m11_rl_v1_200"],
        "M11_dynamic_receding_Astar": ["df_m11_planner_v1_best_200"],
        "M11B_dynamic_trained_RL_no_dynamic_obs": ["df_m11b_rl_200"],
        "M11C_dynamic_trained_RL_dynamic_obs": ["df_m11c_rl_200"],
        "Vienna_M8_reward_transfer": ["df_m12c_reward_transfer_footway_200", "df_m12c_reward_transfer_200"],
        "Vienna_M8_reward_finetuned": ["df_m12c_reward_ft_200"],
        "Vienna_M8_reward_extended": ["df_m12c_reward_ext_200"],
        "Vienna_M7_geometry_only": ["df_m12c_geometry_200"],
        "Vienna_footway_only_M10_planner_200": ["df_m12c_planner_footway_200", "df_m12c_planner_200"],
    }

    for name in mapping.get(label, []):
        if _exists(name):
            return _g(name)

    return None

PAIRED_TEST_SPECS = [
    ("M8_reward_only_RL", "M7_geometry_only_RL", "synthetic_reward_vs_geometry"),
    ("M6_full_semantic_RL", "M7_geometry_only_RL", "synthetic_fullsemantic_vs_geometry"),
    ("M8_reward_only_RL", "M10_fullmap_receding_Astar", "synthetic_reward_vs_fullmap_receding_planner"),

    ("M11_static_trained_RL_dynamic_eval", "M11_dynamic_receding_Astar", "dynamic_staticRL_vs_planner"),
    ("M11B_dynamic_trained_RL_no_dynamic_obs", "M11_dynamic_receding_Astar", "dynamic_noObsRL_vs_planner"),
    ("M11C_dynamic_trained_RL_dynamic_obs", "M11_dynamic_receding_Astar", "dynamic_obsRL_vs_planner"),

    ("Vienna_M8_reward_transfer", "Vienna_M8_reward_extended", "vienna_transfer_vs_extended"),
    ("Vienna_M8_reward_finetuned", "Vienna_M8_reward_extended", "vienna_shortFT_vs_extended"),
    ("Vienna_M8_reward_extended", "Vienna_footway_only_M10_planner_200", "vienna_extended_vs_planner"),
    ("Vienna_M7_geometry_only", "Vienna_M8_reward_extended", "vienna_geometry_vs_extended"),
]

def run_paper_paired_tests():
    rows = []
    pval_rows = []

    for label_a, label_b, comp_name in PAIRED_TEST_SPECS:
        df_a = get_df_for_label(label_a)
        df_b = get_df_for_label(label_b)

        if df_a is None or df_b is None:
            print("[skip paired]", comp_name, "| missing df")
            continue

        paired, summary, pvals = paired_compare(
            df_a=df_a,
            df_b=df_b,
            label_a=label_a,
            label_b=label_b,
        )

        if paired is None:
            print("[skip paired]", comp_name, "| incompatible")
            continue

        paired_path = PAPER_TABLES / f"paired_{comp_name}.csv"
        paired.to_csv(paired_path, index=False)

        summary["comparison"] = comp_name
        summary["paired_csv"] = str(paired_path)

        rows.append(summary)

        pval_row = {"comparison": comp_name}
        pval_row.update(pvals)
        pval_rows.append(pval_row)

        write_json(
            PAPER_JSON / f"paired_{comp_name}_summary.json",
            {"summary": summary, "p_values": pvals},
        )

    summaries_df = pd.DataFrame(rows)
    pvals_df = pd.DataFrame(pval_rows)

    summaries_path = PAPER_TABLES / "paper_paired_test_summaries.csv"
    pvals_path = PAPER_TABLES / "paper_paired_test_p_values.csv"

    summaries_df.to_csv(summaries_path, index=False)
    pvals_df.to_csv(pvals_path, index=False)

    print("Saved:", summaries_path)
    print("Saved:", pvals_path)

    return summaries_df, pvals_df

def paired_m13_from_summary_table(summary_df, n0=0, n_best=15):
    if summary_df is None:
        return None, None

    row0 = summary_df[summary_df["n_gates"].astype(int) == int(n0)].iloc[0]
    rowB = summary_df[summary_df["n_gates"].astype(int) == int(n_best)].iloc[0]

    df0 = pd.read_csv(row0["csv_path"])
    dfB = pd.read_csv(rowB["csv_path"])

    paired, summary, pvals = paired_compare(
        df_a=dfB,
        df_b=df0,
        label_a=f"N{n_best}",
        label_b=f"N{n0}",
    )

    summary["n0"] = int(n0)
    summary["n_best"] = int(n_best)

    paired_path = PAPER_TABLES / f"paired_m13_N{n_best}_vs_N{n0}.csv"
    paired.to_csv(paired_path, index=False)

    json_path = PAPER_JSON / f"paired_m13_N{n_best}_vs_N{n0}_summary.json"
    write_json(json_path, {"summary": summary, "p_values": pvals})

    return summary, pvals

def tail_summary(df, label, col="dynamic_cumulative_risk", budget=10.0):
    if df is None or col not in df.columns:
        return None

    s = df[col].astype(float)

    return {
        "label": label,
        "metric": col,
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=0)),
        "min": float(s.min()),
        "median": float(s.median()),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "p95": float(s.quantile(0.95)),
        "p99": float(s.quantile(0.99)),
        "max": float(s.max()),
        "budget": float(budget),
        "budget_ok_rate": float((s <= budget).mean()),
    }

def build_dynamic_tail_table():
    items = [
        ("M11_staticRL", get_df_for_label("M11_static_trained_RL_dynamic_eval")),
        ("M11_planner", get_df_for_label("M11_dynamic_receding_Astar")),
        ("M11B_noDynObsRL", get_df_for_label("M11B_dynamic_trained_RL_no_dynamic_obs")),
        ("M11C_dynObsRL", get_df_for_label("M11C_dynamic_trained_RL_dynamic_obs")),
    ]

    rows = []
    for label, df in items:
        row = tail_summary(df, label, col="dynamic_cumulative_risk", budget=10.0)
        if row is not None:
            rows.append(row)

    out = pd.DataFrame(rows)
    path = PAPER_TABLES / "paper_dynamic_tail_risk_table.csv"
    out.to_csv(path, index=False)
    print("Saved:", path)

    return out

def coverage_row(world, label):
    return {
        "aoi": label,
        "building": float(np.asarray(world.building_mask).mean()),
        "road": float(np.asarray(world.road_mask).mean()),
        "sidewalk": float(np.asarray(world.sidewalk_mask).mean()),
        "green": float(np.asarray(world.green_mask).mean()),
        "canopy": float(np.asarray(world.canopy_mask).mean()),
        "sensitive": float(np.asarray(world.sensitive_mask).mean()),
    }

def savefig(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    print("Saved figure:", path)
    plt.show()

def plot_bar_rates(df, label_col, rate_cols, title, path, figsize=(10, 5)):
    d = df.copy()
    d = d.dropna(subset=rate_cols, how="all")

    x = np.arange(len(d))
    width = 0.8 / len(rate_cols)

    plt.figure(figsize=figsize)

    for i, col in enumerate(rate_cols):
        plt.bar(x + (i - (len(rate_cols)-1)/2) * width, d[col], width=width, label=col)

    plt.xticks(x, d[label_col], rotation=35, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Rate")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()

    savefig(path)

def select_diagnostic_seeds(df, label, n=5):
    if df is None or len(df) == 0 or "seed" not in df.columns:
        return {}

    out = {}

    if "mission_acceptable" in df.columns:
        acc = df[(df["is_success"] == True) & (df["mission_acceptable"] == True)].copy()
        if len(acc) > 0:
            sort_col = "urban_cumulative_risk" if "urban_cumulative_risk" in acc.columns else "dist_xy"
            out["accepted_success_low_risk"] = acc.sort_values(sort_col)["seed"].head(n).astype(int).tolist()

    nonacc = df[(df["is_success"] == True) & (df["mission_acceptable"] == False)].copy()
    if len(nonacc) > 0:
        sort_col = "urban_cumulative_risk" if "urban_cumulative_risk" in nonacc.columns else "dist_xy"
        out["nonaccepted_success_high_risk"] = nonacc.sort_values(sort_col, ascending=False)["seed"].head(n).astype(int).tolist()

    fail = df[df["is_success"] == False].copy()
    if len(fail) > 0:
        out["failures"] = fail.sort_values("dist_xy")["seed"].head(n).astype(int).tolist()

    if "termination" in df.columns:
        out["termination_examples"] = {}
        for term, group in df.groupby("termination"):
            out["termination_examples"][str(term)] = group["seed"].head(min(n, len(group))).astype(int).tolist()

    return out

def fmt_rate(x):
    if x is None or pd.isna(x):
        return "NA"
    return f"{100*float(x):.1f}%"

def find_row(df, label):
    d = df[df["label"] == label]
    if len(d) == 0:
        return None
    return d.iloc[0]

import numpy as np

import pandas as pd

from pathlib import Path

from datetime import datetime

import json

def bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=123):
    values = np.asarray(values)

    values = values[np.isfinite(values)] if values.dtype.kind in "fc" else values

    if len(values) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)
    stats = []

    n = len(values)

    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        sample = values[idx]
        stats.append(float(stat_fn(sample)))

    stats = np.asarray(stats, dtype=np.float64)

    alpha = 1.0 - float(ci)
    lo = float(np.quantile(stats, alpha / 2.0))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
    mid = float(stat_fn(values))

    return mid, lo, hi

def eval_df_ci_summary(df, label, n_boot=2000):
    rows = []

    metrics = []

    if "is_success" in df.columns:
        metrics.append(("success_rate", df["is_success"].astype(bool).astype(float).values, np.mean))

    if "mission_acceptable" in df.columns:
        metrics.append(("mission_acceptability_rate", df["mission_acceptable"].astype(bool).astype(float).values, np.mean))

    if "dynamic_budget_ok" in df.columns:
        metrics.append(("dynamic_budget_ok_rate", df["dynamic_budget_ok"].astype(bool).astype(float).values, np.mean))

    continuous_cols = [
        "steps",
        "dist_xy",
        "cargo_health",
        "urban_cumulative_risk",
        "dynamic_cumulative_risk",
        "road_time",
        "sidewalk_time",
        "building_time",
        "green_time",
        "canopy_time",
    ]

    for col in continuous_cols:
        if col in df.columns:
            metrics.append((f"mean_{col}", df[col].astype(float).values, np.mean))

    for metric_name, values, fn in metrics:
        mid, lo, hi = bootstrap_ci(values, stat_fn=fn, n_boot=n_boot, seed=123)

        rows.append({
            "label": label,
            "metric": metric_name,
            "estimate": mid,
            "ci95_low": lo,
            "ci95_high": hi,
            "n": int(len(values)),
        })

    return pd.DataFrame(rows)

def build_final_ci_table():
    registry = []

    # Synthetic / dynamic / real, if available
    candidates = [
        ("M8_reward_only_synthetic", "df_m8_reward_200"),
        ("M7_geometry_only_synthetic", "df_geo_eval"),
        ("M10_fullmap_receding_synthetic", "df_m10_b0r"),

        ("M11_staticRL_dynamic", "df_m11_rl_v1_200"),
        ("M11_planner_dynamic", "df_m11_planner_v1_best_200"),
        ("M11C_dynamicObsRL", "df_m11c_rl_200"),

        ("Vienna_planner", "df_m12c_planner_footway_200"),
        ("Vienna_reward_transfer", "df_m12c_reward_transfer_footway_200"),
        ("Vienna_reward_extended", "df_m12c_reward_ext_200"),
        ("Vienna_geometry_only", "df_m12c_geometry_200"),
    ]

    for label, var_name in candidates:
        if var_name in globals():
            registry.append((label, globals()[var_name]))

    # M13 from final table CSV paths
    if "m13_gateaware_confirm_200" in globals():
        for _, row in m13_gateaware_confirm_200.iterrows():
            n_gates = int(row["n_gates"])
            try:
                df = pd.read_csv(row["csv_path"])
                registry.append((f"M13_N{n_gates}_gates", df))
            except Exception as e:
                print("Could not read M13 CSV:", n_gates, repr(e))

    all_ci = []

    for label, df in registry:
        ci_df = eval_df_ci_summary(df, label, n_boot=2000)
        all_ci.append(ci_df)

    out = pd.concat(all_ci, ignore_index=True) if all_ci else pd.DataFrame()

    path = CI_DIR / "paper_final_bootstrap_ci_table.csv"
    out.to_csv(path, index=False)

    print("Saved:", path)
    display(out)

    return out

from pathlib import Path

import numpy as np

import pandas as pd

import json

from datetime import datetime

def make_seed_df(seed0, n):
    return pd.DataFrame({
        "seed": np.arange(int(seed0), int(seed0) + int(n), dtype=int)
    })

def evaluate_gate_aware_planner_variants_oos(
    base_world,
    gates_df,
    fold_name,
    n_gates_list=(0, 15, 30),
    radius_m=30.0,
    road_factor=0.25,
    sidewalk_factor=0.25,
    building_factor=1.0,
    n_episodes=200,
    seed0=7200,
):
    rows = []

    for n_gates in n_gates_list:
        gate_mask = make_circular_gate_mask(
            base_world,
            gates_df,
            n_gates=int(n_gates),
            radius_m=float(radius_m),
        )

        gate_world = clone_world_with_gate_mask(
            base_world,
            gate_mask=gate_mask,
            label_suffix=f"{fold_name}_gates_N{n_gates}_r{radius_m}",
            road_factor=road_factor,
            sidewalk_factor=sidewalk_factor,
            building_factor=building_factor,
        )

        prefix = (
            f"m13_oos_{fold_name}_{M13_WORLD_LABEL}_"
            f"N{n_gates}_r{radius_m}_rf{road_factor}_sf{sidewalk_factor}_"
            f"evalseed{seed0}_n{n_episodes}"
        )

        df_eval, summary, csv_path, json_path = evaluate_m10_receding_planner_drive(
            world=gate_world,
            drive_dirs=M13_DRIVE_DIRS,
            cfg=m13_planner_cfg,
            risk_profile="balanced",
            n_episodes=int(n_episodes),
            seed0=int(seed0),
            stage="C",
            mode="strict_eval",
            n_obstacles=0,
            prefix=prefix,
            save_paths=True,
        )

        rows.append({
            "fold": fold_name,
            "eval_seed0": int(seed0),
            "n_episodes": int(n_episodes),

            "n_gates": int(n_gates),
            "radius_m": float(radius_m),
            "road_factor": float(road_factor),
            "sidewalk_factor": float(sidewalk_factor),
            "building_factor": float(building_factor),
            "gate_mask_coverage": float(gate_mask.mean()),

            "success_rate": summary["success_rate"],
            "mission_acceptability_rate": summary["mission_acceptability_rate"],
            "acceptable_given_success_rate": summary["acceptable_given_success_rate"],

            "mean_steps": summary["mean_steps"],
            "mean_dist_xy": summary["mean_dist_xy"],
            "mean_cumulative_risk": summary["mean_cumulative_risk"],
            "mean_road_time": summary["mean_road_time"],
            "mean_sidewalk_time": summary["mean_sidewalk_time"],
            "mean_building_time": summary["mean_building_time"],
            "mean_green_time": summary["mean_green_time"],
            "mean_canopy_time": summary["mean_canopy_time"],
            "mean_sensitive_time": summary["mean_sensitive_time"],
            "mean_hard_violation_steps": summary["mean_hard_violation_steps"],

            "termination_counts": json.dumps(summary["termination_counts"]),
            "csv_path": csv_path,
            "summary_path": json_path,
        })

    out = pd.DataFrame(rows)

    path = M13_OOS_DIR / f"m13_oos_{fold_name}_summary.csv"
    out.to_csv(path, index=False)

    print("Saved:", path)
    display(out)

    return out

def run_m13_oos_gate_fold(
    fold_name,
    mine_seed0,
    mine_n=100,
    eval_seed0=7200,
    eval_n=200,
    n_gates_list=(0, 15, 30),
    radius_m=30.0,
):
    print("=" * 100)
    print("M13 OOS FOLD")
    print("=" * 100)
    print("fold:", fold_name)
    print("mine_seed0:", mine_seed0, "mine_n:", mine_n)
    print("eval_seed0:", eval_seed0, "eval_n:", eval_n)
    print("=" * 100)

    # 1. Mine telemetry on training/mine seeds
    mine_seed_df = make_seed_df(mine_seed0, mine_n)

    fold_results, fold_telemetry = collect_m13_telemetry(
        world=m13_world_base,
        seed_df=mine_seed_df,
        n_episodes=mine_n,
    )

    telemetry_path = M13_OOS_DIR / f"m13_oos_{fold_name}_mining_telemetry.csv"
    fold_telemetry.to_csv(telemetry_path, index=False)

    # 2. Mine candidates
    heat_raw, heat_smooth = build_m13_exposure_heatmap(
        m13_world_base,
        fold_telemetry,
        use_building=False,
    )

    candidates = select_gate_candidates_from_heatmap(
        heat_smooth,
        m13_world_base,
        n_candidates=40,
        min_distance_m=55.0,
        avoid_sensitive=True,
    )

    candidates_clean = clean_m13_gate_candidates(candidates)

    candidates_path = M13_OOS_DIR / f"m13_oos_{fold_name}_gate_candidates_clean.csv"
    candidates_clean.to_csv(candidates_path, index=False)

    print("Clean candidates:", len(candidates_clean))
    display(candidates_clean.head(20))

    # 3. Evaluate fixed candidates on held-out seeds
    summary_oos = evaluate_gate_aware_planner_variants_oos(
        base_world=m13_world_base,
        gates_df=candidates_clean,
        fold_name=fold_name,
        n_gates_list=n_gates_list,
        radius_m=radius_m,
        road_factor=0.25,
        sidewalk_factor=0.25,
        building_factor=1.0,
        n_episodes=eval_n,
        seed0=eval_seed0,
    )

    summary_oos["mine_seed0"] = int(mine_seed0)
    summary_oos["mine_n"] = int(mine_n)
    summary_oos["eval_seed0"] = int(eval_seed0)
    summary_oos["eval_n"] = int(eval_n)
    summary_oos["candidate_csv"] = str(candidates_path)
    summary_oos["telemetry_csv"] = str(telemetry_path)

    final_path = M13_OOS_DIR / f"m13_oos_{fold_name}_final_summary.csv"
    summary_oos.to_csv(final_path, index=False)

    print("Saved:", final_path)

    return summary_oos, candidates_clean, fold_telemetry

def aggregate_m13_oos(summary_df):
    metrics = [
        "success_rate",
        "mission_acceptability_rate",
        "acceptable_given_success_rate",
        "mean_steps",
        "mean_dist_xy",
        "mean_cumulative_risk",
        "mean_road_time",
        "mean_sidewalk_time",
        "mean_building_time",
        "mean_green_time",
        "gate_mask_coverage",
    ]

    rows = []

    for n_gates, g in summary_df.groupby("n_gates"):
        row = {
            "n_gates": int(n_gates),
            "n_folds": int(g["fold"].nunique()),
        }

        for m in metrics:
            if m in g.columns:
                row[f"{m}_mean"] = float(g[m].mean())
                row[f"{m}_std"] = float(g[m].std(ddof=0))
                row[f"{m}_min"] = float(g[m].min())
                row[f"{m}_max"] = float(g[m].max())

        rows.append(row)

    out = pd.DataFrame(rows).sort_values("n_gates")

    # Add deltas vs N0
    base = out[out["n_gates"] == 0].iloc[0]

    for m in ["success_rate", "mission_acceptability_rate", "mean_cumulative_risk", "mean_road_time", "mean_sidewalk_time"]:
        col = f"{m}_mean"
        if col in out.columns:
            out[f"delta_{m}_vs_N0"] = out[col] - base[col]

    return out

def run_m12c_reward_multiseed(
    seeds=(6001, 7001, 8001),
    steps_A=50_000,
    steps_B=75_000,
    steps_C=150_000,
):
    rows = []

    for seed in seeds:
        exp_name = f"m12c_multiseed_reward_vienna_seed{seed}"

        dirs = setup_ruscuav_drive_project(
            drive_root="/content/drive/MyDrive/RUSCUAV",
            experiment_name=exp_name,
        )

        paths = train_m12c_real_vienna_policy_drive_resume(
            world_path=M12C_VIENNA_FOOTWAY_WORLD_PATH,
            policy_kind="reward_only",
            drive_dirs=dirs,
            seed=int(seed),
            n_obstacles=0,

            steps_A=int(steps_A),
            steps_B=int(steps_B),
            steps_C=int(steps_C),

            initial_model_path=m8_reward_only_model_path,

            save_freq_steps=10_000,
            device="cpu",
            n_envs=4,
            mode="fast_train",
            start_method="fork",
        )

        df_eval, summary_eval, csv_eval, json_eval = evaluate_m12c_real_policy_drive(
            model_path=paths["final_C"],
            world=vienna_world_footway_only,
            policy_kind="reward_only",
            drive_dirs=dirs,
            n_episodes=200,
            seed0=7000,
            prefix=f"m12c_multiseed_reward_seed{seed}_eval200",
            stage="C",
            mode="strict_eval",
            deterministic=True,
            save_paths=False,
        )

        rows.append({
            "seed": int(seed),
            "final_C": paths["final_C"],
            "success_rate": summary_eval["success_rate"],
            "mission_acceptability_rate": summary_eval["mission_acceptability_rate"],
            "acceptable_given_success_rate": summary_eval["acceptable_given_success_rate"],
            "mean_steps": summary_eval["mean_steps"],
            "mean_cumulative_risk": summary_eval["mean_cumulative_risk"],
            "mean_road_time": summary_eval["mean_road_time"],
            "mean_sidewalk_time": summary_eval["mean_sidewalk_time"],
            "mean_building_time": summary_eval["mean_building_time"],
            "csv_path": csv_eval,
            "summary_path": json_eval,
        })

    out = pd.DataFrame(rows)

    path = Path("/content/drive/MyDrive/RUSCUAV/paper_ready_v1/tables/m12c_reward_multiseed_summary.csv")
    out.to_csv(path, index=False)

    display(out)
    print("Saved:", path)

    return out

def run_m12c_geometry_multiseed(
    seeds=(6101, 7101, 8101),
    steps_A=50_000,
    steps_B=75_000,
    steps_C=150_000,
):
    rows = []

    for seed in seeds:
        exp_name = f"m12c_multiseed_geometry_vienna_seed{seed}"

        dirs = setup_ruscuav_drive_project(
            drive_root="/content/drive/MyDrive/RUSCUAV",
            experiment_name=exp_name,
        )

        paths = train_m12c_real_vienna_policy_drive_resume(
            world_path=M12C_VIENNA_FOOTWAY_WORLD_PATH,
            policy_kind="geometry_only",
            drive_dirs=dirs,
            seed=int(seed),
            n_obstacles=0,

            steps_A=int(steps_A),
            steps_B=int(steps_B),
            steps_C=int(steps_C),

            initial_model_path=None,

            save_freq_steps=10_000,
            device="cpu",
            n_envs=4,
            mode="fast_train",
            start_method="fork",
        )

        df_eval, summary_eval, csv_eval, json_eval = evaluate_m12c_real_policy_drive(
            model_path=paths["final_C"],
            world=vienna_world_footway_only,
            policy_kind="geometry_only",
            drive_dirs=dirs,
            n_episodes=200,
            seed0=7000,
            prefix=f"m12c_multiseed_geometry_seed{seed}_eval200",
            stage="C",
            mode="strict_eval",
            deterministic=True,
            save_paths=False,
        )

        rows.append({
            "seed": int(seed),
            "final_C": paths["final_C"],
            "success_rate": summary_eval["success_rate"],
            "mission_acceptability_rate": summary_eval["mission_acceptability_rate"],
            "acceptable_given_success_rate": summary_eval["acceptable_given_success_rate"],
            "mean_steps": summary_eval["mean_steps"],
            "mean_cumulative_risk": summary_eval["mean_cumulative_risk"],
            "mean_road_time": summary_eval["mean_road_time"],
            "mean_sidewalk_time": summary_eval["mean_sidewalk_time"],
            "mean_building_time": summary_eval["mean_building_time"],
            "csv_path": csv_eval,
            "summary_path": json_eval,
        })

    out = pd.DataFrame(rows)

    path = Path("/content/drive/MyDrive/RUSCUAV/paper_ready_v1/tables/m12c_geometry_multiseed_summary.csv")
    out.to_csv(path, index=False)

    display(out)
    print("Saved:", path)

    return out

import os

import json

import math

import traceback

from pathlib import Path

from datetime import datetime

import numpy as np

import pandas as pd

RUN_METHODS = {
    # Core learned baselines requested for Paper A
    "M7_geometry_only": True,
    "M8_reward_only": True,

    # Choose this OR M6_full_semantic depending on compute budget.
    # I recommend leaving obs-only ON for the observation-vs-reward ablation.
    "M8_obs_only": True,

    # Optional: expensive full semantic M6 baseline.
    # Turn ON if you want the headline full-semantic policy also replicated across seeds.
    "M6_full_semantic": False,

    # Dynamic-risk-observation RL
    "M11C_dynamic_obs_reward_only": True,
}

METHOD_ORDER = [
    "M7_geometry_only",
    "M8_reward_only",
    "M8_obs_only",
    "M6_full_semantic",
    "M11C_dynamic_obs_reward_only",
]

TRAIN_SEEDS = [1001, 2001, 3001, 4001, 5001]

N_EVAL_EPISODES = 200

EVAL_SEED0 = 5000

N_OBSTACLES = 0

EVAL_MODE = "strict_eval"

TRAIN_MODE = "fast_train"

MAX_STEPS = 1000

DETERMINISTIC_EVAL = True

MULTISEED_N_ENVS = 4

MULTISEED_DEVICE = "cpu"

MULTISEED_START_METHOD = "fork"

MULTISEED_SAVE_FREQ_STEPS = 10_000

MULTISEED_SAVE_PATHS = False

SKIP_COMPLETE_MULTISEED_RUNS = True

STOP_ON_ERROR = True

M7_TOTAL_STEPS = 200_000

M8_STEPS_A = 50_000

M8_STEPS_B = 75_000

M8_STEPS_C = 75_000

M11C_STEPS_A = 50_000

M11C_STEPS_B = 75_000

M11C_STEPS_C = 75_000

M6_STEPS_A = 150_000

M6_STEPS_B = 250_000

M6_STEPS_C = 400_000

MULTISEED_EXPERIMENT_PREFIX = "paperA_multiseed_rl_v1"

SUMMARY_METRIC_KEYS = [
    "success_rate",
    "mission_acceptability_rate",
    "acceptable_given_success_rate",
    "static_only_acceptability_rate",
    "dynamic_budget_ok_rate",

    "mean_reward",
    "mean_steps",
    "mean_dist_xy",
    "median_dist_xy",
    "mean_cargo_health",

    "mean_urban_risk",
    "mean_urban_max_risk",
    "mean_cumulative_risk",
    "mean_urban_cumulative_risk",

    "mean_dynamic_risk",
    "mean_dynamic_risk_max",
    "mean_dynamic_cumulative_risk",

    "mean_road_time",
    "mean_sidewalk_time",
    "mean_building_time",
    "mean_green_time",
    "mean_canopy_time",
    "mean_sensitive_time",

    "mean_building_collision_steps",
    "mean_canopy_collision_steps",
    "mean_sensitive_low_altitude_violation_steps",
    "mean_hard_violation_steps",
]

PRIMARY_METRICS = [
    "success_rate",
    "mission_acceptability_rate",
    "acceptable_given_success_rate",
    "dynamic_budget_ok_rate",
    "static_only_acceptability_rate",
    "mean_cumulative_risk",
    "mean_urban_cumulative_risk",
    "mean_dynamic_cumulative_risk",
    "mean_dynamic_risk_max",
    "mean_steps",
    "mean_dist_xy",
    "mean_road_time",
    "mean_sidewalk_time",
    "mean_building_time",
    "mean_hard_violation_steps",
]

RATE_METRICS = {
    "success_rate",
    "mission_acceptability_rate",
    "acceptable_given_success_rate",
    "static_only_acceptability_rate",
    "dynamic_budget_ok_rate",
}

def ms_jsonable(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): ms_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [ms_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def ms_atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(ms_jsonable(data), f, indent=2)
    os.replace(tmp, path)

def ms_bool_series(s):
    """
    Robust bool conversion for native bools and CSV-loaded string bools.
    """
    if str(s.dtype) == "bool":
        return s.astype(bool)

    def parse_bool(x):
        if isinstance(x, bool):
            return x
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if pd.isna(x):
            return False
        return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}

    return s.map(parse_bool).astype(bool)

def ms_tcrit_975(df):
    """
    Two-sided 95% t critical value.
    Uses scipy if available; otherwise falls back to common values.
    """
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, int(df)))
    except Exception:
        table = {
            1: 12.706,
            2: 4.303,
            3: 3.182,
            4: 2.776,
            5: 2.571,
            6: 2.447,
            7: 2.365,
            8: 2.306,
            9: 2.262,
            10: 2.228,
        }
        return float(table.get(int(df), 1.96))

def ms_bootstrap_ci(values, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=12345):
    """
    Episode-level bootstrap CI. This is secondary uncertainty only.
    """
    arr = np.asarray(values)

    if arr.dtype.kind in {"b"}:
        arr = arr.astype(float)
    else:
        arr = pd.to_numeric(pd.Series(arr), errors="coerce").dropna().to_numpy(dtype=float)

    if len(arr) == 0:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(int(seed))
    n = len(arr)
    boots = np.empty(int(n_boot), dtype=float)

    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(stat_fn(arr[idx]))

    alpha = 1.0 - float(ci)
    estimate = float(stat_fn(arr))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))

    return estimate, lo, hi

def ms_episode_bootstrap_secondary_table(
    df,
    method,
    train_seed,
    n_boot=2000,
    bootstrap_seed=12345,
):
    """
    Per trained model, compute episode-level bootstrap CI.
    This should be reported as secondary/supplementary uncertainty,
    not as the primary uncertainty for learned-policy claims.
    """
    metric_specs = []

    binary_cols = {
        "success_rate": "is_success",
        "mission_acceptability_rate": "mission_acceptable",
        "static_only_acceptability_rate": "mission_acceptable_static_only",
        "dynamic_budget_ok_rate": "dynamic_budget_ok",
    }

    for metric, col in binary_cols.items():
        if col in df.columns:
            metric_specs.append((metric, ms_bool_series(df[col]).astype(float).to_numpy(), np.mean))

    continuous_cols = {
        "mean_reward": "total_reward",
        "mean_steps": "steps",
        "mean_dist_xy": "dist_xy",
        "mean_cargo_health": "cargo_health",

        # Static summaries use mean_cumulative_risk.
        # Dynamic summaries often use mean_urban_cumulative_risk.
        # Save both aliases for compatibility with existing tables.
        "mean_cumulative_risk": "urban_cumulative_risk",
        "mean_urban_cumulative_risk": "urban_cumulative_risk",

        "mean_dynamic_cumulative_risk": "dynamic_cumulative_risk",
        "mean_dynamic_risk_max": "dynamic_risk_max",

        "mean_road_time": "road_time",
        "mean_sidewalk_time": "sidewalk_time",
        "mean_building_time": "building_time",
        "mean_green_time": "green_time",
        "mean_canopy_time": "canopy_time",
        "mean_sensitive_time": "sensitive_time",

        "mean_hard_violation_steps": "urban_hard_violation_steps",
    }

    for metric, col in continuous_cols.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) > 0:
                metric_specs.append((metric, vals, np.mean))

    rows = []
    for i, (metric, vals, stat_fn) in enumerate(metric_specs):
        est, lo, hi = ms_bootstrap_ci(
            vals,
            stat_fn=stat_fn,
            n_boot=n_boot,
            ci=0.95,
            seed=int(bootstrap_seed + i),
        )

        rows.append({
            "method": method,
            "train_seed": int(train_seed),
            "metric": metric,
            "estimate": est,
            "ci95_low": lo,
            "ci95_high": hi,
            "n_episodes": int(len(vals)),
            "n_boot": int(n_boot),
            "ci_scope": "episode_bootstrap_secondary",
        })

    return pd.DataFrame(rows)

def ms_summary_row_from_run(
    method,
    train_seed,
    model_path,
    summary,
    csv_path,
    summary_path,
    drive_dirs,
):
    row = {
        "method": str(method),
        "train_seed": int(train_seed),
        "run_status": "complete",

        "n_episodes": int(summary.get("n_episodes", N_EVAL_EPISODES)),
        "eval_seed0": int(EVAL_SEED0),
        "n_eval_episodes_requested": int(N_EVAL_EPISODES),

        "final_model_path": str(model_path),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "experiment_dir": str(drive_dirs["exp_dir"]),

        "eval_status": summary.get("status", None),
        "completed_at": datetime.now().isoformat(),
    }

    for key in SUMMARY_METRIC_KEYS:
        value = summary.get(key, np.nan)
        try:
            row[key] = float(value)
        except Exception:
            row[key] = np.nan

    return row

def ms_training_seed_ci_table(seed_level_df, metrics=SUMMARY_METRIC_KEYS):
    """
    Primary uncertainty table:
    for each method and metric, summarize across training seeds.
    CI is a 95% t-CI over seed-level metric values.
    """
    df = seed_level_df.copy()

    if "run_status" in df.columns:
        df = df[df["run_status"].astype(str) == "complete"].copy()

    rows = []

    for method, g in df.groupby("method", sort=False):
        for metric in metrics:
            if metric not in g.columns:
                continue

            vals = pd.to_numeric(g[metric], errors="coerce").dropna().to_numpy(dtype=float)

            if len(vals) == 0:
                continue

            n = int(len(vals))
            mean = float(np.mean(vals))

            if n >= 2:
                std = float(np.std(vals, ddof=1))
                sem = float(std / math.sqrt(n))
                tcrit = ms_tcrit_975(n - 1)
                ci_low = float(mean - tcrit * sem)
                ci_high = float(mean + tcrit * sem)
                ci_method = f"training_seed_t_CI_95_df{n-1}"
            else:
                std = np.nan
                sem = np.nan
                ci_low = np.nan
                ci_high = np.nan
                ci_method = "undefined_n_lt_2"

            rows.append({
                "method": method,
                "metric": metric,
                "n_training_seeds": n,
                "training_seeds": ",".join(str(int(s)) for s in g["train_seed"].dropna().astype(int).tolist()),

                "mean": mean,
                "std": std,
                "sem": sem,
                "ci95_low": ci_low,
                "ci95_high": ci_high,

                "ci_scope": "training_seed_primary",
                "ci_method": ci_method,
            })

    return pd.DataFrame(rows)

def ms_format_value(metric, x):
    if pd.isna(x):
        return ""

    if metric in RATE_METRICS:
        return f"{100.0 * float(x):.1f}%"

    return f"{float(x):.3f}"

def ms_format_mean_std_ci(row):
    metric = row["metric"]
    mean = ms_format_value(metric, row["mean"])
    std = ms_format_value(metric, row["std"])

    lo = ms_format_value(metric, row["ci95_low"])
    hi = ms_format_value(metric, row["ci95_high"])

    if lo == "" or hi == "":
        return f"{mean} ± {std}"

    return f"{mean} ± {std} [{lo}, {hi}]"

def ms_experiment_dirs(method_id, train_seed):
    exp_name = f"{MULTISEED_EXPERIMENT_PREFIX}_{method_id}_trainseed{int(train_seed)}"
    return setup_ruscuav_drive_project(
        drive_root=str(DRIVE_ROOT),
        experiment_name=exp_name,
    )

def ms_existing_complete_row(method_id, train_seed):
    if not SEED_LEVEL_PARTIAL_PATH.exists():
        return None

    df = pd.read_csv(SEED_LEVEL_PARTIAL_PATH)

    if len(df) == 0:
        return None

    mask = (
        (df["method"].astype(str) == str(method_id)) &
        (df["train_seed"].astype(int) == int(train_seed)) &
        (df.get("run_status", "").astype(str) == "complete")
    )

    if mask.sum() == 0:
        return None

    row = df.loc[mask].iloc[-1].to_dict()

    try:
        if int(row.get("n_episodes", -1)) < int(N_EVAL_EPISODES):
            return None
    except Exception:
        return None

    for col in ["final_model_path", "csv_path", "summary_path"]:
        p = row.get(col, None)
        if not p or not Path(str(p)).exists():
            return None

    return row

def ms_upsert_seed_level_row(row):
    if SEED_LEVEL_PARTIAL_PATH.exists():
        df = pd.read_csv(SEED_LEVEL_PARTIAL_PATH)
    else:
        df = pd.DataFrame()

    if len(df) > 0 and {"method", "train_seed"}.issubset(df.columns):
        keep = ~(
            (df["method"].astype(str) == str(row["method"])) &
            (df["train_seed"].astype(int) == int(row["train_seed"]))
        )
        df = df.loc[keep].copy()

    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df.sort_values(["method", "train_seed"]).reset_index(drop=True)
    df.to_csv(SEED_LEVEL_PARTIAL_PATH, index=False)

    return df

def ms_upsert_episode_ci(ep_ci_df):
    if ep_ci_df is None or len(ep_ci_df) == 0:
        return pd.DataFrame()

    method = str(ep_ci_df["method"].iloc[0])
    train_seed = int(ep_ci_df["train_seed"].iloc[0])

    if EPISODE_CI_PARTIAL_PATH.exists():
        df = pd.read_csv(EPISODE_CI_PARTIAL_PATH)
    else:
        df = pd.DataFrame()

    if len(df) > 0 and {"method", "train_seed"}.issubset(df.columns):
        keep = ~(
            (df["method"].astype(str) == method) &
            (df["train_seed"].astype(int) == train_seed)
        )
        df = df.loc[keep].copy()

    df = pd.concat([df, ep_ci_df], ignore_index=True)
    df = df.sort_values(["method", "train_seed", "metric"]).reset_index(drop=True)
    df.to_csv(EPISODE_CI_PARTIAL_PATH, index=False)

    return df

def ms_run_one_train_eval(method_id, train_seed):
    method_id = str(method_id)
    train_seed = int(train_seed)

    print("=" * 120)
    print(f"MULTI-SEED RUN | method={method_id} | train_seed={train_seed}")
    print("=" * 120)

    drive_dirs = ms_experiment_dirs(method_id, train_seed)

    eval_prefix = (
        f"{method_id}_trainseed{train_seed}_"
        f"strict_eval_n{N_EVAL_EPISODES}_evalseed{EVAL_SEED0}"
    )

    if method_id == "M7_geometry_only":
        model_path = train_geo_baseline_m7_drive_resume(
            world_path=WORLD_PATH_DRIVE,
            drive_dirs=drive_dirs,
            total_steps=int(M7_TOTAL_STEPS),
            seed=train_seed,
            n_obstacles=N_OBSTACLES,
            save_freq_steps=MULTISEED_SAVE_FREQ_STEPS,
            device=MULTISEED_DEVICE,
            n_envs=MULTISEED_N_ENVS,
            mode=TRAIN_MODE,
            start_method=MULTISEED_START_METHOD,
        )

        df_eval, summary, csv_path, summary_path = evaluate_geo_baseline_m7_drive(
            model_path=model_path,
            world=world,
            drive_dirs=drive_dirs,
            n_episodes=N_EVAL_EPISODES,
            seed0=EVAL_SEED0,
            n_obstacles=N_OBSTACLES,
            deterministic=DETERMINISTIC_EVAL,
            prefix=eval_prefix,
            mode=EVAL_MODE,
            max_steps=MAX_STEPS,
            save_paths=MULTISEED_SAVE_PATHS,
        )

    elif method_id in {"M8_reward_only", "M8_obs_only"}:
        variant_name = {
            "M8_reward_only": "reward_only",
            "M8_obs_only": "obs_only",
        }[method_id]

        paths = train_m8_ablation_variant_drive_resume(
            world_path=WORLD_PATH_DRIVE,
            variant_name=variant_name,
            drive_dirs=drive_dirs,
            seed=train_seed,
            n_obstacles=N_OBSTACLES,
            steps_A=M8_STEPS_A,
            steps_B=M8_STEPS_B,
            steps_C=M8_STEPS_C,
            save_freq_steps=MULTISEED_SAVE_FREQ_STEPS,
            device=MULTISEED_DEVICE,
            n_envs=MULTISEED_N_ENVS,
            mode=TRAIN_MODE,
            start_method=MULTISEED_START_METHOD,
        )

        model_path = paths["final_C"]

        df_eval, summary, csv_path, summary_path = evaluate_m8_ablation_variant_drive(
            model_path=model_path,
            world=world,
            variant_name=variant_name,
            drive_dirs=drive_dirs,
            n_episodes=N_EVAL_EPISODES,
            seed0=EVAL_SEED0,
            n_obstacles=N_OBSTACLES,
            deterministic=DETERMINISTIC_EVAL,
            prefix=eval_prefix,
            mode=EVAL_MODE,
            max_steps=MAX_STEPS,
            save_paths=MULTISEED_SAVE_PATHS,
        )

    elif method_id == "M6_full_semantic":
        paths = train_ruscuav_m6_vec_drive_resume(
            world_path=WORLD_PATH_DRIVE,
            drive_dirs=drive_dirs,
            seed=train_seed,
            n_obstacles=N_OBSTACLES,
            steps_A=M6_STEPS_A,
            steps_B=M6_STEPS_B,
            steps_C=M6_STEPS_C,
            save_freq_steps=MULTISEED_SAVE_FREQ_STEPS,
            device=MULTISEED_DEVICE,
            n_envs=MULTISEED_N_ENVS,
            mode=TRAIN_MODE,
            start_method=MULTISEED_START_METHOD,
        )

        model_path = paths["final_C"]

        df_eval, summary, csv_path, summary_path = evaluate_and_save_ruscuav_m4_drive(
            model_path=model_path,
            world=world,
            drive_dirs=drive_dirs,
            stage="C",
            n_episodes=N_EVAL_EPISODES,
            seed0=EVAL_SEED0,
            n_obstacles=N_OBSTACLES,
            deterministic=DETERMINISTIC_EVAL,
            prefix=eval_prefix,
            mode=EVAL_MODE,
            max_steps=MAX_STEPS,
            save_paths=MULTISEED_SAVE_PATHS,
        )

    elif method_id == "M11C_dynamic_obs_reward_only":
        paths = train_m11c_dynamic_obs_reward_only_drive_resume(
            world_path=WORLD_PATH_DRIVE,
            dynamic_config=m11_dyn_cfg_v1,
            drive_dirs=drive_dirs,
            seed=train_seed,
            n_obstacles=N_OBSTACLES,
            steps_A=M11C_STEPS_A,
            steps_B=M11C_STEPS_B,
            steps_C=M11C_STEPS_C,
            save_freq_steps=MULTISEED_SAVE_FREQ_STEPS,
            device=MULTISEED_DEVICE,
            n_envs=MULTISEED_N_ENVS,
            mode=TRAIN_MODE,
            start_method=MULTISEED_START_METHOD,
        )

        model_path = paths["final_C"]

        df_eval, summary, csv_path, summary_path = evaluate_m11c_dynamic_obs_rl_drive(
            model_path=model_path,
            world=world,
            drive_dirs=drive_dirs,
            dynamic_config=m11_dyn_cfg_v1,
            n_episodes=N_EVAL_EPISODES,
            seed0=EVAL_SEED0,
            prefix=eval_prefix,
            stage="C",
            mode=EVAL_MODE,
            save_paths=MULTISEED_SAVE_PATHS,
        )

    else:
        raise ValueError(f"Unknown method_id={method_id}")

    seed_row = ms_summary_row_from_run(
        method=method_id,
        train_seed=train_seed,
        model_path=model_path,
        summary=summary,
        csv_path=csv_path,
        summary_path=summary_path,
        drive_dirs=drive_dirs,
    )

    episode_ci = ms_episode_bootstrap_secondary_table(
        df=df_eval,
        method=method_id,
        train_seed=train_seed,
        n_boot=2000,
        bootstrap_seed=10_000 + train_seed,
    )

    print("=" * 120)
    print(f"COMPLETE | method={method_id} | train_seed={train_seed}")
    print(f"SR  = {seed_row.get('success_rate', np.nan):.4f}")
    print(f"MAR = {seed_row.get('mission_acceptability_rate', np.nan):.4f}")
    if "dynamic_budget_ok_rate" in seed_row and np.isfinite(seed_row["dynamic_budget_ok_rate"]):
        print(f"DynOK = {seed_row['dynamic_budget_ok_rate']:.4f}")
    print("CSV:", csv_path)
    print("=" * 120)

    return seed_row, episode_ci

from pathlib import Path

import os

import math

import json

import warnings

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch

from matplotlib.ticker import PercentFormatter

from matplotlib.colors import ListedColormap, BoundaryNorm

FIG_FORMATS = ["pdf", "svg", "png"]

PNG_DPI = 600

FIG_W_SINGLE = 3.50

FIG_W_DOUBLE = 7.16

def read_csv_required(name):
    path = TABLE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing required table: {path}")
    return pd.read_csv(path)

def read_csv_optional(name):
    path = TABLE_DIR / name
    return pd.read_csv(path) if path.exists() else None

def save_pub_fig(fig, stem, close=False):
    paths = []

    for ext in FIG_FORMATS:
        path = PUB_FIG_DIR / f"{stem}.{ext}"
        fig.savefig(
            path,
            bbox_inches="tight",
            dpi=PNG_DPI if ext == "png" else None,
        )
        paths.append(str(path))

    if close:
        plt.close(fig)

    print(f"Saved {stem}:")
    for p in paths:
        print("  ", p)

    return paths

def panel_label(ax, text):
    ax.text(
        -0.12,
        1.04,
        text,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

def ordered_rows(df, labels):
    available = set(df["label"].astype(str))
    existing = [lab for lab in labels if lab in available]
    missing = [lab for lab in labels if lab not in available]

    if missing:
        print("[WARN] Missing labels skipped:", missing)

    return df.set_index("label").loc[existing].reset_index()

def as_percent_axis(ax):
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", linewidth=0.4, alpha=0.35)

RATE_METRICS = {
    "success_rate",
    "mission_acceptability_rate",
    "acceptable_given_success_rate",
    "static_only_acceptability_rate",
    "dynamic_budget_ok_rate",
}

SHORT_LABEL = {
    "M7_geometry_only_RL": "M7 geom.",
    "M8_obs_only_RL": "M8 obs.",
    "M6_full_semantic_RL": "M6 full sem.",
    "M8_reward_only_RL": "M8 reward",
    "M9_static_risk_Astar": "M9 A*",
    "M10_partial_risk_receding_Astar": "M10 partial",
    "M10_fullmap_receding_Astar": "M10 full",

    "M11_static_trained_RL_dynamic_eval": "Static-trained RL",
    "M11_dynamic_receding_Astar": "Receding A*",
    "M11B_dynamic_trained_RL_no_dynamic_obs": "Dyn-trained RL\nno dyn. obs.",
    "M11C_dynamic_trained_RL_dynamic_obs": "Dyn-trained RL\n+ dyn. obs.",

    "Vienna_M7_geometry_only": "Geometry RL",
    "Vienna_M8_reward_transfer": "Zero-shot RL",
    "Vienna_M8_reward_finetuned": "Short FT RL",
    "Vienna_M8_reward_extended": "Extended FT RL",
    "Vienna_footway_only_M10_planner_200": "Receding A*",
}

MULTISEED_LABEL_MAP = {
    "M7_geometry_only_RL": "M7_geometry_only",
    "M8_reward_only_RL": "M8_reward_only",
    "M8_obs_only_RL": "M8_obs_only",
    "M6_full_semantic_RL": "M6_full_semantic",
    "M11C_dynamic_trained_RL_dynamic_obs": "M11C_dynamic_obs_reward_only",
}

BOOT_LABEL_MAP = {
    "M7_geometry_only_RL": "M7_geometry_only_synthetic",
    "M8_reward_only_RL": "M8_reward_only_synthetic",
    "M10_fullmap_receding_Astar": "M10_fullmap_receding_synthetic",

    "M11_static_trained_RL_dynamic_eval": "M11_staticRL_dynamic",
    "M11_dynamic_receding_Astar": "M11_planner_dynamic",
    "M11C_dynamic_trained_RL_dynamic_obs": "M11C_dynamicObsRL",

    "Vienna_M8_reward_extended": "Vienna_reward_extended",
    "Vienna_M7_geometry_only": "Vienna_geometry_only",
}

METRIC_ALIAS_FOR_BOOT = {
    "mean_cumulative_risk": "mean_urban_cumulative_risk",
    "mean_urban_cumulative_risk": "mean_urban_cumulative_risk",
}

METRIC_ALIAS_FOR_MULTI = {
    "mean_urban_cumulative_risk": "mean_cumulative_risk",
}

def wilson_ci_rate(p, n, z=1.96):
    if n is None or n <= 0 or not np.isfinite(p):
        return np.nan, np.nan

    k = float(p) * int(n)
    phat = k / int(n)

    denom = 1.0 + z**2 / int(n)
    center = (phat + z**2 / (2 * int(n))) / denom
    half = z * math.sqrt(
        (phat * (1 - phat) + z**2 / (4 * int(n))) / int(n)
    ) / denom

    return max(0.0, center - half), min(1.0, center + half)

def estimate_ci_for_row(row, metric):
    """
    Returns:
        estimate, ci_low, ci_high, ci_source

    Priority:
        1. training-seed CI for learned policies, if multi-seed table exists;
        2. episode-bootstrap CI from paper_final_bootstrap_ci_table.csv;
        3. Wilson CI for rates;
        4. raw table value with no CI.
    """
    label = str(row["label"])
    n = int(row.get("n_episodes", 0)) if "n_episodes" in row else None

    # 1. Primary uncertainty for learned policies.
    if multiseed_ci is not None and label in MULTISEED_LABEL_MAP:
        method_id = MULTISEED_LABEL_MAP[label]
        metric_candidates = [metric, METRIC_ALIAS_FOR_MULTI.get(metric, metric)]

        for met in metric_candidates:
            sub = multiseed_ci[
                (multiseed_ci["method"].astype(str) == method_id)
                & (multiseed_ci["metric"].astype(str) == met)
            ]

            if len(sub) > 0:
                r = sub.iloc[-1]
                return (
                    float(r["mean"]),
                    float(r["ci95_low"]),
                    float(r["ci95_high"]),
                    "training-seed CI",
                )

    # 2. Secondary episode-bootstrap CI.
    if bootstrap_ci is not None and label in BOOT_LABEL_MAP:
        boot_label = BOOT_LABEL_MAP[label]
        boot_metric = METRIC_ALIAS_FOR_BOOT.get(metric, metric)

        sub = bootstrap_ci[
            (bootstrap_ci["label"].astype(str) == boot_label)
            & (bootstrap_ci["metric"].astype(str) == boot_metric)
        ]

        if len(sub) > 0:
            r = sub.iloc[-1]
            return (
                float(r["estimate"]),
                float(r["ci95_low"]),
                float(r["ci95_high"]),
                "episode-bootstrap CI",
            )

    # 3. Fallback.
    value = float(row[metric]) if metric in row and pd.notna(row[metric]) else np.nan

    if metric in RATE_METRICS and np.isfinite(value):
        lo, hi = wilson_ci_rate(value, n)
        return value, lo, hi, "Wilson episode CI"

    return value, np.nan, np.nan, "no CI"

def asymmetric_yerr(values, lows, highs):
    values = np.asarray(values, dtype=float)
    lows = np.asarray(lows, dtype=float)
    highs = np.asarray(highs, dtype=float)

    lower = np.where(np.isfinite(lows), np.maximum(0.0, values - lows), 0.0)
    upper = np.where(np.isfinite(highs), np.maximum(0.0, highs - values), 0.0)

    return np.vstack([lower, upper])

from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

def save_pub_fig_aliases(fig, stems, close=False):
    """
    Save the same corrected figure under one or more stems.
    This is useful if earlier notebook versions used slightly different names.
    """
    stems = list(dict.fromkeys(stems))

    saved = []
    for stem in stems:
        if "save_pub_fig" in globals():
            saved.extend(save_pub_fig(fig, stem, close=False))
        elif "tits_save_pub_fig" in globals():
            saved.extend(tits_save_pub_fig(fig, stem, close=False))
        else:
            for ext in ["pdf", "svg", "png"]:
                p = PUB_FIG_DIR / f"{stem}.{ext}"
                fig.savefig(
                    p,
                    bbox_inches="tight",
                    dpi=PNG_DPI if ext == "png" else None,
                )
                saved.append(str(p))
                print("Saved:", p)

    if close:
        plt.close(fig)

    return saved

def outside_panel_legend(ax, ncol=2, y=1.18, fontsize=7):
    """
    Move the current axis legend outside the plot area, above the panel.
    """
    handles, labels = ax.get_legend_handles_labels()

    if len(handles) == 0:
        return None

    return ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        handlelength=1.8,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

def add_box(ax, xy, width, height, text, fc="white", lw=0.8):
    x, y = xy

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor="black",
        facecolor=fc,
    )

    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=7,
    )

    return box

def add_arrow(ax, start, end):
    arr = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=8,
        linewidth=0.8,
        color="black",
        shrinkA=4,
        shrinkB=4,
    )

    ax.add_patch(arr)
    return arr

def get_world_object_for_figures():
    if "world" in globals():
        return globals()["world"]

    candidates = []

    if "WORLD_PATH_DRIVE" in globals():
        candidates.append(Path(str(globals()["WORLD_PATH_DRIVE"])))

    candidates.append(DRIVE_ROOT / "worlds" / "AOI_FAKE_001_urban_world_layers.npz")
    candidates.append(PAPER_READY_ROOT / "worlds" / "AOI_FAKE_001_urban_world_layers.npz")

    for path in candidates:
        if path.exists() and "load_world_from_drive" in globals():
            return globals()["load_world_from_drive"](str(path))

        if path.exists() and "UrbanWorldLayers" in globals():
            return globals()["UrbanWorldLayers"].load_npz(str(path))

    return None

def build_semantic_map(world_obj):
    sem = np.zeros(world_obj.shape, dtype=np.int16)

    sem[np.asarray(world_obj.green_mask) > 0.5] = 1
    sem[np.asarray(world_obj.road_mask) > 0.5] = 2
    sem[np.asarray(world_obj.sidewalk_mask) > 0.5] = 3
    sem[np.asarray(world_obj.canopy_mask) > 0.5] = 4
    sem[np.asarray(world_obj.building_mask) > 0.5] = 5
    sem[np.asarray(world_obj.sensitive_mask) > 0.5] = 6

    return sem

STATIC_ORDER = [
    "M7_geometry_only_RL",
    "M8_obs_only_RL",
    "M6_full_semantic_RL",
    "M8_reward_only_RL",
    "M9_static_risk_Astar",
    "M10_partial_risk_receding_Astar",
    "M10_fullmap_receding_Astar",
]

DYNAMIC_ORDER = [
    "M11_static_trained_RL_dynamic_eval",
    "M11_dynamic_receding_Astar",
    "M11B_dynamic_trained_RL_no_dynamic_obs",
    "M11C_dynamic_trained_RL_dynamic_obs",
]

VIENNA_ORDER = [
    "Vienna_M7_geometry_only",
    "Vienna_M8_reward_transfer",
    "Vienna_M8_reward_finetuned",
    "Vienna_M8_reward_extended",
    "Vienna_footway_only_M10_planner_200",
]

from pathlib import Path

import os

import json

import math

import warnings

import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from matplotlib import colors as mcolors

def tits_save_pub_fig(fig, stem, close=False):
    """
    Save figure as PDF/SVG/PNG.
    Uses prior save_pub_fig if it exists; otherwise local fallback.
    """
    if "save_pub_fig" in globals():
        return save_pub_fig(fig, stem, close=close)

    out_paths = []
    for ext in ["pdf", "svg", "png"]:
        p = PUB_FIG_DIR / f"{stem}.{ext}"
        fig.savefig(
            p,
            bbox_inches="tight",
            dpi=PNG_DPI if ext == "png" else None,
        )
        out_paths.append(str(p))

    if close:
        plt.close(fig)

    print(f"Saved {stem}:")
    for p in out_paths:
        print("  ", p)

    return out_paths

def tits_read_table(name):
    p = TABLE_DIR / name
    if not p.exists():
        raise FileNotFoundError(f"Missing table: {p}")
    return pd.read_csv(p)

def tits_as_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)

    if pd.isna(x):
        return False

    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}

def tits_world_extent(world):
    h, w = np.asarray(world.terrain_z).shape
    return [0, w, h, 0]

def tits_rgba(color, alpha):
    return mcolors.to_rgba(color, alpha)

def tits_load_synthetic_world():
    if "world" in globals():
        return globals()["world"]

    candidates = []

    if "WORLD_PATH_DRIVE" in globals():
        candidates.append(Path(str(globals()["WORLD_PATH_DRIVE"])))

    candidates += [
        DRIVE_ROOT / "worlds" / "AOI_FAKE_001_urban_world_layers.npz",
        PAPER_READY_ROOT / "worlds" / "AOI_FAKE_001_urban_world_layers.npz",
        Path("/mnt/data/paper_ready_v1/worlds/AOI_FAKE_001_urban_world_layers.npz"),
    ]

    for p in candidates:
        if p.exists():
            if "load_world_from_drive" in globals():
                return load_world_from_drive(str(p))
            return UrbanWorldLayers.load_npz(str(p))

    raise RuntimeError(
        "Cannot load synthetic world. Expected global `world` or "
        "AOI_FAKE_001_urban_world_layers.npz on Drive."
    )

def tits_load_vienna_world():
    if "vienna_world_footway_only" in globals():
        return globals()["vienna_world_footway_only"]

    candidates = []

    if "M12C_VIENNA_FOOTWAY_WORLD_PATH" in globals():
        candidates.append(Path(str(globals()["M12C_VIENNA_FOOTWAY_WORLD_PATH"])))

    candidates += [
        DRIVE_ROOT / "worlds" / "vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz",
        PAPER_READY_ROOT / "worlds" / "vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz",
        Path("/mnt/data/paper_ready_v1/worlds/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz"),
    ]

    for p in candidates:
        if p.exists():
            return UrbanWorldLayers.load_npz(str(p))

    raise RuntimeError(
        "Cannot load Vienna world. Expected global `vienna_world_footway_only` "
        "or vienna_prater_leopoldstadt_osm_1km_v1_footway_only_urban_world_layers.npz."
    )

SEMANTIC_STYLE = {
    "green":     ("#2ca25f", 0.30, "green"),
    "canopy":    ("#006d2c", 0.32, "canopy"),
    "road":      ("#f16913", 0.42, "road"),
    "sidewalk":  ("#fdae6b", 0.46, "sidewalk / footway"),
    "building":  ("#737373", 0.58, "building"),
    "sensitive": ("#cb181d", 0.42, "sensitive"),
}

TRAJ_STYLE = {
    "M7 geometry-only RL": {
        "color": "#756bb1",
        "linestyle": "--",
        "linewidth": 1.7,
    },
    "M8 reward-only RL": {
        "color": "#238b45",
        "linestyle": "-",
        "linewidth": 1.9,
    },
    "M10 receding A*": {
        "color": "#08519c",
        "linestyle": "-.",
        "linewidth": 1.8,
    },
    "Extended FT RL": {
        "color": "#238b45",
        "linestyle": "-",
        "linewidth": 1.9,
    },
    "Receding A*": {
        "color": "#08519c",
        "linestyle": "-.",
        "linewidth": 1.8,
    },
    "Zero-shot RL": {
        "color": "#756bb1",
        "linestyle": "--",
        "linewidth": 1.7,
    },
}

def tits_overlay_mask(ax, world, mask_name, color, alpha):
    mask = np.asarray(getattr(world, mask_name), dtype=np.float32) > 0.5
    if not np.any(mask):
        return

    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[mask] = tits_rgba(color, alpha)

    ax.imshow(
        rgba,
        origin="upper",
        extent=tits_world_extent(world),
        interpolation="nearest",
    )

def tits_draw_semantic_world(ax, world, title=None, legend=False):
    terrain = np.asarray(world.terrain_z, dtype=np.float32)

    ax.imshow(
        terrain,
        origin="upper",
        extent=tits_world_extent(world),
        cmap="Greys",
        alpha=0.22,
        interpolation="nearest",
    )

    overlay_order = [
        ("green_mask", "green"),
        ("canopy_mask", "canopy"),
        ("road_mask", "road"),
        ("sidewalk_mask", "sidewalk"),
        ("building_mask", "building"),
        ("sensitive_mask", "sensitive"),
    ]

    for mask_name, key in overlay_order:
        color, alpha, _ = SEMANTIC_STYLE[key]
        tits_overlay_mask(ax, world, mask_name, color, alpha)

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(0, world.width)
    ax.set_ylim(world.height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(linewidth=0.25, alpha=0.20)

    if legend:
        handles = [
            Patch(facecolor=color, edgecolor="black", alpha=alpha, label=label)
            for color, alpha, label in SEMANTIC_STYLE.values()
        ]
        ax.legend(
            handles=handles,
            loc="lower right",
            frameon=True,
            fontsize=6,
            ncol=1,
        )

def tits_risk_map(world, agl_m):
    if "UrbanRiskModel" in globals():
        try:
            rm = UrbanRiskModel(
                world=world,
                profile="balanced",
                hard_sensitive_zones=True,
                include_global_altitude_band_cost=True,
            )
            return rm.risk_map_at_agl(float(agl_m))
        except TypeError:
            rm = UrbanRiskModel(
                world=world,
                profile="balanced",
                hard_sensitive_zones=True,
            )
            return rm.risk_map_at_agl(float(agl_m))

    if "compute_base_semantic_risk_map" in globals():
        return compute_base_semantic_risk_map(world, risk_profile="balanced")

    risk = np.zeros(world.shape, dtype=np.float32)
    risk += 0.10 * np.asarray(world.green_mask, dtype=np.float32)
    risk += 0.20 * np.asarray(world.canopy_density, dtype=np.float32)
    risk += 3.00 * np.asarray(world.building_mask, dtype=np.float32)
    risk += 6.00 * np.asarray(world.road_mask, dtype=np.float32)
    risk += 12.00 * np.asarray(world.sidewalk_mask, dtype=np.float32)
    risk += 50.00 * np.asarray(world.sensitive_mask, dtype=np.float32)
    return risk

def tits_draw_risk_world(ax, world, agl_m=40.0, title=None, colorbar=True):
    risk = tits_risk_map(world, agl_m=agl_m)

    im = ax.imshow(
        risk,
        origin="upper",
        extent=tits_world_extent(world),
        cmap="inferno",
        interpolation="nearest",
    )

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(0, world.width)
    ax.set_ylim(world.height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(linewidth=0.25, alpha=0.20)

    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("urban risk", fontsize=7)

    return im

def tits_draw_obstacle_height(ax, world, title=None, colorbar=True):
    obstacle_h = np.maximum(
        np.asarray(world.building_height, dtype=np.float32),
        np.asarray(world.canopy_height, dtype=np.float32),
    )

    im = ax.imshow(
        obstacle_h,
        origin="upper",
        extent=tits_world_extent(world),
        cmap="magma",
        interpolation="nearest",
    )

    if title is not None:
        ax.set_title(title)

    ax.set_xlim(0, world.width)
    ax.set_ylim(world.height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(linewidth=0.25, alpha=0.20)

    if colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("height [m]", fontsize=7)

    return im

def tits_add_paths(ax, rollouts, show_endpoints=True):
    for r in rollouts:
        path = np.asarray(r["path"], dtype=np.float32)
        label = r["label"]
        style = TRAJ_STYLE.get(label, {})

        ax.plot(
            path[:, 0],
            path[:, 1],
            label=label,
            color=style.get("color", None),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 1.7),
            zorder=5,
        )

        if show_endpoints:
            ax.scatter(
                path[-1, 0],
                path[-1, 1],
                s=26,
                marker="x",
                color=style.get("color", "black"),
                linewidth=1.2,
                zorder=6,
            )

    # common start marker, usually same seed and same reset
    p0 = np.asarray(rollouts[0]["path"], dtype=np.float32)[0]
    ax.scatter(
        p0[0],
        p0[1],
        s=34,
        marker="o",
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=7,
        label="start",
    )

def tits_rollout_metrics_text(rollouts):
    lines = []

    for r in rollouts:
        row = r.get("row", {})
        label = r["label"]

        success = tits_as_bool(row.get("is_success", False))
        acceptable = tits_as_bool(row.get("mission_acceptable", False))
        term = row.get("termination", "NA")

        risk = row.get("urban_cumulative_risk", np.nan)
        road = row.get("road_time", np.nan)
        sidewalk = row.get("sidewalk_time", np.nan)
        steps = row.get("steps", np.nan)

        try:
            risk_txt = f"{float(risk):.1f}"
        except Exception:
            risk_txt = "NA"

        try:
            road_txt = f"{float(road):.1f}"
        except Exception:
            road_txt = "NA"

        try:
            sw_txt = f"{float(sidewalk):.1f}"
        except Exception:
            sw_txt = "NA"

        lines.append(
            f"{label}: succ={success}, acc={acceptable}, "
            f"term={term}, risk={risk_txt}, road={road_txt}s, sidewalk={sw_txt}s, steps={steps}"
        )

    return "\n".join(lines)

def tits_table_row(table_name, label):
    df = tits_read_table(table_name)
    sub = df[df["label"].astype(str) == str(label)]

    if len(sub) == 0:
        raise KeyError(f"Label {label!r} not found in {table_name}.")

    return sub.iloc[0].to_dict()

def tits_eval_df_from_table_label(table_name, label):
    row = tits_table_row(table_name, label)

    csv_path = Path(str(row.get("csv_path", "")))
    if csv_path.exists():
        return pd.read_csv(csv_path), csv_path

    df_var = row.get("df_var", None)
    if isinstance(df_var, str) and df_var in globals():
        return globals()[df_var].copy(), csv_path

    return None, csv_path

def tits_summary_json_from_table_label(table_name, label):
    row = tits_table_row(table_name, label)
    summary_path = Path(str(row.get("summary_path", "")))

    if summary_path.exists():
        with open(summary_path, "r") as f:
            return json.load(f), summary_path

    return {}, summary_path

def tits_get_model_path(table_name, label):
    summary, _ = tits_summary_json_from_table_label(table_name, label)

    for key in ["model_path", "final_model_path"]:
        p = summary.get(key, None)
        if p is not None and Path(str(p)).exists():
            return Path(str(p))

    # Runtime fallbacks for common variables.
    fallback = {
        "M7_geometry_only_RL": [
            "final_geo_path",
            "final_geo_path_full",
        ],
        "M8_reward_only_RL": [
            "m8_reward_only_paths",
            "m8_reward_only_model_path",
        ],
        "M8_obs_only_RL": [
            "m8_obs_only_paths",
        ],
        "M6_full_semantic_RL": [
            "final_C_path",
            "train_paths_m6",
        ],
        "Vienna_M8_reward_extended": [
            "latest_reward_ext_path",
            "m12c_reward_extended_paths",
        ],
        "Vienna_M8_reward_transfer": [
            "m8_reward_only_model_path",
        ],
        "Vienna_M8_reward_finetuned": [
            "m12c_reward_paths",
        ],
    }

    for var in fallback.get(label, []):
        if var not in globals():
            continue

        obj = globals()[var]

        if isinstance(obj, (str, Path)) and Path(str(obj)).exists():
            return Path(str(obj))

        if isinstance(obj, dict):
            for k in ["final_C", "latest", "final"]:
                p = obj.get(k, None)
                if p is not None and Path(str(p)).exists():
                    return Path(str(p))

    raise RuntimeError(
        f"Cannot find model path for {label}. "
        "Run the corresponding training/evaluation cells or check summary JSON."
    )

def tits_load_saved_path(table_name, label, seed):
    """
    Try to load an already saved .npy trajectory from eval_dir/paths.
    Works for RL path.npy and M9/M10 actual_path.npy naming conventions.
    """
    df, csv_path = tits_eval_df_from_table_label(table_name, label)

    if df is None or not csv_path.exists():
        return None

    if "seed" not in df.columns:
        return None

    sub = df[df["seed"].astype(int) == int(seed)]
    if len(sub) == 0:
        return None

    row = sub.iloc[0].to_dict()
    ep = int(row["episode"])

    eval_dir = csv_path.parent
    paths_dir = eval_dir / "paths"

    patterns = [
        paths_dir / f"episode_{ep:05d}_seed_{int(seed)}_path.npy",
        paths_dir / f"episode_{ep:05d}_seed_{int(seed)}_actual_path.npy",
        eval_dir / "planned_paths" / f"episode_{ep:05d}_seed_{int(seed)}_planned_path.npy",
    ]

    for p in patterns:
        if p.exists():
            path = np.load(p)
            return {
                "path": path,
                "row": row,
                "path_file": str(p),
            }

    return None

def tits_pick_synthetic_contrast_seed():
    rvg = tits_read_table("paired_synthetic_reward_vs_geometry.csv")
    rvp = tits_read_table("paired_synthetic_reward_vs_fullmap_receding_planner.csv")

    merged = rvg.merge(
        rvp[
            [
                "a_seed",
                "b_is_success",
                "b_mission_acceptable",
                "b_urban_cumulative_risk",
                "b_steps",
            ]
        ],
        on="a_seed",
        suffixes=("", "_planner"),
    )

    # Best qualitative contrast:
    # reward-only accepted, geometry succeeds but is not acceptable, planner accepted.
    cond = (
        (merged["a_mission_acceptable"].map(tits_as_bool) == True)
        & (merged["b_is_success"].map(tits_as_bool) == True)
        & (merged["b_mission_acceptable"].map(tits_as_bool) == False)
        & (merged["b_mission_acceptable_planner"].map(tits_as_bool) == True)
    )

    candidates = merged[cond].copy()

    if len(candidates) > 0:
        candidates["contrast_score"] = (
            candidates["b_urban_cumulative_risk"].astype(float)
            - candidates["a_urban_cumulative_risk"].astype(float)
        )
        best = candidates.sort_values("contrast_score", ascending=False).iloc[0]
        return int(best["a_seed"]), int(best["a_episode"])

    # Fallback: reward accepted and planner accepted, geometry not accepted.
    cond = (
        (merged["a_mission_acceptable"].map(tits_as_bool) == True)
        & (merged["b_mission_acceptable"].map(tits_as_bool) == False)
        & (merged["b_mission_acceptable_planner"].map(tits_as_bool) == True)
    )

    candidates = merged[cond].copy()

    if len(candidates) > 0:
        best = candidates.iloc[0]
        return int(best["a_seed"]), int(best["a_episode"])

    # Last fallback: first seed.
    best = merged.iloc[0]
    return int(best["a_seed"]), int(best["a_episode"])

def tits_static_rollout_from_method(label, seed):
    """
    Returns dict with path and row for selected synthetic method.
    First tries saved .npy paths. If missing, reruns one rollout.
    """
    world_static = tits_load_synthetic_world()

    saved = tits_load_saved_path("paper_table_synthetic_static.csv", label, seed)
    if saved is not None:
        row = saved["row"]
        path = saved["path"]

        short_label = {
            "M7_geometry_only_RL": "M7 geometry-only RL",
            "M8_reward_only_RL": "M8 reward-only RL",
            "M10_fullmap_receding_Astar": "M10 receding A*",
        }.get(label, label)

        return {
            "label": short_label,
            "path": path,
            "row": row,
            "source": "saved_path",
        }

    # Saved paths are absent for some old M8 runs, so rerun a single deterministic rollout.
    from stable_baselines3 import PPO

    if label == "M7_geometry_only_RL":
        if "rollout_geo_baseline_eval_episode" not in globals():
            raise RuntimeError(
                "rollout_geo_baseline_eval_episode is missing. "
                "Run M7 definition cells first."
            )

        model_path = tits_get_model_path("paper_table_synthetic_static.csv", label)
        model = PPO.load(str(model_path), device="cpu")

        row, path = rollout_geo_baseline_eval_episode(
            model=model,
            world=world_static,
            seed=int(seed),
            mode="strict_eval",
            n_obstacles=0,
            deterministic=True,
            max_steps=1000,
        )

        return {
            "label": "M7 geometry-only RL",
            "path": path,
            "row": row,
            "source": "rerun_rollout",
        }

    if label == "M8_reward_only_RL":
        if "rollout_m8_ablation_eval_episode" not in globals():
            raise RuntimeError(
                "rollout_m8_ablation_eval_episode is missing. "
                "Run M8 definition cells first."
            )

        model_path = tits_get_model_path("paper_table_synthetic_static.csv", label)
        model = PPO.load(str(model_path), device="cpu")

        row, path = rollout_m8_ablation_eval_episode(
            model=model,
            world=world_static,
            variant_name="reward_only",
            seed=int(seed),
            stage="C",
            mode="strict_eval",
            n_obstacles=0,
            deterministic=True,
            max_steps=1000,
        )

        return {
            "label": "M8 reward-only RL",
            "path": path,
            "row": row,
            "source": "rerun_rollout",
        }

    if label == "M10_fullmap_receding_Astar":
        if "rollout_m10_receding_planner_episode" not in globals():
            raise RuntimeError(
                "rollout_m10_receding_planner_episode is missing. "
                "Run M10 definition cells first."
            )

        if "m10_cfg_perfect" in globals():
            cfg = m10_cfg_perfect
        else:
            cfg = M10RecedingHorizonPlannerConfig(
                plan_res_m=5.0,
                planner_agl_m=40.0,
                risk_lambda=0.35,
                use_astar=True,
                sensing_radius_m=75.0,
                replan_every_steps=5,
                prior_risk_mode="perfect",
                hard_blockers_known=True,
                waypoint_radius_m=8.0,
                target_speed_mps=12.0,
                max_steps=1000,
            )

        planner = M10RecedingHorizonRiskPlanner(
            world=world_static,
            config=cfg,
            risk_profile="balanced",
        )

        row, path = rollout_m10_receding_planner_episode(
            world=world_static,
            planner=planner,
            seed=int(seed),
            cfg=cfg,
            stage="C",
            mode="strict_eval",
            n_obstacles=0,
            max_steps=1000,
        )

        return {
            "label": "M10 receding A*",
            "path": path,
            "row": row,
            "source": "rerun_rollout",
        }

    raise ValueError(f"Unsupported synthetic method label: {label}")

def tits_pick_vienna_contrast_seed():
    paired = tits_read_table("paired_vienna_extended_vs_planner.csv")

    cond = (
        (paired["a_is_success"].map(tits_as_bool) == True)
        & (paired["a_mission_acceptable"].map(tits_as_bool) == False)
        & (paired["b_is_success"].map(tits_as_bool) == True)
        & (paired["b_mission_acceptable"].map(tits_as_bool) == True)
    )

    candidates = paired[cond].copy()

    if len(candidates) > 0:
        candidates["contrast_score"] = candidates["a_urban_cumulative_risk"].astype(float)
        best = candidates.sort_values("contrast_score", ascending=False).iloc[0]
        return int(best["a_seed"]), int(best["a_episode"])

    # Fallback: both successful, planner acceptable.
    cond = (
        (paired["a_is_success"].map(tits_as_bool) == True)
        & (paired["b_is_success"].map(tits_as_bool) == True)
        & (paired["b_mission_acceptable"].map(tits_as_bool) == True)
    )

    candidates = paired[cond].copy()

    if len(candidates) > 0:
        best = candidates.iloc[0]
        return int(best["a_seed"]), int(best["a_episode"])

    best = paired.iloc[0]
    return int(best["a_seed"]), int(best["a_episode"])

def tits_vienna_trajectory_from_method(label, seed):
    """
    Load saved Vienna .npy path if available; otherwise rerun one rollout.
    Supported:
      - Vienna_M8_reward_extended
      - Vienna_M8_reward_transfer
      - Vienna_footway_only_M10_planner_200
    """
    vienna_world = tits_load_vienna_world()

    saved = tits_load_saved_path("paper_table_real_gis_vienna.csv", label, seed)
    if saved is not None:
        row = saved["row"]
        path = saved["path"]

        short_label = {
            "Vienna_M8_reward_extended": "Extended FT RL",
            "Vienna_M8_reward_transfer": "Zero-shot RL",
            "Vienna_footway_only_M10_planner_200": "Receding A*",
        }.get(label, label)

        return {
            "label": short_label,
            "path": path,
            "row": row,
            "source": "saved_path",
        }

    # Rerun fallback.
    from stable_baselines3 import PPO

    if label in {"Vienna_M8_reward_extended", "Vienna_M8_reward_transfer"}:
        if "rollout_m12c_real_policy_episode" not in globals():
            raise RuntimeError(
                "rollout_m12c_real_policy_episode is missing. "
                "Run M12C/Vienna definition cells first."
            )

        model_path = tits_get_model_path("paper_table_real_gis_vienna.csv", label)
        model = PPO.load(str(model_path), device="cpu")

        row, path = rollout_m12c_real_policy_episode(
            model=model,
            world=vienna_world,
            policy_kind="reward_only",
            seed=int(seed),
            stage="C",
            mode="strict_eval",
            n_obstacles=0,
            deterministic=True,
            max_steps=1000,
        )

        short_label = {
            "Vienna_M8_reward_extended": "Extended FT RL",
            "Vienna_M8_reward_transfer": "Zero-shot RL",
        }[label]

        return {
            "label": short_label,
            "path": path,
            "row": row,
            "source": "rerun_rollout",
        }

    if label == "Vienna_footway_only_M10_planner_200":
        if "rollout_vienna_receding_planner_with_telemetry" in globals():
            rollout = rollout_vienna_receding_planner_with_telemetry(
                world=vienna_world,
                seed=int(seed),
                stage="C",
                mode="strict_eval",
                max_steps=1000,
            )

            final_row = {}
            if len(rollout.get("df", [])) > 0:
                final_row = rollout["df"].iloc[-1].to_dict()

            return {
                "label": "Receding A*",
                "path": rollout["path"],
                "row": final_row,
                "source": "rerun_rollout_with_telemetry",
                "rollout": rollout,
            }

        if "rollout_m10_receding_planner_episode" not in globals():
            raise RuntimeError(
                "rollout_m10_receding_planner_episode is missing. "
                "Run M10/Vienna definition cells first."
            )

        cfg = M10RecedingHorizonPlannerConfig(
            plan_res_m=5.0,
            planner_agl_m=30.0,
            risk_lambda=0.35,
            use_astar=True,
            sensing_radius_m=75.0,
            replan_every_steps=5,
            prior_risk_mode="perfect",
            hard_blockers_known=True,
            waypoint_radius_m=8.0,
            target_speed_mps=12.0,
            max_steps=1000,
        )

        planner = M10RecedingHorizonRiskPlanner(
            world=vienna_world,
            config=cfg,
            risk_profile="balanced",
        )

        row, path = rollout_m10_receding_planner_episode(
            world=vienna_world,
            planner=planner,
            seed=int(seed),
            cfg=cfg,
            stage="C",
            mode="strict_eval",
            n_obstacles=0,
            max_steps=1000,
        )

        return {
            "label": "Receding A*",
            "path": path,
            "row": row,
            "source": "rerun_rollout",
        }

    raise ValueError(f"Unsupported Vienna method label: {label}")

import os

import json

import math

import warnings

from pathlib import Path

from datetime import datetime

import numpy as np

import pandas as pd

def ma_atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    def conv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, dict):
            return {str(k): conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x

    with open(tmp, "w") as f:
        json.dump(conv(obj), f, indent=2)
    os.replace(tmp, path)

def ma_bool_series(s):
    if str(s.dtype) == "bool":
        return s.astype(bool)

    def parse_bool(x):
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if pd.isna(x):
            return False
        return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}

    return s.map(parse_bool).astype(bool)

def ma_tcrit95(df):
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, int(df)))
    except Exception:
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}
        return float(table.get(int(df), 1.96))

def ma_training_seed_ci(values):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    n = int(len(vals))
    if n == 0:
        return dict(n=0, mean=np.nan, std=np.nan, ci95_low=np.nan, ci95_high=np.nan)

    mean = float(np.mean(vals))

    if n == 1:
        return dict(n=1, mean=mean, std=np.nan, ci95_low=np.nan, ci95_high=np.nan)

    std = float(np.std(vals, ddof=1))
    sem = std / math.sqrt(n)
    tcrit = ma_tcrit95(n - 1)

    return dict(
        n=n,
        mean=mean,
        std=std,
        ci95_low=float(mean - tcrit * sem),
        ci95_high=float(mean + tcrit * sem),
    )

def ma_format_rate(x):
    if pd.isna(x):
        return ""
    return f"{100.0 * float(x):.1f}%"

def ma_read_csv_if_exists(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None

def ma_find_existing_path(x):
    if x is None or pd.isna(x):
        return None
    p = Path(str(x))
    if p.exists():
        return p
    return None

def ma_read_json_if_exists(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)

DEFAULT_BUDGETS = {
    "road": 20.0,
    "sidewalk": 5.0,
    "building": 15.0,
    "sensitive": 0.0,
    "cargo_min": 0.80,
    "dynamic": 10.0,
}

def ma_col(df, name, default=0.0):
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default)
    return pd.Series([default] * len(df), index=df.index)

def ma_recompute_acceptability(
    df,
    road_budget_s=20.0,
    sidewalk_budget_s=5.0,
    building_budget_s=15.0,
    sensitive_budget_s=0.0,
    cargo_min=0.80,
    include_dynamic=False,
    dynamic_budget=10.0,
):
    """
    Recompute mission acceptability from logged per-episode CSV.
    Works for M7/M8/M9/M10/M11/M12-like eval CSVs.
    """
    if len(df) == 0:
        return pd.Series([], dtype=bool)

    success = ma_bool_series(df["is_success"]) if "is_success" in df.columns else pd.Series(False, index=df.index)

    cargo = ma_col(df, "cargo_health", default=1.0) >= float(cargo_min)

    road_ok = ma_col(df, "road_time") <= float(road_budget_s)
    sidewalk_ok = ma_col(df, "sidewalk_time") <= float(sidewalk_budget_s)
    building_ok = ma_col(df, "building_time") <= float(building_budget_s)
    sensitive_ok = ma_col(df, "sensitive_time") <= float(sensitive_budget_s)

    hard_col = "urban_hard_violation_steps" if "urban_hard_violation_steps" in df.columns else "hard_violation_steps"
    hard_ok = ma_col(df, hard_col) <= 0

    acc = success & cargo & road_ok & sidewalk_ok & building_ok & sensitive_ok & hard_ok

    if include_dynamic:
        dyn_ok = ma_col(df, "dynamic_cumulative_risk") <= float(dynamic_budget)
        acc = acc & dyn_ok

    return acc.astype(bool)

def ma_summarize_with_acceptability(df, acc, label, budget_name, extra=None):
    success = ma_bool_series(df["is_success"]) if "is_success" in df.columns else pd.Series(False, index=df.index)
    n = int(len(df))

    if n == 0:
        sr = mar = ags = np.nan
    else:
        sr = float(success.mean())
        mar = float(acc.mean())
        ags = float(acc[success].mean()) if success.sum() > 0 else 0.0

    row = {
        "label": label,
        "budget_name": budget_name,
        "n_episodes": n,
        "success_rate": sr,
        "mission_acceptability_rate": mar,
        "acceptable_given_success_rate": ags,

        "mean_road_time": float(ma_col(df, "road_time").mean()) if n else np.nan,
        "mean_sidewalk_time": float(ma_col(df, "sidewalk_time").mean()) if n else np.nan,
        "mean_building_time": float(ma_col(df, "building_time").mean()) if n else np.nan,
        "mean_sensitive_time": float(ma_col(df, "sensitive_time").mean()) if n else np.nan,
        "mean_green_time": float(ma_col(df, "green_time").mean()) if n else np.nan,
        "mean_hard_violation_steps": float(ma_col(df, "urban_hard_violation_steps").mean()) if n else np.nan,
        "mean_cumulative_risk": float(ma_col(df, "urban_cumulative_risk").mean()) if "urban_cumulative_risk" in df.columns else np.nan,
    }

    if extra:
        row.update(extra)

    return row

def ma_collect_csvs_from_paper_tables(table_names):
    """
    Reads paper_ready_v1 tables and collects rows that contain an existing csv_path.
    """
    registry = {}

    for table_name in table_names:
        p = TABLE_DIR / table_name
        if not p.exists():
            continue

        tab = pd.read_csv(p)

        for _, row in tab.iterrows():
            label = None
            for c in ["label", "method", "Method", "name"]:
                if c in row and not pd.isna(row[c]):
                    label = str(row[c])
                    break

            if label is None:
                continue

            csv_path = None
            for c in ["csv_path", "episodes_csv", "eval_csv"]:
                if c in row:
                    csv_path = ma_find_existing_path(row[c])
                    if csv_path is not None:
                        break

            if csv_path is not None:
                try:
                    registry[label] = pd.read_csv(csv_path)
                    registry[label].attrs["csv_path"] = str(csv_path)
                except Exception as e:
                    print(f"[WARN] Could not read {csv_path}: {repr(e)}")

    return registry

def ma_collect_from_globals(name_map):
    """
    Add existing global dataframes if present.
    """
    out = {}
    for label, names in name_map.items():
        for n in names:
            if n in globals() and isinstance(globals()[n], pd.DataFrame):
                out[label] = globals()[n].copy()
                out[label].attrs["source_global"] = n
                break
    return out

STATIC_GLOBAL_DF_MAP = {
    "M7_geometry_only_RL": ["df_geo_eval", "df_m7_geo", "df_geo_eval_200", "df_m7_geometry_only"],
    "M8_obs_only_RL": ["df_m8_obs", "df_m8_obs_only", "df_m8_obs_200"],
    "M8_reward_only_RL": ["df_m8_reward_200", "df_m8_reward", "df_m8_reward_only"],
    "M6_full_semantic_RL": ["df_eval", "df_m6_eval", "df_m6_full_semantic"],
    "M9_static_risk_Astar": ["df_m9_b0", "df_m9_static", "df_m9"],
    "M10_partial_risk_receding_Astar": ["df_m10_b1", "df_m10_partial", "m10_sweep_df"],
    "M10_fullmap_receding_Astar": ["df_m10_b0r", "df_m10_fullmap"],
}

BUDGET_JOINT_SCALES = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]

BUDGET_OAT_SCALES = [0.50, 0.75, 1.00, 1.25, 1.50, 2.00, 3.00]

def make_budget_profiles():
    profiles = []

    for s in BUDGET_JOINT_SCALES:
        profiles.append({
            "budget_name": f"joint_x{s:g}",
            "mode": "joint",
            "scale_road": s,
            "scale_sidewalk": s,
            "scale_building": s,
            "road_budget_s": DEFAULT_BUDGETS["road"] * s,
            "sidewalk_budget_s": DEFAULT_BUDGETS["sidewalk"] * s,
            "building_budget_s": DEFAULT_BUDGETS["building"] * s,
            "sensitive_budget_s": DEFAULT_BUDGETS["sensitive"],
            "cargo_min": DEFAULT_BUDGETS["cargo_min"],
        })

    for var in ["road", "sidewalk", "building"]:
        for s in BUDGET_OAT_SCALES:
            b = {
                "budget_name": f"{var}_x{s:g}",
                "mode": "one_at_a_time",
                "scale_road": 1.0,
                "scale_sidewalk": 1.0,
                "scale_building": 1.0,
                "road_budget_s": DEFAULT_BUDGETS["road"],
                "sidewalk_budget_s": DEFAULT_BUDGETS["sidewalk"],
                "building_budget_s": DEFAULT_BUDGETS["building"],
                "sensitive_budget_s": DEFAULT_BUDGETS["sensitive"],
                "cargo_min": DEFAULT_BUDGETS["cargo_min"],
            }
            b[f"scale_{var}"] = s
            b[f"{var}_budget_s"] = DEFAULT_BUDGETS[var] * s
            profiles.append(b)

    return profiles

def ma_profile_to_dict(profile_name):
    """
    Extract weights and eta scales from UrbanRiskProfile if available.
    """
    if "RUSCUAV_RISK_PROFILES_M2" in globals():
        prof = RUSCUAV_RISK_PROFILES_M2[str(profile_name)]
        return {
            "name": str(profile_name),
            "road_weight": float(prof.road_weight),
            "sidewalk_weight": float(prof.sidewalk_weight),
            "building_weight": float(prof.building_weight),
            "canopy_weight": float(prof.canopy_weight),
            "green_weight": float(prof.green_weight),
            "sensitive_weight": float(prof.sensitive_weight),

            "road_h_scale_m": float(prof.road_h_scale_m),
            "sidewalk_h_scale_m": float(prof.sidewalk_h_scale_m),
            "sensitive_h_scale_m": float(prof.sensitive_h_scale_m),
            "building_clearance_h_scale_m": float(prof.building_clearance_h_scale_m),
            "canopy_clearance_h_scale_m": float(prof.canopy_clearance_h_scale_m),
            "green_h_scale_m": float(prof.green_h_scale_m),
        }

    # Fallback if M2 profiles not in runtime.
    fallback = {
        "efficiency_first": dict(
            road_weight=3.0, sidewalk_weight=6.0, building_weight=1.5,
            canopy_weight=0.10, green_weight=0.05, sensitive_weight=30.0,
            road_h_scale_m=45.0, sidewalk_h_scale_m=55.0, sensitive_h_scale_m=80.0,
            building_clearance_h_scale_m=35.0, canopy_clearance_h_scale_m=20.0,
            green_h_scale_m=35.0,
        ),
        "balanced": dict(
            road_weight=6.0, sidewalk_weight=12.0, building_weight=3.0,
            canopy_weight=0.20, green_weight=0.10, sensitive_weight=50.0,
            road_h_scale_m=45.0, sidewalk_h_scale_m=55.0, sensitive_h_scale_m=80.0,
            building_clearance_h_scale_m=35.0, canopy_clearance_h_scale_m=20.0,
            green_h_scale_m=35.0,
        ),
        "safety_first": dict(
            road_weight=10.0, sidewalk_weight=20.0, building_weight=5.0,
            canopy_weight=0.10, green_weight=0.05, sensitive_weight=100.0,
            road_h_scale_m=60.0, sidewalk_h_scale_m=75.0, sensitive_h_scale_m=100.0,
            building_clearance_h_scale_m=35.0, canopy_clearance_h_scale_m=20.0,
            green_h_scale_m=35.0,
        ),
    }
    d = fallback[str(profile_name)].copy()
    d["name"] = str(profile_name)
    return d

def ma_decay(agl, eta):
    return math.exp(-max(float(agl), 0.0) / max(float(eta), 1e-6))

def ma_weighted_exposure_proxy(df, profile_name, agl_m=40.0, eta_multiplier=1.0):
    """
    Approximate post-hoc risk from logged exposure times.
    This is not a replacement for rerunning planning/training.
    """
    p = ma_profile_to_dict(profile_name)

    eta_mult = float(eta_multiplier)

    road_coeff = p["road_weight"] * ma_decay(agl_m, p["road_h_scale_m"] * eta_mult)
    sidewalk_coeff = p["sidewalk_weight"] * ma_decay(agl_m, p["sidewalk_h_scale_m"] * eta_mult)
    sensitive_coeff = p["sensitive_weight"] * ma_decay(agl_m, p["sensitive_h_scale_m"] * eta_mult)

    # Building/canopy/green are approximated from exposure time because detailed clearance
    # is not recoverable unless per-step trajectory paths are reprocessed.
    building_coeff = p["building_weight"]
    canopy_coeff = p["canopy_weight"]
    green_coeff = p["green_weight"] * ma_decay(agl_m, p["green_h_scale_m"] * eta_mult)

    score = (
        road_coeff * ma_col(df, "road_time")
        + sidewalk_coeff * ma_col(df, "sidewalk_time")
        + building_coeff * ma_col(df, "building_time")
        + canopy_coeff * ma_col(df, "canopy_time")
        + sensitive_coeff * ma_col(df, "sensitive_time")
        + green_coeff * ma_col(df, "green_time")
    )

    return score.astype(float)

PROFILE_NAMES = ["efficiency_first", "balanced", "safety_first"]

ETA_MULTIPLIERS = [0.75, 1.0, 1.25]

RUN_PLANNER_PROFILE_SENSITIVITY = True

N_PROFILE_EPISODES = 100

PROFILE_SEED0 = 5000

RUN_EQUAL_INFORMATION_PLANNERS = True

EQUAL_INFO_N_EPISODES = 200

EQUAL_INFO_SEED0 = 5000

def ma_coverage_row(world_obj, label):
    return {
        "aoi": label,
        "building_coverage": float(np.asarray(world_obj.building_mask).mean()),
        "road_coverage": float(np.asarray(world_obj.road_mask).mean()),
        "pedestrian_coverage": float(np.asarray(world_obj.sidewalk_mask).mean()),
        "green_coverage": float(np.asarray(world_obj.green_mask).mean()),
        "canopy_coverage": float(np.asarray(world_obj.canopy_mask).mean()),
        "sensitive_coverage": float(np.asarray(world_obj.sensitive_mask).mean()),
    }

def ma_summary_to_row(summary, label):
    if summary is None:
        return None
    return {
        "label": label,
        "success_rate": float(summary.get("success_rate", np.nan)),
        "mission_acceptability_rate": float(summary.get("mission_acceptability_rate", np.nan)),
        "acceptable_given_success_rate": float(summary.get("acceptable_given_success_rate", np.nan)),
        "mean_cumulative_risk": float(summary.get("mean_cumulative_risk", summary.get("mean_urban_cumulative_risk", np.nan))),
        "mean_road_time": float(summary.get("mean_road_time", np.nan)),
        "mean_sidewalk_time": float(summary.get("mean_sidewalk_time", np.nan)),
        "mean_building_time": float(summary.get("mean_building_time", np.nan)),
        "mean_green_time": float(summary.get("mean_green_time", np.nan)),
        "mean_canopy_time": float(summary.get("mean_canopy_time", np.nan)),
        "mean_sensitive_time": float(summary.get("mean_sensitive_time", np.nan)),
        "mean_hard_violation_steps": float(summary.get("mean_hard_violation_steps", np.nan)),
        "n_episodes": int(summary.get("n_episodes", summary.get("n_episodes_requested", -1))),
        "csv_path": str(summary.get("csv_path", "")),
        "summary_path": str(summary.get("summary_path", "")),
    }

def ma_first_global(*names):
    for n in names:
        if n in globals():
            return globals()[n]
    return None

def make_ruscuav_m8_ablation_env_m4_mode(
    world,
    variant_name,
    stage="A",
    mode="fast_train",
    risk_profile="balanced",
    n_obstacles=0,
    seed=None,
):
    if variant_name not in M8_VARIANTS:
        raise ValueError(f"Unknown variant_name={variant_name}. Available: {list(M8_VARIANTS.keys())}")

    variant = M8_VARIANTS[variant_name]
    stage = str(stage).upper()
    mode = str(mode).lower()

    if mode == "fast_train":
        urban_lidar_step_m = 5.0
        urban_segment_check_step_m = 2.0
    elif mode == "strict_eval":
        urban_lidar_step_m = 3.0
        urban_segment_check_step_m = 1.0
    else:
        raise ValueError("mode must be 'fast_train' or 'strict_eval'.")

    if stage == "A":
        urban_risk_lambda_base = 0.05
        terminate_on_urban_constraint = False
        hard_sensitive_zones = False
    elif stage == "B":
        urban_risk_lambda_base = 0.10
        terminate_on_urban_constraint = False
        hard_sensitive_zones = True
    elif stage == "C":
        urban_risk_lambda_base = 0.10
        terminate_on_urban_constraint = True
        hard_sensitive_zones = True
    else:
        raise ValueError("stage must be A, B, or C.")

    urban_risk_lambda = urban_risk_lambda_base * float(variant.get("urban_risk_lambda_mult", 1.0))

    cfg = RealisticUrbanEnvConfig(
        risk_profile=risk_profile,

        include_urban_vector_obs=bool(variant["include_urban_vector_obs"]),
        use_urban_risk_costs=bool(variant["use_urban_risk_costs"]),
        urban_risk_lambda=float(urban_risk_lambda),
        green_bonus_lambda=float(variant["green_bonus_lambda"]),

        use_urban_altitude_reference=True,

        terminate_on_urban_constraint=bool(terminate_on_urban_constraint),
        hard_sensitive_zones=bool(hard_sensitive_zones),

        default_n_obstacles=int(n_obstacles),

        mission_accept_cargo_min=0.80,
        mission_accept_road_budget_s=20.0,
        mission_accept_sidewalk_budget_s=5.0,
        mission_accept_building_budget_s=15.0,
        mission_accept_sensitive_budget_s=0.0,

        terminal_road_budget_s=30.0,
        terminal_sidewalk_budget_s=10.0,
        terminal_building_budget_s=20.0,
        terminal_sensitive_budget_s=0.0,
    )

    env = RealisticUrbanCargoUAVEnvM4(
        world=world,
        config=cfg,
        include_shield_features=False,

        urban_lidar_enabled=bool(variant["urban_lidar_enabled"]),
        urban_action_shield_enabled=bool(variant["urban_action_shield_enabled"]),
        urban_geometric_shield_enabled=bool(variant["urban_geometric_shield_enabled"]),

        urban_lidar_step_m=urban_lidar_step_m,
        urban_segment_check_step_m=urban_segment_check_step_m,
        urban_shield_backoff_m=0.50,
        urban_horizontal_safety_margin_m=1.0,
    )

    if seed is not None:
        obs, info = env.reset(seed=int(seed))
        print(
            f"[M8 patched env] variant={variant_name}, stage={stage}, mode={mode}, "
            f"risk_lambda={urban_risk_lambda}, green_bonus={cfg.green_bonus_lambda}, obs.shape={obs.shape}"
        )

    return env

RUN_REWARD_FACTOR_ABLATION = True

REWARD_FACTOR_VARIANTS = ["risk_only", "green_only"]

REWARD_FACTOR_TRAIN_SEEDS = [1001, 2001, 3001, 4001, 5001]

REWARD_FACTOR_STEPS_A = 50_000

REWARD_FACTOR_STEPS_B = 75_000

REWARD_FACTOR_STEPS_C = 75_000

REWARD_FACTOR_N_EVAL = 200

REWARD_FACTOR_EVAL_SEED0 = 5000

def ma_pick_reward_model_path():
    # Prefer multi-seed representative model if available.
    if len(multiseed_seed_level) > 0:
        sub = multiseed_seed_level[
            (multiseed_seed_level["method"].astype(str) == "M8_reward_only")
            & (multiseed_seed_level.get("run_status", "complete").astype(str) == "complete")
        ].copy()
        if len(sub) > 0:
            # Pick seed 1001 if present, otherwise first available.
            sub["seed_pref"] = (sub["train_seed"].astype(int) != 1001).astype(int)
            sub = sub.sort_values(["seed_pref", "train_seed"])
            p = ma_find_existing_path(sub.iloc[0].get("final_model_path"))
            if p is not None:
                return p

    # Fallbacks from notebook globals.
    if "m8_reward_only_paths" in globals():
        p = ma_find_existing_path(m8_reward_only_paths.get("final_C"))
        if p is not None:
            return p

    if "m8_reward_only_model_path" in globals():
        p = ma_find_existing_path(m8_reward_only_model_path)
        if p is not None:
            return p

    raise RuntimeError("Could not find an M8 reward-only model path for shield diagnostics.")

RUN_SHIELD_DIAGNOSTICS = True

SHIELD_DIAG_N_EPISODES = 200

SHIELD_DIAG_SEED0 = 5000

RUN_GLOBAL_GUIDANCE = False

STATIC_ONLY_LABELS = [
    "M7_geometry_only_RL",
    "M8_obs_only_RL",
    "M6_full_semantic_RL",
    "M8_reward_only_RL",
    "M9_static_risk_Astar",
    "M10_partial_risk_receding_Astar",
    "M10_fullmap_receding_Astar",
]

from pathlib import Path

import os

import glob

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

def save_pub_fig_local(fig, stem):
    """
    Save as PDF/SVG/PNG. Uses previous save_pub_fig if available.
    """
    if "save_pub_fig" in globals():
        return save_pub_fig(fig, stem)

    paths = []
    for ext in ["pdf", "svg", "png"]:
        p = PUB_FIG_DIR / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=PNG_DPI if ext == "png" else None)
        paths.append(str(p))
        print("Saved:", p)
    return paths

SEMANTIC_STYLE_W = {
    "green":     ("#2ca25f", 0.30, "green"),
    "canopy":    ("#006d2c", 0.32, "canopy"),
    "road":      ("#f16913", 0.42, "road"),
    "sidewalk":  ("#fdae6b", 0.46, "pedestrian / footway"),
    "building":  ("#737373", 0.58, "building"),
    "sensitive": ("#cb181d", 0.42, "sensitive"),
}

def load_world_by_patterns(global_names, patterns, label):
    """
    Load world from existing global object or matching .npz path.
    """
    for name in global_names:
        if name in globals():
            print(f"Using global {name} for {label}")
            return globals()[name]

    candidate_paths = []
    for pat in patterns:
        candidate_paths.extend(glob.glob(str(pat), recursive=True))

    candidate_paths = sorted(set(candidate_paths))

    if len(candidate_paths) == 0:
        raise FileNotFoundError(
            f"Could not find {label} world. Tried globals={global_names} and patterns={patterns}"
        )

    p = candidate_paths[0]
    print(f"Loading {label} world from:", p)
    return UrbanWorldLayers.load_npz(p)

def load_vienna_footway_world_for_fig():
    patterns = [
        DRIVE_ROOT / "worlds" / "*vienna*footway*urban_world_layers*.npz",
        DRIVE_ROOT / "worlds" / "*vienna*urban_world_layers*.npz",
        DRIVE_ROOT / "experiments" / "**" / "*vienna*footway*urban_world_layers*.npz",
        PAPER_READY_ROOT / "worlds" / "*vienna*footway*urban_world_layers*.npz",
    ]

    return load_world_by_patterns(
        global_names=["vienna_world_footway_only", "vienna_world"],
        patterns=patterns,
        label="Vienna",
    )

def load_warsaw_footway_world_for_fig():
    patterns = [
        DRIVE_ROOT / "worlds" / "*warsaw*footway*urban_world_layers*.npz",
        DRIVE_ROOT / "worlds" / "*warsaw*urban_world_layers*.npz",
        DRIVE_ROOT / "experiments" / "**" / "*warsaw*footway*urban_world_layers*.npz",
        DRIVE_ROOT / "experiments" / "**" / "*warsaw*urban_world_layers*.npz",
        PAPER_READY_ROOT / "worlds" / "*warsaw*footway*urban_world_layers*.npz",
    ]

    return load_world_by_patterns(
        global_names=["real_world_footway_only", "m13_world_base", "real_world"],
        patterns=patterns,
        label="Warsaw",
    )

def world_extent(world):
    h, w = np.asarray(world.terrain_z).shape
    return [0, w, h, 0]

def overlay_mask(ax, world, mask_name, color, alpha):
    mask = np.asarray(getattr(world, mask_name), dtype=np.float32) > 0.5
    if not np.any(mask):
        return

    rgba = np.zeros(mask.shape + (4,), dtype=np.float32)
    rgba[mask] = (*plt.matplotlib.colors.to_rgb(color), alpha)

    ax.imshow(
        rgba,
        origin="upper",
        extent=world_extent(world),
        interpolation="nearest",
    )

def draw_semantic_world(ax, world, title):
    terrain = np.asarray(world.terrain_z, dtype=np.float32)

    ax.imshow(
        terrain,
        origin="upper",
        extent=world_extent(world),
        cmap="Greys",
        alpha=0.22,
        interpolation="nearest",
    )

    for mask_name, key in [
        ("green_mask", "green"),
        ("canopy_mask", "canopy"),
        ("road_mask", "road"),
        ("sidewalk_mask", "sidewalk"),
        ("building_mask", "building"),
        ("sensitive_mask", "sensitive"),
    ]:
        color, alpha, _ = SEMANTIC_STYLE_W[key]
        overlay_mask(ax, world, mask_name, color, alpha)

    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, world.width)
    ax.set_ylim(world.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

def risk_map_for_world(world, agl_m=30.0):
    try:
        rm = UrbanRiskModel(
            world=world,
            profile="balanced",
            hard_sensitive_zones=True,
            include_global_altitude_band_cost=True,
        )
        return rm.risk_map_at_agl(float(agl_m))
    except Exception:
        rm = UrbanRiskModel(
            world=world,
            profile="balanced",
            hard_sensitive_zones=True,
        )
        return rm.risk_map_at_agl(float(agl_m))

def draw_risk_world(ax, world, title, agl_m=30.0, add_colorbar=True):
    risk = risk_map_for_world(world, agl_m=agl_m)

    im = ax.imshow(
        risk,
        origin="upper",
        extent=world_extent(world),
        cmap="inferno",
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, world.width)
    ax.set_ylim(world.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if add_colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("risk", fontsize=6)

    return im

def draw_height_world(ax, world, title, add_colorbar=True):
    obstacle_h = np.maximum(
        np.asarray(world.building_height, dtype=np.float32),
        np.asarray(world.canopy_height, dtype=np.float32),
    )

    im = ax.imshow(
        obstacle_h,
        origin="upper",
        extent=world_extent(world),
        cmap="magma",
        interpolation="nearest",
    )

    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, world.width)
    ax.set_ylim(world.height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    if add_colorbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=6)
        cb.set_label("height [m]", fontsize=6)

    return im

def semantic_legend_handles():
    return [
        Patch(facecolor=color, edgecolor="black", alpha=alpha, label=label)
        for color, alpha, label in SEMANTIC_STYLE_W.values()
    ]

def find_warsaw_eval_df(prefer="footway_B"):
    """
    Prefer Stage B no-terminal footway-only because it illustrates:
    high success but low MAR.
    """
    candidates = []

    if prefer == "footway_B":
        candidates = [
            "df_m12_stepA_foot_B",
            "df_m12_m10_footway_only_no_terminal",
            "df_m12_m10_no_terminal",
            "df_m12_stepA_foot_C",
            "df_m12_m10_footway_only",
        ]
    else:
        candidates = [
            "df_m12_stepA_foot_C",
            "df_m12_m10_footway_only",
            "df_m12_stepA_foot_B",
            "df_m12_m10_no_terminal",
        ]

    for name in candidates:
        if name in globals() and isinstance(globals()[name], pd.DataFrame):
            print("Using Warsaw dataframe:", name)
            return globals()[name].copy(), name

    raise RuntimeError(
        "No Warsaw evaluation dataframe found. Expected one of: "
        "df_m12_stepA_foot_B, df_m12_stepA_foot_C, df_m12_m10_footway_only, etc."
    )

def bool_col(s):
    if str(s.dtype) == "bool":
        return s.astype(bool)

    return s.map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y", "t"}).astype(bool)

def pick_warsaw_seed(df, success=True, acceptable=None, prefer_high_exposure=True):
    """
    Pick a seed for accepted or successful-but-unacceptable mission.
    """
    d = df.copy()

    if "is_success" in d.columns:
        d = d[bool_col(d["is_success"]) == bool(success)]

    if acceptable is not None and "mission_acceptable" in d.columns:
        d = d[bool_col(d["mission_acceptable"]) == bool(acceptable)]

    if len(d) == 0:
        raise RuntimeError(
            f"No Warsaw episode matching success={success}, acceptable={acceptable}."
        )

    sort_candidates = [
        "sidewalk_time",
        "road_time",
        "urban_cumulative_risk",
        "dist_xy",
    ]

    if prefer_high_exposure:
        for c in sort_candidates:
            if c in d.columns:
                d = d.sort_values(c, ascending=False)
                break
    else:
        if "dist_xy" in d.columns:
            d = d.sort_values("dist_xy", ascending=True)

    row = d.iloc[0]
    return int(row["seed"]), int(row.get("episode", -1)), row.to_dict()

def make_warsaw_planner_cfg():
    if "m12_viz_cfg_agl30" in globals():
        return m12_viz_cfg_agl30

    return M10RecedingHorizonPlannerConfig(
        plan_res_m=5.0,
        planner_agl_m=30.0,
        risk_lambda=0.35,
        use_astar=True,
        sensing_radius_m=75.0,
        replan_every_steps=5,
        prior_risk_mode="perfect",
        hard_blockers_known=True,
        waypoint_radius_m=8.0,
        target_speed_mps=12.0,
        max_steps=1000,
    )

def rollout_warsaw_planner_for_fig(world, seed, stage="B"):
    """
    Replays a Warsaw planner rollout with telemetry.
    """
    if "rollout_m12_receding_planner_with_telemetry" not in globals():
        raise RuntimeError(
            "rollout_m12_receding_planner_with_telemetry is not defined. "
            "Run the M12-VIZ cells first."
        )

    cfg = make_warsaw_planner_cfg()

    return rollout_m12_receding_planner_with_telemetry(
        world=world,
        seed=int(seed),
        cfg=cfg,
        stage=str(stage),
        mode="strict_eval",
        n_obstacles=0,
    )

def plot_path_on_ax(ax, rollout, label, linestyle="-", linewidth=1.8):
    path = np.asarray(rollout["path"], dtype=np.float32)

    ax.plot(
        path[:, 0],
        path[:, 1],
        linestyle=linestyle,
        linewidth=linewidth,
        label=label,
        zorder=5,
    )

    ax.scatter(
        path[0, 0],
        path[0, 1],
        marker="o",
        s=28,
        facecolor="white",
        edgecolor="black",
        linewidth=0.8,
        zorder=6,
    )

    ax.scatter(
        path[-1, 0],
        path[-1, 1],
        marker="x",
        s=32,
        linewidth=1.2,
        zorder=6,
    )

def final_rollout_text(label, rollout):
    df = rollout.get("df", None)

    if isinstance(df, pd.DataFrame) and len(df) > 0:
        r = df.iloc[-1]
        return (
            f"{label}: termination={rollout.get('termination', 'NA')}, "
            f"steps={len(df)}, "
            f"road={float(r.get('road_time', r.get('urban_road_time', np.nan))):.1f}s, "
            f"ped={float(r.get('sidewalk_time', r.get('urban_sidewalk_time', np.nan))):.1f}s"
        )

    return f"{label}: termination={rollout.get('termination', 'NA')}"
