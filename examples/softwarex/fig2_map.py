#!/usr/bin/env python3
"""Fig. 2 of the SoftwareX paper: one mission in each world, the paths of FULL_RH_Astar and of the external
planner over the semantic layers that decide acceptability.

    python fig2_map.py --panel WORLD FULL_PATHS_DIR GEO_CSV SEED TARGET_X,Y TITLE [--panel ...] --output STEM

FULL_PATHS_DIR is the paths folder that `ruscuav evaluate --save-paths` wrote for FULL_RH_Astar; GEO_CSV the
trajectories of geometric_planner.py. Writes STEM.png and STEM.pdf (figures.sh gives the paper's panels).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ruscuav_benchmark import UrbanWorldLayers  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
ROAD = "#d9d8d1"
BUILDING = "#a8a69e"
PEDESTRIAN = "#1baf7a"
SENSITIVE = "#52514e"
FULL = "#2a78d6"
GEO = "#eb6834"


def blend(fg: str, alpha: float, bg: str = SURFACE) -> np.ndarray:
    return alpha * np.array(to_rgb(fg)) + (1 - alpha) * np.array(to_rgb(bg))


def layer_image(world: UrbanWorldLayers) -> np.ndarray:
    img = np.empty(world.terrain_z.shape + (3,))
    img[:] = to_rgb(SURFACE)
    img[world.road_mask > 0.5] = to_rgb(ROAD)
    img[world.sidewalk_mask > 0.5] = blend(PEDESTRIAN, 0.55)
    img[world.building_mask > 0.5] = to_rgb(BUILDING)
    return img


def geo_path(csv_path: str, seed: int) -> np.ndarray:
    with open(csv_path, encoding="utf-8") as f:
        return np.array([(float(r["x"]), float(r["y"])) for r in csv.DictReader(f) if int(r["seed"]) == seed])


def draw(ax, world: UrbanWorldLayers, full_path: np.ndarray, geo: np.ndarray, target_xy, title: str) -> None:
    # the grid frame of the environment: metres from the corner of the first cell; y grows downward (southward)
    width, height = world.width * world.resolution_m, world.height * world.resolution_m
    ax.imshow(layer_image(world), extent=(0.0, width, height, 0.0), origin="upper", interpolation="nearest")
    sens = (world.sensitive_mask > 0.5).astype(float)
    x = np.linspace(0.0, width, sens.shape[1])
    y = np.linspace(0.0, height, sens.shape[0])
    ax.contourf(x, y, sens, levels=[0.5, 1.5], colors="none", hatches=["////"])
    ax.contour(x, y, sens, levels=[0.5], colors=[SENSITIVE], linewidths=0.6)
    ax.plot(full_path[:, 0], full_path[:, 1], color=FULL, lw=1.6, solid_capstyle="round", zorder=4)
    ax.plot(geo[:, 0], geo[:, 1], color=GEO, lw=1.6, ls=(0, (4, 2)), zorder=5)
    ax.plot(*full_path[0, :2], marker="o", ms=6, mfc=INK, mec="white", mew=1.0, zorder=6)
    tx, ty = target_xy
    ax.add_patch(plt.Circle((tx, ty), 30.0, fill=False, ec=INK, lw=0.8, ls=":", zorder=6))
    ax.plot(tx, ty, marker="*", ms=9, mfc=INK, mec="white", mew=0.8, zorder=6)
    ax.set_title(title, loc="left", fontsize=8, color=INK)
    ax.set_xlim(0.0, width)
    ax.set_ylim(height, 0.0)
    ax.set_xticks(np.arange(0.0, width + 1.0, 250.0))
    ax.set_yticks(np.arange(0.0, height + 1.0, 250.0))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.tick_params(length=2, width=0.5)
    ax.set_aspect("equal")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panel", nargs=6, action="append", required=True,
                    metavar=("WORLD", "FULL_PATHS_DIR", "GEO_CSV", "SEED", "TARGET_X,Y", "TITLE"))
    ap.add_argument("--output", required=True, help="file stem; writes .png and .pdf")
    args = ap.parse_args()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#c3c2b7",
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
                         "hatch.color": SENSITIVE, "hatch.linewidth": 0.5})
    fig, axes = plt.subplots(1, len(args.panel), figsize=(7.2, 3.95), facecolor="white")
    for ax, (world_path, full_dir, geo_csv, seed, target_xy, title) in zip(np.atleast_1d(axes), args.panel):
        world = UrbanWorldLayers.load_npz(world_path)
        full_path = np.load(next(Path(full_dir).glob(f"*seed_{int(seed)}_actual_path.npy")))
        target = tuple(float(v) for v in target_xy.split(","))
        draw(ax, world, full_path, geo_path(geo_csv, int(seed)), target, title)
    handles = [Line2D([0], [0], color=FULL, lw=1.6, label="FULL_RH_Astar (package planner)"),
               Line2D([0], [0], color=GEO, lw=1.6, ls=(0, (4, 2)), label="GEO_SHORTEST (external planner)"),
               Line2D([0], [0], marker="o", ls="none", mfc=INK, mec="white", ms=6, label="start"),
               Line2D([0], [0], marker="*", ls="none", mfc=INK, mec="white", ms=9,
                      label="target (30 m success radius)"),
               Patch(facecolor=blend(PEDESTRIAN, 0.55), label="pedestrian ways"),
               Patch(facecolor=ROAD, label="roads"),
               Patch(facecolor=BUILDING, label="buildings"),
               Patch(facecolor="white", edgecolor=SENSITIVE, hatch="////", lw=0.6, label="sensitive zones")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7, handlelength=2.4,
               columnspacing=1.4, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.25, wspace=0.22)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.output}.{ext}", dpi=300, facecolor="white")
    print(f"written {args.output}.png and .pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
