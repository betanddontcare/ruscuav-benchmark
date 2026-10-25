from __future__ import annotations

from pathlib import Path

import numpy as np

from .risk import UrbanRiskModel
from .world import UrbanWorldLayers


def save_world_quicklook(world: UrbanWorldLayers, output_path: str | Path, agl_m: float = 40.0) -> str:
    """Save a compact three-panel quicklook PNG for one world."""
    import matplotlib.pyplot as plt

    risk = UrbanRiskModel(world, profile="balanced").risk_map_at_agl(agl_m)
    obstacle_h = np.maximum(world.building_height, world.canopy_height)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    ax = axes[0]
    ax.imshow(world.green_mask, origin="lower", alpha=0.25)
    ax.imshow(np.ma.masked_where(world.road_mask <= 0.5, world.road_mask), origin="lower", alpha=0.45)
    ax.imshow(np.ma.masked_where(world.sidewalk_mask <= 0.5, world.sidewalk_mask), origin="lower", alpha=0.45)
    ax.imshow(np.ma.masked_where(world.building_mask <= 0.5, world.building_mask), origin="lower", alpha=0.55)
    ax.imshow(np.ma.masked_where(world.sensitive_mask <= 0.5, world.sensitive_mask), origin="lower", alpha=0.55)
    ax.set_title("Semantic layers")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    im = axes[1].imshow(obstacle_h, origin="lower")
    axes[1].set_title("Obstacle height [m]")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    im = axes[2].imshow(risk, origin="lower")
    axes[2].set_title(f"Risk at {agl_m:g} m AGL")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    for a in axes:
        a.set_xlim(0, world.width)
        a.set_ylim(0, world.height)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)
