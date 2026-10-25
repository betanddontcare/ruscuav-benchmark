"""RUSCUAV benchmark package.

A compact, reproducible implementation of the semantic mission-acceptability
benchmark for low-altitude urban cargo-UAV navigation.
"""

from .world import (
    AOIManifestEntry,
    LEGACY_AOI_ALIASES,
    UrbanWorldLayers,
    WORLD_SCHEMA_VERSION,
    canonical_aoi_id,
    run_world_gate_checks,
)
from .risk import UrbanRiskModel, UrbanRiskProfile, RISK_PROFILES
from .metrics import ExposureBudgets, evaluate_path, summarize_episode_records
from .planners import RiskAStarConfig, RiskCostAStarPlanner

#: Single source of the package version (pyproject.toml reads it from here).
__version__ = "2.1.0.dev0"

__all__ = [
    "AOIManifestEntry",
    "LEGACY_AOI_ALIASES",
    "UrbanWorldLayers",
    "WORLD_SCHEMA_VERSION",
    "canonical_aoi_id",
    "run_world_gate_checks",
    "UrbanRiskModel",
    "UrbanRiskProfile",
    "RISK_PROFILES",
    "ExposureBudgets",
    "evaluate_path",
    "summarize_episode_records",
    "RiskAStarConfig",
    "RiskCostAStarPlanner",
]
