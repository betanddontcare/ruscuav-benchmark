"""RUSCUAV-v0 benchmark package.

A compact, reproducible implementation of the semantic mission-acceptability
benchmark for low-altitude urban cargo-UAV navigation.
"""

from .world import AOIManifestEntry, UrbanWorldLayers, run_world_gate_checks
from .risk import UrbanRiskModel, UrbanRiskProfile, RISK_PROFILES
from .metrics import ExposureBudgets, evaluate_path, summarize_episode_records
from .planners import RiskAStarConfig, RiskCostAStarPlanner

__version__ = "0.1.0"

__all__ = [
    "AOIManifestEntry",
    "UrbanWorldLayers",
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
