"""Risk-cost A* planners of the RUSCUAV paper and their evaluation.

>>> from ruscuav_benchmark import UrbanWorldLayers
>>> from ruscuav_benchmark.planners import evaluate_planner
>>> world = UrbanWorldLayers.load_npz("worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz")  # doctest: +SKIP
>>> result = evaluate_planner(world, "FULL_RH_Astar", n_episodes=5)                              # doctest: +SKIP
>>> result.summary["mission_acceptability_rate"]                                                  # doctest: +SKIP

Planners (:data:`PLANNER_PRESETS`): ``STATIC_Astar``, ``LOCAL_RH_Astar``,
``FULL_RH_Astar`` and ``DYN_RH_Astar``; see docs/PLANNERS.md. The planner classes
need only NumPy; flying them (:mod:`.rollout`, :func:`evaluate_planner`) needs the
environment, i.e. the ``env`` extra.
"""

from .config import (
    DYNAMIC_PRIOR_MODES,
    PLANNER_KINDS,
    PLANNER_PRESETS,
    PRIOR_RISK_MODES,
    REPORTED_DYN_RH_CONFIG,
    REPORTED_FULL_RH_CONFIG,
    REPORTED_LOCAL_RH_CONFIG,
    REPORTED_STATIC_CONFIG,
    PlannerConfig,
    PlannerSpec,
    RecedingPlannerConfig,
    get_planner,
)
from .controller import waypoint_action
from .dynamic import DynamicRecedingPlanner
from ..scenarios import missions_from_csv
from ..tables import write_rows_csv
from .evaluation import PlannerEvaluation, evaluate_planner, make_planner, summarize_planner_rows
from .grid import RiskCostGrid, astar, path_length_xy, plan_path
from .receding import RecedingHorizonPlanner
from .rollout import run_dynamic_episode, run_episode, run_receding_episode, run_static_episode
from .static import StaticPlanner

__all__ = [
    "PLANNER_KINDS", "PLANNER_PRESETS", "PRIOR_RISK_MODES", "DYNAMIC_PRIOR_MODES",
    "REPORTED_STATIC_CONFIG", "REPORTED_LOCAL_RH_CONFIG", "REPORTED_FULL_RH_CONFIG", "REPORTED_DYN_RH_CONFIG",
    "PlannerConfig", "RecedingPlannerConfig", "PlannerSpec", "get_planner",
    "RiskCostGrid", "astar", "plan_path", "path_length_xy", "waypoint_action",
    "StaticPlanner", "RecedingHorizonPlanner", "DynamicRecedingPlanner", "make_planner",
    "run_static_episode", "run_receding_episode", "run_dynamic_episode", "run_episode",
    "evaluate_planner", "missions_from_csv", "PlannerEvaluation", "summarize_planner_rows", "write_rows_csv",
]
