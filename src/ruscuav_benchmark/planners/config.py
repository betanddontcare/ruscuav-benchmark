"""Planner configurations and the planners of the paper.

All planners search a coarse risk-cost grid with A* at a fixed planning altitude
and fly the path with a waypoint controller (:mod:`.controller`). They differ in
what they know and how often they plan:

``static``
    plans once, at the start, on the full risk map (M9 in the notebook);
``receding``
    replans every ``replan_every_steps`` steps from the current position; risk is
    known within ``sensing_radius_m`` of positions visited so far and estimated
    elsewhere by ``prior_risk_mode`` (M10). With the ``"perfect"`` prior the whole
    map is known, which is the full-map receding planner;
``dynamic``
    a receding planner that also adds the episode's dynamic risk, known within the
    sensing radius and estimated elsewhere by ``dynamic_prior_mode`` (M11).

:data:`PLANNER_PRESETS` holds the four planners of the paper with the exact
settings of the archived runs.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Dict

PLANNER_KINDS = ("static", "receding", "dynamic")
PRIOR_RISK_MODES = ("zero", "mean", "blurred", "perfect")
DYNAMIC_PRIOR_MODES = ("zero", "mean")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _check_field_types(config: Any) -> None:
    """Every field must have the kind of value its annotation names; numbers are
    stored as Python floats and ints."""
    for f in fields(config):
        value = getattr(config, f.name)
        kind = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", "")
        if kind == "float":
            ok = isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)
            expected = "a finite number"
        elif kind == "int":
            ok = isinstance(value, numbers.Integral) and not isinstance(value, bool)
            expected = "an integer"
        elif kind == "bool":
            ok = isinstance(value, bool)
            expected = "True or False"
        elif kind == "str":
            ok = isinstance(value, str)
            expected = "a string"
        else:
            continue
        _require(ok, f"{f.name} must be {expected}, got {value!r}")
        if kind in ("float", "int"):
            object.__setattr__(config, f.name, float(value) if kind == "float" else int(value))


@dataclass(frozen=True)
class PlannerConfig:
    """Risk-cost A* on a coarse grid, followed by a waypoint controller.

    The cost of moving between neighbouring cells is their distance times
    ``1 + risk_lambda * r``, with ``r`` the semantic risk at ``planner_agl_m``
    averaged over the cell and divided by its 95th percentile over the map (capped at
    10). Cells where buildings or canopy, plus a clearance, reach the planning
    altitude are blocked, and so are sensitive zones when the planning altitude is
    below ``sensitive_min_agl_m``.
    """

    #: Cell size of the planning grid in metres (rounded to a multiple of the world resolution).
    plan_res_m: float = 5.0
    #: Altitude above ground at which risk and blockers are evaluated.
    planner_agl_m: float = 40.0
    #: Weight of normalised risk in the cell cost.
    risk_lambda: float = 0.35
    #: A* (True) or Dijkstra (False).
    use_astar: bool = True
    block_buildings: bool = True
    block_canopy: bool = True
    block_sensitive_low_altitude: bool = True
    building_clearance_m: float = 5.0
    canopy_clearance_m: float = 3.0
    sensitive_min_agl_m: float = 80.0
    #: The controller moves on to the next waypoint within this distance.
    waypoint_radius_m: float = 8.0
    #: Recorded with the configuration; the controller does not use it.
    waypoint_lookahead_m: float = 12.0
    #: Cruise speed. Closer to the target than 1.25 times this many metres, the
    #: controller flies at 0.8 m/s per metre of remaining distance (at least 2 m/s).
    target_speed_mps: float = 12.0
    #: Episode step limit of the rollout.
    max_steps: int = 1000

    def __post_init__(self) -> None:
        _check_field_types(self)
        _require(self.plan_res_m > 0, "plan_res_m must be positive")
        _require(self.risk_lambda >= 0, "risk_lambda must not be negative")
        _require(self.target_speed_mps > 0, "target_speed_mps must be positive")
        _require(self.max_steps >= 1, "max_steps must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecedingPlannerConfig(PlannerConfig):
    """:class:`PlannerConfig` of a planner that replans as it flies."""

    #: At every replan the true risk becomes known within this radius of the UAV.
    sensing_radius_m: float = 75.0
    replan_every_steps: int = 5
    #: Risk assumed where it is not known yet: "zero", "mean" (map mean), "blurred"
    #: (Gaussian-blurred true risk, sigma of 5 cells) or "perfect" (true risk everywhere).
    prior_risk_mode: str = "mean"
    #: Blocked cells known in advance everywhere (True) or only within the sensing radius.
    hard_blockers_known: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _require(self.sensing_radius_m > 0, "sensing_radius_m must be positive")
        _require(self.replan_every_steps >= 1, "replan_every_steps must be at least 1")
        _require(self.prior_risk_mode.lower() in PRIOR_RISK_MODES,
                 f"prior_risk_mode={self.prior_risk_mode!r}; expected one of {list(PRIOR_RISK_MODES)}")


@dataclass(frozen=True)
class PlannerSpec:
    """A named planner: its kind, configuration and (dynamic planner) dynamic-risk prior."""

    name: str
    kind: str
    config: PlannerConfig
    dynamic_prior_mode: str = "zero"
    description: str = ""

    def __post_init__(self) -> None:
        _require(self.kind in PLANNER_KINDS, f"kind={self.kind!r}; expected one of {list(PLANNER_KINDS)}")
        if self.kind == "static":
            _require(type(self.config) is PlannerConfig, "a static planner takes a PlannerConfig")
        else:
            _require(isinstance(self.config, RecedingPlannerConfig),
                     f"a {self.kind} planner takes a RecedingPlannerConfig")
        _require(self.dynamic_prior_mode in DYNAMIC_PRIOR_MODES,
                 f"dynamic_prior_mode={self.dynamic_prior_mode!r}; expected one of {list(DYNAMIC_PRIOR_MODES)}")

    @property
    def env_method(self) -> str:
        """Method preset of the environment the archived runs flew this planner in."""
        return "DYN-RISK-RL" if self.kind == "dynamic" else "RISK+OBS-RL"

    def with_config(self, **changes: Any) -> "PlannerSpec":
        """The same planner with some configuration fields (or ``dynamic_prior_mode``) changed."""
        changes = dict(changes)
        spec_changes = {"dynamic_prior_mode": changes.pop("dynamic_prior_mode")} if "dynamic_prior_mode" in changes else {}
        known = {f.name for f in fields(self.config)}
        unknown = sorted(set(changes) - known)
        _require(not unknown, f"unknown {type(self.config).__name__} fields: {unknown}")
        return replace(self, config=replace(self.config, **changes), **spec_changes)

    def to_dict(self) -> Dict[str, Any]:
        out = {"name": self.name, "kind": self.kind, "config": self.config.to_dict()}
        if self.kind == "dynamic":
            out["dynamic_prior_mode"] = self.dynamic_prior_mode
        return out


#: Settings of the archived runs (REPORTED_M9_CONFIG ... REPORTED_M11_PLANNER_CONFIG of
#: the v2.0.0 submission-repair code); fields not listed keep their defaults.
REPORTED_STATIC_CONFIG: Dict[str, Any] = {
    "plan_res_m": 5.0,
    "planner_agl_m": 40.0,
    "risk_lambda": 0.35,
    "use_astar": True,
    "waypoint_radius_m": 8.0,
    "target_speed_mps": 12.0,
    "max_steps": 1000,
}
REPORTED_LOCAL_RH_CONFIG: Dict[str, Any] = {
    **REPORTED_STATIC_CONFIG,
    "sensing_radius_m": 75.0,
    "replan_every_steps": 5,
    "prior_risk_mode": "mean",
    "hard_blockers_known": True,
}
REPORTED_FULL_RH_CONFIG: Dict[str, Any] = {**REPORTED_LOCAL_RH_CONFIG, "prior_risk_mode": "perfect"}
REPORTED_DYN_RH_CONFIG: Dict[str, Any] = {**REPORTED_FULL_RH_CONFIG, "replan_every_steps": 3}

#: The planners of the paper. The real-GIS runs flew FULL_RH_Astar at
#: ``planner_agl_m=30`` (``PLANNER_PRESETS["FULL_RH_Astar"].with_config(planner_agl_m=30.0)``).
PLANNER_PRESETS: Dict[str, PlannerSpec] = {
    "STATIC_Astar": PlannerSpec(
        "STATIC_Astar", "static", PlannerConfig(**REPORTED_STATIC_CONFIG),
        description="full-map risk-cost A*, planned once at the start",
    ),
    "LOCAL_RH_Astar": PlannerSpec(
        "LOCAL_RH_Astar", "receding", RecedingPlannerConfig(**REPORTED_LOCAL_RH_CONFIG),
        description="receding-horizon A*, risk known within 75 m, map mean elsewhere",
    ),
    "FULL_RH_Astar": PlannerSpec(
        "FULL_RH_Astar", "receding", RecedingPlannerConfig(**REPORTED_FULL_RH_CONFIG),
        description="receding-horizon A* on the full risk map",
    ),
    "DYN_RH_Astar": PlannerSpec(
        "DYN_RH_Astar", "dynamic", RecedingPlannerConfig(**REPORTED_DYN_RH_CONFIG), dynamic_prior_mode="zero",
        description="receding-horizon A* on the full static map plus dynamic risk sensed within 75 m",
    ),
}


def get_planner(name: str, **config_changes: Any) -> PlannerSpec:
    """The preset ``name``, optionally with some configuration fields changed."""
    if name not in PLANNER_PRESETS:
        raise ValueError(f"Unknown planner {name!r}; available: {list(PLANNER_PRESETS)}")
    spec = PLANNER_PRESETS[name]
    return spec.with_config(**config_changes) if config_changes else spec
