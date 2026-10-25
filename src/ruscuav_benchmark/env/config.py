"""Configuration of the RUSCUAV cargo-UAV environment.

One environment class, :class:`~ruscuav_benchmark.env.RuscuavEnv`, covers every
variant used in the paper. A variant is chosen with a few high-level settings:

``observation``
    ``"geometry"`` (80 values: kinematics, target, 64 LiDAR rays, cargo),
    ``"semantic"`` (geometry + 20 static urban-semantics values),
    ``"dynamic"`` (geometry + 6 dynamic-risk values) or ``"semantic+dynamic"``.
``urban_reward``
    Which urban terms enter the reward: ``"none"``, ``"risk"`` (semantic risk
    penalty), ``"green"`` (green-corridor bonus) or ``"risk+green"``.
``stage``
    Curriculum stage ``"A"``, ``"B"`` or ``"C"`` (see :data:`STAGE_SETTINGS`). The
    paper's evaluation regimes are stages: the primary no-terminal regime is stage B
    and the secondary operational-stopping regime is stage C (:data:`REGIMES`).
``mode``
    ``"fast_train"`` or ``"strict_eval"``: sampling resolution of the urban LiDAR
    and of the geometric shield (:data:`MODE_SETTINGS`).
``dynamic_risk``
    Optional :class:`DynamicRiskConfig`; adds an episode-specific dynamic risk
    field to the reward, the metrics and (for dynamic observations) the observation.

The detailed constants (dynamics, reward weights, budgets, ...) live in the
parameter groups below. Their defaults are the values of the archived RUSCUAV
v2.0.0 notebook environment, so the defaults reproduce the paper.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Optional, Tuple, Union

from ..risk import RISK_PROFILES, UrbanRiskProfile
from .dynamic_risk import REPORTED_DYNAMIC_CONFIG, DynamicRiskConfig

def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


STAGES = ("A", "B", "C")
MODES = ("fast_train", "strict_eval")
OBSERVATIONS = ("geometry", "semantic", "dynamic", "semantic+dynamic")
URBAN_REWARDS = ("none", "risk", "green", "risk+green")


@dataclass(frozen=True)
class StageSettings:
    """Settings that change between curriculum stages."""

    #: Weight of the semantic risk penalty (before the reward-kind multiplier).
    urban_risk_lambda: float
    #: End the episode on urban constraint violations and exceeded terminal budgets.
    terminate_on_urban_constraint: bool
    #: Treat sensitive zones below the overflight altitude as hard constraints.
    hard_sensitive_zones: bool


STAGE_SETTINGS: Dict[str, StageSettings] = {
    "A": StageSettings(urban_risk_lambda=0.05, terminate_on_urban_constraint=False, hard_sensitive_zones=False),
    "B": StageSettings(urban_risk_lambda=0.10, terminate_on_urban_constraint=False, hard_sensitive_zones=True),
    "C": StageSettings(urban_risk_lambda=0.10, terminate_on_urban_constraint=True, hard_sensitive_zones=True),
}


@dataclass(frozen=True)
class SensingResolution:
    lidar_step_m: float
    segment_check_step_m: float


MODE_SETTINGS: Dict[str, SensingResolution] = {
    "fast_train": SensingResolution(lidar_step_m=5.0, segment_check_step_m=2.0),
    "strict_eval": SensingResolution(lidar_step_m=3.0, segment_check_step_m=1.0),
}

@dataclass(frozen=True)
class EvaluationRegime:
    """An evaluation regime of the paper: a stage evaluated in a sensing mode."""

    name: str
    stage: str
    mode: str = "strict_eval"
    terminate_on_soft_budgets: bool = False
    description: str = ""


PRIMARY_NO_TERMINAL = EvaluationRegime(
    name="primary_no_terminal",
    stage="B",
    mode="strict_eval",
    terminate_on_soft_budgets=False,
    description=(
        "Hard physical and low-altitude sensitive-zone constraints are active; "
        "road, pedestrian, building, and dynamic budget exceedances are logged "
        "but do not terminate the episode."
    ),
)
SECONDARY_OPERATIONAL_STOP = EvaluationRegime(
    name="secondary_operational_stopping",
    stage="C",
    mode="strict_eval",
    terminate_on_soft_budgets=True,
    description=(
        "Stage C operational stress test: selected semantic-budget exceedances "
        "can terminate the episode."
    ),
)
REGIMES: Dict[str, EvaluationRegime] = {r.name: r for r in (PRIMARY_NO_TERMINAL, SECONDARY_OPERATIONAL_STOP)}

#: Urban reward kind -> (use the urban risk terms, risk-weight multiplier, green bonus).
URBAN_REWARD_SETTINGS: Dict[str, Tuple[bool, float, float]] = {
    "none": (False, 0.0, 0.0),
    "risk": (True, 1.0, 0.0),
    "green": (True, 0.0, 0.03),
    "risk+green": (True, 1.0, 0.03),
}


@dataclass(frozen=True)
class DynamicsConfig:
    """Point-mass dynamics: filtered horizontal acceleration command, jerk limit and an
    altitude autopilot that tracks a terrain-following reference (metres, seconds)."""

    dt: float = 0.5
    v_max_xy: float = 18.0
    v_max_z: float = 6.0
    a_max_xy: float = 6.0
    a_max_z: float = 3.0
    jerk_max: float = 30.0
    #: First-order filter weights of the previous acceleration.
    alpha_xy: float = 0.2
    alpha_z: float = 0.2
    #: Altitude band above terrain tracked by the autopilot.
    cruise_agl_m: float = 30.0
    min_agl_m: float = 2.0
    max_agl_m: float = 80.0
    kp_z: float = 0.35
    kd_z: float = 0.90
    #: Raise the altitude reference over buildings and canopy (roof/top + minimum clearance).
    urban_altitude_reference: bool = True
    #: Upper altitude bound of the workspace, in the height reference of the world's
    #: terrain. Worlds with terrain heights above sea level need it raised (or
    #: terrain heights relative to the area's lowest point).
    z_max_m: float = 150.0
    #: Constant horizontal wind drift (m/s). It only shifts the position update; the
    #: velocity state and the action shield's one-step prediction do not include it.
    wind_xy: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        _require(self.dt > 0, "dynamics.dt must be positive")
        _require(self.min_agl_m <= self.cruise_agl_m <= self.max_agl_m,
                 "dynamics needs min_agl_m <= cruise_agl_m <= max_agl_m")


@dataclass(frozen=True)
class CargoConfig:
    """Cargo health decays quadratically with the acceleration above the soft limit."""

    g0: float = 9.81
    overload_soft_g: float = 0.30
    health_decay_k: float = 0.005


@dataclass(frozen=True)
class LidarConfig:
    """Planar range sensor centred on the direction to the target."""

    #: Number of rays; the observation has 16 + n_rays geometry values.
    n_rays: int = 64
    half_angle_deg: float = 135.0
    range_m: float = 250.0
    #: Detect urban hazards (buildings, canopy, sensitive volumes, terrain). When
    #: False every ray reads the maximum range.
    urban: bool = True
    #: Ray-marching step; None takes it from ``mode``.
    step_m: Optional[float] = None

    def __post_init__(self) -> None:
        _require(int(self.n_rays) >= 1, "lidar.n_rays must be at least 1")
        _require(self.range_m > 0, "lidar.range_m must be positive")


@dataclass(frozen=True)
class ShieldConfig:
    """Urban safety shields."""

    #: Replace actions whose one-step prediction enters an urban hazard.
    action_shield: bool = True
    #: Stop the position update before it enters an urban hazard.
    geometric_shield: bool = True
    #: Distance kept before the first hazard on the motion segment.
    backoff_m: float = 0.50
    #: Margin subtracted from the horizontal distance to building, canopy and
    #: sensitive footprints in the signed-clearance proxy that scores action-shield
    #: candidates. It does not enlarge the hazards themselves, which are per cell.
    horizontal_margin_m: float = 1.0
    #: Sampling step along the motion segment; None takes it from ``mode``.
    segment_check_step_m: Optional[float] = None


@dataclass(frozen=True)
class RewardConfig:
    """Weights of the reward terms (see :mod:`ruscuav_benchmark.env.reward`)."""

    progress_scale: float = 12.0
    align_k: float = 0.8
    speed_min_mps: float = 2.0
    low_speed_penalty: float = 0.5
    time_penalty: float = 0.1
    cte_soft_limit_m: float = 20.0
    cte_corridor_limit_m: float = 100.0
    cte_quad_k: float = 0.002
    cte_lin_k: float = 0.03
    cte_cap: float = 14.0
    lidar_k: float = 150.0
    lidar_trigger: float = 0.25
    lidar_cap: float = 30.0
    terrain_collision_penalty: float = 500.0
    shield_penalty: float = 5.0
    #: Extra penalty when a shielded step moves less than ``shield_blocked_step_m``.
    shield_blocked_penalty: float = 10.0
    shield_blocked_step_m: float = 0.1
    shield_stuck_penalty: float = 1500.0
    altitude_k: float = 0.04
    altitude_ref_agl_m: float = 30.0
    acceleration_k: float = 0.001
    overload_k: float = 80.0
    boundary_margin_m: float = 35.0
    boundary_k: float = 0.015
    boundary_cap: float = 25.0
    success_bonus: float = 3000.0
    near_success_bonus: float = 1500.0
    stuck_penalty: float = 200.0
    cargo_failure_penalty: float = 200.0
    #: Penalty for an urban termination (stage C).
    urban_failure_penalty: float = 1000.0
    #: Bonus for a success that also meets the mission-acceptability budgets.
    acceptable_success_bonus: float = 500.0


@dataclass(frozen=True)
class EpisodeConfig:
    """Termination conditions."""

    max_steps: int = 1000
    success_radius_m: float = 30.0
    near_success_radius_m: float = 45.0
    #: Distance improvement that resets the stuck counter.
    improve_eps_m: float = 0.5
    stuck_patience_steps: int = 300
    shield_stuck_patience_steps: int = 40
    shield_progress_eps_m: float = 0.20
    shield_min_step_xy_m: float = 0.05
    #: Height above terrain at or below which the episode ends with
    #: ``terrain_collision``. (The risk profile's terrain clearance, 2 m, defines the
    #: terrain hard violation and the stage C ``urban_terrain_constraint`` stop.)
    terrain_margin_m: float = 1.0

    def __post_init__(self) -> None:
        _require(int(self.max_steps) >= 1, "episode.max_steps must be at least 1")
        _require(self.success_radius_m <= self.near_success_radius_m,
                 "episode.success_radius_m must not exceed near_success_radius_m")


@dataclass(frozen=True)
class MissionConfig:
    """Mission-acceptability budgets, operational-stopping budgets and safe endpoints."""

    accept_cargo_min: float = 0.80
    accept_road_s: float = 20.0
    accept_sidewalk_s: float = 5.0
    accept_building_s: float = 15.0
    accept_sensitive_s: float = 0.0
    terminal_road_s: float = 30.0
    terminal_sidewalk_s: float = 10.0
    terminal_building_s: float = 20.0
    terminal_sensitive_s: float = 0.0
    #: Resample missions whose start or target lies on a building, road, sidewalk or
    #: sensitive cell, violates a hard constraint, or has risk above ``start_target_max_risk``.
    enforce_safe_start_target: bool = True
    start_target_max_risk: float = 1.0
    max_reset_attempts: int = 300

    def __post_init__(self) -> None:
        _require(int(self.max_reset_attempts) >= 1, "mission.max_reset_attempts must be at least 1")


@dataclass(frozen=True)
class ScenarioConfig:
    """Random missions: integer start and target cells at least ``min_distance_m``
    apart, with a feasible altitude profile along a grid path between them."""

    spawn_margin_m: int = 50
    min_distance_m: float = 500.0
    max_attempts: int = 1000

    def __post_init__(self) -> None:
        _require(int(self.max_attempts) >= 1, "scenario.max_attempts must be at least 1")
        _require(self.feasibility_grid_m > 0, "scenario.feasibility_grid_m must be positive")
    feasibility_check: bool = True
    feasibility_grid_m: float = 10.0
    max_detour_ratio: float = 2.75
    nominal_speed_mps: float = 12.0
    max_vz_ratio: float = 0.90
    max_az_ratio: float = 0.90
    profile_step_m: float = 5.0


@dataclass(frozen=True)
class EnvConfig:
    """Complete configuration of :class:`~ruscuav_benchmark.env.RuscuavEnv`."""

    observation: str = "geometry"
    urban_reward: str = "risk+green"
    stage: str = "B"
    mode: str = "strict_eval"
    #: Name of a profile in ``ruscuav_benchmark.RISK_PROFILES`` or an ``UrbanRiskProfile``.
    risk_profile: Union[str, UrbanRiskProfile] = "balanced"
    #: Dynamic risk field (a ``DynamicRiskConfig`` or a dict of its fields), or None.
    dynamic_risk: Optional[DynamicRiskConfig] = None
    #: Overrides of stage- and reward-derived values (None: derived).
    urban_risk_lambda: Optional[float] = None
    green_bonus_lambda: Optional[float] = None
    hard_sensitive_zones: Optional[bool] = None
    terminate_on_urban_constraint: Optional[bool] = None
    #: Free-form label, e.g. the method preset this configuration came from.
    label: str = ""

    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    cargo: CargoConfig = field(default_factory=CargoConfig)
    lidar: LidarConfig = field(default_factory=LidarConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    mission: MissionConfig = field(default_factory=MissionConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)

    def __post_init__(self) -> None:
        for name, value, allowed in (
            ("observation", self.observation, OBSERVATIONS),
            ("urban_reward", self.urban_reward, URBAN_REWARDS),
            ("stage", self.stage, STAGES),
            ("mode", self.mode, MODES),
        ):
            if value not in allowed:
                raise ValueError(f"{name}={value!r}; expected one of {list(allowed)}")
        if not isinstance(self.risk_profile, UrbanRiskProfile) and self.risk_profile not in RISK_PROFILES:
            raise ValueError(f"risk_profile={self.risk_profile!r}; expected an UrbanRiskProfile or one of "
                             f"{list(RISK_PROFILES)}")
        if isinstance(self.dynamic_risk, dict):
            object.__setattr__(self, "dynamic_risk", DynamicRiskConfig(**self.dynamic_risk))
        if self.dynamic_risk is not None and not isinstance(self.dynamic_risk, DynamicRiskConfig):
            raise TypeError(f"dynamic_risk must be a DynamicRiskConfig, a dict or None, not "
                            f"{type(self.dynamic_risk).__name__}")
        if self.dynamic_observation and self.dynamic_risk is None:
            raise ValueError(f"observation={self.observation!r} needs a dynamic_risk configuration")
        if self.urban_reward == "none" and (self.urban_risk_lambda or self.green_bonus_lambda):
            raise ValueError("urban_reward='none' has no urban reward terms; urban_risk_lambda and "
                             "green_bonus_lambda would have no effect")

    # ------------------------------------------------------------------ derived values
    @property
    def semantic_observation(self) -> bool:
        return self.observation in ("semantic", "semantic+dynamic")

    @property
    def dynamic_observation(self) -> bool:
        return self.observation in ("dynamic", "semantic+dynamic")

    @property
    def use_urban_risk_costs(self) -> bool:
        return URBAN_REWARD_SETTINGS[self.urban_reward][0]

    @property
    def resolved_urban_risk_lambda(self) -> float:
        if self.urban_risk_lambda is not None:
            return float(self.urban_risk_lambda)
        multiplier = URBAN_REWARD_SETTINGS[self.urban_reward][1]
        return float(STAGE_SETTINGS[self.stage].urban_risk_lambda * float(multiplier))

    @property
    def resolved_green_bonus_lambda(self) -> float:
        if self.green_bonus_lambda is not None:
            return float(self.green_bonus_lambda)
        return float(URBAN_REWARD_SETTINGS[self.urban_reward][2])

    @property
    def resolved_hard_sensitive_zones(self) -> bool:
        if self.hard_sensitive_zones is not None:
            return bool(self.hard_sensitive_zones)
        return STAGE_SETTINGS[self.stage].hard_sensitive_zones

    @property
    def resolved_terminate_on_urban_constraint(self) -> bool:
        if self.terminate_on_urban_constraint is not None:
            return bool(self.terminate_on_urban_constraint)
        return STAGE_SETTINGS[self.stage].terminate_on_urban_constraint

    @property
    def resolved_lidar_step_m(self) -> float:
        step = self.lidar.step_m if self.lidar.step_m is not None else MODE_SETTINGS[self.mode].lidar_step_m
        return float(max(0.5, step))

    @property
    def resolved_segment_check_step_m(self) -> float:
        step = self.shield.segment_check_step_m
        if step is None:
            step = MODE_SETTINGS[self.mode].segment_check_step_m
        return float(max(0.25, step))

    @property
    def observation_dim(self) -> int:
        geometry = 16 + int(self.lidar.n_rays)
        return geometry + (20 if self.semantic_observation else 0) + (6 if self.dynamic_observation else 0)

    def with_changes(self, **changes: Any) -> "EnvConfig":
        """Return a copy with some fields replaced (``dataclasses.replace``)."""
        return replace(self, **changes)

    def to_dict(self) -> Dict[str, Any]:
        """Plain-dict form including the derived values, e.g. for experiment logs."""
        d = asdict(self)
        d["resolved"] = {
            "observation_dim": self.observation_dim,
            "use_urban_risk_costs": self.use_urban_risk_costs,
            "urban_risk_lambda": self.resolved_urban_risk_lambda,
            "green_bonus_lambda": self.resolved_green_bonus_lambda,
            "hard_sensitive_zones": self.resolved_hard_sensitive_zones,
            "terminate_on_urban_constraint": self.resolved_terminate_on_urban_constraint,
            "lidar_step_m": self.resolved_lidar_step_m,
            "segment_check_step_m": self.resolved_segment_check_step_m,
        }
        return d


#: Learned-policy methods of the paper (short names of the method-naming table)
#: and the observation, urban reward and dynamic risk they use.
METHOD_PRESETS: Dict[str, Dict[str, Any]] = {
    "GEO-RL": {"observation": "geometry", "urban_reward": "none"},
    "OBS-RL": {"observation": "semantic", "urban_reward": "none"},
    "RISK-RL": {"observation": "geometry", "urban_reward": "risk"},
    "GREEN-RL": {"observation": "geometry", "urban_reward": "green"},
    "RISK+GREEN-RL": {"observation": "geometry", "urban_reward": "risk+green"},
    "RISK+OBS-RL": {"observation": "semantic", "urban_reward": "risk+green"},
    "DYN-RISK-RL": {"observation": "geometry", "urban_reward": "risk+green", "dynamic_risk": REPORTED_DYNAMIC_CONFIG},
    "DYN-OBS-RL": {"observation": "dynamic", "urban_reward": "risk+green", "dynamic_risk": REPORTED_DYNAMIC_CONFIG},
}


def make_config(
    method: str = "RISK+GREEN-RL",
    *,
    regime: "str | EvaluationRegime | None" = None,
    stage: Optional[str] = None,
    mode: Optional[str] = None,
    **overrides: Any,
) -> EnvConfig:
    """Configuration of a method preset in an evaluation regime or a training stage.

    ``regime`` (``"primary_no_terminal"``, ``"secondary_operational_stopping"`` or an
    :class:`EvaluationRegime`) sets stage, mode and operational stopping
    (``terminate_on_soft_budgets``); otherwise ``stage`` defaults to ``"B"`` and
    ``mode`` to ``"strict_eval"``. Further keyword arguments override
    :class:`EnvConfig` fields.

    >>> make_config("RISK+GREEN-RL", regime="primary_no_terminal").stage
    'B'
    >>> make_config("OBS-RL", stage="A", mode="fast_train").observation_dim
    100
    """
    if method not in METHOD_PRESETS:
        raise ValueError(f"Unknown method {method!r}; available: {list(METHOD_PRESETS)}")
    if regime is not None:
        if isinstance(regime, str):
            if regime not in REGIMES:
                raise ValueError(f"Unknown regime {regime!r}; available: {list(REGIMES)}")
            regime = REGIMES[regime]
        if stage is not None or mode is not None:
            raise ValueError("Give either regime or stage/mode, not both")
        stage, mode = regime.stage, regime.mode
        overrides.setdefault("terminate_on_urban_constraint", bool(regime.terminate_on_soft_budgets))
    settings = dict(METHOD_PRESETS[method])
    settings.update(stage=stage or "B", mode=mode or "strict_eval", label=method)
    settings.update(overrides)
    return EnvConfig(**settings)
