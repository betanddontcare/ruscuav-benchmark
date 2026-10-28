"""The factory functions of the archived RUSCUAV v2.0.0 notebook, on top of RuscuavEnv.

Each function has the name and signature of a notebook factory and returns a
:class:`RuscuavEnv` configured like the notebook environment it replaces. They are
used by ``scripts/migration/env_parity.py`` to check, step by step, that the two
give identical trajectories, and by ``scripts/legacy_parity.py --env refactored``
to run the archived evaluation code (planners, PPO evaluators) on the new
environment. Notebook diagnostics that reach into private methods of the notebook
classes (``_ground_at``, ``_apply_safety_shield``, ...) are not supported. New code
should use :func:`ruscuav_benchmark.env.make_env`.

Procedural cylinder/cone obstacles of the v14 base environment are not supported
(no RUSCUAV experiment used them), so ``n_obstacles`` must be 0.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Optional

from ..world import UrbanWorldLayers
from .config import (
    PRIMARY_NO_TERMINAL,
    SECONDARY_OPERATIONAL_STOP,
    EvaluationRegime,
    LidarConfig,
    ShieldConfig,
    make_config,
)
from .dynamic_risk import REPORTED_DYNAMIC_CONFIG as _REPORTED_DYNAMIC_CONFIG
from .dynamic_risk import DynamicRiskConfig
from .env import RuscuavEnv

__all__ = [
    "UrbanWorldLayers", "EvaluationRegime", "PRIMARY_NO_TERMINAL", "SECONDARY_OPERATIONAL_STOP",
    "M11DynamicRiskConfig", "REPORTED_DYNAMIC_CONFIG", "reported_dynamic_config", "M8_VARIANT_PRESETS",
    "make_ruscuav_env_m4", "make_ruscuav_env_m4_mode", "make_ruscuav_m8_ablation_env_m4_mode",
    "make_ruscuav_geo_baseline_env_m4_mode", "make_ruscuav_env_m11_dynamic_mode",
    "make_ruscuav_env_m11c_dynamic_obs_mode", "make_m12c_real_vienna_env_mode", "make_geo_eval_env",
    "make_static_eval_env",
]

#: The notebook's dynamic configuration class and reported configuration (a dict, as
#: in the notebook: ``M11DynamicRiskConfig(**REPORTED_DYNAMIC_CONFIG)``).
M11DynamicRiskConfig = DynamicRiskConfig
REPORTED_DYNAMIC_CONFIG = asdict(_REPORTED_DYNAMIC_CONFIG)

#: M8 ablation variants of the notebook -> (method preset, safety-stack overrides).
M8_VARIANT_PRESETS = {
    "reward_only": ("RISK+GREEN-RL", {}),
    "obs_only": ("OBS-RL", {}),
    "risk_only": ("RISK-RL", {}),
    "green_only": ("GREEN-RL", {}),
    "no_urban_lidar": ("RISK+OBS-RL", {"lidar": False}),
    "no_action_shield": ("RISK+OBS-RL", {"action_shield": False}),
    "no_geom_shield": ("RISK+OBS-RL", {"geometric_shield": False}),
    "reward_only_full_stack_diag": ("RISK+GREEN-RL", {}),
    "reward_only_no_action_shield_diag": ("RISK+GREEN-RL", {"action_shield": False}),
    "reward_only_no_geom_shield_diag": ("RISK+GREEN-RL", {"geometric_shield": False}),
    "reward_only_no_shields_diag": ("RISK+GREEN-RL", {"action_shield": False, "geometric_shield": False}),
}


def reported_dynamic_config() -> DynamicRiskConfig:
    return DynamicRiskConfig(**REPORTED_DYNAMIC_CONFIG)


def _dynamic_risk(dynamic_config: Any) -> Any:
    """The dynamic configuration for ``make_config``: the default one for None, and a
    :class:`DynamicRiskConfig` for the notebook's own ``M11DynamicRiskConfig`` (a
    dataclass with the same fields), which the archived evaluation code passes."""
    if dynamic_config is None:
        return DynamicRiskConfig()
    if is_dataclass(dynamic_config) and not isinstance(dynamic_config, (type, DynamicRiskConfig)):
        return DynamicRiskConfig(**asdict(dynamic_config))
    return dynamic_config


def _check_obstacles(n_obstacles: int) -> None:
    if int(n_obstacles) != 0:
        raise ValueError("Procedural obstacles are not supported; n_obstacles must be 0")


def _stage_mode(stage: str, mode: str) -> tuple:
    stage, mode = str(stage).upper(), str(mode).lower()
    if mode not in ("fast_train", "strict_eval"):
        raise ValueError("mode must be 'fast_train' or 'strict_eval'.")
    if stage not in ("A", "B", "C"):
        raise ValueError("stage must be A, B, or C.")
    return stage, mode


def _finish(env: RuscuavEnv, seed: Optional[int]) -> RuscuavEnv:
    if seed is not None:
        env.reset(seed=int(seed))
    return env


def _safety(lidar: bool = True, action_shield: bool = True, geometric_shield: bool = True) -> dict:
    return {
        "lidar": LidarConfig(urban=bool(lidar)),
        "shield": ShieldConfig(action_shield=bool(action_shield), geometric_shield=bool(geometric_shield)),
    }


def make_ruscuav_env_m4(world, stage="A", risk_profile="balanced", n_obstacles=0, seed=None,
                        urban_lidar_enabled=True, urban_action_shield_enabled=True,
                        urban_geometric_shield_enabled=True):
    """Full semantic environment at strict sampling resolution, with safety-stack switches
    (the factory of the M4/M6 training code)."""
    _check_obstacles(n_obstacles)
    stage, _ = _stage_mode(stage, "strict_eval")
    cfg = make_config("RISK+OBS-RL", stage=stage, mode="strict_eval", risk_profile=risk_profile,
                      **_safety(urban_lidar_enabled, urban_action_shield_enabled, urban_geometric_shield_enabled))
    return _finish(RuscuavEnv(world, cfg), seed)


def make_ruscuav_env_m4_mode(world, stage="A", mode="fast_train", risk_profile="balanced", n_obstacles=0,
                             seed=None):
    """Full semantic environment (RISK+OBS-RL); also the environment of the planners."""
    _check_obstacles(n_obstacles)
    stage, mode = _stage_mode(stage, mode)
    cfg = make_config("RISK+OBS-RL", stage=stage, mode=mode, risk_profile=risk_profile)
    return _finish(RuscuavEnv(world, cfg), seed)


def make_ruscuav_m8_ablation_env_m4_mode(world, variant_name, stage="A", mode="fast_train",
                                         risk_profile="balanced", n_obstacles=0, seed=None):
    if variant_name not in M8_VARIANT_PRESETS:
        raise ValueError(f"Unknown variant_name={variant_name}. Available: {list(M8_VARIANT_PRESETS)}")
    _check_obstacles(n_obstacles)
    stage, mode = _stage_mode(stage, mode)
    method, safety = M8_VARIANT_PRESETS[variant_name]
    cfg = make_config(method, stage=stage, mode=mode, risk_profile=risk_profile, **_safety(**safety))
    return _finish(RuscuavEnv(world, cfg), seed)


def make_ruscuav_geo_baseline_env_m4_mode(world, mode="fast_train", stage="GEO", risk_profile="balanced",
                                          n_obstacles=0, seed=None):
    """Geometry-only baseline of the M7 experiments: training without urban stopping and
    hard sensitive zones, strict evaluation with both."""
    _check_obstacles(n_obstacles)
    mode = str(mode).lower()
    if mode not in ("fast_train", "strict_eval"):
        raise ValueError("mode must be 'fast_train' or 'strict_eval'.")
    strict = mode == "strict_eval"
    cfg = make_config("GEO-RL", stage="C" if strict else "A", mode=mode, risk_profile=risk_profile)
    return _finish(RuscuavEnv(world, cfg), seed)


def make_ruscuav_env_m11_dynamic_mode(world, stage="C", mode="strict_eval", risk_profile="balanced",
                                      n_obstacles=0, seed=None, dynamic_config: DynamicRiskConfig = None,
                                      reward_only_obs: bool = True):
    """Dynamic-risk environment (DYN-RISK-RL; with reward_only_obs=False also the semantic observation)."""
    _check_obstacles(n_obstacles)
    stage, mode = _stage_mode(stage, mode)
    cfg = make_config(
        "DYN-RISK-RL" if reward_only_obs else "RISK+OBS-RL", stage=stage, mode=mode, risk_profile=risk_profile,
        dynamic_risk=_dynamic_risk(dynamic_config),
    )
    return _finish(RuscuavEnv(world, cfg), seed)


def make_ruscuav_env_m11c_dynamic_obs_mode(world, stage="C", mode="strict_eval", risk_profile="balanced",
                                           n_obstacles=0, seed=None, dynamic_config: DynamicRiskConfig = None):
    """Dynamic-risk environment with the 6-value dynamic observation (DYN-OBS-RL)."""
    _check_obstacles(n_obstacles)
    stage, mode = _stage_mode(stage, mode)
    cfg = make_config(
        "DYN-OBS-RL", stage=stage, mode=mode, risk_profile=risk_profile,
        dynamic_risk=_dynamic_risk(dynamic_config),
    )
    return _finish(RuscuavEnv(world, cfg), seed)


def make_m12c_real_vienna_env_mode(world, policy_kind="reward_only", stage="C", mode="strict_eval",
                                   risk_profile="balanced", n_obstacles=0, seed=None, verbose=False):
    """Real-GIS environment of the M12C transfer experiments (RISK+GREEN-RL or GEO-RL)."""
    policy_kind = str(policy_kind).lower()
    if policy_kind not in ("reward_only", "geometry_only"):
        raise ValueError("policy_kind must be 'reward_only' or 'geometry_only'.")
    _check_obstacles(n_obstacles)
    stage, mode = _stage_mode(stage, mode)
    method = "RISK+GREEN-RL" if policy_kind == "reward_only" else "GEO-RL"
    cfg = make_config(method, stage=stage, mode=mode, risk_profile=risk_profile)
    return _finish(RuscuavEnv(world, cfg), seed)


def make_geo_eval_env(world, regime: EvaluationRegime, seed: int, n_obstacles: int = 0):
    """GEO-RL evaluation environment of the submission-repair evaluations (strict sampling,
    hard sensitive zones, operational stopping as the regime says)."""
    _check_obstacles(n_obstacles)
    cfg = make_config("GEO-RL", stage=regime.stage, mode="strict_eval", hard_sensitive_zones=True,
                      terminate_on_urban_constraint=bool(regime.terminate_on_soft_budgets))
    env = RuscuavEnv(world, cfg)
    env.reset(seed=int(seed))
    return env


_STATIC_METHOD_VARIANTS = {
    "OBS-RL": "obs_only",
    "RISK+GREEN-RL": "reward_only",
    "RISK-RL": "risk_only",
    "GREEN-RL": "green_only",
}


def make_static_eval_env(method: str, world, regime: EvaluationRegime, seed: int, n_obstacles: int = 0):
    """Evaluation environment of a static learned-policy method in a regime."""
    method = str(method)
    if method == "GEO-RL":
        return make_geo_eval_env(world, regime, seed, n_obstacles=n_obstacles)
    if method in _STATIC_METHOD_VARIANTS:
        return make_ruscuav_m8_ablation_env_m4_mode(
            world=world, variant_name=_STATIC_METHOD_VARIANTS[method], stage=regime.stage, mode=regime.mode,
            risk_profile="balanced", n_obstacles=n_obstacles, seed=seed,
        )
    if method == "RISK+OBS-RL":
        return make_ruscuav_env_m4_mode(world=world, stage=regime.stage, mode=regime.mode, risk_profile="balanced",
                                        n_obstacles=n_obstacles, seed=seed)
    raise ValueError(f"Unsupported static policy method: {method}")
