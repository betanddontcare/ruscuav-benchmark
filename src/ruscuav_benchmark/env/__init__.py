"""RUSCUAV cargo-UAV Gymnasium environment.

The environment needs the ``env`` extra (``pip install -e ".[env]"`` in the repository:
gymnasium, scipy); the configuration classes import without it.

>>> from ruscuav_benchmark import UrbanWorldLayers
>>> from ruscuav_benchmark.env import make_env
>>> world = UrbanWorldLayers.load_npz("worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz")
>>> env = make_env(world, "RISK+GREEN-RL", regime="primary_no_terminal")
>>> obs, info = env.reset(seed=5000)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from ..world import UrbanWorldLayers
from .config import (
    METHOD_PRESETS,
    MODE_SETTINGS,
    PRIMARY_NO_TERMINAL,
    REGIMES,
    SECONDARY_OPERATIONAL_STOP,
    STAGE_SETTINGS,
    CargoConfig,
    DynamicsConfig,
    EnvConfig,
    EpisodeConfig,
    EvaluationRegime,
    LidarConfig,
    MissionConfig,
    RewardConfig,
    ScenarioConfig,
    ShieldConfig,
    make_config,
)
from .dynamic_risk import REPORTED_DYNAMIC_CONFIG, DynamicRiskConfig, DynamicRiskField

if TYPE_CHECKING:
    from .env import RuscuavEnv

_EXTRA_HINT = 'ruscuav_benchmark.env needs the "env" extra: pip install -e ".[env]" in the repository'


def _import_env():
    try:
        from .env import RuscuavEnv
    except ModuleNotFoundError as exc:
        if exc.name in ("gymnasium", "scipy"):
            raise ModuleNotFoundError(f"{_EXTRA_HINT} (missing {exc.name})", name=exc.name) from exc
        raise
    return RuscuavEnv


def __getattr__(name: str):
    if name == "RuscuavEnv":
        return _import_env()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def make_env(world: UrbanWorldLayers, method: str = "RISK+GREEN-RL", *, regime: Any = None,
             stage: Optional[str] = None, mode: Optional[str] = None, seed: Optional[int] = None,
             **overrides: Any) -> "RuscuavEnv":
    """Environment of a method preset (see :data:`METHOD_PRESETS`) in a regime or stage.

    Arguments are those of :func:`make_config`; with ``seed`` the environment is reset once.
    """
    env = _import_env()(world, make_config(method, regime=regime, stage=stage, mode=mode, **overrides))
    if seed is not None:
        env.reset(seed=int(seed))
    return env


__all__ = [
    "RuscuavEnv", "make_env", "make_config", "EnvConfig", "EvaluationRegime", "METHOD_PRESETS", "REGIMES",
    "STAGE_SETTINGS", "MODE_SETTINGS", "PRIMARY_NO_TERMINAL", "SECONDARY_OPERATIONAL_STOP",
    "DynamicsConfig", "CargoConfig", "LidarConfig", "ShieldConfig", "RewardConfig", "EpisodeConfig",
    "MissionConfig", "ScenarioConfig", "DynamicRiskConfig", "DynamicRiskField", "REPORTED_DYNAMIC_CONFIG",
]
