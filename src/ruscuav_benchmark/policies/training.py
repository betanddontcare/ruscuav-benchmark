"""Train a PPO policy with the settings and curriculum of the paper.

The archived policies were trained with Stable-Baselines3 PPO (``MlpPolicy``, two
hidden layers of 256 units) on four parallel environments in the ``fast_train``
sensing mode, through the stages A, B and C of the curriculum
(:data:`TRAINING_SCHEDULES`); each stage continues from the model of the previous
one. :func:`train_ppo` follows that recipe on any world. The package checks that it
runs, saves and reloads; it was not used to re-train the archived checkpoints, and
training is not bit-for-bit reproducible across machines anyway.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from ..world import UrbanWorldLayers

__all__ = ["PPO_SETTINGS", "TRAINING_SCHEDULES", "MONITOR_INFO_KEYWORDS", "make_training_env", "train_ppo"]

#: PPO hyperparameters of the archived training runs. ``n_steps`` was 512 with four
#: or more parallel environments and 1024 otherwise.
PPO_SETTINGS: Dict[str, Any] = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "batch_size": 256,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "policy_kwargs": {"net_arch": [256, 256]},
}

#: Curriculum of the archived training runs: (stage, environment steps) per method
#: preset. GEO-RL trained in one stage (stage A settings); RISK+OBS-RL is the longer
#: single-checkpoint run of the full semantic environment.
TRAINING_SCHEDULES: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "GEO-RL": (("A", 200_000),),
    "OBS-RL": (("A", 50_000), ("B", 75_000), ("C", 75_000)),
    "RISK-RL": (("A", 50_000), ("B", 75_000), ("C", 75_000)),
    "GREEN-RL": (("A", 50_000), ("B", 75_000), ("C", 75_000)),
    "RISK+GREEN-RL": (("A", 50_000), ("B", 75_000), ("C", 75_000)),
    "RISK+OBS-RL": (("A", 150_000), ("B", 250_000), ("C", 400_000)),
    "DYN-RISK-RL": (("A", 50_000), ("B", 75_000), ("C", 75_000)),
    "DYN-OBS-RL": (("A", 50_000), ("B", 75_000), ("C", 75_000)),
}

#: Episode-end info entries recorded by the training monitor.
MONITOR_INFO_KEYWORDS = (
    "termination", "is_success", "mission_acceptable", "urban_risk", "urban_mean_risk", "urban_max_risk",
    "urban_cumulative_risk", "urban_road_time", "urban_sidewalk_time", "urban_building_time", "urban_green_time",
    "urban_canopy_time", "urban_sensitive_time", "urban_building_collision_steps",
    "urban_building_clearance_violation_steps", "urban_canopy_collision_steps",
    "urban_canopy_clearance_violation_steps", "urban_sensitive_low_altitude_violation_steps",
    "urban_hard_violation_steps", "urban_action_shielded", "urban_geom_shielded", "urban_shielded_any",
    "urban_lidar_min_norm", "urban_signed_clearance_proxy_m", "cargo_health", "dist_xy",
)

_RL_HINT = 'training needs the "rl" extra: pip install "ruscuav-benchmark[rl]"'


def _sb3():
    try:
        import stable_baselines3
        from stable_baselines3.common import vec_env
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{_RL_HINT} (missing {exc.name})", name=exc.name) from exc
    return stable_baselines3, vec_env


class _EnvFactory:
    """Picklable environment constructor for the vectorized training environment."""

    def __init__(self, world: UrbanWorldLayers, config: Any, seed: int) -> None:
        self.world, self.config, self.seed = world, config, int(seed)

    def __call__(self):
        from ..env import RuscuavEnv

        env = RuscuavEnv(self.world, self.config)
        env.reset(seed=self.seed)
        return env


def make_training_env(world: UrbanWorldLayers, method: str, stage: str, *, mode: str = "fast_train", n_envs: int = 4,
                      base_seed: int = 1001, vec_env: str = "subproc", start_method: Optional[str] = None,
                      dynamic_risk: Any = None, env_overrides: Optional[Dict[str, Any]] = None):
    """The vectorized training environment of one curriculum stage: ``n_envs`` copies
    of the method preset's environment, environment k reset with seed ``base_seed + k``,
    in a ``VecMonitor``."""
    from ..env import make_config

    _, sb3_vec = _sb3()
    overrides = dict(env_overrides or {})
    if dynamic_risk is not None:
        overrides["dynamic_risk"] = dynamic_risk
    config = make_config(method, stage=stage, mode=mode, **overrides)
    factories = [_EnvFactory(world, config, base_seed + k) for k in range(int(n_envs))]
    if vec_env == "subproc" and int(n_envs) > 1:
        venv = sb3_vec.SubprocVecEnv(factories, start_method=start_method)
    elif vec_env in ("subproc", "dummy"):
        venv = sb3_vec.DummyVecEnv(factories)
    else:
        raise ValueError(f"vec_env={vec_env!r}; expected 'subproc' or 'dummy'")
    return sb3_vec.VecMonitor(venv, info_keywords=MONITOR_INFO_KEYWORDS)


def train_ppo(
    world: UrbanWorldLayers,
    method: str,
    output_dir: Union[str, Path],
    *,
    seed: int = 1001,
    schedule: Optional[Sequence[Tuple[str, int]]] = None,
    n_envs: int = 4,
    mode: str = "fast_train",
    vec_env: str = "subproc",
    start_method: Optional[str] = None,
    device: str = "cpu",
    dynamic_risk: Any = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    ppo_overrides: Optional[Dict[str, Any]] = None,
    verbose: int = 0,
) -> Dict[str, Path]:
    """Train ``method`` (a preset of ``ruscuav_benchmark.env.METHOD_PRESETS``) on
    ``world`` through ``schedule`` (default: :data:`TRAINING_SCHEDULES`).

    The first stage creates the PPO model with ``seed`` and :data:`PPO_SETTINGS`
    (``ppo_overrides`` change them); every later stage loads the previous stage's model
    into a new environment. Stage k's environments are seeded from ``seed + 100 k``, as
    in the archived runs. The model after each stage is saved as
    ``<output_dir>/<method>_seed<seed>_stage<k>_<stage>.zip``; the paths are returned
    by stage name (``"final"`` is the last).
    """
    sb3, _ = _sb3()
    schedule = tuple(TRAINING_SCHEDULES[method] if schedule is None else schedule)
    if not schedule:
        raise ValueError("schedule has no stages")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = dict(PPO_SETTINGS)
    settings["n_steps"] = 512 if int(n_envs) >= 4 else 1024
    settings.update(ppo_overrides or {})
    policy = settings.pop("policy")

    paths: Dict[str, Path] = {}
    model = None
    previous = ""
    for k, (stage, steps) in enumerate(schedule):
        venv = make_training_env(world, method, stage, mode=mode, n_envs=n_envs, base_seed=int(seed) + 100 * k,
                                 vec_env=vec_env, start_method=start_method, dynamic_risk=dynamic_risk,
                                 env_overrides=env_overrides)
        try:
            if model is None:
                model = sb3.PPO(policy, venv, seed=int(seed), device=device, verbose=verbose, **settings)
            else:
                model = sb3.PPO.load(str(paths[previous]), env=venv, device=device)
            model.learn(total_timesteps=int(steps), reset_num_timesteps=bool(model.num_timesteps == 0))
            path = out_dir / f"{method}_seed{int(seed)}_stage{k}_{stage}.zip"
            model.save(str(path))
        finally:
            venv.close()
        previous = f"{k}_{stage}"
        paths[previous] = path
    paths["final"] = path
    return paths
