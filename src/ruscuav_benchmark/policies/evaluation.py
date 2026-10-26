"""Evaluate a policy on a world: one episode per seed or mission, a per-episode CSV
and a summary, in the formats of the archived evaluations.

Each episode runs in a fresh environment of the method preset in the chosen regime.
The policy is a Stable-Baselines3 checkpoint (needs the ``rl`` extra), any object
with a ``predict(obs, deterministic=...)`` method, or a callable ``policy(obs)``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..posthoc import DT_S, summarize_posthoc, with_derived_metrics
from ..scenarios import Episode, episode_list
from ..tables import Row, is_true, jsonable, mean_of, median_of, std_of, termination_counts, write_rows_csv
from ..world import UrbanWorldLayers, canonical_aoi_id
from .checkpoints import ARCHIVED_EVALUATIONS, archived_checkpoint, checkpoint_env_overrides, sha256_file
from .rollout import ROW_FORMATS, run_policy_episode

__all__ = [
    "PolicyEvaluation", "load_policy", "evaluate_policy", "evaluate_archived", "default_row_format",
    "default_label", "summarize_policy_rows",
]

_RL_HINT = 'loading a PPO checkpoint needs the "rl" extra: pip install "ruscuav-benchmark[rl]"'


@dataclass
class PolicyEvaluation:
    """Result of :func:`evaluate_policy`."""

    method: str
    row_format: str
    rows: List[Row]
    summary: Dict[str, Any]
    csv_path: Optional[Path] = None
    summary_path: Optional[Path] = None
    paths: Dict[int, np.ndarray] = field(default_factory=dict)


def load_policy(path: Union[str, Path], device: str = "cpu") -> Any:
    """A Stable-Baselines3 PPO checkpoint (``.zip``), loaded for inference."""
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(f"{_RL_HINT} (missing {exc.name})", name=exc.name) from exc
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"policy checkpoint not found: {path}")
    return PPO.load(str(path), device=device)


def default_row_format(method: str) -> str:
    """``"dynamic"`` for the dynamic-risk presets, otherwise ``"posthoc"``."""
    return "dynamic" if str(method).startswith("DYN-") else "posthoc"


def default_label(method: str, row_format: str) -> str:
    """The ``method`` (``policy_kind``) column the archived evaluations wrote; the
    method name where they wrote none."""
    if row_format == "dynamic":
        return {"DYN-OBS-RL": "RL_dynamic_obs_reward_only", "DYN-RISK-RL": "RL_reward_only_dynamic_eval"}.get(
            method, method)
    if row_format == "transfer":
        return {"GEO-RL": "geometry_only", "RISK+GREEN-RL": "reward_only"}.get(method, method)
    return method


# ---------------------------------------------------------------------- summaries
def _summary_dynamic(rows: List[Row]) -> Dict[str, Any]:
    """``summarize_m11_dynamic_eval_df``."""
    if not rows:
        return {"n_episodes": 0}
    s: Dict[str, Any] = {
        "n_episodes": len(rows),
        "success_rate": mean_of(rows, "is_success"),
        "mission_acceptability_rate": mean_of(rows, "mission_acceptable"),
        "static_only_acceptability_rate": mean_of(rows, "mission_acceptable_static_only"),
        "mean_reward": mean_of(rows, "total_reward"),
        "std_reward": std_of(rows, "total_reward"),
        "mean_steps": mean_of(rows, "steps"),
        "mean_dist_xy": mean_of(rows, "dist_xy"),
        "median_dist_xy": median_of(rows, "dist_xy"),
        "mean_cargo_health": mean_of(rows, "cargo_health"),
        "mean_urban_risk": mean_of(rows, "urban_mean_risk"),
        "mean_urban_cumulative_risk": mean_of(rows, "urban_cumulative_risk"),
        "mean_dynamic_risk": mean_of(rows, "dynamic_risk_mean"),
        "mean_dynamic_risk_max": mean_of(rows, "dynamic_risk_max"),
        "mean_dynamic_cumulative_risk": mean_of(rows, "dynamic_cumulative_risk"),
        "dynamic_budget_ok_rate": mean_of(rows, "dynamic_budget_ok"),
        "mean_road_time": mean_of(rows, "road_time"),
        "mean_sidewalk_time": mean_of(rows, "sidewalk_time"),
        "mean_green_time": mean_of(rows, "green_time"),
        "mean_sensitive_time": mean_of(rows, "sensitive_time"),
        "mean_hard_violation_steps": mean_of(rows, "urban_hard_violation_steps"),
        "termination_counts": termination_counts(rows),
    }
    successes = [r for r in rows if is_true(r.get("is_success"))]
    s["acceptable_given_success_rate"] = mean_of(successes, "mission_acceptable") if successes else 0.0
    return s


def _summary_transfer(rows: List[Row]) -> Dict[str, Any]:
    """``summarize_m12c_real_policy_eval_df``."""
    if not rows:
        return {"n_episodes": 0}
    s: Dict[str, Any] = {
        "n_episodes": len(rows),
        "success_rate": mean_of(rows, "is_success"),
        "mission_acceptability_rate": mean_of(rows, "mission_acceptable"),
        "mean_reward": mean_of(rows, "total_reward"),
        "std_reward": std_of(rows, "total_reward"),
        "mean_steps": mean_of(rows, "steps"),
        "mean_dist_xy": mean_of(rows, "dist_xy"),
        "median_dist_xy": median_of(rows, "dist_xy"),
        "mean_cargo_health": mean_of(rows, "cargo_health"),
        "mean_urban_risk": mean_of(rows, "urban_mean_risk"),
        "mean_urban_max_risk": mean_of(rows, "urban_max_risk"),
        "mean_cumulative_risk": mean_of(rows, "urban_cumulative_risk"),
        "mean_road_time": mean_of(rows, "road_time"),
        "mean_sidewalk_time": mean_of(rows, "sidewalk_time"),
        "mean_building_time": mean_of(rows, "building_time"),
        "mean_green_time": mean_of(rows, "green_time"),
        "mean_canopy_time": mean_of(rows, "canopy_time"),
        "mean_sensitive_time": mean_of(rows, "sensitive_time"),
        "mean_hard_violation_steps": mean_of(rows, "urban_hard_violation_steps"),
        "termination_counts": termination_counts(rows),
    }
    successes = [r for r in rows if is_true(r.get("is_success"))]
    s["acceptable_given_success_rate"] = mean_of(successes, "mission_acceptable") if successes else 0.0
    return s


def summarize_policy_rows(rows: List[Row], row_format: str, dt_s: float = DT_S) -> Dict[str, Any]:
    """Summary of a policy evaluation with the statistics of the archived summaries
    of the row format (the posthoc format: rates and means over all, successful and
    acceptable episodes; ``dt_s`` is the step length of its derived rates)."""
    if row_format == "posthoc":
        return summarize_posthoc(rows, dynamic=False, dt_s=dt_s)
    if row_format == "dynamic":
        return _summary_dynamic(rows)
    if row_format == "transfer":
        return _summary_transfer(rows)
    raise ValueError(f"row_format={row_format!r}; expected one of {list(ROW_FORMATS)}")


def _episode_rows(rows: List[Row], row_format: str, dt_s: float = DT_S) -> List[Row]:
    """The rows as the archived evaluations wrote them (posthoc: with the derived metrics)."""
    return with_derived_metrics(rows, dt_s=dt_s) if row_format == "posthoc" else rows


# ---------------------------------------------------------------------- evaluation
#: EnvConfig fields set by the regime (or stage and mode), not by ``env_overrides``.
_REGIME_FIELDS = ("stage", "mode", "terminate_on_urban_constraint", "label")


def _env_config(method: str, regime: Optional[str], stage: Optional[str], mode: Optional[str], risk_profile: str,
                dynamic_risk: Any, env_overrides: Optional[Dict[str, Any]]):
    from .. import env as env_package

    clashes = sorted(set(env_overrides or {}) & set(_REGIME_FIELDS + ("risk_profile", "dynamic_risk")))
    if clashes:
        raise ValueError(f"env_overrides cannot set {clashes}: give regime (or stage and mode), risk_profile "
                         "and dynamic_risk instead")
    overrides: Dict[str, Any] = {"risk_profile": risk_profile, **(env_overrides or {})}
    if dynamic_risk is not None:
        overrides["dynamic_risk"] = dynamic_risk
    if stage is not None or mode is not None:
        if regime is not None:
            raise ValueError("give a regime or a stage/mode, not both")
        config = env_package.make_config(method, stage=None if stage is None else str(stage).upper(),
                                         mode=None if mode is None else str(mode).lower(), **overrides)
        regime_obj = env_package.EvaluationRegime(
            name=f"stage_{config.stage}_{config.mode}", stage=config.stage, mode=config.mode,
            terminate_on_soft_budgets=bool(config.resolved_terminate_on_urban_constraint))
    else:
        name = regime or "primary_no_terminal"
        if name not in env_package.REGIMES:
            raise ValueError(f"Unknown regime {name!r}; available: {list(env_package.REGIMES)}")
        config = env_package.make_config(method, regime=name, **overrides)
        regime_obj = env_package.REGIMES[name]
    return config, regime_obj


def _check_spaces(policy: Any, config: Any) -> None:
    space = getattr(policy, "observation_space", None)
    shape = getattr(space, "shape", None)
    if shape is not None and tuple(shape) != (int(config.observation_dim),):
        raise ValueError(f"the policy expects observations of shape {tuple(shape)}, the environment gives "
                         f"({config.observation_dim},): choose a method preset with the observation the policy "
                         "was trained with")


def _settings_digest(settings: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(jsonable(settings), sort_keys=True, default=str).encode()).hexdigest()[:8]


def evaluate_policy(
    world: UrbanWorldLayers,
    policy: Any,
    method: str,
    *,
    regime: Optional[str] = None,
    stage: Optional[str] = None,
    mode: Optional[str] = None,
    seeds: Optional[Iterable[int]] = None,
    missions: Optional[Sequence[Episode]] = None,
    n_episodes: Optional[int] = None,
    seed0: int = 5000,
    row_format: Optional[str] = None,
    label: Optional[str] = None,
    deterministic: bool = True,
    max_steps: int = 1000,
    risk_profile: str = "balanced",
    dynamic_risk: Any = None,
    env_overrides: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path, None] = None,
    prefix: Optional[str] = None,
    save_paths: bool = False,
    progress: Optional[Callable[[int, int, Row], None]] = None,
) -> PolicyEvaluation:
    """Fly ``policy`` in the environment of ``method`` (a preset of
    ``ruscuav_benchmark.env.METHOD_PRESETS``) on episodes of ``world``.

    ``policy`` is a path to a Stable-Baselines3 checkpoint, a model with ``predict``,
    or a callable ``obs -> action``; the method preset fixes the observation, so it
    must be the one the policy was trained with. Episodes are ``missions``
    (``(seed, reset options)``, e.g. from ``ruscuav_benchmark.scenarios.missions_from_csv``),
    ``seeds``, or by default ``n_episodes`` seeds from ``seed0`` (200). The regime is
    ``"primary_no_terminal"`` (default) or ``"secondary_operational_stopping"``, or
    give ``stage`` and ``mode``. ``env_overrides`` change ``EnvConfig`` fields (for
    example ``lidar``); ``dynamic_risk`` replaces the dynamic risk configuration.

    ``row_format`` is ``"posthoc"`` (default; ``"dynamic"`` for the dynamic presets,
    and only for them) or ``"transfer"`` (see :mod:`ruscuav_benchmark.policies.rollout`);
    ``label`` fills the method column. With ``deterministic=False`` the policy samples
    its actions; a Stable-Baselines3 model is then seeded with the episode seed, so
    runs repeat.

    With ``output_dir``, ``<prefix>_episodes.csv`` (rewritten after every episode),
    ``<prefix>_summary.json`` and ``<prefix>_config.json`` are written there; with
    ``save_paths`` also the flown paths, in ``<prefix>_paths/``. The default prefix is
    ``<method>_<world>_<regime>_<checksum>``, the checksum of the settings (policy file
    and its SHA-256, or the policy's class; environment, format, label), so that runs
    of other policies or settings do not overwrite each other.
    """
    from ..env import RuscuavEnv  # needs the "env" extra

    row_format = default_row_format(method) if row_format is None else row_format
    if row_format not in ROW_FORMATS:
        raise ValueError(f"row_format={row_format!r}; expected one of {list(ROW_FORMATS)}")
    label = default_label(method, row_format) if label is None else str(label)
    episodes = episode_list(seeds, missions, n_episodes, seed0)
    config, regime_obj = _env_config(method, regime, stage, mode, risk_profile, dynamic_risk, env_overrides)
    if row_format == "dynamic" and config.dynamic_risk is None:
        raise ValueError(f"the dynamic format records dynamic risk; {method} has none (use a DYN- preset or give "
                         "dynamic_risk)")
    dt_s = float(config.dynamics.dt)

    policy_info: Dict[str, Any] = {"kind": type(policy).__name__}
    if isinstance(policy, (str, Path)):
        policy_path = Path(policy)
        if not policy_path.is_file():
            raise FileNotFoundError(f"policy checkpoint not found: {policy_path}")
        policy_info = {"kind": "stable_baselines3_checkpoint", "path": str(policy_path),
                       "sha256": sha256_file(policy_path)}
        policy = load_policy(policy_path)
    _check_spaces(policy, config)

    world_id = canonical_aoi_id(world.aoi_id)
    settings = {
        "method": method,
        "row_format": row_format,
        "label": label,
        "world_id": world_id,
        "regime": asdict(regime_obj),
        "deterministic": bool(deterministic),
        "max_steps": int(max_steps),
        "policy": policy_info,
        "environment": jsonable(config.to_dict()),
    }
    if prefix is None:
        prefix = f"{method}_{world_id}_{regime_obj.name}_{_settings_digest(settings)}"
    out_dir = None if output_dir is None else Path(output_dir)
    csv_path = summary_path = paths_dir = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{prefix}_episodes.csv"
        summary_path = out_dir / f"{prefix}_summary.json"
        (out_dir / f"{prefix}_config.json").write_text(json.dumps(jsonable(settings), indent=2) + "\n",
                                                       encoding="utf-8")
        if save_paths:
            paths_dir = out_dir / f"{prefix}_paths"
            paths_dir.mkdir(exist_ok=True)

    rows: List[Row] = []
    paths: Dict[int, np.ndarray] = {}
    reseed = getattr(policy, "set_random_seed", None) if not deterministic else None
    for index, (seed, options) in enumerate(episodes):
        env = RuscuavEnv(world, config)
        if reseed is not None and seed is not None:
            reseed(int(seed))
        try:
            row, path = run_policy_episode(env, policy, seed, row_format=row_format, label=label, regime=regime_obj,
                                           options=options, deterministic=deterministic, max_steps=max_steps)
        finally:
            env.close()
        row["episode"] = index
        rows.append(row)
        paths[index] = path
        if paths_dir is not None:
            np.save(paths_dir / f"episode_{index:05d}_seed_{seed}_path.npy", path)
        if csv_path is not None:
            write_rows_csv(csv_path, _episode_rows(rows, row_format, dt_s))
        if progress is not None:
            progress(index, len(episodes), row)

    summary = summarize_policy_rows(rows, row_format, dt_s)
    summary.update({
        **settings,
        "legacy_world_id": world.aoi_id,
        "seeds": [row["seed"] for row in rows],
        "missions_given": missions is not None,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    if summary_path is not None:
        summary["csv_path"] = str(csv_path)
        summary_path.write_text(json.dumps(jsonable(summary), indent=2) + "\n", encoding="utf-8")
    return PolicyEvaluation(method, row_format, _episode_rows(rows, row_format, dt_s), summary, csv_path,
                            summary_path, paths)


#: The canonical world inside the extracted archive (the AOI_SYNTH_000 layers under
#: their v2.0.0 name).
ARCHIVE_CANONICAL_WORLD = "worlds/synthetic/AOI_FAKE_001_urban_world_layers.npz"


def evaluate_archived(name: str, archive: Union[str, Path], world: Optional[UrbanWorldLayers] = None,
                      check_world: bool = True, **kwargs: Any) -> PolicyEvaluation:
    """Re-run the archived policy evaluation ``name`` (see
    :data:`~ruscuav_benchmark.policies.checkpoints.ARCHIVED_EVALUATIONS`) with its
    checkpoint from the extracted archive ``archive``. The canonical world is read
    from the archive; the Vienna evaluations need ``world``, the processed
    footway-only world, which is not distributed; with ``check_world`` the world's
    layers checksum must be the archived one. Keyword arguments go to
    :func:`evaluate_policy`, e.g. ``n_episodes``, ``seed0`` or ``output_dir``; the
    settings of the evaluation itself are fixed."""
    if name not in ARCHIVED_EVALUATIONS:
        raise KeyError(f"unknown archived evaluation {name!r}; available: {sorted(ARCHIVED_EVALUATIONS)}")
    entry = ARCHIVED_EVALUATIONS[name]
    fixed = sorted(set(kwargs) & {"regime", "stage", "mode", "row_format", "label", "risk_profile", "dynamic_risk",
                                  "env_overrides", "deterministic", "max_steps", "seeds", "missions"})
    if fixed:
        raise ValueError(f"{fixed} are fixed by the archived evaluation; use evaluate_policy to change them")
    if world is None:
        if entry.world != "canonical":
            raise ValueError(f"{name} ran in the {entry.world} world, which is not in the archive; pass world=")
        world = UrbanWorldLayers.load_npz(Path(archive) / ARCHIVE_CANONICAL_WORLD)
    if check_world and world.layers_sha256() != entry.world_layers_sha256:
        raise ValueError(f"{name} ran in a world with layers checksum {entry.world_layers_sha256}; the given world "
                         f"has {world.layers_sha256()} (check_world=False flies it anyway)")
    checkpoint = archived_checkpoint(entry.checkpoint, archive)
    kwargs.setdefault("n_episodes", entry.n_episodes)
    kwargs.setdefault("seed0", entry.seed0)
    kwargs.setdefault("prefix", name.replace("/", "_"))
    return evaluate_policy(world, checkpoint, entry.method, regime="primary_no_terminal", row_format=entry.row_format,
                           label=entry.label, env_overrides=checkpoint_env_overrides(entry.checkpoint), **kwargs)
