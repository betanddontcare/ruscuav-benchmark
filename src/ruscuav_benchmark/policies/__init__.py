"""Learned policies: the archived PPO checkpoints, their evaluation and training.

Evaluate a Stable-Baselines3 checkpoint, an object with ``predict``, or a function of
the observation in the RUSCUAV environment, with the per-episode rows and summaries
of the archived evaluations::

    from ruscuav_benchmark import UrbanWorldLayers
    from ruscuav_benchmark.policies import archived_checkpoint, evaluate_policy

    world = UrbanWorldLayers.load_npz("worlds/synthetic/AOI_SYNTH_000/urban_world_layers.npz")
    ckpt = archived_checkpoint("RISK+GREEN-RL/4001", archive="/data/RUSCUAV-v2.0.0")  # doctest: +SKIP
    result = evaluate_policy(world, ckpt, "RISK+GREEN-RL", n_episodes=20)            # doctest: +SKIP
    print(result.summary["mission_acceptability_rate"])                               # doctest: +SKIP

Loading checkpoints and training need the ``rl`` extra (Stable-Baselines3, PyTorch);
the environment needs the ``env`` extra. See docs/POLICIES.md.
"""

from .checkpoints import (
    ARCHIVED_CHECKPOINTS,
    ARCHIVED_EVALUATIONS,
    ArchivedCheckpoint,
    ArchivedEvaluation,
    archived_checkpoint,
    checkpoint_env_overrides,
    sha256_file,
)
from .evaluation import (
    ALGORITHMS,
    PolicyEvaluation,
    default_label,
    default_row_format,
    evaluate_archived,
    evaluate_policy,
    load_policy,
    summarize_policy_rows,
)
from .rollout import ROW_FORMATS, fly, policy_action, run_policy_episode
from .simple import ConstantPolicy, TargetSeekingPolicy
from .training import PPO_SETTINGS, TRAINING_SCHEDULES, make_training_env, train_ppo

__all__ = [
    "ARCHIVED_CHECKPOINTS", "ARCHIVED_EVALUATIONS", "ArchivedCheckpoint", "ArchivedEvaluation", "archived_checkpoint",
    "checkpoint_env_overrides", "sha256_file", "ALGORITHMS", "PolicyEvaluation", "evaluate_policy", "evaluate_archived",
    "load_policy", "summarize_policy_rows",
    "default_row_format", "default_label", "ROW_FORMATS", "run_policy_episode", "fly", "policy_action",
    "PPO_SETTINGS", "TRAINING_SCHEDULES", "make_training_env", "train_ppo", "TargetSeekingPolicy", "ConstantPolicy",
]
