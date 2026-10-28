"""The trained policies of the RUSCUAV v2.0.0 archive and the evaluations that used them.

The checkpoints are Stable-Baselines3 PPO archives in the Zenodo release (not in
this repository). :data:`ARCHIVED_CHECKPOINTS` lists them by key, e.g.
``"RISK+GREEN-RL/4001"`` (method preset / training seed) or
``"VIENNA/RISK+GREEN-RL/short_ft"``, with their path in the extracted archive and
their SHA-256; :func:`archived_checkpoint` finds and checks one.
:data:`ARCHIVED_EVALUATIONS` lists the primary (no-terminal) policy evaluations of
the paper: which checkpoint, environment preset, world and row format each used,
and where its per-episode CSV is in the archive.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

__all__ = [
    "ArchivedCheckpoint", "ARCHIVED_CHECKPOINTS", "archived_checkpoint", "ArchivedEvaluation",
    "ARCHIVED_EVALUATIONS", "WORLD_LAYERS_SHA256", "checkpoint_env_overrides", "sha256_file",
]


@dataclass(frozen=True)
class ArchivedCheckpoint:
    """A PPO checkpoint of the archive."""

    key: str
    #: Method preset (ruscuav_benchmark.env.METHOD_PRESETS) of the environment the
    #: policy was trained in; the policy expects that preset's observation.
    method: str
    train_seed: Optional[int]
    #: Name of the training run in the notebook.
    training_run: str
    #: Path inside the extracted v2.0.0 archive.
    path: str
    sha256: str
    size_bytes: int
    note: str = ""
    #: False for the ablation trained without the urban LiDAR (all readings 1).
    urban_lidar: bool = True


def _ckpt(key, method, train_seed, training_run, path, sha256, size_bytes, note) -> Tuple[str, ArchivedCheckpoint]:
    return key, ArchivedCheckpoint(key, method, train_seed, training_run, path, sha256, size_bytes, note,
                                   urban_lidar=key != "RISK+OBS-RL/no_urban_lidar")


#: Checkpoints of the archive (models/WEIGHTS_REQUIRED.csv and the two Vienna
#: fine-tuning checkpoints of models/vienna_finetuning/).
ARCHIVED_CHECKPOINTS: Dict[str, ArchivedCheckpoint] = dict([
    _ckpt('GEO-RL/1001', 'GEO-RL', 1001, 'M7_geometry_only',
          'experiments/paperA_multiseed_rl_v1_M7_geometry_only_trainseed1001/models/ppo_ruscuav_m4_after_stage_GEO.zip',
          '9c1de48ad653e873ca0abfa643a16fef32342d36fbc79d7d1c0083594413fedd', 2154111,
          'training seed 1001 (M7_geometry_only)'),
    _ckpt('GEO-RL/2001', 'GEO-RL', 2001, 'M7_geometry_only',
          'experiments/paperA_multiseed_rl_v1_M7_geometry_only_trainseed2001/models/ppo_ruscuav_m4_after_stage_GEO.zip',
          '9fd22891423d38ec2544f9b1481098d927da64d939fd1bb16b65665ac2f4e7c0', 2154111,
          'training seed 2001 (M7_geometry_only)'),
    _ckpt('GEO-RL/3001', 'GEO-RL', 3001, 'M7_geometry_only',
          'experiments/paperA_multiseed_rl_v1_M7_geometry_only_trainseed3001/models/ppo_ruscuav_m4_after_stage_GEO.zip',
          '2be1c3fdc4fc85bbbb7a0dee3381c345007c96f684866a6fad110a1a9dab7a87', 2154115,
          'training seed 3001 (M7_geometry_only)'),
    _ckpt('GEO-RL/4001', 'GEO-RL', 4001, 'M7_geometry_only',
          'experiments/paperA_multiseed_rl_v1_M7_geometry_only_trainseed4001/models/ppo_ruscuav_m4_after_stage_GEO.zip',
          'e721729c47484b5508a2a9c19c3797ff0414eccde24e3d0245cc5b15ad8b766e', 2154115,
          'training seed 4001 (M7_geometry_only)'),
    _ckpt('GEO-RL/5001', 'GEO-RL', 5001, 'M7_geometry_only',
          'experiments/paperA_multiseed_rl_v1_M7_geometry_only_trainseed5001/models/ppo_ruscuav_m4_after_stage_GEO.zip',
          '82ccc39b569b61a0ec5e07175bbc0afdd044845e6cbbfad8033fc0f01e279a90', 2154111,
          'training seed 5001 (M7_geometry_only)'),
    _ckpt('GEO-RL/single', 'GEO-RL', None, 'M7_geometry_only',
          'experiments/m7_geometry_only_baseline/models/ppo_ruscuav_m4_after_stage_GEO.zip',
          '1453127e0f8148539e4a2bf92685bcf6830ce0509a7f3d277f0c7473c23d425e', 2154136,
          'single checkpoint (M7_geometry_only)'),
    _ckpt('OBS-RL/1001', 'OBS-RL', 1001, 'M8_obs_only',
          'experiments/paperA_multiseed_rl_v1_M8_obs_only_trainseed1001/models/'
          'ppo_ruscuav_m4_after_stage_M8_obs_only_C.zip',
          'aa27297bb3911144321e75be7241f47d6859c72946bb192fc44e906c8e7b9d50', 2278182,
          'training seed 1001 (M8_obs_only)'),
    _ckpt('OBS-RL/2001', 'OBS-RL', 2001, 'M8_obs_only',
          'experiments/paperA_multiseed_rl_v1_M8_obs_only_trainseed2001/models/'
          'ppo_ruscuav_m4_after_stage_M8_obs_only_C.zip',
          '8135293c8b72e3a271749d4426d10a3b705807f7e63edbce39beed9925d75ea0', 2278458,
          'training seed 2001 (M8_obs_only)'),
    _ckpt('OBS-RL/3001', 'OBS-RL', 3001, 'M8_obs_only',
          'experiments/paperA_multiseed_rl_v1_M8_obs_only_trainseed3001/models/'
          'ppo_ruscuav_m4_after_stage_M8_obs_only_C.zip',
          '89ac99b8361030c063adf99e51d9bee7129b16d2136b808ec8e4d5a69c53dfd8', 2278334,
          'training seed 3001 (M8_obs_only)'),
    _ckpt('OBS-RL/4001', 'OBS-RL', 4001, 'M8_obs_only',
          'experiments/paperA_multiseed_rl_v1_M8_obs_only_trainseed4001/models/'
          'ppo_ruscuav_m4_after_stage_M8_obs_only_C.zip',
          '438db3d428defaaaecd06b93d5961a5d14e9a95f4674adb091c0a2a13a020a53', 2278526,
          'training seed 4001 (M8_obs_only)'),
    _ckpt('OBS-RL/5001', 'OBS-RL', 5001, 'M8_obs_only',
          'experiments/paperA_multiseed_rl_v1_M8_obs_only_trainseed5001/models/'
          'ppo_ruscuav_m4_after_stage_M8_obs_only_C.zip',
          '1d8bea62131c753b6b0cfef4c48af0a949d5b5e54a3cbb068deab7d12d314d06', 2278215,
          'training seed 5001 (M8_obs_only)'),
    _ckpt('OBS-RL/single', 'OBS-RL', None, 'M8_obs_only',
          'experiments/m8_ablation_obs_only/models/ppo_ruscuav_m4_after_stage_M8_obs_only_C.zip',
          '2d0168c6f784b5852f6ce203fff7a4bb90bb249255396362d7fa9d7f0e46628b', 2278292,
          'single checkpoint (M8_obs_only)'),
    _ckpt('RISK+GREEN-RL/1001', 'RISK+GREEN-RL', 1001, 'M8_reward_only',
          'experiments/paperA_multiseed_rl_v1_M8_reward_only_trainseed1001/models/'
          'ppo_ruscuav_m4_after_stage_M8_reward_only_C.zip',
          '263e844844616c936f5f6cf343cd3d144c69b6c8827dc01111e8dff7bcb4e920', 2154147,
          'training seed 1001 (M8_reward_only)'),
    _ckpt('RISK+GREEN-RL/2001', 'RISK+GREEN-RL', 2001, 'M8_reward_only',
          'experiments/paperA_multiseed_rl_v1_M8_reward_only_trainseed2001/models/'
          'ppo_ruscuav_m4_after_stage_M8_reward_only_C.zip',
          'b087a6a17a871ff279f50cc5edc1a9f530934bb76bdab2c03277ed35997b43ad', 2154331,
          'training seed 2001 (M8_reward_only)'),
    _ckpt('RISK+GREEN-RL/3001', 'RISK+GREEN-RL', 3001, 'M8_reward_only',
          'experiments/paperA_multiseed_rl_v1_M8_reward_only_trainseed3001/models/'
          'ppo_ruscuav_m4_after_stage_M8_reward_only_C.zip',
          '8dfd035e841828a99385a268a69e5abdada7fae1ffca159bd7a19f61c3e15ef5', 2154179,
          'training seed 3001 (M8_reward_only)'),
    _ckpt('RISK+GREEN-RL/4001', 'RISK+GREEN-RL', 4001, 'M8_reward_only',
          'experiments/paperA_multiseed_rl_v1_M8_reward_only_trainseed4001/models/'
          'ppo_ruscuav_m4_after_stage_M8_reward_only_C.zip',
          '1a9662bc1fa25000ed8d3b325861b66e4e827f17c89aa0537ce86174f01b8841', 2154243,
          'training seed 4001 (M8_reward_only)'),
    _ckpt('RISK+GREEN-RL/5001', 'RISK+GREEN-RL', 5001, 'M8_reward_only',
          'experiments/paperA_multiseed_rl_v1_M8_reward_only_trainseed5001/models/'
          'ppo_ruscuav_m4_after_stage_M8_reward_only_C.zip',
          '530ee88ab9b6bae310ba92bbf80dd92d5fd59f13ecf74cf8a47ef297b5014775', 2154299,
          'training seed 5001 (M8_reward_only)'),
    _ckpt('RISK+GREEN-RL/single', 'RISK+GREEN-RL', None, 'M8_reward_only',
          'experiments/m8_ablation_reward_only/models/ppo_ruscuav_m4_after_stage_M8_reward_only_C.zip',
          '0a01a4e90e2e687b69dfcd1aefc03b2083818f51d9c03a277ab81182c4c03d08', 2154194,
          'single checkpoint (M8_reward_only)'),
    _ckpt('RISK-RL/1001', 'RISK-RL', 1001, 'M8_risk_only',
          'experiments/paperA_reward_factor_risk_only_seed1001/models/ppo_ruscuav_m4_after_stage_M8_risk_only_C.zip',
          '8aec139329fc9432a40724fb3f90d8b615377401dbc8689ed7122be2a79d6ef2', 2154312,
          'training seed 1001 (M8_risk_only)'),
    _ckpt('RISK-RL/2001', 'RISK-RL', 2001, 'M8_risk_only',
          'experiments/paperA_reward_factor_risk_only_seed2001/models/ppo_ruscuav_m4_after_stage_M8_risk_only_C.zip',
          '06d350ef34cb6bab1d29b5bd4d4ba57a966d547db95887978b83f3b48df479e6', 2154220,
          'training seed 2001 (M8_risk_only)'),
    _ckpt('RISK-RL/3001', 'RISK-RL', 3001, 'M8_risk_only',
          'experiments/paperA_reward_factor_risk_only_seed3001/models/ppo_ruscuav_m4_after_stage_M8_risk_only_C.zip',
          '14223ac53ec3be8ed35d954b83130ce52531bc0a5ab24a38870976e1e38d7754', 2154224,
          'training seed 3001 (M8_risk_only)'),
    _ckpt('RISK-RL/4001', 'RISK-RL', 4001, 'M8_risk_only',
          'experiments/paperA_reward_factor_risk_only_seed4001/models/ppo_ruscuav_m4_after_stage_M8_risk_only_C.zip',
          '9c4b6a8678a65e7d26d6818a902702947cc25c0bdcc94ebfe540dc4b7c8b5e8a', 2154252,
          'training seed 4001 (M8_risk_only)'),
    _ckpt('RISK-RL/5001', 'RISK-RL', 5001, 'M8_risk_only',
          'experiments/paperA_reward_factor_risk_only_seed5001/models/ppo_ruscuav_m4_after_stage_M8_risk_only_C.zip',
          'c6db73239a5ec080254a09a3b24c38d144c4e38f07584c5a1593a677fd165152', 2154216,
          'training seed 5001 (M8_risk_only)'),
    _ckpt('GREEN-RL/1001', 'GREEN-RL', 1001, 'M8_green_only',
          'experiments/paperA_reward_factor_green_only_seed1001/models/ppo_ruscuav_m4_after_stage_M8_green_only_C.zip',
          'faaf12ddb15e3035d638ef0e8cb119d49d7fb05df2122b877b2920d21a4bc78f', 2154248,
          'training seed 1001 (M8_green_only)'),
    _ckpt('GREEN-RL/2001', 'GREEN-RL', 2001, 'M8_green_only',
          'experiments/paperA_reward_factor_green_only_seed2001/models/ppo_ruscuav_m4_after_stage_M8_green_only_C.zip',
          '7d946b83fa937806b8e2e07c2504338d6b40842d83c4f8b059605ec417b8663d', 2154248,
          'training seed 2001 (M8_green_only)'),
    _ckpt('GREEN-RL/3001', 'GREEN-RL', 3001, 'M8_green_only',
          'experiments/paperA_reward_factor_green_only_seed3001/models/ppo_ruscuav_m4_after_stage_M8_green_only_C.zip',
          '89f25d11979dadd4cb43948ba9d138f662e88fc15eab7136f3c0d91af58e2c5c', 2154224,
          'training seed 3001 (M8_green_only)'),
    _ckpt('GREEN-RL/4001', 'GREEN-RL', 4001, 'M8_green_only',
          'experiments/paperA_reward_factor_green_only_seed4001/models/ppo_ruscuav_m4_after_stage_M8_green_only_C.zip',
          '59f2e72d7b637acbfe8d65806f2b2ffd74b66750a89b1762ee19f35b35dc5934', 2154160,
          'training seed 4001 (M8_green_only)'),
    _ckpt('GREEN-RL/5001', 'GREEN-RL', 5001, 'M8_green_only',
          'experiments/paperA_reward_factor_green_only_seed5001/models/ppo_ruscuav_m4_after_stage_M8_green_only_C.zip',
          'd479da24157c6d76a1742f445dc3510bfafd67bb856aa392db730bcc8df049e1', 2154220,
          'training seed 5001 (M8_green_only)'),
    _ckpt('RISK+OBS-RL/no_urban_lidar', 'RISK+OBS-RL', None, 'm8_ablation_no_urban_lidar',
          'experiments/m8_ablation_no_urban_lidar/models/ppo_ruscuav_m4_after_stage_M8_no_urban_lidar_C.zip',
          'a7880f8cc6d6336e4d911df8cf0fff5b773388830f4ca32854c5283549a8e115', 2278385,
          'safety-stack ablation without the urban LiDAR (M8)'),
    _ckpt('RISK+OBS-RL/single', 'RISK+OBS-RL', None, 'M6_full_semantic_RL',
          'experiments/m5_m4_ppo_debug/models/ppo_ruscuav_m4_after_stage_C.zip',
          'a03ea42d74bd9018bdb5f7f5e7b23ccc440b8a4428be2cc4d44caab62d581eac', 2278143,
          'single checkpoint (M6_full_semantic_RL)'),
    _ckpt('DYN-RISK-RL/single', 'DYN-RISK-RL', None, 'M11B_dynamic_reward_only',
          'experiments/m11b_dynamic_trained_reward_only/models/'
          'ppo_ruscuav_m4_after_stage_M11B_dynamic_reward_only_C.zip',
          '5615b0adc43c10c73f37319a4621334d127d390a82cf3e73cd6558d91fbd95e3', 2161614,
          'single checkpoint (M11B_dynamic_reward_only)'),
    _ckpt('DYN-OBS-RL/1001', 'DYN-OBS-RL', 1001, 'M11C_dynamic_obs_reward_only',
          'experiments/paperA_multiseed_rl_v1_M11C_dynamic_obs_reward_only_trainseed1001/models/'
          'ppo_ruscuav_m4_after_stage_M11C_dynamic_obs_reward_only_C.zip',
          'fece42dc49c2ff7ce9b22af4143b3653dd343cafdfe646a2f71e61389b92fb56', 2198707,
          'training seed 1001 (M11C_dynamic_obs_reward_only)'),
    _ckpt('DYN-OBS-RL/2001', 'DYN-OBS-RL', 2001, 'M11C_dynamic_obs_reward_only',
          'experiments/paperA_multiseed_rl_v1_M11C_dynamic_obs_reward_only_trainseed2001/models/'
          'ppo_ruscuav_m4_after_stage_M11C_dynamic_obs_reward_only_C.zip',
          'f0785eecc9d950d317d40b063cd7c31271dbbd7e343777dcadc3d48ac2e7f56c', 2198767,
          'training seed 2001 (M11C_dynamic_obs_reward_only)'),
    _ckpt('DYN-OBS-RL/3001', 'DYN-OBS-RL', 3001, 'M11C_dynamic_obs_reward_only',
          'experiments/paperA_multiseed_rl_v1_M11C_dynamic_obs_reward_only_trainseed3001/models/'
          'ppo_ruscuav_m4_after_stage_M11C_dynamic_obs_reward_only_C.zip',
          '385d4e539a7203b90c316f6952f08c564df58c55e37ae775ecf519b52bf10f6f', 2199475,
          'training seed 3001 (M11C_dynamic_obs_reward_only)'),
    _ckpt('DYN-OBS-RL/4001', 'DYN-OBS-RL', 4001, 'M11C_dynamic_obs_reward_only',
          'experiments/paperA_multiseed_rl_v1_M11C_dynamic_obs_reward_only_trainseed4001/models/'
          'ppo_ruscuav_m4_after_stage_M11C_dynamic_obs_reward_only_C.zip',
          'cf2ec9cacd79bc5d1c1a908656f8ced174fb5d1553cdd5fcbfe73084240235d5', 2198711,
          'training seed 4001 (M11C_dynamic_obs_reward_only)'),
    _ckpt('DYN-OBS-RL/5001', 'DYN-OBS-RL', 5001, 'M11C_dynamic_obs_reward_only',
          'experiments/paperA_multiseed_rl_v1_M11C_dynamic_obs_reward_only_trainseed5001/models/'
          'ppo_ruscuav_m4_after_stage_M11C_dynamic_obs_reward_only_C.zip',
          'b449ea2b8cb68a4adbdb30663cd1dc8f7cff4ae048ce70438021d4f28753dcc2', 2198799,
          'training seed 5001 (M11C_dynamic_obs_reward_only)'),
    _ckpt('DYN-OBS-RL/single', 'DYN-OBS-RL', None, 'M11C_dynamic_obs_reward_only',
          'experiments/m11c_dynamic_obs_reward_only/models/'
          'ppo_ruscuav_m4_after_stage_M11C_dynamic_obs_reward_only_C.zip',
          '625b3526a8379f5c2cf72dc846c0843e0ff85d86eafd6c142cc7521ba5f121a2', 2198818,
          'single checkpoint (M11C_dynamic_obs_reward_only)'),
    _ckpt('VIENNA/GEO-RL/final', 'GEO-RL', None, 'Vienna_geometry_only',
          'experiments/m12C_real_vienna_footway_learning/models/'
          'ppo_ruscuav_m4_after_stage_M12C_vienna_geometry_only_C.zip',
          '08ff7d8d97f8277ce02a07cfcc515452e1e904c8fdf303b735e054aead1ead7e', 2154702,
          'GEO-RL fine-tuned in the Vienna footway-only world (M12C)'),
    _ckpt('VIENNA/RISK+GREEN-RL/extended_ft', 'RISK+GREEN-RL', None, 'M12C_vienna_reward_only',
          'models/vienna_finetuning/stage_M12C_vienna_reward_only_C_ckpt_completed_000251904_global_000581632.zip',
          'c0949b21a44efabced360aac56707d778642065cb306d8ac6acbd56e3f3b9417', 2154375,
          'extended Vienna fine-tuning of RISK+GREEN-RL/1001 (251,904 stage C steps)'),
    _ckpt('VIENNA/RISK+GREEN-RL/final', 'RISK+GREEN-RL', None, 'Vienna_reward_only_shortFT',
          'experiments/m12C_real_vienna_footway_learning/models/'
          'ppo_ruscuav_m4_after_stage_M12C_vienna_reward_only_C.zip',
          'c0949b21a44efabced360aac56707d778642065cb306d8ac6acbd56e3f3b9417', 2154375,
          'final RISK+GREEN-RL fine-tuning checkpoint in the Vienna footway-only world (M12C); the same file as '
          'VIENNA/RISK+GREEN-RL/extended_ft, although the v1 manifest calls it shortFT'),
    _ckpt('VIENNA/RISK+GREEN-RL/short_ft', 'RISK+GREEN-RL', None, 'M12C_vienna_reward_only',
          'models/vienna_finetuning/stage_M12C_vienna_reward_only_C_ckpt_completed_000075776_global_000405504.zip',
          'b331236e3b41d790733c6f6ba7a9f030c0718088cf4acb20c5be43d9ee48bbf0', 2154536,
          'short Vienna fine-tuning of RISK+GREEN-RL/1001 (75,776 stage C steps; the primary Vienna table)'),
])


def checkpoint_env_overrides(key: str) -> Optional[Dict[str, object]]:
    """``EnvConfig`` overrides for the environment checkpoint ``key`` was trained in,
    beyond its method preset (None for all but the no-LiDAR ablation)."""
    if not ARCHIVED_CHECKPOINTS[key].urban_lidar:
        from ..env.config import LidarConfig

        return {"lidar": LidarConfig(urban=False)}
    return None


def sha256_file(path: Union[str, Path], chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def archived_checkpoint(key: str, archive: Union[str, Path], verify: bool = True) -> Path:
    """Path of checkpoint ``key`` in the extracted archive ``archive``; with ``verify``
    its SHA-256 is checked against the archive's manifest."""
    if key not in ARCHIVED_CHECKPOINTS:
        raise KeyError(f"unknown checkpoint {key!r}; available: {sorted(ARCHIVED_CHECKPOINTS)}")
    entry = ARCHIVED_CHECKPOINTS[key]
    path = Path(archive) / entry.path
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint {key!r} not found at {path}")
    if verify:
        actual = sha256_file(path)
        if actual != entry.sha256:
            raise ValueError(f"checkpoint {key!r} at {path} has SHA-256 {actual}, expected {entry.sha256}")
    return path


@dataclass(frozen=True)
class ArchivedEvaluation:
    """A primary (no-terminal) policy evaluation of the archive."""

    name: str
    checkpoint: str
    #: Environment preset of the evaluation (it can differ from the training preset:
    #: the static-trained RISK+GREEN-RL policies were evaluated with dynamic risk).
    method: str
    #: "canonical" (AOI_SYNTH_000) or "vienna" (the footway-only Vienna world, not distributed).
    world: str
    row_format: str
    #: The ``method`` (posthoc and dynamic formats) or ``policy_kind`` (transfer) column.
    label: str
    seed0: int
    n_episodes: int
    #: Per-episode CSV and summary inside the extracted archive.
    episodes_csv: str
    summary_json: str
    #: ``UrbanWorldLayers.layers_sha256()`` of the world the evaluation ran in.
    world_layers_sha256: str = ""


_STATIC = "experiments/paperA_submission_repair_v1/primary_no_terminal/static"
_DYNAMIC = "experiments/paperA_primary_no_terminal_dynamic/eval"
_VIENNA = "experiments/paperA_primary_no_terminal_vienna/eval"


def _evaluations():
    out = []
    for method in ("GEO-RL", "OBS-RL", "RISK+GREEN-RL"):
        for seed in (1001, 2001, 3001, 4001, 5001):
            d = f"{_STATIC}/{method}/trainseed_{seed}"
            out.append(ArchivedEvaluation(f"static/{method}/{seed}", f"{method}/{seed}", method, "canonical",
                                          "posthoc", method, 5000, 200, f"{d}/episodes.csv", f"{d}/summary.json"))
    d = f"{_STATIC}/RISK+OBS-RL/single_checkpoint"
    out.append(ArchivedEvaluation("static/RISK+OBS-RL/single", "RISK+OBS-RL/single", "RISK+OBS-RL", "canonical",
                                  "posthoc", "RISK+OBS-RL", 5000, 200, f"{d}/episodes.csv", f"{d}/summary.json"))
    for seed in (1001, 2001, 3001, 4001, 5001):
        p = f"primary_no_terminal_static_trained_RISK_GREEN_seed{seed}"
        out.append(ArchivedEvaluation(f"dynamic/RISK+GREEN-RL/{seed}", f"RISK+GREEN-RL/{seed}", "DYN-RISK-RL",
                                      "canonical", "dynamic", "RL_reward_only_dynamic_eval", 5000, 200,
                                      f"{_DYNAMIC}/{p}/{p}_episodes.csv", f"{_DYNAMIC}/{p}/{p}_summary.json"))
    for seed in (1001, 2001, 3001, 4001, 5001):
        p = f"primary_no_terminal_M11C_seed{seed}"
        out.append(ArchivedEvaluation(f"dynamic/DYN-OBS-RL/{seed}", f"DYN-OBS-RL/{seed}", "DYN-OBS-RL", "canonical",
                                      "dynamic", "RL_dynamic_obs_reward_only", 5000, 200,
                                      f"{_DYNAMIC}/{p}/{p}_episodes.csv", f"{_DYNAMIC}/{p}/{p}_summary.json"))
    p = "primary_no_terminal_M11B_single_checkpoint"
    out.append(ArchivedEvaluation("dynamic/DYN-RISK-RL/single", "DYN-RISK-RL/single", "DYN-RISK-RL", "canonical",
                                  "dynamic", "RL_reward_only_dynamic_eval", 5000, 200,
                                  f"{_DYNAMIC}/{p}/{p}_episodes.csv", f"{_DYNAMIC}/{p}/{p}_summary.json"))
    for name, ckpt in (("zero_shot", "RISK+GREEN-RL/1001"), ("short_finetuned", "VIENNA/RISK+GREEN-RL/short_ft"),
                       ("extended_finetuned", "VIENNA/RISK+GREEN-RL/extended_ft")):
        p = f"primary_no_terminal_Vienna_{name}_RISK_GREEN"
        out.append(ArchivedEvaluation(f"vienna/RISK+GREEN-RL/{name}", ckpt, "RISK+GREEN-RL", "vienna", "transfer",
                                      "reward_only", 7000, 200, f"{_VIENNA}/{p}/{p}_episodes.csv",
                                      f"{_VIENNA}/{p}/{p}_summary.json"))
    d = f"{_VIENNA}/VIENNA-GEO/trainseed_1001"
    out.append(ArchivedEvaluation("vienna/GEO-RL/1001", "GEO-RL/1001", "GEO-RL", "vienna", "posthoc", "VIENNA-GEO",
                                  7000, 200, f"{d}/episodes.csv", f"{d}/summary.json"))
    return {e.name: e for e in out}


#: Layers checksums of the worlds of the archived evaluations: the canonical world
#: (AOI_SYNTH_000, the archive's AOI_FAKE_001) and the processed footway-only Vienna
#: world of the primary Vienna evaluations
#: (scenarios/real_gis/vienna_prater_leopoldstadt_osm_1km_v1_footway_only_seed7000_n200.json).
WORLD_LAYERS_SHA256 = {
    "canonical": "be2bdac959eb9507598b77f8f0531dedf73e4f723bfacf162ea0cb9c4f785be4",
    "vienna": "fa99c5710f9bcb2345901f986f0e97f59bddf26d775047c8c5f2e6079f692786",
}

#: The primary policy evaluations of the paper, by name: ``static/<method>/<seed>``
#: (canonical world), ``dynamic/...`` (canonical world with dynamic risk) and
#: ``vienna/...``. All ran in the primary no-terminal regime, deterministically.
ARCHIVED_EVALUATIONS: Dict[str, ArchivedEvaluation] = {
    name: replace(entry, world_layers_sha256=WORLD_LAYERS_SHA256[entry.world])
    for name, entry in _evaluations().items()
}
