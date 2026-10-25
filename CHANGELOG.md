# Changelog

## Unreleased (2.1.0.dev0)

- Legacy notebook code runs outside Colab: `ruscuav_benchmark.legacy.runtime` and
  `scripts/legacy_parity.py`, which reproduces archived primary episodes
  (see docs/REPRODUCIBILITY.md).
- World files are opened with `allow_pickle=False`; worlds from untrusted sources can
  no longer execute code when loaded.
- World NPZ schema `ruscuav-world/1.0` (docs/WORLD_NPZ_SCHEMA.md): `schema_version` and
  optional `meta_json` provenance; `ruscuav validate` checks files against it.
- New `ruscuav` command (`validate`, `evaluate`); `ruscuav-evaluate` is unchanged.
- The canonical world `AOI_FAKE_001` is renamed `AOI_SYNTH_000` and its CRS label
  corrected from `EPSG:2180` to `LOCAL_METRIC_SYNTHETIC`; the old ID still resolves.
  The four synthetic worlds are now part of the repository (layers unchanged).
- Canonical-world missions for seeds 5000–5199 recorded in
  `scenarios/canonical/AOI_SYNTH_000_seed5000_n200.csv`.
- The quickstart example runs from a repository checkout.
- One package version, read from `ruscuav_benchmark.__version__`.
- New Gymnasium environment `ruscuav_benchmark.env.RuscuavEnv` (extra `env`), one
  configurable class in place of the six notebook environment classes: method
  presets of the paper (GEO-RL … DYN-OBS-RL), stages A/B/C, the primary and secondary
  regimes, and separate modules for dynamics, shields, cargo, LiDAR, reward,
  observations, missions and dynamic risk. It reproduces the notebook environments
  step for step (docs/ENVIRONMENT.md). `ruscuav_benchmark.env.legacy_factories`
  provides the notebook factory names on top of it.
- The legacy runtime also restores the notebook's reward-factor variants
  (`risk_only`, `green_only`) and shield-diagnostic variants of `M8_VARIANTS`,
  which the definitions-only extraction had dropped; the RISK-RL and GREEN-RL
  environments can be created again.
- `scripts/legacy_parity.py --env refactored` re-runs archived episodes in the new
  environment, and `--method REWARD_FACTOR:…` the archived reward-factor
  evaluations; `scripts/migration/env_parity.py` and `env_parity_paths.py` compare
  the two environments step by step (docs/REPRODUCIBILITY.md).

## 2.0.0 — 2026-07-24

Major reproducibility and evaluation-protocol release.

- Changed the headline evaluation from strict soft-budget termination to primary no-terminal evaluation.
- Retained strict Stage C as a separately labelled operational-stopping stress test.
- Added five-seed primary no-terminal static and dynamic PPO evaluations.
- Added scenario-aligned primary no-terminal Vienna and Warsaw evaluations.
- Added zero-shot morphology-shift evaluation on AOI_SYNTH_001–003 as a secondary strict stress test.
- Added conditional/normalized risk, non-acceptability-cause decomposition and dynamic-budget sensitivity.
- Added complete fixed-checkpoint Holm families and provenance manifests.
- Added exact immutable Vienna short and extended fine-tuning checkpoints.
- Added final manuscript and supplementary PDFs and vector-quality final figures.
- Removed processed Vienna/Warsaw NPZ worlds from the public release; retained generation scripts and AOI metadata.
- Added v2 release verifier, full file/model/world manifests and SHA-256 sums.

## 1.0.0

Previous Zenodo release archived at DOI `10.5281/zenodo.21331403`.
