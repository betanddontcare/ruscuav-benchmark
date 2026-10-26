# RUSCUAV v2.0.0 — Zenodo reproducibility release

**RUSCUAV** (Reproducible Urban Semantic Cargo-UAV) is a benchmark for evaluating low-altitude urban cargo-UAV navigation using **benchmark-defined semantic mission acceptability**, rather than target-reaching success alone.

This release supports the manuscript:

> **RUSCUAV: A Semantic Mission-Acceptability Benchmark for Low-Altitude Urban Cargo UAV Navigation**  
> Igor Betkier and Mateusz Oszczypała

The release contains source code, synthetic worlds, scenario data, PPO weights, planner and policy outputs, per-episode CSVs, frozen configurations, paper-ready tables and figures, statistical families, provenance manifests, and the manuscript/supplement PDFs. The supplementary material documents the 20-D semantic vector, 6-D dynamic-risk vector, PPO hyperparameters, reward/budget values, method naming, and frozen planner/risk/dynamics parameters.

## Version and record

- Release version: **2.0.0**
- Previous Zenodo version: **10.5281/zenodo.21331403**
- This package is intended for publication as the **next version of the same Zenodo record**.
- Zenodo will assign a new version-specific DOI when the new version is published.

## What changed in v2.0.0

Version 2.0.0 changes the primary evaluation protocol and therefore constitutes a major release:

1. **Primary no-terminal protocol.** Soft road, pedestrian, building, and dynamic-risk budget exceedances are recorded but do not abort the mission. Success is target-region completion; mission acceptability is computed post hoc.
2. **Secondary operational-stopping protocol.** The former strict Stage-C stopping rule is retained as a separate stress test and is not pooled with primary success rate.
3. **Five-seed learned-policy reporting.** Headline PPO results use training-seed mean, sample standard deviation, and 95% t-confidence intervals.
4. **Multi-world and Warsaw zero-shot analyses.** The archive includes the three-world synthetic morphology-shift experiment and scenario-aligned Vienna/Warsaw real-GIS simulation evaluations.
5. **Full provenance and statistical families.** Claim-driving artifacts, checkpoint resolution, configuration audit, conditional/normalized risk, non-acceptability causes, dynamic-budget sensitivity, and complete Holm families are archived.
6. **Immutable Vienna checkpoints.** The exact short (75,776 Stage-C steps) and extended (251,904 Stage-C steps) fine-tuning checkpoints are included.

## Primary benchmark distinction

```text
Success = target-region completion
Mission acceptability = Success
                        AND cargo health >= 0.80
                        AND road time <= 20 s
                        AND pedestrian time <= 5 s
                        AND building time <= 15 s
                        AND sensitive-zone time <= 0 s
                        AND hard-violation steps = 0
```

The dynamic regime additionally reports whether cumulative dynamic risk is within the benchmark budget of 10. These thresholds are benchmark parameters, not regulatory approval criteria or validated societal-preference thresholds.

## Archive layout

```text
RUSCUAV-v2.0.0/
├── README.md, CITATION.cff, codemeta.json, .zenodo.json
├── BENCHMARK_SPEC.md, CHANGELOG.md
├── LICENSE_CODE_MIT, LICENSE_DATA_CC_BY_4_0
├── code/submission_repair/          final repair and one-click runner code
├── source_code/                     full supplied Colab/Python exports
├── src/, scripts/, tests/, examples/
├── configs/                         frozen experiment configurations
├── worlds/synthetic/                canonical and AOI_SYNTH_001–003 worlds
├── worlds/real_gis/metadata/        Vienna/Warsaw metadata only
├── scenarios/
├── models/                          model manifest + immutable Vienna checkpoints
├── experiments/                     raw per-episode CSVs, summaries and model weights
├── results/                         curated archived references
├── paper_ready_v2/                  final tables, JSON and vector/raster figures
├── paper/                           manuscript and supplementary PDFs
├── provenance/                      configs, hashes, resolved checkpoints, run state
├── reproducibility/                 environment snapshots and verification assets
├── manifests/                       file, model and world manifests
└── SHA256SUMS
```

## Real-GIS release policy

Processed Vienna and Warsaw `UrbanWorldLayers` NPZ files are **not included** in v2.0.0. The release includes:

- real-GIS generation code in the full source export;
- the frozen real-GIS pipeline configuration;
- AOI metadata and layer-construction parameters;
- per-episode evaluation outputs and summaries;
- OpenStreetMap attribution information.

Users can regenerate the real-GIS worlds from OpenStreetMap data with `ruscuav build-aoi --preset vienna` (or `warsaw`), the package's port of the archived pipeline, and build worlds of other areas the same way (`docs/BUILDING_WORLDS.md`). See `docs/OSM_ATTRIBUTION.md` and `worlds/real_gis/README.md`.

## Verification

From the extracted archive root:

```bash
python scripts/verify_v2_release.py --root .
```

The verifier checks required files, file hashes, exact Vienna checkpoints, final PDFs, primary tables, representative Warsaw paths, and the absence of processed Vienna/Warsaw NPZ files.

For a complete checksum verification:

```bash
sha256sum -c SHA256SUMS
```

## Key result locations

- Primary static five-seed summary: `paper_ready_v2/tables/paper_table_primary_no_terminal_static_training_seed_ci.csv`
- All primary run summaries: `paper_ready_v2/tables/paper_table_primary_no_terminal_all_run_summaries.csv`
- Training-seed summary across scopes: `paper_ready_v2/tables/paper_table_primary_no_terminal_training_seed_ci_all_scopes.csv`
- Conditional and normalized risk: `paper_ready_v2/tables/paper_table_conditional_normalized_risk_metrics.csv`
- Non-acceptability causes: `paper_ready_v2/tables/paper_table_non_acceptability_cause_decomposition.csv`
- Dynamic-budget sensitivity: `paper_ready_v2/tables/paper_table_dynamic_budget_sensitivity.csv`
- Complete Holm families: `paper_ready_v2/tables/paper_table_holm_full_families.csv`
- Multi-world strict stress test: `paper_ready_v2/tables/paper_table_multiworld_synthetic_per_world.csv`
- Final figures: `paper_ready_v2/figures_final/`
- Exact run and checkpoint provenance: `provenance/`

See `docs/TABLE_FIGURE_ARTIFACT_MAP.md` for the full mapping.

## Environment

The archived Colab snapshot is stored under `reproducibility/v1_environment_and_audit/uploaded_assets/`. Asset verification requires only NumPy, pandas and PyYAML. Model loading and complete re-evaluation require Stable-Baselines3, PyTorch, Gymnasium, SciPy and the GIS stack listed in `requirements-reproducibility.txt`.

## Licenses

- **Source code and executable scripts:** MIT License (`LICENSE_CODE_MIT`).
- **Synthetic worlds, trained weights, results, tables, figures, documentation and paper files:** CC BY 4.0 (`LICENSE_DATA_CC_BY_4_0`).
- **OpenStreetMap-derived metadata:** released with attribution and subject to the upstream OpenStreetMap/ODbL terms described in `docs/OSM_ATTRIBUTION.md`.

## Authors

- **Igor Betkier** — ORCID: 0000-0002-5437-7669; corresponding author; igor.betkier@wat.edu.pl
- **Mateusz Oszczypała** — ORCID: 0000-0002-1194-6913; mateusz.oszczypala@wat.edu.pl

Faculty of Civil Engineering and Geodesy, Military University of Technology, gen. Sylwestra Kaliskiego Street 2, 00-908 Warsaw, Poland.

## Citation

Use the version-specific Zenodo DOI assigned when v2.0.0 is published and cite the associated manuscript. Machine-readable citation metadata are provided in `CITATION.cff` and `codemeta.json`.
