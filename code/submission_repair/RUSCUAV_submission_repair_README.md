# RUSCUAV submission-repair pipeline

This package contains the code required to close the remaining submission-blocking issues without retraining PPO.

## Files

- `RUSCUAV_submission_repair_pipeline.ipynb` — Colab notebook; recommended entry point.
- `RUSCUAV_submission_repair_cells.py` — the same pipeline as `# %%` cells, convenient for version control and `%run`.

## Prerequisites

1. Mount Google Drive.
2. Run all class and function definition cells from the current `TOP_UAV` notebook, or import the frozen benchmark definitions.
3. Confirm that `/content/drive/MyDrive/RUSCUAV` contains the trained checkpoints and evaluation artifacts.
4. Do not change the five PPO training seeds: `1001, 2001, 3001, 4001, 5001`.

All expensive switches are `False` by default. Enable one `RUN_*` block at a time. The evaluators are resume-safe and write an episode CSV after each completed episode.

## Recommended execution order

1. **Provenance audit** — set `RUN_PROVENANCE_AUDIT = True`.
   - Produces checksums, exact paths, extracted configuration parameters, and a conflict report.
   - Use these outputs to finalize Methodology and Appendix A6.
2. **Primary static no-terminal evaluation** — enable `RUN_STATIC_PRIMARY_RL`, then `RUN_STATIC_PRIMARY_PLANNERS`.
   - Soft road/pedestrian/building budgets are logged but do not stop the episode.
   - Mission acceptability is recomputed post hoc.
3. **Primary dynamic no-terminal evaluation** — set `RUN_DYNAMIC_PRIMARY = True`.
4. **Vienna and Warsaw no-terminal evaluation** — enable `RUN_VIENNA_PRIMARY` and `RUN_WARSAW_PRIMARY` separately.
   - Use the same 200 scenario seeds for the learned policies and planner within each AOI.
5. **Derived metrics and cause decomposition** — run the cells that create conditional/normalized risk and non-acceptability causes.
6. **Dynamic-budget sensitivity** — recompute DynOK over a grid of `beta_dyn` values from existing per-episode dynamic-risk CSVs.
7. **Full Holm families** — populate `CSV_REGISTRY` and run `compute_holm_families(...)`.
8. **Fig. 9 validation/regeneration** — generate the trajectory figure only from a row whose post-hoc acceptability flag is true.
9. **Release audit** — set `RUN_RELEASE_AUDIT = True` after all outputs are complete.

## Primary and secondary evaluation regimes

- **Primary:** `stage="B"`, strict geometric checks, no termination on soft exposure-budget exceedance. Report SR and compute MAR post hoc.
- **Secondary operational-stopping regime:** `stage="C"`, where selected soft-budget exceedances may terminate a mission. Do not call its completion rate ordinary SR without the regime qualifier.

## Outputs that must replace manuscript tables

- Table I / main static results: training-seed aggregates from the primary no-terminal evaluation.
- Table II / dynamic results: primary no-terminal rates, dynamic-risk distribution, and `beta_dyn` sensitivity.
- Table III / Vienna: scenario-aligned primary no-terminal evaluation.
- Table IV: full reproducible paired-test families with raw and Holm-adjusted p-values.
- Tables XII-XIII: rerun or relabel the existing strict results as secondary operational-stopping stress tests.
- Supplement: conditional risk, risk per step/second/metre, exposure fractions, cause decomposition, complete Holm families, and provenance manifest.

## Important interpretation rules

- Algorithm-level RL claims use all five training seeds.
- A single checkpoint may be used only for trajectory figures or diagnostics and must be identified by training seed and SHA-256.
- Do not select a checkpoint using the reported test MAR.
- Time-based exposure budgets are benchmark parameters, not regulatory limits, and depend on both route and speed.
- The green bonus is a benchmark heuristic; it does not establish that parks or green areas are intrinsically safe flight corridors.
