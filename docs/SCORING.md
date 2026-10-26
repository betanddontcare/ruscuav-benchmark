# Scoring results

`ruscuav score` (`ruscuav_benchmark.scoring`) turns per-episode results into the
tables of the paper: for each run the success rate, the mission acceptability
computed post hoc with the budgets, the conditional and normalized risk and
exposure, the causes of non-acceptability and, for dynamic runs, a sweep of the
dynamic-risk budget; over training seeds the mean with a Student-t interval; and
between runs, exact paired tests of declared comparisons with Holm's correction.
It uses the archive's arithmetic: from the archived per-episode CSVs it rebuilds the
paper's eight primary tables character for character (docs/REPRODUCIBILITY.md).

A *run* is the per-episode CSV of one evaluation: one planner, one policy checkpoint
or any other method, as `ruscuav evaluate` writes it, as the archive has it, or
written by your own code with the columns below.

## On the command line

```bash
ruscuav score --protocol canonical-primary \
    MINE=runs/mine/MINE_canonical-primary_episodes.csv \
    FULL=runs/full/FULL_RH_Astar_canonical-primary_episodes.csv \
    GEO-RL@1001=runs/geo1001.csv GEO-RL@2001=runs/geo2001.csv \
    --compare MINE:FULL:success --compare MINE:FULL:MAR_static \
    --output-dir scores/
```

Each run is `[LABEL[@SEED]=]CSV`: the label names the method in the tables (by
default the rows' `method`), the seed is the training seed of a learned method, so
that runs of one method over several seeds are summarized together. A registry CSV
(`--registry`, the columns `label`, `csv_path` and optionally `dynamic`,
`train_seed`, `scope`, `regime`, as the archive's
`paper_table_primary_no_terminal_csv_registry.csv`) lists many runs at once.

With `--protocol` every run must have been evaluated under the protocol, and is
scored with the protocol's budgets and acceptability (dynamic acceptability in
`canonical-dynamic`). `ruscuav score` refuses a run

- with episodes of other seeds, or a seed twice (a run of the first missions only
  is accepted and reported as incomplete);
- without dynamic risk (the `dynamic_cumulative_risk` column) in a dynamic
  protocol, or with dynamic risk in a static one;
- whose rows were evaluated in another regime (`evaluation_regime`);
- whose evaluation recorded another protocol (by its digest), or another world or
  regime, in the `<prefix>_config.json` or `<prefix>_summary.json` that
  `ruscuav evaluate` writes next to `<prefix>_episodes.csv`. Archived CSVs and CSVs
  of your own code have no such record: their seeds, columns and regime are checked.

`score.json` records the checks of each run (`protocol_checks`), and the tables give
every run the protocol's regime. Without a protocol the paper's budgets apply and
runs whose rows have `dynamic_cumulative_risk` are scored as dynamic.

`--compare` declares comparisons: `A:B:METRIC` (repeatable, one family), or a CSV
with the columns `family`, `comparison_id`, `method_a`, `method_b`, `metric` and
optionally `claim_driving`, as the paper's families
(`ruscuav_benchmark.scoring.PAPER_HOLM_COMPARISONS`). The metrics are `success`,
`MAR_static` (static acceptability), `DynOK` (cumulative dynamic risk within the
budget) and `MAR_dynamic`. Compared methods must be single runs (one label each).

The output directory gets

| File | Content | Archive counterpart |
|---|---|---|
| `run_summaries.csv` | one row per run | `paper_table_primary_no_terminal_all_run_summaries.csv` |
| `training_seed_ci.csv` | per scope, regime and method with two or more training seeds: n, mean, std, 95 % interval of success, acceptability and acceptability given success | `paper_table_primary_no_terminal_training_seed_ci_all_scopes.csv` (primary runs only, without the regime column) |
| `non_acceptability_causes.csv` | per run and cause | `paper_table_non_acceptability_cause_decomposition.csv` |
| `dynamic_budget_sensitivity.csv` | per dynamic run and budget 2.5–35 | `paper_table_dynamic_budget_sensitivity.csv` |
| `paired_tests_holm.csv` | per comparison | `paper_table_holm_full_families.csv` |
| `score.json` | the inputs (the recorded and the local path, the SHA-256 of the file read), the protocol checks, the summaries and tests | |

and the summary table is printed.

## The statistics

- **Post-hoc acceptability**: success, cargo health at least 0.80, at most 20 s over
  roads, 5 s over pedestrian ways, 15 s over buildings and 0 s over sensitive zones,
  no hard violation; dynamic acceptability adds a cumulative dynamic risk of at most
  10 (`ruscuav_benchmark.posthoc`). Acceptability given success (AGS) is the share
  of acceptable episodes among the successful ones.
- **Training-seed interval** (`stats.training_seed_ci`): the mean over the training
  seeds of a learned method, the standard deviation (n − 1) and the two-sided 95 %
  Student-t interval. This is the primary uncertainty of learned-policy results
  (BENCHMARK_SPEC.md).
- **Paired test** (`stats.mcnemar_exact_p`): on the missions both runs flew, the
  exact two-sided McNemar test of the discordant missions (a binomial test with
  p = 1/2), computed with SciPy's `binomtest` as the archive did; the rates are those
  of each whole run and the effect is their difference in percentage points.
- **Holm's correction** (`stats.holm_adjust`): within each family, the k-th smallest
  p-value times (m − k + 1), made non-decreasing; significant when below 0.05.
  Fixed-checkpoint paired tests are exploratory and need complete, declared
  families.
- **Episode bootstrap** (`stats.episode_bootstrap_ci`): the percentile interval over
  2,000 resamples of the episodes of one run, the secondary (fixed-policy)
  uncertainty of the archive's multi-seed tables.

The intervals and the p-values depend on SciPy (the Student-t quantile and
`binomtest`) and NumPy in their last digits. The archive's values are reproduced
exactly with the versions of `requirements-reproducibility.txt` (NumPy 2.0.2, SciPy
1.16.3); with NumPy 2.5.4 and SciPy 1.18.1, for example, the training-seed bounds of
the paper's tables differ by at most 5.4e-15, relatively, and every other number is
the same (docs/REPRODUCIBILITY.md).

## Reading CSVs

The archive built its tables from the per-episode CSVs read back with
`pandas.read_csv`, whose default float parser differs from exact parsing in the last
bit for about a fifth of the cells. `ruscuav_benchmark.tables.read_rows_csv` reads
CSVs as pandas does, without pandas: the same column types (integers, floats,
booleans, text, missing cells), the same skipped lines (blank lines, a UTF-8
byte-order mark) and, by default, the same floats; `--exact-floats`
(`exact_floats=True`) reads them correctly rounded instead, as
`float_precision="round_trip"` does. With exact parsing 16 of the 52 archived run
summaries would differ in the last digits.

## The columns of a run

The tables read these columns of the per-episode rows; the CSVs of
`ruscuav evaluate` have them (docs/PLANNERS.md, docs/POLICIES.md,
docs/TRAJECTORIES.md):

| Column | Meaning |
|---|---|
| `seed` | the mission's seed (pairs missions across runs) |
| `is_success` | reached the target region |
| `steps` | environment steps of 0.5 s |
| `cargo_health` | cargo health at the end (1 at the start) |
| `road_time`, `sidewalk_time`, `building_time`, `green_time`, `canopy_time`, `sensitive_time` | seconds over each layer |
| `urban_hard_violation_steps` (or `hard_violation_steps`) | steps with a hard violation |
| `urban_cumulative_risk` | time-integrated semantic risk |
| `path_length_xy` | flown horizontal distance (m), for the risk per metre |
| `dist_xy` | final horizontal distance to the target (m) |
| `dynamic_cumulative_risk` | time-integrated dynamic risk (dynamic runs) |
| `termination` | how the episode ended (counted in the summaries) |

## In Python

```python
from ruscuav_benchmark import scoring

runs = [scoring.load_run("runs/mine.csv", "MINE"), scoring.load_run("runs/full.csv", "FULL")]
tables = scoring.score_runs(runs, comparisons=[
    {"family": "mine", "comparison_id": "mine_vs_full_SR", "method_a": "MINE", "method_b": "FULL",
     "metric": "success"}])
tables.write("scores/")
print(tables.summaries[0]["mission_acceptability_rate"], tables.paired_tests[0]["p_holm"])
```

`ruscuav_benchmark.paper_tables.write_paper_tables(archive, directory)` rebuilds the
paper's primary tables from an extracted v2.0.0 archive under their archived names.
