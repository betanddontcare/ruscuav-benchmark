"""The paper's primary tables, rebuilt from the archived per-episode CSVs.

The v2.0.0 post-processing (``stage_postprocess``, ``stage_holm`` and
``_write_static_aggregate`` of the one-click runner) built the primary tables of the
paper from the per-episode CSVs of the primary (no-terminal) evaluations, listed in
``paper_ready_v2/tables/paper_table_primary_no_terminal_csv_registry.csv``.
:func:`paper_tables` rebuilds them with :mod:`ruscuav_benchmark.scoring` from an
extracted copy of the archive, and :func:`write_paper_tables` writes them under the
archived file names, so that they can be compared file by file
(``scripts/migration/score_parity.py``).

The archive read the CSVs back with ``pandas.read_csv``, whose default float
parser differs from exact parsing in the last bit for about a fifth of the cells;
the tables are rebuilt from the values as pandas read them
(:func:`ruscuav_benchmark.tables.parse_float`). The static five-seed table summarized
the evaluator's own summaries, computed before the rows were written, and is
rebuilt from the exactly parsed rows.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Union

from .scoring import (PAPER_HOLM_COMPARISONS, Run, cause_table, dynamic_budget_table, holm_table, load_run,
                      read_registry, summary_table, training_seed_table)
from .tables import Row, write_rows_csv

__all__ = ["ARCHIVE_DRIVE_ROOT", "REGISTRY", "TABLE_FILES", "HOLM_INPUTS", "STATIC_SEED_METRICS", "archived_runs",
           "paper_tables", "write_paper_tables"]

#: The Google Drive folder the archived paths start with.
ARCHIVE_DRIVE_ROOT = "/content/drive/MyDrive/RUSCUAV/"
#: The registry of the primary per-episode CSVs, inside the archive.
REGISTRY = "paper_ready_v2/tables/paper_table_primary_no_terminal_csv_registry.csv"
_TABLES = "paper_ready_v2/tables/"
#: The rebuilt tables and their archived files.
TABLE_FILES = {
    "all_run_summaries": _TABLES + "paper_table_primary_no_terminal_all_run_summaries.csv",
    "conditional_normalized_risk_metrics": _TABLES + "paper_table_conditional_normalized_risk_metrics.csv",
    "non_acceptability_cause_decomposition": _TABLES + "paper_table_non_acceptability_cause_decomposition.csv",
    "dynamic_budget_sensitivity": _TABLES + "paper_table_dynamic_budget_sensitivity.csv",
    "training_seed_ci_all_scopes": _TABLES + "paper_table_primary_no_terminal_training_seed_ci_all_scopes.csv",
    "static_training_seed_ci": _TABLES + "paper_table_primary_no_terminal_static_training_seed_ci.csv",
    "holm_full_families": _TABLES + "paper_table_holm_full_families.csv",
    "holm_claim_driving_subset": _TABLES + "paper_table_holm_claim_driving_subset.csv",
}
_SR = "experiments/paperA_submission_repair_v1/primary_no_terminal/static/"
_DYN = "experiments/paperA_primary_no_terminal_dynamic/eval/"
_VIE = "experiments/paperA_primary_no_terminal_vienna/eval/"
#: The runs of the paper's Holm families (``stage_holm``), inside the archive.
HOLM_INPUTS = {
    "GEO-RL": _SR + "GEO-RL/trainseed_1001/episodes.csv",
    "RISK+GREEN-RL": _SR + "RISK+GREEN-RL/trainseed_1001/episodes.csv",
    "RISK+OBS-RL": _SR + "RISK+OBS-RL/single_checkpoint/episodes.csv",
    "FULL-RH-A*": "experiments/paperA_primary_no_terminal_static_planners/eval/primary_no_terminal_FULL_RH_Astar_n200_"
                  "seed5000/primary_no_terminal_FULL_RH_Astar_n200_seed5000_episodes.csv",
    "STATIC-RL-DYN": _DYN + "primary_no_terminal_static_trained_RISK_GREEN_seed1001/"
                            "primary_no_terminal_static_trained_RISK_GREEN_seed1001_episodes.csv",
    "DYN-RISK-RL": _DYN + "primary_no_terminal_M11B_single_checkpoint/primary_no_terminal_M11B_single_checkpoint_"
                          "episodes.csv",
    "DYN-OBS-RL": _DYN + "primary_no_terminal_M11C_seed1001/primary_no_terminal_M11C_seed1001_episodes.csv",
    "DYN-RH-A*": _DYN + "primary_no_terminal_dynamic_receding_Astar/primary_no_terminal_dynamic_receding_Astar_"
                        "episodes.csv",
    "VIENNA-ZERO": _VIE + "primary_no_terminal_Vienna_zero_shot_RISK_GREEN/primary_no_terminal_Vienna_zero_shot_"
                          "RISK_GREEN_episodes.csv",
    "VIENNA-EXT": _VIE + "primary_no_terminal_Vienna_extended_finetuned_RISK_GREEN/primary_no_terminal_Vienna_"
                         "extended_finetuned_RISK_GREEN_episodes.csv",
    "VIENNA-PLANNER": _VIE + "primary_no_terminal_Vienna_FULL_RH_Astar/primary_no_terminal_Vienna_FULL_RH_Astar_"
                             "episodes.csv",
    "VIENNA-GEO": _VIE + "VIENNA-GEO/trainseed_1001/episodes.csv",
}
#: The statistics of the static five-seed table (``_write_static_aggregate``).
STATIC_SEED_METRICS = (
    "success_rate", "mission_acceptability_rate", "acceptable_given_success_rate", "mean_urban_cumulative_risk",
    "mean_urban_cumulative_risk_given_success", "mean_urban_risk_per_step", "mean_urban_risk_per_s",
    "mean_urban_risk_per_m_xy", "mean_road_fraction", "mean_ped_fraction", "mean_building_fraction",
)
_STATIC_RL = ("GEO-RL", "OBS-RL", "RISK+GREEN-RL")


def archived_runs(archive: Union[str, Path], exact_floats: bool = False) -> List[Run]:
    """The 53 runs of the archived registry (52 primary, one secondary), read from
    the archive; the tables keep the recorded Drive paths."""
    archive = Path(archive)
    return read_registry(archive / REGISTRY, base=archive, replace_prefix=(ARCHIVE_DRIVE_ROOT, archive),
                         exact_floats=exact_floats)


def paper_tables(archive: Union[str, Path]) -> Dict[str, List[Row]]:
    """The tables of :data:`TABLE_FILES`, rebuilt from the archived per-episode CSVs."""
    archive = Path(archive)
    primary = [run for run in archived_runs(archive) if run.regime == "primary_no_terminal"]
    summaries = summary_table(primary)
    exact = [run for run in archived_runs(archive, exact_floats=True)
             if run.regime == "primary_no_terminal" and run.scope == "static" and run.label in _STATIC_RL]
    holm_runs = {label: load_run(archive / path, label) for label, path in HOLM_INPUTS.items()}
    holm = holm_table(PAPER_HOLM_COMPARISONS, holm_runs)
    return {
        "all_run_summaries": summaries,
        "conditional_normalized_risk_metrics": summaries,
        "non_acceptability_cause_decomposition": cause_table(primary),
        "dynamic_budget_sensitivity": dynamic_budget_table(primary),
        "training_seed_ci_all_scopes": training_seed_table(summaries),
        "static_training_seed_ci": training_seed_table(summary_table(exact), STATIC_SEED_METRICS, group_by=("label",),
                                                       min_seeds=1),
        "holm_full_families": holm,
        "holm_claim_driving_subset": [row for row in holm if row["claim_driving"]],
    }


def write_paper_tables(archive: Union[str, Path], directory: Union[str, Path]) -> Dict[str, Path]:
    """Write the rebuilt tables into ``directory`` under the archived file names."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return {name: write_rows_csv(directory / Path(TABLE_FILES[name]).name, rows)
            for name, rows in paper_tables(archive).items()}
