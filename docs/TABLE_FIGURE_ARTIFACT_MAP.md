# Paper table and figure artifact map

The manuscript and supplementary PDFs are the authoritative source for numbering. The following files are the canonical machine-readable inputs.

## Primary tables

| Scope | Canonical artifact |
|---|---|
| Static five-seed PPO | `paper_ready_v2/tables/paper_table_primary_no_terminal_static_training_seed_ci.csv` |
| Static seed-level runs | `paper_ready_v2/tables/paper_table_primary_no_terminal_static_seed_level.csv` |
| All primary runs | `paper_ready_v2/tables/paper_table_primary_no_terminal_all_run_summaries.csv` |
| All replicated scopes | `paper_ready_v2/tables/paper_table_primary_no_terminal_training_seed_ci_all_scopes.csv` |
| Conditional/normalized risk | `paper_ready_v2/tables/paper_table_conditional_normalized_risk_metrics.csv` |
| Non-acceptability causes | `paper_ready_v2/tables/paper_table_non_acceptability_cause_decomposition.csv` |
| Full Holm families | `paper_ready_v2/tables/paper_table_holm_full_families.csv` |
| Dynamic-budget sensitivity | `paper_ready_v2/tables/paper_table_dynamic_budget_sensitivity.csv` |
| Learning vs planning by information | `paper_ready_v2/tables/paper_table_equal_information_learning_vs_planning.csv` |

## Secondary tables

| Scope | Canonical artifact |
|---|---|
| Reward-factor diagnostic | `paper_ready_v2/tables/paper_table_reward_factor_ablation_training_seed_ci.csv` |
| Shield diagnostic | `paper_ready_v2/tables/paper_table_shield_diagnostic_reward_only.csv` |
| Multi-world strict stress test | `paper_ready_v2/tables/paper_table_multiworld_synthetic_per_world.csv` |
| Multi-world seed-level data | `paper_ready_v2/tables/paper_table_multiworld_synthetic_seed_level.csv` |
| Budget sensitivity | `paper_ready_v2/tables/paper_table_budget_sensitivity_multiseed_training_seed_ci.csv` |

## Final figures

| Figure content | PDF/SVG/PNG stem |
|---|---|
| Static primary no-terminal benchmark | `paper_ready_v2/figures_final/fig04_static_primary_no_terminal` |
| Dynamic primary no-terminal benchmark and ECDF | `paper_ready_v2/figures_final/fig06_dynamic_primary_no_terminal` |
| Vienna primary no-terminal transfer/fine-tuning | `paper_ready_v2/figures_final/fig07_vienna_primary_no_terminal` |
| Warsaw verified trajectories | `paper_ready_v2/figures_final/fig09_warsaw_verified_primary_no_terminal` |

## Representative Warsaw trajectories

```text
experiments/paperA_primary_no_terminal_warsaw_scenario_aligned/eval/
  primary_no_terminal_Warsaw_FULL_RH_Astar_seed7000/paths/
    episode_00000_seed_7000_actual_path.npy
    episode_00174_seed_7174_actual_path.npy
```

Episode 174 is benchmark-acceptable; episode 0 is successful but violates the pedestrian exposure budget.
