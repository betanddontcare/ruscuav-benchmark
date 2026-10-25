# RUSCUAV one-click submission-repair runner

## Required notebook state

Use the old `TOP_UAV` notebook. Before launching the runner:

1. Mount Google Drive.
2. Execute all TOP_UAV class/function definition cells.
3. Execute all cells from `RUSCUAV_submission_repair_cells.py` once. Their `RUN_*` flags may remain `False`; the one-click runner calls the functions directly.
4. Save `RUSCUAV_submission_repair_one_click_runner.py` on Drive, for example:
   `/content/drive/MyDrive/RUSCUAV/code/RUSCUAV_submission_repair_one_click_runner.py`.
5. Execute one line:

```python
%run -i /content/drive/MyDrive/RUSCUAV/code/RUSCUAV_submission_repair_one_click_runner.py
```

## Checkpoint preflight

The runner attempts to discover M6, M11B, Vienna short-fine-tuned, and Vienna extended-fine-tuned checkpoints. If any are ambiguous or missing, it stops **before** expensive evaluations and writes:

`/content/drive/MyDrive/RUSCUAV/experiments/paperA_submission_repair_v1/master_run/resolved_checkpoints.csv`

Set the unresolved paths once in `CHECKPOINT_OVERRIDES` at the top of the runner and execute the same `%run -i` command again.

## Resume behavior

The runner is resume-safe. After a Colab restart, rerun TOP_UAV definitions, repair definitions, and the same one-line `%run -i` command. Completed master stages are skipped and partial per-episode evaluations resume from CSV files.

## Logs and state

- Log: `.../paperA_submission_repair_v1/master_run/master_run.log`
- State: `.../paperA_submission_repair_v1/master_run/master_run_state.json`
- Checkpoint resolution: `.../paperA_submission_repair_v1/master_run/resolved_checkpoints.csv`

## Execution order

1. Provenance/config audit
2. Static primary no-terminal RL
3. Static primary no-terminal planners
4. Dynamic primary no-terminal evaluation
5. Vienna primary no-terminal evaluation
6. Warsaw primary no-terminal and aligned planner Stage C evaluation
7. Conditional/normalized risk, cause decomposition, beta-dynamic sensitivity
8. Complete Holm families
9. Verified Warsaw Fig. 9
10. Release-readiness audit
