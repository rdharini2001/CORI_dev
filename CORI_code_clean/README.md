# CORIdeep v4 notebooks — Croydon-train, other-centers-test

These notebooks implement the requested split: **train only on Croydon** and evaluate on **all non-Croydon centers**. They also follow the senior multivariable-analysis pattern: fit CoxPHFitter on train, predict partial hazards on test, and report train/test C-index plus HR/CI/p from the fitted Cox model.

Run order:
1. H1 — primary pan-cancer model and subtype analyses.
2. H2 — cancer CORIdeep vs non-cancer M-MACE cross-prediction.
3. H3 — treatment-stratified analyses and interaction tests.
4. H4 — visit-aware CMR correlates.

Edit only the path block in the first code cell if needed. The train center is controlled by `TRAIN_CENTER_PATTERN = "croydon"`. If your center column is numeric-coded, the notebook will stop and print the top center labels so you can replace the pattern/code.
