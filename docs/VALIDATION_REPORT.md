# Validation report

## Source review

The refactor was built after inspecting:

- all nine notebooks in `old_code_with_all_experiments(1).zip`;
- `clean_demo(1).zip`, including `101_main.ipynb` and its helper modules;
- the July 18 handoff and collaborator comments;
- the uploaded clean locked CLS cohort schemas;
- the treatment, clinical-status, and CMR file schemas.

## Automated checks

`pytest` smoke tests cover:

1. deletion of known stale CORI/MMACE score columns;
2. treatment-file `usecols` isolation;
3. one-to-one merge membership/order preservation;
4. training and scoring through the single generic `train_model()` function.

All smoke tests passed during package creation.

## Feature-ranking check

The vectorized univariate Cox ranking helper was checked against `lifelines.CoxPHFitter` on the uploaded D1 CLS cohort. For the top ten features, the fast and lifelines z statistics had correlation greater than 0.999999 and identical ordering. The helper avoids the many-minute feature-by-feature loop while the final multivariable model is still fitted directly with lifelines.

## Runtime checks completed

- D1 CLS: 2,009 participants and 1,024 features loaded and ranked successfully.
- D3 CLS: 14,159 participants and 1,024 features loaded and ranked successfully.
- Synthetic end-to-end model fit, scoring, adjusted Cox analysis, and bootstrap C-index completed successfully.
- A two-repetition event/covariate-matched MMACE smoke run completed on the uploaded clean CLS cohorts.

## Not executed end to end here

The complete notebook was not run end to end in the sandbox because the user-local `meanpool_pre`, all seven representation folders, `CORI_allcancer_8Jan2026.csv`, and handcrafted locked cohorts were not all uploaded. The notebook performs a path audit and stops before analysis when any required local input is absent.
