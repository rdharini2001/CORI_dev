# Refactor Notes

## Main changes

- Replaced the 58-code-cell main notebook with thin calls to `prepare_shared_data`, `run_h1`, `run_h2`, `run_h3`, `run_h4`, `run_hcori`, and `run_hmmace`.
- Split the 1,454-line utility file into domain modules instead of importing everything with `*`.
- Consolidated the two 388-line mediation scripts into one configurable implementation.
- Moved reclassification, calibration, CMR violin/cluster/forest, and triangle plots out of notebook cells.
- Removed duplicated triangle-plot definitions, duplicate imports, empty cells, repeated path setup, global warning suppression in primary runners, hard-coded Windows/Linux drive fallbacks, and the hard-coded proteomics path.
- Replaced candidate cache filename scanning in the active pipeline with the single configured cache path:
  `H1_CORI_LOCKED_MODEL_v13/handcrafted_HCORI/tables/H1_handcrafted_subject_level_features_cached.csv`.
- Added exact-path and exact-clinical-column configuration.
- Added legacy import wrappers so old imports such as `from cori_pipeline_utils_v13 import ...` still work after installing this package.

## Exceptions and branches intentionally retained

Some conditional logic is part of the analysis rather than clutter and remains in the core functions:

- Event-count and sample-size checks before Cox/KM estimation.
- Per-feature failure handling during large univariate Cox screens.
- Development-only model/threshold derivation and held-out application.
- Underpowered treatment-stratum reporting.
- CMR family/QC selection.
- Raw handcrafted per-subject CSV loading when the explicit cache is rebuilt.

Removing these branches would alter the analysis or make sparse analyses fail completely.

## Source-level parity

`tests/test_logic_parity.py` verifies AST hashes for 98 moved function/class bodies against the uploaded source files. Pipeline orchestration, path handling, and notebook presentation were rewritten; core statistical function bodies were not.

## Important correction from the uploaded notebook

The uploaded main notebook imported `one_age_covars` even though the visible helper snippet only showed `one_age_covars_htn_diabetes`. The clean package provides the explicit alias in both the modern and compatibility APIs.

## Files intentionally excluded

Compiled Python cache files (`*.pyc`) were not copied. They are environment-specific, not source code, and are recreated automatically by Python.
