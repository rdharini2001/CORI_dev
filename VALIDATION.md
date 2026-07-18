# Validation report

## Completed checks

- All Python sources and runner scripts compile with `python -m compileall`.
- Eighteen package modules import successfully in one clean process.
- All three command-line runners expose valid `--help` interfaces.
- All three notebooks parse as valid notebook v4 files and contain no saved outputs.
- `pytest -q` passes all six tests.
- The parity test checks AST hashes for 98 moved core functions/classes against hashes computed from the uploaded implementations.
- No wildcard imports remain.
- The built wheel installs successfully and imports both the modern package API and explicit legacy compatibility modules.
- Compiled cache files are excluded from the distributable archive.

## Logic intentionally unchanged

The moved statistical implementations retain the original function bodies for cohort construction, feature filtering, Cox fitting/ranking, locked-model application, performance estimation, clinical models, treatment analyses, CMR analyses, handcrafted-feature models, mediation, and the original reusable plotting functions covered by the parity manifest.

## What cannot be verified in this environment

The private CSV datasets and precomputed feature directories were not uploaded with this request. Therefore, end-to-end numerical parity of cohort counts, selected features, model coefficients, C-indices, hazard ratios, and manuscript tables cannot be executed here. The recommended data-level parity procedure is described in `README.md`.
