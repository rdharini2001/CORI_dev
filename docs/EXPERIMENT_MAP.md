# Legacy-to-refactored experiment map

| Legacy source | Refactored notebook section | Main outputs |
|---|---|---|
| `CORI_FINAL_COHORT_REPAIR_AND_LOCK_v3.ipynb` and `cohort.ipynb` | Sections 2–4 | Input-path audit, locked cohort audit, stale-score removal audit |
| `H1_CORI_locked_model...ipynb` | Sections 5–9 | Baseline table, CORI model, primary validation, horizons, clinical/combined models, calibration and reclassification |
| `H2_MMACE_comparison...ipynb` | Sections 6–12 | Full MMACE, matched MMACE, cross-domain comparison, paired delta-C, CORI/MMACE reclassification, residualization, incremental value, feature-rank comparison |
| `H3_treatment...ipynb` | Section 13 | Treatment-stratified performance, generic adjusted Cox models, treatment interactions |
| `CORI_center_based...ipynb` | Section 14 | Center performance and forest plot |
| `CORI_additional_analyses...ipynb` | Sections 15–18 | Age/sex/treatment/site subgroups, representation specificity, score interactions, handcrafted comparator |
| `H4_CMR...ipynb` | Section 19 | Explicit CMR feature extraction, adjusted associations, domain ACAT and domain-PC global test |
| `CORI_only_mediation...ipynb` and legacy R mediation | Section 20 plus `R/mediation_DAG_2_3_4.R` | Newly trained held-out score export, two single-mediator DAGs and one joint DAG |

## Deliberate retirements

- Old `train_ready_f1024` risk-score columns are not inputs.
- Old H1/H2 model bundles are not inputs.
- The treatment file is not a source of CORI or MMACE scores.
- Cohorts are not rebuilt inside the manuscript notebook; the authoritative July 18 locked cohorts are audited and used as-is.
- Specialized treatment-model functions were replaced by the generic `adjusted_cox()` helper.
- CMR variables are an explicit list in the notebook, not inferred from arbitrary numeric column names.
