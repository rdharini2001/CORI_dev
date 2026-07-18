# Original-to-clean file map

| Uploaded source | Clean replacement | Role |
|---|---|---|
| `101_main.ipynb` | `notebooks/101_main_clean.ipynb` and `src/cori_analysis/pipelines/` | Thin orchestration notebook plus reusable H1/H2/H3/H4 runners. |
| `102_mediation.py` | `src/cori_analysis/mediation.py`, `scripts/run_mediation.py`, `notebooks/102_mediation_clean.ipynb` | Primary mediation configuration. |
| `102_mediation_cancerStatus_adjusted.py` | Same configurable mediation module with `--adjust-mediator-for-cancer-status` | Removes the duplicate script while preserving its formula/output/replicate differences. |
| `103_proteomics.ipynb` | `src/cori_analysis/proteomics.py`, `scripts/run_proteomics.py`, `notebooks/103_proteomics_clean.ipynb` | Reusable proteomics analysis and a thin reporting notebook. |
| `cori_pipeline_utils_v13(4).py` | `common.py`, `cohorts.py`, `survival.py`, `evaluation.py`, `clinical_models.py`, `plots.py` | Monolithic helper file split by responsibility. |
| `clinical_covariates.py` | `clinical_data.py` | Clinical status loading, merge, and covariate policy. |
| `cmr_helper.py` | `cmr.py` and `visualization.py` | CMR curation/preprocessing and CMR-specific plots. |
| `handcrafted_features.py` | `handcrafted.py` and `pipelines/handcrafted.py` | Handcrafted feature loading, locked models, and experiment orchestration. |
| `performance.py` | `horizon.py` | Safe 3/5/10-year performance evaluation. |
| `rank_transition_plot.py` | `rank_transition.py` | Feature-rank transition visualization. |
| `treatment_analysis.py` | `treatment.py` and `pipelines/h3.py` | Treatment-adjusted models and H3 orchestration. |
| `utils.py` | `common.py`; compatibility aliases retained in `src/utils.py` | Shared formatting, ID, and figure helpers. |
| `*.pyc` | Excluded | Python recreates these environment-specific cache files automatically. |

## Compatibility

The top-level modules under `src/` explicitly re-export the old public function names. Existing imports can therefore be migrated gradually, while all new code should import from `cori_analysis.<domain>`.
