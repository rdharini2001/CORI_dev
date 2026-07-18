from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from ..clinical_data import one_age_covars
from ..cohorts import merge_treatment_labels
from ..common import QCLogger, ensure_dirs, safe_name
from ..config import AnalysisConfig
from ..evaluation import lrt_interaction, performance_row
from ..horizon import safe_horizon_performance
from ..plots import forest_plot, plot_km_high_low, plot_km_tertiles
from ..survival import apply_model_bundle, load_model_bundle
from ..treatment import fit_treatment_adjusted_cox, zscore_from_train_apply
from .common import SharedData


@dataclass
class H3Result:
    cancer_train: pd.DataFrame
    cancer_test: pd.DataFrame
    stratified_performance: pd.DataFrame
    horizon_performance: pd.DataFrame
    adjusted_models: pd.DataFrame
    interaction_tests: pd.DataFrame


def _interaction_or_status(df, score, flag, covariates, label):
    try:
        return lrt_interaction(df, score, flag, covars=covariates, label=label)
    except (ValueError, RuntimeError) as error:
        return {'analysis': label, 'status': str(error)}


def run_h3(shared: SharedData, config: AnalysisConfig) -> H3Result:
    run_dir = config.paths.output_dir / 'H3_treatment_locked_CORI_v13'
    outdir, figdir, tabledir, _, qcdir = ensure_dirs(run_dir)
    logger = QCLogger(qcdir / 'H3_QC_log.txt')
    logger.section('H3: treatment analyses with locked CORI')
    if not config.cori_bundle.exists():
        raise FileNotFoundError(f'Missing H1 locked CORI bundle: {config.cori_bundle}')

    cancer_train = merge_treatment_labels(shared.cancer_train.copy(), treatment_csv=config.paths.treatment, chemo_csv=config.paths.chemo, logger=logger)
    cancer_test = merge_treatment_labels(shared.cancer_test.copy(), treatment_csv=config.paths.treatment, chemo_csv=config.paths.chemo, logger=logger)
    shared.cancer_audit.to_csv(tabledir / 'H3_Table_00A_cohort_audit.csv', index=False)
    shared.cancer_split.to_csv(tabledir / 'H3_Table_01_center_split.csv', index=False)
    treatment_counts = pd.DataFrame([
        {'label': 'Any systemic treatment', 'N': int(cancer_train['treatment_any'].sum() + cancer_test['treatment_any'].sum())},
        {'label': 'Treatment naive', 'N': int(cancer_train['treatment_naive'].sum() + cancer_test['treatment_naive'].sum())},
        {'label': 'Chemotherapy', 'N': int(cancer_train['chemo_any'].sum() + cancer_test['chemo_any'].sum())},
        {'label': 'Immunotherapy', 'N': int(cancer_train['io_any'].sum() + cancer_test['io_any'].sum())},
    ])
    treatment_counts.to_csv(tabledir / 'H3_Table_00B_treatment_label_counts_overall.csv', index=False)

    bundle = load_model_bundle(config.cori_bundle)
    cancer_train = apply_model_bundle(bundle, cancer_train, score_col='CORI_score', prefix='CORI', logger=logger)
    cancer_test = apply_model_bundle(bundle, cancer_test, score_col='CORI_score', prefix='CORI', logger=logger)
    score_columns = [column for column in ['eid', shared.center_col, 'time', 'event', 'CORI_score', 'CORI_high_risk', 'CORI_risk_tertile', 'treatment_any', 'treatment_naive', 'chemo_any', 'io_any'] if column in cancer_test.columns]
    pd.concat([
        cancer_train[score_columns].assign(split='development'),
        cancer_test[score_columns].assign(split='held_out'),
    ]).to_csv(tabledir / 'H3_Table_02_CORI_scores_with_treatment.csv', index=False)

    strata = [
        ('All held-out cancer', cancer_test),
        ('Treatment-naive', cancer_test[cancer_test['treatment_naive'].eq(1)]),
        ('Any systemic treatment', cancer_test[cancer_test['treatment_any'].eq(1)]),
        ('No chemotherapy', cancer_test[cancer_test['chemo_any'].eq(0)]),
        ('Chemotherapy', cancer_test[cancer_test['chemo_any'].eq(1)]),
        ('No immunotherapy', cancer_test[cancer_test['io_any'].eq(0)]),
        ('Immunotherapy', cancer_test[cancer_test['io_any'].eq(1)]),
    ]
    km_dir = figdir / 'H3_treatment_KM_curves'
    km_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, subset in strata:
        row = performance_row(subset, label, 'CORI_score', 'CORI_high_risk') if len(subset) else {'cohort': label, 'N': 0, 'Events': 0}
        row['stratum'] = label
        row['status'] = 'analyzable' if row.get('N', 0) >= 30 and row.get('Events', 0) >= 5 else 'underpowered'
        rows.append(row)
        if row['status'] == 'analyzable':
            plot_km_high_low(subset, 'CORI_score', 'CORI_high_risk', f'H1 CORI high vs low — {label}', km_dir / f'H3_KM_{safe_name(label)}_high_low')
            if subset['CORI_risk_tertile'].nunique(dropna=True) >= 2:
                plot_km_tertiles(subset, 'CORI_score', 'CORI_risk_tertile', f'H1 CORI tertiles — {label}', km_dir / f'H3_KM_{safe_name(label)}_tertiles')
    stratified = pd.DataFrame(rows)
    stratified.to_csv(tabledir / 'H3_Table_03_treatment_stratified_performance.csv', index=False)
    forest_data = stratified.copy()
    forest_data['label'] = forest_data['stratum']
    forest_plot(forest_data.dropna(subset=['HR_High_vs_Low']), 'label', 'HR_High_vs_Low', 'HR_CI_Low', 'HR_CI_High', 'LogRank_p', 'N', 'Events', 'H1 CORI performance across treatment strata', figdir / 'H3_Fig01_treatment_forest')

    horizon_tables = []
    for label, subset in strata:
        table = safe_horizon_performance(
            subset,
            score_col='CORI_score',
            group_col='CORI_high_risk',
            horizon_cols=config.model.horizon_columns,
            label=label,
            tabledir=tabledir,
            prefix=f'H3_{safe_name(label)}',
            n_boot=300,
        )
        if len(table):
            horizon_tables.append(table)
    horizon = pd.concat(horizon_tables, ignore_index=True) if horizon_tables else pd.DataFrame()
    if not horizon.empty:
        horizon.to_csv(tabledir / 'H3_Table_04_treatment_horizon_performance_all.csv', index=False)

    cancer_train, cancer_test, z_parameters = zscore_from_train_apply(cancer_train, cancer_test, col='CORI_score', out_col='CORI_z')
    pd.DataFrame([{
        'score': 'CORI_score',
        'z_mean_from_development': z_parameters['mean'],
        'z_sd_from_development': z_parameters['sd'],
        'note': 'Held-out CORI_z uses development-set mean/SD only; CORI model itself is locked.',
    }]).to_csv(tabledir / 'H3_Table_treatment_adjustment_CORI_z_params.csv', index=False)
    models_to_run = [
        ('CORI adjusted \n Any systemic', 'CORI_z', ['treatment_any']),
        ('CORI adjusted \n Chemo & immuno', 'CORI_z', ['chemo_any', 'io_any']),
        ('CORI high-risk adjusted \n Any systemic', 'CORI_high_risk', ['treatment_any']),
        ('CORI high-risk adjusted \n Chemo & immuno', 'CORI_high_risk', ['chemo_any', 'io_any']),
    ]
    adjusted_tables = [
        fit_treatment_adjusted_cox(cancer_test, score_col=score, covars=covariates, label=label, tabledir=tabledir, prefix='H3', penalizer=0.05)[0]
        for label, score, covariates in models_to_run
    ]
    adjusted = pd.concat(adjusted_tables, ignore_index=True)
    adjusted.to_csv(tabledir / 'H3_Table_treatment_adjusted_CORI_models_all.csv', index=False)
    if 'variable' in adjusted.columns:
        cori_terms = adjusted[adjusted['variable'].isin(['CORI_z', 'CORI_high_risk'])].copy()
        if not cori_terms.empty:
            cori_terms['label'] = cori_terms['analysis']
            forest_plot(cori_terms, 'label', 'HR', 'HR_low', 'HR_high', 'p', 'N', 'Events', 'Subgroup Analysis after treatment adjustment', figdir / 'H3_Fig_treatment_adjusted_CORI_forest')

    clinical_covariates, age_column = one_age_covars(cancer_test, logger=logger)
    interaction_rows = []
    for flag, label in [
        ('treatment_any', 'H1 CORI x any systemic treatment'),
        ('chemo_any', 'H1 CORI x chemotherapy'),
        ('io_any', 'H1 CORI x immunotherapy'),
    ]:
        if flag in cancer_test.columns:
            interaction_rows.append(_interaction_or_status(cancer_test, 'CORI_score', flag, [], label + ' unadjusted'))
            interaction_rows.append(_interaction_or_status(cancer_test, 'CORI_score', flag, clinical_covariates, label + ' adjusted'))
    interactions = pd.DataFrame(interaction_rows)
    interactions.to_csv(tabledir / 'H3_Table_05_treatment_interaction_LRT.csv', index=False)

    manifest = {
        'notebook': 'H3',
        'purpose': 'Load H1 locked CORI and evaluate treatment strata; no CORI retraining.',
        'H1_CORI_bundle': str(config.cori_bundle),
        'treatment_files': [str(config.paths.treatment), str(config.paths.chemo)],
        'clinical_covariates_for_adjusted_LRT': clinical_covariates,
        'age_variable': age_column,
    }
    (outdir / 'H3_output_manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    return H3Result(cancer_train, cancer_test, stratified, horizon, adjusted, interactions)
