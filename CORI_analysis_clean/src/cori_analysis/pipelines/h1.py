from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tableone import TableOne

from ..clinical_data import load_clinical_status_exact, merge_htn_diabetes_status, one_age_covars
from ..clinical_models import train_clinical_and_stacked
from ..common import QCLogger, ensure_dirs
from ..config import AnalysisConfig
from ..evaluation import performance_row, predict_10yr, risk_tertiles_from_pred
from ..horizon import safe_horizon_performance
from ..plots import plot_km_high_low, plot_km_tertiles
from ..survival import train_locked_model
from ..visualization import plot_calibration_deciles, transition_heatmap
from ..plots import alluvial_static
from .common import SharedData


@dataclass
class H1Result:
    cancer_train: pd.DataFrame
    cancer_test: pd.DataFrame
    selected_features: list[str]
    feature_ranking: pd.DataFrame
    bundle: dict
    model_manifest: dict
    feature_count_cv: pd.DataFrame
    clinical_fits: dict
    clinical_comparison: pd.DataFrame
    clinical_covariates: list[str]
    age_column: str | None


def _baseline_characteristics(data: pd.DataFrame, tabledir) -> None:
    subtype_columns = [
        'DigestiveCancer_present', 'RespiCancer_present', 'BreastCancer_present',
        'FemRepoCancer_present', 'MaleRepoCancer_present', 'UrinaryTractCancer_present',
        'EndocrineCancer_present', 'HeamatoCancer_present', 'InsituCancer_present',
        'LipOralCancer_present', 'BoneCancer_present', 'SkinCancer_present',
        'MesotheliumCancer_present', 'EyeCNSCancer_present', 'SecondaryCancer_present',
        'UnknownCancer_present',
    ]
    columns = [
        'age_at_image_visit', 'sex', 'height', 'assessment_center_at_image_visit',
        'AnyCancer_present', 'MACE_in_allCancer_10yr_censored_status', *subtype_columns,
    ]
    categorical = [
        'sex', *subtype_columns, 'AnyCancer_present',
        'MACE_in_allCancer_10yr_censored_status', 'assessment_center_at_image_visit',
    ]
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise ValueError(f'Baseline table is missing configured columns: {missing}')
    table = TableOne(
        data,
        columns=columns,
        categorical=categorical,
        groupby='MACE_in_allCancer_10yr_censored_status',
        pval=True,
    )
    table.to_csv(tabledir / 'H1_Table_02_baseline_characteristics.csv')


def run_h1(shared: SharedData, config: AnalysisConfig) -> H1Result:
    outdir, figdir, tabledir, modeldir, qcdir = ensure_dirs(config.h1_dir)
    logger = QCLogger(qcdir / 'H1_QC_log.txt')
    logger.section('H1: locked pan-cancer CORI')
    _baseline_characteristics(shared.cancer_df, tabledir)

    cancer_train = shared.cancer_train.copy()
    cancer_test = shared.cancer_test.copy()
    cancer_train, cancer_test, selected, ranking, bundle, model_manifest, feature_cv = train_locked_model(
        cancer_train,
        cancer_test,
        shared.feature_cols,
        prefix='CORI',
        outdir=tabledir,
        modeldir=modeldir,
        logger=logger,
        variance_threshold=config.model.variance_threshold,
        max_missing=config.model.max_feature_missing,
        candidate_ks=config.model.candidate_feature_counts,
        cox_rank_penalizer=config.model.univariate_cox_penalizer,
    )
    risk_columns = [
        column for column in [
            'eid', shared.center_col, 'time', 'event', 'CORI_score',
            'CORI_high_risk', 'CORI_risk_tertile',
        ] if column in cancer_train.columns
    ]
    pd.concat(
        [
            cancer_train[risk_columns].assign(set='development'),
            cancer_test[risk_columns].assign(set='held_out'),
        ],
        ignore_index=True,
    ).to_csv(tabledir / 'H1_CORI_risk_scores_all_sets.csv', index=False)

    performance = pd.DataFrame([
        performance_row(cancer_train.loc[:, ~cancer_train.columns.duplicated()], 'Development', 'CORI_score', 'CORI_high_risk'),
        performance_row(cancer_test.loc[:, ~cancer_test.columns.duplicated()], 'Held-out centers', 'CORI_score', 'CORI_high_risk'),
    ])
    performance.to_csv(tabledir / 'H1_Table_02_primary_CORI_performance_10yr.csv', index=False)
    logger.df('Primary CORI performance', performance)
    plot_km_high_low(cancer_test, 'CORI_score', 'CORI_high_risk', 'CORI high vs low risk — held-out centers', figdir / 'H1_Fig01_KM_CORI_high_low_heldout')
    plot_km_tertiles(cancer_test, 'CORI_score', 'CORI_risk_tertile', 'CORI tertiles — held-out centers', figdir / 'H1_Fig02_KM_CORI_tertiles_heldout')
    safe_horizon_performance(
        cancer_test,
        score_col='CORI_score',
        group_col='CORI_high_risk',
        horizon_cols=config.model.horizon_columns,
        label='Held-out centers',
        tabledir=tabledir,
        prefix='H1_Table_03_CORI',
        n_boot=300,
    )

    clinical_table = load_clinical_status_exact(
        config.paths.clinical,
        sex_col=config.columns.clinical_sex,
        diabetes_col=config.columns.clinical_diabetes,
        htn_col=config.columns.clinical_htn,
    )
    cancer_train = merge_htn_diabetes_status(cancer_train, clinical_table)
    cancer_test = merge_htn_diabetes_status(cancer_test, clinical_table)
    clinical_covariates, age_column = one_age_covars(cancer_train, logger=logger)
    clinical_fits, clinical_comparison = train_clinical_and_stacked(
        cancer_train,
        cancer_test,
        'CORI_score',
        clinical_covariates,
        logger,
        tabledir,
    )
    clinical_comparison.to_csv(tabledir / 'H1_Table_04_clinical_CORI_stacked_model_comparison.csv', index=False)
    coefficient_tables = []
    for model_name, fit in clinical_fits.items():
        summary = fit['model'].summary.reset_index().rename(columns={'covariate': 'predictor', 'index': 'predictor'})
        summary.insert(0, 'model', model_name)
        coefficient_tables.append(summary)
    pd.concat(coefficient_tables, ignore_index=True).to_csv(tabledir / 'H1_Table_05_multivariable_model_coefficients.csv', index=False)

    clinical_fit = clinical_fits['Clinical']
    stacked_fit = clinical_fits['Stacked clinical-risk + CORI-risk']
    clinical_test = clinical_fit['test_matrix'].copy()
    stacked_test = stacked_fit['test_matrix'].copy()
    clinical_test['pred10_clinical'] = predict_10yr(clinical_fit['model'], clinical_test)
    stacked_test['pred10_stacked'] = predict_10yr(stacked_fit['model'], stacked_test)
    reclassification = pd.DataFrame({
        'event': stacked_test['event'].values,
        'pred10_stacked': stacked_test['pred10_stacked'].values,
        'pred10_clinical': pd.Series(clinical_test['pred10_clinical'].values, index=clinical_test.index).reindex(stacked_test.index).values,
    }).dropna()
    reclassification['clinical_pred_tertile'] = risk_tertiles_from_pred(reclassification['pred10_clinical'])
    reclassification['stacked_pred_tertile'] = risk_tertiles_from_pred(reclassification['pred10_stacked'])
    counts, event_rates, mean_predictions = transition_heatmap(
        reclassification,
        'clinical_pred_tertile',
        'stacked_pred_tertile',
        'event',
        'pred10_stacked',
        'Predicted-risk reclassification: clinical → clinical+CORI',
        figdir / 'H1_Fig03_reclassification_matrix_predicted_risk',
    )
    counts.to_csv(tabledir / 'H1_Table_06_reclassification_counts.csv')
    event_rates.to_csv(tabledir / 'H1_Table_07_reclassification_event_rates.csv')
    mean_predictions.to_csv(tabledir / 'H1_Table_08_reclassification_mean_predicted_risk.csv')
    alluvial_static(
        reclassification,
        'clinical_pred_tertile',
        'stacked_pred_tertile',
        'Clinical → clinical+CORI predicted-risk strata',
        figdir / 'H1_Fig04_alluvial_clinical_to_CORI',
    )
    reclassification['decile'] = pd.qcut(reclassification['pred10_stacked'].rank(method='first'), 10, labels=False) + 1
    calibration = reclassification.groupby('decile').agg(
        N=('event', 'size'),
        Events=('event', 'sum'),
        Observed_rate=('event', 'mean'),
        Mean_predicted_risk=('pred10_stacked', 'mean'),
    ).reset_index()
    calibration.to_csv(tabledir / 'H1_Table_09_stacked_model_calibration_deciles.csv', index=False)
    plot_calibration_deciles(calibration, figdir / 'H1_Fig05_calibration_stacked_deciles')

    manifest = {
        'notebook': 'H1',
        'purpose': 'Train locked pan-cancer CORI model once; downstream analyses load it without retraining.',
        'center_col': shared.center_col,
        'train_centers_requested': config.model.train_centers,
        'development_N': len(cancer_train),
        'development_events': int(cancer_train.event.sum()),
        'heldout_N': len(cancer_test),
        'heldout_events': int(cancer_test.event.sum()),
        'feature_selection': model_manifest,
        'clinical_covariates': clinical_covariates,
        'age_variable': age_column,
        'locked_model_bundle': str(config.cori_bundle),
    }
    (outdir / 'H1_output_manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    return H1Result(
        cancer_train=cancer_train,
        cancer_test=cancer_test,
        selected_features=selected,
        feature_ranking=ranking,
        bundle=bundle,
        model_manifest=model_manifest,
        feature_count_cv=feature_cv,
        clinical_fits=clinical_fits,
        clinical_comparison=clinical_comparison,
        clinical_covariates=clinical_covariates,
        age_column=age_column,
    )
