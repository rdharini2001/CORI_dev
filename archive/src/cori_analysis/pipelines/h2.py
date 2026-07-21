from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from ..clinical_data import one_age_covars
from ..common import QCLogger, ensure_dirs, safe_name
from ..config import AnalysisConfig
from ..evaluation import lrt_interaction, performance_row
from ..plots import alluvial_static, forest_plot, plot_km_high_low, plot_km_tertiles
from ..rank_transition import feature_rank_shift_bump_FIXED
from ..residualization import residualize_against_mmace
from ..survival import (
    apply_model_bundle,
    apply_thresholds_from_dict,
    derive_thresholds,
    load_model_bundle,
    train_locked_model,
)
from ..visualization import transition_heatmap
from .common import SharedData
from .h1 import H1Result


@dataclass
class H2Result:
    cancer_train: pd.DataFrame
    cancer_test: pd.DataFrame
    noncancer_train: pd.DataFrame
    noncancer_test: pd.DataFrame
    mmace_features: list[str]
    mmace_ranking: pd.DataFrame
    mmace_bundle: dict
    mmace_manifest: dict
    mmace_cv: pd.DataFrame
    clinical_covariates: list[str]
    age_column: str | None


def run_h2(shared: SharedData, h1: H1Result, config: AnalysisConfig) -> H2Result:
    run_dir = config.paths.output_dir / 'H2_MCORI_MMACE_locked_CORI_v13'
    outdir, figdir, tabledir, modeldir, qcdir = ensure_dirs(run_dir)
    logger = QCLogger(qcdir / 'H2_QC_log.txt')
    logger.section('H2: locked CORI versus non-cancer-trained M-MACE')
    if not config.cori_bundle.exists():
        raise FileNotFoundError(f'Missing H1 locked CORI bundle: {config.cori_bundle}')

    cancer_train = shared.cancer_train.copy()
    cancer_test = shared.cancer_test.copy()
    noncancer_train = shared.noncancer_train.copy()
    noncancer_test = shared.noncancer_test.copy()
    shared.cancer_audit.to_csv(tabledir / 'H2_Table_00A_cancer_audit.csv', index=False)
    shared.noncancer_audit.to_csv(tabledir / 'H2_Table_00B_noncancer_audit.csv', index=False)
    shared.cancer_split.to_csv(tabledir / 'H2_Table_01A_cancer_center_split.csv', index=False)
    shared.noncancer_split.to_csv(tabledir / 'H2_Table_01B_noncancer_center_split.csv', index=False)

    cori_bundle = load_model_bundle(config.cori_bundle)
    cancer_train = apply_model_bundle(cori_bundle, cancer_train, score_col='CORI_score', prefix='CORI', logger=logger)
    cancer_test = apply_model_bundle(cori_bundle, cancer_test, score_col='CORI_score', prefix='CORI', logger=logger)
    noncancer_train = apply_model_bundle(cori_bundle, noncancer_train, score_col='CORI_score', prefix='CORI', logger=logger)
    noncancer_test = apply_model_bundle(cori_bundle, noncancer_test, score_col='CORI_score', prefix='CORI', logger=logger)

    noncancer_train, noncancer_test, mmace_features, mmace_ranking, mmace_bundle, mmace_manifest, mmace_cv = train_locked_model(
        noncancer_train,
        noncancer_test,
        shared.feature_cols_noncancer,
        prefix='MMACE',
        outdir=tabledir,
        modeldir=modeldir,
        logger=logger,
        variance_threshold=config.model.variance_threshold,
        max_missing=config.model.max_feature_missing,
        candidate_ks=config.model.candidate_feature_counts,
        cox_rank_penalizer=config.model.univariate_cox_penalizer,
    )
    cancer_train = apply_model_bundle(mmace_bundle, cancer_train, score_col='MMACE_score', prefix='MMACE', logger=logger)
    cancer_test = apply_model_bundle(mmace_bundle, cancer_test, score_col='MMACE_score', prefix='MMACE', logger=logger)

    performance_rows = []
    cohorts = [
        ('Cancer development\n', cancer_train),
        ('Cancer held-out\n', cancer_test),
        ('Non-cancer development\n', noncancer_train),
        ('Non-cancer held-out\n', noncancer_test),
    ]
    models = [
        ('H1 CORI \n cancer-trained', 'CORI_score', 'CORI_high_risk'),
        ('M-MACE \n non-cancer-trained', 'MMACE_score', 'MMACE_high_risk'),
    ]
    for cohort_label, cohort in cohorts:
        for model_label, score, group in models:
            if score in cohort.columns and group in cohort.columns and cohort[score].notna().sum() > 20:
                row = performance_row(cohort, cohort_label + ' | ' + model_label, score, group)
                row.update({'cohort': cohort_label, 'model': model_label})
                performance_rows.append(row)
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(tabledir / 'H2_Table_02_cross_prediction_performance.csv', index=False)
    held_out = performance[performance['cohort'].str.contains('held-out', case=False, na=False)].copy()
    held_out['label'] = held_out['cohort'] + ' | ' + held_out['model']
    forest_plot(
        held_out,
        'label',
        'HR_High_vs_Low',
        'HR_CI_Low',
        'HR_CI_High',
        'LogRank_p',
        'N',
        'Events',
        'Held-out cross-prediction: H1 CORI vs M-MACE',
        figdir / 'H2_Fig01_cross_prediction_forest',
    )
    for cohort_label, cohort in [('Cancer held-out', cancer_test), ('Non-cancer held-out', noncancer_test)]:
        for model_label, score, group, tertile in [
            ('CORI', 'CORI_score', 'CORI_high_risk', 'CORI_risk_tertile'),
            ('M-MACE', 'MMACE_score', 'MMACE_high_risk', 'MMACE_risk_tertile'),
        ]:
            plot_km_high_low(cohort, score, group, f'{cohort_label}: {model_label} high vs low', figdir / f'H2_KM_{safe_name(cohort_label)}_{model_label}_highlow')
            plot_km_tertiles(cohort, score, tertile, f'{cohort_label}: {model_label} tertiles', figdir / f'H2_KM_{safe_name(cohort_label)}_{model_label}_tertiles')

    pooled = pd.concat(
        [
            cancer_test.assign(cancer_status=1, cohort='Cancer'),
            noncancer_test.assign(cancer_status=0, cohort='Non-cancer'),
        ],
        ignore_index=True,
    )
    clinical_covariates, age_column = one_age_covars(pooled, logger=logger)
    interaction_rows = []
    for score, label in [('CORI_score', 'H1 CORI x cancer status'), ('MMACE_score', 'M-MACE x cancer status')]:
        interaction_rows.append(lrt_interaction(pooled, score, 'cancer_status', covars=[], label=label + ' unadjusted'))
        interaction_rows.append(lrt_interaction(pooled, score, 'cancer_status', covars=clinical_covariates, label=label + ' adjusted'))
    pd.DataFrame(interaction_rows).to_csv(tabledir / 'H2_Table_03_cancer_status_interaction_LRT.csv', index=False)

    _, residual_model = residualize_against_mmace(cancer_train)
    logger.log(
        f"CORI ~ MMACE_score: R^2={residual_model.rsquared:.4f}, "
        f"beta={residual_model.params['MMACE_score']:.4f}, "
        f"p={residual_model.pvalues['MMACE_score']:.3g}"
    )
    cancer_train['CORI_resid_MMACE'], _ = residualize_against_mmace(cancer_train, fit_on=cancer_train)
    cancer_test['CORI_resid_MMACE'], _ = residualize_against_mmace(cancer_test, fit_on=cancer_train)
    residual_thresholds = derive_thresholds(cancer_train['CORI_resid_MMACE'])
    cancer_train = apply_thresholds_from_dict(cancer_train, 'CORI_resid_MMACE', 'CORIresidMMACE', residual_thresholds)
    cancer_test = apply_thresholds_from_dict(cancer_test, 'CORI_resid_MMACE', 'CORIresidMMACE', residual_thresholds)
    residual_performance = pd.DataFrame([
        performance_row(cancer_test, 'Held-out centers | CORI (raw)', 'CORI_score', 'CORI_high_risk'),
        performance_row(cancer_test, 'Held-out centers | CORI resid. M-MACE', 'CORI_resid_MMACE', 'CORIresidMMACE_high_risk'),
    ])
    residual_performance.to_csv(tabledir / 'H2_Table_04_CORI_residualized_MMACE_performance.csv', index=False)
    plot_km_high_low(cancer_test, 'CORI_resid_MMACE', 'CORIresidMMACE_high_risk', 'CORI (residualized against M-MACE) high vs low — held-out centers', figdir / 'H2_Fig02_KM_CORIresidMMACE_high_low_heldout')
    plot_km_tertiles(cancer_test, 'CORI_resid_MMACE', 'CORIresidMMACE_risk_tertile', 'CORI (residualized against M-MACE) tertiles — held-out centers', figdir / 'H2_Fig03_KM_CORIresidMMACE_tertiles_heldout')

    for cohort_label, cohort in [('Cancer held-out', cancer_test), ('Non-cancer held-out', noncancer_test)]:
        reclassification = cohort[['event', 'MMACE_risk_tertile', 'CORI_risk_tertile', 'MMACE_score', 'CORI_score']].dropna().copy()
        if reclassification.empty:
            continue
        counts, event_rates, mean_scores = transition_heatmap(
            reclassification,
            'MMACE_risk_tertile',
            'CORI_risk_tertile',
            'event',
            'CORI_score',
            f'M-MACE → H1 CORI predicted strata ({cohort_label})',
            figdir / f'H2_reclassification_{safe_name(cohort_label)}',
        )
        counts.to_csv(tabledir / f'H2_reclassification_counts_{safe_name(cohort_label)}.csv')
        event_rates.to_csv(tabledir / f'H2_reclassification_event_rates_{safe_name(cohort_label)}.csv')
        mean_scores.to_csv(tabledir / f'H2_reclassification_mean_CORI_score_{safe_name(cohort_label)}.csv')
        alluvial_static(reclassification, 'MMACE_risk_tertile', 'CORI_risk_tertile', f'M-MACE → H1 CORI strata ({cohort_label})', figdir / f'H2_alluvial_{safe_name(cohort_label)}')

    feature_rank_shift_bump_FIXED(h1.feature_ranking, mmace_ranking, figdir, tabledir, top_n=30, prefix='H2')
    manifest = {
        'notebook': 'H2',
        'purpose': 'Load H1 locked CORI; train only M-MACE; compare cancer-trained and non-cancer-trained retinal signatures.',
        'H1_CORI_bundle': str(config.cori_bundle),
        'center_col': shared.center_col,
        'cancer_development_N': len(cancer_train),
        'cancer_heldout_N': len(cancer_test),
        'noncancer_development_N': len(noncancer_train),
        'noncancer_heldout_N': len(noncancer_test),
        'MMACE_manifest': mmace_manifest,
        'clinical_covariates_for_LRT': clinical_covariates,
        'age_variable': age_column,
    }
    (outdir / 'H2_output_manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    return H2Result(
        cancer_train=cancer_train,
        cancer_test=cancer_test,
        noncancer_train=noncancer_train,
        noncancer_test=noncancer_test,
        mmace_features=mmace_features,
        mmace_ranking=mmace_ranking,
        mmace_bundle=mmace_bundle,
        mmace_manifest=mmace_manifest,
        mmace_cv=mmace_cv,
        clinical_covariates=clinical_covariates,
        age_column=age_column,
    )
