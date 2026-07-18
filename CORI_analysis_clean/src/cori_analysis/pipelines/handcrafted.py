from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import clean_id
from ..config import AnalysisConfig
from ..handcrafted import (
    apply_handcrafted_bundle,
    hc_performance_row,
    load_handcrafted_features,
    merge_handcrafted_cached_safe,
    plot_handcrafted_feature_transition_clean,
    plot_hc_km,
    set_logger,
    train_handcrafted_survival_model,
)
from ..survival import apply_model_bundle, load_model_bundle
from .common import SharedData
from .h1 import H1Result
from .h2 import H2Result
from .h4 import H4Result


HCORI_LOCKED_FEATURES = [
    'HC_vessel_fundus_ring_125_175STD_branch_length',
    'HC_vessel_fundus_ring_125_175STD_branch_distance',
    'HC_vessel_fundus_ring_125_175pixel_degree_3',
    'HC_vein_fundus_ring_125_175total_connected_pixels',
    'HC_vein_fundus_ring_125_175pixel_degree_2',
]


@dataclass
class HCORIResult:
    cancer_train: pd.DataFrame
    cancer_test: pd.DataFrame
    cached_features: pd.DataFrame
    selected_features: list[str]
    ranking: pd.DataFrame
    bundle: dict
    cv: pd.DataFrame
    performance: pd.DataFrame


@dataclass
class HMMACEResult:
    cancer_test: pd.DataFrame
    noncancer_test: pd.DataFrame
    selected_features: list[str]
    ranking: pd.DataFrame
    bundle: dict
    cv: pd.DataFrame
    performance: pd.DataFrame


def _load_or_build_cache(config: AnalysisConfig, rebuild: bool) -> pd.DataFrame:
    cache = config.handcrafted_cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not rebuild:
        features = pd.read_csv(cache, low_memory=False)
    else:
        features, errors = load_handcrafted_features(config.paths.handcrafted_feature_dir, id_col_name='eid')
        features.to_csv(cache, index=False)
        if not errors.empty:
            errors.to_csv(cache.parent / 'H1_handcrafted_feature_read_errors.csv', index=False)
    if 'eid' not in features.columns:
        raise ValueError(f'Handcrafted cache must contain eid: {cache}')
    features['eid'] = clean_id(features['eid'])
    return features.drop_duplicates('eid', keep='first')


def _merge_exact(cohort: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    left = cohort.copy()
    right = features.copy()
    left['eid'] = clean_id(left['eid'])
    right['eid'] = clean_id(right['eid'])
    overlap = [column for column in right.columns if column != 'eid' and column in left.columns]
    if overlap:
        left = left.drop(columns=overlap)
    return left.merge(right, on='eid', how='left', validate='m:1')


def run_hcori(
    h1: H1Result,
    config: AnalysisConfig,
    *,
    rebuild_cache: bool = False,
    max_features_for_ranking: int = 75,
    max_missing: float = 0.50,
    min_variance: float = 1e-8,
    min_nonmissing_train: int = 50,
) -> HCORIResult:
    from ..common import QCLogger

    output = config.hcori_dir
    tabledir = output / 'tables'
    modeldir = output / 'models'
    figdir = output / 'figures'
    qcdir = output / 'qc'
    for directory in [output, tabledir, modeldir, figdir, qcdir]:
        directory.mkdir(parents=True, exist_ok=True)
    logger = QCLogger(qcdir / 'H1_HCORI_QC_log.txt')
    set_logger(logger)

    features = _load_or_build_cache(config, rebuild_cache)
    cancer_train = _merge_exact(h1.cancer_train, features)
    cancer_test = _merge_exact(h1.cancer_test, features)
    handcrafted_columns = [column for column in cancer_train.columns if column.startswith('HC_')]
    merge_qc = pd.DataFrame([
        {
            'set': 'development',
            'N': len(cancer_train),
            'with_any_HC': int(cancer_train[handcrafted_columns].notna().any(axis=1).sum()) if handcrafted_columns else 0,
            'n_HC_columns': len(handcrafted_columns),
            'id_col': 'eid',
        },
        {
            'set': 'held_out',
            'N': len(cancer_test),
            'with_any_HC': int(cancer_test[handcrafted_columns].notna().any(axis=1).sum()) if handcrafted_columns else 0,
            'n_HC_columns': len(handcrafted_columns),
            'id_col': 'eid',
        },
    ])
    merge_qc.to_csv(tabledir / 'H1_handcrafted_merge_QC.csv', index=False)

    prefilter_rows = []
    candidates = []
    for column in handcrafted_columns:
        values = pd.to_numeric(cancer_train[column], errors='coerce')
        nonmissing = int(values.notna().sum())
        missing_fraction = float(values.isna().mean())
        variance = float(values.var(skipna=True)) if nonmissing > 1 else np.nan
        keep = (
            nonmissing >= min_nonmissing_train
            and missing_fraction <= max_missing
            and np.isfinite(variance)
            and variance > min_variance
        )
        prefilter_rows.append({
            'feature': column,
            'n_nonmissing_train': nonmissing,
            'missing_fraction_train': missing_fraction,
            'variance_train': variance,
            'status': 'keep' if keep else 'drop',
        })
        if keep:
            candidates.append(column)
    pd.DataFrame(prefilter_rows).to_csv(tabledir / 'H1_handcrafted_prefilter_QC.csv', index=False)

    event = pd.to_numeric(cancer_train['event'], errors='coerce').fillna(0).astype(int)
    rank_rows = []
    for column in candidates:
        values = pd.to_numeric(cancer_train[column], errors='coerce')
        non_events = values[event.eq(0)].dropna()
        events = values[event.eq(1)].dropna()
        score = np.nan
        if len(non_events) >= 20 and len(events) >= 5:
            pooled = np.sqrt((non_events.var(ddof=1) + events.var(ddof=1)) / 2)
            if pooled and np.isfinite(pooled):
                score = abs((events.mean() - non_events.mean()) / pooled)
        rank_rows.append({'feature': column, 'fast_abs_standardized_difference': score})
    fast_rank = pd.DataFrame(rank_rows).sort_values('fast_abs_standardized_difference', ascending=False, na_position='last')
    fast_rank.to_csv(tabledir / 'H1_handcrafted_fast_univariate_prefilter_rank.csv', index=False)
    selected_for_modeling = fast_rank['feature'].dropna().head(max_features_for_ranking).tolist()
    pd.DataFrame({'HC_feature_used_for_modeling': selected_for_modeling}).to_csv(tabledir / 'H1_handcrafted_features_passed_to_survival_model.csv', index=False)

    metadata_columns = [column for column in cancer_train.columns if not column.startswith('HC_')]
    train_fast = cancer_train[metadata_columns + selected_for_modeling].copy()
    test_fast = cancer_test[[column for column in cancer_test.columns if not column.startswith('HC_')] + selected_for_modeling].copy()
    train_fast, test_fast, selected, ranking, bundle, cv = train_handcrafted_survival_model(
        train_fast,
        test_fast,
        prefix='HCORI',
        outdir=output,
    )
    for column in ['HCORI_score', 'HCORI_high_risk', 'HCORI_risk_tertile']:
        cancer_train[column] = train_fast[column].values
        cancer_test[column] = test_fast[column].values
    performance_rows = [
        hc_performance_row(cancer_train, 'Development', 'HCORI_score', 'HCORI_high_risk'),
        hc_performance_row(cancer_test, 'Held-out centers', 'HCORI_score', 'HCORI_high_risk'),
    ]
    if 'CORI_score' in cancer_test.columns:
        performance_rows.append(hc_performance_row(cancer_test, 'Held-out centers', 'CORI_score', 'CORI_high_risk' if 'CORI_high_risk' in cancer_test.columns else None))
    performance = pd.DataFrame(performance_rows)
    performance.to_csv(tabledir / 'H1_handcrafted_vs_retfound_performance.csv', index=False)
    plot_hc_km(cancer_test, 'HCORI_high_risk', 'HCORI high vs low risk — held-out centers', figdir / 'H1_HCORI_KM_high_low_heldout')
    return HCORIResult(cancer_train, cancer_test, features, selected, ranking, bundle, cv, performance)


def run_hmmace(h2: H2Result, hcori: HCORIResult, config: AnalysisConfig) -> HMMACEResult:
    output = config.paths.output_dir / 'H2_MCORI_MMACE_locked_CORI_v13' / 'handcrafted_HMMACE'
    tabledir = output / 'tables'
    modeldir = output / 'models'
    figdir = output / 'figures'
    for directory in [output, tabledir, modeldir, figdir]:
        directory.mkdir(parents=True, exist_ok=True)

    features = hcori.cached_features
    cancer_train, cancer_train_qc = merge_handcrafted_cached_safe(h2.cancer_train, features, cohort_name='cancer_train')
    cancer_test, cancer_test_qc = merge_handcrafted_cached_safe(h2.cancer_test, features, cohort_name='cancer_test')
    noncancer_train, noncancer_train_qc = merge_handcrafted_cached_safe(h2.noncancer_train, features, cohort_name='noncancer_train')
    noncancer_test, noncancer_test_qc = merge_handcrafted_cached_safe(h2.noncancer_test, features, cohort_name='noncancer_test')
    pd.concat([cancer_train_qc, cancer_test_qc, noncancer_train_qc, noncancer_test_qc], ignore_index=True).to_csv(tabledir / 'H2_handcrafted_cache_merge_QC.csv', index=False)

    if not config.hcori_bundle.exists():
        raise FileNotFoundError(f'Missing locked HCORI bundle: {config.hcori_bundle}')
    cancer_test = apply_handcrafted_bundle(config.hcori_bundle, cancer_test, prefix='HCORI')
    noncancer_test = apply_handcrafted_bundle(config.hcori_bundle, noncancer_test, prefix='HCORI')
    noncancer_train, noncancer_test, selected, ranking, bundle, cv = train_handcrafted_survival_model(
        noncancer_train,
        noncancer_test,
        prefix='HMMACE',
        outdir=output,
    )
    bundle_path = modeldir / 'HMMACE_locked_handcrafted_model_bundle.pkl'
    cancer_test = apply_handcrafted_bundle(bundle_path, cancer_test, prefix='HMMACE')

    rows = []
    for cohort_label, cohort in [('Cancer held-out', cancer_test), ('Non-cancer held-out', noncancer_test)]:
        for model_label, score, group in [
            ('HCORI cancer-trained handcrafted', 'HCORI_score', 'HCORI_high_risk'),
            ('H-M-MACE non-cancer-trained handcrafted', 'HMMACE_score', 'HMMACE_high_risk'),
        ]:
            if score in cohort.columns:
                row = hc_performance_row(cohort, cohort_label, score, group if group in cohort.columns else None)
                row.update({'model': model_label, 'score_col': score, 'group_col': group if group in cohort.columns else None})
                rows.append(row)
    performance = pd.DataFrame(rows)
    performance.to_csv(tabledir / 'H2_handcrafted_HCORI_vs_HMMACE_cross_prediction.csv', index=False)
    plot_hc_km(cancer_test, 'HCORI_high_risk', 'Cancer held-out: HCORI high vs low', figdir / 'H2_KM_cancer_HCORI')
    plot_hc_km(cancer_test, 'HMMACE_high_risk', 'Cancer held-out: H-M-MACE high vs low', figdir / 'H2_KM_cancer_HMMACE')
    plot_hc_km(noncancer_test, 'HCORI_high_risk', 'Non-cancer held-out: HCORI high vs low', figdir / 'H2_KM_noncancer_HCORI')
    plot_hc_km(noncancer_test, 'HMMACE_high_risk', 'Non-cancer held-out: H-M-MACE high vs low', figdir / 'H2_KM_noncancer_HMMACE')

    transition_selected, transition_full, transition_overlap = plot_handcrafted_feature_transition_clean(
        hcori.ranking,
        ranking,
        figdir=figdir,
        tabledir=tabledir,
        left_label='HCORI',
        right_label='HMMACE',
        top_n=20,
        top_each_side=10,
        prefix='H2_HC',
    )
    transition_selected.to_csv(tabledir / 'H2_handcrafted_feature_transition_CLEAN_selected.csv', index=False)
    transition_full.to_csv(tabledir / 'H2_handcrafted_feature_transition_FULL.csv', index=False)
    transition_overlap.to_csv(tabledir / 'H2_handcrafted_feature_overlap_by_topk.csv', index=False)
    manifest = {
        'analysis': 'H2 handcrafted HCORI vs H-M-MACE',
        'feature_source': str(config.handcrafted_cache),
        'HCORI_bundle_path': str(config.hcori_bundle),
        'HMMACE_bundle_path': str(bundle_path),
        'n_HMMACE_selected_features': len(selected),
        'HMMACE_selected_features': selected,
    }
    (tabledir / 'H2_handcrafted_cache_based_manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    return HMMACEResult(cancer_test, noncancer_test, selected, ranking, bundle, cv, performance)


def apply_hcori_to_cmr(h4: H4Result, config: AnalysisConfig) -> pd.DataFrame:
    if not config.hcori_bundle.exists():
        raise FileNotFoundError(f'Missing locked HCORI bundle: {config.hcori_bundle}')
    features = pd.read_csv(config.handcrafted_cache, low_memory=False)
    if 'eid' not in features.columns:
        raise ValueError(f'Handcrafted cache must contain eid: {config.handcrafted_cache}')
    missing = [column for column in HCORI_LOCKED_FEATURES if column not in features.columns]
    if missing:
        raise ValueError(f'Handcrafted cache is missing locked HCORI features: {missing}')
    features['eid'] = clean_id(features['eid'])
    cmr = h4.cmr_merged.copy()
    cmr['eid'] = clean_id(cmr['eid'])
    cmr = cmr.drop(columns=HCORI_LOCKED_FEATURES, errors='ignore').merge(
        features[['eid'] + HCORI_LOCKED_FEATURES].drop_duplicates('eid'),
        on='eid',
        how='left',
        validate='m:1',
    )
    if not cmr[HCORI_LOCKED_FEATURES].notna().any(axis=1).any():
        raise ValueError('No eid overlap between CMR rows and cached handcrafted features.')
    bundle = load_model_bundle(config.hcori_bundle)
    return apply_model_bundle(bundle, cmr, score_col='HCORI_score', prefix='HCORI')
