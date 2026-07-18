from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..clinical_data import one_age_covars
from ..cmr import CMR_FEATURE_FAMILIES, build_cmr_variable_inventory
from ..cohorts import reshape_cmr_long
from ..common import QCLogger, clean_id, ensure_dirs, load_csv
from ..config import AnalysisConfig
from ..evaluation import adjusted_linear_regressions, cmr_analysis_table
from ..survival import apply_model_bundle, load_model_bundle
from ..visualization import (
    cmr_cluster_map,
    cmr_violin_grid,
    plot_adjusted_beta_forest,
    plot_cmr_family_counts,
)
from .common import SharedData


@dataclass
class H4Result:
    cmr_long: pd.DataFrame
    cmr_merged: pd.DataFrame
    primary_cmr: pd.DataFrame
    inventory: pd.DataFrame
    selected_features: list[str]
    association_table: pd.DataFrame
    regression_table: pd.DataFrame


def run_h4(shared: SharedData, config: AnalysisConfig) -> H4Result:
    run_dir = config.paths.output_dir / 'H4_CMR_locked_CORI_v13'
    outdir, figdir, tabledir, _, qcdir = ensure_dirs(run_dir)
    logger = QCLogger(qcdir / 'H4_QC_log.txt')
    logger.section('H4: CMR phenotyping with locked CORI')
    if not config.cori_bundle.exists():
        raise FileNotFoundError(f'Missing H1 locked CORI bundle: {config.cori_bundle}')
    if not config.paths.cardiac_mri.exists():
        raise FileNotFoundError(f'Configured CMR file not found: {config.paths.cardiac_mri}')

    cancer_train = shared.cancer_train.copy()
    cancer_test = shared.cancer_test.copy()
    shared.cancer_audit.to_csv(tabledir / 'H4_Table_00_cohort_audit.csv', index=False)
    shared.cancer_split.to_csv(tabledir / 'H4_Table_01_center_split.csv', index=False)
    bundle = load_model_bundle(config.cori_bundle)
    cancer_train = apply_model_bundle(bundle, cancer_train, score_col='CORI_score', prefix='CORI', logger=logger)
    cancer_test = apply_model_bundle(bundle, cancer_test, score_col='CORI_score', prefix='CORI', logger=logger)
    score_df = pd.concat([
        cancer_train.assign(split='development'),
        cancer_test.assign(split='held_out'),
    ], ignore_index=True)

    cmr = load_csv(config.paths.cardiac_mri, logger)
    if 'eid' not in cmr.columns:
        raise ValueError('CMR file must contain eid column.')
    cmr['eid'] = clean_id(cmr['eid'])
    cmr_long = reshape_cmr_long(cmr, instances=(2, 3))
    cmr_long.to_csv(tabledir / 'H4_Table_02_CMR_long_visit_aware.csv', index=False)
    merge_columns = [column for column in ['eid', 'time', 'event', 'CORI_score', 'CORI_high_risk', 'CORI_risk_tertile', 'split'] if column in score_df.columns]
    cmr_merged = cmr_long.merge(score_df[merge_columns], on='eid', how='inner')
    cmr_merged.to_csv(tabledir / 'H4_Table_03_CORI_CMR_merged.csv', index=False)
    merge_summary = cmr_merged.groupby(['split', 'cmr_instance'], dropna=False).agg(
        N=('eid', 'nunique'), Rows=('eid', 'size'), Events=('event', 'sum')
    ).reset_index()
    merge_summary.to_csv(tabledir / 'H4_Table_04_CMR_merge_summary.csv', index=False)
    primary = cmr_merged[cmr_merged['split'].eq('held_out')].copy()
    if len(primary) < 30:
        logger.log('Held-out CMR sample too small; using all scored CMR participants.')
        primary = cmr_merged.copy()

    inventory = build_cmr_variable_inventory(
        primary,
        CMR_FEATURE_FAMILIES,
        exact_variables=config.cmr.exact_variables,
        user_keywords=config.cmr.keywords,
        min_nonmissing=config.cmr.min_nonmissing,
        min_unique=config.cmr.min_unique_values,
        primary_cmr_families=config.cmr.primary_families,
    )
    inventory.to_csv(tabledir / 'H4_Table_05_CMR_variable_inventory_curated.csv', index=False)
    selected = inventory.loc[inventory['selected'], 'column'].tolist()
    family_table = inventory[inventory['selected']].groupby('family').agg(
        n_features=('column', 'count'),
        median_nonmissing=('nonmissing', 'median'),
        min_nonmissing=('nonmissing', 'min'),
        max_nonmissing=('nonmissing', 'max'),
    ).reset_index().sort_values('n_features', ascending=False)
    family_table.to_csv(tabledir / 'H4_Table_06_selected_CMR_feature_families.csv', index=False)
    if not selected:
        raise ValueError('No CMR variables passed the configured curated-family and QC rules.')

    associations = cmr_analysis_table(primary, 'CORI_score', 'CORI_high_risk', selected, tabledir, prefix='H4_curated')
    if not associations.empty:
        family_map = inventory[['column', 'clean_label', 'family']].rename(columns={'column': 'feature'})
        associations = associations.merge(family_map, on='feature', how='left').sort_values(['Spearman_p', 'Welch_p'], na_position='last')
        associations.to_csv(tabledir / 'H4_Table_07_curated_CMR_associations_with_family.csv', index=False)

    family_summary_rows = []
    if not associations.empty:
        for family, group in associations.groupby('family', dropna=False):
            top = group.sort_values('Spearman_p', na_position='last').iloc[0]
            family_summary_rows.append({
                'family': family,
                'n_tested': len(group),
                'top_feature': top['feature'],
                'top_feature_clean_label': top.get('clean_label', top['feature']),
                'top_N': top['N'],
                'top_Spearman_r': top['Spearman_r'],
                'top_Spearman_p': top['Spearman_p'],
                'top_Spearman_q': top.get('Spearman_q', np.nan),
                'top_Cohen_d_high_minus_low': top.get('Cohen_d_high_minus_low', np.nan),
            })
    family_summary = pd.DataFrame(family_summary_rows)
    if not family_summary.empty:
        family_summary = family_summary.sort_values('top_Spearman_p', na_position='last')
    family_summary.to_csv(tabledir / 'H4_Table_08_CMR_family_summary.csv', index=False)
    plot_cmr_family_counts(family_table, figdir / 'H4_Fig01_CMR_selected_family_counts')

    top_overall = associations.sort_values(['Spearman_p', 'Welch_p'], na_position='last').head(config.cmr.top_features_overall)['feature'].tolist() if not associations.empty else []
    cmr_violin_grid(primary, top_overall, 'CORI_high_risk', figdir, prefix='H4_curated_overall')
    top_by_family: list[str] = []
    if not associations.empty:
        for _, group in associations.groupby('family', dropna=False):
            top_by_family.extend(group.sort_values('Spearman_p', na_position='last').head(config.cmr.top_features_per_family)['feature'].tolist())
    top_by_family = list(dict.fromkeys(top_by_family))
    pd.DataFrame({'feature': top_by_family}).to_csv(tabledir / 'H4_Table_09_top_CMR_features_by_family_for_plots.csv', index=False)
    cmr_cluster_map(primary, top_by_family[:30], figdir, prefix='H4_curated_family_selected')

    clinical_covariates, age_column = one_age_covars(primary, logger=logger)
    features_for_regression = top_overall
    pd.DataFrame({'feature_for_adjusted_regression': features_for_regression}).to_csv(tabledir / 'H4_Table_10_features_entering_adjusted_CMR_regressions.csv', index=False)
    regressions = adjusted_linear_regressions(primary, 'CORI_score', features_for_regression, clinical_covariates, tabledir, prefix='H4_curated')
    if not regressions.empty:
        regressions = regressions.merge(
            inventory[['column', 'clean_label', 'family']].rename(columns={'column': 'feature'}),
            on='feature',
            how='left',
        ).sort_values('p', na_position='last')
        regressions.to_csv(tabledir / 'H4_Table_11_adjusted_CMR_regressions_curated_with_family.csv', index=False)
    plot_adjusted_beta_forest(regressions, figdir / 'H4_Fig03_adjusted_CMR_beta_forest_curated')

    manifest = {
        'notebook': 'H4',
        'purpose': 'Load H1 locked CORI and evaluate curated visit-aware CMR phenotype families; no CORI retraining.',
        'H1_CORI_bundle': str(config.cori_bundle),
        'CARDIAC_MRI_CSV': str(config.paths.cardiac_mri),
        'primary_CMR_rows': len(primary),
        'primary_CMR_unique_eids': int(primary.eid.nunique()) if 'eid' in primary.columns else None,
        'CMR_feature_families': CMR_FEATURE_FAMILIES,
        'PRIMARY_CMR_FAMILIES': config.cmr.primary_families,
        'EXACT_CMR_VARIABLES': config.cmr.exact_variables,
        'USER_CMR_KEYWORDS': config.cmr.keywords,
        'MIN_CMR_NONMISSING': config.cmr.min_nonmissing,
        'MIN_CMR_UNIQUE_VALUES': config.cmr.min_unique_values,
        'n_selected_CMR_features': len(selected),
        'selected_CMR_features': selected,
        'clinical_covariates_for_adjusted_regression': clinical_covariates,
        'age_variable': age_column,
    }
    (outdir / 'H4_output_manifest.json').write_text(json.dumps(manifest, indent=2, default=str), encoding='utf-8')
    return H4Result(cmr_long, cmr_merged, primary, inventory, selected, associations, regressions)
