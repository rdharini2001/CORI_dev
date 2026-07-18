from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from .common import clean_id


@dataclass(frozen=True)
class ProteomicsConfig:
    source_file: Path = Path('./data/source_population_with_retinal_scores.csv')
    clinical_file: Path = Path('./data/final_df_HTN_DB_Status.csv')
    proteomics_file: Path = Path('./data/proteomics_50k_instance_0_sdf.csv')
    protein_columns_file: Path = Path('./data/alz_proteomics_columns.txt')
    output_dir: Path = Path('./figures/proteomics')


def load_proteomics_data(config: ProteomicsConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(config.source_file)
    clinical = pd.read_csv(config.clinical_file, usecols=['eid', 'HTN', 'Diabetes'])
    scores = scores.merge(clinical, on='eid', how='left', validate='m:1').dropna(subset=['height']).copy()
    proteomics = pd.read_csv(config.proteomics_file, low_memory=False)
    scores['eid'] = clean_id(scores['eid'])
    proteomics['eid'] = clean_id(proteomics['eid'])
    merged = scores.merge(proteomics, on='eid', how='inner', validate='m:1')
    train = merged[merged['analysis_role'].eq('Development')].copy()
    test = merged[merged['analysis_role'].ne('Development')].copy()
    train_cori_median = train['M_CORI_z'].median()
    train_mmace_median = train['M_MMACE_z'].median()
    test['CORI_z_class'] = (test['M_CORI_z'] > train_cori_median).astype(int)
    test['MMACE_z_class'] = (test['M_MMACE_z'] > train_mmace_median).astype(int)
    return train, test


def differential_proteomics(data: pd.DataFrame, protein_columns: list[str], group_column: str) -> pd.DataFrame:
    low = data[data[group_column].eq(0)]
    high = data[data[group_column].eq(1)]
    rows = []
    for protein in protein_columns:
        mean_low = low[protein].mean()
        mean_high = high[protein].mean()
        _, p_value = ttest_ind(high[protein], low[protein], nan_policy='omit')
        rows.append({'protein': protein, 'log2FC': np.log2(mean_high + 1e-8) - np.log2(mean_low + 1e-8), 'pval': p_value})
    result = pd.DataFrame(rows)
    result['pval_fdr'] = multipletests(result['pval'], alpha=0.05, method='fdr_bh')[1]
    result['-log10(pval)'] = -np.log10(result['pval'])
    result['-log10(pval_fdr)'] = -np.log10(result['pval_fdr'])
    return result.round(4)


def plot_log2fc_comparison(cori, mmace, save_base):
    merged = cori[['protein', 'log2FC', 'pval_fdr']].merge(
        mmace[['protein', 'log2FC', 'pval_fdr']], on='protein', suffixes=('_CORI', '_MMACE'), validate='1:1'
    )
    cori_sig = merged['pval_fdr_CORI'] < 0.05
    mmace_sig = merged['pval_fdr_MMACE'] < 0.05
    colors = np.select(
        [cori_sig & mmace_sig, cori_sig, mmace_sig],
        ['#3DBB08DC', '#EB331EDC', '#0000FFD6'],
        default='#80808013',
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(merged['log2FC_CORI'], merged['log2FC_MMACE'], c=colors)
    ax.axvline(0, color='grey', linestyle='--')
    ax.axhline(0, color='grey', linestyle='--')
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    ax.set_xlabel('log2FC (CORI)')
    ax.set_ylabel('log2FC (MMACE)')
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#EB331EDC', markersize=10, label='CORI sig'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#0000FFD6', markersize=10, label='MMACE sig'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#3DBB08DC', markersize=10, label='Both sig'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#80808013', markersize=10, label='Not sig'),
    ]
    ax.legend(handles=handles, title='Significance (FDR)', fontsize=10, frameon=False)
    save_base = Path(save_base)
    save_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(save_base.with_suffix('.svg'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    return merged


def run_proteomics(config: ProteomicsConfig) -> dict[str, pd.DataFrame]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _, test = load_proteomics_data(config)
    protein_columns = [line.strip() for line in config.protein_columns_file.read_text().splitlines() if line.strip()]
    results = {
        'CORI': differential_proteomics(test, protein_columns, 'CORI_z_class'),
        'MMACE': differential_proteomics(test, protein_columns, 'MMACE_z_class'),
    }
    for name, table in results.items():
        table.to_csv(config.output_dir / f'volcano_{name}.csv', index=False)
    comparison = plot_log2fc_comparison(results['CORI'], results['MMACE'], config.output_dir / 'log2FC_scatter_CORI_MMACE')
    comparison.to_csv(config.output_dir / 'log2FC_comparison.csv', index=False)
    return results
