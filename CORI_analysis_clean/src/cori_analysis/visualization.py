from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import ttest_ind

from .common import savefig
from .evaluation import transition_tables

DEFAULT_COLORS = {
    'dark_red1': '#c23537', 'dark_red2': '#d95d5b',
    'light_red1': '#ea999c', 'light_red2': '#f8c9c7',
    'light_blue1': '#c1e3fa', 'light_blue2': '#91c3e8',
    'dark_blue1': '#0067a8', 'dark_blue2': '#4c8fca',
    'light_green1': '#56aa3e', 'light_green2': '#95c36e',
    'dark_green1': '#1e662a', 'dark_green2': '#3c892d',
}


def transition_heatmap(df, left, right, event_col, value_col, title, save_base):
    counts, event_rates, values = transition_tables(df, left, right, event_col, value_col=value_col)
    annot = np.empty(values.shape, dtype=object)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            n = int(counts.iloc[i, j])
            event_rate = event_rates.iloc[i, j]
            annot[i, j] = f'N={n}\nER={event_rate:.1%}'
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sns.heatmap(
        values.astype(float), annot=annot, fmt='', cmap='coolwarm', cbar=True,
        cbar_kws={'label': f'Mean {value_col}' if value_col else 'Event rate'},
        linewidths=2, linecolor='#fcfcfb', annot_kws={'fontsize': 12, 'color': 'black'},
        xticklabels=values.columns, yticklabels=values.index, ax=ax,
    )
    ax.set_xlabel(right)
    ax.set_ylabel(left)
    ax.set_title(title, fontweight='bold')
    fig.tight_layout()
    savefig(fig, save_base)
    plt.close(fig)
    return counts, event_rates, values


def plot_calibration_deciles(calibration, save_base):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(calibration['Mean_predicted_risk'], calibration['Observed_rate'], s=50)
    upper = max(calibration['Mean_predicted_risk'].max(), calibration['Observed_rate'].max())
    ax.plot([0, upper], [0, upper], linestyle='--', color='black')
    ax.set_xlabel('Mean predicted 10-year risk')
    ax.set_ylabel('Observed event rate')
    ax.set_title('Calibration: stacked clinical-risk + CORI-risk model', fontweight='bold')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    savefig(fig, save_base)
    plt.close(fig)


def plot_cmr_family_counts(selected_family_table, save_base):
    if selected_family_table.empty:
        return
    tmp = selected_family_table.sort_values('n_features').copy()
    tmp['family'] = tmp['family'].str.replace('_', ' ', regex=False)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.45 * len(tmp) + 1)))
    sns.barplot(x='n_features', y='family', data=tmp, ax=ax, palette=sns.color_palette('Blues', n_colors=len(tmp)))
    ax.set_xlabel('Number of selected CMR phenotypes')
    ax.set_ylabel('')
    ax.set_title('Curated CMR phenotype families selected for H4', fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    savefig(fig, save_base)
    plt.close(fig)


def cmr_violin_grid(df, features, group_col, figdir, prefix='H4', colors=DEFAULT_COLORS):
    feats = [feature for feature in features if feature in df.columns][:6]
    if not feats:
        return
    fig, axes = plt.subplots(len(feats), 1, figsize=(4.5, 3 * len(feats)), squeeze=False)
    for ax, feature in zip(axes.ravel(), feats):
        data = df[[feature, group_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        denominator = data[feature].max() - data[feature].min()
        data[feature] = 0.0 if denominator == 0 else (data[feature] - data[feature].min()) / denominator
        low = pd.to_numeric(data.loc[data[group_col] == 0, feature], errors='coerce').dropna().values
        high = pd.to_numeric(data.loc[data[group_col] == 1, feature], errors='coerce').dropna().values
        sns.violinplot(data=[low, high], ax=ax, inner='quartile', palette=[colors['dark_blue1'], colors['dark_red2']])
        if len(low) > 1 and len(high) > 1:
            _, p_value = ttest_ind(low, high, equal_var=False)
            label = 'p < 0.001' if p_value < 0.001 else f'p = {p_value:.3f}'
            ax.plot([0, 0, 0.9, 0.9], [0.82, 0.84, 0.84, 0.82], c='black', lw=1)
            ax.text(0.5, 0.845, label, ha='center', va='bottom', fontsize=12)
        ax.set_ylim(0, 1.08)
        ax.set_xticks([0, 1], ['Low CORI', 'High CORI'])
        ax.set_title(feature.split('|')[0].title(), fontsize=12)
        ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    savefig(fig, Path(figdir) / f'{prefix}_CMR_top_violin_grid')
    plt.close(fig)


def cmr_cluster_map(df, features, figdir, prefix='H4'):
    clean_features = [feature for feature in features if '| Array' not in feature]
    data = df.copy()
    data.columns = [column.split('|')[0].strip() for column in data.columns]
    clean_features = [feature.split('|')[0].strip() for feature in clean_features]
    clean_features = [feature for feature in clean_features if feature in data.columns][:30]
    if len(clean_features) < 3:
        return
    matrix = data[clean_features].apply(pd.to_numeric, errors='coerce')
    matrix = matrix.fillna(matrix.median())
    correlation = matrix.corr(method='spearman')
    distance = 1 - np.abs(correlation.values)
    condensed = distance[np.triu_indices_from(distance, k=1)]
    try:
        order = leaves_list(linkage(condensed, method='average'))
    except ValueError:
        # Degenerate correlation structures have no valid hierarchical ordering.
        order = np.arange(len(clean_features))
    ordered = correlation.iloc[order, order]
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(ordered, ax=ax, cmap='coolwarm', center=0, cbar_kws={'label': 'Spearman correlation'})
    ax.set_title('CMR phenotype cluster map (Spearman correlation)', fontweight='bold')
    fig.tight_layout()
    savefig(fig, Path(figdir) / f'{prefix}_CMR_cluster_map')
    plt.close(fig)


def plot_adjusted_beta_forest(regression_table, save_base, color=DEFAULT_COLORS['dark_blue1']):
    if regression_table.empty:
        return
    plot_df = regression_table.dropna(subset=['beta_per_1SD_CORI', 'ci_low', 'ci_high']).sort_values('p').head(15).iloc[::-1].copy()
    if plot_df.empty:
        return
    labels = plot_df['clean_label'].fillna(plot_df['feature']).astype(str).str.slice(0, 55)
    fig, ax = plt.subplots(figsize=(8.0, max(4, 0.42 * len(plot_df) + 1)))
    y = np.arange(len(plot_df))
    ax.errorbar(
        plot_df['beta_per_1SD_CORI'], y,
        xerr=[plot_df['beta_per_1SD_CORI'] - plot_df['ci_low'], plot_df['ci_high'] - plot_df['beta_per_1SD_CORI']],
        fmt='s', capsize=3, color=color,
    )
    ax.axvline(0, color='black', linestyle='--')
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel('Adjusted beta per 1-SD CORI')
    ax.set_title('Adjusted associations between locked H1 CORI and curated CMR phenotypes', fontweight='bold')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    savefig(fig, save_base)
    plt.close(fig)


def _corr_series(df, variables, other, method='pearson'):
    return np.array([df[variable].corr(df[other], method=method) for variable in variables])


def _cross_corr_matrix(df, row_variables, column_variables, method='pearson'):
    return np.array([[df[row].corr(df[column], method=method) for column in column_variables] for row in row_variables])


def triangle_corr_plot(
    df: pd.DataFrame,
    row_vars: Sequence[str],
    col_vars: Sequence[str],
    side_var_left: str,
    side_var_bottom: str,
    *,
    method: str = 'pearson',
    shape: str = 'lower',
    size_scale: float = 9900.0,
    cmap: str = 'RdBu_r',
    vmin: float | None = -1.0,
    vmax: float | None = 1.0,
    figsize: tuple[float, float] = (10, 10),
    label_row: str = 'Row vars',
    label_col: str = 'Col vars',
    label_left: str | None = None,
    label_bottom: str | None = None,
):
    if shape not in {'lower', 'full'}:
        raise ValueError("shape must be 'lower' or 'full'")
    n_rows, n_cols = len(row_vars), len(col_vars)
    label_left = label_left or side_var_left
    label_bottom = label_bottom or side_var_bottom
    correlation = _cross_corr_matrix(df, row_vars, col_vars, method)
    left = _corr_series(df, row_vars, side_var_left, method)
    bottom = _corr_series(df, col_vars, side_var_bottom, method)
    finite = np.concatenate([correlation.ravel(), left, bottom])
    finite = finite[np.isfinite(finite)]
    if vmin is None:
        vmin = float(finite.min()) if finite.size else -1.0
    if vmax is None:
        vmax = float(finite.max()) if finite.size else 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.get_cmap(cmap)
    fig = plt.figure(figsize=figsize)
    grid = GridSpec(2, 2, width_ratios=[0.6, n_cols], height_ratios=[n_rows, 0.6], wspace=0.03, hspace=0.03, figure=fig)
    ax_left = fig.add_subplot(grid[0, 0])
    ax_main = fig.add_subplot(grid[0, 1], sharey=ax_left)
    ax_bottom = fig.add_subplot(grid[1, 1], sharex=ax_main)
    fig.add_subplot(grid[1, 0]).axis('off')
    for i in range(n_rows):
        for j in range(n_cols):
            if shape == 'lower' and j > i:
                continue
            value = correlation[i, j]
            ax_main.scatter(j, i, s=abs(value) * size_scale, c=[colormap(norm(value))], edgecolors='black', linewidths=0.5, zorder=3)
    for i in range(n_rows):
        ax_main.axhline(i, color='lightgray', lw=0.4, zorder=1)
    for j in range(n_cols):
        ax_main.axvline(j, color='lightgray', lw=0.4, zorder=1)
    ax_main.set_xlim(-0.7, n_cols - 0.3)
    ax_main.set_ylim(n_rows - 0.3, -0.7)
    ax_main.set_xticks(range(n_cols), col_vars, rotation=90)
    ax_main.set_yticks(range(n_rows), row_vars)
    ax_main.set_xlabel(label_col)
    ax_main.xaxis.set_label_position('top')
    ax_main.tick_params(labelbottom=False, labeltop=True, labelleft=False)
    ax_main.set_frame_on(False)
    for i, value in enumerate(left):
        ax_left.scatter(0, i, s=abs(value) * size_scale, c=[colormap(norm(value))], edgecolors='black', linewidths=0.5)
    ax_left.set_xlim(-0.5, 0.5)
    ax_left.set_ylim(n_rows - 0.3, -0.7)
    ax_left.set_xticks([])
    ax_left.set_yticks(range(n_rows), row_vars)
    ax_left.set_ylabel(label_row)
    ax_left.set_title(label_left, fontsize=9)
    ax_left.set_frame_on(False)
    for j, value in enumerate(bottom):
        ax_bottom.scatter(j, 0, s=abs(value) * size_scale, c=[colormap(norm(value))], edgecolors='black', linewidths=0.5)
    ax_bottom.set_ylim(-0.5, 0.5)
    ax_bottom.set_xlim(-0.7, n_cols - 0.3)
    ax_bottom.set_yticks([])
    ax_bottom.set_xticks(range(n_cols), col_vars, rotation=90)
    ax_bottom.set_xlabel(label_bottom)
    ax_bottom.set_frame_on(False)
    scalar = cm.ScalarMappable(norm=norm, cmap=colormap)
    scalar.set_array([])
    fig.colorbar(scalar, ax=[ax_main, ax_left, ax_bottom], shrink=0.5, pad=0.02).set_label(f'{method} correlation')
    return fig, (ax_main, ax_left, ax_bottom)
