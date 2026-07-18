from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..cohorts import reshape_cmr_long
from ..common import clean_id, load_csv
from ..config import AnalysisConfig
from ..mediation import MediationConfig, load_and_merge_data
from ..visualization import triangle_corr_plot
from .handcrafted import HCORI_LOCKED_FEATURES


def run_cross_modal_triangle(config: AnalysisConfig):
    """Run the final exploratory CMR–handcrafted-retina–CORI triangle plot."""
    source = load_and_merge_data(
        MediationConfig(
            source_file=config.paths.source_population,
            clinical_file=config.paths.clinical,
        )
    )
    source['eid'] = clean_id(source['eid'])

    cmr = load_csv(config.paths.cardiac_mri)
    cmr['eid'] = clean_id(cmr['eid'])
    cmr_long = reshape_cmr_long(cmr, instances=(2, 3))
    cmr_variables = [
        column for column in cmr_long.columns
        if column != 'eid' and np.issubdtype(cmr_long[column].dtype, np.floating)
    ]
    ranges = cmr_long[cmr_variables].max() - cmr_long[cmr_variables].min()
    nonconstant = ranges[ranges.ne(0)].index.tolist()
    cmr_long[nonconstant] = (
        cmr_long[nonconstant] - cmr_long[nonconstant].min()
    ) / ranges[nonconstant]
    cmr_variables = nonconstant

    handcrafted = pd.read_csv(config.handcrafted_cache, low_memory=False)
    handcrafted['eid'] = clean_id(handcrafted['eid'])
    missing = [column for column in HCORI_LOCKED_FEATURES if column not in handcrafted.columns]
    if missing:
        raise ValueError(f'Handcrafted cache is missing locked variables: {missing}')

    merged = source.merge(cmr_long[['eid'] + cmr_variables], on='eid', how='inner')
    merged = merged.merge(
        handcrafted[['eid'] + HCORI_LOCKED_FEATURES].drop_duplicates('eid'),
        on='eid',
        how='left',
        validate='m:1',
    )
    cmr_variables = [column for column in cmr_variables if merged[column].var() > 1e-2]
    cmr_variables = [column for column in cmr_variables if merged[column].notna().mean() > 0.2]
    correlations = merged[cmr_variables].corrwith(merged['M_CORI_z'])
    cmr_variables = correlations[correlations.abs() > 0.05].index.tolist()
    if not cmr_variables:
        raise ValueError('No CMR variables passed the original variance, missingness, and CORI-correlation thresholds.')

    figure, axes = triangle_corr_plot(
        merged,
        row_vars=cmr_variables[:5],
        col_vars=HCORI_LOCKED_FEATURES,
        side_var_left='M_CORI_z',
        side_var_bottom='M_CORI_z',
        shape='lower',
        label_row='CMR',
        label_col='Retinal handcrafted features',
        label_left='vs CORI',
        label_bottom='vs CORI',
        cmap='RdBu_r',
        vmin=None,
        vmax=None,
        figsize=(10, 10),
    )
    output = config.paths.output_dir / 'cross_modal_triangle'
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / 'CMR_retina_CORI_triangle.png', dpi=300, bbox_inches='tight')
    figure.savefig(output / 'CMR_retina_CORI_triangle.svg', bbox_inches='tight')
    return merged, cmr_variables, figure, axes
