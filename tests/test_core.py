from __future__ import annotations

import numpy as np
import pandas as pd

from cori_analysis.cohorts import build_allcancer_cohort, build_noncancer_ready_cohort
from cori_analysis.common import clean_id, fmt_ci, pformat, safe_name
from cori_analysis.survival import apply_thresholds_from_dict, derive_thresholds, variance_filter


def test_small_helpers():
    assert clean_id(pd.Series([1.0, ' 2 '])).tolist() == ['1', '2']
    assert safe_name('CORI / M-MACE') == 'CORI_M_MACE'
    assert pformat(0.0004) == '<0.001'
    assert fmt_ci(1.2, 1.0, 1.4) == '1.20 (1.00-1.40)'


def test_all_cancer_cohort_logic():
    data = pd.DataFrame({
        'eid': ['1', '1', '2', '3'],
        'image_visit': [0, 0, 0, 0],
        'allcancer_event_status': [1, 1, 1, 0],
        'MACE_in_allCancer_10yr_censored_time': [100, 200, 300, 400],
        'MACE_in_allCancer_10yr_censored_status': [0, 1, 1, 0],
        'f0': [1.0, 2.0, 3.0, 4.0],
    })
    cohort, audit = build_allcancer_cohort(data, ['f0'], horizon=10)
    assert cohort['eid'].tolist() == ['1', '2']
    assert cohort.loc[cohort['eid'].eq('1'), 'time'].item() == 200
    assert audit.iloc[-1]['unique_eids'] == 2


def test_noncancer_cohort_logic():
    data = pd.DataFrame({
        'eid': ['1', '2'],
        'image_visit': [0, 0],
        'mace_10y_time_days': [100, 200],
        'mace_10y_event': [0, 1],
        'f0': [1.0, 2.0],
    })
    cohort, _ = build_noncancer_ready_cohort(data, ['f0'])
    assert cohort[['time', 'event']].to_dict('list') == {'time': [100, 200], 'event': [0, 1]}


def test_thresholds_and_variance_filter():
    scores = pd.Series([0.0, 1.0, 2.0, 3.0])
    thresholds = derive_thresholds(scores)
    frame = apply_thresholds_from_dict(pd.DataFrame({'score': scores}), 'score', 'X', thresholds)
    assert frame['X_high_risk'].tolist() == [0, 0, 1, 1]
    train = pd.DataFrame({'f0': [1, 2, 3], 'f1': [1, 1, 1], 'f2': [1, np.nan, np.nan]})
    keep, table = variance_filter(train, ['f0', 'f1', 'f2'], max_missing=0.5)
    assert keep == ['f0']
    assert set(table['status']) == {'keep', 'drop'}
