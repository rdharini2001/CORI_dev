"""Compatibility wrapper for :mod:`cori_analysis.clinical_data`."""

from cori_analysis.clinical_data import (
    find_col,
    clean_binary_col,
    load_and_prepare_htn_db,
    merge_htn_diabetes_status,
    one_age_covars_htn_diabetes,
    choose_adjustment_covars_for_h4_hcori,
    one_age_covars,
    load_clinical_status_exact,
)

__all__ = [
    'find_col',
    'clean_binary_col',
    'load_and_prepare_htn_db',
    'merge_htn_diabetes_status',
    'one_age_covars_htn_diabetes',
    'choose_adjustment_covars_for_h4_hcori',
    'one_age_covars',
    'load_clinical_status_exact',
]
