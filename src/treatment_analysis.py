"""Compatibility wrapper for :mod:`cori_analysis.treatment`."""

from cori_analysis.treatment import (
    zscore_from_train_apply,
    fit_treatment_adjusted_cox,
)

__all__ = [
    'zscore_from_train_apply',
    'fit_treatment_adjusted_cox',
]
