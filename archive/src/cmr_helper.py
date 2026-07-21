"""Compatibility wrapper for :mod:`cori_analysis.cmr`."""

from cori_analysis.cmr import (
    CMR_FEATURE_FAMILIES,
    preprocess_cmr_features,
    preprocess_cmr_features_for_hcori,
    classify_cmr_feature,
    build_cmr_variable_inventory,
)

__all__ = [
    'CMR_FEATURE_FAMILIES',
    'preprocess_cmr_features',
    'preprocess_cmr_features_for_hcori',
    'classify_cmr_feature',
    'build_cmr_variable_inventory',
]
