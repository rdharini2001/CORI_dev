"""Compatibility wrapper for :mod:`cori_analysis.handcrafted`."""

from cori_analysis.handcrafted import (
    set_logger,
    load_handcrafted_features,
    merge_handcrafted,
    merge_handcrafted_cached_safe,
    discover_handcrafted_cols,
    prepare_hc_matrix,
    univariate_cox_rank_hc,
    internal_choose_k_hc,
    train_handcrafted_survival_model,
    apply_handcrafted_bundle,
    hc_performance_row,
    plot_hc_km,
    plot_handcrafted_feature_transition,
    plot_handcrafted_feature_transition_clean,
    load_cached_handcrafted_features,
    load_cached_handcrafted_features_exact,
)

__all__ = [
    'set_logger',
    'load_handcrafted_features',
    'merge_handcrafted',
    'merge_handcrafted_cached_safe',
    'discover_handcrafted_cols',
    'prepare_hc_matrix',
    'univariate_cox_rank_hc',
    'internal_choose_k_hc',
    'train_handcrafted_survival_model',
    'apply_handcrafted_bundle',
    'hc_performance_row',
    'plot_hc_km',
    'plot_handcrafted_feature_transition',
    'plot_handcrafted_feature_transition_clean',
    'load_cached_handcrafted_features',
    'load_cached_handcrafted_features_exact',
]
