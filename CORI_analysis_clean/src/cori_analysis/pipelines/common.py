from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..cohorts import (
    build_allcancer_cohort,
    build_noncancer_ready_cohort,
    discover_feature_cols,
    find_center_col,
    make_center_split,
)
from ..common import QCLogger, ensure_dirs, load_csv
from ..config import AnalysisConfig


@dataclass
class SharedData:
    master: pd.DataFrame
    feature_cols: list[str]
    feature_cols_noncancer: list[str]
    cancer_df: pd.DataFrame
    cancer_train: pd.DataFrame
    cancer_test: pd.DataFrame
    noncancer_df: pd.DataFrame
    noncancer_train: pd.DataFrame
    noncancer_test: pd.DataFrame
    cancer_audit: pd.DataFrame
    noncancer_audit: pd.DataFrame
    cancer_split: pd.DataFrame
    noncancer_split: pd.DataFrame
    center_col: str


def prepare_shared_data(config: AnalysisConfig) -> SharedData:
    outdir, _, tabledir, _, qcdir = ensure_dirs(config.h1_dir)
    logger = QCLogger(qcdir / 'H1_QC_log.txt')
    logger.section('Load master and build shared cohorts')

    master = load_csv(config.paths.master, logger)
    feature_cols = discover_feature_cols(master)
    cancer_df, cancer_audit = build_allcancer_cohort(
        master,
        feature_cols,
        horizon=config.model.primary_horizon,
        logger=logger,
    )
    center_col = config.columns.center if config.columns.center in cancer_df.columns else find_center_col(cancer_df)
    cancer_train, cancer_test, cancer_split, center_col = make_center_split(
        cancer_df,
        config.model.train_centers,
        center_col=center_col,
        logger=logger,
    )

    noncancer_raw = load_csv(config.paths.noncancer, logger)
    feature_cols_noncancer = [column for column in feature_cols if column in noncancer_raw.columns]
    noncancer_df, noncancer_audit = build_noncancer_ready_cohort(
        noncancer_raw,
        feature_cols_noncancer,
        logger=logger,
    )
    noncancer_train, noncancer_test, noncancer_split, _ = make_center_split(
        noncancer_df,
        config.model.train_centers,
        center_col=center_col,
        logger=logger,
    )

    cancer_audit.to_csv(tabledir / 'H1_Table_00_cohort_audit.csv', index=False)
    cancer_split.to_csv(tabledir / 'H1_Table_01_center_split.csv', index=False)
    logger.log(
        f'Shared cohorts ready: cancer development N={len(cancer_train)}, held-out N={len(cancer_test)}; '
        f'non-cancer development N={len(noncancer_train)}, held-out N={len(noncancer_test)}.'
    )
    return SharedData(
        master=master,
        feature_cols=feature_cols,
        feature_cols_noncancer=feature_cols_noncancer,
        cancer_df=cancer_df,
        cancer_train=cancer_train,
        cancer_test=cancer_test,
        noncancer_df=noncancer_df,
        noncancer_train=noncancer_train,
        noncancer_test=noncancer_test,
        cancer_audit=cancer_audit,
        noncancer_audit=noncancer_audit,
        cancer_split=cancer_split,
        noncancer_split=noncancer_split,
        center_col=center_col,
    )
