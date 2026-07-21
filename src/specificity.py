from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import chi2, pearsonr, norm

from .models import rank_features


def interaction_slopes(cancer: pd.DataFrame, control: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    cancer_rank = rank_features(cancer, feature_columns)
    control_rank = rank_features(control, feature_columns)
    merged = cancer_rank[["feature", "coef", "se"]].rename(
        columns={"coef": "cancer_coef", "se": "cancer_se"}
    ).merge(
        control_rank[["feature", "coef", "se"]].rename(
            columns={"coef": "control_coef", "se": "control_se"}
        ),
        on="feature",
        how="inner",
    )
    merged["difference"] = merged["cancer_coef"] - merged["control_coef"]
    merged["difference_se"] = np.sqrt(merged["cancer_se"] ** 2 + merged["control_se"] ** 2)
    merged["interaction_z"] = merged["difference"] / merged["difference_se"]
    merged["interaction_p"] = chi2.sf(merged["interaction_z"] ** 2, 1)
    return merged.sort_values("interaction_p").reset_index(drop=True)


def interaction_replication(
    development_cancer: pd.DataFrame,
    development_control: pd.DataFrame,
    heldout_cancer: pd.DataFrame,
    heldout_control: pd.DataFrame,
    feature_columns: list[str],
):
    development = interaction_slopes(development_cancer, development_control, feature_columns)
    heldout = interaction_slopes(heldout_cancer, heldout_control, feature_columns)
    merged = development[["feature", "interaction_z", "interaction_p"]].rename(
        columns={"interaction_z": "development_z", "interaction_p": "development_p"}
    ).merge(
        heldout[["feature", "interaction_z", "interaction_p"]].rename(
            columns={"interaction_z": "heldout_z", "interaction_p": "heldout_p"}
        ),
        on="feature",
        how="inner",
    )

    replication_r, replication_p = pearsonr(merged["development_z"], merged["heldout_z"])
    fisher_statistic = float(-2 * np.log(np.clip(merged["development_p"], 1e-300, 1)).sum())
    global_p = float(chi2.sf(fisher_statistic, 2 * len(merged)))

    summary = {
        "n_features": len(merged),
        "development_nominal_p_lt_0_05": int((merged["development_p"] < 0.05).sum()),
        "development_global_fisher_p": global_p,
        "heldout_replication_r": float(replication_r),
        "heldout_replication_p": float(replication_p),
    }
    return merged, summary


def score_interaction_test(
    cancer: pd.DataFrame,
    control: pd.DataFrame,
    score_column: str,
    covariates: list[str],
):
    """
    Test multiplicative effect modification while allowing separate baseline
    hazards by cancer status and center.
    """
    data = pd.concat(
        [
            cancer.assign(cancer_status=1),
            control.assign(cancer_status=0),
        ],
        ignore_index=True,
    )
    interaction_column = f"{score_column}_x_cancer"
    data[interaction_column] = data[score_column] * data["cancer_status"]
    data["analysis_stratum"] = (
        data["cancer_status"].astype(str)
        + "__"
        + data["center"].astype(str)
    )

    numeric_covariates = [
        column for column in covariates if column != "center"
    ]
    terms = [score_column, interaction_column, *numeric_covariates]
    model_data = data[
        ["time_years", "Y_mace", *terms, "analysis_stratum"]
    ].copy()
    model_data[
        ["time_years", "Y_mace", *terms]
    ] = model_data[
        ["time_years", "Y_mace", *terms]
    ].apply(pd.to_numeric, errors="coerce")
    model_data = model_data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()

    varying_terms = [
        column
        for column in terms
        if model_data[column].nunique() > 1
        and np.isfinite(model_data[column].std(ddof=0))
        and model_data[column].std(ddof=0) > 1e-12
    ]
    if score_column not in varying_terms or interaction_column not in varying_terms:
        raise ValueError("Score and cancer interaction must vary in pooled data.")

    fit_data = model_data[
        ["time_years", "Y_mace", *varying_terms, "analysis_stratum"]
    ].copy()
    model = CoxPHFitter(penalizer=0.0)
    model.fit(
        fit_data,
        duration_col="time_years",
        event_col="Y_mace",
        strata=["analysis_stratum"],
        robust=True,
        show_progress=False,
    )

    row = model.summary.loc[interaction_column]
    beta_score = float(model.params_.loc[score_column])
    beta_interaction = float(model.params_.loc[interaction_column])
    covariance = model.variance_matrix_
    cancer_variance = float(
        covariance.loc[score_column, score_column]
        + covariance.loc[interaction_column, interaction_column]
        + 2 * covariance.loc[score_column, interaction_column]
    )
    cancer_se = np.sqrt(max(cancer_variance, 0.0))
    cancer_slope = beta_score + beta_interaction

    return {
        "never_cancer_HR": float(np.exp(beta_score)),
        "cancer_HR": float(np.exp(cancer_slope)),
        "cancer_HR_low": float(np.exp(cancer_slope - 1.96 * cancer_se)),
        "cancer_HR_high": float(np.exp(cancer_slope + 1.96 * cancer_se)),
        "interaction_HR": float(np.exp(row["coef"])),
        "interaction_HR_low": float(np.exp(row["coef lower 95%"])),
        "interaction_HR_high": float(np.exp(row["coef upper 95%"])),
        "interaction_p": float(row["p"]),
        "N": int(len(model_data)),
        "events": int(model_data["Y_mace"].sum()),
    }


def dual_score_interaction_test(
    cancer: pd.DataFrame,
    control: pd.DataFrame,
    cori_score: str,
    mmace_score: str,
    covariates: list[str],
) -> dict[str, float]:
    """Directly test whether cancer modifies CORI more strongly than MMACE."""
    data = pd.concat(
        [
            cancer.assign(cancer_status=1),
            control.assign(cancer_status=0),
        ],
        ignore_index=True,
    )
    data["cancer_x_CORI"] = data["cancer_status"] * data[cori_score]
    data["cancer_x_MMACE"] = data["cancer_status"] * data[mmace_score]
    data["analysis_stratum"] = (
        data["cancer_status"].astype(str)
        + "__"
        + data["center"].astype(str)
    )

    numeric_covariates = [
        column for column in covariates if column != "center"
    ]
    terms = [
        cori_score,
        mmace_score,
        "cancer_x_CORI",
        "cancer_x_MMACE",
        *numeric_covariates,
    ]
    model_data = data[
        ["time_years", "Y_mace", *terms, "analysis_stratum"]
    ].copy()
    model_data[
        ["time_years", "Y_mace", *terms]
    ] = model_data[
        ["time_years", "Y_mace", *terms]
    ].apply(pd.to_numeric, errors="coerce")
    model_data = model_data.replace(
        [np.inf, -np.inf], np.nan
    ).dropna().copy()

    varying_terms = [
        column
        for column in terms
        if model_data[column].nunique() > 1
        and np.isfinite(model_data[column].std(ddof=0))
        and model_data[column].std(ddof=0) > 1e-12
    ]
    required = {cori_score, mmace_score, "cancer_x_CORI", "cancer_x_MMACE"}
    if not required.issubset(varying_terms):
        raise ValueError("Both scores and both interaction terms must vary.")

    fit_data = model_data[
        ["time_years", "Y_mace", *varying_terms, "analysis_stratum"]
    ].copy()
    model = CoxPHFitter(penalizer=0.0)
    model.fit(
        fit_data,
        duration_col="time_years",
        event_col="Y_mace",
        strata=["analysis_stratum"],
        robust=True,
        show_progress=False,
    )

    beta_cori = float(model.params_.loc["cancer_x_CORI"])
    beta_mmace = float(model.params_.loc["cancer_x_MMACE"])
    variance = model.variance_matrix_
    difference = beta_cori - beta_mmace
    difference_variance = float(
        variance.loc["cancer_x_CORI", "cancer_x_CORI"]
        + variance.loc["cancer_x_MMACE", "cancer_x_MMACE"]
        - 2 * variance.loc["cancer_x_CORI", "cancer_x_MMACE"]
    )
    difference_se = np.sqrt(max(difference_variance, 1e-15))
    z_value = difference / difference_se

    return {
        "MMACE_comparator": mmace_score,
        "CORI_interaction_HR": float(np.exp(beta_cori)),
        "CORI_interaction_p": float(model.summary.loc["cancer_x_CORI", "p"]),
        "MMACE_interaction_HR": float(np.exp(beta_mmace)),
        "MMACE_interaction_p": float(model.summary.loc["cancer_x_MMACE", "p"]),
        "interaction_ratio_CORI_vs_MMACE": float(np.exp(difference)),
        "difference_z": float(z_value),
        "difference_p": float(2 * norm.sf(abs(z_value))),
        "N": int(len(model_data)),
        "events": int(model_data["Y_mace"].sum()),
    }


def permuted_replication_p(
    table: pd.DataFrame,
    repetitions: int = 5000,
    seed: int = 20260714,
) -> float:
    rng = np.random.default_rng(seed)
    observed = np.corrcoef(table["development_z"], table["heldout_z"])[0, 1]
    null = []
    heldout = table["heldout_z"].to_numpy().copy()
    for _ in range(repetitions):
        rng.shuffle(heldout)
        null.append(np.corrcoef(table["development_z"], heldout)[0, 1])
    return float((1 + np.sum(np.asarray(null) >= observed)) / (repetitions + 1))
