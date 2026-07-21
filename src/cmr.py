from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import combine_pvalues, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

from scipy.stats import norm


def add_directional_p_values(
    table: pd.DataFrame,
    directions: dict[str, int],
) -> pd.DataFrame:
    """
    Convert two-sided standardized-beta results into
    prespecified directional p-values.

    Direction must be fixed biologically before testing:
    +1 means a positive beta is adverse-aligned;
    -1 means a negative beta is adverse-aligned.
    """
    output = table.copy()

    standard_normal_975 = norm.ppf(0.975)

    output["se"] = (
        output["CI_high"]
        - output["CI_low"]
    ) / (
        2 * standard_normal_975
    )

    output["direction"] = (
        output["phenotype"]
        .map(directions)
    )

    output["directional_z"] = (
        output["direction"]
        * output["beta"]
        / output["se"]
    )

    output["directional_p"] = norm.sf(
        output["directional_z"]
    )

    return output


def _prepare_linear_model_data(
    df: pd.DataFrame,
    outcome: str,
    predictors: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare one numeric complete-case table for linear regression.

    Center is dummy-coded. Constant predictors are removed.
    All remaining values are converted to ordinary float64 so
    statsmodels never receives pandas object/nullable dtypes.
    """

    columns = list(
        dict.fromkeys(
            [
                outcome,
                *predictors,
            ]
        )
    )

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing linear-model columns: "
            f"{missing_columns}"
        )

    data = df[columns].copy()

    categorical_columns = [
        column
        for column in predictors
        if column == "center"
    ]

    # Missing categorical values should not silently become
    # the reference category.
    if categorical_columns:
        data = data.dropna(
            subset=categorical_columns
        )

        data = pd.get_dummies(
            data,
            columns=categorical_columns,
            drop_first=True,
            dtype=np.float64,
        )

    # Convert every remaining column explicitly.
    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    data = (
        data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .astype(np.float64)
    )

    predictor_columns = [
        column
        for column in data.columns
        if column != outcome
    ]

    # Remove zero-variance adjustment columns.
    predictor_columns = [
        column
        for column in predictor_columns
        if (
            data[column].nunique(dropna=True) > 1
            and np.isfinite(
                data[column].std(ddof=0)
            )
            and data[column].std(ddof=0) > 1e-12
        )
    ]

    return (
        data[
            [
                outcome,
                *predictor_columns,
            ]
        ].copy(),
        predictor_columns,
    )


def adjusted_association(
    df: pd.DataFrame,
    outcome: str,
    score: str,
    covariates: list[str],
) -> dict:
    """
    Estimate the adjusted association between a standardized
    retinal score and a standardized CMR phenotype.

    Beta is the change in phenotype SD per 1-SD higher score.
    HC3 robust standard errors are used.
    """

    result = {
        "phenotype": outcome,
        "N": 0,
        "beta": np.nan,
        "CI_low": np.nan,
        "CI_high": np.nan,
        "p": np.nan,
    }

    data, predictor_columns = (
        _prepare_linear_model_data(
            df,
            outcome=outcome,
            predictors=[
                score,
                *covariates,
            ],
        )
    )

    result["N"] = len(data)

    if len(data) < 30:
        return result

    if data[outcome].nunique() < 5:
        return result

    if score not in predictor_columns:
        return result

    outcome_sd = data[outcome].std(ddof=0)
    score_sd = data[score].std(ddof=0)

    if (
        not np.isfinite(outcome_sd)
        or not np.isfinite(score_sd)
        or outcome_sd <= 1e-12
        or score_sd <= 1e-12
    ):
        return result

    data[outcome] = (
        data[outcome] - data[outcome].mean()
    ) / outcome_sd

    data[score] = (
        data[score] - data[score].mean()
    ) / score_sd

    design = data[
        predictor_columns
    ].astype(np.float64)

    design = sm.add_constant(
        design,
        has_constant="add",
    ).astype(np.float64)

    response = data[outcome].astype(
        np.float64
    )

    model = sm.OLS(
        response,
        design,
        missing="raise",
    ).fit(
        cov_type="HC3"
    )

    interval = model.conf_int().loc[score]

    result.update(
        {
            "beta": float(
                model.params.loc[score]
            ),
            "CI_low": float(
                interval.iloc[0]
            ),
            "CI_high": float(
                interval.iloc[1]
            ),
            "p": float(
                model.pvalues.loc[score]
            ),
        }
    )

    return result


def association_table(
    df: pd.DataFrame,
    phenotypes: list[str],
    score: str,
    covariates: list[str],
) -> pd.DataFrame:
    rows = [
        adjusted_association(
            df,
            phenotype,
            score,
            covariates,
        )
        for phenotype in phenotypes
    ]

    table = pd.DataFrame(rows)

    valid = table["p"].notna()

    if valid.any():
        table.loc[valid, "q"] = multipletests(
            table.loc[valid, "p"],
            method="fdr_bh",
        )[1]

    return table


def acat(p_values) -> float:
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    p_values = p_values[
        np.isfinite(p_values)
    ]

    if len(p_values) == 0:
        return np.nan

    p_values = np.clip(
        p_values,
        1e-15,
        1 - 1e-15,
    )

    statistic = np.mean(
        np.tan(
            (0.5 - p_values) * np.pi
        )
    )

    return float(
        0.5
        - np.arctan(statistic) / np.pi
    )


def domain_acat(
    table: pd.DataFrame,
    domains: dict[str, list[str]],
    p_column: str = "p",
) -> pd.DataFrame:
    rows = []

    for domain, phenotypes in domains.items():
        subset = table.loc[
            table["phenotype"].isin(
                phenotypes
            )
        ]

        rows.append(
            {
                "domain": domain,
                "n_phenotypes": int(
                    subset[p_column]
                    .notna()
                    .sum()
                ),
                "ACAT_p": acat(
                    subset[p_column]
                ),
            }
        )

    result = pd.DataFrame(rows)

    valid = result["ACAT_p"].notna()

    if valid.any():
        result.loc[valid, "q"] = (
            multipletests(
                result.loc[
                    valid,
                    "ACAT_p",
                ],
                method="fdr_bh",
            )[1]
        )

    return result


def residualized_domain_pc(
    df: pd.DataFrame,
    phenotypes: list[str],
    covariates: list[str],
) -> pd.Series:
    """
    Residualize each phenotype against covariates and calculate
    an unsupervised PC1 from the complete residual matrix.
    """

    residuals = []

    for phenotype in phenotypes:
        series = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
            name=phenotype,
        )

        data, predictor_columns = (
            _prepare_linear_model_data(
                df,
                outcome=phenotype,
                predictors=covariates,
            )
        )

        if (
            len(data) >= 30
            and data[phenotype].nunique() >= 5
        ):
            if predictor_columns:
                design = data[
                    predictor_columns
                ].astype(np.float64)

                design = sm.add_constant(
                    design,
                    has_constant="add",
                ).astype(np.float64)
            else:
                design = pd.DataFrame(
                    {
                        "const": np.ones(
                            len(data),
                            dtype=float,
                        )
                    },
                    index=data.index,
                )

            response = data[
                phenotype
            ].astype(np.float64)

            model = sm.OLS(
                response,
                design,
                missing="raise",
            ).fit()

            series.loc[data.index] = (
                np.asarray(
                    model.resid,
                    dtype=float,
                )
            )

        residuals.append(series)

    matrix = pd.concat(
        residuals,
        axis=1,
    )

    usable = [
        column
        for column in matrix.columns
        if matrix[column].notna().sum() >= 30
    ]

    matrix = matrix[usable]

    score = pd.Series(
        np.nan,
        index=df.index,
        dtype=float,
    )

    if len(usable) < 2:
        return score

    complete = matrix.notna().all(axis=1)

    if complete.sum() < 30:
        return score

    scaled = StandardScaler().fit_transform(
        matrix.loc[complete]
    )

    score.loc[complete] = (
        PCA(n_components=1)
        .fit_transform(scaled)
        .reshape(-1)
    )

    return score


def domain_pc_table(
    df: pd.DataFrame,
    domains: dict[str, list[str]],
    score: str,
    covariates: list[str],
):
    score_values = pd.to_numeric(
        df[score],
        errors="coerce",
    )

    rows = []

    for domain, phenotypes in domains.items():
        pc = residualized_domain_pc(
            df,
            phenotypes,
            covariates,
        )

        valid = (
            pc.notna()
            & score_values.notna()
        )

        if valid.sum() >= 30:
            rho, p = spearmanr(
                pc.loc[valid],
                score_values.loc[valid],
            )
        else:
            rho, p = np.nan, np.nan

        rows.append(
            {
                "domain": domain,
                "N": int(valid.sum()),
                "rho": float(rho)
                if np.isfinite(rho)
                else np.nan,
                "p": float(p)
                if np.isfinite(p)
                else np.nan,
            }
        )

    table = pd.DataFrame(rows)

    valid = table["p"].notna()

    if valid.any():
        table.loc[valid, "q"] = multipletests(
            table.loc[valid, "p"],
            method="fdr_bh",
        )[1]

        global_p = combine_pvalues(
            table.loc[valid, "p"],
            method="fisher",
        ).pvalue
    else:
        global_p = np.nan

    return table, float(global_p)

def signed_domain_association_table(
    df: pd.DataFrame,
    signed_domains: dict[str, dict[str, float]],
    score: str,
    covariates: list[str],
    require_complete: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build biologically signed standardized domain composites and test their
    adjusted association with a retinal score.

    Example direction: -1 for LVEF and +1 for LV volumes/mass.
    """
    out = df.copy()
    rows = []

    for domain, directions in signed_domains.items():
        missing = [
            phenotype for phenotype in directions
            if phenotype not in out.columns
        ]
        if missing:
            rows.append(
                {
                    "phenotype": domain,
                    "N": 0,
                    "beta": np.nan,
                    "CI_low": np.nan,
                    "CI_high": np.nan,
                    "p": np.nan,
                    "missing_components": ", ".join(missing),
                }
            )
            continue

        component_columns = []
        for phenotype, direction in directions.items():
            values = pd.to_numeric(out[phenotype], errors="coerce")
            standard_deviation = values.std(ddof=0)
            component = f"__signed__{domain}__{phenotype}"
            if (
                not np.isfinite(standard_deviation)
                or standard_deviation <= 1e-12
            ):
                out[component] = np.nan
            else:
                out[component] = (
                    float(direction)
                    * (values - values.mean())
                    / standard_deviation
                )
            component_columns.append(component)

        out[domain] = out[component_columns].mean(
            axis=1,
            skipna=not require_complete,
        )
        result = adjusted_association(
            out,
            outcome=domain,
            score=score,
            covariates=covariates,
        )
        result["missing_components"] = ""
        result["n_components"] = len(component_columns)
        rows.append(result)

    table = pd.DataFrame(rows)
    valid = table["p"].notna()
    if valid.any():
        table.loc[valid, "q"] = multipletests(
            table.loc[valid, "p"],
            method="fdr_bh",
        )[1]
    return table, out
