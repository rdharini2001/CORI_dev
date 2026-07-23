from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from scipy.stats import chi2, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .data import truncate_followup


def c_index(df: pd.DataFrame, score_column: str, time_column: str = "time_years", event_column: str = "Y_mace") -> float:
    d = df[[time_column, event_column, score_column]].dropna()
    return float(concordance_index(d[time_column], -d[score_column], d[event_column]))


def bootstrap_c_index(
    df: pd.DataFrame,
    score_column: str,
    n_bootstraps: int = 1000,
    seed: int = 20260714,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(n_bootstraps):
        sample = df.iloc[rng.integers(0, len(df), len(df))]
        if sample["Y_mace"].nunique() < 2:
            continue
        estimates.append(c_index(sample, score_column))
    estimates = np.asarray(estimates)
    return {
        "C_index": c_index(df, score_column),
        "C_low": float(np.quantile(estimates, 0.025)),
        "C_high": float(np.quantile(estimates, 0.975)),
        "successful_bootstraps": int(len(estimates)),
    }


def paired_delta_c(
    df: pd.DataFrame,
    score_a: str,
    score_b: str,
    n_bootstraps: int = 1000,
    seed: int = 20260714,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstraps):
        sample = df.iloc[rng.integers(0, len(df), len(df))]
        if sample["Y_mace"].nunique() < 2:
            continue
        deltas.append(c_index(sample, score_a) - c_index(sample, score_b))
    deltas = np.asarray(deltas)
    estimate = c_index(df, score_a) - c_index(df, score_b)
    return {
        "delta_C": estimate,
        "delta_low": float(np.quantile(deltas, 0.025)),
        "delta_high": float(np.quantile(deltas, 0.975)),
        "p_two_sided": float(2 * min(np.mean(deltas <= 0), np.mean(deltas >= 0))),
    }


def fit_cox_columns(
    train_df: pd.DataFrame,
    columns: list[str],
    penalizer: float = 0.05,
    weight_column: str | None = None,
):
    data = train_df[["time_years", "Y_mace", *columns] + ([weight_column] if weight_column else [])].copy()
    data[columns] = data[columns].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["time_years", "Y_mace", *columns])

    arguments = {"duration_col": "time_years", "event_col": "Y_mace"}
    if weight_column:
        arguments["weights_col"] = weight_column
        arguments["robust"] = True

    model = CoxPHFitter(penalizer=penalizer)
    model.fit(data, **arguments)
    return model


def score_cox_model(model: CoxPHFitter, df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    design = df[columns].apply(pd.to_numeric, errors="coerce")
    design = design.fillna(design.median())
    return np.asarray(model.predict_log_partial_hazard(design)).reshape(-1)


def predicted_risk(model: CoxPHFitter, df: pd.DataFrame, columns: list[str], horizon: float = 10.0) -> np.ndarray:
    design = df[columns].apply(pd.to_numeric, errors="coerce")
    design = design.fillna(design.median())
    survival = model.predict_survival_function(design, times=[horizon]).iloc[0].to_numpy()
    return 1 - survival


def likelihood_ratio_test(reduced_model: CoxPHFitter, full_model: CoxPHFitter, degrees_of_freedom: int) -> dict[str, float]:
    statistic = 2 * (full_model.log_likelihood_ - reduced_model.log_likelihood_)
    return {
        "LR_chi2": float(statistic),
        "df": int(degrees_of_freedom),
        "p": float(chi2.sf(statistic, degrees_of_freedom)),
    }


def high_low_hr(df: pd.DataFrame, group_column: str) -> dict[str, float]:
    data = df[["time_years", "Y_mace", group_column]].dropna().copy()
    model = CoxPHFitter(penalizer=0.0)
    model.fit(data, duration_col="time_years", event_col="Y_mace")
    row = model.summary.loc[group_column]
    return {
        "HR": float(np.exp(row["coef"])),
        "HR_low": float(np.exp(row["coef lower 95%"])),
        "HR_high": float(np.exp(row["coef upper 95%"])),
        "p": float(row["p"]),
    }


def performance_row(df: pd.DataFrame, cohort: str, score_column: str, high_column: str, n_bootstraps: int = 1000) -> dict:
    c = bootstrap_c_index(df, score_column, n_bootstraps=n_bootstraps)
    hr = high_low_hr(df, high_column)
    return {
        "cohort": cohort,
        "N": len(df),
        "events": int(df["Y_mace"].sum()),
        **c,
        **hr,
    }


def horizon_table(df: pd.DataFrame, score_column: str, high_column: str, horizons=(3, 5, 10)) -> pd.DataFrame:
    rows = []
    for horizon in horizons:
        truncated = truncate_followup(df, horizon).rename(
            columns={"time_horizon": "time_original", "event_horizon": "event_original"}
        )
        analysis = truncated.copy()
        analysis["time_years"] = analysis["time_original"]
        analysis["Y_mace"] = analysis["event_original"]
        row = performance_row(analysis, f"{horizon}-year", score_column, high_column, n_bootstraps=500)
        row["horizon_years"] = horizon
        rows.append(row)
    return pd.DataFrame(rows)


def baseline_table(df: pd.DataFrame, group_column: str, continuous: list[str], categorical: list[str]) -> pd.DataFrame:
    rows = []
    groups = ["Overall", *df[group_column].dropna().astype(str).unique().tolist()]
    for variable in continuous:
        row = {"variable": variable}
        for group in groups:
            values = df[variable] if group == "Overall" else df.loc[df[group_column].astype(str) == group, variable]
            values = pd.to_numeric(values, errors="coerce")
            row[group] = f"{values.mean():.1f} ({values.std(ddof=1):.1f})"
        rows.append(row)
    for variable in categorical:
        row = {"variable": variable}
        for group in groups:
            values = df[variable] if group == "Overall" else df.loc[df[group_column].astype(str) == group, variable]
            values = pd.to_numeric(values, errors="coerce").fillna(0)
            row[group] = f"{int(values.sum()):,} ({100 * values.mean():.1f}%)"
        rows.append(row)
    return pd.DataFrame(rows)


def reclassification_tables(
    df: pd.DataFrame,
    old_group: str,
    new_group: str,
    event_column: str = "Y_mace",
    value_column: str | None = None,
):
    counts = pd.crosstab(df[old_group], df[new_group])
    event_rates = pd.pivot_table(df, index=old_group, columns=new_group, values=event_column, aggfunc="mean")
    values = None
    if value_column is not None:
        values = pd.pivot_table(df, index=old_group, columns=new_group, values=value_column, aggfunc="mean")
    return counts, event_rates, values


def categorical_nri(df: pd.DataFrame, old_group: str, new_group: str, event_column: str = "Y_mace") -> dict[str, float]:
    order = {"Low": 0, "Middle": 1, "High": 2}
    old = df[old_group].map(order)
    new = df[new_group].map(order)
    event = df[event_column].astype(int)

    event_up = ((new > old) & (event == 1)).sum() / max((event == 1).sum(), 1)
    event_down = ((new < old) & (event == 1)).sum() / max((event == 1).sum(), 1)
    nonevent_down = ((new < old) & (event == 0)).sum() / max((event == 0).sum(), 1)
    nonevent_up = ((new > old) & (event == 0)).sum() / max((event == 0).sum(), 1)

    event_nri = event_up - event_down
    nonevent_nri = nonevent_down - nonevent_up
    return {
        "event_NRI": float(event_nri),
        "nonevent_NRI": float(nonevent_nri),
        "total_NRI": float(event_nri + nonevent_nri),
    }


def idi(df: pd.DataFrame, old_risk: str, new_risk: str, event_column: str = "Y_mace") -> dict[str, float]:
    event = df[event_column].astype(int)
    old_discrimination = df.loc[event == 1, old_risk].mean() - df.loc[event == 0, old_risk].mean()
    new_discrimination = df.loc[event == 1, new_risk].mean() - df.loc[event == 0, new_risk].mean()
    return {
        "old_discrimination_slope": float(old_discrimination),
        "new_discrimination_slope": float(new_discrimination),
        "IDI": float(new_discrimination - old_discrimination),
    }


def residualize(train_df: pd.DataFrame, apply_df: pd.DataFrame, target: str, reference: str, output: str) -> tuple[pd.DataFrame, dict]:
    x = np.column_stack([np.ones(len(train_df)), train_df[reference].to_numpy(dtype=float)])
    y = train_df[target].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    out = apply_df.copy()
    out[output] = out[target] - (coefficients[0] + coefficients[1] * out[reference])
    return out, {"intercept": float(coefficients[0]), "slope": float(coefficients[1])}





def adjusted_cox(
    df: pd.DataFrame,
    score_column: str,
    covariates: list[str],
    interaction_column: str | None = None,
    penalizer: float = 0.01,
) -> tuple[pd.DataFrame, CoxPHFitter]:
    """
    Fit an adjusted Cox model.

    Constant covariates are removed automatically within the
    analysis subset. This is necessary for subgroup analyses,
    where variables such as sex may be constant by definition.
    """

    model_columns = [
        "time_years",
        "Y_mace",
        score_column,
        *covariates,
    ]

    if interaction_column is not None:
        model_columns.append(interaction_column)

    # Preserve order while removing repeated column names.
    model_columns = list(dict.fromkeys(model_columns))

    missing_columns = [
        column
        for column in model_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Missing columns for adjusted Cox model: "
            f"{missing_columns}"
        )

    data = df[model_columns].copy()

    # Center is categorical; all other model variables are numeric.
    numeric_columns = [
        column
        for column in model_columns
        if column != "center"
    ]

    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    data = (
        data
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    if interaction_column is not None:
        interaction_term = (
            f"{score_column}_x_{interaction_column}"
        )

        data[interaction_term] = (
            data[score_column]
            * data[interaction_column]
        )

    if "center" in covariates:
        data = pd.get_dummies(
            data,
            columns=["center"],
            prefix="center",
            drop_first=True,
            dtype=float,
        )

    candidate_columns = [
        column
        for column in data.columns
        if column not in {
            "time_years",
            "Y_mace",
        }
    ]

    # Remove non-estimable zero-variance columns.
    varying_columns = []
    dropped_constant = []

    for column in candidate_columns:
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        variance = values.var(ddof=0)

        if (
            values.nunique(dropna=True) >= 2
            and np.isfinite(variance)
            and variance > 1e-12
        ):
            varying_columns.append(column)
        else:
            dropped_constant.append(column)

    if score_column not in varying_columns:
        raise ValueError(
            f"{score_column} has no variation in this subset."
        )

    model_data = data[
        [
            "time_years",
            "Y_mace",
            *varying_columns,
        ]
    ].copy()

    if model_data["Y_mace"].sum() == 0:
        raise ValueError(
            "The analysis subset contains no MACE events."
        )

    model = CoxPHFitter(
        penalizer=penalizer,
    )

    model.fit(
        model_data,
        duration_col="time_years",
        event_col="Y_mace",
    )

    summary = (
        model.summary
        .reset_index()
        .rename(columns={"covariate": "term"})
    )

    # Support lifelines versions whose reset index is named "index".
    if "term" not in summary.columns:
        summary = summary.rename(
            columns={summary.columns[0]: "term"}
        )

    summary["HR"] = np.exp(summary["coef"])
    summary["HR_low"] = np.exp(
        summary["coef lower 95%"]
    )
    summary["HR_high"] = np.exp(
        summary["coef upper 95%"]
    )

    summary["N"] = len(model_data)
    summary["events"] = int(
        model_data["Y_mace"].sum()
    )
    summary["adjusted_for"] = ", ".join(
        column
        for column in varying_columns
        if column != score_column
        and not column.startswith(
            f"{score_column}_x_"
        )
    )
    summary["dropped_constant"] = ", ".join(
        dropped_constant
    )

    return summary, model


def subgroup_cox(
    df: pd.DataFrame,
    score_column: str,
    subgroup_column: str,
    minimum_events: int = 20,
    covariates: list[str] | None = None,
    penalizer: float = 0.01,
) -> pd.DataFrame:
    """
    Estimate the score association separately within each subgroup.

    Each level is fit with train_cox_model(), reusing its Cox fit and
    forest_df summary. Covariates that are constant within a level
    (e.g. sex inside a sex subgroup) are dropped before fitting.
    """
    from src2.train import train_cox_model

    if covariates is None:
        covariates = ["age", "female", "height", "HTN", "Diabetes"]

    if subgroup_column not in df.columns:
        raise KeyError(f"Subgroup column not found: {subgroup_column}")

    analysis_columns = list(dict.fromkeys(["time_years", "Y_mace", score_column, *covariates]))

    rows = []

    for level, subset in df.groupby(subgroup_column, observed=True, dropna=True):
        complete_subset = (
            subset[analysis_columns]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        events = int(complete_subset["Y_mace"].sum())

        if events < minimum_events:
            continue

        if complete_subset[score_column].nunique(dropna=True) < 2:
            continue

        varying_covariates = [
            c for c in covariates
            if complete_subset[c].nunique(dropna=True) >= 2
            and complete_subset[c].var(ddof=0) > 1e-12
        ]
        dropped_constant = [c for c in covariates if c not in varying_covariates]

        result = train_cox_model(
            complete_subset,
            complete_subset,
            selected_features=[score_column, *varying_covariates],
            time_col="time_years",
            event_col="Y_mace",
            penalizer=penalizer,
            verbose=False,
            plot_km=False,
        )

        score_row = result["forest_df"].loc[
            result["forest_df"]["covariate"] == score_column
        ].iloc[0]

        rows.append(
            {
                "subgroup": subgroup_column,
                "level": str(level),
                "N": len(complete_subset),
                "events": events,
                "HR": float(score_row["hr"]),
                "HR_low": float(score_row["ci_low"]),
                "HR_high": float(score_row["ci_high"]),
                "p": float(score_row["p_value"]),
                "adjusted_for": ", ".join(varying_covariates),
                "dropped_constant": ", ".join(dropped_constant),
            }
        )

    return pd.DataFrame(rows)


def stratified_incremental_test(
    df: pd.DataFrame,
    base_score: str,
    added_score: str,
    covariates: list[str],
    strata_column: str = "center",
) -> dict[str, float]:
    """Compare nested Cox models using an unpenalized, center-stratified likelihood ratio test."""
    numeric_covariates = [
        column for column in covariates if column != strata_column
    ]
    columns = [
        "time_years",
        "Y_mace",
        base_score,
        added_score,
        *numeric_covariates,
        strata_column,
    ]
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for incremental Cox test: {missing}")

    data = df[columns].copy()
    for column in columns:
        if column != strata_column:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()

    candidate_terms = [base_score, added_score, *numeric_covariates]
    varying_terms = [
        column
        for column in candidate_terms
        if data[column].nunique(dropna=True) > 1
        and np.isfinite(data[column].std(ddof=0))
        and data[column].std(ddof=0) > 1e-12
    ]
    if base_score not in varying_terms or added_score not in varying_terms:
        raise ValueError("Both base and added scores must vary in the analysis cohort.")

    reduced_terms = [base_score, *[
        column for column in numeric_covariates if column in varying_terms
    ]]
    full_terms = [base_score, added_score, *[
        column for column in numeric_covariates if column in varying_terms
    ]]

    reduced_data = data[
        ["time_years", "Y_mace", *reduced_terms, strata_column]
    ].copy()
    full_data = data[
        ["time_years", "Y_mace", *full_terms, strata_column]
    ].copy()

    reduced = CoxPHFitter(penalizer=0.0)
    reduced.fit(
        reduced_data,
        duration_col="time_years",
        event_col="Y_mace",
        strata=[strata_column],
        robust=True,
    )

    full = CoxPHFitter(penalizer=0.0)
    full.fit(
        full_data,
        duration_col="time_years",
        event_col="Y_mace",
        strata=[strata_column],
        robust=True,
    )

    statistic = 2 * (full.log_likelihood_ - reduced.log_likelihood_)
    row = full.summary.loc[added_score]
    reduced_score = np.asarray(
        reduced.predict_log_partial_hazard(reduced_data[reduced_terms])
    ).reshape(-1)
    full_score = np.asarray(
        full.predict_log_partial_hazard(full_data[full_terms])
    ).reshape(-1)

    evaluation = data[["time_years", "Y_mace"]].copy()
    evaluation["reduced_score"] = reduced_score
    evaluation["full_score"] = full_score

    return {
        "base_score": base_score,
        "added_score": added_score,
        "N": int(len(data)),
        "events": int(data["Y_mace"].sum()),
        "added_HR": float(np.exp(row["coef"])),
        "added_HR_low": float(np.exp(row["coef lower 95%"])),
        "added_HR_high": float(np.exp(row["coef upper 95%"])),
        "added_p": float(row["p"]),
        "LR_chi2": float(statistic),
        "LR_p": float(chi2.sf(statistic, 1)),
        "C_reduced": c_index(evaluation, "reduced_score"),
        "C_full": c_index(evaluation, "full_score"),
        "delta_C": c_index(evaluation, "full_score")
        - c_index(evaluation, "reduced_score"),
    }


def partial_spearman(
    df: pd.DataFrame,
    x: str,
    y: str,
    covariates: list[str],
) -> dict[str, float]:
    """Partial Spearman correlation obtained by residualizing ranked variables."""
    columns = list(dict.fromkeys([x, y, *covariates]))
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for partial Spearman analysis: {missing}")

    data = df[columns].copy()
    categorical = [column for column in covariates if column == "center"]
    numeric = [column for column in columns if column not in categorical]

    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    data = data.replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(data) < 30 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return {"N": int(len(data)), "rho": np.nan, "p": np.nan}

    design_parts = []
    continuous_covariates = [
        column for column in covariates if column not in categorical
    ]
    if continuous_covariates:
        design_parts.append(data[continuous_covariates].astype(float))
    if categorical:
        design_parts.append(
            pd.get_dummies(
                data[categorical],
                drop_first=True,
                dtype=float,
            )
        )
    if design_parts:
        design = pd.concat(design_parts, axis=1)
        varying = [
            column
            for column in design.columns
            if design[column].nunique() > 1
            and design[column].std(ddof=0) > 1e-12
        ]
        design = design[varying]
        design = sm.add_constant(design.astype(float), has_constant="add")
    else:
        design = pd.DataFrame(
            {"const": np.ones(len(data), dtype=float)},
            index=data.index,
        )

    rank_x = data[x].rank(method="average").astype(float)
    rank_y = data[y].rank(method="average").astype(float)
    residual_x = sm.OLS(rank_x, design).fit().resid
    residual_y = sm.OLS(rank_y, design).fit().resid
    rho, p_value = pearsonr(residual_x, residual_y)
    return {"N": int(len(data)), "rho": float(rho), "p": float(p_value)}


def handcrafted_replication_table(
    development: pd.DataFrame,
    heldout: pd.DataFrame,
    score_column: str,
    feature_columns: list[str],
    covariates: list[str],
    top_n: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select score-morphology correlates in development and replicate them in held-out data."""
    rows = []
    for feature in feature_columns:
        development_result = partial_spearman(
            development, score_column, feature, covariates
        )
        heldout_result = partial_spearman(
            heldout, score_column, feature, covariates
        )
        raw = heldout[[score_column, feature]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        if len(raw) >= 30 and raw[score_column].nunique() > 1 and raw[feature].nunique() > 1:
            raw_rho, raw_p = spearmanr(raw[score_column], raw[feature])
        else:
            raw_rho, raw_p = np.nan, np.nan

        rows.append(
            {
                "feature": feature,
                "development_N": development_result["N"],
                "development_partial_rho": development_result["rho"],
                "development_p": development_result["p"],
                "heldout_N": heldout_result["N"],
                "heldout_partial_rho": heldout_result["rho"],
                "heldout_p": heldout_result["p"],
                "heldout_raw_rho": float(raw_rho) if np.isfinite(raw_rho) else np.nan,
                "heldout_raw_p": float(raw_p) if np.isfinite(raw_p) else np.nan,
            }
        )

    table = pd.DataFrame(rows)
    valid = table["heldout_p"].notna()
    if valid.any():
        table.loc[valid, "heldout_q"] = multipletests(
            table.loc[valid, "heldout_p"],
            method="fdr_bh",
        )[1]

    table["abs_development_rho"] = table["development_partial_rho"].abs()
    selected_features = (
        table.sort_values("abs_development_rho", ascending=False)
        .head(top_n)["feature"]
        .tolist()
    )
    selected = table.loc[table["feature"].isin(selected_features)].copy()
    selected["same_direction"] = (
        np.sign(selected["development_partial_rho"])
        == np.sign(selected["heldout_partial_rho"])
    )
    selected = selected.sort_values(
        "abs_development_rho", ascending=False
    ).reset_index(drop=True)
    return table.drop(columns="abs_development_rho"), selected.drop(
        columns="abs_development_rho"
    )
