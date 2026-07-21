from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy.stats import norm
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from lifelines.utils import concordance_index


@dataclass
class LockedCoxModel:
    name: str
    feature_columns: list[str]
    medians: pd.Series
    means: pd.Series
    standard_deviations: pd.Series
    score_mean: float
    score_standard_deviation: float
    thresholds: dict[str, float]
    penalizer: float
    cox_model: CoxPHFitter
    ranking: pd.DataFrame


def _usable_features(df: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    usable = []
    for column in feature_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.notna().mean() < 0.50:
            continue
        if values.var(skipna=True) <= 1e-8:
            continue
        usable.append(column)
    return usable


def _standardized_matrix(df: pd.DataFrame, feature_columns: list[str]):
    values = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    medians = values.median()
    values = values.fillna(medians)
    means = values.mean()
    standard_deviations = values.std(ddof=0).replace(0, 1)
    matrix = ((values - means) / standard_deviations).to_numpy(dtype=float)
    return matrix, medians, means, standard_deviations


def _univariate_cox_chunk(
    matrix: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    penalizer: float,
    max_iterations: int = 30,
    tolerance: float = 1e-7,
):
    """Fit independent univariate Cox models together using Breslow ties."""
    order = np.argsort(-time, kind="mergesort")
    x = matrix[order]
    t = time[order]
    e = event[order].astype(float)

    starts = np.r_[0, np.flatnonzero(t[1:] != t[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(t) - 1]
    deaths = np.add.reduceat(e, starts)
    event_x = np.add.reduceat(e[:, None] * x, starts, axis=0)

    beta = np.zeros(x.shape[1], dtype=float)

    for _ in range(max_iterations):
        eta = np.clip(x * beta[None, :], -40, 40)
        risk = np.exp(eta)
        s0 = np.cumsum(risk, axis=0)[ends]
        s1 = np.cumsum(risk * x, axis=0)[ends]
        s2 = np.cumsum(risk * x * x, axis=0)[ends]

        mean = s1 / np.maximum(s0, 1e-12)
        information = np.sum(
            deaths[:, None] * (s2 / np.maximum(s0, 1e-12) - mean * mean),
            axis=0,
        )
        score = np.sum(event_x - deaths[:, None] * mean, axis=0)

        score -= penalizer * beta
        information += penalizer
        step = score / np.maximum(information, 1e-12)
        beta_new = beta + np.clip(step, -1, 1)

        if np.max(np.abs(beta_new - beta)) < tolerance:
            beta = beta_new
            break
        beta = beta_new

    eta = np.clip(x * beta[None, :], -40, 40)
    risk = np.exp(eta)
    s0 = np.cumsum(risk, axis=0)[ends]
    s1 = np.cumsum(risk * x, axis=0)[ends]
    s2 = np.cumsum(risk * x * x, axis=0)[ends]
    mean = s1 / np.maximum(s0, 1e-12)
    information = np.sum(
        deaths[:, None] * (s2 / np.maximum(s0, 1e-12) - mean * mean),
        axis=0,
    ) + penalizer

    standard_error = 1 / np.sqrt(np.maximum(information, 1e-12))
    z = beta / standard_error
    p = 2 * norm.sf(np.abs(z))
    return beta, standard_error, z, p


def rank_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    penalizer: float = 0.01,
    chunk_size: int = 128,
) -> pd.DataFrame:
    """Rank features by absolute univariate Cox z statistic."""
    usable = _usable_features(df, feature_columns)
    matrix, _, _, _ = _standardized_matrix(df, usable)
    time = pd.to_numeric(df["time_years"], errors="raise").to_numpy(dtype=float)
    event = pd.to_numeric(df["Y_mace"], errors="raise").to_numpy(dtype=int)

    rows = []
    for start in range(0, len(usable), chunk_size):
        columns = usable[start : start + chunk_size]
        beta, standard_error, z, p = _univariate_cox_chunk(
            matrix[:, start : start + len(columns)],
            time,
            event,
            penalizer,
        )
        for index, column in enumerate(columns):
            rows.append(
                {
                    "feature": column,
                    "coef": beta[index],
                    "se": standard_error[index],
                    "z": z[index],
                    "abs_z": abs(z[index]),
                    "p": p[index],
                }
            )

    return pd.DataFrame(rows).sort_values("abs_z", ascending=False).reset_index(drop=True)


def _design_from_training(df: pd.DataFrame, model: LockedCoxModel) -> pd.DataFrame:
    values = df[model.feature_columns].apply(pd.to_numeric, errors="coerce")
    values = values.fillna(model.medians)
    return (values - model.means) / model.standard_deviations


def train_model(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    name: str,
    n_features: int,
    penalizer: float,
    rank_penalizer: float = 0.01,
    weight_column: str | None = None,
) -> LockedCoxModel:
    """Rank, select, fit and freeze one Cox model."""
    ranking = rank_features(train_df, feature_columns, penalizer=rank_penalizer)
    selected = ranking.head(n_features)["feature"].tolist()
    matrix, medians, means, standard_deviations = _standardized_matrix(train_df, selected)

    fit_data = pd.DataFrame(matrix, columns=selected, index=train_df.index)
    fit_data["time_years"] = train_df["time_years"].to_numpy()
    fit_data["Y_mace"] = train_df["Y_mace"].to_numpy()

    fit_arguments = {
        "duration_col": "time_years",
        "event_col": "Y_mace",
        "show_progress": False,
    }
    if weight_column is not None:
        fit_data[weight_column] = train_df[weight_column].to_numpy()
        fit_arguments["weights_col"] = weight_column
        fit_arguments["robust"] = True

    cox_model = CoxPHFitter(penalizer=penalizer)
    cox_model.fit(fit_data, **fit_arguments)

    raw_score = np.asarray(cox_model.predict_log_partial_hazard(fit_data)).reshape(-1)
    score_mean = float(np.mean(raw_score))
    score_standard_deviation = float(np.std(raw_score, ddof=0))
    if score_standard_deviation == 0:
        score_standard_deviation = 1.0
    score_z = (raw_score - score_mean) / score_standard_deviation

    thresholds = {
        "median": float(np.quantile(score_z, 0.50)),
        "tertile_1": float(np.quantile(score_z, 1 / 3)),
        "tertile_2": float(np.quantile(score_z, 2 / 3)),
    }

    return LockedCoxModel(
        name=name,
        feature_columns=selected,
        medians=medians,
        means=means,
        standard_deviations=standard_deviations,
        score_mean=score_mean,
        score_standard_deviation=score_standard_deviation,
        thresholds=thresholds,
        penalizer=penalizer,
        cox_model=cox_model,
        ranking=ranking,
    )


def score_model(model: LockedCoxModel, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    design = _design_from_training(out, model)
    raw_score = np.asarray(model.cox_model.predict_log_partial_hazard(design)).reshape(-1)
    score_z = (raw_score - model.score_mean) / model.score_standard_deviation

    out[f"{model.name}_raw"] = raw_score
    out[f"{model.name}_z"] = score_z
    out[f"{model.name}_high"] = (score_z > model.thresholds["median"]).astype(int)
    out[f"{model.name}_tertile"] = pd.cut(
        score_z,
        bins=[-np.inf, model.thresholds["tertile_1"], model.thresholds["tertile_2"], np.inf],
        labels=["Low", "Middle", "High"],
        include_lowest=True,
    ).astype(str)
    return out


def save_model(model: LockedCoxModel, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(model, handle)


def load_model(path: str | Path) -> LockedCoxModel:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def score_values(model: LockedCoxModel, df: pd.DataFrame) -> np.ndarray:
    design = _design_from_training(df, model)
    raw_score = np.asarray(model.cox_model.predict_log_partial_hazard(design)).reshape(-1)
    return (raw_score - model.score_mean) / model.score_standard_deviation


def predict_risk(model: LockedCoxModel, df: pd.DataFrame, horizon: float = 10.0) -> np.ndarray:
    design = _design_from_training(df, model)
    survival = model.cox_model.predict_survival_function(design, times=[horizon]).iloc[0].to_numpy()
    return 1 - survival


def fit_model_from_ranking(
    train_df: pd.DataFrame,
    ranking: pd.DataFrame,
    name: str,
    n_features: int,
    penalizer: float,
    weight_column: str | None = None,
) -> LockedCoxModel:
    """Fit and freeze a Cox model from an already computed development ranking."""
    selected = ranking.head(min(n_features, len(ranking)))["feature"].tolist()
    if not selected:
        raise ValueError("No usable features were available for model fitting.")

    matrix, medians, means, standard_deviations = _standardized_matrix(
        train_df, selected
    )
    fit_data = pd.DataFrame(matrix, columns=selected, index=train_df.index)
    fit_data["time_years"] = pd.to_numeric(
        train_df["time_years"], errors="raise"
    ).to_numpy(dtype=float)
    fit_data["Y_mace"] = pd.to_numeric(
        train_df["Y_mace"], errors="raise"
    ).to_numpy(dtype=int)

    fit_arguments = {
        "duration_col": "time_years",
        "event_col": "Y_mace",
        "show_progress": False,
    }
    if weight_column is not None:
        fit_data[weight_column] = pd.to_numeric(
            train_df[weight_column], errors="raise"
        ).to_numpy(dtype=float)
        fit_arguments["weights_col"] = weight_column
        fit_arguments["robust"] = True

    cox_model = CoxPHFitter(penalizer=penalizer)
    cox_model.fit(fit_data, **fit_arguments)

    raw_score = np.asarray(
        cox_model.predict_log_partial_hazard(fit_data)
    ).reshape(-1)
    score_mean = float(raw_score.mean())
    score_standard_deviation = float(raw_score.std(ddof=0))
    if not np.isfinite(score_standard_deviation) or score_standard_deviation <= 0:
        score_standard_deviation = 1.0

    score_z = (raw_score - score_mean) / score_standard_deviation
    thresholds = {
        "median": float(np.quantile(score_z, 0.50)),
        "tertile_1": float(np.quantile(score_z, 1 / 3)),
        "tertile_2": float(np.quantile(score_z, 2 / 3)),
    }

    return LockedCoxModel(
        name=name,
        feature_columns=selected,
        medians=medians,
        means=means,
        standard_deviations=standard_deviations,
        score_mean=score_mean,
        score_standard_deviation=score_standard_deviation,
        thresholds=thresholds,
        penalizer=penalizer,
        cox_model=cox_model,
        ranking=ranking.copy(),
    )


def _cv_strata_labels(
    df: pd.DataFrame,
    folds: int,
    center_column: str = "center",
) -> pd.Series:
    """Use center-by-event strata when sufficiently populated; otherwise event only."""
    event = pd.to_numeric(df["Y_mace"], errors="raise").astype(int).astype(str)
    if center_column in df.columns:
        combined = df[center_column].astype(str) + "__" + event
        if combined.value_counts().min() >= folds:
            return combined
    return event


def tune_model_cv(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    name: str,
    candidate_k: list[int],
    candidate_penalizers: list[float],
    folds: int = 3,
    repeats: int = 3,
    seed: int = 20260714,
    rule: str = "one_se",
) -> tuple[LockedCoxModel, pd.DataFrame, dict]:
    """
    Tune model complexity using development-only repeated CV.

    Feature ranking is recomputed inside each training fold. CORI and MMACE
    should receive the same candidate grid, number of folds, and repeats.
    """
    labels = _cv_strata_labels(train_df, folds=folds)
    splitter = RepeatedStratifiedKFold(
        n_splits=folds,
        n_repeats=repeats,
        random_state=seed,
    )

    rows: list[dict] = []
    dummy = np.zeros(len(train_df), dtype=int)

    for split, (train_index, valid_index) in enumerate(
        splitter.split(dummy, labels)
    ):
        development = train_df.iloc[train_index].copy()
        validation = train_df.iloc[valid_index].copy()
        ranking = rank_features(development, feature_columns, penalizer=0.01)

        for n_features in candidate_k:
            if n_features > len(ranking):
                continue
            for penalizer in candidate_penalizers:
                model = fit_model_from_ranking(
                    development,
                    ranking,
                    name=f"{name}_cv",
                    n_features=n_features,
                    penalizer=penalizer,
                )
                values = score_values(model, validation)
                c_value = concordance_index(
                    pd.to_numeric(validation["time_years"], errors="raise"),
                    -values,
                    pd.to_numeric(validation["Y_mace"], errors="raise"),
                )
                rows.append(
                    {
                        "split": split,
                        "n_features": int(n_features),
                        "penalizer": float(penalizer),
                        "C_index": float(c_value),
                    }
                )

    if not rows:
        raise RuntimeError("No valid hyperparameter configuration was evaluated.")

    summary = (
        pd.DataFrame(rows)
        .groupby(["n_features", "penalizer"], as_index=False)
        .agg(
            mean_C=("C_index", "mean"),
            sd_C=("C_index", "std"),
            n=("C_index", "size"),
        )
    )
    summary["sd_C"] = summary["sd_C"].fillna(0.0)
    summary["se_C"] = summary["sd_C"] / np.sqrt(summary["n"])

    best = summary.loc[summary["mean_C"].idxmax()].copy()

    if rule == "max":
        chosen = best
    elif rule == "one_se":
        eligible = summary.loc[
            summary["mean_C"] >= best["mean_C"] - best["se_C"]
        ].copy()
        chosen = eligible.sort_values(
            ["n_features", "penalizer"],
            ascending=[True, False],
        ).iloc[0]
    else:
        raise ValueError("rule must be either 'max' or 'one_se'.")

    final_model = train_model(
        train_df,
        feature_columns,
        name=name,
        n_features=int(chosen["n_features"]),
        penalizer=float(chosen["penalizer"]),
    )
    choice = {
        "rule": rule,
        "n_features": int(chosen["n_features"]),
        "penalizer": float(chosen["penalizer"]),
        "mean_internal_C": float(chosen["mean_C"]),
        "best_internal_C": float(best["mean_C"]),
        "cv_splits": int(folds * repeats),
    }
    return final_model, summary, choice


def out_of_fold_score(
    df: pd.DataFrame,
    feature_columns: list[str],
    name: str,
    n_features: int,
    penalizer: float,
    folds: int = 5,
    seed: int = 20260714,
) -> np.ndarray:
    """Generate participant-level scores from models that excluded that participant."""
    labels = _cv_strata_labels(df, folds=folds)
    splitter = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=seed,
    )
    output = np.full(len(df), np.nan, dtype=float)
    dummy = np.zeros(len(df), dtype=int)

    for fold, (train_index, valid_index) in enumerate(
        splitter.split(dummy, labels)
    ):
        model = train_model(
            df.iloc[train_index],
            feature_columns,
            name=f"{name}_fold_{fold}",
            n_features=n_features,
            penalizer=penalizer,
        )
        output[valid_index] = score_values(model, df.iloc[valid_index])

    if np.isnan(output).any():
        raise RuntimeError("Out-of-fold scoring left missing participant scores.")
    return output
