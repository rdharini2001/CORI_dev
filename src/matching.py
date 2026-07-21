from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .models import train_model, score_values
from .metrics import c_index


def propensity_odds(source: pd.DataFrame, target: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Odds that a source participant resembles the target development cohort."""
    if len(columns) == 0:
        return pd.Series(1.0, index=source.index)

    source_covariates = source[columns].copy()
    target_covariates = target[columns].copy()
    pooled = pd.concat([source_covariates, target_covariates], ignore_index=True)
    label = np.r_[np.zeros(len(source)), np.ones(len(target))]

    categorical = [column for column in columns if pooled[column].dtype == object]
    continuous = [column for column in columns if column not in categorical]

    transformers = []
    if continuous:
        transformers.append(("continuous", StandardScaler(), continuous))
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))

    preprocessing = ColumnTransformer(transformers)
    model = make_pipeline(preprocessing, LogisticRegression(max_iter=2000, C=1.0))
    model.fit(pooled, label)

    probability = model.predict_proba(source_covariates)[:, 1]
    probability = np.clip(probability, 0.01, 0.99)
    return pd.Series(probability / (1 - probability), index=source.index)


def draw_event_matched_sample(
    source: pd.DataFrame,
    target: pd.DataFrame,
    odds: pd.Series,
    rng: np.random.Generator,
) -> pd.DataFrame:
    target_events = int(target["Y_mace"].sum())
    target_nonevents = len(target) - target_events

    event_index = source.index[source["Y_mace"] == 1].to_numpy()
    nonevent_index = source.index[source["Y_mace"] == 0].to_numpy()

    # Covariate matching enters HERE: draw never-cancer donors with probability
    # proportional to their odds of resembling the cancer development cohort, so the
    # training sample matches the cancer cohort on age/sex/height, not just on event
    # count.  match_columns=[] gives odds==1 everywhere -> crude uniform sampling.
    p_event = odds.loc[event_index].to_numpy()
    p_nonevent = odds.loc[nonevent_index].to_numpy()
    p_event = p_event / p_event.sum()
    p_nonevent = p_nonevent / p_nonevent.sum()

    selected_events = rng.choice(event_index, target_events, replace=False, p=p_event)
    selected_nonevents = rng.choice(nonevent_index, target_nonevents, replace=False, p=p_nonevent)
    selected = np.r_[selected_events, selected_nonevents]

    sample = source.loc[selected].copy()
    # The fit weight is now ONLY the case-cohort inverse-probability correction for the
    # outcome-dependent (event vs non-event) sampling.  The covariate matching already
    # lives in the sample composition above, so we do not multiply by odds again.
    sample["sampling_weight"] = np.where(
        sample["Y_mace"] == 1,
        len(event_index) / target_events,
        len(nonevent_index) / target_nonevents,
    )
    sample["sampling_weight"] /= sample["sampling_weight"].mean()
    return sample


def matched_mmace_ensemble(
    source_development: pd.DataFrame,
    target_development: pd.DataFrame,
    evaluation_cohorts: dict[str, pd.DataFrame],
    feature_columns: list[str],
    n_features: int,
    penalizer: float,
    match_columns: list[str],
    repetitions: int = 60,
    seed: int = 20260714,
):
    """Train event- and covariate-matched models and average locked z-scores."""
    odds = propensity_odds(source_development, target_development, match_columns)
    rng = np.random.default_rng(seed)
    predictions = {name: [] for name in evaluation_cohorts}
    learning_rows = []
    models = []

    for repetition in range(repetitions):
        sample = draw_event_matched_sample(source_development, target_development, odds, rng)
        model = train_model(
            sample,
            feature_columns,
            name=f"MMACE_matched_{repetition:03d}",
            n_features=n_features,
            penalizer=penalizer,
            weight_column="sampling_weight",
        )
        models.append(model)

        row = {
            "repetition": repetition,
            "train_N": len(sample),
            "train_events": int(sample["Y_mace"].sum()),
        }
        for name, cohort in evaluation_cohorts.items():
            values = score_values(model, cohort)
            predictions[name].append(values)
            temp = cohort[["time_years", "Y_mace"]].copy()
            temp["score"] = values
            row[f"C_{name}"] = c_index(temp, "score")
        learning_rows.append(row)

    ensemble = {
        name: np.mean(np.column_stack(values), axis=1)
        for name, values in predictions.items()
    }
    return models, ensemble, pd.DataFrame(learning_rows)


def information_matched_learning_curve(
    source_development: pd.DataFrame,
    target_development: pd.DataFrame,
    evaluation: pd.DataFrame,
    feature_columns: list[str],
    n_features: int,
    penalizer: float,
    match_columns: list[str],
    event_repetitions: dict[int, int],
    seed: int = 20260714,
) -> pd.DataFrame:
    """Learning curve indexed by training events using target event fraction."""
    odds = propensity_odds(source_development, target_development, match_columns)
    rng = np.random.default_rng(seed)
    target_event_fraction = target_development["Y_mace"].mean()
    source_events = int(source_development["Y_mace"].sum())
    rows = []

    for event_count, repetitions in event_repetitions.items():
        for repetition in range(repetitions):
            if event_count >= source_events:
                sample = source_development.copy()
                sample["sampling_weight"] = odds.loc[sample.index].to_numpy()
            else:
                total_n = int(round(event_count / target_event_fraction))
                nonevent_count = total_n - event_count
                event_index = source_development.index[source_development["Y_mace"] == 1].to_numpy()
                nonevent_index = source_development.index[source_development["Y_mace"] == 0].to_numpy()
                selected = np.r_[
                    rng.choice(event_index, event_count, replace=False),
                    rng.choice(nonevent_index, nonevent_count, replace=False),
                ]
                sample = source_development.loc[selected].copy()
                sample["sampling_weight"] = np.where(
                    sample["Y_mace"] == 1,
                    len(event_index) / event_count,
                    len(nonevent_index) / nonevent_count,
                )
                sample["sampling_weight"] *= odds.loc[sample.index].to_numpy()

            lower, upper = sample["sampling_weight"].quantile([0.01, 0.99])
            sample["sampling_weight"] = sample["sampling_weight"].clip(lower, upper)
            sample["sampling_weight"] /= sample["sampling_weight"].mean()

            model = train_model(
                sample,
                feature_columns,
                name=f"MMACE_curve_{event_count}_{repetition}",
                n_features=n_features,
                penalizer=penalizer,
                weight_column="sampling_weight",
            )
            values = score_values(model, evaluation)
            scored = evaluation[["time_years", "Y_mace"]].copy()
            scored["score"] = values
            rows.append(
                {
                    "train_events": event_count,
                    "train_N": len(sample),
                    "repetition": repetition,
                    "C_D2": c_index(scored, "score"),
                }
            )
    return pd.DataFrame(rows)


def matched_curve_summary(draws: pd.DataFrame, c_column: str, cori_reference_c: float) -> dict:
    """Honest one-row summary of a matched-MMACE draw table: the median transported
    C, its 95% interval across draws, and the fraction of draws that beat CORI.
    Report this (median + CI), never a single point estimate."""
    c = draws[c_column].to_numpy()
    return {
        "reps": int(len(c)),
        "median_C": round(float(np.median(c)), 4),
        "C_low": round(float(np.percentile(c, 2.5)), 4),
        "C_high": round(float(np.percentile(c, 97.5)), 4),
        "P_greater_than_CORI": round(float((c > cori_reference_c).mean()), 3),
    }