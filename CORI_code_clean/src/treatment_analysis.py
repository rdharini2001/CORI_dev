import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from cori_pipeline_utils_v13 import fmt_ci, pformat, safe_name


def zscore_from_train_apply(train_df, test_df, col, out_col):
    """
    Z-score a column using development-set mean/SD and apply to held-out set.
    Used only for continuous CORI score scaling.
    """
    mu = pd.to_numeric(train_df[col], errors="coerce").mean()
    sd = pd.to_numeric(train_df[col], errors="coerce").std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0

    train_df[out_col] = (pd.to_numeric(train_df[col], errors="coerce") - mu) / sd
    test_df[out_col] = (pd.to_numeric(test_df[col], errors="coerce") - mu) / sd
    return train_df, test_df, {"mean": mu, "sd": sd}


def fit_treatment_adjusted_cox(
    df,
    score_col,
    covars,
    label,
    tabledir,
    prefix,
    penalizer=0.05,
):
    """
    Fits Cox model in held-out cancer cohort:
        time, event ~ CORI + treatment covariates

    This does NOT retrain CORI.
    It only estimates whether the locked CORI score remains associated with MACE
    after accounting for treatment labels.
    """
    cols = ["time", "event", score_col] + [c for c in covars if c in df.columns]
    d = df[cols].copy()
    d = d.replace([np.inf, -np.inf], np.nan)

    # Coerce numeric columns
    d["time"] = pd.to_numeric(d["time"], errors="coerce")
    d["event"] = pd.to_numeric(d["event"], errors="coerce")
    d[score_col] = pd.to_numeric(d[score_col], errors="coerce")

    for c in covars:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["time", "event", score_col]).copy()
    d = d[d["time"] > 0].copy()
    d["event"] = (d["event"] > 0).astype(int)

    # Keep treatment covariates only if present and variable
    usable_covars = []
    for c in covars:
        if c not in d.columns:
            continue
        # Missing treatment labels are conservatively filled as 0 only after checking variability
        d[c] = d[c].fillna(0).astype(int)
        if d[c].nunique(dropna=True) > 1:
            usable_covars.append(c)

    model_cols = ["time", "event", score_col] + usable_covars
    d_model = d[model_cols].dropna().copy()

    out_base = {
        "analysis": label,
        "score_col": score_col,
        "N": len(d_model),
        "Events": int(d_model["event"].sum()) if len(d_model) else 0,
        "treatment_covariates_requested": ", ".join(covars),
        "treatment_covariates_used": ", ".join(usable_covars),
        "penalizer": penalizer,
    }

    if len(d_model) < 30 or d_model["event"].sum() < 5:
        out = pd.DataFrame([{**out_base, "status": "underpowered"}])
        out.to_csv(tabledir / f"{prefix}_{safe_name(label)}_treatment_adjusted_summary.csv", index=False)
        return out, None

    try:
        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(d_model, duration_col="time", event_col="event")

        summ = cph.summary.reset_index().rename(columns={"covariate": "variable"})
        summ["HR"] = np.exp(summ["coef"])
        summ["HR_low"] = np.exp(summ["coef lower 95%"])
        summ["HR_high"] = np.exp(summ["coef upper 95%"])
        summ["HR_95CI"] = summ.apply(
            lambda r: fmt_ci(r["HR"], r["HR_low"], r["HR_high"], 2),
            axis=1,
        )
        summ["p_fmt"] = summ["p"].map(pformat)
        summ["analysis"] = label
        summ["N"] = len(d_model)
        summ["Events"] = int(d_model["event"].sum())
        summ["status"] = "ok"

        # C-index of adjusted model
        risk = np.log(np.asarray(cph.predict_partial_hazard(d_model)).reshape(-1))
        cidx = concordance_index(d_model["time"], -risk, d_model["event"])
        summ["model_C_index"] = cidx

        summ.to_csv(
            tabledir / f"{prefix}_{safe_name(label)}_treatment_adjusted_coefficients.csv",
            index=False,
        )

        compact = summ[
            [
                "analysis",
                "variable",
                "N",
                "Events",
                "HR",
                "HR_low",
                "HR_high",
                "HR_95CI",
                "p",
                "p_fmt",
                "model_C_index",
                "status",
            ]
        ].copy()

        compact.to_csv(
            tabledir / f"{prefix}_{safe_name(label)}_treatment_adjusted_summary.csv",
            index=False,
        )

        return compact, cph

    except Exception as e:
        out = pd.DataFrame([{**out_base, "status": f"failed: {str(e)[:120]}"}])
        out.to_csv(tabledir / f"{prefix}_{safe_name(label)}_treatment_adjusted_summary.csv", index=False)
        return out, None
