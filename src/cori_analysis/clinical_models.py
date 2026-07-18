from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from .cohorts import find_first_existing

def one_age_covars_htn_diabetes(df, logger=None):
    age_col = find_first_existing(
        df,
        [
            "age_at_image_visit",
            "Age at image visit",
            "age_at_retinal_imaging",
            "Age at recruitment",
            "age",
            "Age",
        ],
    )

    preferred = [
        age_col,
        "sex_clinical",
        "height_clinical",
        "Diabetes_clinical",
        "HTN_clinical",
    ]

    fallback = [
        "sex",
        "Sex",
        "height",
        "Height",
        "Standing height",
        "Diabetes_status",
        "Diabetes_present",
        "diabetes",
        "HTN_status",
        "Hypertension_present",
        "hypertension",
    ]

    age_like = {
        "Age at recruitment",
        "age_at_image_visit",
        "Age at image visit",
        "age_at_retinal_imaging",
        "age",
        "Age",
    }

    covars = []
    for c in preferred + fallback:
        if c is None or c not in df.columns:
            continue
        if c != age_col and c in age_like:
            continue
        if c in covars:
            continue

        s = df[c]
        if s.notna().sum() >= 30 and s.nunique(dropna=True) > 1:
            covars.append(c)

    concept_priority = {
        "sex": ["sex_clinical", "sex", "Sex"],
        "height": ["height_clinical", "height", "Height", "Standing height"],
        "diabetes": ["Diabetes_clinical", "Diabetes_status", "Diabetes_present", "diabetes"],
        "htn": ["HTN_clinical", "HTN_status", "Hypertension_present", "hypertension"],
    }

    final = []
    for c in covars:
        duplicate = False
        for cols in concept_priority.values():
            if c in cols and any(x in final for x in cols):
                duplicate = True
                break
        if not duplicate:
            final.append(c)

    if logger:
        logger.log(f"One-age clinical policy: age variable={age_col}; covariates={final}")

    return final, age_col

def encode_train_test(train_df, test_df, covars):
    cols = ["time","event"] + [c for c in covars if c in train_df.columns and c in test_df.columns]
    tr = train_df[cols].copy()
    te = test_df[cols].copy()
    tr = tr.replace([np.inf,-np.inf], np.nan).dropna(subset=["time","event"])
    te = te.replace([np.inf,-np.inf], np.nan).dropna(subset=["time","event"])
    tr["time"] = pd.to_numeric(tr["time"], errors="coerce"); tr["event"] = pd.to_numeric(tr["event"], errors="coerce").astype(int)
    te["time"] = pd.to_numeric(te["time"], errors="coerce"); te["event"] = pd.to_numeric(te["event"], errors="coerce").astype(int)
    cat = []
    for c in cols[2:]:
        if tr[c].dtype == "object" or str(tr[c].dtype).startswith("string") or str(tr[c].dtype) == "category":
            cat.append(c)
        else:
            tr[c] = pd.to_numeric(tr[c], errors="coerce")
            te[c] = pd.to_numeric(te[c], errors="coerce")
    full = pd.concat([tr.assign(_set="train"), te.assign(_set="test")], axis=0)
    if cat:
        full = pd.get_dummies(full, columns=cat, drop_first=True, dummy_na=False, dtype=float)
    for c in full.columns:
        if c not in ["time","event","_set"]:
            full[c] = pd.to_numeric(full[c], errors="coerce")
            med = full.loc[full._set.eq("train"), c].median()
            full[c] = full[c].fillna(0 if pd.isna(med) else med)
    tr2 = full[full._set.eq("train")].drop(columns="_set")
    te2 = full[full._set.eq("test")].drop(columns="_set")
    # drop constant cols based on training
    keep = ["time","event"] + [c for c in tr2.columns if c not in ["time","event"] and tr2[c].nunique(dropna=True) > 1]
    return tr2[keep], te2.reindex(columns=keep, fill_value=0), keep[2:]

def train_cox_matrix(train_df, test_df, covars, penalizer=0.05):
    tr, te, used = encode_train_test(train_df, test_df, covars)
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(tr, "time", "event")
    rtr = np.log(np.asarray(cph.predict_partial_hazard(tr)).reshape(-1))
    rte = np.log(np.asarray(cph.predict_partial_hazard(te)).reshape(-1))
    c_train = concordance_index(tr.time, -rtr, tr.event) if tr.event.nunique() > 1 else np.nan
    c_test = concordance_index(te.time, -rte, te.event) if te.event.nunique() > 1 else np.nan
    return {"model": cph, "train_matrix": tr, "test_matrix": te, "train_risk": rtr, "test_risk": rte, "c_train": c_train, "c_test": c_test, "used_covars": used, "penalizer": penalizer}

def train_clinical_and_stacked(train_df, test_df, cori_score_col, clinical_covars, logger, tabledir):
    # z-score CORI using development only
    tr = train_df.copy(); te = test_df.copy()
    mu = pd.to_numeric(tr[cori_score_col], errors="coerce").mean()
    sd = pd.to_numeric(tr[cori_score_col], errors="coerce").std(ddof=0)
    if not np.isfinite(sd) or sd == 0: sd = 1.0
    tr["CORI_z"] = (pd.to_numeric(tr[cori_score_col], errors="coerce") - mu) / sd
    te["CORI_z"] = (pd.to_numeric(te[cori_score_col], errors="coerce") - mu) / sd
    fits = {}
    fits["Clinical"] = train_cox_matrix(tr, te, clinical_covars, penalizer=0.05)
    fits["CORI"] = train_cox_matrix(tr, te, ["CORI_z"], penalizer=0.01)
    fits["Clinical + CORI"] = train_cox_matrix(tr, te, clinical_covars + ["CORI_z"], penalizer=0.05)
    # stacked model uses clinical risk and CORI risk only, then Cox
    tr_stack = pd.DataFrame({"time": tr["time"].values, "event": tr["event"].values, "clinical_risk": fits["Clinical"]["train_risk"], "CORI_risk": fits["CORI"]["train_risk"]})
    te_stack = pd.DataFrame({"time": te["time"].values, "event": te["event"].values, "clinical_risk": fits["Clinical"]["test_risk"], "CORI_risk": fits["CORI"]["test_risk"]})
    fits["Stacked clinical-risk + CORI-risk"] = train_cox_matrix(tr_stack, te_stack, ["clinical_risk","CORI_risk"], penalizer=0.05)
    rows = []
    for name, fit in fits.items():
        rows.append({"model": name, "train_N": len(fit["train_matrix"]), "train_events": int(fit["train_matrix"].event.sum()), "test_N": len(fit["test_matrix"]), "test_events": int(fit["test_matrix"].event.sum()), "train_C_index": fit["c_train"], "test_C_index": fit["c_test"], "predictors": ", ".join(fit["used_covars"]), "penalizer": fit["penalizer"]})
    comp = pd.DataFrame(rows)
    logger.df("Clinical/CORI/stacked model comparison", comp)
    return fits, comp

one_age_covars = one_age_covars_htn_diabetes
