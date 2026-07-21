from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import clean_id

def discover_feature_cols(df):
    # RetFound features: f0-f1023 exactly
    feats = []
    for c in df.columns:
        m = re.fullmatch(r"f(\d+)", str(c))
        if m:
            feats.append(c)
    feats = sorted(feats, key=lambda x: int(x[1:]))
    if len(feats) == 0:
        raise ValueError("No RetFound f0..f1023 feature columns found.")
    return feats

def find_center_col(df):
    candidates = [
        "UK Biobank assessment centre | Instance 0",
        "assessment_centre",
        "assessment_center",
        "center",
        "centre",
        "batch_label",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: any column with center/centre
    for c in df.columns:
        if "centre" in str(c).lower() or "center" in str(c).lower():
            return c
    raise ValueError("Could not identify assessment center column.")

def find_first_existing(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None

def cohort_audit_row(name, d):
    return {
        "step": name,
        "rows": len(d),
        "unique_eids": d["eid"].nunique() if "eid" in d.columns else np.nan,
        "events": int(pd.to_numeric(d["event"], errors="coerce").fillna(0).sum()) if "event" in d.columns else np.nan,
    }

def build_allcancer_cohort(master, feature_cols, horizon=10, logger=None):
    d = master.copy()
    if "eid" not in d.columns:
        raise ValueError("Master file must contain eid column.")
    d["eid"] = clean_id(d["eid"])
    audit = [cohort_audit_row("raw master", d)]

    if "image_visit" in d.columns:
        d = d[pd.to_numeric(d["image_visit"], errors="coerce").eq(0)].copy()
        audit.append(cohort_audit_row("baseline image_visit == 0", d))

    # all-cancer filter: prefer allcancer_event_status
    if "allcancer_event_status" in d.columns:
        d = d[pd.to_numeric(d["allcancer_event_status"], errors="coerce").eq(1)].copy()
        audit.append(cohort_audit_row("allcancer_event_status == 1", d))
    elif "AnyCancer_present" in d.columns:
        d = d[pd.to_numeric(d["AnyCancer_present"], errors="coerce").eq(1)].copy()
        audit.append(cohort_audit_row("AnyCancer_present == 1", d))
    else:
        # if no explicit cancer col, assume master is already cancer
        audit.append(cohort_audit_row("no explicit cancer column; retained all rows", d))

    time_col = f"MACE_in_allCancer_{horizon}yr_censored_time"
    event_col = f"MACE_in_allCancer_{horizon}yr_censored_status"
    if time_col not in d.columns or event_col not in d.columns:
        # fallback to generic MACE columns
        time_col = "scan_to_MACE_event_time" if "scan_to_MACE_event_time" in d.columns else None
        event_col = "MACE_event_status" if "MACE_event_status" in d.columns else None
    if time_col is None or event_col is None:
        raise ValueError("Could not identify cancer MACE endpoint columns.")
    d["time"] = pd.to_numeric(d[time_col], errors="coerce")
    d["event"] = pd.to_numeric(d[event_col], errors="coerce").fillna(0).astype(int)
    d = d.dropna(subset=["time", "event"]).copy()
    d = d[d["time"] > 0].copy()
    audit.append(cohort_audit_row(f"valid endpoint: {time_col} / {event_col}", d))

    present_features = [c for c in feature_cols if c in d.columns]
    d = d[d[present_features].notna().any(axis=1)].copy()
    audit.append(cohort_audit_row("has at least one RetFound feature", d))

    # one row per participant, preserve baseline row with longest available follow-up
    d = d.sort_values(["eid", "time"], ascending=[True, False]).drop_duplicates("eid", keep="first").copy()
    audit.append(cohort_audit_row("one row per eid", d))

    audit_df = pd.DataFrame(audit)
    if logger: logger.df("All-cancer cohort audit", audit_df)
    return d, audit_df

def build_noncancer_ready_cohort(df, feature_cols, logger=None):
    d = df.copy()
    if "eid" not in d.columns:
        raise ValueError("Non-cancer file must contain eid column.")
    d["eid"] = clean_id(d["eid"])
    audit = [cohort_audit_row("raw non-cancer file", d)]
    if "image_visit" in d.columns:
        d = d[pd.to_numeric(d["image_visit"], errors="coerce").eq(0)].copy()
        audit.append(cohort_audit_row("baseline image_visit == 0", d))

    if {"mace_10y_time_days", "mace_10y_event"}.issubset(d.columns):
        d["time"] = pd.to_numeric(d["mace_10y_time_days"], errors="coerce")
        d["event"] = pd.to_numeric(d["mace_10y_event"], errors="coerce").fillna(0).astype(int)
        endpoint = "mace_10y_time_days / mace_10y_event"
    elif {"mace_time_days_final", "mace_event_final"}.issubset(d.columns):
        d["time"] = pd.to_numeric(d["mace_time_days_final"], errors="coerce")
        d["event"] = pd.to_numeric(d["mace_event_final"], errors="coerce").fillna(0).astype(int)
        endpoint = "mace_time_days_final / mace_event_final"
    else:
        raise ValueError("Could not identify non-cancer MACE endpoint columns.")
    d = d.dropna(subset=["time", "event"]).copy()
    d = d[d["time"] > 0].copy()
    audit.append(cohort_audit_row(f"valid endpoint: {endpoint}", d))

    present_features = [c for c in feature_cols if c in d.columns]
    if not present_features:
        raise ValueError("No RetFound features present in non-cancer file.")
    d = d[d[present_features].notna().any(axis=1)].copy()
    audit.append(cohort_audit_row("has at least one RetFound feature", d))
    d = d.sort_values(["eid", "time"], ascending=[True, False]).drop_duplicates("eid", keep="first").copy()
    audit.append(cohort_audit_row("one row per eid", d))
    audit_df = pd.DataFrame(audit)
    if logger: logger.df("Non-cancer cohort audit", audit_df)
    return d, audit_df

def make_center_split(df, train_patterns, center_col=None, logger=None, min_train=20, min_test=20):
    d = df.copy()
    center_col = center_col or find_center_col(d)
    d[center_col] = d[center_col].astype(str).str.strip()
    pattern = "|".join([re.escape(str(x).strip()) for x in train_patterns if str(x).strip()])
    if not pattern:
        raise ValueError("No train center patterns supplied.")
    m = d[center_col].str.contains(pattern, case=False, regex=True, na=False)
    train = d[m].copy()
    test = d[~m].copy()
    tab = d.groupby(center_col).agg(N=("eid","count"), Events=("event","sum"), Event_rate=("event","mean")).reset_index()
    tab["split"] = np.where(tab[center_col].str.contains(pattern, case=False, regex=True, na=False), "development", "held_out")
    if logger:
        logger.section("Center split diagnostics")
        logger.log(f"Center column: {center_col}")
        logger.log(f"Requested development center patterns: {train_patterns}")
        logger.df("Available centers and split", tab.sort_values(["split","N"], ascending=[True,False]))
        logger.log(f"Split result: development N={len(train)}, events={int(train.event.sum())}; held-out N={len(test)}, events={int(test.event.sum())}")
    if len(train) < min_train or len(test) < min_test:
        raise ValueError(f"Too-small train/test after center split: train={len(train)}, test={len(test)}. Check center labels and requested TRAIN_CENTER_PATTERNS.")
    return train, test, tab, center_col

def cancer_subtype_columns(df):
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if c.endswith("_present") and ("cancer" in cl or "neoplasm" in cl or any(x in cl for x in ["repo","urinarytract","skin_","eyecns","digestive_","respi_"])):
            if c not in ["AnyCancer_present"]:
                vals = pd.to_numeric(df[c], errors="coerce")
                if vals.fillna(0).sum() > 0:
                    cols.append(c)
    return cols

def merge_treatment_labels(df, treatment_csv=None, chemo_csv=None, logger=None):
    d = df.copy()
    d["eid"] = clean_id(d["eid"])
    for path in [treatment_csv, chemo_csv]:
        if path is None:
            continue
        path = Path(path)
        if not path.exists():
            if logger: logger.log(f"Treatment file not found, skipping: {path}")
            continue
        t = pd.read_csv(path, low_memory=False)
        if "eid" not in t.columns:
            if logger: logger.log(f"No eid column in treatment file, skipping: {path}")
            continue
        t["eid"] = clean_id(t["eid"])
        t = t.drop_duplicates("eid", keep="first")
        add_cols = [c for c in t.columns if c != "eid" and c not in d.columns]
        d = d.merge(t[["eid"]+add_cols], on="eid", how="left")
        if logger: logger.log(f"Merged treatment file: {path}; added columns={len(add_cols)}")
    def find_first(cands):
        return find_first_existing(d, cands)
    any_col = find_first(["has_target_drug","has_treatment","treated","any_treatment","treatment_any"])
    chemo_col = find_first(["has_chemo","chemo","chemotherapy","has_chemotherapy","chemo_any"])
    io_col = find_first(["has_io","io","immunotherapy","has_immunotherapy","io_any"])
    def to01(s):
        if s is None:
            return pd.Series(0, index=d.index)
        x = d[s]
        if x.dtype == "object" or str(x.dtype).startswith("string"):
            xl = x.astype(str).str.lower()
            return xl.isin(["1","true","yes","y","treated","present"]).astype(int)
        return (pd.to_numeric(x, errors="coerce").fillna(0) > 0).astype(int)
    d["chemo_any"] = to01(chemo_col)
    d["io_any"] = to01(io_col)
    d["treatment_any"] = to01(any_col) if any_col else ((d["chemo_any"].eq(1)) | (d["io_any"].eq(1))).astype(int)
    d["treatment_naive"] = (d["treatment_any"].eq(0)).astype(int)
    if logger:
        logger.log(f"Treatment columns used: any={any_col}, chemo={chemo_col}, io={io_col}")
        logger.df("Treatment counts", pd.DataFrame([
            {"label":"Any systemic treatment","N":int(d.treatment_any.sum())},
            {"label":"Treatment naive","N":int(d.treatment_naive.sum())},
            {"label":"Chemotherapy","N":int(d.chemo_any.sum())},
            {"label":"Immunotherapy","N":int(d.io_any.sum())},
        ]))
    return d

def reshape_cmr_long(cmr_df, instances=(2,3)):
    rows = []
    for inst in instances:
        pat = re.compile(rf"\|\s*Instance\s*{inst}\b", flags=re.I)
        inst_cols = [c for c in cmr_df.columns if pat.search(str(c))]
        if not inst_cols:
            continue
        rename = {c: re.sub(r"\s*\|\s*Instance\s*\d+\s*", "", str(c), flags=re.I).strip() for c in inst_cols}
        tmp = cmr_df[["eid"]+inst_cols].rename(columns=rename).copy()
        tmp["cmr_instance"] = inst
        rows.append(tmp)
    if not rows:
        # Already long or no instance suffix; use all columns except eid
        tmp = cmr_df.copy()
        if "cmr_instance" not in tmp.columns:
            tmp["cmr_instance"] = np.nan
        rows.append(tmp)
    long = pd.concat(rows, ignore_index=True)
    long = long.loc[:, ~long.columns.duplicated()].copy()
    return long

def cmr_feature_cols(df, min_nonmissing=30):
    exclude_words = ["eid","time","event","score","risk","tertile","visit","date","status","label","split","image","path","filename"]
    cols = []
    for c in df.columns:
        cl = str(c).lower()
        if any(w in cl for w in exclude_words):
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        if x.notna().sum() >= min_nonmissing and x.dropna().nunique() > 5:
            cols.append(c)
    return cols
