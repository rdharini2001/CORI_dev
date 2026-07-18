from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .common import safe_name

_safe_name = safe_name

CMR_FEATURE_FAMILIES = {
    "LV_structure_function": {
        "include": [
            r"\bleft ventricular\b", r"\bLV\b",
            r"end[- ]diastolic volume", r"end[- ]systolic volume",
            r"stroke volume", r"ejection fraction", r"cardiac output",
            r"myocardial mass", r"left ventricle",
        ],
        "exclude": [r"image", r"date", r"quality", r"path"],
    },
    "RV_structure_function": {
        "include": [
            r"\bright ventricular\b", r"\bRV\b", r"right ventricle",
            r"end[- ]diastolic volume", r"end[- ]systolic volume",
            r"stroke volume", r"ejection fraction",
        ],
        "exclude": [r"image", r"date", r"quality", r"path"],
    },
    "Atrial_structure": {
        "include": [
            r"\bleft atri", r"\bright atri", r"\bLA\b", r"\bRA\b",
            r"atrial volume", r"atrial ejection", r"atrial emptying",
        ],
        "exclude": [r"image", r"date", r"quality", r"path"],
    },
    "Aortic_geometry": {
        "include": [
            r"aorta.*area", r"aortic.*area",
            r"ascending aorta", r"descending aorta",
            r"proximal descending", r"distal descending",
            r"maximum area", r"minimum area", r"mean area",
        ],
        "exclude": [r"distensibility", r"strain", r"flow", r"velocity", r"date"],
    },
    "Aortic_stiffness_function": {
        "include": [
            r"aortic.*distensibility", r"distensibility",
            r"aortic.*strain", r"strain.*aorta",
            r"pulse wave velocity", r"\bPWV\b", r"arterial stiffness",
        ],
        "exclude": [r"image", r"date", r"path"],
    },
    "Blood_pressure_PWA": {
        "include": [
            r"central systolic", r"central diastolic",
            r"brachial systolic", r"brachial diastolic",
            r"systolic blood pressure", r"diastolic blood pressure",
            r"pulse pressure", r"augmentation index", r"\bPWA\b", r"\bPVR\b",
            r"end systolic pressure",
        ],
        "exclude": [r"date", r"path"],
    },
    "Aortic_flow": {
        "include": [
            r"aortic.*flow", r"flow rate", r"forward flow", r"backward flow",
            r"regurgitant fraction", r"peak velocity", r"velocity",
        ],
        "exclude": [r"date", r"path"],
    },
    "Myocardial_tissue": {
        "include": [
            r"\bT1\b", r"native T1", r"shMOLLI", r"myocardial T1",
            r"extracellular volume", r"\bECV\b",
        ],
        "exclude": [r"date", r"path"],
    },
    "Myocardial_strain": {
        "include": [
            r"global longitudinal strain", r"\bGLS\b",
            r"circumferential strain", r"radial strain",
            r"longitudinal strain", r"strain rate",
        ],
        "exclude": [r"aorta", r"aortic", r"date", r"path"],
    },
}

def preprocess_cmr_features(
    df,
    cmr_cols,
    *,
    winsorize=True,
    lower_q=0.01,
    upper_q=0.99,
    z_prefix="z__",
    min_nonmissing=30,
    logger=None,
):
    """
    Creates standardized CMR analysis columns while preserving raw CMR columns.

    For each selected CMR feature:
      1. converts to numeric
      2. sets infinite values to NaN
      3. optionally winsorizes at lower_q / upper_q
      4. z-scores using the analytic H4 cohort distribution

    This is appropriate for association/cluster/regression analyses.
    Raw columns should still be used for clinically interpretable descriptive plots.
    """

    out = df.copy()
    qc_rows = []
    z_cols = []

    for col in cmr_cols:
        if col not in out.columns:
            continue

        x_raw = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        n_raw = int(x_raw.notna().sum())
        n_unique = int(x_raw.nunique(dropna=True))

        if n_raw < min_nonmissing or n_unique <= 2:
            qc_rows.append({
                "feature": col,
                "status": "excluded_low_nonmissing_or_low_variance",
                "n_nonmissing": n_raw,
                "n_unique": n_unique,
                "raw_mean": x_raw.mean(),
                "raw_sd": x_raw.std(ddof=0),
                "lower_clip": np.nan,
                "upper_clip": np.nan,
                "z_col": None,
            })
            continue

        x = x_raw.copy()

        lo = np.nan
        hi = np.nan

        if winsorize:
            lo = x.quantile(lower_q)
            hi = x.quantile(upper_q)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                x = x.clip(lower=lo, upper=hi)

        mu = x.mean()
        sd = x.std(ddof=0)

        if not np.isfinite(sd) or sd == 0:
            qc_rows.append({
                "feature": col,
                "status": "excluded_zero_sd_after_cleaning",
                "n_nonmissing": n_raw,
                "n_unique": n_unique,
                "raw_mean": x_raw.mean(),
                "raw_sd": x_raw.std(ddof=0),
                "lower_clip": lo,
                "upper_clip": hi,
                "z_col": None,
            })
            continue

        z_col = z_prefix + safe_name(col)
        out[z_col] = (x - mu) / sd
        z_cols.append(z_col)

        qc_rows.append({
            "feature": col,
            "status": "included",
            "n_nonmissing": n_raw,
            "n_unique": n_unique,
            "raw_mean": x_raw.mean(),
            "raw_sd": x_raw.std(ddof=0),
            "processed_mean": x.mean(),
            "processed_sd": sd,
            "lower_clip": lo,
            "upper_clip": hi,
            "z_col": z_col,
        })

    qc = pd.DataFrame(qc_rows)

    if logger:
        logger.df("CMR preprocessing QC", qc)

    return out, z_cols, qc

def preprocess_cmr_features_for_hcori(
    df,
    cmr_cols,
    *,
    winsorize=True,
    lower_q=0.01,
    upper_q=0.99,
    z_prefix="zHC__",
    min_nonmissing=30,
):
    """
    H4-handcrafted-HCORI variant of preprocess_cmr_features.

    Same winsorize/z-score logic, but also returns a z_col -> raw_col map
    (needed downstream to report associations in raw CMR units) and does
    not take a logger (the H4 HCORI cell logs the returned qc table itself).
    """
    out = df.copy()
    qc_rows = []
    z_cols = []
    z_to_raw = {}

    for col in cmr_cols:
        if col not in out.columns:
            continue

        x_raw = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        n_nonmissing = int(x_raw.notna().sum())
        n_unique = int(x_raw.nunique(dropna=True))

        if n_nonmissing < min_nonmissing or n_unique <= 2:
            qc_rows.append({
                "feature": col,
                "status": "excluded_low_nonmissing_or_low_variance",
                "n_nonmissing": n_nonmissing,
                "n_unique": n_unique,
                "raw_mean": x_raw.mean(),
                "raw_sd": x_raw.std(ddof=0),
                "lower_clip": np.nan,
                "upper_clip": np.nan,
                "z_col": None,
            })
            continue

        x = x_raw.copy()
        lo = np.nan
        hi = np.nan

        if winsorize:
            lo = x.quantile(lower_q)
            hi = x.quantile(upper_q)
            if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                x = x.clip(lower=lo, upper=hi)

        mu = x.mean()
        sd = x.std(ddof=0)

        if not np.isfinite(sd) or sd == 0:
            qc_rows.append({
                "feature": col,
                "status": "excluded_zero_sd_after_processing",
                "n_nonmissing": n_nonmissing,
                "n_unique": n_unique,
                "raw_mean": x_raw.mean(),
                "raw_sd": x_raw.std(ddof=0),
                "processed_mean": x.mean(),
                "processed_sd": sd,
                "lower_clip": lo,
                "upper_clip": hi,
                "z_col": None,
            })
            continue

        z_col = z_prefix + _safe_name(col)
        out[z_col] = (x - mu) / sd
        z_cols.append(z_col)
        z_to_raw[z_col] = col

        qc_rows.append({
            "feature": col,
            "status": "included",
            "n_nonmissing": n_nonmissing,
            "n_unique": n_unique,
            "raw_mean": x_raw.mean(),
            "raw_sd": x_raw.std(ddof=0),
            "processed_mean": x.mean(),
            "processed_sd": sd,
            "lower_clip": lo,
            "upper_clip": hi,
            "z_col": z_col,
        })

    qc = pd.DataFrame(qc_rows)
    return out, z_cols, z_to_raw, qc

def _clean_cmr_label_for_matching(x):
    """Normalize column label for regex matching."""
    s = str(x)
    s = re.sub(r"\s*\|\s*Instance\s*\d+\s*", "", s, flags=re.I)
    s = re.sub(r"\s*\|\s*Array\s*\d+\s*", "", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _is_numeric_usable(df, col, min_nonmissing=30, min_unique=5):
    x = pd.to_numeric(df[col], errors="coerce")
    return (x.notna().sum() >= min_nonmissing) and (x.dropna().nunique() >= min_unique)

def classify_cmr_feature(col, family_map):
    """
    Return the first matched CMR family for a phenotype column.
    If multiple families match, priority follows dictionary order.
    """
    label = _clean_cmr_label_for_matching(col)
    low = label.lower()

    for family, rules in family_map.items():
        include = rules.get("include", [])
        exclude = rules.get("exclude", [])

        inc = any(re.search(pat, low, flags=re.I) for pat in include)
        exc = any(re.search(pat, low, flags=re.I) for pat in exclude)

        if inc and not exc:
            return family

    return "Unclassified"

def build_cmr_variable_inventory(
    df,
    family_map,
    exact_variables=None,
    user_keywords=None,
    min_nonmissing=30,
    min_unique=5,
    primary_cmr_families="ALL_CURATED",
):
    """
    Build a transparent inventory of candidate CMR phenotypes.
    This table is the audit trail for why a variable was included/excluded.
    """
    exact_variables = set(exact_variables or [])
    user_keywords = user_keywords or []

    # Columns that should never be treated as CMR phenotypes
    hard_exclude_terms = [
        "eid", "time", "event", "score", "risk", "tertile", "split",
        "image", "path", "filename", "date", "status", "label",
        "cancer", "mace", "retfound", "f0", "f1"
    ]

    rows = []
    for col in df.columns:
        label = _clean_cmr_label_for_matching(col)
        low = label.lower()

        x = pd.to_numeric(df[col], errors="coerce")
        nonmissing = int(x.notna().sum())
        unique_vals = int(x.dropna().nunique()) if nonmissing else 0
        numeric_usable = _is_numeric_usable(df, col, min_nonmissing, min_unique)

        hard_excluded = any(term in low for term in hard_exclude_terms)
        family = classify_cmr_feature(col, family_map)
        exact_include = col in exact_variables or label in exact_variables
        keyword_include = any(re.search(k, low, flags=re.I) for k in user_keywords)

        include_by_family = family != "Unclassified"
        selected = numeric_usable and (not hard_excluded) and (include_by_family or exact_include or keyword_include)

        if exact_include or keyword_include:
            selected = numeric_usable and (not hard_excluded)

        if exact_include:
            selection_reason = "force_include_exact"
        elif keyword_include:
            selection_reason = "force_include_keyword"
        elif selected and include_by_family:
            selection_reason = f"family:{family}"
        elif hard_excluded:
            selection_reason = "hard_excluded_metadata_or_outcome"
        elif not numeric_usable:
            selection_reason = "not_numeric_or_insufficient_nonmissing_unique"
        else:
            selection_reason = "unclassified_not_selected"

        rows.append({
            "column": col,
            "clean_label": label,
            "family": family,
            "nonmissing": nonmissing,
            "unique_values": unique_vals,
            "numeric_usable": numeric_usable,
            "hard_excluded": hard_excluded,
            "exact_include": exact_include,
            "keyword_include": keyword_include,
            "selected": bool(selected),
            "selection_reason": selection_reason,
        })

    inv = pd.DataFrame(rows)

    # Restrict to requested primary families if needed
    if primary_cmr_families != "ALL_CURATED":
        allowed = set(primary_cmr_families)
        inv.loc[~inv["family"].isin(allowed) & ~inv["exact_include"] & ~inv["keyword_include"], "selected"] = False
        inv.loc[~inv["family"].isin(allowed) & ~inv["exact_include"] & ~inv["keyword_include"], "selection_reason"] = "not_in_PRIMARY_CMR_FAMILIES"

    return inv