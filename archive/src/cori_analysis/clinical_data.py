from __future__ import annotations

import numpy as np
import pandas as pd

from .common import clean_id
from .cohorts import find_first_existing

def find_col(df, candidates):
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None

def clean_binary_col(s):
    if s.dtype == object or str(s.dtype).startswith("string"):
        x = s.astype(str).str.strip().str.lower()
        return x.isin(["1", "yes", "y", "true", "present", "positive", "diabetes", "hypertension", "htn"]).astype(int)
    return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)

def load_and_prepare_htn_db(htn_db_csv):
    """Load final_df_HTN_DB_Status.csv and build a clean eid/sex/diabetes/HTN table."""
    htn_db = pd.read_csv(htn_db_csv, low_memory=False)

    if "eid" not in htn_db.columns:
        raise ValueError("final_df_HTN_DB_Status.csv must contain an eid column.")

    htn_db["eid"] = clean_id(htn_db["eid"])

    print("HTN/DB file columns:")
    print(htn_db.columns.tolist())

    sex_col = find_col(htn_db, ["sex", "Sex", "gender", "Gender"])
    diab_col = find_col(htn_db, [
        "Diabetes_clinical",
        "Diabetes_status",
        "Diabetes_present",
        "diabetes",
        "DB_status",
        "DM_status",
    ])
    htn_col = find_col(htn_db, [
        "HTN_clinical",
        "HTN_status",
        "Hypertension_present",
        "hypertension",
        "HTN",
    ])

    keep = ["eid"]
    rename = {}

    if sex_col:
        keep.append(sex_col)
        rename[sex_col] = "sex_clinical"

    if diab_col:
        keep.append(diab_col)
        rename[diab_col] = "Diabetes_clinical"

    if htn_col:
        keep.append(htn_col)
        rename[htn_col] = "HTN_clinical"

    htn_db_small = (
        htn_db[keep]
        .rename(columns=rename)
        .drop_duplicates("eid", keep="first")
    )

    for c in ["Diabetes_clinical", "HTN_clinical"]:
        if c in htn_db_small.columns:
            htn_db_small[c] = clean_binary_col(htn_db_small[c])

    return htn_db_small

def merge_htn_diabetes_status(df, htn_db_small):
    """Merge cleaned sex/Diabetes_clinical/HTN_clinical columns into df by eid."""
    d = df.copy()
    d["eid"] = clean_id(d["eid"])

    # Drop stale clinical columns if already present, then remerge
    for c in ["sex_clinical", "Diabetes_clinical", "HTN_clinical"]:
        if c in d.columns:
            d = d.drop(columns=[c])

    d = d.merge(htn_db_small, on="eid", how="left")
    return d

def one_age_covars_htn_diabetes(df, logger=None):
    """One-age clinical covariate policy: prefer cleaned _clinical columns, fall back
    to raw master-file columns, and never include more than one variable per concept
    (age/sex/height/diabetes/HTN)."""
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

def choose_adjustment_covars_for_h4_hcori(df):
    """Covariate policy for H4 handcrafted-HCORI adjusted CMR regressions.

    Distinct from one_age_covars_htn_diabetes: includes BMI and dedups by
    substring-matched concept rather than an explicit priority table.
    """
    preferred = [
        "age_at_image_visit",
        "Age at image visit",
        "age_at_retinal_imaging",
        "Age at recruitment",
        "sex_clinical",
        "sex",
        "Sex",
        "height_clinical",
        "height",
        "Height",
        "BMI",
        "body_mass_index",
        "Diabetes_clinical",
        "HTN_clinical",
        "Diabetes_present",
        "Hypertension_present",
    ]

    selected = []
    used_concepts = set()

    for c in preferred:
        if c not in df.columns:
            continue

        cl = c.lower()
        if "age" in cl:
            concept = "age"
        elif "sex" in cl:
            concept = "sex"
        elif "height" in cl:
            concept = "height"
        elif "bmi" in cl or "body_mass" in cl:
            concept = "bmi"
        elif "diabetes" in cl:
            concept = "diabetes"
        elif "htn" in cl or "hypertension" in cl:
            concept = "htn"
        else:
            concept = c

        if concept in used_concepts:
            continue

        s = df[c]
        if s.notna().sum() >= 30 and s.nunique(dropna=True) > 1:
            selected.append(c)
            used_concepts.add(concept)

    return selected

one_age_covars = one_age_covars_htn_diabetes


def load_clinical_status_exact(
    path,
    *,
    id_col='eid',
    sex_col='sex',
    diabetes_col='Diabetes',
    htn_col='HTN',
):
    """Load the explicitly configured clinical columns without alias searching."""
    data = pd.read_csv(path, low_memory=False)
    required = [id_col, diabetes_col, htn_col]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f'Clinical status file is missing configured columns {missing}. '
            'Update ColumnSchema rather than adding filename/column search logic.'
        )
    keep = required + ([sex_col] if sex_col in data.columns else [])
    data = data[keep].copy()
    rename = {
        id_col: 'eid',
        diabetes_col: 'Diabetes_clinical',
        htn_col: 'HTN_clinical',
    }
    if sex_col in data.columns:
        rename[sex_col] = 'sex_clinical'
    data = data.rename(columns=rename)
    data['eid'] = clean_id(data['eid'])
    data['Diabetes_clinical'] = clean_binary_col(data['Diabetes_clinical'])
    data['HTN_clinical'] = clean_binary_col(data['HTN_clinical'])
    return data.drop_duplicates('eid', keep='first')
