from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


CORE_COLUMNS = [
    "eid",
    "A_cancer",
    "Y_mace",
    "time_years",
    "age",
    "female",
    "height",
    "center",
]

# These names appeared in older train-ready or score-export files. They are
# explicitly removed so the new pipeline cannot accidentally reuse old scores.
OLD_SCORE_COLUMNS = [
    "M_CORI_z",
    "M_MMACE_z",
    "CORI_score",
    "CORI_score_z",
    "CORI_raw_score",
    "CORI_risk_score",
    "MMACE_score",
    "MMACE_score_z",
    "HCORI_score",
    "HMMACE_score",
    "score_z",
]

TREATMENT_COLUMNS = ["has_target_drug", "has_chemo", "has_io"]


def _compression(path: Path) -> str | None:
    """Read gzip files correctly even when Windows hides the .gz suffix."""
    with path.open("rb") as handle:
        return "gzip" if handle.read(2) == b"\x1f\x8b" else None


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    return pd.read_csv(path, compression=_compression(path), low_memory=False, **kwargs)


def clean_eid(values: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="raise")
    if not np.allclose(values, np.round(values)):
        raise ValueError("eid contains non-integer values")
    return values.round().astype("int64")


def remove_old_scores(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    dropped = [column for column in OLD_SCORE_COLUMNS if column in df.columns]
    return df.drop(columns=dropped), dropped


def load_cohort(path: str | Path, feature_columns: list[str]) -> pd.DataFrame:
    """Load one locked cohort and enforce the common analysis schema."""
    df = read_csv(path)
    df, dropped = remove_old_scores(df)

    missing = [column for column in CORE_COLUMNS + feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{Path(path).name} is missing columns: {missing[:20]}")

    df["eid"] = clean_eid(df["eid"])
    if df["eid"].duplicated().any():
        raise ValueError(f"{Path(path).name} contains duplicate eid values")

    df["Y_mace"] = pd.to_numeric(df["Y_mace"], errors="raise").astype(int)
    df["A_cancer"] = pd.to_numeric(df["A_cancer"], errors="raise").astype(int)
    df["time_years"] = pd.to_numeric(df["time_years"], errors="raise")

    if (df["time_years"] <= 0).any():
        raise ValueError(f"{Path(path).name} contains non-positive follow-up")

    df.attrs["old_score_columns_removed"] = dropped
    return df


def load_clinical(path: str | Path) -> pd.DataFrame:
    clinical = read_csv(path, usecols=["eid", "HTN", "Diabetes", "Sex"])
    clinical["eid"] = clean_eid(clinical["eid"])
    clinical = clinical.drop_duplicates("eid", keep="first")
    clinical["HTN"] = pd.to_numeric(clinical["HTN"], errors="coerce")
    clinical["Diabetes"] = pd.to_numeric(clinical["Diabetes"], errors="coerce")
    return clinical


def load_treatment(path: str | Path) -> pd.DataFrame:
    """Read only treatment labels. Old risk-score columns never enter memory."""
    treatment = read_csv(path, usecols=["eid", *TREATMENT_COLUMNS])
    treatment["eid"] = clean_eid(treatment["eid"])
    treatment = treatment.drop_duplicates("eid", keep="first")
    for column in TREATMENT_COLUMNS:
        treatment[column] = pd.to_numeric(treatment[column], errors="coerce").fillna(0).astype(int)
    treatment["any_treatment"] = treatment[TREATMENT_COLUMNS].max(axis=1)
    return treatment


def load_sites(path: str | Path, site_columns: list[str]) -> pd.DataFrame:
    sites = read_csv(path, usecols=["eid", *site_columns])
    sites["eid"] = clean_eid(sites["eid"])
    sites = sites.drop_duplicates("eid", keep="first")
    for column in site_columns:
        values = sites[column]
        if values.dtype == object:
            values = values.astype(str).str.strip().str.lower().map({
                "true": 1, "false": 0, "1": 1, "0": 0,
                "yes": 1, "no": 0, "present": 1, "absent": 0,
            })
        sites[column] = pd.to_numeric(values, errors="coerce").fillna(0).astype(int)
    return sites


def merge_columns(df: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """One participant in, one participant out. Never change cohort membership."""
    before = df[["eid"]].copy()
    out = df.merge(other, on="eid", how="left", validate="one_to_one", sort=False)
    if not before["eid"].equals(out["eid"]):
        raise AssertionError("Merge changed participant order or membership")
    return out


def cohort_audit(cohorts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in cohorts.items():
        rows.append(
            {
                "cohort": name,
                "N": len(df),
                "events": int(df["Y_mace"].sum()),
                "event_rate": float(df["Y_mace"].mean()),
                "centers": int(df["center"].nunique()),
                "old_score_columns_removed": ", ".join(df.attrs.get("old_score_columns_removed", [])),
            }
        )
    return pd.DataFrame(rows)


def truncate_followup(df: pd.DataFrame, horizon_years: float) -> pd.DataFrame:
    out = df.copy()
    original_time = pd.to_numeric(out["time_years"], errors="coerce")
    original_event = pd.to_numeric(out["Y_mace"], errors="coerce").astype(int)
    out["time_horizon"] = np.minimum(original_time, horizon_years)
    out["event_horizon"] = ((original_event == 1) & (original_time <= horizon_years)).astype(int)
    return out


def read_cmr_zip(path: str | Path, member: str = "cardiac_mri.csv") -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        with archive.open(member) as handle:
            cmr = pd.read_csv(handle, low_memory=False)
    cmr["eid"] = clean_eid(cmr["eid"])
    return cmr


def collapse_cmr_columns(cmr: pd.DataFrame, column_map: dict[str, tuple[str, str]]) -> pd.DataFrame:
    """Collapse explicit Instance 2/3 CMR columns to one value per participant."""
    out = pd.DataFrame({"eid": cmr["eid"]})
    for output_name, (instance_2, instance_3) in column_map.items():
        values_2 = (
            pd.to_numeric(cmr[instance_2], errors="coerce")
            if instance_2 in cmr.columns
            else pd.Series(np.nan, index=cmr.index)
        )
        values_3 = (
            pd.to_numeric(cmr[instance_3], errors="coerce")
            if instance_3 in cmr.columns
            else pd.Series(np.nan, index=cmr.index)
        )
        out[output_name] = values_2.combine_first(values_3)
    return out.drop_duplicates("eid", keep="first")
