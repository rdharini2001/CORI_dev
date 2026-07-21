from __future__ import annotations

import numpy as np
import pandas as pd

from .common import RANDOM_SEED, fmt_ci, pformat
from .evaluation import bootstrap_cindex, km_hr

def safe_horizon_performance(
    df,
    score_col,
    group_col,
    horizon_cols,
    label,
    tabledir,
    prefix,
    n_boot=300,
):
    """
    Compute 3/5/10-year performance safely for a cohort or a treatment stratum.

    Important:
    This function intentionally creates a new dataframe with only:
        time, event, score_col, group_col
    for each horizon, so duplicate time/event columns cannot occur.
    """

    rows = []

    base = df.copy()
    base = base.loc[:, ~base.columns.duplicated()].copy()

    for horizon_years, cols in horizon_cols.items():
        time_col, event_col = cols

        if time_col not in base.columns or event_col not in base.columns:
            rows.append({
                "cohort": label,
                "horizon_years": horizon_years,
                "N": 0,
                "Events": 0,
                "C_index": np.nan,
                "C_index_low": np.nan,
                "C_index_high": np.nan,
                "C_index_95CI": "NA",
                "HR_High_vs_Low": np.nan,
                "HR_CI_Low": np.nan,
                "HR_CI_High": np.nan,
                "HR_95CI": "NA",
                "LogRank_p": np.nan,
                "LogRank_p_fmt": "NA",
                "status": f"missing horizon columns: {time_col}, {event_col}",
            })
            continue

        # Build clean horizon-specific dataframe
        tmp = pd.DataFrame({
            "time": pd.to_numeric(base[time_col], errors="coerce"),
            "event": pd.to_numeric(base[event_col], errors="coerce"),
            score_col: pd.to_numeric(base[score_col], errors="coerce"),
            group_col: pd.to_numeric(base[group_col], errors="coerce"),
        })

        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna().copy()
        tmp = tmp[tmp["time"] > 0].copy()

        if len(tmp) == 0:
            rows.append({
                "cohort": label,
                "horizon_years": horizon_years,
                "N": 0,
                "Events": 0,
                "C_index": np.nan,
                "C_index_low": np.nan,
                "C_index_high": np.nan,
                "C_index_95CI": "NA",
                "HR_High_vs_Low": np.nan,
                "HR_CI_Low": np.nan,
                "HR_CI_High": np.nan,
                "HR_95CI": "NA",
                "LogRank_p": np.nan,
                "LogRank_p_fmt": "NA",
                "status": "empty after cleaning",
            })
            continue

        tmp["event"] = (tmp["event"] > 0).astype(int)
        tmp[group_col] = tmp[group_col].astype(int)

        n = len(tmp)
        events = int(tmp["event"].sum())
        n_groups = int(tmp[group_col].nunique())

        if n < 10 or tmp["event"].nunique() < 2 or n_groups < 2:
            rows.append({
                "cohort": label,
                "horizon_years": horizon_years,
                "N": n,
                "Events": events,
                "C_index": np.nan,
                "C_index_low": np.nan,
                "C_index_high": np.nan,
                "C_index_95CI": "NA",
                "HR_High_vs_Low": np.nan,
                "HR_CI_Low": np.nan,
                "HR_CI_High": np.nan,
                "HR_95CI": "NA",
                "LogRank_p": np.nan,
                "LogRank_p_fmt": "NA",
                "status": "insufficient N/events/groups",
            })
            continue

        point, lo, hi = bootstrap_cindex(
            tmp,
            score_col,
            n_boot=n_boot,
            seed=RANDOM_SEED,
        )

        met = km_hr(
            tmp,
            score_col,
            group_col,
            f"{label} {horizon_years}yr",
        )

        rows.append({
            "cohort": label,
            "horizon_years": horizon_years,
            "N": n,
            "Events": events,
            "C_index": point,
            "C_index_low": lo,
            "C_index_high": hi,
            "C_index_95CI": fmt_ci(point, lo, hi, 3),
            "HR_High_vs_Low": met.get("HR_High_vs_Low", np.nan),
            "HR_CI_Low": met.get("HR_CI_Low", np.nan),
            "HR_CI_High": met.get("HR_CI_High", np.nan),
            "HR_95CI": fmt_ci(
                met.get("HR_High_vs_Low", np.nan),
                met.get("HR_CI_Low", np.nan),
                met.get("HR_CI_High", np.nan),
                2,
            ),
            "LogRank_p": met.get("LogRank_p", np.nan),
            "LogRank_p_fmt": pformat(met.get("LogRank_p", np.nan)),
            "status": "ok",
        })

    out = pd.DataFrame(rows)
    out.to_csv(tabledir / f"{prefix}_horizon_performance.csv", index=False)
    return out