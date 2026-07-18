from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy.stats import chi2, norm, pearsonr, spearmanr, ttest_ind
from statsmodels.stats.multitest import multipletests

from .common import RANDOM_SEED, fmt_ci
from .clinical_models import encode_train_test

def bootstrap_cindex(df, score_col, n_boot=300, seed=RANDOM_SEED):
    d = df[["time","event",score_col]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    if len(d) < 10 or d.event.nunique() < 2:
        return np.nan, np.nan, np.nan
    point = concordance_index(d["time"], -d[score_col], d["event"])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(len(d)), size=len(d), replace=True)
        b = d.iloc[idx]
        if b.event.nunique() < 2:
            continue
        try:
            vals.append(concordance_index(b["time"], -b[score_col], b["event"]))
        except Exception:
            pass
    if not vals:
        return point, np.nan, np.nan
    return float(point), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def km_hr(df, score_col, group_col, label=""):
    d = df[["time","event",score_col,group_col]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    out = {"label": label, "N": len(d), "Events": int(d.event.sum()) if len(d) else 0}
    if len(d) < 10 or d[group_col].nunique() < 2 or d.event.sum() < 2:
        out.update({"HR_High_vs_Low": np.nan, "HR_CI_Low": np.nan, "HR_CI_High": np.nan, "LogRank_p": np.nan})
        return out
    try:
        cph = CoxPHFitter(penalizer=0.01)
        dd = d[["time","event",group_col]].copy()
        cph.fit(dd, "time", "event")
        row = cph.summary.loc[group_col]
        out.update({"HR_High_vs_Low": float(np.exp(row["coef"])), "HR_CI_Low": float(np.exp(row["coef lower 95%"])), "HR_CI_High": float(np.exp(row["coef upper 95%"]))})
    except Exception:
        out.update({"HR_High_vs_Low": np.nan, "HR_CI_Low": np.nan, "HR_CI_High": np.nan})
    try:
        g0 = d[d[group_col].eq(0)]
        g1 = d[d[group_col].eq(1)]
        lr = logrank_test(g0.time, g1.time, event_observed_A=g0.event, event_observed_B=g1.event)
        out["LogRank_p"] = float(lr.p_value)
    except Exception:
        out["LogRank_p"] = np.nan
    return out

def performance_row(df, cohort, score_col, group_col):
    point, lo, hi = bootstrap_cindex(df, score_col)
    m = km_hr(df, score_col, group_col, cohort)
    return {"cohort": cohort, "N": len(df), "Events": int(df.event.sum()), "C_index": point, "C_index_low": lo, "C_index_high": hi, "C_index_95CI": fmt_ci(point,lo,hi,3), "HR_95CI": fmt_ci(m.get("HR_High_vs_Low",np.nan), m.get("HR_CI_Low",np.nan), m.get("HR_CI_High",np.nan),2), **m}

def horizon_performance(df, score_col, group_col, horizon_cols, label, tabledir, prefix):
    # horizon_cols: dict years -> (time_col,event_col)
    rows = []
    for yr, (tcol, ecol) in horizon_cols.items():
        if tcol not in df.columns or ecol not in df.columns:
            continue
        tmp = df.copy()
        tmp["time_h"] = pd.to_numeric(tmp[tcol], errors="coerce")
        tmp["event_h"] = pd.to_numeric(tmp[ecol], errors="coerce").fillna(0).astype(int)
        tmp = tmp.dropna(subset=["time_h","event_h",score_col,group_col]).copy()
        tmp2 = tmp.rename(columns={"time_h":"time", "event_h":"event"})
        rows.append({"cohort": label, "horizon_years": yr, **performance_row(tmp2, label, score_col, group_col)})
    out = pd.DataFrame(rows)
    if len(out):
        out.to_csv(Path(tabledir)/f"{prefix}_horizon_performance.csv", index=False)
    return out

def predict_10yr(cph, mat):
    # crude Cox absolute risk at 10 years based on baseline survival
    try:
        t = 365.25*10
        bs = cph.baseline_survival_
        idx = np.argmin(np.abs(bs.index.values - t))
        s0 = float(bs.iloc[idx,0])
        hr = np.asarray(cph.predict_partial_hazard(mat)).reshape(-1)
        risk = 1 - np.power(s0, hr)
        return np.clip(risk, 0, 1)
    except Exception:
        # fallback to normalized partial hazard
        r = np.asarray(cph.predict_partial_hazard(mat)).reshape(-1)
        return (r - np.min(r)) / (np.max(r) - np.min(r) + 1e-9)

def lrt_interaction(df, score_col, status_col, covars=None, label="interaction"):
    covars = covars or []
    d = df[["time","event",score_col,status_col] + [c for c in covars if c in df.columns]].copy()
    d = d.replace([np.inf,-np.inf], np.nan).dropna(subset=["time","event",score_col,status_col])
    d["score_z"] = (pd.to_numeric(d[score_col], errors="coerce") - pd.to_numeric(d[score_col], errors="coerce").mean()) / (pd.to_numeric(d[score_col], errors="coerce").std(ddof=0) or 1)
    d[status_col] = pd.to_numeric(d[status_col], errors="coerce").fillna(0).astype(int)
    d["score_x_status"] = d["score_z"] * d[status_col]
    base_covars = ["score_z", status_col] + [c for c in covars if c not in ["score_z", status_col]]
    inter_covars = base_covars + ["score_x_status"]
    base, _, _ = encode_train_test(d, d, base_covars)
    inter, _, _ = encode_train_test(d, d, inter_covars)
    common = base.index.intersection(inter.index)
    base = base.loc[common]; inter = inter.loc[common]
    c0 = CoxPHFitter(penalizer=0.05); c1 = CoxPHFitter(penalizer=0.05)
    c0.fit(base, "time", "event"); c1.fit(inter, "time", "event")
    lr = 2*(c1.log_likelihood_ - c0.log_likelihood_)
    df_diff = max(1, len(c1.params_) - len(c0.params_))
    p = chi2.sf(lr, df_diff) if chi2 is not None else np.nan
    r0 = np.log(np.asarray(c0.predict_partial_hazard(base)).reshape(-1))
    r1 = np.log(np.asarray(c1.predict_partial_hazard(inter)).reshape(-1))
    return {"analysis": label, "N": len(inter), "Events": int(inter.event.sum()), "LRT_chi_square": lr, "df": df_diff, "p": p, "base_c_index": concordance_index(base.time, -r0, base.event), "interaction_c_index": concordance_index(inter.time, -r1, inter.event), "covariates": ", ".join(covars)}

def risk_tertiles_from_pred(pred):
    return pd.qcut(pd.Series(pred).rank(method="first"), 3, labels=[1,2,3]).astype(int)

def transition_tables(df, left, right, event_col="event", value_col=None):
    d = df[[left,right,event_col] + ([value_col] if value_col and value_col in df.columns else [])].dropna().copy()
    counts = pd.crosstab(d[left], d[right])
    events = pd.crosstab(d[left], d[right], values=d[event_col], aggfunc="sum").fillna(0)
    event_rates = (events / counts.replace(0,np.nan)).fillna(0)
    if value_col and value_col in d.columns:
        values = pd.crosstab(d[left], d[right], values=d[value_col], aggfunc="mean")
    else:
        values = event_rates
    return counts, event_rates, values

def cmr_analysis_table(df, score_col, group_col, cmr_cols, tabledir, prefix="H4"):
    rows = []
    for feat in cmr_cols:
        d = df[[score_col, group_col, feat]].copy().replace([np.inf,-np.inf], np.nan).dropna()
        if len(d) < 30 or d[group_col].nunique() < 2:
            continue
        x = pd.to_numeric(d[score_col], errors="coerce")
        y = pd.to_numeric(d[feat], errors="coerce")
        try:
            pr, pp = pearsonr(x,y)
        except Exception:
            pr, pp = np.nan, np.nan
        try:
            sr, sp = spearmanr(x,y)
        except Exception:
            sr, sp = np.nan, np.nan
        lo = y[d[group_col].eq(0)]
        hi = y[d[group_col].eq(1)]
        pooled = np.sqrt(((len(hi)-1)*hi.var(ddof=1)+(len(lo)-1)*lo.var(ddof=1))/max(len(hi)+len(lo)-2,1))
        cd = (hi.mean()-lo.mean())/pooled if pooled and np.isfinite(pooled) else np.nan
        try:
            wp = ttest_ind(hi,lo,equal_var=False,nan_policy="omit").pvalue
        except Exception:
            wp = np.nan
        rows.append({"feature": feat, "N": len(d), "N_low": len(lo), "N_high": len(hi), "Pearson_r": pr, "Pearson_p": pp, "Spearman_r": sr, "Spearman_p": sp, "Cohen_d_high_minus_low": cd, "Welch_p": wp})
    out = pd.DataFrame(rows)
    if len(out):
        for col in ["Pearson_p","Spearman_p","Welch_p"]:
            if col in out and multipletests is not None:
                mask = out[col].notna()
                out.loc[mask, col.replace("_p","_q")] = multipletests(out.loc[mask,col], method="fdr_bh")[1]
        out = out.sort_values(["Spearman_p","Welch_p"], na_position="last")
        out.to_csv(Path(tabledir)/f"{prefix}_CMR_correlation_group_comparison.csv", index=False)
    return out

def adjusted_linear_regressions(df, score_col, features, covars, tabledir, prefix="H4"):
    rows = []
    for feat in features:
        cols = [feat, score_col] + [c for c in covars if c in df.columns]
        d = df[cols].copy().replace([np.inf,-np.inf], np.nan).dropna()
        if len(d) < 50:
            continue
        # z-score outcome and predictors for comparable beta
        y = pd.to_numeric(d[feat], errors="coerce")
        y = (y - y.mean())/(y.std(ddof=0) or 1)
        X = pd.DataFrame({"intercept": 1.0, "score_z": (pd.to_numeric(d[score_col], errors="coerce") - pd.to_numeric(d[score_col], errors="coerce").mean())/(pd.to_numeric(d[score_col], errors="coerce").std(ddof=0) or 1)})
        for c in covars:
            if c not in d.columns: continue
            if d[c].dtype == "object" or str(d[c].dtype).startswith("string"):
                dum = pd.get_dummies(d[c].astype(str), prefix=c, drop_first=True, dtype=float)
                X = pd.concat([X, dum.reset_index(drop=True)], axis=1)
            else:
                z = pd.to_numeric(d[c], errors="coerce")
                X[c] = (z - z.mean())/(z.std(ddof=0) or 1)
        X = X.replace([np.inf,-np.inf], np.nan).fillna(0)
        try:
            beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
            yhat = X.values @ beta
            resid = y.values - yhat
            n,p = X.shape
            sigma2 = (resid @ resid)/max(n-p,1)
            covb = sigma2 * np.linalg.pinv(X.values.T @ X.values)
            se = np.sqrt(np.diag(covb))
            idx = list(X.columns).index("score_z")
            b, s = beta[idx], se[idx]
            zstat = b/s if s and np.isfinite(s) else np.nan
            pval = 2*(1-norm.cdf(abs(zstat))) if norm is not None and np.isfinite(zstat) else np.nan
            rows.append({"feature": feat, "N": len(d), "beta_per_1SD_CORI": b, "se": s, "ci_low": b-1.96*s, "ci_high": b+1.96*s, "p": pval})
        except Exception as e:
            rows.append({"feature": feat, "N": len(d), "beta_per_1SD_CORI": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p": np.nan, "error": str(e)[:120]})
    out = pd.DataFrame(rows)
    if len(out) and multipletests is not None:
        mask = out["p"].notna()
        out.loc[mask, "q"] = multipletests(out.loc[mask,"p"], method="fdr_bh")[1]
    out.to_csv(Path(tabledir)/f"{prefix}_adjusted_CMR_regressions.csv", index=False)
    return out
