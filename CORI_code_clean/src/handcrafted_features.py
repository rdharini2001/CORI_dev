import json
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
except Exception as e:
    raise ImportError("Please install lifelines in this kernel: pip install lifelines") from e

try:
    from sklearn.model_selection import StratifiedKFold
except Exception:
    StratifiedKFold = None

from cori_pipeline_utils_v13 import clean_id


# ------------------------------------------------------------
# Section logger. The notebook has one QCLogger per H-section (H1/H2/H3/H4);
# call set_logger(logger) once near the top of each section so _hc_log/_hc_section
# route into that section's own QC log instead of just printing.
# ------------------------------------------------------------
_active_logger = None


def set_logger(logger):
    global _active_logger
    _active_logger = logger


def _hc_log(msg):
    if _active_logger is not None and hasattr(_active_logger, "log"):
        _active_logger.log(str(msg))
    else:
        print(msg)


def _hc_section(title):
    if _active_logger is not None and hasattr(_active_logger, "section"):
        _active_logger.section(title)
    else:
        print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def _hc_pformat(p):
    try:
        if p is None or not np.isfinite(p): return "NA"
        if p < 0.001: return "<0.001"
        return f"{p:.3f}"
    except Exception:
        return "NA"

def _hc_fmt_ci(x, lo, hi, digits=2):
    vals = [x, lo, hi]
    if not all(pd.notna(v) and np.isfinite(v) for v in vals): return "NA"
    return f"{x:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"

def _hc_safe_name(x):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")

def _hc_extract_subject_id_from_filename(path):
    stem = Path(path).stem
    m = re.search(r"\d+", stem)
    return m.group(0) if m else stem

def load_handcrafted_features(feature_dir, id_col_name="eid"):
    """Load one handcrafted-feature CSV per subject/image and aggregate to subject level."""
    feature_dir = Path(feature_dir)
    files = sorted(feature_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No handcrafted feature CSVs found under: {feature_dir}")
    rows, errors = [], []
    for fp in files:
        sid = _hc_extract_subject_id_from_filename(fp)
        try:
            dat = pd.read_csv(fp, low_memory=False)
            if dat.empty:
                raise ValueError("empty CSV")
            dat = dat.loc[:, [c for c in dat.columns if not str(c).lower().startswith("unnamed")]]
            numeric = dat.apply(pd.to_numeric, errors="coerce")
            numeric_cols = [c for c in numeric.columns if numeric[c].notna().any()]
            if len(numeric_cols) == 0:
                raise ValueError("no numeric handcrafted columns")
            vals = numeric[numeric_cols].mean(axis=0, skipna=True).to_dict()
            vals[id_col_name] = str(sid)
            vals["n_handcrafted_rows_in_file"] = len(dat)
            vals["handcrafted_source_file"] = str(fp)
            rows.append(vals)
        except Exception as e:
            errors.append({"file": str(fp), "error": repr(e)})
    hc = pd.DataFrame(rows)
    if hc.empty:
        raise RuntimeError("No handcrafted feature files could be read.")
    non_feature_cols = {id_col_name, "handcrafted_source_file"}
    numeric_cols = [c for c in hc.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(hc[c])]
    hc_subject = hc.groupby(id_col_name, as_index=False)[numeric_cols].mean()
    rename = {c: f"HC_{c}" for c in numeric_cols if c != "n_handcrafted_rows_in_file"}
    hc_subject = hc_subject.rename(columns=rename)
    counts = hc.groupby(id_col_name).size().reset_index(name="n_handcrafted_files")
    hc_subject = hc_subject.merge(counts, on=id_col_name, how="left")
    err_df = pd.DataFrame(errors)
    _hc_log(f"Handcrafted files found: {len(files)}; subjects loaded: {hc_subject[id_col_name].nunique()}; errors: {len(err_df)}")
    return hc_subject, err_df

def merge_handcrafted(df, hc, id_candidates=("eid", "subject_id", "participant_id", "ID2", "id")):
    d, h = df.copy(), hc.copy()
    chosen = next((c for c in id_candidates if c in d.columns), None)
    if chosen is None:
        raise ValueError(f"No ID column found. Tried: {id_candidates}")
    d["_hc_id"] = d[chosen].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    h["_hc_id"] = h.iloc[:, 0].astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    out = d.merge(h.drop(columns=[h.columns[0]]), on="_hc_id", how="left")
    out = out.drop(columns=["_hc_id"])
    return out, chosen

# ------------------------------------------------------------
# 1. Merge cached handcrafted features into H2 cohorts
# ROBUST FIX: do not rely on older merge_handcrafted() return type
# ------------------------------------------------------------

def merge_handcrafted_cached_safe(
    cohort_df,
    hc_df,
    *,
    cohort_name,
    id_col="eid",
    feature_prefix="HC_",
):
    """
    Robust cached handcrafted-feature merge.

    Returns
    -------
    merged_df : pd.DataFrame
        Cohort dataframe with handcrafted features merged by eid.
    qc_df : pd.DataFrame
        One-row dataframe summarizing merge success.

    Why this exists
    ---------------
    Some earlier merge_handcrafted() versions return a string/status object
    as the second output. The H2 cell expects a dataframe and then calls
    .assign(...), which causes:
        AttributeError: 'str' object has no attribute 'assign'
    This helper always returns a dataframe QC object.
    """

    if cohort_df is None or not isinstance(cohort_df, pd.DataFrame):
        raise TypeError(f"{cohort_name}: cohort_df must be a pandas DataFrame.")

    if hc_df is None or not isinstance(hc_df, pd.DataFrame):
        raise TypeError(f"{cohort_name}: hc_df must be a pandas DataFrame.")

    if id_col not in cohort_df.columns:
        raise ValueError(f"{cohort_name}: cohort dataframe missing id column '{id_col}'.")

    if id_col not in hc_df.columns:
        raise ValueError(f"{cohort_name}: handcrafted dataframe missing id column '{id_col}'.")

    left = cohort_df.copy()
    right = hc_df.copy()

    left[id_col] = clean_id(left[id_col])
    right[id_col] = clean_id(right[id_col])

    n_left_raw = len(left)
    n_left_unique = left[id_col].nunique(dropna=True)

    n_right_raw = len(right)
    n_right_unique = right[id_col].nunique(dropna=True)

    # Ensure one row per subject in handcrafted matrix
    right = right.drop_duplicates(id_col, keep="first").copy()

    # Identify usable handcrafted features
    hc_candidate_cols = [c for c in right.columns if c != id_col]

    numeric_hc_cols = []
    for c in hc_candidate_cols:
        x = pd.to_numeric(right[c], errors="coerce")
        if x.notna().sum() > 0 and x.nunique(dropna=True) > 1:
            numeric_hc_cols.append(c)
            right[c] = x

    if len(numeric_hc_cols) == 0:
        raise ValueError(f"{cohort_name}: no usable numeric handcrafted features found.")

    # Avoid duplicate feature columns if rerunning cell
    existing_overlap = [c for c in numeric_hc_cols if c in left.columns]
    if existing_overlap:
        left = left.drop(columns=existing_overlap)

    before_cols = set(left.columns)

    merged = left.merge(
        right[[id_col] + numeric_hc_cols],
        on=id_col,
        how="left",
        validate="m:1",
    )

    matched_mask = merged[numeric_hc_cols].notna().any(axis=1)
    n_matched = int(matched_mask.sum())
    n_unmatched = int(len(merged) - n_matched)

    # Keep all rows, because downstream functions may handle missing features;
    # but record how many actually have feature coverage.
    added_cols = [c for c in merged.columns if c not in before_cols and c != id_col]

    qc_df = pd.DataFrame([{
        "cohort": cohort_name,
        "cohort_rows_raw": n_left_raw,
        "cohort_unique_eids": n_left_unique,
        "handcrafted_rows_raw": n_right_raw,
        "handcrafted_unique_eids": n_right_unique,
        "merged_rows": len(merged),
        "matched_rows_any_handcrafted_feature": n_matched,
        "unmatched_rows": n_unmatched,
        "match_rate": n_matched / max(len(merged), 1),
        "n_numeric_handcrafted_features_available": len(numeric_hc_cols),
        "n_handcrafted_features_added": len(added_cols),
        "feature_examples": str(numeric_hc_cols[:10]),
        "status": "ok" if n_matched > 0 else "no_matches",
    }])

    if n_matched == 0:
        print(f"WARNING: {cohort_name}: no rows matched handcrafted features by {id_col}.")

    return merged, qc_df


def discover_handcrafted_cols(df, min_nonmissing=0.50):
    cols = [c for c in df.columns if str(c).startswith("HC_")]
    good = []
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().mean() >= min_nonmissing and s.nunique(dropna=True) > 1:
            good.append(c)
    return good

def prepare_hc_matrix(train_df, test_df, cols):
    Xtr = train_df[cols].apply(pd.to_numeric, errors="coerce").copy()
    Xte = test_df[cols].apply(pd.to_numeric, errors="coerce").copy()
    med = Xtr.median(axis=0, skipna=True).fillna(0)
    Xtr, Xte = Xtr.fillna(med), Xte.fillna(med)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=0).replace(0, 1).fillna(1)
    return (Xtr - mu) / sd, (Xte - mu) / sd, med, mu, sd

def univariate_cox_rank_hc(train_df, feature_cols, time_col="time", event_col="event", penalizer=0.01):
    rows = []
    base = train_df[[time_col, event_col] + feature_cols].copy()
    base[time_col] = pd.to_numeric(base[time_col], errors="coerce")
    base[event_col] = pd.to_numeric(base[event_col], errors="coerce").fillna(0).astype(int)
    for col in feature_cols:
        d = base[[time_col, event_col, col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        d[col] = pd.to_numeric(d[col], errors="coerce")
        if len(d) < 30 or d[event_col].sum() < 5 or d[col].nunique() < 2:
            rows.append({"feature": col, "status": "skip", "p": np.nan, "hr_per_sd": np.nan, "z_abs": np.nan})
            continue
        sd = d[col].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            rows.append({"feature": col, "status": "skip", "p": np.nan, "hr_per_sd": np.nan, "z_abs": np.nan})
            continue
        d[col] = (d[col] - d[col].mean()) / sd
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(d.rename(columns={time_col:"time", event_col:"event"}), duration_col="time", event_col="event")
            row = cph.summary.loc[col]
            rows.append({
                "feature": col, "status": "ok", "coef": float(row["coef"]),
                "hr_per_sd": float(np.exp(row["coef"])), "p": float(row["p"]),
                "z_abs": float(abs(row.get("z", np.nan))),
                "importance": float(-np.log10(max(float(row["p"]), 1e-300))),
            })
        except Exception as e:
            rows.append({"feature": col, "status": f"error: {e}", "p": np.nan, "hr_per_sd": np.nan, "z_abs": np.nan})
    rank = pd.DataFrame(rows)
    if "importance" not in rank.columns: rank["importance"] = np.nan
    rank = rank.sort_values(["status", "p", "z_abs"], ascending=[True, True, False]).reset_index(drop=True)
    ok = rank["status"].eq("ok")
    rank.loc[ok, "rank"] = np.arange(1, ok.sum()+1)
    return rank

def _hc_cindex(df, score_col, time_col="time", event_col="event"):
    d = df[[time_col, event_col, score_col]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 10 or d[event_col].nunique() < 2: return np.nan
    return float(concordance_index(d[time_col], -d[score_col], d[event_col]))

def _fit_hc_cox(train_df, features, penalizer=0.10, prefix="HCORI"):
    d = train_df[["time", "event"] + features].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 30 or d["event"].sum() < 5:
        raise ValueError(f"Not enough rows/events to fit {prefix}: N={len(d)}, events={int(d['event'].sum())}")
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(d, duration_col="time", event_col="event")
    return cph

def _score_from_fit(cph, df, features):
    coefs = cph.params_.reindex(features).fillna(0)
    return np.asarray(df[features] @ coefs)

def internal_choose_k_hc(train_df, rank, candidate_ks=(5,10,15,20,30,50), penalizer=0.10, random_seed=2026):
    ok_features = rank.loc[rank.status.eq("ok"), "feature"].tolist()
    if len(ok_features) == 0:
        raise ValueError("No valid handcrafted features after Cox ranking.")
    candidate_ks = [int(k) for k in candidate_ks if int(k) <= len(ok_features)] or [min(5, len(ok_features))]
    d = train_df[["time", "event"] + ok_features].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if StratifiedKFold is None or d["event"].sum() < 10 or len(d) < 80:
        chosen = min(10, len(ok_features))
        return chosen, pd.DataFrame([{"k": chosen, "mean_cindex": np.nan, "sd_cindex": np.nan, "note": "fallback_event_limited"}])
    y = d["event"].astype(int).values
    n_splits = min(5, int(d["event"].sum()), int((1-y).sum()))
    if n_splits < 2:
        chosen = min(10, len(ok_features))
        return chosen, pd.DataFrame([{"k": chosen, "mean_cindex": np.nan, "sd_cindex": np.nan, "note": "fallback_insufficient_folds"}])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    rows = []
    for k in candidate_ks:
        feats = ok_features[:k]
        vals = []
        for tr_idx, va_idx in cv.split(d, y):
            tr, va = d.iloc[tr_idx].copy(), d.iloc[va_idx].copy()
            try:
                cph = _fit_hc_cox(tr, feats, penalizer=penalizer, prefix=f"CV_k{k}")
                va[f"_score_k{k}"] = _score_from_fit(cph, va, feats)
                vals.append(_hc_cindex(va, f"_score_k{k}"))
            except Exception:
                vals.append(np.nan)
        vals = [v for v in vals if pd.notna(v)]
        rows.append({"k": k, "mean_cindex": float(np.mean(vals)) if vals else np.nan, "sd_cindex": float(np.std(vals)) if vals else np.nan, "note": "cv"})
    cv_df = pd.DataFrame(rows)
    ok = cv_df.dropna(subset=["mean_cindex"]).copy()
    if ok.empty:
        return min(10, len(ok_features)), cv_df
    best = ok.loc[ok["mean_cindex"].idxmax()]
    threshold = best["mean_cindex"] - (0 if pd.isna(best["sd_cindex"]) else best["sd_cindex"])
    chosen = int(ok[ok["mean_cindex"] >= threshold].sort_values("k").iloc[0]["k"])
    cv_df["chosen"] = cv_df["k"].eq(chosen)
    return chosen, cv_df

def train_handcrafted_survival_model(train_df, test_df, outdir, prefix="HCORI",
                                     candidate_ks=(5,10,15,20,30,50), max_missing=0.50,
                                     univariate_penalizer=0.01, multivariable_penalizer=0.10):
    outdir = Path(outdir)
    tabledir, modeldir, figdir = outdir / "tables", outdir / "models", outdir / "figures"
    for p in [tabledir, modeldir, figdir]: p.mkdir(parents=True, exist_ok=True)
    hc_cols = discover_handcrafted_cols(train_df, min_nonmissing=1-max_missing)
    keep = []
    for c in hc_cols:
        s = pd.to_numeric(train_df[c], errors="coerce")
        if s.notna().mean() >= (1-max_missing) and s.nunique(dropna=True) > 1 and s.var(skipna=True) > 1e-12:
            keep.append(c)
    hc_cols = keep
    _hc_log(f"{prefix}: candidate handcrafted columns after filters: {len(hc_cols)}")
    rank = univariate_cox_rank_hc(train_df, hc_cols, penalizer=univariate_penalizer)
    rank.to_csv(tabledir / f"{prefix}_handcrafted_univariate_cox_ranking.csv", index=False)
    chosen_k, cv_df = internal_choose_k_hc(train_df, rank, candidate_ks=candidate_ks, penalizer=multivariable_penalizer)
    cv_df.to_csv(tabledir / f"{prefix}_handcrafted_feature_count_cv.csv", index=False)
    selected = rank.loc[rank.status.eq("ok"), "feature"].tolist()[:chosen_k]
    pd.DataFrame({"feature": selected, "rank": np.arange(1, len(selected)+1)}).to_csv(tabledir / f"{prefix}_selected_handcrafted_features.csv", index=False)
    rank.head(10).to_csv(tabledir / f"{prefix}_top10_handcrafted_features.csv", index=False)
    Xtrz, Xtez, med, mu, sd = prepare_hc_matrix(train_df, test_df, selected)
    train_model_df = pd.concat([train_df[["time","event"]].reset_index(drop=True), Xtrz.reset_index(drop=True)], axis=1)
    test_model_df = pd.concat([test_df[["time","event"]].reset_index(drop=True), Xtez.reset_index(drop=True)], axis=1)
    cph = _fit_hc_cox(train_model_df, selected, penalizer=multivariable_penalizer, prefix=prefix)
    train_df[f"{prefix}_score"] = _score_from_fit(cph, train_model_df, selected)
    test_df[f"{prefix}_score"] = _score_from_fit(cph, test_model_df, selected)
    med_thr = float(np.nanmedian(train_df[f"{prefix}_score"])); q1, q2 = np.nanquantile(train_df[f"{prefix}_score"], [1/3, 2/3])
    for d in [train_df, test_df]:
        d[f"{prefix}_high_risk"] = (d[f"{prefix}_score"] >= med_thr).astype(int)
        d[f"{prefix}_risk_tertile"] = pd.cut(d[f"{prefix}_score"], bins=[-np.inf, q1, q2, np.inf], labels=[1,2,3], include_lowest=True).astype(int)
    manifest = {"prefix": prefix, "n_candidate_handcrafted_after_filter": len(hc_cols), "n_selected_features": len(selected), "selected_features": selected, "candidate_ks": list(candidate_ks), "chosen_k": int(chosen_k), "univariate_penalizer": univariate_penalizer, "multivariable_penalizer": multivariable_penalizer, "score_threshold_median": med_thr, "score_threshold_tertile_1": float(q1), "score_threshold_tertile_2": float(q2)}
    with open(tabledir / f"{prefix}_handcrafted_model_manifest.json", "w") as f: json.dump(manifest, f, indent=2, default=str)
    bundle = {"prefix": prefix, "model": cph, "features": selected, "feature_medians": med, "feature_means": mu, "feature_sds": sd, "thresholds": {"median": med_thr, "tertile_1": float(q1), "tertile_2": float(q2)}, "manifest": manifest, "feature_ranking": rank}
    with open(modeldir / f"{prefix}_locked_handcrafted_model_bundle.pkl", "wb") as f: pickle.dump(bundle, f)
    return train_df, test_df, selected, rank, bundle, cv_df

def apply_handcrafted_bundle(bundle_path_or_obj, df, prefix=None):
    if isinstance(bundle_path_or_obj, (str, Path)):
        with open(bundle_path_or_obj, "rb") as f: bundle = pickle.load(f)
    else:
        bundle = bundle_path_or_obj
    prefix = prefix or bundle.get("prefix", "HCORI")
    features = bundle["features"]
    d = df.copy()
    X = d[features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(bundle["feature_medians"].reindex(features).fillna(0))
    X = (X - bundle["feature_means"].reindex(features).fillna(0)) / bundle["feature_sds"].reindex(features).replace(0, 1).fillna(1)
    d[f"{prefix}_score"] = _score_from_fit(bundle["model"], X, features)
    thr = bundle["thresholds"]
    d[f"{prefix}_high_risk"] = (d[f"{prefix}_score"] >= thr["median"]).astype(int)
    d[f"{prefix}_risk_tertile"] = pd.cut(d[f"{prefix}_score"], bins=[-np.inf, thr["tertile_1"], thr["tertile_2"], np.inf], labels=[1,2,3], include_lowest=True).astype(int)
    return d

def hc_performance_row(df, label, score_col, group_col=None):
    row = {"cohort": label, "model": score_col.replace("_score", ""), "N": len(df), "Events": int(pd.to_numeric(df["event"], errors="coerce").fillna(0).sum())}
    row["C_index"] = _hc_cindex(df, score_col)
    if group_col is not None and group_col in df.columns and df[group_col].nunique() >= 2 and row["Events"] >= 5:
        d = df[["time","event",group_col]].dropna().copy()
        d[group_col] = pd.to_numeric(d[group_col], errors="coerce").astype(int)
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(d.rename(columns={group_col:"group"}), duration_col="time", event_col="event")
            r = cph.summary.loc["group"]
            row["HR_high_vs_low"] = float(np.exp(r["coef"])); row["HR_low"] = float(np.exp(r["coef lower 95%"])); row["HR_high"] = float(np.exp(r["coef upper 95%"])); row["HR_95CI"] = _hc_fmt_ci(row["HR_high_vs_low"], row["HR_low"], row["HR_high"], 2); row["p"] = float(r["p"]); row["p_fmt"] = _hc_pformat(row["p"])
        except Exception:
            row["HR_high_vs_low"] = np.nan; row["HR_95CI"] = "NA"; row["p"] = np.nan; row["p_fmt"] = "NA"
    return row

def plot_hc_km(df, group_col, title, out_base):
    d = df[["time", "event", group_col]].dropna().copy()
    if len(d) < 10 or d["event"].sum() < 5 or d[group_col].nunique() < 2:
        _hc_log(f"Skipping KM {title}: insufficient N/events/groups.")
        return
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for val, lab in [(0,"Low risk"), (1,"High risk")]:
        sub = d[pd.to_numeric(d[group_col], errors="coerce").astype(int).eq(val)]
        if len(sub) == 0: continue
        kmf = KaplanMeierFitter(); kmf.fit(sub["time"]/365.25, sub["event"], label=f"{lab}: N={len(sub)}, events={int(sub['event'].sum())}"); kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2)
    ax.set_title(title, fontweight="bold"); ax.set_xlabel("Years"); ax.set_ylabel("MACE-free survival"); ax.grid(alpha=0.25); ax.spines[["top","right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(str(out_base)+".png", dpi=300, bbox_inches="tight"); fig.savefig(str(out_base)+".pdf", bbox_inches="tight"); plt.show()

def plot_handcrafted_feature_transition(rank_left, rank_right, figdir, tabledir, left_label="HCORI", right_label="HMMACE", top_n=30, prefix="H2_HC"):
    figdir, tabledir = Path(figdir), Path(tabledir); figdir.mkdir(parents=True, exist_ok=True); tabledir.mkdir(parents=True, exist_ok=True)
    def clean(r, label):
        x = r.copy(); x = x[x["status"].eq("ok")].copy() if "status" in x.columns else x.copy(); x["p"] = pd.to_numeric(x.get("p", np.nan), errors="coerce"); x["importance"] = -np.log10(x["p"].clip(lower=1e-300)); x = x.sort_values(["p","importance"], ascending=[True,False]).drop_duplicates("feature").reset_index(drop=True); x[f"{label}_rank"] = np.arange(1, len(x)+1); keep = ["feature", f"{label}_rank", "importance", "hr_per_sd"]; return x[[c for c in keep if c in x.columns]].rename(columns={"importance":f"{label}_importance", "hr_per_sd":f"{label}_hr"})
    L, R = clean(rank_left, left_label), clean(rank_right, right_label)
    m = L.merge(R, on="feature", how="outer"); m = m[(m[f"{left_label}_rank"].le(top_n)) | (m[f"{right_label}_rank"].le(top_n))].copy()
    if m.empty: _hc_log("No features for handcrafted transition plot."); return m
    cap = top_n + 1; m[f"{left_label}_plot_rank"] = m[f"{left_label}_rank"].where(m[f"{left_label}_rank"].le(top_n), cap).fillna(cap); m[f"{right_label}_plot_rank"] = m[f"{right_label}_rank"].where(m[f"{right_label}_rank"].le(top_n), cap).fillna(cap); m["best_rank"] = m[[f"{left_label}_plot_rank", f"{right_label}_plot_rank"]].min(axis=1); m = m.sort_values(["best_rank", f"{left_label}_plot_rank", f"{right_label}_plot_rank"]).head(top_n).copy(); m.to_csv(tabledir / f"{prefix}_handcrafted_feature_transition_table.csv", index=False)
    fig_h = max(7, min(13, 0.3*len(m)+2)); fig, ax = plt.subplots(figsize=(10, fig_h))
    for _, r in m.iterrows():
        y0, y1 = r[f"{left_label}_plot_rank"], r[f"{right_label}_plot_rank"]; color = "tab:blue" if y1 < y0 else ("tab:red" if y1 > y0 else "0.45"); ax.plot([0,1], [y0,y1], color=color, alpha=0.75, linewidth=1.5); ax.scatter([0,1], [y0,y1], color=color, s=45, zorder=3)
        lab = lambda v: f">{top_n}" if pd.isna(v) or v > top_n else str(int(v)); ax.text(-0.04, y0, f"{str(r['feature']).replace('HC_','')} ({lab(r.get(f'{left_label}_rank'))})", ha="right", va="center", fontsize=7); ax.text(1.04, y1, f"{str(r['feature']).replace('HC_','')} ({lab(r.get(f'{right_label}_rank'))})", ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(cap+0.8, 0.5); ax.set_xticks([0,1]); ax.set_xticklabels([f"{left_label}\nCancer-trained", f"{right_label}\nNon-cancer-trained"]); yticks = list(range(1, top_n+1, 2)) + [cap]; ax.set_yticks(yticks); ax.set_yticklabels([str(x) for x in yticks[:-1]]+[f">{top_n}"]); ax.set_ylabel("Handcrafted feature rank; 1 = strongest"); ax.set_title("Interpretable handcrafted feature-rank transition", fontweight="bold"); ax.grid(axis="y", alpha=0.2); ax.spines[["top","right"]].set_visible(False); fig.tight_layout(); fig.savefig(figdir / f"{prefix}_handcrafted_feature_transition.png", dpi=300, bbox_inches="tight"); fig.savefig(figdir / f"{prefix}_handcrafted_feature_transition.pdf", bbox_inches="tight"); plt.show(); return m


# ============================================================
# Better handcrafted feature-transition visualization for H2
# Replaces plot_handcrafted_feature_transition(...)
# ============================================================

def plot_handcrafted_feature_transition_clean(
    hcori_rank,
    hmmace_rank,
    figdir,
    tabledir,
    *,
    left_label="HCORI",
    right_label="HMMACE",
    top_n=20,
    top_each_side=12,
    prefix="H2_HC",
):
    """
    Cleaner visualization for interpretable handcrafted feature-rank transition.

    Why this is better than the old ribbon plot:
    - Avoids plotting dozens of '>top_n' labels on the same row.
    - Shortens long handcrafted feature names.
    - Separates three ideas:
        1) rank transition among selected top features,
        2) strongest cancer-trained HCORI features,
        3) strongest non-cancer H-M-MACE features.
    """

    figdir = Path(figdir)
    tabledir = Path(tabledir)
    figdir.mkdir(parents=True, exist_ok=True)
    tabledir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _clean_feature_name(x, max_len=42):
        x = str(x)
        x = x.replace("HC_", "")
        x = x.replace("handcrafted_", "")
        x = x.replace("vessel_fundus_ring_", "vessel fundus ring ")
        x = x.replace("vein_fundus_ring_", "vein fundus ring ")
        x = x.replace("artery_fundus_ring_", "artery fundus ring ")
        x = x.replace("vessel_disc_ring_", "vessel disc ring ")
        x = x.replace("vessel_All_", "vessel all ")
        x = x.replace("_", " ")
        x = " ".join(x.split())
        if len(x) > max_len:
            return x[:max_len - 1] + "…"
        return x

    def _prepare_rank_table(rank_df, label):
        r = rank_df.copy()

        if "status" in r.columns:
            r = r[r["status"].astype(str).eq("ok")].copy()

        if "feature" not in r.columns:
            raise ValueError(f"{label} rank table must contain a 'feature' column.")

        for c in ["p", "z_abs", "importance", "coef", "hr_per_sd"]:
            if c in r.columns:
                r[c] = pd.to_numeric(r[c], errors="coerce")

        if "importance" not in r.columns:
            r["importance"] = np.nan

        if "p" in r.columns:
            r["importance"] = r["importance"].fillna(
                -np.log10(r["p"].clip(lower=1e-300))
            )

        if "p" in r.columns and "z_abs" in r.columns:
            r = r.sort_values(["p", "z_abs"], ascending=[True, False]).copy()
        elif "importance" in r.columns:
            r = r.sort_values("importance", ascending=False).copy()
        else:
            r = r.reset_index(drop=True).copy()

        r = r.drop_duplicates("feature", keep="first").reset_index(drop=True)
        r[f"{label}_rank"] = np.arange(1, len(r) + 1)
        r[f"{label}_importance"] = r["importance"]

        if "hr_per_sd" in r.columns:
            r[f"{label}_hr"] = r["hr_per_sd"]
        elif "coef" in r.columns:
            r[f"{label}_hr"] = np.exp(r["coef"])
        else:
            r[f"{label}_hr"] = np.nan

        keep = ["feature", f"{label}_rank", f"{label}_importance", f"{label}_hr"]
        return r[keep]

    hc = _prepare_rank_table(hcori_rank, "HCORI")
    hm = _prepare_rank_table(hmmace_rank, "HMMACE")

    merged = hc.merge(hm, on="feature", how="outer")

    # True full-rank difference
    merged["rank_shift_HMMACE_minus_HCORI"] = (
        merged["HMMACE_rank"] - merged["HCORI_rank"]
    )

    merged["best_rank"] = merged[["HCORI_rank", "HMMACE_rank"]].min(axis=1)

    def _membership(row):
        in_hc = pd.notna(row["HCORI_rank"]) and row["HCORI_rank"] <= top_n
        in_hm = pd.notna(row["HMMACE_rank"]) and row["HMMACE_rank"] <= top_n
        if in_hc and in_hm:
            return "Top in both"
        if in_hc:
            return "HCORI top only"
        if in_hm:
            return "H-M-MACE top only"
        return "Outside top set"

    merged["membership"] = merged.apply(_membership, axis=1)
    merged["feature_short"] = merged["feature"].map(_clean_feature_name)

    # Save full transition table
    merged.sort_values(["best_rank", "HCORI_rank", "HMMACE_rank"]).to_csv(
        tabledir / f"{prefix}_handcrafted_feature_transition_FULL_TABLE.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plot 1: clean selected bump plot
    # --------------------------------------------------------
    selected_hc = merged[merged["HCORI_rank"].le(top_each_side)].copy()
    selected_hm = merged[merged["HMMACE_rank"].le(top_each_side)].copy()

    selected = (
        pd.concat([selected_hc, selected_hm], axis=0)
        .drop_duplicates("feature", keep="first")
        .copy()
    )

    selected["best_rank"] = selected[["HCORI_rank", "HMMACE_rank"]].min(axis=1)
    selected = selected.sort_values(["best_rank", "HCORI_rank", "HMMACE_rank"]).copy()

    cap = top_n + 1
    selected["HCORI_plot_rank"] = selected["HCORI_rank"].where(
        selected["HCORI_rank"].le(top_n), cap
    ).fillna(cap)

    selected["HMMACE_plot_rank"] = selected["HMMACE_rank"].where(
        selected["HMMACE_rank"].le(top_n), cap
    ).fillna(cap)

    def _color(row):
        if row["membership"] == "HCORI top only":
            return "tab:red"
        if row["membership"] == "H-M-MACE top only":
            return "tab:blue"
        return "0.35"

    def _rank_label(x):
        if pd.isna(x) or x > top_n:
            return f">{top_n}"
        return str(int(x))

    # Stagger bottom labels slightly to avoid pile-up
    bottom_hc_counter = 0
    bottom_hm_counter = 0
    hc_y = []
    hm_y = []
    for _, row in selected.iterrows():
        y0 = row["HCORI_plot_rank"]
        y1 = row["HMMACE_plot_rank"]

        if y0 == cap:
            y0 = cap + 0.10 * bottom_hc_counter
            bottom_hc_counter += 1
        if y1 == cap:
            y1 = cap + 0.10 * bottom_hm_counter
            bottom_hm_counter += 1

        hc_y.append(y0)
        hm_y.append(y1)

    selected["HCORI_y"] = hc_y
    selected["HMMACE_y"] = hm_y

    fig_h = max(6.5, min(10.5, 0.28 * len(selected) + 2.5))
    fig, ax = plt.subplots(figsize=(10.8, fig_h))

    for _, row in selected.iterrows():
        c = _color(row)

        ax.plot(
            [0, 1],
            [row["HCORI_y"], row["HMMACE_y"]],
            color=c,
            linewidth=1.6,
            alpha=0.75,
            zorder=1,
        )

        ax.scatter(
            [0, 1],
            [row["HCORI_y"], row["HMMACE_y"]],
            s=55,
            color=c,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

        # Label only if top-ranked on that side.
        if pd.notna(row["HCORI_rank"]) and row["HCORI_rank"] <= top_n:
            ax.text(
                -0.04,
                row["HCORI_y"],
                f"{row['feature_short']} ({_rank_label(row['HCORI_rank'])})",
                ha="right",
                va="center",
                fontsize=7.2,
            )

        if pd.notna(row["HMMACE_rank"]) and row["HMMACE_rank"] <= top_n:
            ax.text(
                1.04,
                row["HMMACE_y"],
                f"{row['feature_short']} ({_rank_label(row['HMMACE_rank'])})",
                ha="left",
                va="center",
                fontsize=7.2,
            )

    ymax = cap + max(bottom_hc_counter, bottom_hm_counter) * 0.10 + 0.8
    ax.set_ylim(ymax, 0.5)
    ax.set_xlim(-0.58, 1.58)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        [
            f"{left_label}\nCancer-trained",
            f"{right_label}\nNon-cancer-trained",
        ],
        fontsize=10,
    )

    yticks = list(range(1, top_n + 1, 2)) + [cap]
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(x) for x in yticks[:-1]] + [f">{top_n}"])
    ax.set_ylabel("Feature rank; 1 = strongest prognostic feature")
    ax.set_title(
        "Handcrafted retinal feature-rank transition",
        fontweight="bold",
    )

    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    legend_items = [
        Line2D([0], [0], color="tab:red", lw=2.5, label="Cancer-trained HCORI top feature"),
        Line2D([0], [0], color="tab:blue", lw=2.5, label="Non-cancer H-M-MACE top feature"),
        Line2D([0], [0], color="0.35", lw=2.5, label="Top-ranked in both"),
    ]

    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=1,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(
        figdir / f"{prefix}_handcrafted_feature_transition_CLEAN_BUMP.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        figdir / f"{prefix}_handcrafted_feature_transition_CLEAN_BUMP.pdf",
        bbox_inches="tight",
    )
    plt.show()

    # Save selected transition table
    selected.to_csv(
        tabledir / f"{prefix}_handcrafted_feature_transition_CLEAN_BUMP_table.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plot 2 and 3: top feature bar plots by model
    # --------------------------------------------------------
    def _top_barplot(rank_df, rank_col, importance_col, title, save_name, color):
        top = rank_df.copy()
        top = top[pd.to_numeric(top[rank_col], errors="coerce").le(top_n)].copy()
        top = top.sort_values(rank_col).head(top_n).copy()
        top["feature_short"] = top["feature"].map(lambda x: _clean_feature_name(x, max_len=55))

        # Use importance if available; otherwise inverse rank
        top["plot_value"] = pd.to_numeric(top[importance_col], errors="coerce")
        top["plot_value"] = top["plot_value"].fillna(top_n + 1 - top[rank_col])

        top = top.iloc[::-1].copy()

        fig_h = max(5.2, 0.30 * len(top) + 1.5)
        fig, ax = plt.subplots(figsize=(8.8, fig_h))

        ax.barh(top["feature_short"], top["plot_value"], color=color, alpha=0.85)
        ax.set_xlabel("Univariate Cox importance, -log10(p)")
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", alpha=0.20)
        ax.spines[["top", "right"]].set_visible(False)

        # Add rank labels
        for i, (_, r) in enumerate(top.iterrows()):
            ax.text(
                r["plot_value"],
                i,
                f"  rank {int(r[rank_col])}",
                va="center",
                fontsize=7.5,
            )

        fig.tight_layout()
        fig.savefig(figdir / f"{save_name}.png", dpi=300, bbox_inches="tight")
        fig.savefig(figdir / f"{save_name}.pdf", bbox_inches="tight")
        plt.show()

        return top

    top_hc = _top_barplot(
        merged.dropna(subset=["HCORI_rank"]),
        "HCORI_rank",
        "HCORI_importance",
        "Top handcrafted features in cancer-trained HCORI",
        f"{prefix}_top_HCORI_handcrafted_features",
        "tab:red",
    )

    top_hm = _top_barplot(
        merged.dropna(subset=["HMMACE_rank"]),
        "HMMACE_rank",
        "HMMACE_importance",
        "Top handcrafted features in non-cancer H-M-MACE",
        f"{prefix}_top_HMMACE_handcrafted_features",
        "tab:blue",
    )

    top_hc.to_csv(
        tabledir / f"{prefix}_top_HCORI_handcrafted_features.csv",
        index=False,
    )
    top_hm.to_csv(
        tabledir / f"{prefix}_top_HMMACE_handcrafted_features.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Plot 4: overlap summary
    # --------------------------------------------------------
    overlap_rows = []
    for k in [5, 10, 15, 20, 30]:
        hc_set = set(merged.loc[merged["HCORI_rank"].le(k), "feature"])
        hm_set = set(merged.loc[merged["HMMACE_rank"].le(k), "feature"])
        overlap_rows.append(
            {
                "top_k": k,
                "HCORI_top_k": len(hc_set),
                "HMMACE_top_k": len(hm_set),
                "overlap_n": len(hc_set.intersection(hm_set)),
                "jaccard": (
                    len(hc_set.intersection(hm_set)) / len(hc_set.union(hm_set))
                    if len(hc_set.union(hm_set)) else np.nan
                ),
            }
        )

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(
        tabledir / f"{prefix}_handcrafted_feature_overlap_by_topk.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(overlap["top_k"], overlap["jaccard"], marker="o", linewidth=2)
    ax.set_xlabel("Top-k features")
    ax.set_ylabel("Jaccard overlap")
    ax.set_title("Feature-overlap between HCORI and H-M-MACE", fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(
        figdir / f"{prefix}_handcrafted_feature_overlap_jaccard.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        figdir / f"{prefix}_handcrafted_feature_overlap_jaccard.pdf",
        bbox_inches="tight",
    )
    plt.show()

    return selected, merged, overlap


# ============================================================
# Cached handcrafted-feature loaders (H2/H3 reuse H1's cache instead
# of re-reading the raw one-CSV-per-subject OneDrive folder).
# ============================================================

def load_cached_handcrafted_features(cache_candidates, id_col_name="eid"):
    """
    Load a pre-cached handcrafted-feature matrix, trying each candidate path
    in order and using the first one that exists.

    This function intentionally does NOT read the original OneDrive folder of
    one CSV per subject. It only loads a previously cached subject-level matrix.

    Required output format:
        one row per eid
        eid column
        handcrafted feature columns
    """
    cache_candidates = [Path(p) for p in cache_candidates if p is not None]

    existing = [p for p in cache_candidates if p.exists()]

    if len(existing) == 0:
        msg = (
            "No cached handcrafted feature matrix found.\n\n"
            "Expected one of these files:\n"
            + "\n".join([str(p) for p in cache_candidates])
            + "\n\n"
            "Fix: run the H1 handcrafted cache-building cell first, or manually save a subject-level "
            "handcrafted feature matrix as one of the paths above. This H2 cell intentionally avoids "
            "reading the OneDrive one-CSV-per-subject feature folder."
        )
        raise FileNotFoundError(msg)

    cache_path = existing[0]
    print(f"Loading cached handcrafted features from: {cache_path}")

    suffix = cache_path.suffix.lower()

    if suffix in [".pkl", ".pickle"]:
        hc = pd.read_pickle(cache_path)
    elif suffix == ".parquet":
        hc = pd.read_parquet(cache_path)
    elif suffix == ".csv":
        hc = pd.read_csv(cache_path, low_memory=False)
    else:
        raise ValueError(f"Unsupported handcrafted cache format: {cache_path}")

    if id_col_name not in hc.columns:
        # Try common alternatives
        alt_id_cols = ["eid", "subject_id", "participant_id", "ID", "id"]
        found = None
        for c in alt_id_cols:
            if c in hc.columns:
                found = c
                break
        if found is None:
            raise ValueError(
                f"Cached handcrafted feature file must contain '{id_col_name}' or one of {alt_id_cols}. "
                f"Columns found: {hc.columns[:20].tolist()}"
            )
        hc = hc.rename(columns={found: id_col_name})

    hc[id_col_name] = clean_id(hc[id_col_name])

    # Drop duplicate subjects after loading cache
    n_before = len(hc)
    hc = hc.drop_duplicates(id_col_name, keep="first").copy()
    n_after = len(hc)

    # Basic feature sanity check
    non_id_cols = [c for c in hc.columns if c != id_col_name]
    numeric_feature_cols = []
    for c in non_id_cols:
        x = pd.to_numeric(hc[c], errors="coerce")
        if x.notna().sum() > 0 and x.nunique(dropna=True) > 1:
            numeric_feature_cols.append(c)

    print(f"Cached handcrafted rows before duplicate removal: {n_before}")
    print(f"Cached handcrafted unique subjects: {n_after}")
    print(f"Usable numeric handcrafted feature columns: {len(numeric_feature_cols)}")

    if len(numeric_feature_cols) == 0:
        raise ValueError("Cached handcrafted feature matrix has no usable numeric feature columns.")

    return hc, {
        "cache_path": str(cache_path),
        "n_rows_raw": n_before,
        "n_unique_subjects": n_after,
        "n_numeric_features": len(numeric_feature_cols),
        "numeric_feature_examples": numeric_feature_cols[:20],
    }


def load_cached_handcrafted_features_exact(path, id_col_name="eid"):
    """
    Load one specific cached handcrafted-feature CSV (no candidate search).

    Used by H3, which always reuses H1's single cached feature file rather
    than searching several possible locations (that's what
    load_cached_handcrafted_features is for).
    """
    hc = pd.read_csv(path, low_memory=False)

    if id_col_name not in hc.columns:
        first_col = hc.columns[0]
        _hc_log(f"'{id_col_name}' not found in cached handcrafted file; renaming first column '{first_col}' to '{id_col_name}'.")
        hc = hc.rename(columns={first_col: id_col_name})

    hc[id_col_name] = (
        hc[id_col_name]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
    )

    numeric_cols = [
        c for c in hc.columns
        if c != id_col_name and pd.api.types.is_numeric_dtype(hc[c])
    ]

    if hc[id_col_name].duplicated().any():
        _hc_log(
            f"Cached handcrafted file has duplicate {id_col_name}s: "
            f"{int(hc[id_col_name].duplicated().sum())}. Aggregating numeric features by mean."
        )
        hc = (
            hc[[id_col_name] + numeric_cols]
            .groupby(id_col_name, as_index=False)
            .mean()
        )

    _hc_log(
        f"Cached handcrafted subjects loaded: {hc[id_col_name].nunique()} | "
        f"numeric feature columns: {len(numeric_cols)}"
    )

    return hc
