
from __future__ import annotations

import json, re, math, warnings, pickle
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

try:
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.utils import concordance_index
    from lifelines.statistics import logrank_test, multivariate_logrank_test
except Exception as e:
    CoxPHFitter = None
    KaplanMeierFitter = None
    concordance_index = None
    logrank_test = None
    multivariate_logrank_test = None

try:
    from scipy.stats import chi2, norm, ttest_ind, mannwhitneyu, pearsonr, spearmanr
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import pdist
except Exception:
    chi2 = norm = ttest_ind = mannwhitneyu = pearsonr = spearmanr = linkage = dendrogram = leaves_list = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

RANDOM_SEED = 2026
np.random.seed(RANDOM_SEED)

# -----------------------------
# Logging and output helpers
# -----------------------------
class QCLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"CORI QC LOG\nStarted: {datetime.now()}\n{'='*100}\n", encoding="utf-8")
    def log(self, msg: str):
        print(msg)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    def section(self, title: str):
        self.log("\n" + "="*100 + f"\n{title}\n" + "="*100)
    def df(self, name: str, df: pd.DataFrame, max_rows: int = 30):
        self.section(name)
        if df is None:
            self.log("<None>")
            return
        self.log(df.head(max_rows).to_string(index=False))
        self.log(f"[shape={df.shape}]")

def ensure_dirs(base: Path):
    base = Path(base)
    fig = base / "figures"
    tab = base / "tables"
    mod = base / "models"
    qc = base / "qc"
    for p in [base, fig, tab, mod, qc]:
        p.mkdir(parents=True, exist_ok=True)
    return base, fig, tab, mod, qc

def savefig(fig, basepath):
    basepath = Path(basepath)
    basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(basepath)+".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(basepath)+".pdf", bbox_inches="tight")
    fig.savefig(str(basepath)+".svg", bbox_inches="tight")
    
    print("Images saved to:", str(basepath)+".{png,pdf,svg}")
    

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def clean_id(x):
    return pd.Series(x).astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

def load_csv(path, logger=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    if logger: logger.log(f"Loading CSV: {path}")
    return pd.read_csv(path, low_memory=False)

def as_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def pformat(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def fmt_ci(x, lo, hi, digits=2):
    if not all(np.isfinite(v) for v in [x, lo, hi]):
        return "NA"
    return f"{x:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"

# -----------------------------
# Cohort and column helpers
# -----------------------------
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

# -----------------------------
# Feature handling and Cox model
# -----------------------------
def variance_filter(train_df, feature_cols, threshold=1e-8, max_missing=0.50):
    rows = []
    keep = []
    for c in feature_cols:
        if c not in train_df.columns:
            continue
        x = pd.to_numeric(train_df[c], errors="coerce")
        miss = x.isna().mean()
        var = x.var(skipna=True)
        ok = (miss <= max_missing) and np.isfinite(var) and (var > threshold)
        rows.append({"feature": c, "missing_fraction": miss, "variance": var, "status": "keep" if ok else "drop"})
        if ok:
            keep.append(c)
    return keep, pd.DataFrame(rows)

def _prepare_design_from_training(train_df, test_df, features):
    # Development-only imputation and scaling. Returns matrices and preprocessing parameters.
    Xtr = train_df[features].apply(pd.to_numeric, errors="coerce").copy()
    Xte = test_df[features].apply(pd.to_numeric, errors="coerce").copy()
    med = Xtr.median(axis=0)
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)
    mean = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0, ddof=0).replace(0, 1.0).fillna(1.0)
    Xtr = (Xtr - mean) / sd
    Xte = (Xte - mean) / sd
    return Xtr, Xte, med, mean, sd

def prepare_design_with_params(df, features, med, mean, sd):
    X = df[features].apply(pd.to_numeric, errors="coerce").copy()
    X = X.fillna(med)
    X = (X - mean) / sd.replace(0, 1.0).fillna(1.0)
    return X

def fit_cox_model(train_df, test_df, features, penalizer_grid=(0.01,0.05,0.1,0.25)):
    if CoxPHFitter is None:
        raise ImportError("lifelines is required.")
    Xtr, Xte, med, mean, sd = _prepare_design_from_training(train_df, test_df, features)
    tr = pd.concat([train_df[["time","event"]].reset_index(drop=True), Xtr.reset_index(drop=True)], axis=1)
    te = pd.concat([test_df[["time","event"]].reset_index(drop=True), Xte.reset_index(drop=True)], axis=1)
    best = None
    errs = []
    for pen in penalizer_grid:
        try:
            cph = CoxPHFitter(penalizer=pen)
            cph.fit(tr, duration_col="time", event_col="event")
            risk = np.log(np.asarray(cph.predict_partial_hazard(te)).reshape(-1))
            cidx = concordance_index(te["time"], -risk, te["event"]) if te["event"].nunique() > 1 else np.nan
            if best is None or (np.isfinite(cidx) and cidx > best["c_index"]):
                best = {"model": cph, "penalizer": pen, "c_index": cidx, "train_matrix": tr, "test_matrix": te, "median": med, "mean": mean, "sd": sd, "features": list(features)}
        except Exception as e:
            errs.append((pen, str(e)))
    if best is None:
        raise RuntimeError("All Cox model fits failed: " + repr(errs[:3]))
    return best

def univariate_cox_rank(train_df, feature_cols, penalizer=0.01, logger=None):
    rows = []
    for i,c in enumerate(feature_cols):
        d = train_df[["time","event",c]].copy()
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d = d.replace([np.inf,-np.inf], np.nan).dropna()
        if len(d) < 50 or d["event"].sum() < 5 or d[c].nunique() <= 2:
            rows.append({"feature": c, "status": "skip", "n": len(d), "events": int(d["event"].sum()) if len(d) else 0, "hr_per_sd": np.nan, "p": np.nan, "z_abs": np.nan, "importance": np.nan})
            continue
        sd = d[c].std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            rows.append({"feature": c, "status": "skip", "n": len(d), "events": int(d["event"].sum()), "hr_per_sd": np.nan, "p": np.nan, "z_abs": np.nan, "importance": np.nan})
            continue
        d[c] = (d[c] - d[c].mean()) / sd
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(d, "time", "event")
            summ = cph.summary.loc[c]
            p = float(summ.get("p", np.nan))
            z = float(abs(summ.get("z", np.nan)))
            hr = float(np.exp(summ.get("coef", np.nan)))
            imp = -np.log10(max(p, 1e-300)) if np.isfinite(p) else np.nan
            rows.append({"feature": c, "status": "ok", "n": len(d), "events": int(d["event"].sum()), "hr_per_sd": hr, "p": p, "z_abs": z, "importance": imp, "coef": float(summ.get("coef", np.nan))})
        except Exception as e:
            rows.append({"feature": c, "status": f"fail: {str(e)[:80]}", "n": len(d), "events": int(d["event"].sum()), "hr_per_sd": np.nan, "p": np.nan, "z_abs": np.nan, "importance": np.nan})
    rank = pd.DataFrame(rows)
    # Important: p ascending, then z descending
    rank["p_sort"] = rank["p"].fillna(1.0)
    rank["z_sort"] = rank["z_abs"].fillna(0.0)
    rank = rank.sort_values(["status","p_sort","z_sort"], ascending=[True, True, False]).drop(columns=["p_sort","z_sort"])
    # Put ok first explicitly
    rank["status_order"] = np.where(rank["status"].eq("ok"), 0, 1)
    rank = rank.sort_values(["status_order","p","z_abs"], ascending=[True, True, False]).drop(columns=["status_order"]).reset_index(drop=True)
    rank["rank"] = np.arange(1, len(rank)+1)
    if logger:
        logger.log(f"Univariate Cox ranking: {rank.status.eq('ok').sum()} usable features out of {len(feature_cols)}")
    return rank

def internal_cv_choose_k(train_df, ranked_features, candidate_ks=(5,10,15,20,30,50), n_splits=3, penalizer_grid=(0.01,0.05,0.1), logger=None):
    ok_feats = ranked_features.loc[ranked_features.status.eq("ok"), "feature"].tolist()
    candidate_ks = [int(k) for k in candidate_ks if int(k) <= len(ok_feats)]
    if not candidate_ks:
        candidate_ks = [min(10, len(ok_feats))]
    d = train_df.copy().reset_index(drop=True)
    rng = np.random.default_rng(RANDOM_SEED)
    idx_event = d.index[d.event.astype(int).eq(1)].to_numpy(copy=True)
    idx_none = d.index[d.event.astype(int).eq(0)].to_numpy(copy=True)
    rng.shuffle(idx_event)
    rng.shuffle(idx_none)
    folds = [[] for _ in range(n_splits)]
    for arr in [idx_event, idx_none]:
        for j, idx in enumerate(arr):
            folds[j % n_splits].append(idx)
    rows = []
    for k in candidate_ks:
        feats = ok_feats[:k]
        cvals = []
        for f in range(n_splits):
            val_idx = np.array(folds[f], dtype=int)
            tr_idx = np.setdiff1d(d.index.to_numpy(copy=True), val_idx)
            tr = d.loc[tr_idx].copy()
            va = d.loc[val_idx].copy()
            try:
                fit = fit_cox_model(tr, va, feats, penalizer_grid=penalizer_grid)
                cvals.append(float(fit["c_index"]))
            except Exception:
                cvals.append(np.nan)
        rows.append({"k": k, "mean_c_index": np.nanmean(cvals), "sd_c_index": np.nanstd(cvals), "fold_c_indices": cvals})
    cv = pd.DataFrame(rows).sort_values("k")
    # One-SE-style rule: choose smallest k within one SD of best mean to reduce overfitting.
    best_idx = cv["mean_c_index"].idxmax()
    best_mean = cv.loc[best_idx, "mean_c_index"]
    best_sd = cv.loc[best_idx, "sd_c_index"]
    threshold = best_mean - (best_sd if np.isfinite(best_sd) else 0)
    eligible = cv[cv["mean_c_index"] >= threshold].sort_values("k")
    chosen_k = int(eligible.iloc[0]["k"]) if len(eligible) else int(cv.loc[best_idx, "k"])
    cv["selection_rule"] = f"smallest k within 1 SD of best CV C-index; chosen_k={chosen_k}"
    if logger:
        logger.df("Internal development-only CV for feature count", cv)
        logger.log(f"Chosen feature count: {chosen_k}")
    return chosen_k, cv

def train_locked_model(train_df, test_df, feature_cols, prefix, outdir, modeldir, logger, variance_threshold=1e-8, max_missing=0.50, candidate_ks=(5,10,15,20,30,50), cox_rank_penalizer=0.01):
    outdir = Path(outdir); modeldir = Path(modeldir)
    outdir.mkdir(parents=True, exist_ok=True); modeldir.mkdir(parents=True, exist_ok=True)
    logger.section(f"{prefix}: feature selection and locked model training")
    vf, vt = variance_filter(train_df, feature_cols, threshold=variance_threshold, max_missing=max_missing)
    vt.to_csv(outdir / f"{prefix}_feature_variance_filter.csv", index=False)
    logger.log(f"{prefix}: initial features={len(feature_cols)}; retained after variance/missingness={len(vf)}")
    rank = univariate_cox_rank(train_df, vf, penalizer=cox_rank_penalizer, logger=logger)
    rank.to_csv(outdir / f"{prefix}_feature_univariate_cox_ranking.csv", index=False)
    chosen_k, cv = internal_cv_choose_k(train_df, rank, candidate_ks=candidate_ks, logger=logger)
    cv.to_csv(outdir / f"{prefix}_feature_count_internal_cv.csv", index=False)
    selected = rank.loc[rank.status.eq("ok"), "feature"].tolist()[:chosen_k]
    if len(selected) == 0:
        raise ValueError(f"{prefix}: no selected features.")
    pd.DataFrame({"selected_feature": selected, "rank": np.arange(1, len(selected)+1)}).to_csv(outdir / f"{prefix}_selected_features.csv", index=False)
    pd.DataFrame({"top5_feature": selected[:5]}).to_csv(outdir / f"{prefix}_top5_features.csv", index=False)
    pd.DataFrame({"top10_feature": selected[:10]}).to_csv(outdir / f"{prefix}_top10_features.csv", index=False)
    fit = fit_cox_model(train_df, test_df, selected)
    train_risk = np.log(np.asarray(fit["model"].predict_partial_hazard(fit["train_matrix"])).reshape(-1))
    test_risk = np.log(np.asarray(fit["model"].predict_partial_hazard(fit["test_matrix"])).reshape(-1))
    train_df = train_df.copy(); test_df = test_df.copy()
    train_df[f"{prefix}_score"] = train_risk
    test_df[f"{prefix}_score"] = test_risk
    thresholds = derive_thresholds(train_df[f"{prefix}_score"])
    apply_thresholds_from_dict(train_df, f"{prefix}_score", prefix, thresholds)
    apply_thresholds_from_dict(test_df, f"{prefix}_score", prefix, thresholds)
    manifest = {
        "prefix": prefix,
        "n_initial_features": len(feature_cols),
        "n_after_variance_filter": len(vf),
        "n_selected_features": len(selected),
        "selected_features": selected,
        "top5_features": selected[:5],
        "top10_features": selected[:10],
        "variance_threshold": variance_threshold,
        "max_missing": max_missing,
        "cox_rank_penalizer": cox_rank_penalizer,
        "candidate_ks": list(candidate_ks),
        "chosen_k": chosen_k,
        "feature_count_selection_rule": "development-only internal CV; smallest k within 1 SD of best mean C-index",
        "final_cox_penalizer": fit["penalizer"],
        "thresholds": thresholds,
        "model_type": "penalized Cox proportional hazards model on RetFound embedding features",
    }
    (outdir / f"{prefix}_model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    bundle = {
        "prefix": prefix,
        "model": fit["model"],
        "features": selected,
        "feature_medians": fit["median"],
        "feature_means": fit["mean"],
        "feature_sds": fit["sd"],
        "thresholds": thresholds,
        "manifest": manifest,
        "feature_ranking": rank,
        "cv_feature_count": cv,
    }
    with open(modeldir / f"{prefix}_locked_model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)
    logger.log(f"Saved locked model bundle: {modeldir / f'{prefix}_locked_model_bundle.pkl'}")
    return train_df, test_df, selected, rank, bundle, manifest, cv

def load_model_bundle(path):
    with open(Path(path), "rb") as f:
        return pickle.load(f)

def apply_model_bundle(bundle_or_path, df, score_col=None, prefix=None, logger=None):
    bundle = load_model_bundle(bundle_or_path) if not isinstance(bundle_or_path, dict) else bundle_or_path
    prefix = prefix or bundle.get("prefix", "MODEL")
    score_col = score_col or f"{prefix}_score"
    features = bundle["features"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Input dataframe missing {len(missing)} model features, e.g. {missing[:5]}")
    X = prepare_design_with_params(df, features, bundle["feature_medians"], bundle["feature_means"], bundle["feature_sds"])
    dmat = pd.concat([df[["time","event"]].reset_index(drop=True), X.reset_index(drop=True)], axis=1)
    risk = np.log(np.asarray(bundle["model"].predict_partial_hazard(dmat)).reshape(-1))
    out = df.copy()
    out[score_col] = risk
    apply_thresholds_from_dict(out, score_col, prefix, bundle.get("thresholds", derive_thresholds(pd.Series(risk))))
    if logger:
        logger.log(f"Applied locked {prefix} model to N={len(out)}; score_col={score_col}")
    return out

def derive_thresholds(score):
    s = pd.to_numeric(score, errors="coerce").dropna()
    return {"median": float(s.quantile(0.5)), "tertile_1": float(s.quantile(1/3)), "tertile_2": float(s.quantile(2/3))}


def apply_thresholds_from_dict(df, score_col, prefix, thresholds):
    print(thresholds)
    s = pd.to_numeric(df[score_col], errors="coerce")
    df[f"{prefix}_high_risk"] = (s > thresholds["median"]).astype(int)
    def tert(x):
        if pd.isna(x): return np.nan
        if x <= thresholds["tertile_1"]: return 1
        if x <= thresholds["tertile_2"]: return 2
        return 3
    df[f"{prefix}_risk_tertile"] = s.map(tert).astype("Int64")
    return df

# -----------------------------
# Metrics and plotting
# -----------------------------
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

def plot_km_high_low(df, score_col, group_col, title, save_base):
    d = df[["time","event",score_col,group_col]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    if len(d) == 0 or d[group_col].nunique() < 2:
        print(f"Skipping KM {title}: insufficient groups")
        return
    met = performance_row(d, title, score_col, group_col)
    fig, ax = plt.subplots(figsize=(8.2,5.6))
    colors = {0: "tab:blue", 1: "tab:red"}
    for val, lab in [(0,"Low risk"), (1,"High risk")]:
        sub = d[d[group_col].eq(val)]
        kmf = KaplanMeierFitter()
        kmf.fit(sub.time/365.25, sub.event, label=f"{lab}: N={len(sub)}, events={int(sub.event.sum())}")
        kmf.plot_survival_function(ax=ax, ci_show=True, color=colors[val])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time from retinal imaging (years)")
    ax.set_ylabel("MACE-free survival probability")
    ax.grid(alpha=0.25)
    txt = f"C-index: {met['C_index_95CI']}\nHR: {met['HR_95CI']}\nLog-rank p: {pformat(met.get('LogRank_p'))}"
    ax.text(0.02, 0.05, txt, transform=ax.transAxes, bbox=dict(facecolor="white", edgecolor="0.3", alpha=0.9))
    fig.tight_layout()
    savefig(fig, save_base)
    plt.show()

def plot_km_tertiles(df, score_col, tertile_col, title, save_base):
    d = df[["time","event",score_col,tertile_col]].replace([np.inf,-np.inf], np.nan).dropna().copy()
    if len(d) == 0 or d[tertile_col].nunique() < 2:
        print(f"Skipping tertile KM {title}: insufficient groups")
        return
    fig, ax = plt.subplots(figsize=(8.2,5.6))
    for val in sorted(d[tertile_col].dropna().unique()):
        sub = d[d[tertile_col].eq(val)]
        kmf = KaplanMeierFitter()
        kmf.fit(sub.time/365.25, sub.event, label=f"T{int(val)}: N={len(sub)}, events={int(sub.event.sum())}")
        kmf.plot_survival_function(ax=ax, ci_show=True)
    try:
        lr = multivariate_logrank_test(d.time/365.25, d[tertile_col], d.event)
        ptxt = pformat(lr.p_value)
    except Exception:
        ptxt = "NA"
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Time from retinal imaging (years)")
    ax.set_ylabel("MACE-free survival probability")
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.05, f"Tertile log-rank p: {ptxt}", transform=ax.transAxes, bbox=dict(facecolor="white", edgecolor="0.3", alpha=0.9))
    fig.tight_layout()
    savefig(fig, save_base)
    plt.show()


def forest_plot(df, label_col, hr_col, lo_col, hi_col, p_col, n_col, event_col, title, save_base):
    d = df.dropna(subset=[hr_col, lo_col, hi_col]).copy()
    if d.empty:
        print(f"Skipping forest {title}: empty")
        return
    d = d.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(d))
    fig, (ax_text, ax) = plt.subplots(1,2, figsize=(8, max(4, 0.35*len(d)+1)), gridspec_kw={"width_ratios":[1.4,1.3]})
    ax_text.axis("off")
    ax_text.set_ylim(-1, len(d))
    ax.set_ylim(-1, len(d))
    ax_text.text(0.00, len(d)-0.2, "Group", fontweight="bold")
    ax_text.text(0.45, len(d)-0.2, "N/events", fontweight="bold")
    ax_text.text(0.78, len(d)-0.2, "p", fontweight="bold")
    for i,r in d.iterrows():
        ax_text.text(0.00, i, str(r[label_col]), va="center", fontsize=8.5)
        ax_text.text(0.45, i, f"{int(r[n_col])}/{int(r[event_col])}", va="center", fontsize=8.5)
        ax_text.text(0.78, i, pformat(r[p_col]), va="center", fontsize=8.5)
    ax.errorbar(d[hr_col], y, xerr=[d[hr_col]-d[lo_col], d[hi_col]-d[hr_col]], fmt="s", capsize=2, color="black") #, markersize=5)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    # ax.set_xscale("log")
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.set_yticks([])
    ax.grid(axis="x", alpha=0.25)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    savefig(fig, save_base)
    plt.show()
    

# def forest_plot(df, label_col, hr_col, lo_col, hi_col, p_col, n_col, event_col, title, save_base):
#     d = df.dropna(subset=[hr_col, lo_col, hi_col]).copy()
#     if d.empty:
#         print(f"Skipping forest {title}: empty")
#         return
#     # Try to use the `forestplot` package if available; otherwise fallback to matplotlib
#     import forestplot as fp

#     # Prepare a dataframe compatible with fp.forestplot
#     d2 = d.copy()
#     # Ensure numeric columns
#     for c in [hr_col, lo_col, hi_col]:
#         d2[c] = pd.to_numeric(d2[c], errors="coerce")

#     # Add simple annotation columns for left (N, Events) and ensure p-val column exists
#     if n_col not in d2.columns:
#         d2[n_col] = np.nan
#     if event_col not in d2.columns:
#         d2[event_col] = np.nan
#     if p_col not in d2.columns:
#         d2[p_col] = np.nan

#     # Use fp to draw the forest plot. Provide annote columns for N and Events and p-value on right.
#     fp.forestplot(
#         d2,
#         estimate=hr_col,
#         ll=lo_col,
#         hl=hi_col,
#         varlabel=label_col,
#         annote=[n_col, event_col],
#         annoteheaders=["N", "Events"],
#         pval=p_col,
#         xlabel="Hazard ratio (95% CI)",
#         ylabel="",
#     )
#     fig = plt.gcf()
#     savefig(fig, save_base)
#     plt.show()



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

# -----------------------------
# Clinical covariates and models
# -----------------------------
# ============================================================
# Updated one-age clinical covariate policy with HTN/diabetes CSV labels
# ============================================================

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


# Make existing notebook calls use this updated function.
one_age_covars = one_age_covars_htn_diabetes


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

# -----------------------------
# Reclassification and alluvial plots
# -----------------------------
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



def alluvial_static(df, left, right, title, save_base):
    d = df[[left,right]].dropna().astype(str)
    flows = d.groupby([left,right]).size().reset_index(name="count")
    left_levels = sorted(d[left].unique(), key=str)
    right_levels = sorted(d[right].unique(), key=str)
    lpos = {k:i for i,k in enumerate(left_levels)}
    rpos = {k:i for i,k in enumerate(right_levels)}
    fig, ax = plt.subplots(figsize=(8,5.5))
    max_count = max(flows["count"].max(), 1)
    for _,r in flows.iterrows():
        y0 = lpos[r[left]]
        y1 = rpos[r[right]]
        lw = 1 + 9*r["count"]/max_count
        ax.plot([0,1],[y0,y1], alpha=0.35, linewidth=lw)
        ax.text(0.5, (y0+y1)/2, str(int(r["count"])), fontsize=7, alpha=0.7)
    ax.scatter([0]*len(left_levels), list(lpos.values()), s=80)
    ax.scatter([1]*len(right_levels), list(rpos.values()), s=80)
    for k,v in lpos.items():
        ax.text(-0.03, v, f"{left}: {k}", ha="right", va="center")
    for k,v in rpos.items():
        ax.text(1.03, v, f"{right}: {k}", ha="left", va="center")
    ax.set_xlim(-0.35,1.35); ax.set_ylim(-0.5, max(len(left_levels),len(right_levels))-0.5)
    ax.set_xticks([0,1]); ax.set_xticklabels([left,right])
    ax.set_yticks([]); ax.set_title(title, fontweight="bold")
    ax.spines[["top","right","left"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, save_base)
    plt.show()

# -----------------------------
# Subgroup, treatment, and CMR
# -----------------------------
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

def subgroup_analysis(df, subgroup_cols, score_col, group_col, figdir, tabledir, min_n=50, min_events=5, prefix="subgroup"):
    rows = []
    km_dir = Path(figdir) / f"{prefix}_KM"
    km_dir.mkdir(parents=True, exist_ok=True)
    for c in subgroup_cols:
        sub = df[pd.to_numeric(df[c], errors="coerce").fillna(0).eq(1)].copy()
        label = c.replace("_present","")
        row = {"subgroup": label, "column": c, "N": len(sub), "Events": int(sub.event.sum()) if len(sub) else 0}
        if len(sub) >= min_n and int(sub.event.sum()) >= min_events and sub[group_col].nunique(dropna=True) >= 2:
            perf = performance_row(sub, label, score_col, group_col)
            row.update(perf)
            row["status"] = "analyzable"
            plot_km_high_low(sub, score_col, group_col, f"{label}: CORI high vs low", km_dir / f"{safe_name(label)}_KM_high_low")
        else:
            row.update({"C_index":np.nan,"C_index_low":np.nan,"C_index_high":np.nan,"HR_High_vs_Low":np.nan,"HR_CI_Low":np.nan,"HR_CI_High":np.nan,"LogRank_p":np.nan})
            row["status"] = "underpowered"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(Path(tabledir)/f"{prefix}_performance.csv", index=False)
    if len(out) and out["HR_High_vs_Low"].notna().sum() > 0:
        fp = out.dropna(subset=["HR_High_vs_Low"]).copy()
        fp["label"] = fp["subgroup"]
        forest_plot(fp, "label", "HR_High_vs_Low", "HR_CI_Low", "HR_CI_High", "LogRank_p", "N", "Events", "Subtype/subgroup analysis: pan-cancer CORI model", Path(figdir)/f"{prefix}_forest")
    return out

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

# -----------------------------
# Feature-rank shift visualization
# -----------------------------
def feature_rank_shift_bump(
    cori_rank,
    mmace_rank,
    figdir,
    tabledir,
    top_n=30,
    prefix="H2",
    max_labels=30,
):
    """
    Publication-ready feature-rank transition plot.

    Purpose
    -------
    Compare how RetFound embedding features rank in:
        1. cancer-trained CORI
        2. non-cancer-trained M-MACE

    Key design choices
    ------------------
    - Uses only features that are top_n in either model.
    - Y-axis is capped at top_n + 1.
    - Features outside the top_n in the other model are shown at '>top_n'
      rather than stretching the y-axis to rank 200, 500, or 1000.
    - Rank 1 is shown at the top.
    - Blue line: feature becomes more important in M-MACE than CORI.
    - Red line: feature becomes less important in M-MACE than CORI.
    - Gray line: rank is similar.
    - Dot size reflects Cox-ranking importance, using -log10(p).
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    figdir = Path(figdir)
    tabledir = Path(tabledir)
    figdir.mkdir(parents=True, exist_ok=True)
    tabledir.mkdir(parents=True, exist_ok=True)

    def _clean_rank_table(rank_df, model_label):
        r = rank_df.copy()

        # Keep only usable rows when status exists
        if "status" in r.columns:
            r = r[r["status"].astype(str).eq("ok")].copy()

        if "feature" not in r.columns:
            raise ValueError(f"{model_label}: rank table must contain a 'feature' column.")

        # Ensure numeric p, z, HR, importance
        for col in ["p", "z_abs", "hr_per_sd", "importance", "coef"]:
            if col in r.columns:
                r[col] = pd.to_numeric(r[col], errors="coerce")

        # If importance is absent, use -log10(p)
        if "importance" not in r.columns:
            r["importance"] = np.nan
        if "p" in r.columns:
            r["importance"] = r["importance"].fillna(
                -np.log10(r["p"].clip(lower=1e-300))
            )

        # Sort by p first, then z_abs if available
        if "p" in r.columns and "z_abs" in r.columns:
            r = r.sort_values(["p", "z_abs"], ascending=[True, False]).copy()
        elif "p" in r.columns:
            r = r.sort_values("p", ascending=True).copy()
        elif "importance" in r.columns:
            r = r.sort_values("importance", ascending=False).copy()
        else:
            r = r.reset_index(drop=True).copy()

        r = r.drop_duplicates("feature", keep="first").reset_index(drop=True)

        # Recompute rank after sorting to avoid stale rank columns
        r[f"{model_label}_rank"] = np.arange(1, len(r) + 1)

        # Direction from HR or coefficient
        if "hr_per_sd" in r.columns:
            r[f"{model_label}_hr"] = r["hr_per_sd"]
            r[f"{model_label}_direction"] = np.where(
                r["hr_per_sd"] >= 1, "risk-increasing", "risk-decreasing"
            )
        elif "coef" in r.columns:
            r[f"{model_label}_hr"] = np.exp(r["coef"])
            r[f"{model_label}_direction"] = np.where(
                r["coef"] >= 0, "risk-increasing", "risk-decreasing"
            )
        else:
            r[f"{model_label}_hr"] = np.nan
            r[f"{model_label}_direction"] = "unknown"

        r[f"{model_label}_importance"] = r["importance"]

        keep = [
            "feature",
            f"{model_label}_rank",
            f"{model_label}_importance",
            f"{model_label}_hr",
            f"{model_label}_direction",
        ]
        return r[keep]

    # Clean and rank both tables
    cori = _clean_rank_table(cori_rank, "CORI")
    mmace = _clean_rank_table(mmace_rank, "MMACE")

    # Merge full rankings
    merged = cori.merge(mmace, on="feature", how="outer")

    # Select top union: top_n in either model
    selected = merged[
        (merged["CORI_rank"].le(top_n)) | (merged["MMACE_rank"].le(top_n))
    ].copy()

    if selected.empty:
        print("No features available for feature-rank transition plot.")
        return selected

    # Sort features by best rank across the two models
    selected["best_rank"] = selected[["CORI_rank", "MMACE_rank"]].min(axis=1)
    selected = selected.sort_values(["best_rank", "CORI_rank", "MMACE_rank"]).copy()

    # Limit number of drawn labels if requested
    if max_labels is not None:
        selected = selected.head(max_labels).copy()

    capped_rank = top_n + 1

    # Plot ranks: anything outside top_n goes to a single '>top_n' row
    selected["CORI_plot_rank"] = selected["CORI_rank"].where(
        selected["CORI_rank"].le(top_n), capped_rank
    )
    selected["MMACE_plot_rank"] = selected["MMACE_rank"].where(
        selected["MMACE_rank"].le(top_n), capped_rank
    )

    selected["CORI_plot_rank"] = selected["CORI_plot_rank"].fillna(capped_rank)
    selected["MMACE_plot_rank"] = selected["MMACE_plot_rank"].fillna(capped_rank)

    selected["rank_shift_MMACE_minus_CORI"] = (
        selected["MMACE_plot_rank"] - selected["CORI_plot_rank"]
    )

    def _membership(row):
        in_cori = pd.notna(row["CORI_rank"]) and row["CORI_rank"] <= top_n
        in_mmace = pd.notna(row["MMACE_rank"]) and row["MMACE_rank"] <= top_n
        if in_cori and in_mmace:
            return "Top in both"
        if in_cori and not in_mmace:
            return "CORI-specific top feature"
        if in_mmace and not in_cori:
            return "M-MACE-specific top feature"
        return "Outside top set"

    selected["feature_membership"] = selected.apply(_membership, axis=1)

    def _line_color(row):
        # Blue: stronger rank in M-MACE; Red: stronger rank in CORI; Gray: similar
        if row["feature_membership"] == "CORI-specific top feature":
            return "tab:red"
        if row["feature_membership"] == "M-MACE-specific top feature":
            return "tab:blue"
        shift = row["rank_shift_MMACE_minus_CORI"]
        if shift <= -2:
            return "tab:blue"
        if shift >= 2:
            return "tab:red"
        return "0.45"

    def _dot_size(x):
        if pd.isna(x) or not np.isfinite(x):
            return 30
        return float(np.clip(28 + 16 * x, 30, 180))

    # Save table before plotting
    selected.to_csv(
        tabledir / f"{prefix}_feature_rank_transition_CORI_vs_MMACE.csv",
        index=False,
    )

    # Figure height scales with number of features but is capped
    fig_h = min(max(7.0, 0.30 * len(selected) + 2.5), 13.5)
    fig, ax = plt.subplots(figsize=(9.5, fig_h))

    # Draw lines
    for _, row in selected.iterrows():
        color = _line_color(row)
        y0 = row["CORI_plot_rank"]
        y1 = row["MMACE_plot_rank"]

        # Thicker if larger movement
        lw = 1.2 + min(abs(row["rank_shift_MMACE_minus_CORI"]) / 4.0, 3.0)

        ax.plot(
            [0, 1],
            [y0, y1],
            color=color,
            alpha=0.68,
            linewidth=lw,
            zorder=1,
        )

        ax.scatter(
            0,
            y0,
            s=_dot_size(row.get("CORI_importance", np.nan)),
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.scatter(
            1,
            y1,
            s=_dot_size(row.get("MMACE_importance", np.nan)),
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )

    # Label each feature at both sides, but show capped ranks as >top_n
    def _rank_label(rank_val):
        if pd.isna(rank_val):
            return f">{top_n}"
        if rank_val > top_n:
            return f">{top_n}"
        return str(int(rank_val))

    for _, row in selected.iterrows():
        left_rank = _rank_label(row["CORI_rank"])
        right_rank = _rank_label(row["MMACE_rank"])

        ax.text(
            -0.035,
            row["CORI_plot_rank"],
            f"{row['feature']} ({left_rank})",
            ha="right",
            va="center",
            fontsize=7.4,
        )
        ax.text(
            1.035,
            row["MMACE_plot_rank"],
            f"{row['feature']} ({right_rank})",
            ha="left",
            va="center",
            fontsize=7.4,
        )

    # Axis formatting
    ax.set_xlim(-0.46, 1.46)
    ax.set_ylim(capped_rank + 0.8, 0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Cancer-trained CORI\nfeature rank", "Non-cancer M-MACE\nfeature rank"],
        fontsize=10,
    )

    yticks = list(range(1, top_n + 1))
    if top_n > 20:
        yticks = list(range(1, top_n + 1, 2))
    yticks = yticks + [capped_rank]
    yticklabels = [str(y) for y in yticks[:-1]] + [f">{top_n}"]

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.set_ylabel("Feature rank; 1 = most prognostic by univariate Cox", fontsize=10)
    ax.set_title(
        "Feature-rank transition: cancer-trained CORI vs non-cancer M-MACE",
        fontweight="bold",
        fontsize=12,
    )

    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    # Add explanatory legend manually
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color="tab:red", lw=2.5, label="Higher rank in CORI / drops in M-MACE"),
        Line2D([0], [0], color="tab:blue", lw=2.5, label="Higher rank in M-MACE / rises in M-MACE"),
        Line2D([0], [0], color="0.45", lw=2.5, label="Similar top-rank feature"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    savefig(fig, figdir / f"{prefix}_feature_rank_transition_bump_FIXED")
    plt.show()

    # Also save a compact top-table for manuscript inspection
    compact_cols = [
        "feature",
        "feature_membership",
        "CORI_rank",
        "MMACE_rank",
        "rank_shift_MMACE_minus_CORI",
        "CORI_importance",
        "MMACE_importance",
        "CORI_hr",
        "MMACE_hr",
        "CORI_direction",
        "MMACE_direction",
    ]
    compact_cols = [c for c in compact_cols if c in selected.columns]
    selected[compact_cols].to_csv(
        tabledir / f"{prefix}_feature_rank_transition_compact_CORI_vs_MMACE.csv",
        index=False,
    )

    return selected