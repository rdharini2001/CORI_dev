from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .common import RANDOM_SEED
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

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
