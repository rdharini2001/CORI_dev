#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_coxnet_by_center_fixed.py

Leakage-safe baseline survival modeling (center-based split) + safer plotting.

Fixes vs your previous baseline script:
1) Deterministic baseline row selection (date tie-breaker + stable fallback tie-breakers).
2) Forestplot safety: skips forestplot if HR/CI are non-finite (prevents NaN/Inf crashes).
3) Feature discovery is more robust:
   - Uses your --feat_regex if it matches columns.
   - If it matches nothing, falls back to a generic embedding pattern ^emb\\d+_f\\d+$ with a warning.

Still does:
- HARD filter to baseline visit BEFORE selection/cleaning.
- Canonicalize endpoint columns to TIME_COL/EVENT_COL.
- Train/test split by baseline centers.
- Coxnet training on baseline only.
- Saves: risk_scores_baseline_<tag>.csv, KM plots, metrics, subtype analyses, forestplot (if valid), model_bundle.pkl

Example:
python baseline_coxnet_by_center_fixed.py \
  --csv retonco_merged_allcancer.csv \
  --outdir outputs/allcancer \
  --train_centers "Hounslow,Bury" \
  --visit_col image_visit_subid --baseline_visit 0 \
  --feat_regex "^emb2_f\\d+$" \
  --time_col allcancer_10yr_censored_time --event_col allcancer_10yr_censored_status
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Dict, Sequence, Tuple, List, Optional

import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv

from src.constants import TIME_COL, EVENT_COL  # expected canonical names, e.g. "time", "event"
from src import (
    fit_transform_train,
    transform_test,
    select_features,
    compute_group_metrics,
    plot_km_single,
    plot_km_grid,
)
from src.forestplot_utils import make_forestplot


CENTER_COL_DEFAULT = "UK Biobank assessment centre | Instance 0"
BASELINE_DATE_COL = "scan_date"

SUBTYPE_COLS = [
    "DigestiveCancer_present", "RespiCancer_present", "BreastCancer_present",
    "FemRepoCancer_present", "UrinaryTractCancer_present", "HeamatoCancer_present",
    "Digestive_01_Colorectal_present", "Digestive_02_Liver_gall_present",
    "Digestive_03_Pancreas_present", "Digestive_04_others_present",
    "Respi_01_Lung_present", "Respi_02_others_present",
    "FemRepo_01_ovary_present", "FemRepo_02_uterus_present", "FemRepo_03_others_present",
    "MaleRepo_01_prostate_present", "MaleRepo_02_others_present",
    "UrinaryTract_01_kidney_present", "UrinaryTract_02_bladder_present", "UrinaryTract_03_others_present",
    "MaleRepoCancer_present", "EndocrineCancer_present", "InsituCancer_present",
    "BenignNeoplasm_present", "LipOralCancer_present", "BoneCancer_present",
    "SkinCancer_present", "MesotheliumCancer_present", "EyeCNSCancer_present",
    "SecondaryCancer_present", "UnknownCancer_present", "Skin_01_melanoma_present",
    "Skin_02_nonmelanoma_present", "EyeCNS_01_Brain_present", "EyeCNS_02_Eye_present",
    "EyeCNS_03_OtherCNS_present", "BenignNeoplasm_01_colon_present",
    "BenignNeoplasm_02_digestive_present", "BenignNeoplasm_03_uterus_present",
    "BenignNeoplasm_04_others_present",
]


def _rename_cancer(col: str) -> str:
    return col.replace("_present", "").replace("_", " ")


def _fmt_p(p):
    if pd.isna(p):
        return ""
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def _infer_tag_from_endpoint_cols(time_col_raw: str, event_col_raw: str) -> str:
    s = f"{time_col_raw} {event_col_raw}".lower()
    m = re.search(r"(\b\d+\s*yr\b|\b\d+\s*y\b|\b\d+yr\b|\b\d+y\b)", s)
    if m:
        return re.sub(r"\s+", "", m.group(1)).replace("yr", "y")
    m = re.search(r"\b(\d+)\s*[-_ ]?\s*(year|years)\b", s)
    if m:
        return f"{m.group(1)}y"
    return "endpoint"


def minimal_survival_clean(df: pd.DataFrame, *, time_col: str, event_col: str, center_col: str) -> pd.DataFrame:
    """
    Minimal survival cleaning (baseline rows only):
    - coerce eid/time/event numeric
    - event -> {0,1}
    - require eid + center not null
    - require time > 0
    """
    df = df.copy()

    if "eid" not in df.columns:
        raise ValueError("Input CSV must contain 'eid'.")
    for c in [center_col, time_col, event_col]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["eid"] = pd.to_numeric(df["eid"], errors="coerce").astype("Int64")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[event_col] = pd.to_numeric(df[event_col], errors="coerce")

    df[event_col] = (df[event_col].fillna(0) > 0).astype(int)

    df = df.dropna(subset=["eid", center_col]).copy()
    df = df[df[time_col].notna() & (df[time_col] > 0)].copy()
    return df


def _stable_sort_cols_for_baseline(df: pd.DataFrame) -> List[str]:
    """
    Deterministic tie-breakers to avoid 'CSV-order randomness' when multiple baseline images exist.
    Priority:
      1) baseline assessment date (Instance 0) if present
      2) scan_date if present
      3) image_visit_subid / image_visit if present
      4) image_basename if present
      5) image_fullpath if present
    Always includes 'eid' first.
    """
    cols = ["eid"]
    if BASELINE_DATE_COL in df.columns:
        cols.append("_baseline_dt")
    if "scan_date" in df.columns:
        cols.append("_scan_dt")
    for c in ["image_visit_subid", "image_visit"]:
        if c in df.columns:
            cols.append(c)
    for c in ["image_basename", "image_fullpath"]:
        if c in df.columns:
            cols.append(c)
    return cols


def select_baseline_one_row_per_eid(df: pd.DataFrame, visit_col: str, baseline_visit: int) -> pd.DataFrame:
    """
    df is already baseline-filtered, but we keep it robust:
    - re-filter by visit_col
    - sort by deterministic tie-breakers
    - take first per eid
    """
    d = df.copy()
    d[visit_col] = pd.to_numeric(d[visit_col], errors="coerce").fillna(-999).astype(int)
    d = d[d[visit_col] == int(baseline_visit)].copy()

    if d.empty:
        return d

    # Build sortable helper dates
    if BASELINE_DATE_COL in d.columns:
        d["_baseline_dt"] = pd.to_datetime(d[BASELINE_DATE_COL], errors="coerce")
    else:
        d["_baseline_dt"] = pd.NaT

    if "scan_date" in d.columns:
        d["_scan_dt"] = pd.to_datetime(d["scan_date"], errors="coerce")
    else:
        d["_scan_dt"] = pd.NaT

    sort_cols = _stable_sort_cols_for_baseline(d)
    # ensure missing helper cols exist
    for c in sort_cols:
        if c not in d.columns:
            d[c] = np.nan

    d = d.sort_values(sort_cols, na_position="last")
    d = d.drop_duplicates("eid", keep="first")

    # drop helper cols
    d = d.drop(columns=[c for c in ["_baseline_dt", "_scan_dt"] if c in d.columns], errors="ignore")
    return d

def _discover_feature_columns(df: pd.DataFrame, feat_regex: str) -> Tuple[List[str], str]:
    """
    Returns (feature_columns, regex_used).
    No fallback: if feat_regex matches none, raise.
    """
    cols = [c for c in df.columns if re.match(feat_regex, str(c))]
    if cols:
        return cols, feat_regex

    raise ValueError(
        f"No feature columns matched --feat_regex='{feat_regex}'. "
        "Tip: try --feat_regex '^(artery|vein|vessel)_' or '^(artery|vein|vessel)'."
    )

def _finite_rows_for_forestplot(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows with finite positive HR and CI.
    Prevents forestplot crashes due to NaN/Inf axis limits.
    """
    req = ["HR_High_vs_Low", "HR_CI_Low", "HR_CI_High"]
    for c in req:
        if c not in metrics_df.columns:
            return metrics_df.iloc[0:0].copy()

    d = metrics_df.copy()
    for c in req:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=req)
    d = d[(d["HR_High_vs_Low"] > 0) & (d["HR_CI_Low"] > 0) & (d["HR_CI_High"] > 0)]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument("--train_centers", type=str, required=True, help="Comma-separated center names for training")
    ap.add_argument("--center_col", type=str, default=CENTER_COL_DEFAULT)

    # Baseline visit filter (CRITICAL)
    ap.add_argument("--visit_col", type=str, default="image_visit")
    ap.add_argument("--baseline_visit", type=int, default=0)

    # Features + endpoint columns (raw names in your CSV)
    
    ap.add_argument("--feat_regex", type=str, default=r"^(artery|vein|vessel)_")

    ap.add_argument("--time_col", type=str, required=True)
    ap.add_argument("--event_col", type=str, required=True)

    # Model options
    ap.add_argument("--l1_ratio", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)

    # Optional filename tag override
    ap.add_argument("--tag", type=str, default=None, help="Optional tag for output filenames (e.g., 10y, 5y, 3y).")

    # Plot safety
    ap.add_argument("--min_events_for_stats", type=int, default=3)

    args = ap.parse_args()

    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_centers = [x.strip() for x in args.train_centers.split(",") if x.strip()]
    if not train_centers:
        raise ValueError("train_centers is empty after parsing. Provide comma-separated center names.")

    tag = args.tag if args.tag else _infer_tag_from_endpoint_cols(args.time_col, args.event_col)

    df_all = pd.read_csv(args.csv, low_memory=False)

    # -----------------------------
    # (A) HARD filter to baseline visit FIRST
    # -----------------------------
    if args.visit_col not in df_all.columns:
        raise ValueError(
            f"Missing visit_col='{args.visit_col}'. "
            "To guarantee baseline-only, you must provide a visit column (e.g., image_visit_subid)."
        )

    df_all[args.visit_col] = pd.to_numeric(df_all[args.visit_col], errors="coerce").fillna(-999).astype(int)
    df_all = df_all[df_all[args.visit_col] == int(args.baseline_visit)].copy()

    # -----------------------------
    # (B) Clean endpoints on baseline visit only
    # -----------------------------
    df_all = minimal_survival_clean(
        df_all,
        time_col=args.time_col,
        event_col=args.event_col,
        center_col=args.center_col,
    )

    # Canonicalize endpoint columns for ALL downstream code
    df_all = df_all.rename(columns={args.time_col: TIME_COL, args.event_col: EVENT_COL})

    # -----------------------------
    # (C) Baseline selection: one row per eid (deterministic)
    # -----------------------------
    df_base = select_baseline_one_row_per_eid(df_all, args.visit_col, args.baseline_visit)

    # Strict assertions: baseline visit only + 1 row per eid
    assert (df_base[args.visit_col].astype(int) == int(args.baseline_visit)).all(), \
        "df_base contains non-baseline visits — baseline filtering is not working."
    assert df_base["eid"].nunique() == len(df_base), \
        "df_base is not 1 row per eid after baseline selection."

    # subtype columns present
    subtype_cols_present = [c for c in SUBTYPE_COLS if c in df_base.columns]
    for c in subtype_cols_present:
        df_base[c] = pd.to_numeric(df_base[c], errors="coerce").fillna(0).astype(int)

    # -----------------------------
    # (D) Split based on baseline only (center-based)
    # -----------------------------
    train_mask = df_base[args.center_col].isin(train_centers)
    train_df = df_base[train_mask].copy()
    test_df = df_base[~train_mask].copy()

    # Guarantees: 1 row per eid per split
    if train_df["eid"].nunique() != len(train_df):
        raise ValueError("Train baseline table is not 1 row per eid. Fix baseline selection.")
    if test_df["eid"].nunique() != len(test_df):
        raise ValueError("Test baseline table is not 1 row per eid. Fix baseline selection.")

    print(f"[{tag}] BASELINE TRAIN N:", train_df["eid"].nunique(), "| BASELINE TEST N:", test_df["eid"].nunique())

    # -----------------------------
    # (E) Feature discovery + preprocessing
    # -----------------------------
    feat_cols_all, regex_used = _discover_feature_columns(df_base, args.feat_regex)
    if regex_used != args.feat_regex:
        print(f"[INFO] Using features discovered by regex: {regex_used}")
    print(f"[INFO] #features matched: {len(feat_cols_all)}")

    train_df[feat_cols_all] = train_df[feat_cols_all].apply(pd.to_numeric, errors="coerce")
    test_df[feat_cols_all] = test_df[feat_cols_all].apply(pd.to_numeric, errors="coerce")

    train_df, art = fit_transform_train(train_df, feat_cols_all)
    test_df = transform_test(test_df, art)

    # feature selection on train only
    feat_cols = select_features(
        train_df,
        art.valid_features,
        seed=args.seed,
        event_col=EVENT_COL,
    )
    if not feat_cols:
        raise ValueError("No features selected. Check missingness/variance thresholds inside select_features().")

    # -----------------------------
    # (F) Fit Coxnet (baseline only)
    # -----------------------------
    train_df[EVENT_COL] = train_df[EVENT_COL].astype(bool)
    test_df[EVENT_COL] = test_df[EVENT_COL].astype(bool)

    y_train = Surv.from_dataframe(EVENT_COL, TIME_COL, train_df)
    X_train = train_df[feat_cols].values

    model = CoxnetSurvivalAnalysis(l1_ratio=args.l1_ratio, alpha_min_ratio=0.01, n_alphas=100)
    model.fit(X_train, y_train)

    train_risk = model.predict(X_train)
    test_risk = model.predict(test_df[feat_cols].values)

    thr = float(np.median(train_risk))
    train_df = train_df.assign(risk_score=train_risk, high_risk=(train_risk >= thr).astype(int))
    test_df = test_df.assign(risk_score=test_risk, high_risk=(test_risk >= thr).astype(int))

    # -----------------------------
    # (G) Save baseline risk scores (canonical + raw endpoint columns for convenience)
    # -----------------------------
    train_out = train_df.assign(**{args.time_col: train_df[TIME_COL], args.event_col: train_df[EVENT_COL].astype(int)})
    test_out = test_df.assign(**{args.time_col: test_df[TIME_COL], args.event_col: test_df[EVENT_COL].astype(int)})

    risk_path = outdir / f"risk_scores_baseline_{tag}.csv"
    risk_df = pd.concat(
        [
            train_out[["eid", args.center_col, args.visit_col, TIME_COL, EVENT_COL, args.time_col, args.event_col, "risk_score", "high_risk"]]
            .assign(set="train"),
            test_out[["eid", args.center_col, args.visit_col, TIME_COL, EVENT_COL, args.time_col, args.event_col, "risk_score", "high_risk"]]
            .assign(set="test"),
        ],
        ignore_index=True,
    )
    risk_df.to_csv(risk_path, index=False)

    # Hard count check: written must match baseline (1 row per eid)
    expected_unique = train_df["eid"].nunique() + test_df["eid"].nunique()
    written_unique = risk_df["eid"].nunique()
    assert written_unique == expected_unique, (
        f"Risk file unique eids ({written_unique}) != expected baseline eids ({expected_unique})"
    )

    # -----------------------------
    # (H) Baseline KM + metrics (with minimal safety)
    # -----------------------------
    rows = []
    for split_name, split_df in [("train", train_df), ("test", test_df)]:
        # safety: need 2 groups and enough events for meaningful stats/plots
        n_events = int(split_df[EVENT_COL].astype(int).sum())
        if split_df["high_risk"].nunique() < 2 or n_events < int(args.min_events_for_stats):
            print(f"[WARN] {tag} {split_name}: skip KM/metrics (groups={split_df['high_risk'].nunique()}, events={n_events}).")
            continue

        m = compute_group_metrics(split_df, "high_risk")
        pd.DataFrame([m]).to_csv(outdir / f"{split_name}_metrics_baseline_{tag}.csv", index=False)

        plot_km_single(
            split_df,
            "high_risk",
            {0: "Low risk", 1: "High risk"},
            f"Baseline (visit={args.baseline_visit}) — {tag} — {split_name}",
            m,
            show_risk_table=True,
            save_path=outdir / f"KM_baseline_{tag}_{split_name}.png",
        )

        if split_name == "test":
            rows.append({**m, "Cancer_Type": "PanCancer", "Display_Label": "PanCancer"})

    # -----------------------------
    # (I) Subtype analysis (TEST only)
    # -----------------------------
    if subtype_cols_present:
        test_base = test_df.copy()

        # controls = cancer-free (all subtype flags 0)
        ctrl_mask = (test_base[subtype_cols_present].fillna(0).max(axis=1) == 0)
        controls = test_base.loc[ctrl_mask].copy()

        grid_entries = [("PanCancer", test_base)]
        for cancer_col in subtype_cols_present:
            disp = _rename_cancer(cancer_col)
            suffix = cancer_col.replace("_present", "")
            cases = test_base.loc[test_base[cancer_col].fillna(0).astype(int) == 1].copy()
            temp = pd.concat([cases, controls], ignore_index=True)

            n_events = int(temp[EVENT_COL].astype(int).sum())
            if temp["high_risk"].nunique() < 2 or n_events < int(args.min_events_for_stats) or len(temp) < 30:
                continue

            m = compute_group_metrics(temp, "high_risk")
            pd.DataFrame([m]).to_csv(outdir / f"test_metrics_baseline_{tag}_{suffix}.csv", index=False)
            grid_entries.append((disp, temp))
            rows.append({**m, "Cancer_Type": suffix, "Display_Label": disp})

        if grid_entries and len(grid_entries) > 1:
            plot_km_grid(grid_entries, outdir / f"KM_baseline_{tag}_pan_and_subtypes.png", ncols=3)

    # -----------------------------
    # (J) Forestplot (test summary) — SAFE
    # -----------------------------
    if rows:
        metrics_df = pd.DataFrame(rows)
        metrics_df.to_csv(outdir / f"all_test_metrics_baseline_{tag}.csv", index=False)

        finite_metrics = _finite_rows_for_forestplot(metrics_df)
        if finite_metrics.empty:
            print(f"[WARN] {tag}: No finite HR/CI rows for forestplot; skipping forestplot.")
        else:
            plot_df = finite_metrics.rename(columns={"HR_High_vs_Low": "r", "HR_CI_Low": "ll", "HR_CI_High": "hl"}).copy()
            plot_df["label"] = plot_df["Display_Label"]
            plot_df["group"] = "Cancer"
            plot_df["HR_95CI"] = plot_df.apply(lambda x: f"{x['r']:.2f} ({x['ll']:.2f}–{x['hl']:.2f})", axis=1)
            plot_df["C_index_fmt"] = plot_df["C_index_cont"].where(
                plot_df["C_index_cont"].notna(), plot_df["C_index"]
            ).map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
            plot_df["LogRank_p_fmt"] = plot_df["LogRank_p"].map(_fmt_p)
            plot_df["Events_fmt"] = plot_df.apply(lambda x: f"{int(x['Test_Events'])}/{int(x['Test_N'])}", axis=1)

            # final axis safety
            plot_df = plot_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["r", "ll", "hl"])
            if plot_df.empty:
                print(f"[WARN] {tag}: Forestplot DF empty after filtering; skipping forestplot.")
            else:
                make_forestplot(plot_df, outdir / f"forestplot_test_metrics_baseline_{tag}.png")

    # -----------------------------
    # (K) Save bundle
    # -----------------------------
    bundle = {
        "model": model,
        "art": art,
        "feat_cols": feat_cols,
        "thr": float(thr),
        "train_centers": list(train_centers),
        "center_col": args.center_col,
        "visit_col": args.visit_col,
        "baseline_visit": int(args.baseline_visit),
        "feat_regex": regex_used,
        "baseline_date_col": BASELINE_DATE_COL,
        "time_col_trained_raw": args.time_col,
        "event_col_trained_raw": args.event_col,
        "time_col_canonical": TIME_COL,
        "event_col_canonical": EVENT_COL,
        "tag": tag,
        "min_events_for_stats": int(args.min_events_for_stats),
    }
    with open(outdir / "model_bundle.pkl", "wb") as f:
        pickle.dump(bundle, f)

    print("Saved risk ->", risk_path)
    print("Saved bundle ->", outdir / "model_bundle.pkl")


if __name__ == "__main__":
    main()
