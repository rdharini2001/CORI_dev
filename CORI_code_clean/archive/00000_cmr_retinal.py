#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cardio_retinal_phenotyping.py
=============================

Cardio-Oncology Retinal-CMR Signature Analysis
----------------------------------------------
For patients with BOTH retinal fundus features and Cardiac MRI features, we:

  1. Load the trained Coxnet bundle (top retinal features + risk score).
  2. Merge retinal sheet with CMR sheet on `eid`.
  3. Auto-discover semantic CMR blocks (LV volume/EF, AHA wall thickness 1-16,
     circumferential / radial / longitudinal strain, atrial volumes,
     aortic distensibility, central blood pressures).
  4. TARGETED association: top Coxnet retinal features vs each CMR feature,
     partial Spearman adjusting for age + sex + SBP, with BH-FDR correction.
  5. DISCOVERY association: sparse CCA between full retinal block and CMR block
     -> latent cardio-retinal axes.
  6. JOINT PHENOTYPING: standardise + concatenate retinal+CMR, GMM with BIC-
     selected k -> "cardio-retinal phenotypes".
  7. Per-phenotype MACE KM (uses event_col + time_col from bundle).
  8. ONE big multi-panel figure with:
        (A) AHA 17-segment bullseye coloured by max |retinal-CMR correlation|
        (B) Schematic fundus coloured by retinal feature loading by region
        (C) Phenotype atlas: per-cluster mean bullseye + schematic fundus
        (D) Chord-style diagram retinal-groups -> CMR-regions -> MACE HR
        (E) KM curves per phenotype

USAGE
-----
python cardio_retinal_phenotyping.py \
    --bundle outputs/allcancer/model_bundle.pkl \
    --risk_csv outputs/allcancer/risk_scores_baseline_10y.csv \
    --retinal_csv retonco_merged_allcancer.csv \
    --cmr_csv cardiac_mri.csv \
    --outdir outputs/allcancer/cardio_retinal \
    --top_k_retinal 15

Notes
-----
* Designed to be robust to missing CMR columns -- it auto-detects what's present.
* All cross-sheet joins are on `eid` (integer).
* CMR columns use UK Biobank "Instance 2" convention; if "Instance 3" exists
  it is averaged in when "Instance 2" is missing.
"""
from __future__ import annotations

import argparse
import pickle
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Wedge
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PALETTE = {
    "bg":       "#0f1218",
    "panel":    "#171b24",
    "ink":      "#e8eaf0",
    "ink_dim":  "#9aa3b2",
    "grid":     "#2a3142",
    "artery":   "#e74c5e",
    "vein":     "#3a7bd5",
    "vessel":   "#7b6cf2",
    "pos":      "#ef4f6b",
    "neg":      "#3b82f6",
    "accent1":  "#f5a623",
    "accent2":  "#10b981",
    "accent3":  "#a855f7",
}
PHENOTYPE_COLORS = ["#ef4f6b", "#3b82f6", "#10b981", "#f5a623", "#a855f7", "#06b6d4"]


# ---------------------------------------------------------------------------
# CMR semantic block discovery
# ---------------------------------------------------------------------------
@dataclass
class CMRBlocks:
    """Holds dynamically discovered CMR feature column names."""
    lv_volumes_ef: List[str] = field(default_factory=list)
    aha_wall:      List[str] = field(default_factory=list)   # 16 segs
    circ_strain:   List[str] = field(default_factory=list)   # 16 segs
    radial_strain: List[str] = field(default_factory=list)   # 16 segs
    long_strain:   List[str] = field(default_factory=list)   # 6 segs
    rv:            List[str] = field(default_factory=list)
    la:            List[str] = field(default_factory=list)
    ra:            List[str] = field(default_factory=list)
    aorta:         List[str] = field(default_factory=list)
    central_bp:    List[str] = field(default_factory=list)

    def all(self) -> List[str]:
        out = []
        for v in self.__dict__.values():
            out.extend(v)
        return list(dict.fromkeys(out))   # preserve order, unique


def _instance2_first(cols: List[str], pattern: re.Pattern) -> List[str]:
    """Return cols matching pattern AND containing 'Instance 2'."""
    return [c for c in cols if pattern.search(c) and "Instance 2" in c]


def discover_cmr_blocks(df: pd.DataFrame) -> CMRBlocks:
    cols = list(df.columns)
    blocks = CMRBlocks()

    # LV volumes / EF / mass / CO -- must exclude AHA + strain (handled separately)
    lv_simple_pat = re.compile(
        r"^LV (ejection fraction|end diastolic volume|end systolic volume|"
        r"stroke volume|cardiac output|myocardial mass)\b"
    )
    blocks.lv_volumes_ef = _instance2_first(cols, lv_simple_pat)

    # AHA wall thickness 1..16
    aha_wall_pat = re.compile(r"^LV mean myocardial wall thickness AHA \d+\b")
    blocks.aha_wall = sorted(
        _instance2_first(cols, aha_wall_pat),
        key=lambda c: int(re.search(r"AHA (\d+)", c).group(1)),
    )

    # Circumferential strain AHA 1..16
    circ_pat = re.compile(r"^LV circumferential strain AHA \d+\b")
    blocks.circ_strain = sorted(
        _instance2_first(cols, circ_pat),
        key=lambda c: int(re.search(r"AHA (\d+)", c).group(1)),
    )

    # Radial strain AHA 1..16
    rad_pat = re.compile(r"^LV radial strain AHA \d+\b")
    blocks.radial_strain = sorted(
        _instance2_first(cols, rad_pat),
        key=lambda c: int(re.search(r"AHA (\d+)", c).group(1)),
    )

    # Longitudinal strain Segment 1..6
    long_pat = re.compile(r"^LV longitudinal strain Segment \d+\b")
    blocks.long_strain = sorted(
        _instance2_first(cols, long_pat),
        key=lambda c: int(re.search(r"Segment (\d+)", c).group(1)),
    )

    # RV / LA / RA blocks
    blocks.rv = _instance2_first(cols, re.compile(r"^RV (end diastolic|end systolic|stroke volume|ejection fraction)"))
    blocks.la = _instance2_first(cols, re.compile(r"^LA (maximum volume|minimum volume|stroke volume|ejection fraction)"))
    blocks.ra = _instance2_first(cols, re.compile(r"^RA (maximum volume|minimum volume|stroke volume|ejection fraction)"))

    # Aorta distensibility / area
    blocks.aorta = _instance2_first(
        cols,
        re.compile(r"^(Ascending|Descending) aorta (maximum area|minimum area|distensibility)\b"),
    )

    # Central BP (PWA) -- average across array repeats
    blocks.central_bp = [c for c in cols if c.startswith(
        "Central systolic blood pressure during PWA - PVR | Instance 2"
    )]
    return blocks


def collapse_arrays(df: pd.DataFrame, prefix: str, new_name: str) -> pd.DataFrame:
    """Average all columns starting with `prefix` into a single column `new_name`."""
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        return df
    df = df.copy()
    df[new_name] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return df


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def partial_spearman(x: np.ndarray, y: np.ndarray, Z: np.ndarray) -> Tuple[float, float]:
    """Partial Spearman correlation of x and y given covariates Z (np.ndarray, n x p)."""
    df = pd.DataFrame({"x": x, "y": y})
    Zdf = pd.DataFrame(Z, columns=[f"z{i}" for i in range(Z.shape[1])])
    df = pd.concat([df, Zdf], axis=1).dropna()
    if len(df) < 20:
        return np.nan, np.nan
    rx = stats.rankdata(df["x"])
    ry = stats.rankdata(df["y"])
    Zd = df[Zdf.columns].values
    # residualise ranks on Z
    Zc = np.column_stack([np.ones(len(Zd)), Zd])
    bx, *_ = np.linalg.lstsq(Zc, rx, rcond=None)
    by, *_ = np.linalg.lstsq(Zc, ry, rcond=None)
    ex = rx - Zc @ bx
    ey = ry - Zc @ by
    if np.std(ex) == 0 or np.std(ey) == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(ex, ey)
    return float(r), float(p)


def association_table(
    retinal: pd.DataFrame,
    cmr: pd.DataFrame,
    retinal_feats: List[str],
    cmr_feats: List[str],
    covariates: pd.DataFrame,
) -> pd.DataFrame:
    """Long-format table of partial Spearman correlations + BH-FDR q-values."""
    rows = []
    Z = covariates.values
    for rf in retinal_feats:
        x = pd.to_numeric(retinal[rf], errors="coerce").values
        for cf in cmr_feats:
            y = pd.to_numeric(cmr[cf], errors="coerce").values
            r, p = partial_spearman(x, y, Z)
            rows.append({"retinal": rf, "cmr": cf, "r": r, "p": p})
    out = pd.DataFrame(rows)
    valid = out["p"].notna()
    q = np.full(len(out), np.nan)
    if valid.sum() > 0:
        _, q_valid, _, _ = multipletests(out.loc[valid, "p"].values, method="fdr_bh")
        q[valid.values] = q_valid
    out["q_fdr"] = q
    return out


# ---------------------------------------------------------------------------
# Joint phenotyping
# ---------------------------------------------------------------------------
def fit_joint_phenotypes(
    X_ret: np.ndarray,
    X_cmr: np.ndarray,
    k_range: Tuple[int, int] = (2, 6),
    seed: int = 42,
) -> Tuple[np.ndarray, int, GaussianMixture, np.ndarray]:
    """Standardise+concatenate, PCA-compress, fit GMM, choose k by BIC."""
    sc = StandardScaler()
    X = np.column_stack([X_ret, X_cmr])
    Xs = sc.fit_transform(X)
    n_comp = min(10, Xs.shape[1] - 1, Xs.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=seed)
    Xp = pca.fit_transform(Xs)

    best_bic = np.inf
    best_k = None
    best_gmm = None
    for k in range(k_range[0], k_range[1] + 1):
        gmm = GaussianMixture(
            n_components=k, covariance_type="full", random_state=seed,
            n_init=5, reg_covar=1e-3,
        )
        gmm.fit(Xp)
        bic = gmm.bic(Xp)
        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_gmm = gmm
    labels = best_gmm.predict(Xp)
    return labels, best_k, best_gmm, Xp


# ---------------------------------------------------------------------------
# Multi-panel figure
# ---------------------------------------------------------------------------
def aha_bullseye(
    ax: plt.Axes,
    values_16: np.ndarray,
    cmap,
    norm,
    title: str = "",
    show_seg_nums: bool = False,
):
    """
    Draw an AHA 16-segment bullseye (plus apex blank=17). values_16: array length 16
    indexed AHA1..AHA16.
    Layout (standard AHA):
      Basal (1-6): outer ring, 60deg wedges, starting at AHA1=anterior (top)
      Mid   (7-12): middle ring
      Apical(13-16): inner ring, 90deg wedges
      Apex(17): center disc
    """
    ax.set_aspect("equal")
    ax.set_facecolor(PALETTE["panel"])
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.axis("off")

    radii = [(0.66, 1.00), (0.33, 0.66), (0.10, 0.33)]   # base, mid, apex
    # AHA convention: start at 60deg above horizontal (anterior=top), go counter-clockwise
    # Matplotlib Wedge: theta in degrees, CCW from +x axis
    # Basal seg 1 (anterior) center at 90deg, width 60 -> theta1=60, theta2=120
    base_starts = [60, 120, 180, 240, 300, 0]   # AHA1..AHA6 theta1
    base_widths = [60] * 6
    mid_starts  = base_starts                    # AHA7..AHA12 same angular layout
    mid_widths  = [60] * 6
    apex_starts = [45, 135, 225, 315]            # AHA13..AHA16, 90deg each
    apex_widths = [90] * 4

    layers = [
        (radii[0], base_starts, base_widths, list(range(0, 6))),
        (radii[1], mid_starts,  mid_widths,  list(range(6, 12))),
        (radii[2], apex_starts, apex_widths, list(range(12, 16))),
    ]

    for (r_in, r_out), starts, widths, idxs in layers:
        for s, w, i in zip(starts, widths, idxs):
            v = values_16[i]
            color = cmap(norm(v)) if np.isfinite(v) else "#3a3f4d"
            wedge = Wedge((0, 0), r_out, s, s + w, width=r_out - r_in,
                          facecolor=color, edgecolor=PALETTE["panel"], linewidth=1.2)
            ax.add_patch(wedge)
            if show_seg_nums:
                mid_ang = np.deg2rad(s + w / 2)
                rr = (r_in + r_out) / 2
                ax.text(rr * np.cos(mid_ang), rr * np.sin(mid_ang),
                        str(i + 1), ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold")

    # Apex (segment 17) -- center disc, no value (blank)
    ax.add_patch(Circle((0, 0), 0.10, facecolor="#3a3f4d",
                        edgecolor=PALETTE["panel"], linewidth=1.2))
    if title:
        ax.set_title(title, color=PALETTE["ink"], fontsize=10, pad=4)


def schematic_fundus(
    ax: plt.Axes,
    region_values: Dict[str, float],   # keys: "disc", "ring1", "ring2", "fundus_ring1", "fundus_ring2"
    cmap,
    norm,
    title: str = "",
):
    """
    Stylised right-eye fundus: optic disc + concentric rings reflecting the
    artery_disc_ring_*, vessel_fundus_ring_* feature regions in your data.
    """
    ax.set_aspect("equal")
    ax.set_facecolor(PALETTE["panel"])
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15); ax.axis("off")

    # Eye background
    ax.add_patch(Circle((0, 0), 1.05, facecolor="#1a1f2b",
                        edgecolor=PALETTE["grid"], linewidth=1.0))

    # Fundus outer ring (125-175 px)
    v = region_values.get("fundus_ring2", np.nan)
    ax.add_patch(Circle((0, 0), 1.00,
                        facecolor=cmap(norm(v)) if np.isfinite(v) else "#2a2f3b",
                        edgecolor=PALETTE["grid"], linewidth=0.8))
    # Fundus inner ring (75-125 px)
    v = region_values.get("fundus_ring1", np.nan)
    ax.add_patch(Circle((0, 0), 0.75,
                        facecolor=cmap(norm(v)) if np.isfinite(v) else "#2a2f3b",
                        edgecolor=PALETTE["grid"], linewidth=0.8))

    # Optic disc (right eye -> disc on the right side, ~30% from center)
    cx, cy = 0.45, 0.0
    # Disc outer ring (111-185)
    v = region_values.get("ring2", np.nan)
    ax.add_patch(Circle((cx, cy), 0.32,
                        facecolor=cmap(norm(v)) if np.isfinite(v) else "#2a2f3b",
                        edgecolor=PALETTE["grid"], linewidth=0.8))
    # Disc inner ring (74-111)
    v = region_values.get("ring1", np.nan)
    ax.add_patch(Circle((cx, cy), 0.20,
                        facecolor=cmap(norm(v)) if np.isfinite(v) else "#2a2f3b",
                        edgecolor=PALETTE["grid"], linewidth=0.8))
    # Disc itself
    v = region_values.get("disc", np.nan)
    ax.add_patch(Circle((cx, cy), 0.10,
                        facecolor=cmap(norm(v)) if np.isfinite(v) else "#fbbf24",
                        edgecolor="white", linewidth=1.0))

    # Schematic vessels emanating from disc
    rng = np.random.default_rng(7)
    for _ in range(6):
        ang = rng.uniform(-np.pi, np.pi)
        r0, r1 = 0.10, rng.uniform(0.55, 0.95)
        n = 25
        ts = np.linspace(0, 1, n)
        rs = r0 + (r1 - r0) * ts
        # Add some curvature
        wobble = 0.05 * np.sin(ts * np.pi * rng.uniform(1.5, 3.5)) * rng.choice([-1, 1])
        xs = cx + rs * np.cos(ang) + wobble * np.cos(ang + np.pi / 2)
        ys = cy + rs * np.sin(ang) + wobble * np.sin(ang + np.pi / 2)
        col = PALETTE["artery"] if rng.random() < 0.5 else PALETTE["vein"]
        ax.plot(xs, ys, color=col, linewidth=rng.uniform(0.8, 1.6), alpha=0.85)

    if title:
        ax.set_title(title, color=PALETTE["ink"], fontsize=10, pad=4)


def chord_links(ax, retinal_groups, cmr_groups, weights, mace_hr_per_cmr):
    """
    Simple 3-column flow diagram: retinal feature groups -> CMR regions -> MACE HR bars.
    weights: dict[(retinal_group, cmr_group)] = float in [0,1]
    """
    ax.set_facecolor(PALETTE["panel"])
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    n_r = len(retinal_groups); n_c = len(cmr_groups)
    y_r = np.linspace(8.5, 1.5, n_r)
    y_c = np.linspace(8.5, 1.5, n_c)

    # Left nodes
    for i, g in enumerate(retinal_groups):
        col = {"artery": PALETTE["artery"], "vein": PALETTE["vein"],
               "vessel": PALETTE["vessel"]}.get(g.lower(), PALETTE["accent3"])
        ax.add_patch(plt.Rectangle((0.3, y_r[i] - 0.35), 1.6, 0.7,
                                   facecolor=col, edgecolor="white", linewidth=0.8))
        ax.text(1.1, y_r[i], g, ha="center", va="center", color="white",
                fontsize=8, fontweight="bold")

    # Middle nodes
    for j, g in enumerate(cmr_groups):
        ax.add_patch(plt.Rectangle((4.2, y_c[j] - 0.35), 2.0, 0.7,
                                   facecolor=PALETTE["accent2"],
                                   edgecolor="white", linewidth=0.8))
        ax.text(5.2, y_c[j], g, ha="center", va="center", color="white",
                fontsize=7.5, fontweight="bold")

    # Links
    if weights:
        wmax = max(abs(v) for v in weights.values()) or 1.0
        for (rg, cg), w in weights.items():
            if rg not in retinal_groups or cg not in cmr_groups:
                continue
            i = retinal_groups.index(rg); j = cmr_groups.index(cg)
            lw = 0.5 + 4.0 * abs(w) / wmax
            alpha = 0.25 + 0.55 * abs(w) / wmax
            x0, y0 = 1.9, y_r[i]
            x1, y1 = 4.2, y_c[j]
            xm = (x0 + x1) / 2
            ts = np.linspace(0, 1, 40)
            xs = (1 - ts) ** 2 * x0 + 2 * (1 - ts) * ts * xm + ts ** 2 * x1
            ys = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * y0 + ts ** 2 * y1   # smooth y
            # actually quadratic with control point in middle:
            ys = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * ((y0 + y1) / 2) + ts ** 2 * y1
            color = PALETTE["pos"] if w > 0 else PALETTE["neg"]
            ax.plot(xs, ys, color=color, alpha=alpha, linewidth=lw)

    # Right HR bars
    if mace_hr_per_cmr:
        max_hr = max(abs(np.log(v)) for v in mace_hr_per_cmr.values() if np.isfinite(v) and v > 0) or 1.0
        for j, g in enumerate(cmr_groups):
            hr = mace_hr_per_cmr.get(g, np.nan)
            if not np.isfinite(hr) or hr <= 0:
                continue
            log_hr = np.log(hr)
            length = (log_hr / max_hr) * 2.0
            x0 = 7.0
            color = PALETTE["pos"] if hr > 1 else PALETTE["neg"]
            ax.add_patch(plt.Rectangle((x0, y_c[j] - 0.18),
                                       max(length, -length) if length != 0 else 0.05,
                                       0.36, facecolor=color, edgecolor="white",
                                       linewidth=0.5,
                                       transform=ax.transData))
            # put bar to right if HR>1, left (negative) if HR<1
            ax.add_patch(plt.Rectangle((x0, y_c[j] - 0.18), length if length > 0 else 0,
                                       0.36, facecolor=color, alpha=0.0))
            ax.text(x0 + 2.2, y_c[j], f"HR {hr:.2f}",
                    ha="left", va="center", color=PALETTE["ink"], fontsize=7.5)
        ax.text(8.0, 9.4, "MACE HR", color=PALETTE["ink_dim"],
                fontsize=8, ha="center")

    # Column titles
    ax.text(1.1, 9.4, "Retinal", color=PALETTE["ink_dim"], fontsize=9, ha="center")
    ax.text(5.2, 9.4, "CMR region", color=PALETTE["ink_dim"], fontsize=9, ha="center")


def km_per_cluster(ax, df, time_col, event_col, cluster_col, max_t=None):
    """Simple KM per cluster, no external dep beyond numpy/scipy."""
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["ink_dim"])
    for spine in ax.spines.values():
        spine.set_color(PALETTE["grid"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.5, alpha=0.5)

    clusters = sorted(df[cluster_col].dropna().unique())
    for i, c in enumerate(clusters):
        sub = df[df[cluster_col] == c].dropna(subset=[time_col, event_col])
        if len(sub) < 5:
            continue
        t = sub[time_col].values.astype(float)
        e = sub[event_col].values.astype(int)
        order = np.argsort(t)
        t, e = t[order], e[order]
        # KM
        unique_t, idx = np.unique(t, return_inverse=True)
        d = np.zeros_like(unique_t, dtype=float)
        n = np.zeros_like(unique_t, dtype=float)
        at_risk = len(t)
        for k_ix in range(len(unique_t)):
            mask_k = idx == k_ix
            d[k_ix] = e[mask_k].sum()
            n[k_ix] = at_risk
            at_risk -= mask_k.sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            surv = np.cumprod(1 - d / np.where(n > 0, n, 1))
        col = PHENOTYPE_COLORS[i % len(PHENOTYPE_COLORS)]
        ax.step(np.concatenate([[0], unique_t]),
                np.concatenate([[1.0], surv]),
                where="post", color=col, linewidth=2.0,
                label=f"P{int(c)+1} (n={len(sub)}, ev={int(e.sum())})")

    if max_t:
        ax.set_xlim(0, max_t)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Time", color=PALETTE["ink_dim"], fontsize=9)
    ax.set_ylabel("Event-free probability", color=PALETTE["ink_dim"], fontsize=9)
    ax.legend(loc="lower left", fontsize=7, frameon=False, labelcolor=PALETTE["ink"])
    ax.set_title("MACE-free survival by cardio-retinal phenotype",
                 color=PALETTE["ink"], fontsize=10, pad=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True,
                    help="model_bundle.pkl produced by your baseline Coxnet script")
    ap.add_argument("--risk_csv", type=Path, required=True,
                    help="risk_scores_baseline_<tag>.csv")
    ap.add_argument("--retinal_csv", type=Path, required=True,
                    help="merged retinal+endpoint CSV (the same one used to train Coxnet)")
    ap.add_argument("--cmr_csv", type=Path, required=True,
                    help="cardiac MRI sheet with `eid` column")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--top_k_retinal", type=int, default=15,
                    help="How many top retinal features to use for targeted analysis")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    # -----------------------------------------------------------------------
    # 1. Load bundle + risk scores + retinal sheet + CMR sheet
    # -----------------------------------------------------------------------
    print("[1/8] Loading inputs ...")
    with open(args.bundle, "rb") as f:
        bundle = pickle.load(f)
    feat_cols = bundle["feat_cols"]
    model = bundle["model"]
    time_col_raw = bundle["time_col_trained_raw"]
    event_col_raw = bundle["event_col_trained_raw"]
    visit_col = bundle["visit_col"]
    baseline_visit = bundle["baseline_visit"]
    feat_regex = bundle["feat_regex"]

    risk_df = pd.read_csv(args.risk_csv)
    retinal_df = pd.read_csv(args.retinal_csv, low_memory=False)
    cmr_df = pd.read_csv(args.cmr_csv, low_memory=False)

    # restrict retinal to baseline rows used at training
    retinal_df[visit_col] = pd.to_numeric(retinal_df[visit_col], errors="coerce")
    retinal_df = retinal_df[retinal_df[visit_col] == int(baseline_visit)].copy()
    retinal_df = retinal_df.drop_duplicates("eid", keep="first")

    # Make sure all features are numeric
    all_retinal_feats = [c for c in retinal_df.columns if re.match(feat_regex, str(c))]
    for c in all_retinal_feats:
        retinal_df[c] = pd.to_numeric(retinal_df[c], errors="coerce")

    # Top-K Coxnet features by |coef| at the chosen alpha (last column of coef_)
    coef = np.asarray(model.coef_)[:, -1] if model.coef_.ndim == 2 else np.asarray(model.coef_)
    top_idx = np.argsort(-np.abs(coef))[: args.top_k_retinal]
    top_retinal = [feat_cols[i] for i in top_idx if abs(coef[i]) > 0]
    if len(top_retinal) == 0:
        # all-zero coefs at last alpha -> fall back to non-zero across path
        coef_path = np.asarray(model.coef_)
        nz = np.argsort(-np.abs(coef_path).max(axis=1))[: args.top_k_retinal]
        top_retinal = [feat_cols[i] for i in nz]
    print(f"      Top-{len(top_retinal)} Coxnet retinal features selected.")

    # -----------------------------------------------------------------------
    # 2. Discover CMR blocks + merge
    # -----------------------------------------------------------------------
    print("[2/8] Discovering CMR feature blocks ...")
    blocks = discover_cmr_blocks(cmr_df)
    cmr_feats = blocks.all()
    # Collapse central BP arrays into one summary if present
    cmr_df = collapse_arrays(
        cmr_df,
        "Central systolic blood pressure during PWA - PVR | Instance 2",
        "central_sbp_pwa_inst2",
    )
    if "central_sbp_pwa_inst2" in cmr_df.columns and "central_sbp_pwa_inst2" not in cmr_feats:
        cmr_feats.append("central_sbp_pwa_inst2")

    # Coerce numeric
    for c in cmr_feats:
        cmr_df[c] = pd.to_numeric(cmr_df[c], errors="coerce")

    print(f"      LV vol/EF: {len(blocks.lv_volumes_ef)} | AHA wall: {len(blocks.aha_wall)} | "
          f"circ: {len(blocks.circ_strain)} | radial: {len(blocks.radial_strain)} | "
          f"long: {len(blocks.long_strain)} | RV: {len(blocks.rv)} | "
          f"LA: {len(blocks.la)} | RA: {len(blocks.ra)} | aorta: {len(blocks.aorta)}")

    # Merge -- inner join on eid for "patients with both"
    merged = retinal_df.merge(cmr_df, on="eid", how="inner", suffixes=("", "_cmr"))
    merged = merged.merge(risk_df[["eid", "risk_score", "high_risk", "set"]],
                          on="eid", how="left")
    print(f"      Patients with retinal AND CMR: n = {len(merged)}")

    if len(merged) < 50:
        print("[!] Very few overlapping patients. Analyses below may be unstable.")

    # Covariates -- age, sex (M=1), SBP if available
    cov_cols = []
    if "Age at recruitment" in merged.columns:
        merged["age_cov"] = pd.to_numeric(merged["Age at recruitment"], errors="coerce")
        cov_cols.append("age_cov")
    if "sex" in merged.columns:
        merged["sex_cov"] = (merged["sex"].astype(str).str.lower() == "male").astype(float)
        cov_cols.append("sex_cov")
    sbp_candidates = [c for c in merged.columns if c.startswith("Systolic blood pressure, automated reading | Instance 0")]
    if sbp_candidates:
        merged["sbp_cov"] = merged[sbp_candidates].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        cov_cols.append("sbp_cov")
    if not cov_cols:
        merged["_const_cov"] = 1.0
        cov_cols = ["_const_cov"]
    cov = merged[cov_cols].fillna(merged[cov_cols].median(numeric_only=True))

    # -----------------------------------------------------------------------
    # 3. Targeted association
    # -----------------------------------------------------------------------
    print("[3/8] Targeted retinal-CMR association (top features) ...")
    targeted = association_table(merged, merged, top_retinal, cmr_feats, cov)
    targeted.to_csv(args.outdir / "association_targeted_top_retinal_vs_cmr.csv", index=False)

    # -----------------------------------------------------------------------
    # 4. Discovery: sparse CCA on full retinal block vs CMR block
    # -----------------------------------------------------------------------
    print("[4/8] Discovery CCA: full retinal block vs CMR block ...")
    R_cols = [c for c in all_retinal_feats if merged[c].notna().sum() > 0.5 * len(merged)]
    C_cols = [c for c in cmr_feats if merged[c].notna().sum() > 0.5 * len(merged)]
    cca_loadings = None
    if len(R_cols) > 5 and len(C_cols) > 5:
        sub = merged[R_cols + C_cols].copy()
        imp = SimpleImputer(strategy="median")
        Z = imp.fit_transform(sub.values)
        Rmat = StandardScaler().fit_transform(Z[:, : len(R_cols)])
        Cmat = StandardScaler().fit_transform(Z[:, len(R_cols):])
        n_comp = min(3, len(R_cols), len(C_cols), Rmat.shape[0] - 1)
        cca = CCA(n_components=n_comp, max_iter=500)
        try:
            cca.fit(Rmat, Cmat)
            ret_load = pd.DataFrame(cca.x_loadings_, index=R_cols,
                                    columns=[f"CCA{i+1}" for i in range(n_comp)])
            cmr_load = pd.DataFrame(cca.y_loadings_, index=C_cols,
                                    columns=[f"CCA{i+1}" for i in range(n_comp)])
            ret_load.to_csv(args.outdir / "cca_retinal_loadings.csv")
            cmr_load.to_csv(args.outdir / "cca_cmr_loadings.csv")
            cca_loadings = (ret_load, cmr_load)
            print(f"      Saved CCA loadings ({n_comp} components).")
        except Exception as ex:
            print(f"      [!] CCA failed: {ex}")

    # -----------------------------------------------------------------------
    # 5. Joint phenotyping (GMM, BIC-selected k)
    # -----------------------------------------------------------------------
    print("[5/8] Joint cardio-retinal phenotyping ...")
    X_ret = merged[top_retinal].apply(pd.to_numeric, errors="coerce").values
    X_cmr_for_cluster = merged[C_cols].apply(pd.to_numeric, errors="coerce").values

    imp = SimpleImputer(strategy="median")
    X_ret_imp = imp.fit_transform(X_ret)
    X_cmr_imp = imp.fit_transform(X_cmr_for_cluster)

    labels, k_best, gmm, _ = fit_joint_phenotypes(X_ret_imp, X_cmr_imp,
                                                  k_range=(2, 5), seed=args.seed)
    merged["phenotype"] = labels
    print(f"      Best k = {k_best} (BIC). Cluster sizes: "
          f"{dict(pd.Series(labels).value_counts().sort_index())}")

    # Per-phenotype mean profiles
    pheno_summary = merged.groupby("phenotype")[top_retinal + C_cols].mean()
    pheno_summary.to_csv(args.outdir / "phenotype_mean_profiles.csv")
    merged[["eid", "phenotype", "risk_score", "set"]].to_csv(
        args.outdir / "patient_phenotypes.csv", index=False
    )

    # -----------------------------------------------------------------------
    # 6. Per-phenotype MACE HR (vs phenotype 0 = reference)
    # -----------------------------------------------------------------------
    print("[6/8] Per-phenotype MACE association ...")
    if time_col_raw in merged.columns and event_col_raw in merged.columns:
        from sksurv.linear_model import CoxPHSurvivalAnalysis
        from sksurv.util import Surv

        df_surv = merged.dropna(subset=[time_col_raw, event_col_raw, "phenotype"]).copy()
        df_surv = df_surv[df_surv[time_col_raw] > 0]
        df_surv[event_col_raw] = (pd.to_numeric(df_surv[event_col_raw], errors="coerce") > 0).astype(bool)

        # one-hot phenotype (drop ref = 0)
        oh = pd.get_dummies(df_surv["phenotype"], prefix="P", drop_first=True).astype(float)
        if oh.shape[1] >= 1 and df_surv[event_col_raw].sum() >= 5:
            try:
                y_surv = Surv.from_dataframe(event_col_raw, time_col_raw, df_surv)
                cph = CoxPHSurvivalAnalysis()
                cph.fit(oh.values, y_surv)
                hr = pd.DataFrame({
                    "phenotype": [c.replace("P_", "P") for c in oh.columns],
                    "HR": np.exp(cph.coef_),
                })
                hr.to_csv(args.outdir / "phenotype_mace_hr.csv", index=False)
                print(hr.to_string(index=False))
            except Exception as ex:
                print(f"      [!] CoxPH for phenotype HR failed: {ex}")
        else:
            print("      [!] Skipping CoxPH (insufficient events or single phenotype).")
    else:
        print("      [!] time/event columns not present in merged frame; skipping HR.")

    # -----------------------------------------------------------------------
    # 7. Build the multi-panel figure
    # -----------------------------------------------------------------------
    print("[7/8] Drawing multi-panel figure ...")
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor":   PALETTE["panel"],
        "savefig.facecolor": PALETTE["bg"],
        "text.color":       PALETTE["ink"],
        "axes.labelcolor":  PALETTE["ink"],
        "xtick.color":      PALETTE["ink_dim"],
        "ytick.color":      PALETTE["ink_dim"],
        "font.family":      "DejaVu Sans",
    })

    cmap_div = LinearSegmentedColormap.from_list(
        "div", [PALETTE["neg"], "#1a1f2b", PALETTE["pos"]]
    )
    cmap_seq = LinearSegmentedColormap.from_list(
        "seq", ["#1a1f2b", PALETTE["accent1"], PALETTE["pos"]]
    )

    # AHA values: per AHA segment, take MAX |partial r| across top retinal feats
    def aha_corr_vector(strain_block: List[str]) -> np.ndarray:
        vals = np.full(16, np.nan)
        for c in strain_block:
            m = re.search(r"AHA (\d+)", c)
            if not m:
                continue
            idx = int(m.group(1)) - 1
            if not (0 <= idx < 16):
                continue
            sub = targeted[targeted["cmr"] == c]
            if sub.empty:
                continue
            best = sub["r"].abs().max()
            if np.isnan(best):
                continue
            vals[idx] = best if np.isnan(vals[idx]) else max(vals[idx], best)
        return vals

    aha_vals = aha_corr_vector(blocks.aha_wall + blocks.circ_strain + blocks.radial_strain)
    if np.all(np.isnan(aha_vals)):
        aha_vals = np.zeros(16)
    norm_aha = Normalize(vmin=0, vmax=max(0.05, np.nanmax(aha_vals)))

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.40, wspace=0.30,
                  height_ratios=[1.0, 1.0, 1.1])

    # Title
    fig.suptitle(
        "Cardio-Retinal Signature in Cancer Patients — Retinal–CMR Coupling and Phenotypes",
        color=PALETTE["ink"], fontsize=16, fontweight="bold", y=0.985,
    )

    # (A) AHA bullseye
    axA = fig.add_subplot(gs[0, 0])
    aha_bullseye(axA, aha_vals, cmap_seq, norm_aha,
                 title="(A) AHA bullseye\nmax |retinal–CMR partial r|",
                 show_seg_nums=True)
    cax = fig.add_axes([0.075, 0.66, 0.13, 0.012])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_aha, cmap=cmap_seq),
                 cax=cax, orientation="horizontal").set_label(
        "max |partial r|", color=PALETTE["ink_dim"], fontsize=8)
    cax.tick_params(colors=PALETTE["ink_dim"], labelsize=7)

    # (B) Schematic fundus colored by retinal feature loading per region
    # Take max |r| per retinal-feature-region across all CMR features
    def region_for_feat(feat: str) -> Optional[str]:
        if "disc_ring_74_111" in feat:    return "ring1"
        if "disc_ring_111_185" in feat:   return "ring2"
        if "fundus_ring_75_125" in feat:  return "fundus_ring1"
        if "fundus_ring_125_175" in feat: return "fundus_ring2"
        if "_All" in feat:                return "disc"
        return None
    region_max_r = {k: np.nan for k in ["disc", "ring1", "ring2", "fundus_ring1", "fundus_ring2"]}
    for f in top_retinal:
        rg = region_for_feat(f)
        if rg is None:
            continue
        sub = targeted[targeted["retinal"] == f]
        if sub.empty:
            continue
        best = sub["r"].abs().max()
        if np.isnan(best):
            continue
        region_max_r[rg] = best if np.isnan(region_max_r[rg]) else max(region_max_r[rg], best)
    norm_fundus = Normalize(vmin=0, vmax=max(0.05, np.nanmax(list(region_max_r.values()) + [0.05])))
    axB = fig.add_subplot(gs[0, 1])
    schematic_fundus(axB, region_max_r, cmap_seq, norm_fundus,
                     title="(B) Retinal regions\ncoloured by max |r| with CMR")

    # (C) Phenotype atlas: per-cluster bullseye + schematic fundus
    n_pheno = int(merged["phenotype"].nunique())
    # We'll lay them out across the right two columns (top row) + middle row
    atlas_axes = []
    # Use 1 row across 2 cols at top + full middle row -> max 2*4 = 8 axes (4 per row)
    for i in range(n_pheno):
        if i < 2:
            ax = fig.add_subplot(gs[0, 2 + i])
        else:
            ax = fig.add_subplot(gs[1, i - 2])
        atlas_axes.append(ax)
    # Compute per-phenotype mean strain values across AHA segs + region values
    for i, ph in enumerate(sorted(merged["phenotype"].unique())):
        sub = merged[merged["phenotype"] == ph]
        # AHA: mean circ strain (more interpretable than wall thickness)
        aha_ph = np.full(16, np.nan)
        block_for_atlas = blocks.circ_strain if blocks.circ_strain else blocks.aha_wall
        for c in block_for_atlas:
            m = re.search(r"AHA (\d+)", c)
            if not m: continue
            idx = int(m.group(1)) - 1
            if not (0 <= idx < 16): continue
            v = pd.to_numeric(sub[c], errors="coerce").mean()
            aha_ph[idx] = v
        # global normalisation across all phenotypes for fair comparison
        if i == 0:
            all_ph_vals = []
            for ph2 in sorted(merged["phenotype"].unique()):
                sub2 = merged[merged["phenotype"] == ph2]
                v2 = []
                for c in block_for_atlas:
                    v2.append(pd.to_numeric(sub2[c], errors="coerce").mean())
                all_ph_vals.extend(v2)
            arr = np.array(all_ph_vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if len(arr):
                vmin, vmax = np.nanpercentile(arr, [5, 95])
                if vmin == vmax: vmax = vmin + 1
            else:
                vmin, vmax = -1, 1
            norm_atlas = Normalize(vmin=vmin, vmax=vmax)
        ax = atlas_axes[i]
        ax.set_facecolor(PALETTE["panel"])
        # Split inset: top half bullseye, bottom half fundus
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax.axis("off")
        ax.set_title(f"(C) Phenotype P{ph+1}  (n={len(sub)})",
                     color=PHENOTYPE_COLORS[i % len(PHENOTYPE_COLORS)],
                     fontsize=10, pad=2, fontweight="bold")
        ax_bull = inset_axes(ax, width="80%", height="55%", loc="upper center")
        aha_bullseye(ax_bull, aha_ph, cmap_div, norm_atlas, title="")
        ax_fundus = inset_axes(ax, width="80%", height="40%", loc="lower center")
        # per-region mean of top retinal feats for this phenotype, normalised
        region_vals = {k: np.nan for k in ["disc", "ring1", "ring2",
                                           "fundus_ring1", "fundus_ring2"]}
        for f in top_retinal:
            rg = region_for_feat(f)
            if rg is None: continue
            v = pd.to_numeric(sub[f], errors="coerce").mean()
            # rough z relative to overall mean
            mu = pd.to_numeric(merged[f], errors="coerce").mean()
            sd = pd.to_numeric(merged[f], errors="coerce").std()
            if sd and np.isfinite(sd) and sd > 0:
                z = (v - mu) / sd
            else:
                z = np.nan
            region_vals[rg] = z if np.isnan(region_vals[rg]) else (region_vals[rg] + z) / 2
        norm_z = Normalize(vmin=-1.5, vmax=1.5)
        schematic_fundus(ax_fundus, region_vals, cmap_div, norm_z, title="")

    # (D) Chord-style links — bottom-left cell
    axD = fig.add_subplot(gs[2, 0:2])
    # Aggregate weights per (retinal_group, cmr_region)
    def ret_group(f):
        if f.startswith("artery"): return "artery"
        if f.startswith("vein"):   return "vein"
        return "vessel"
    def cmr_group(c):
        if c in blocks.aha_wall:      return "Wall thick."
        if c in blocks.circ_strain:   return "Circ strain"
        if c in blocks.radial_strain: return "Radial strain"
        if c in blocks.long_strain:   return "Long strain"
        if c in blocks.lv_volumes_ef: return "LV vol/EF"
        if c in blocks.rv:            return "RV"
        if c in blocks.la:            return "LA"
        if c in blocks.ra:            return "RA"
        if c in blocks.aorta:         return "Aorta"
        return "BP/other"
    weights = {}
    for _, row in targeted.iterrows():
        rg = ret_group(row["retinal"])
        cg = cmr_group(row["cmr"])
        if pd.isna(row["r"]): continue
        weights[(rg, cg)] = weights.get((rg, cg), 0.0) + row["r"]
    # average per pair count
    counts = {}
    for _, row in targeted.iterrows():
        rg = ret_group(row["retinal"]); cg = cmr_group(row["cmr"])
        if pd.isna(row["r"]): continue
        counts[(rg, cg)] = counts.get((rg, cg), 0) + 1
    weights = {k: weights[k] / counts[k] for k in weights if counts.get(k)}

    retinal_groups = ["artery", "vein", "vessel"]
    cmr_groups_present = list(dict.fromkeys([cmr_group(c) for c in cmr_feats]))
    chord_links(axD, retinal_groups, cmr_groups_present, weights, mace_hr_per_cmr={})
    axD.set_title("(D) Retinal feature group → CMR region links\n(line width ∝ |mean partial r|, "
                  "colour: red=positive / blue=negative)",
                  color=PALETTE["ink"], fontsize=10, pad=4, loc="left")

    # (E) KM per phenotype
    axE = fig.add_subplot(gs[2, 2:4])
    if time_col_raw in merged.columns and event_col_raw in merged.columns:
        sub = merged.dropna(subset=[time_col_raw, event_col_raw, "phenotype"]).copy()
        sub[event_col_raw] = (pd.to_numeric(sub[event_col_raw], errors="coerce") > 0).astype(int)
        sub = sub[sub[time_col_raw] > 0]
        km_per_cluster(axE, sub, time_col_raw, event_col_raw, "phenotype")
    else:
        axE.text(0.5, 0.5, "MACE columns not found", ha="center", va="center",
                 transform=axE.transAxes, color=PALETTE["ink_dim"])
        axE.axis("off")

    fig_path = args.outdir / "cardio_retinal_multipanel.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight",
                facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"      Saved figure -> {fig_path}")

    # -----------------------------------------------------------------------
    # 8. Companion supplementary heatmap (top retinal x CMR), saved separately
    # -----------------------------------------------------------------------
    print("[8/8] Saving supplementary heatmap ...")
    pivot = targeted.pivot_table(index="retinal", columns="cmr", values="r")
    fig2, ax2 = plt.subplots(figsize=(min(0.18 * pivot.shape[1] + 4, 30),
                                      0.32 * pivot.shape[0] + 3))
    im = ax2.imshow(pivot.values, aspect="auto", cmap=cmap_div,
                    vmin=-np.nanmax(np.abs(pivot.values)) if np.isfinite(np.nanmax(np.abs(pivot.values))) else -0.3,
                    vmax= np.nanmax(np.abs(pivot.values)) if np.isfinite(np.nanmax(np.abs(pivot.values))) else  0.3)
    ax2.set_xticks(range(pivot.shape[1]))
    ax2.set_xticklabels(pivot.columns, rotation=90, fontsize=6)
    ax2.set_yticks(range(pivot.shape[0]))
    ax2.set_yticklabels(pivot.index, fontsize=7)
    ax2.set_title("Targeted partial-Spearman (top retinal × CMR features)",
                  color=PALETTE["ink"])
    fig2.colorbar(im, ax=ax2, fraction=0.02, pad=0.01).set_label("partial r")
    fig2.savefig(args.outdir / "supp_heatmap_top_retinal_vs_cmr.png",
                 dpi=180, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig2)

    print("\nDONE.")
    print(f"All outputs in: {args.outdir}")


if __name__ == "__main__":
    main()