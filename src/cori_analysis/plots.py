from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from matplotlib.lines import Line2D

from .common import pformat, safe_name, savefig
from .evaluation import performance_row

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
