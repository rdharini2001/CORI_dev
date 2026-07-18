from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .common import savefig

def feature_rank_shift_bump_FIXED(
    cori_rank,
    mmace_rank,
    figdir,
    tabledir,
    top_n=30,
    prefix="H2",
):
    """
    Clean feature-rank transition plot for CORI vs M-MACE.

    Fixes the giant-white-space issue by:
    1. plotting only top_n ranks on the y-axis,
    2. mapping any feature outside top_n to a single '>top_n' row,
    3. saving to a new *_FIXED_COMPACT filename,
    4. returning a table showing true full ranks and capped plot ranks.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from matplotlib.lines import Line2D

    figdir = Path(figdir)
    tabledir = Path(tabledir)
    figdir.mkdir(parents=True, exist_ok=True)
    tabledir.mkdir(parents=True, exist_ok=True)

    def _prep(rank_df, label):
        r = rank_df.copy()

        if "feature" not in r.columns:
            raise ValueError(f"{label} ranking table must have a feature column.")

        if "status" in r.columns:
            r = r[r["status"].astype(str).eq("ok")].copy()

        for c in ["p", "z_abs", "importance", "hr_per_sd", "coef"]:
            if c in r.columns:
                r[c] = pd.to_numeric(r[c], errors="coerce")

        if "importance" not in r.columns:
            r["importance"] = np.nan

        if "p" in r.columns:
            p_clean = r["p"].clip(lower=1e-300)
            r["importance"] = r["importance"].fillna(-np.log10(p_clean))

        if "p" in r.columns and "z_abs" in r.columns:
            r = r.sort_values(["p", "z_abs"], ascending=[True, False])
        elif "importance" in r.columns:
            r = r.sort_values("importance", ascending=False)
        else:
            r = r.reset_index(drop=True)

        r = r.drop_duplicates("feature", keep="first").reset_index(drop=True)
        r[f"{label}_rank"] = np.arange(1, len(r) + 1)
        r[f"{label}_importance"] = r["importance"]

        if "hr_per_sd" in r.columns:
            r[f"{label}_hr"] = r["hr_per_sd"]
        elif "coef" in r.columns:
            r[f"{label}_hr"] = np.exp(r["coef"])
        else:
            r[f"{label}_hr"] = np.nan

        return r[["feature", f"{label}_rank", f"{label}_importance", f"{label}_hr"]]

    cori = _prep(cori_rank, "CORI")
    mmace = _prep(mmace_rank, "MMACE")

    merged = cori.merge(mmace, on="feature", how="outer")

    # Select top_n in either model
    sel = merged[
        (merged["CORI_rank"].le(top_n)) | (merged["MMACE_rank"].le(top_n))
    ].copy()

    if sel.empty:
        print("No top-ranked features found for transition plot.")
        return sel

    sel["best_rank"] = sel[["CORI_rank", "MMACE_rank"]].min(axis=1)
    sel = sel.sort_values(["best_rank", "CORI_rank", "MMACE_rank"]).reset_index(drop=True)

    # Cap everything outside top_n to one row
    capped = top_n + 1
    sel["CORI_plot_rank"] = sel["CORI_rank"].where(sel["CORI_rank"].le(top_n), capped).fillna(capped)
    sel["MMACE_plot_rank"] = sel["MMACE_rank"].where(sel["MMACE_rank"].le(top_n), capped).fillna(capped)

    # This is the plotted shift, not the full-rank shift
    sel["plot_rank_shift_MMACE_minus_CORI"] = sel["MMACE_plot_rank"] - sel["CORI_plot_rank"]

    # True full-rank shift for table
    sel["full_rank_shift_MMACE_minus_CORI"] = sel["MMACE_rank"] - sel["CORI_rank"]

    def rank_label(x):
        if pd.isna(x) or x > top_n:
            return f">{top_n}"
        return str(int(x))

    def membership(row):
        in_cori = pd.notna(row["CORI_rank"]) and row["CORI_rank"] <= top_n
        in_mmace = pd.notna(row["MMACE_rank"]) and row["MMACE_rank"] <= top_n
        if in_cori and in_mmace:
            return "Top in both"
        if in_cori:
            return "CORI top only"
        if in_mmace:
            return "M-MACE top only"
        return "Other"

    sel["membership"] = sel.apply(membership, axis=1)

    def color(row):
        if row["membership"] == "CORI top only":
            return "tab:red"
        if row["membership"] == "M-MACE top only":
            return "tab:blue"
        shift = row["plot_rank_shift_MMACE_minus_CORI"]
        if shift >= 2:
            return "tab:red"
        if shift <= -2:
            return "tab:blue"
        return "0.45"

    def dot_size(x):
        if pd.isna(x) or not np.isfinite(x):
            return 35
        return float(np.clip(30 + 16 * x, 35, 160))

    # Save the table first
    sel.to_csv(
        tabledir / f"{prefix}_feature_rank_transition_FIXED_COMPACT_table.csv",
        index=False,
    )

    fig_h = max(7, min(12, 0.28 * len(sel) + 2.5))
    fig, ax = plt.subplots(figsize=(9.8, fig_h))

    for _, row in sel.iterrows():
        c = color(row)

        y0 = row["CORI_plot_rank"]
        y1 = row["MMACE_plot_rank"]

        ax.plot(
            [0, 1],
            [y0, y1],
            color=c,
            alpha=0.70,
            linewidth=1.4,
            zorder=1,
        )

        ax.scatter(
            [0],
            [y0],
            s=dot_size(row["CORI_importance"]),
            color=c,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

        ax.scatter(
            [1],
            [y1],
            s=dot_size(row["MMACE_importance"]),
            color=c,
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )

        ax.text(
            -0.035,
            y0,
            f"{row['feature']} ({rank_label(row['CORI_rank'])})",
            ha="right",
            va="center",
            fontsize=7.2,
        )

        ax.text(
            1.035,
            y1,
            f"{row['feature']} ({rank_label(row['MMACE_rank'])})",
            ha="left",
            va="center",
            fontsize=7.2,
        )

    # Critical: force compact y-axis.
    ax.set_ylim(capped + 0.8, 0.5)

    yticks = list(range(1, top_n + 1, 2)) + [capped]
    ylabels = [str(y) for y in yticks[:-1]] + [f">{top_n}"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)

    ax.set_xlim(-0.52, 1.52)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(
        ["Cancer-trained CORI\nfeature rank", "Non-cancer M-MACE\nfeature rank"],
        fontsize=10,
    )

    ax.set_ylabel("Rank; 1 = strongest univariate Cox feature")
    ax.set_title(
        f"Top-{top_n} feature-rank transition: CORI vs M-MACE",
        fontweight="bold",
    )

    ax.grid(axis="y", alpha=0.18)
    ax.spines[["top", "right"]].set_visible(False)

    legend_items = [
        Line2D([0], [0], color="tab:red", lw=2.5, label="CORI top-ranked / lower in M-MACE"),
        Line2D([0], [0], color="tab:blue", lw=2.5, label="M-MACE top-ranked / lower in CORI"),
        Line2D([0], [0], color="0.45", lw=2.5, label="Top-ranked in both / similar"),
    ]
    ax.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.17),
        frameon=False,
        fontsize=8,
    )

    fig.tight_layout()

    # Save to a NEW filename so you know this fixed function ran.
    savefig(fig, figdir / f"{prefix}_feature_rank_transition_FIXED_COMPACT")
    plt.show()

    print(f"Saved fixed feature-rank plot to: {figdir / f'{prefix}_feature_rank_transition_FIXED_COMPACT.png'}")
    print(f"Y-axis capped at >{top_n}. If you still see huge whitespace, you are viewing the old file.")

    return sel