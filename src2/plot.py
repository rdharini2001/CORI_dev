
# Standard library
import argparse
import glob
import json
import os
import re
import warnings
from datetime import datetime
from pathlib import Path

# Third-party
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import forestplot as fp
import tqdm
from sklearn.linear_model import LassoCV

# lifelines
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index

def compare_quantile_logrank(
    train_df,
    test_df,
    risk_score_col,
    time_col,
    event_col,
    n_groups=2,
    return_details=True,
    show_counts=False,
    plot_km=True,
    km_figsize=(8, 8),
    cmap="tab10",
    metadata=None,
    percentiles=None,
    verbose=False,
    suppress_convergence_warnings=True,
    organ_name=None,
    save_dir=None,
    ax=None,
):
    """
    Split train by quantiles into n_groups, apply same cutpoints to test,
    run multivariate log-rank on train and test, and optionally plot KM curves.

    Computes hazard ratios (HR) with 95% CI and p-values for each group
    vs the *medium-risk group* (middle quantile group) using CoxPH.

    - If metadata contains "group_labels", those are used for labeling groups
      (must match the number of groups actually used). Otherwise deterministic
      generated labels are used.
    - If metadata contains "group_colors" it will be used for plotting (optional).
    """
    # --- Data prep ---
    required = [risk_score_col, time_col, event_col]

    tr = train_df[required].dropna().copy()
    te = test_df[required].dropna().copy()
    if tr.shape[0] == 0 or te.shape[0] == 0:
        raise ValueError("Empty train or test after dropping NA in required columns.")

    # Split-specific C-index (train on train rows, test on test rows)
    train_cindex = concordance_index(tr[time_col], -tr[risk_score_col], tr[event_col])
    test_cindex = concordance_index(te[time_col], -te[risk_score_col], te[event_col])

    # --- Define quantile bins on train ---
    if percentiles is None:
        percentiles = np.linspace(0, 1, n_groups + 1)
        if verbose:
            print("Percentiles not provided, using:", percentiles)
    else:
        if verbose:
            print("Using provided percentiles:", percentiles)

    edges = np.unique(tr[risk_score_col].quantile(percentiles).values)
    if edges.size < 3:
        med = tr[risk_score_col].median()
        bins = np.array([-np.inf, med, np.inf])
    else:
        interior = edges[1:-1]
        bins = np.concatenate(([-np.inf], interior, [np.inf])) if interior.size > 0 else np.array([-np.inf, tr[risk_score_col].median(), np.inf])

    labels = list(range(1, len(bins)))
    if len(labels) < 2:
        raise ValueError("Unable to create at least two groups for log-rank comparison.")

    # --- Assign groups ---
    tr_groups = pd.cut(tr[risk_score_col], bins=bins, labels=labels, include_lowest=True).astype(int)
    te_groups = pd.cut(te[risk_score_col], bins=bins, labels=labels, include_lowest=True).astype(int)

    if show_counts and verbose:
        print("Train group counts:\n", tr_groups.value_counts().sort_index())
        print("Test group counts:\n", te_groups.value_counts().sort_index())

    # --- Run logrank tests ---
    res_train = multivariate_logrank_test(tr[time_col], tr_groups, tr[event_col])
    res_test = multivariate_logrank_test(te[time_col], te_groups, te[event_col])
    n_groups_used = len(labels)
    df_train = getattr(res_train, "degrees_freedom", max(n_groups_used - 1, 1))
    df_test = getattr(res_test, "degrees_freedom", max(n_groups_used - 1, 1))

    # --- Group labels ---
    if metadata and "group_labels" in metadata:
        group_labels = list(metadata["group_labels"])
        if len(group_labels) != n_groups_used:
            raise ValueError(f"metadata['group_labels'] must have {n_groups_used} elements.")
    else:
        if n_groups_used == 2:
            group_labels = ["Low risk", "High risk"]
        else:
            group_labels = [f"G{i}" for i in labels]

    label_map = dict(zip(labels, group_labels))
    ordered_group_labels = [label_map[i] for i in labels]

    tr_labeled = tr_groups.map(label_map)
    te_labeled = te_groups.map(label_map)

    # --- Outputs ---
    out = {
        "n_groups_requested": n_groups,
        "n_groups_used": n_groups_used,
        "group_labels": ordered_group_labels,
        "train": {
            "test_statistic": float(res_train.test_statistic),
            "p_value": float(res_train.p_value),
            "degrees_freedom": int(df_train),
            "c_index": float(train_cindex),
        },
        "test": {
            "test_statistic": float(res_test.test_statistic),
            "p_value": float(res_test.p_value),
            "degrees_freedom": int(df_test),
            "c_index": float(test_cindex),
        },
        "cutpoints": bins,
    }

    tr_out, te_out = tr.assign(_group=tr_labeled), te.assign(_group=te_labeled)
    if return_details or plot_km:
        out["train_details"], out["test_details"] = tr_out, te_out

    # --- Helper: compute HR vs ref group ---
    def _compute_hr_vs_medium(df_with_label):
        df = df_with_label[[time_col, event_col, "_group"]].copy()
        df["_group"] = pd.Categorical(df["_group"], categories=ordered_group_labels, ordered=True)

        ref_group = ordered_group_labels[0]

        dummies = pd.get_dummies(df["_group"], prefix="_grp")
        ref_col = f"_grp_{ref_group}"
        if ref_col not in dummies.columns:
            return pd.DataFrame(columns=["hr", "ci_lower", "ci_upper", "p_value"]), ref_group

        dummies = dummies.drop(columns=[ref_col], errors="ignore")
        if dummies.shape[1] == 0:
            return pd.DataFrame(columns=["hr", "ci_lower", "ci_upper", "p_value"]), ref_group

        cox_df = pd.concat([df[[time_col, event_col]], dummies], axis=1)
        cph = CoxPHFitter(penalizer=0.01)

        if suppress_convergence_warnings:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", UserWarning)
                cph.fit(cox_df, duration_col=time_col, event_col=event_col, show_progress=False)
        else:
            cph.fit(cox_df, duration_col=time_col, event_col=event_col, show_progress=False)

        summary = cph.summary
        results = []
        for dummy_col in dummies.columns:
            target_label = dummy_col.split("_grp_", 1)[1] if dummy_col.startswith("_grp_") else dummy_col
            if dummy_col in summary.index:
                row = summary.loc[dummy_col]
                hr = float(row["exp(coef)"])
                ci_lower = float(row["exp(coef) lower 95%"])
                ci_upper = float(row["exp(coef) upper 95%"])
                pval = float(row["p"])
                results.append({"group": target_label, "hr": hr, "ci_lower": ci_lower, "ci_upper": ci_upper, "p_value": pval})

        if not results:
            return pd.DataFrame(columns=["hr", "ci_lower", "ci_upper", "p_value"]), ref_group

        res_df = pd.DataFrame(results).set_index("group")
        desired_order = [g for g in ordered_group_labels if g != ref_group and g in res_df.index]
        res_df = res_df.loc[desired_order] if len(desired_order) > 0 else res_df
        return res_df, ref_group

    # compute HRs once for train and test
    hr_train_df, ref_train = _compute_hr_vs_medium(tr_out)
    hr_test_df, ref_test = _compute_hr_vs_medium(te_out)
    out["train"]["hr_comparisons"] = hr_train_df.to_dict(orient="index")
    out["test"]["hr_comparisons"] = hr_test_df.to_dict(orient="index")
    out["train_reference_group"] = ref_train
    out["test_reference_group"] = ref_test

    if verbose:
        def _print_hr_df(df, ref_group, title):
            print(f"\n{title}")
            if df.shape[0] == 0:
                print("  (no comparisons - only reference group present)")
                return
            for grp, row in df.iterrows():
                print(
                    f"  {grp} vs {ref_group}: HR = {row['hr']:.3f} "
                    f"(95% CI {row['ci_lower']:.3f} - {row['ci_upper']:.3f}), p = {row['p_value']:.3g}"
                )

        _print_hr_df(hr_train_df, ref_train, "Hazard Ratios (Train) -- each vs ref group")
        _print_hr_df(hr_test_df, ref_test, "Hazard Ratios (Test)  -- each vs ref group")

    # --- KM plots ---
    if plot_km:
        blue, red = "#0921ff", "#ff001e"
        cmap_blue_red = mpl.colors.LinearSegmentedColormap.from_list("blue_red", [blue, red])

        def _color_for_index(idx):
            return cmap_blue_red(idx / (n_groups_used - 1)) if n_groups_used > 1 else cmap_blue_red(0.5)

        def _plot_with_at_risk(df_with_group, title, ax=None):
            if ax is None:
                fig, ax = plt.subplots(figsize=km_figsize, dpi=300)
            else:
                fig = ax.figure
            kmf_list = []
            for i, glabel in enumerate(ordered_group_labels):
                grp = df_with_group[df_with_group["_group"] == glabel]
                if len(grp) == 0:
                    continue
                kmf = KaplanMeierFitter()
                kmf.fit(durations=grp[time_col], event_observed=grp[event_col], label=f"G{i+1}")
                color = metadata.get("group_colors")[i] if metadata and "group_colors" in metadata and i < len(metadata["group_colors"]) else _color_for_index(i)
                kmf.plot_survival_function(ax=ax, ci_show=True, color=color, alpha=0.9)
                kmf_list.append(kmf)
            ax.set_title(title)
            ax.set_xlabel("Time (years)", fontsize=16)
            ax.set_ylabel("Survival probability",fontsize=16)
            ax.tick_params(axis='both', labelsize=14)
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=16, frameon=False)
            ax.grid(alpha=0.6)
            if kmf_list:
                add_at_risk_counts(*kmf_list, ax=ax, fontsize=16)
            #plt.tight_layout()
            return fig, ax

        def _format_pvalue(p):
            return "<0.001" if p < 0.001 else np.round(p, 3)

        def _add_hr_annotation(ax, hr_df, c_index):
            """
            Add C-index and HR with CI annotation to KM plot.
            """
            annotation_y = 0.10
            lines = [f"C-index: {c_index:.4f}"]

            if hr_df is None or hr_df.empty:
                annotation_text = "\n".join(lines)
                ax.text(
                    0.02,
                    annotation_y,
                    annotation_text,
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=16,
                    bbox=dict(
                        boxstyle="round",
                        facecolor="white",
                        alpha=0.85
                    )
                )
                return

            for grp, row in hr_df.iterrows():
                lines.append(
                    f"HR {row['hr']:.2f} "
                    f"({row['ci_lower']:.2f}-{row['ci_upper']:.2f})"
                )

            # append log-rank p={_format_pvalue(out['test']['p_value'])} to annotation
            p_val = hr_df["p_value"].min() if "p_value" in hr_df.columns else None
            if p_val is not None:
                lines.append(f"log-rank p={_format_pvalue(p_val)}")

            annotation_text = "\n".join(lines)

            ax.text(
                0.02,
                annotation_y,
                annotation_text,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=16,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.85
                )
            )

        title = metadata.get("title", "KM by quantile group") if metadata else "KM by quantile group"
        organ_txt = f"{organ_name} - " if organ_name else ""


        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=300)

        _add_hr_annotation(ax, hr_test_df, out["test"]["c_index"])
        _plot_with_at_risk(
            te_out,
            f"{organ_txt}",
            ax=ax
        )

        if save_dir is not None:
            prefix = organ_name if organ_name else "km"
            km_path = os.path.join(save_dir, f"{prefix}_km.png")

            fig.savefig(km_path, dpi=300, bbox_inches="tight")

            out["plot_path"] = km_path

            print(f"Saved: {km_path}")

        if ax is None:
            plt.show()

    return out