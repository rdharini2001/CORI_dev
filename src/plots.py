from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test


def save_figure(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def km_plot(df: pd.DataFrame, group_column: str, labels: dict, title: str, path: str | Path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fitter = KaplanMeierFitter()

    for value, label in labels.items():
        subset = df.loc[df[group_column] == value]
        fitter.fit(subset["time_years"], subset["Y_mace"], label=label)
        fitter.plot_survival_function(ax=ax, ci_show=True)

    values = list(labels)
    if len(values) == 2:
        a = df.loc[df[group_column] == values[0]]
        b = df.loc[df[group_column] == values[1]]
        result = logrank_test(
            a["time_years"],
            b["time_years"],
            event_observed_A=a["Y_mace"],
            event_observed_B=b["Y_mace"],
        )
        ax.text(0.03, 0.05, f"Log-rank p={result.p_value:.3g}", transform=ax.transAxes)

    ax.set(title=title, xlabel="Years since retinal imaging", ylabel="MACE-free survival")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, path)
    return fig


def forest_plot(df: pd.DataFrame, label: str, estimate: str, lower: str, upper: str, title: str, path: str | Path, reference: float = 1.0):
    plot_df = df.dropna(subset=[estimate, lower, upper]).copy().reset_index(drop=True)
    y = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(7, max(3, 0.42 * len(plot_df) + 1.5)))
    ax.errorbar(
        plot_df[estimate],
        y,
        xerr=[plot_df[estimate] - plot_df[lower], plot_df[upper] - plot_df[estimate]],
        fmt="o",
        capsize=3,
    )
    ax.axvline(reference, linestyle="--", linewidth=1)
    ax.set_yticks(y, plot_df[label])
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel("Hazard ratio (95% CI)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, path)
    return fig


def reclassification_heatmap(table: pd.DataFrame, title: str, path: str | Path, percent: bool = True):
    values = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(values, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            text = f"{100 * values[row, column]:.1f}%" if percent else f"{values[row, column]:.0f}"
            ax.text(column, row, text, ha="center", va="center")
    ax.set_xticks(np.arange(table.shape[1]), table.columns)
    ax.set_yticks(np.arange(table.shape[0]), table.index)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    save_figure(fig, path)
    return fig


def calibration_plot(df: pd.DataFrame, risk_column: str, event_column: str, title: str, path: str | Path, bins: int = 10):
    data = df[[risk_column, event_column]].dropna().copy()
    data["bin"] = pd.qcut(data[risk_column], q=bins, duplicates="drop")
    calibration = data.groupby("bin", observed=True).agg(
        predicted=(risk_column, "mean"),
        observed=(event_column, "mean"),
        N=(event_column, "size"),
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.plot(calibration["predicted"], calibration["observed"], marker="o")
    ax.set(xlabel="Predicted 10-year risk", ylabel="Observed 10-year event rate", title=title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, path)
    return calibration, fig


def learning_curve_plot(table: pd.DataFrame, event_column: str, performance_column: str, reference: float, title: str, path: str | Path):
    summary = table.groupby(event_column)[performance_column].agg(
        median="median",
        lower=lambda x: x.quantile(0.025),
        upper=lambda x: x.quantile(0.975),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(summary[event_column], summary["median"], marker="o")
    ax.fill_between(summary[event_column], summary["lower"], summary["upper"], alpha=0.2)
    ax.axhline(reference, linestyle="--", label="CORI reference")
    ax.set(xlabel="Training events", ylabel="Held-out C-index", title=title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, path)
    return summary, fig


def rank_transition(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str, top_n: int, title: str, path: str | Path):
    left_rank = left.reset_index(drop=True).reset_index().rename(columns={"index": "left_rank"})
    right_rank = right.reset_index(drop=True).reset_index().rename(columns={"index": "right_rank"})
    left_rank["left_rank"] += 1
    right_rank["right_rank"] += 1
    merged = left_rank[["feature", "left_rank"]].merge(
        right_rank[["feature", "right_rank"]], on="feature", how="outer"
    )
    selected = merged.loc[(merged["left_rank"] <= top_n) | (merged["right_rank"] <= top_n)].copy()
    selected = selected.fillna(top_n + 1).sort_values(["left_rank", "right_rank"])

    fig, ax = plt.subplots(figsize=(8, max(5, 0.25 * len(selected) + 2)))
    for _, row in selected.iterrows():
        ax.plot([0, 1], [row["left_rank"], row["right_rank"]], linewidth=1)
        ax.text(-0.03, row["left_rank"], row["feature"], ha="right", va="center", fontsize=7)
        ax.text(1.03, row["right_rank"], row["feature"], ha="left", va="center", fontsize=7)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylim(top_n + 1.5, 0.5)
    ax.set_xticks([0, 1], [left_name, right_name])
    ax.set_ylabel("Feature rank")
    ax.set_title(title)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.tight_layout()
    save_figure(fig, path)
    return selected, fig
