
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def nri_heatmap(df,
                new_col,
                event_col='MACE_event',
                baseline_col='ASCVD_risk',
                n_quantiles=2):

    # --------------------------------------------------------
    # Compute common category cutoffs
    # --------------------------------------------------------
    def get_bins(scores):
        qs = [scores.quantile(i / n_quantiles)
              for i in range(1, n_quantiles)]
        return [-np.inf] + qs + [np.inf]

    base_bins = get_bins(df[baseline_col])
    new_bins = get_bins(df[new_col])

    order = list(range(1, n_quantiles + 1))

    def categorize(series, bins):
        return pd.cut(
            series,
            bins=bins,
            labels=order,
            include_lowest=True
        )

    # --------------------------------------------------------
    # Helper: build reclassification matrix
    # --------------------------------------------------------
    def get_matrix(subset):

        base = categorize(subset[baseline_col], base_bins)
        new = categorize(subset[new_col], new_bins)

        valid = (~base.isna()) & (~new.isna())

        base = base[valid]
        new = new[valid]

        matrix = pd.crosstab(base, new)

        matrix = matrix.reindex(
            index=order,
            columns=order,
            fill_value=0
        )

        return matrix

    event_matrix = get_matrix(df[df[event_col] == 1])
    nonevent_matrix = get_matrix(df[df[event_col] == 0])

    # --------------------------------------------------------
    # Standard category NRI
    # --------------------------------------------------------

    def compute_nri(subset, event=True):

        base = categorize(subset[baseline_col], base_bins)
        new = categorize(subset[new_col], new_bins)

        valid = (~base.isna()) & (~new.isna())

        base = base[valid].cat.codes
        new = new[valid].cat.codes

        up = (new > base).sum()
        down = (new < base).sum()
        same = (new == base).sum()

        n = len(base)

        if event:
            nri = (up - down) / n
        else:
            nri = (down - up) / n

        return nri, up, down, same, n

    event_nri, e_up, e_down, e_same, e_n = compute_nri(
        df[df[event_col] == 1],
        event=True
    )

    nonevent_nri, ne_up, ne_down, ne_same, ne_n = compute_nri(
        df[df[event_col] == 0],
        event=False
    )

    total_nri = event_nri + nonevent_nri

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    titles = ["Events", "Non-events", "Overall"]

    matrices = [
        event_matrix,
        nonevent_matrix,
        event_matrix + nonevent_matrix
    ]

    scores = [event_nri, nonevent_nri, total_nri]

    cmaps = ["Blues", "Greens", "Purples"]

    for ax, title, matrix, score, cmap in zip(
        axes, titles, matrices, scores, cmaps
    ):

        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=False,
            ax=ax
        )

        ax.set_title(
            f"{title}\nNet contribution = {score:.3f}",
            fontsize=12
        )

        ax.set_xlabel(new_col)
        ax.set_ylabel(baseline_col)

    plt.tight_layout()
    plt.show()

    print("=" * 60)
    print("Category-based Net Reclassification Improvement")
    print("=" * 60)

    print(f"\nEvents (N={e_n})")
    print(f"  Up   : {e_up} ({e_up/e_n:.3f})")
    print(f"  Down : {e_down} ({e_down/e_n:.3f})")
    print(f"  Same : {e_same} ({e_same/e_n:.3f})")
    print(f"  Event NRI = {event_nri:.4f}")

    print(f"\nNon-events (N={ne_n})")
    print(f"  Down : {ne_down} ({ne_down/ne_n:.3f})")
    print(f"  Up   : {ne_up} ({ne_up/ne_n:.3f})")
    print(f"  Same : {ne_same} ({ne_same/ne_n:.3f})")
    print(f"  Non-event NRI = {nonevent_nri:.4f}")

    print("\n-----------------------------------------------")
    print(f"Overall NRI = {total_nri:.4f}")
    print("-----------------------------------------------")

    return {
        "Event NRI": event_nri,
        "Non-event NRI": nonevent_nri,
        "Overall NRI": total_nri,
        "Event Matrix": event_matrix,
        "Non-event Matrix": nonevent_matrix
    }