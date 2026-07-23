import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from tqdm import tqdm


class FeatureResidualizer(BaseEstimator, TransformerMixin):
    """
    Residualize features with respect to clinical covariates.

    A separate regression model is fit for each feature using the training
    cohort and then applied unchanged to future cohorts.
    """

    def __init__(self, model=None):
        self.model = LinearRegression() if model is None else model

    def fit(self, X, covariates):
        X = np.asarray(X)
        C = np.asarray(covariates)

        if C.ndim == 1:
            C = C[:, None]

        self.models_ = []

        for j in tqdm(range(X.shape[1]), desc="Fitting residualization models"):
            model = clone(self.model)
            model.fit(C, X[:, j])
            self.models_.append(model)

        return self

    def transform(self, X, covariates):
        if not hasattr(self, "models_"):
            raise RuntimeError("Must call fit() before transform().")

        X = np.asarray(X)
        C = np.asarray(covariates)

        if C.ndim == 1:
            C = C[:, None]

        X_resid = np.empty_like(X, dtype=float)

        for j, model in enumerate(self.models_):
            X_resid[:, j] = X[:, j] - model.predict(C)

        return X_resid

    def fit_transform(self, X, covariates):
        return self.fit(X, covariates).transform(
            X, covariates
        )
 

    def plot_residualization(self, X, X_resid, covariates):
        """
        Compare feature-covariate correlations before and after residualization.
        Clustering is learned from the pre-residualization matrix and applied
        unchanged to the residualized matrix.
        """

        X = pd.DataFrame(X).reset_index(drop=True)
        X_resid = pd.DataFrame(X_resid).reset_index(drop=True)
        C = pd.DataFrame(covariates).reset_index(drop=True)

        corr_before = pd.DataFrame(
            index=X.columns,
            columns=C.columns,
            dtype=float,
        )

        corr_after = corr_before.copy()

        for f in X.columns:
            for c in C.columns:
                corr_before.loc[f, c] = X[f].corr(C[c])
                corr_after.loc[f, c] = X_resid[f].corr(C[c])

        # Absolute correlations for clustering
        cluster_matrix = np.abs(corr_before).fillna(0)

        # Learn clustering on original data
        row_order = leaves_list(
            linkage(cluster_matrix.values, method="average")
        )

        col_order = leaves_list(
            linkage(cluster_matrix.values.T, method="average")
        )

        # Apply learned ordering to both matrices
        corr_before_ordered = corr_before.iloc[
            row_order, col_order
        ]

        corr_after_ordered = corr_after.iloc[
            row_order, col_order
        ]

        # Same color scale
        vmax = np.nanmax(np.abs(corr_before.values))

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 10),
        )

        sns.heatmap(
            np.abs(corr_before_ordered),
            ax=axes[0],
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            xticklabels=True,
            yticklabels=False,
        )
        axes[0].set_title("Before residualization")
        axes[0].set_xlabel("Clinical covariates")
        axes[0].set_ylabel("Features")

        sns.heatmap(
            np.abs(corr_after_ordered),
            ax=axes[1],
            cmap="viridis",
            vmin=0,
            vmax=vmax,
            xticklabels=True,
            yticklabels=False,
        )
        axes[1].set_title("After residualization")
        axes[1].set_xlabel("Clinical covariates")
        axes[1].set_ylabel("")

        plt.tight_layout()

        print(
            f"Mean |corr| before: "
            f"{np.nanmean(np.abs(corr_before.values)):.3f}"
        )
        print(
            f"Mean |corr| after : "
            f"{np.nanmean(np.abs(corr_after.values)):.3f}"
        )