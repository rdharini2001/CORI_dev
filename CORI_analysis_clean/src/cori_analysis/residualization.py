from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def residualize_against_mmace(df: pd.DataFrame, fit_on: pd.DataFrame | None = None):
    """Regress CORI_score on MMACE_score and z-standardize the residual."""
    fit_df = fit_on if fit_on is not None else df
    model = smf.ols('CORI_score ~ MMACE_score', data=fit_df).fit()
    prediction = model.predict(df[['MMACE_score']])
    residual = df['CORI_score'] - prediction
    residual_z = (residual - residual.mean()) / residual.std(ddof=1)
    return residual_z, model
