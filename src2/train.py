
from lifelines import CoxPHFitter
import numpy as np
from lifelines.utils import concordance_index


# Third-party
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import forestplot as fp
from src2.plot import compare_quantile_logrank

def train_cox_model(
    train_df,
    test_df,
    selected_features,
    time_col="time",
    event_col="event",
    organ_name="organ",
    save_dir=None,
    plot_km=True,
    plot_groups=[2],
    verbose=True,
    percentiles=None,
    penalizer=None
):
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    if verbose:
        print(" train shape:", train_df.shape)
        print(" test shape:", test_df.shape)

    # -------------------------------
    # 1. Fit Cox model
    # -------------------------------
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(
        train_df[[time_col, event_col] + selected_features],
        duration_col=time_col,
        event_col=event_col
    )

    # -------------------------------
    # 2. Risk scores
    # -------------------------------
    train_df["risk_score"] = cph.predict_log_partial_hazard(train_df[selected_features])
    test_df["risk_score"] = cph.predict_log_partial_hazard(test_df[selected_features])
    
    # zscore normalization of risk scores
    train_df["risk_score"] = (train_df["risk_score"] - train_df["risk_score"].mean()) / train_df["risk_score"].std()
    test_df["risk_score"] = (test_df["risk_score"] - test_df["risk_score"].mean()) / test_df["risk_score"].std()

    coef = cph.params_.sort_values()
    plt.figure(figsize=(6, 4))
    coef.plot(kind="barh")
    plt.axvline(0, linestyle="--")
    plt.title("Cox Coefficients")
    plt.tight_layout()
    if verbose:
        plt.show()
    else:
        plt.close()
        
    
    forest_df = cph.summary.reset_index().rename(
        columns={
            "covariate": "covariate",
            "exp(coef)": "hr",
            "exp(coef) lower 95%": "ci_low",
            "exp(coef) upper 95%": "ci_high",
            "p": "p_value",})
    forest_df = forest_df[["covariate", "hr", "ci_low", "ci_high", "p_value"]].copy()
    forest_df["p_text"] = forest_df["p_value"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")

    forest_df = forest_df.replace([np.inf, -np.inf], np.nan)
    forest_df = forest_df.dropna(subset=["hr", "ci_low", "ci_high", "p_value"])

    forest_df["label"] = forest_df["covariate"]
    forest_df.columns.tolist() 

    # -----------------------------
    # 8. Plot
    # -----------------------------
    # xmin   = 0.4
    # xmax   = 2
    # xticks = np.linspace(0, 2, num=6).tolist()
    fp.forestplot(
        forest_df,
        estimate="hr",
        ll="ci_low",
        hl="ci_high",
        varlabel="label",
        pval="p_value",
        rightannote=["p_text"],
        right_annoteheaders=["p-value"],
        color_alt_rows=True,
        # xlim=(xmin, xmax),
        # xticks=xticks,
        # ci_report=False,
        flush=True,
        xlabel="Hazard Ratio (95% CI)",
        figsize=(12, max(3, len(forest_df) * 0.45)),
    )

    plt.axvline(1.0, color="black", linestyle="-", linewidth=0.8, alpha=0.8)
    plt.tight_layout()
    if verbose:
        plt.show()
    else:
        plt.close()

    results = None
    for group_count in plot_groups:
        results = compare_quantile_logrank(
            train_df,
            test_df,
            risk_score_col="risk_score",
            time_col=time_col,
            event_col=event_col,
            n_groups=group_count,
            plot_km=plot_km,
            verbose=verbose,
            km_figsize=(1, 1),
            percentiles=percentiles
            
        )
    

    return {
        "cox_model": cph,
        "train_df": train_df,
        "test_df": test_df,
        "forest_df": forest_df,
        "results"  : results
    }



def do_multivar(
    df,
    selected_features,
    time_col="time",
    event_col="event",
    label_map=None,
    verbose=True,
):

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        df[[time_col, event_col] + selected_features],
        duration_col=time_col,
        event_col=event_col
    )
    
    
    forest_df = cph.summary.reset_index().rename(
        columns={
            "covariate": "covariate",
            "exp(coef)": "hr",
            "exp(coef) lower 95%": "ci_low",
            "exp(coef) upper 95%": "ci_high",
            "p": "p_value",})
    forest_df = forest_df[["covariate", "hr", "ci_low", "ci_high", "p_value"]].copy()
    forest_df["p_text"] = forest_df["p_value"].apply(
        lambda p: "<0.001" if p < 0.001 else f"{p:.3f}")

    forest_df = forest_df.replace([np.inf, -np.inf], np.nan)
    forest_df = forest_df.dropna(subset=["hr", "ci_low", "ci_high", "p_value"])
    
    forest_df["label"] = forest_df["covariate"].apply(lambda c: label_map.get(c, c.replace("_", " ").title()))
    forest_df.columns.tolist() 

    # -----------------------------
    # 8. Plot
    # -----------------------------
    # xmin   = 0.4
    # xmax   = 2
    # xticks = np.linspace(0, 2, num=6).tolist()
    fp.forestplot(
        forest_df,
        estimate="hr",
        ll="ci_low",
        hl="ci_high",
        varlabel="label",
        pval="p_value",
        rightannote=["p_text"],
        right_annoteheaders=["p-value"],
        color_alt_rows=True,
        # xlim=(xmin, xmax),
        # xticks=xticks,
        # ci_report=False,
        flush=True,
        xlabel="Hazard Ratio (95% CI)",
        figsize=(12, max(3, len(forest_df) * 0.45)),
    )

    plt.axvline(1.0, color="black", linestyle="-", linewidth=0.8, alpha=0.8)
    plt.tight_layout()
    
    return forest_df
