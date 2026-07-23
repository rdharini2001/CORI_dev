import argparse
import glob
import os
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.exceptions import ConvergenceWarning
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import concordance_index
from sklearn.linear_model import LassoCV


def parse_args(
    default_feature_dir,
    default_risk_scores_dir,
    default_clinical_sheet,
    default_output_dir,
    default_organs,
):
    parser = argparse.ArgumentParser(
        description="Organ-agnostic feature selection + multivariable Cox forest pipeline"
    )
    parser.add_argument(
        "--feat_dir",
        type=str,
        default=default_feature_dir,
        help="Directory containing *_all_features.parquet files",
    )
    parser.add_argument(
        "--risk_scores_dir",
        type=str,
        default=default_risk_scores_dir,
        help="Directory to save/read organ risk score csv files",
    )
    parser.add_argument(
        "--clinical_sheet",
        type=str,
        default=default_clinical_sheet,
        help="Path to clinical csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=default_output_dir,
        help="Directory to save aggregate outputs",
    )
    parser.add_argument(
        "--organ_list",
        nargs="+",
        default=default_organs,
        help="Organs to run",
    )
    parser.add_argument("--max_years", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corr_threshold", type=float, default=0.95)
    parser.add_argument("--var_threshold", type=float, default=0.001)
    parser.add_argument("--n_folds", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--bootstrap_iterations", type=int, default=20)
    parser.add_argument("--bootstrap_percent", type=float, default=0.80)
    parser.add_argument("--n_groups", type=int, default=2)
    parser.add_argument("--skip_km", action="store_true")

    args, _ = parser.parse_known_args()
    return args


def extract_organ_name_from_filename(file_path):
    """Extract organ name from known score filename patterns."""
    name = Path(file_path).stem

    if name.startswith("risk_scores_"):
        return name.replace("risk_scores_", "", 1)
    if name.endswith("_risk_scores"):
        return name.replace("_risk_scores", "", 1)
    if name.endswith("_features_wide_all"):
        return name.replace("_features_wide_all", "", 1)
    if "_features_wide_" in name:
        return name.split("_features_wide_", 1)[0]
    if name.endswith("_all_features"):
        return name.replace("_all_features", "", 1)

    return name


def discover_organ_feature_files(feature_dir):
    """Return {organ: parquet_path} across supported feature parquet naming patterns."""
    patterns = [
        "*_all.parquet",
        "*_all_features.parquet",
        "*_features_wide_all.parquet",
        "*_features_wide_*.parquet",
    ]

    feature_files = []
    for pat in patterns:
        feature_files.extend(glob.glob(os.path.join(feature_dir, pat)))

    # Preserve deterministic order and keep first match per organ.
    files_by_organ = {}
    for p in sorted(set(feature_files)):
        organ = extract_organ_name_from_filename(p)
        files_by_organ.setdefault(organ, p)

    return files_by_organ


def generate_time_to_event_labels(
    temp_df,
    scan_date_col,
    event_date_col,
    lost_to_followup_col,
    death_date_col,
    event_col_name="event",
    time_col_name="time",
):
    import pandas as pd
 
    def parse_date(date):
        return pd.to_datetime(date, errors='coerce')
   
    df = temp_df.copy()
 
    # Parse dates
    df['_scan_date'] = df[scan_date_col].apply(parse_date)
    df['_event_date'] = df[event_date_col].apply(parse_date)
    df['_lost_to_followup'] = df[lost_to_followup_col].apply(parse_date) if lost_to_followup_col in df else pd.NaT
    df['_death_date'] = df[death_date_col].apply(parse_date) if death_date_col in df else pd.NaT
 
    # Calculate last followup date as minimum of lost_to_followup and death_date
    df['_last_followup'] = df[['_lost_to_followup', '_death_date']].min(axis=1)
    # Fill nan values as max value of lost_to_followup and death_date
    df['_last_followup'] = df['_last_followup'].fillna(df[['_lost_to_followup', '_death_date']].max().max())
   
    # print df shape
    print("Generating time-to-event labels for", df.shape[0], "records.")
 
    # Event: 1 if event_date exists, else 0
    df[event_col_name] = (~df['_event_date'].isna()).astype(int)
   
    print("Event counts:\n", df[event_col_name].value_counts())
 
    # Time: If event, time to event_date; else, time to last followup
    def calc_time(row):
        if row[event_col_name] == 1:
            return (row['_event_date'] - row['_scan_date']).days
        else:
            return (row['_last_followup'] - row['_scan_date']).days

    df[time_col_name] = df.apply(calc_time, axis=1) / 365.25
    # keep only records with non-negative time
    df = df[df[time_col_name] >= 0].copy()
    print("Records with non-negative time:", df.shape[0])   
    # print final event counts
    print("Final event counts:\n", df[event_col_name].value_counts())   
    # print shape and describe time column
    print("Final shape:", df.shape)
 
    # Clean up temp columns
    df.drop(['_scan_date', '_event_date', '_lost_to_followup', '_death_date', '_last_followup'], axis=1, inplace=True)
    return df

def censor_time_to_event(df, time_col='time', event_col='event', max_years=10):
    """
    Censor time-to-event data at a maximum follow-up time (in years).
    If time > max_years, set time = max_years and event = 0 (censored).
    Returns a copy of the dataframe with new columns: time_censored, event_censored.
    """
   
    print("Censornig data to max years:", max_years)
    df = df.copy()
    max_days = int(max_years * 365.25) 
    df['time'] = df[time_col].clip(upper=max_days)
    df['event'] = df[event_col]
    # If censored due to max_days, set event to 0
    df.loc[df[time_col] > max_days, 'event'] = 0
    return df

def set_seeds(seed=42):
    import random

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def residualize_features(
    train_df,
    test_df,
    feature_cols,
    covariate_cols=("age", "sex", "hypertension", "height", "BMI"),
    verbose=False,
):
    """Regress each feature on covariates and replace it with the residual.

    This removes the linear effect of demographic/clinical covariates (e.g.
    sex, age, hypertension, height, BMI) from each organ-fat feature, so that
    downstream feature selection targets organ-specific signal rather than
    signal explained by those covariates. Covariate coefficients are fit on
    train only and applied to test to avoid leakage.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    missing_cov = [c for c in covariate_cols if c not in train_df.columns or c not in test_df.columns]
    if missing_cov:
        raise ValueError(f"Missing covariate columns for residualization: {missing_cov}")

    def _encode_covariates(df):
        cov = pd.DataFrame(index=df.index)
        for c in covariate_cols:
            col = df[c]
            if pd.api.types.is_bool_dtype(col):
                cov[c] = col.astype(float)
            elif col.dtype == object or isinstance(col.dtype, pd.CategoricalDtype):
                dummies = pd.get_dummies(col, prefix=c, drop_first=True)
                cov = pd.concat([cov, dummies.astype(float)], axis=1)
            else:
                cov[c] = col.astype(float)
        return cov

    train_cov = _encode_covariates(train_df)
    test_cov = _encode_covariates(test_df).reindex(columns=train_cov.columns, fill_value=0.0)

    cov_medians = train_cov.median()
    train_cov = train_cov.fillna(cov_medians)
    test_cov = test_cov.fillna(cov_medians)

    train_design = np.column_stack([np.ones(len(train_cov)), train_cov.values])
    test_design = np.column_stack([np.ones(len(test_cov)), test_cov.values])

    numeric_feature_cols = [
        c
        for c in feature_cols
        if c in train_df.columns and c in test_df.columns and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    n_residualized = 0
    for col in numeric_feature_cols:
        y_train = train_df[col].astype(float)
        valid = y_train.notna().values
        if valid.sum() < train_design.shape[1] + 1:
            continue

        beta, *_ = np.linalg.lstsq(train_design[valid], y_train.values[valid], rcond=None)

        train_df[col] = train_df[col].astype(float) - train_design @ beta
        test_df[col] = test_df[col].astype(float) - test_design @ beta
        n_residualized += 1

    if verbose:
        print(f"Residualized {n_residualized}/{len(numeric_feature_cols)} features against covariates: {list(covariate_cols)}")

    return train_df, test_df


def preprocess_and_select_features(
    train_df,
    test_df,
    feature_cols,
    time_col="time",
    event_col="event",
    corr_threshold=0.95,
    var_threshold=1e-10,
    n_folds=5,
    top_k=5,
    bootstrap_iterations=100,
    bootstrap_percent=0.7,
    verbose=True,
    make_plots=True,
    random_state=None,
):
    train_df = train_df.copy()
    test_df = test_df.copy()

    if verbose:
        print("Initial train shape:", train_df.shape)
        print("Initial test shape:", test_df.shape)

    candidate = [
        c
        for c in feature_cols
        if c in train_df.columns and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    if verbose:
        print("Initial features:", len(candidate))

    candidate = [c for c in candidate if c in test_df.columns]
    if len(candidate) == 0:
        raise ValueError("No shared numeric feature columns found between train and test dataframes.")

    # numpy>=2 can error on quantiles of bool dtype; cast bool features to numeric early.
    bool_cols = [
        c
        for c in candidate
        if pd.api.types.is_bool_dtype(train_df[c]) or pd.api.types.is_bool_dtype(test_df[c])
    ]
    if bool_cols:
        train_df[bool_cols] = train_df[bool_cols].astype(float)
        test_df[bool_cols] = test_df[bool_cols].astype(float)

    missing_frac = train_df[candidate].isna().mean()
    dropped_feature_cols = missing_frac[missing_frac > 0.10].index.tolist()
    candidate = [c for c in candidate if c not in dropped_feature_cols]

    if len(candidate) == 0:
        raise ValueError("All candidate features were removed by the >10% missingness filter.")

    if verbose:
        print("Dropped features (>10% missing):", len(dropped_feature_cols))
        print("Remaining features:", len(candidate))

    train_df = train_df.dropna(subset=[time_col, event_col]).copy()
    test_df = test_df.dropna(subset=[time_col, event_col]).copy()

    if verbose:
        print("Train shape after dropping missing time/event:", train_df.shape)
        print("Test shape after dropping missing time/event:", test_df.shape)

    train_missing_per_row = train_df[candidate].isna().mean(axis=1)
    test_missing_per_row = test_df[candidate].isna().mean(axis=1)
    train_df = train_df[train_missing_per_row <= 0.80].copy()
    test_df = test_df[test_missing_per_row <= 0.80].copy()

    if verbose:
        print("Train shape after dropping patients with >80% missing:", train_df.shape)
        print("Test shape after dropping patients with >80% missing:", test_df.shape)

    impute_medians = train_df[candidate].median()
    train_df[candidate] = train_df[candidate].fillna(impute_medians)
    test_df[candidate] = test_df[candidate].fillna(impute_medians)

    if verbose:
        print("Train shape after missing-data handling:", train_df.shape)
        print("Test shape after missing-data handling:", test_df.shape)

    for col in candidate:
        p1 = train_df[col].quantile(0.01)
        p99 = train_df[col].quantile(0.99)
        train_df[col] = train_df[col].clip(p1, p99)
        test_df[col] = test_df[col].clip(p1, p99)

    var = train_df[candidate].var()
    candidate = var[var > var_threshold].index.tolist()

    if verbose:
        print("After variance filtering:", len(candidate))

    corr = train_df[candidate].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    candidate = [c for c in candidate if c not in to_drop]

    if verbose:
        print("After correlation filtering:", len(candidate))

    mean_vals = train_df[candidate].mean()
    std_vals = train_df[candidate].std()

    train_df[candidate] = (train_df[candidate] - mean_vals) / std_vals
    test_df[candidate] = (test_df[candidate] - mean_vals) / std_vals

    X = train_df[candidate].values
    y = train_df[event_col].values

    # A dedicated RNG (rather than the global np.random state) makes bootstrap
    # sampling reproducible on its own: identical for the same random_state no
    # matter how many other calls (other hyperparameter combos, other organs)
    # happened earlier in the process - e.g. this call inside a big grid
    # search vs. the same call standalone in a later replication script.
    rng = np.random.RandomState(random_state)

    coef_records = []
    bootstrap_range = tqdm.tqdm(range(bootstrap_iterations)) if verbose else range(bootstrap_iterations)
    for _ in bootstrap_range:
        idx = rng.choice(len(X), size=int(len(X) * bootstrap_percent), replace=True)
        lasso = LassoCV(cv=n_folds, random_state=random_state)
        lasso.fit(X[idx], y[idx])
        coef_records.append(lasso.coef_)

    coef_df = pd.DataFrame(coef_records, columns=candidate).T
    coef_df["mean_coef"] = np.nanmean(coef_df, axis=1)
    coef_df = coef_df.sort_values("mean_coef", ascending=False)

    selected_features = coef_df.index.tolist()[:top_k]

    if verbose:
        print("\nSelected features:", selected_features)

    if make_plots:
        plt.figure(figsize=(8, 6))
        coef_df.head(20)["mean_coef"].sort_values().plot(kind="barh")
        plt.title("LASSO Stability")
        plt.xlabel("Mean Coefficient")
        plt.tight_layout()
        plt.close()

    return {
        "train_df": train_df,
        "test_df": test_df,
        "selected_features": selected_features,
        "coef_df": coef_df,
        "mean_vals": mean_vals,
        "std_vals": std_vals,
    }


def train_cox_and_evaluate(
    train_df,
    test_df,
    selected_features,
    time_col="time",
    event_col="event",
    organ_name="organ",
    save_dir=None,
    coeff_plot_dir=None,
    verbose=True,
    make_plots=True,
):
    train_df = train_df.copy()
    test_df = test_df.copy()
    if verbose:
        print("Final train shape:", train_df.shape)
        print("Final test shape:", test_df.shape)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(
        train_df[[time_col, event_col] + selected_features],
        duration_col=time_col,
        event_col=event_col,
    )

    train_df["risk_score"] = cph.predict_log_partial_hazard(train_df[selected_features])
    test_df["risk_score"] = cph.predict_log_partial_hazard(test_df[selected_features])

    train_cindex = concordance_index(
        train_df[time_col], -train_df["risk_score"], train_df[event_col]
    )

    test_cindex = concordance_index(
        test_df[time_col], -test_df["risk_score"], test_df[event_col]
    )

    if verbose:
        print("\nC-index:")
        print("Train:", f"{train_cindex:.4f}")
        print("Test :", f"{test_cindex:.4f}")

    if make_plots:
        coef = cph.params_.sort_values()

        plt.figure(figsize=(6, 4))
        coef.plot(kind="barh")
        plt.axvline(0, linestyle="--")
        plt.title("Cox Coefficients")
        plt.tight_layout()

        if coeff_plot_dir is not None:
            os.makedirs(coeff_plot_dir, exist_ok=True)
            coef_plot_path = os.path.join(coeff_plot_dir, f"cox_coefficients_{organ_name}.png")
            plt.savefig(coef_plot_path, dpi=300, bbox_inches="tight")
            if verbose:
                print(f"Saved coefficient plot -> {coef_plot_path}")

        plt.close()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        train_export = train_df[["image_id", time_col, event_col, "risk_score"]].copy()
        test_export = test_df[["image_id", time_col, event_col, "risk_score"]].copy()

        train_export["dataset"] = "train"
        test_export["dataset"] = "test"

        export_df = pd.concat([train_export, test_export])
        export_df = export_df.rename(columns={time_col: "MACE_time", event_col: "MACE_event"})

        path = os.path.join(save_dir, f"risk_scores_{organ_name}.csv")
        export_df.to_csv(path, index=False)

        if verbose:
            print(f"Saved -> {path}")

    return {
        "cox_model": cph,
        "train_df": train_df,
        "test_df": test_df,
        "train_cindex": train_cindex,
        "test_cindex": test_cindex,
    }


def load_merged_organ_df(organ_name, labels_df, organ_feature_files):
    """Load an organ feature parquet and merge it with labels. No train/test split.

    Separated from the split step so callers (e.g. a hyperparameter search over
    train_center) can load/merge each organ's features once and reuse it across
    many splits instead of re-reading the parquet file every time.
    """
    feature_key = organ_name
    if feature_key not in organ_feature_files:
        if feature_key.endswith("s") and feature_key[:-1] in organ_feature_files:
            feature_key = feature_key[:-1]
        else:
            print(f"Skipping {organ_name}: feature file not found")
            return None

    feat_df = pd.read_parquet(organ_feature_files[feature_key])
    print(f"\nOrgan: {organ_name} - Loaded feature dataframe shape: {feat_df.shape}")

    if "patient_id" in feat_df.columns:
        feat_df = feat_df.rename(columns={"patient_id": "image_id"})

    if "image_id" not in feat_df.columns:
        print(f"Skipping {organ_name}: no image_id column in feature file")
        return None
    # feat_df["image_id"] = pd.to_numeric(feat_df["image_id"], errors="coerce").astype("Int64")

    merged_df = pd.merge(labels_df, feat_df, on="image_id", how="inner")
    if merged_df.empty:
        print(f"Skipping {organ_name}: no merged rows")
        return None

    print(f"{organ_name} - Label dataframe shape: {labels_df.shape}")
    print(f"{organ_name} - Feature dataframe shape: {feat_df.shape}")
    print(f"{organ_name} - Merged dataframe shape: {merged_df.shape}")

    return merged_df


def split_train_test_by_center(
    merged_df,
    train_center,
    centre_col="UK Biobank assessment centre | Instance 2",
    organ_name=None,
):
    """Split a merged organ dataframe into train/test by assessment centre.

    Rows whose centre equals `train_center` go to train; every other row (of the
    centres present in `merged_df`) goes to test.
    """
    train_df = merged_df[merged_df[centre_col] == train_center].copy()
    test_df = merged_df[merged_df[centre_col] != train_center].copy()

    label = organ_name or ""
    print(f"{label} - Train center: {train_center} - Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    if train_df.empty or test_df.empty:
        print(f"Skipping {label}: empty train or test split for train_center={train_center}")
        return None

    return train_df, test_df


def prepare_data_for_organ(organ_name, labels_df, organ_feature_files, train_center="Reading (imaging)"):
    """Load an organ feature parquet, merge with labels, and split train/test.

    Kept for backward compatibility; internally delegates to
    load_merged_organ_df + split_train_test_by_center.
    """
    merged_df = load_merged_organ_df(organ_name, labels_df, organ_feature_files)
    if merged_df is None:
        return None

    centre_col = "UK Biobank assessment centre | Instance 2"
    split = split_train_test_by_center(merged_df, train_center, centre_col=centre_col, organ_name=organ_name)
    if split is None:
        return None

    train_df, test_df = split
    return merged_df, train_df, test_df


def compare_quantile_logrank(
    train_df,
    test_df,
    risk_score_col,
    time_col,
    event_col,
    n_groups=2,
    return_details=False,
    show_counts=False,
    plot_km=True,
    km_figsize=(8, 6),
    cmap="tab10",
    metadata=None,
    percentiles=None,
    verbose=False,
    suppress_convergence_warnings=True,
    organ_name=None,
    save_dir=None,
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
    missing_train = [c for c in required if c not in train_df.columns]
    missing_test = [c for c in required if c not in test_df.columns]
    if missing_train or missing_test:
        raise KeyError(
            f"Missing required columns. train missing={missing_train}, test missing={missing_test}. "
            f"train sample cols={list(train_df.columns)[:20]}"
        )

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
            ax.set_xlabel("Time (years)")
            ax.set_ylabel("Survival probability")
            ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=4, fontsize=8, frameon=False)
            ax.grid(alpha=0.3)
            if kmf_list:
                add_at_risk_counts(*kmf_list, ax=ax)
            plt.tight_layout()
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
                    fontsize=9,
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
                fontsize=9,
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    alpha=0.85
                )
            )

        title = metadata.get("title", "KM by quantile group") if metadata else "KM by quantile group"
        organ_txt = f"{organ_name} - " if organ_name else ""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
        fig_tr, ax_tr = _plot_with_at_risk(
            tr_out,
            f"{organ_txt} (Train)",
            ax=axes[0],
        )
        _add_hr_annotation(ax_tr, hr_train_df, out["train"]["c_index"])
        fig_te, ax_te = _plot_with_at_risk(
            te_out,
            f"{organ_txt} (Test)",
            ax=axes[1],
        )
        _add_hr_annotation(ax_te, hr_test_df, out["test"]["c_index"])

        if save_dir is not None:
            prefix = organ_name if organ_name else "km"
            km_path = os.path.join(save_dir, f"{prefix}_km.png")

            fig.savefig(km_path, dpi=300, bbox_inches="tight", transparent=False)

            out["plot_path"] = km_path

            print(f"Saved: {km_path}")

        plt.close()

    return out

def _pick_test_hr_row(quantile_result):
    test_block = quantile_result.get("test", {})
    hr_comparisons = test_block.get("hr_comparisons", {})
    if not hr_comparisons:
        return None

    hr_df = pd.DataFrame.from_dict(hr_comparisons, orient="index")
    if hr_df.empty:
        return None

    hr_df = hr_df.sort_values("hr", ascending=False)
    return hr_df.iloc[0]


def _get_test_cohort_n(quantile_result):
    """Return test cohort size from quantile result payload if available."""
    test_details = quantile_result.get("test_details")
    if isinstance(test_details, pd.DataFrame):
        return int(test_details.shape[0])

    test_block = quantile_result.get("test", {})
    for key in ["n", "N", "n_test", "test_n"]:
        if key in test_block and pd.notna(test_block[key]):
            return int(test_block[key])

    return np.nan


def save_forest_summary_csv(
    quantile_results_by_model,
    output_csv_path,
    model_name_map=None,
):
    rows = []
    for model_key, qres in quantile_results_by_model.items():
        chosen = _pick_test_hr_row(qres)
        if chosen is None:
            continue

        test_block = qres.get("test", {})
        test_n = _get_test_cohort_n(qres)
        model_name = model_name_map.get(model_key, model_key) if model_name_map else model_key

        rows.append(
            {
                "Model": model_name,
                "Test HR": float(chosen["hr"]),
                "Test CI Lower (l2)": float(chosen["ci_lower"]),
                "Test CI Upper (h2)": float(chosen["ci_upper"]),
                "pvalue test": float(chosen["p_value"]),
                "Test c index": float(test_block.get("c_index", np.nan)),
                "Test N": test_n,
                
            }
        )

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError("No model rows available to export forest summary CSV.")

    out_df.round(4).to_csv(output_csv_path, index=False)
    print(f"Saved forest summary CSV -> {output_csv_path}")
    return out_df

