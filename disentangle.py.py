# %% [markdown]
# # Imports
# 

# %%
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path("./") #Path.home() / "Documents" / "GitHub" / "CORI_dev"
SRC_DIR = PROJECT_DIR / "code/src"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.data import (
    TREATMENT_COLUMNS,
    cohort_audit,
    load_clinical,
    load_cohort,
    load_sites,
    load_treatment,
    merge_columns,
    collapse_cmr_columns,
    read_csv,
)
 
from src.metrics import (
    baseline_table, 
)
 

SEED = 20260714
np.random.seed(SEED)

# %% [markdown]
# # Data paths

# %%
DATA_DIR = PROJECT_DIR / "data"
CLEAN_DIR = DATA_DIR / "CLEAN_COHORTS_21JUL"

OUTPUT_DIR = PROJECT_DIR / "outputs_refactored"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"
SCORE_DIR = OUTPUT_DIR / "scores"
QC_DIR = OUTPUT_DIR / "qc"

for directory in [TABLE_DIR, FIGURE_DIR, MODEL_DIR, SCORE_DIR, QC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


COHORT_FILES = {
    "D1":  "./data/CORI_input_files_21Jul/D1_CORI_cancer_development_train_ready_f1024.csv",
    "D2": "./data/CORI_input_files_21Jul/D2_CORI_cancer_heldout_train_ready_f1024.csv",
    "D3": "./data/CORI_input_files_21Jul/D3_MMACEv2_never_cancer_development_train_ready_f1024.csv",
    "D4": "./data/CORI_input_files_21Jul/D4_MMACEv2_never_cancer_heldout_train_ready_f1024.csv",
    "D6": "./data/CORI_input_files_21Jul/D6_CMR_subset_train_ready_f1024.csv",
} 

# Preserve the historical variable name so downstream cells remain unchanged.
MEANPOOL_FILES = COHORT_FILES

CLINICAL_FILE = DATA_DIR / "final_df_HTN_DB_Status.csv"
TREATMENT_FILE = DATA_DIR / "risk_score_df_final_shared_22April_2026.csv"
CANCER_SITE_FILE = DATA_DIR / "CORI_allcancer_8Jan2026.csv"
CMR_FILE = DATA_DIR / "cardiac_mri.csv"
HANDCRAFTED_FILE = DATA_DIR / "H1_handcrafted_subject_level_features_cached.csv"

DEEP_FEATURES = [f"f{i}" for i in range(1024)]
FEATURE_VIEW_COLUMN = "feature_prefix_used"
FEATURE_SOURCE_COLUMN = "source_name"

CLINICAL_VARIABLES = ["age", "female", "height", "Diabetes", "HTN"]
PRIMARY_ADJUSTMENT = ["age", "female", "Diabetes", "HTN"]
MATCH_VARIABLES = ["age", "female", "height"]
CANCER_SITE_COLUMNS = [
    "DigestiveCancer_present", "RespiCancer_present", "BreastCancer_present",
    "FemRepoCancer_present", "MaleRepoCancer_present", "UrinaryTractCancer_present",
    "EndocrineCancer_present", "HeamatoCancer_present", "InsituCancer_present",
    "LipOralCancer_present", "BoneCancer_present", "SkinCancer_present",
    "MesotheliumCancer_present", "EyeCNSCancer_present", "SecondaryCancer_present",
    "UnknownCancer_present",
]


TUNING_K = [10, 15, 20]
TUNING_PENALTIES = [0.001, 0.01, 0.05, 0.1]
TUNING_FOLDS = 3
TUNING_REPEATS = 3
TUNING_RULE = "one_se"

BOOTSTRAPS = 1000
MATCHED_REPETITIONS = 200
CRUDE_MATCHED_REPETITIONS = 200

RUN_FULL_LEARNING_CURVE = True
RUN_ALL_REPRESENTATIONS = False
RUN_HANDCRAFTED = True
RUN_CMR = True
RUN_CROSSFIT_MEDIATION = True

# Records the explicit decision to retain both pre-existing
# RETFound-derived feature provenance groups in the train-ready cohorts.
ALLOW_MIXED_RETFOUND_VIEWS = True


# %%
def save_table(df, name):
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(path)
    return df


def add_groups(df, score_column, thresholds, prefix):
    out = df.copy()
    out[f"{prefix}_high"] = (out[score_column] > thresholds["median"]).astype(int)
    out[f"{prefix}_tertile"] = pd.cut(
        out[score_column],
        [-np.inf, thresholds["tertile_1"], thresholds["tertile_2"], np.inf],
        labels=["Low", "Middle", "High"],
        include_lowest=True,
    ).astype(str)
    return out


def score_thresholds(values):
    values = np.asarray(values, dtype=float)
    return {
        "median": float(np.quantile(values, 0.50)),
        "tertile_1": float(np.quantile(values, 1 / 3)),
        "tertile_2": float(np.quantile(values, 2 / 3)),
    }


# %% [markdown]
# # Load data

# %%
# ============================================================
# Load train-ready RETFound-derived cohorts
# ============================================================

meanpool = {
    cohort: load_cohort(COHORT_FILES[cohort], DEEP_FEATURES)
    for cohort in COHORT_FILES.keys()
}

# Harmonize the authoritative cancer indicator and retain provenance.
for cohort, cohort_df in meanpool.items():
    if "A_cancer_primary" in cohort_df.columns:
        cohort_df["A_cancer"] = pd.to_numeric(
            cohort_df["A_cancer_primary"],
            errors="raise",
        ).astype(int)

    if FEATURE_VIEW_COLUMN not in cohort_df.columns:
        cohort_df[FEATURE_VIEW_COLUMN] = "unknown"

    if FEATURE_SOURCE_COLUMN not in cohort_df.columns:
        cohort_df[FEATURE_SOURCE_COLUMN] = "unknown"

    cohort_df[FEATURE_VIEW_COLUMN] = (
        cohort_df[FEATURE_VIEW_COLUMN]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )
    cohort_df[FEATURE_SOURCE_COLUMN] = (
        cohort_df[FEATURE_SOURCE_COLUMN]
        .fillna("unknown")
        .astype(str)
        .str.strip()
    )

    # Retained for optional provenance-adjusted sensitivity analyses.
    cohort_df["feature_view_emb2"] = (
        cohort_df[FEATURE_VIEW_COLUMN]
        .eq("emb2_f")
        .astype(int)
    )


# ============================================================
# Merge approved external metadata exactly as before
# ============================================================

clinical = load_clinical(CLINICAL_FILE)
treatment = load_treatment(TREATMENT_FILE)
sites = load_sites(CANCER_SITE_FILE, CANCER_SITE_COLUMNS)

for cohort in ["D1", "D2", "D3", "D4", "D6"]:
    meanpool[cohort] = merge_columns(meanpool[cohort], clinical)

for cohort in ["D1", "D2", "D6"]:
    meanpool[cohort] = merge_columns(meanpool[cohort], treatment)

for cohort in ["D1", "D2", "D6"]:
    meanpool[cohort] = merge_columns(meanpool[cohort], sites)

meanpool["D1"]["split"] = "Development"
meanpool["D2"]["split"] = "Held-out"
meanpool["D3"]["split"] = "Development"
meanpool["D4"]["split"] = "Held-out"


# %%
cancer_all = pd.concat([meanpool["D1"], meanpool["D2"]], ignore_index=True)
table_1 = baseline_table(
    cancer_all,
    group_column="split",
    continuous=["age", "height"],
    categorical=["female", "Diabetes", "HTN", "Y_mace"],
)
# "Table_01_baseline_characteristics_D1_D2")


time_col = "time_years"
event_col = "Y_mace"

Noncancer_all = pd.concat([meanpool["D3"], meanpool["D4"]], ignore_index=True)
table_1 = baseline_table(
    Noncancer_all,
    group_column="split",
    continuous=["age", "height"],
    categorical=["female", "Diabetes", "HTN", "Y_mace"],
)


# %%

def preprocess_features(train_df, test_df, test2_df, test3_df, candidate, var_threshold=0.01, corr_threshold=0.9, verbose=True):
    for col in candidate:
        p1 = train_df[col].quantile(0.01)
        p99 = train_df[col].quantile(0.99)
        train_df[col] = train_df[col].clip(p1, p99)
        test_df[col] = test_df[col].clip(p1, p99)
        test2_df[col] = test2_df[col].clip(p1, p99)
        test3_df[col] = test3_df[col].clip(p1, p99)

    mean_vals = train_df[candidate].mean()
    std_vals = train_df[candidate].std()

    train_df[candidate] = (train_df[candidate] - mean_vals) / std_vals
    test_df[candidate] = (test_df[candidate] - mean_vals) / std_vals
    test2_df[candidate] = (test2_df[candidate] - mean_vals) / std_vals
    test3_df[candidate] = (test3_df[candidate] - mean_vals) / std_vals
    
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

    return train_df, test_df, test2_df, test3_df, candidate, mean_vals, std_vals

d1_coriPreProcess , d2_coriPreProcess , d3_coriPreProcess ,d4_coriPreProcess , Cori_filtered_features, mean_vals, std_vals = preprocess_features(meanpool["D1"], meanpool["D2"], meanpool["D3"], meanpool["D4"],
                                                                                     DEEP_FEATURES, var_threshold=0.1, 
                                                                                     corr_threshold=0.8, verbose=True)

d3_MMACEPreProcess , d4_MMACEPreProcess , d1_MMACEPreProcess , d2_MMACEPreProcess , MMACE_filtered_features, mean_vals, std_vals = preprocess_features(meanpool["D3"], meanpool["D4"], meanpool["D1"], meanpool["D2"],
                                                                                     DEEP_FEATURES, var_threshold=0.1, 
                                                                                     corr_threshold=0.8, verbose=True)



# %%
# cori_cohort_dict = {
#     "D1": d1_coriPreProcess,
#     "D2": d2_coriPreProcess,
#     "D4": d4_coriPreProcess,
# }

# mmace_cohort_dict = {
#     "D3": d3_MMACEPreProcess,
#     "D4": d4_MMACEPreProcess,
#     "D2": d2_MMACEPreProcess,
# }

meanpool['D1_cori'] = d1_coriPreProcess
meanpool['D2_cori'] = d2_coriPreProcess
meanpool['D3_cori'] = d3_coriPreProcess
meanpool['D4_cori'] = d4_coriPreProcess

meanpool['D1_mmace'] = d1_MMACEPreProcess
meanpool['D2_mmace'] = d2_MMACEPreProcess
meanpool['D3_mmace'] = d3_MMACEPreProcess
meanpool['D4_mmace'] = d4_MMACEPreProcess

# %% [markdown]
# # Disentanglement learning

# %%
"""
Disentangled multi-task learning: shared + private encoders for two label sets
(Y1, Y2) from a common feature input X.

Architecture
    X -> shared_encoder      -> E1 (shared embedding)
    X -> private_encoder_1   -> P1 (task-1-private embedding)
    X -> private_encoder_2   -> P2 (task-2-private embedding)
    [E1;P1] -> decoder_1 -> E2_1 -> classifier_1 -> Y1 logit
    [E1;P2] -> decoder_2 -> E2_2 -> classifier_2 -> Y2 logit
    (optional) [E1;P1] -> recon_1 -> X_hat_1
    (optional) [E1;P2] -> recon_2 -> X_hat_2

Losses
    - masked BCE per task (handles missing labels), combined via learned
      homoscedastic uncertainty weighting (Kendall et al. 2018)
    - DSN-style difference/orthogonality loss between E1 and each private embedding
    - optional reconstruction loss

Disentanglement diagnostics (printed during training)
    - task AUCs from the deployed classifiers (want: high, improving)
    - E1 -> Y1 / E1 -> Y2 linear-probe AUC (sanity: E1 should stay predictive of both)
    - P1 -> Y2 / P2 -> Y1 linear-probe AUC (leakage: should sit near chance, ~0.5,
      if the private embeddings are genuinely task-specific)
    - mean |cosine similarity| between E1 and each private embedding (want: shrinking toward 0)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# 1. Dataset / DataLoader
# ---------------------------------------------------------------------------

class MultiTaskDataset(Dataset):
    """Wraps X, Y1, Y2. Labels may contain NaN for missing entries; a per-sample
    mask is returned alongside each label so the loss can skip missing ones."""

    def __init__(self, X, Y1, Y2):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        y1 = np.asarray(Y1, dtype=np.float32)
        y2 = np.asarray(Y2, dtype=np.float32)
        self.mask1 = torch.as_tensor(~np.isnan(y1), dtype=torch.float32)
        self.mask2 = torch.as_tensor(~np.isnan(y2), dtype=torch.float32)
        self.Y1 = torch.as_tensor(np.nan_to_num(y1, nan=0.0), dtype=torch.float32)
        self.Y2 = torch.as_tensor(np.nan_to_num(y2, nan=0.0), dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y1[idx], self.Y2[idx], self.mask1[idx], self.mask2[idx]


def make_loaders(X_train, Y1_train, Y2_train, X_test, Y1_test, Y2_test, batch_size=128):
    train_ds = MultiTaskDataset(X_train, Y1_train, Y2_train)
    test_ds = MultiTaskDataset(X_test, Y1_test, Y2_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------

def mlp(in_dim, out_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(inplace=True),
        nn.LayerNorm(hidden),
        nn.Linear(hidden, out_dim),
    )


class DisentangledMultiTaskNet(nn.Module):
    def __init__(self, in_dim, shared_dim=64, private_dim=32, e2_dim=32, use_recon=False):
        super().__init__()
        self.use_recon = use_recon

        self.shared_encoder = mlp(in_dim, shared_dim)
        self.private_encoder_1 = mlp(in_dim, private_dim)
        self.private_encoder_2 = mlp(in_dim, private_dim)

        self.decoder_1 = mlp(shared_dim + private_dim, e2_dim)
        self.decoder_2 = mlp(shared_dim + private_dim, e2_dim)

        self.classifier_1 = nn.Linear(e2_dim, 1)
        self.classifier_2 = nn.Linear(e2_dim, 1)

        if use_recon:
            self.recon_1 = mlp(shared_dim + private_dim, in_dim)
            self.recon_2 = mlp(shared_dim + private_dim, in_dim)

        # learned log-variance terms for uncertainty-weighted multi-task loss
        self.log_sigma1 = nn.Parameter(torch.zeros(()))
        self.log_sigma2 = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        e1 = self.shared_encoder(x)
        p1 = self.private_encoder_1(x)
        p2 = self.private_encoder_2(x)

        e2_1 = self.decoder_1(torch.cat([e1, p1], dim=-1))
        e2_2 = self.decoder_2(torch.cat([e1, p2], dim=-1))

        logit1 = self.classifier_1(e2_1).squeeze(-1)
        logit2 = self.classifier_2(e2_2).squeeze(-1)

        out = {
            "e1": e1, "p1": p1, "p2": p2,
            "e2_1": e2_1, "e2_2": e2_2,
            "logit1": logit1, "logit2": logit2,
        }
        if self.use_recon:
            out["xhat1"] = self.recon_1(torch.cat([e1, p1], dim=-1))
            out["xhat2"] = self.recon_2(torch.cat([e1, p2], dim=-1))
        return out


# ---------------------------------------------------------------------------
# 3. Losses
# ---------------------------------------------------------------------------

def difference_loss(shared, private):
    """DSN-style soft orthogonality constraint: mean-center and L2-normalize
    both embeddings row-wise, then penalize the squared Frobenius norm of
    their cross-correlation. Pushes shared vs. private toward independent
    directions instead of encoding the same information twice."""
    s = shared - shared.mean(dim=0, keepdim=True)
    p = private - private.mean(dim=0, keepdim=True)
    s = nn.functional.normalize(s, dim=1)
    p = nn.functional.normalize(p, dim=1)
    corr = torch.matmul(s.t(), p)
    return (corr ** 2).mean()


def masked_bce(logits, targets, mask, pos_weight=None):
    if mask.sum() == 0:
        return logits.sum() * 0.0  # no labels in this batch; keep graph valid, contribute 0
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    per_sample = loss_fn(logits, targets)
    return (per_sample * mask).sum() / mask.sum().clamp(min=1.0)


def compute_losses(out, y1, y2, m1, m2, x, pos_weight1, pos_weight2,
                    lambda_orth, lambda_recon, model):
    task1_loss = masked_bce(out["logit1"], y1, m1, pos_weight1)
    task2_loss = masked_bce(out["logit2"], y2, m2, pos_weight2)

    # homoscedastic uncertainty weighting (Kendall, Gal & Cipolla 2018)
    weighted_task_loss = (
        torch.exp(-model.log_sigma1) * task1_loss + model.log_sigma1
        + torch.exp(-model.log_sigma2) * task2_loss + model.log_sigma2
    )

    orth1 = difference_loss(out["e1"], out["p1"])
    orth2 = difference_loss(out["e1"], out["p2"])
    orth_loss = orth1 + orth2

    total = weighted_task_loss + lambda_orth * orth_loss

    recon_loss = torch.tensor(0.0, device=x.device)
    if lambda_recon > 0 and "xhat1" in out:
        recon_loss = (
            nn.functional.mse_loss(out["xhat1"], x) + nn.functional.mse_loss(out["xhat2"], x)
        )
        total = total + lambda_recon * recon_loss

    return {
        "total": total,
        "task1": task1_loss.detach(),
        "task2": task2_loss.detach(),
        "orth1": orth1.detach(),
        "orth2": orth2.detach(),
        "recon": recon_loss.detach(),
    }


# ---------------------------------------------------------------------------
# 4. Disentanglement diagnostics
# ---------------------------------------------------------------------------

def safe_auc(y_true, scores):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, scores)


@torch.no_grad()
def extract_embeddings(model, loader, device):
    model.eval()
    E1, P1, P2, Y1, Y2, M1, M2 = [], [], [], [], [], [], []
    for x, y1, y2, m1, m2 in loader:
        x = x.to(device)
        out = model(x)
        E1.append(out["e1"].cpu().numpy())
        P1.append(out["p1"].cpu().numpy())
        P2.append(out["p2"].cpu().numpy())
        Y1.append(y1.numpy()); Y2.append(y2.numpy())
        M1.append(m1.numpy()); M2.append(m2.numpy())
    cat = lambda arrs: np.concatenate(arrs, axis=0)
    return cat(E1), cat(P1), cat(P2), cat(Y1), cat(Y2), cat(M1), cat(M2)


def probe_auc(train_feat, train_label, train_mask, test_feat, test_label, test_mask):
    """Fit a small linear probe on train embeddings -> label, score AUC on
    held-out test embeddings. Used both as a sanity check (E1 -> its own task)
    and as a leakage check (P1 -> the OTHER task)."""
    tr_idx = train_mask > 0.5
    te_idx = test_mask > 0.5
    if tr_idx.sum() < 10 or te_idx.sum() < 10 or len(np.unique(train_label[tr_idx])) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_feat[tr_idx], train_label[tr_idx])
    scores = clf.predict_proba(test_feat[te_idx])[:, 1]
    return safe_auc(test_label[te_idx], scores)


def cosine_overlap(shared, private):
    """Mean |cross-correlation| between shared and private embeddings.

    shared and private can have different widths (e.g. shared_dim=64 vs
    private_dim=32), so a per-sample dot product isn't defined -- mirrors
    difference_loss: row-normalize each embedding, then reduce the
    (shared_dim x private_dim) cross-correlation matrix over the batch to a
    single scalar via mean absolute value."""
    s = shared - shared.mean(axis=0, keepdims=True)
    p = private - private.mean(axis=0, keepdims=True)
    s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)
    p = p / (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    corr = s.T @ p
    return float(np.abs(corr).mean())


def subgroup_auc(y_target, scores_target, mask_target, y_other, mask_other):
    """AUC of the target task split by the OTHER task's label (0 vs 1), restricted
    to samples with valid labels for both -- e.g. cancer AUC computed separately
    within the MACE and non-MACE subgroups."""
    valid = (mask_target > 0.5) & (mask_other > 0.5)
    out = {}
    for name, group_val in [("neg", 0), ("pos", 1)]:
        idx = valid & (y_other == group_val)
        out[name] = safe_auc(y_target[idx], scores_target[idx]) if idx.sum() else float("nan")
    return out


@torch.no_grad()
def evaluate_task_performance(model, loader, device):
    """AUC from the actual deployed classifiers (E2 path) -- the metric that matters.

    Also breaks each task's AUC down by the OTHER task's label, e.g. cancer AUC
    within the MACE-positive vs. MACE-negative subgroup, and vice versa -- this
    surfaces whether performance is uneven across those subgroups rather than
    genuinely task-specific."""
    model.eval()
    logits1, logits2, y1s, y2s, m1s, m2s = [], [], [], [], [], []
    for x, y1, y2, m1, m2 in loader:
        x = x.to(device)
        out = model(x)
        logits1.append(torch.sigmoid(out["logit1"]).cpu().numpy())
        logits2.append(torch.sigmoid(out["logit2"]).cpu().numpy())
        y1s.append(y1.numpy()); y2s.append(y2.numpy())
        m1s.append(m1.numpy()); m2s.append(m2.numpy())
    logits1, logits2 = np.concatenate(logits1), np.concatenate(logits2)
    y1s, y2s = np.concatenate(y1s), np.concatenate(y2s)
    m1s, m2s = np.concatenate(m1s), np.concatenate(m2s)
    auc1 = safe_auc(y1s[m1s > 0.5], logits1[m1s > 0.5]) if (m1s > 0.5).sum() else float("nan")
    auc2 = safe_auc(y2s[m2s > 0.5], logits2[m2s > 0.5]) if (m2s > 0.5).sum() else float("nan")
    # cancer (task1) AUC within MACE-negative / MACE-positive subgroups
    auc1_by_y2 = subgroup_auc(y1s, logits1, m1s, y2s, m2s)
    # MACE (task2) AUC within cancer-negative / cancer-positive subgroups
    auc2_by_y1 = subgroup_auc(y2s, logits2, m2s, y1s, m1s)
    return auc1, auc2, auc1_by_y2, auc2_by_y1


def log_disentanglement_metrics(model, train_loader, test_loader, device, epoch):
    E1_tr, P1_tr, P2_tr, Y1_tr, Y2_tr, M1_tr, M2_tr = extract_embeddings(model, train_loader, device)
    E1_te, P1_te, P2_te, Y1_te, Y2_te, M1_te, M2_te = extract_embeddings(model, test_loader, device)

    # sanity: E1 alone should still predict both tasks reasonably well
    e1_on_y1 = probe_auc(E1_tr, Y1_tr, M1_tr, E1_te, Y1_te, M1_te)
    e1_on_y2 = probe_auc(E1_tr, Y2_tr, M2_tr, E1_te, Y2_te, M2_te)

    # leakage: each private embedding predicting the OTHER task should sit near chance
    leak_p1_on_y2 = probe_auc(P1_tr, Y2_tr, M2_tr, P1_te, Y2_te, M2_te)
    leak_p2_on_y1 = probe_auc(P2_tr, Y1_tr, M1_tr, P2_te, Y1_te, M1_te)

    overlap1 = cosine_overlap(E1_te, P1_te)
    overlap2 = cosine_overlap(E1_te, P2_te)

    print(
        f"[epoch {epoch:03d}] disentanglement | "
        f"E1->Y1(sanity, want high)={e1_on_y1:.3f}  E1->Y2(sanity, want high)={e1_on_y2:.3f}  |  "
        f"P1->Y2(leak, want ~0.5)={leak_p1_on_y2:.3f}  P2->Y1(leak, want ~0.5)={leak_p2_on_y1:.3f}  |  "
        f"cos(E1,P1)={overlap1:.3f}  cos(E1,P2)={overlap2:.3f}"
    )


# ---------------------------------------------------------------------------
# 5. Train loop
# ---------------------------------------------------------------------------

def compute_pos_weight(y, mask):
    y_obs = y[mask > 0.5]
    if len(y_obs) == 0:
        return None
    pos = y_obs.sum()
    neg = len(y_obs) - pos
    if pos == 0:
        return None
    return torch.tensor(neg / max(pos, 1.0), dtype=torch.float32)


def train(model, train_loader, test_loader, device, epochs=50, lr=1e-3,
          lambda_orth=0.1, lambda_recon=0.0, diagnostics_every=5):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    all_y1 = np.concatenate([b[1].numpy() for b in train_loader])
    all_y2 = np.concatenate([b[2].numpy() for b in train_loader])
    all_m1 = np.concatenate([b[3].numpy() for b in train_loader])
    all_m2 = np.concatenate([b[4].numpy() for b in train_loader])
    pw1 = compute_pos_weight(all_y1, all_m1)
    pw2 = compute_pos_weight(all_y2, all_m2)
    if pw1 is not None: pw1 = pw1.to(device)
    if pw2 is not None: pw2 = pw2.to(device)

    history = {
        "epoch": [], "total": [], "task1": [], "task2": [], "orth1": [], "orth2": [], "recon": [],
        "auc_epoch": [], "train_auc1": [], "train_auc2": [], "test_auc1": [], "test_auc2": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        running = {"total": 0.0, "task1": 0.0, "task2": 0.0, "orth1": 0.0, "orth2": 0.0, "recon": 0.0}
        n_batches = 0
        for x, y1, y2, m1, m2 in train_loader:
            x, y1, y2, m1, m2 = [t.to(device) for t in (x, y1, y2, m1, m2)]
            out = model(x)
            losses = compute_losses(out, y1, y2, m1, m2, x, pw1, pw2, lambda_orth, lambda_recon, model)
            opt.zero_grad()
            losses["total"].backward()
            opt.step()
            for k in running:
                v = losses[k]
                running[k] += (v.item() if torch.is_tensor(v) else v)
            n_batches += 1

        for k in running:
            running[k] /= max(n_batches, 1)

        history["epoch"].append(epoch)
        for k in running:
            history[k].append(running[k])

        print(
            f"[epoch {epoch:03d}] train | total={running['total']:.4f}  "
            f"task1={running['task1']:.4f}  task2={running['task2']:.4f}  "
            f"orth1={running['orth1']:.4f}  orth2={running['orth2']:.4f}  recon={running['recon']:.4f}"
        )

        if epoch % diagnostics_every == 0 or epoch == epochs:
            train_auc1, train_auc2, train_auc1_by_y2, train_auc2_by_y1 = evaluate_task_performance(
                model, train_loader, device
            )
            test_auc1, test_auc2, test_auc1_by_y2, test_auc2_by_y1 = evaluate_task_performance(
                model, test_loader, device
            )
            history["auc_epoch"].append(epoch)
            history["train_auc1"].append(train_auc1)
            history["train_auc2"].append(train_auc2)
            history["test_auc1"].append(test_auc1)
            history["test_auc2"].append(test_auc2)
            print(
                f"[epoch {epoch:03d}] train | task1 AUC={train_auc1:.3f}  task2 AUC={train_auc2:.3f}  |  "
                f"test  | task1 AUC={test_auc1:.3f}  task2 AUC={test_auc2:.3f}"
            )
            print(
                f"[epoch {epoch:03d}] subgroup | "
                f"train cancer AUC[MACE-]={train_auc1_by_y2['neg']:.3f} [MACE+]={train_auc1_by_y2['pos']:.3f}  "
                f"MACE AUC[cancer-]={train_auc2_by_y1['neg']:.3f} [cancer+]={train_auc2_by_y1['pos']:.3f}  |  "
                f"test  cancer AUC[MACE-]={test_auc1_by_y2['neg']:.3f} [MACE+]={test_auc1_by_y2['pos']:.3f}  "
                f"MACE AUC[cancer-]={test_auc2_by_y1['neg']:.3f} [cancer+]={test_auc2_by_y1['pos']:.3f}"
            )
            log_disentanglement_metrics(model, train_loader, test_loader, device, epoch)

    return model, history


def plot_training_history(history):
    """Single figure, 2x3 grid: AUC curves (train vs test) for each task,
    task/orth losses, and recon/total loss (kept on their own axes since
    recon can sit on a very different scale from the other loss terms)."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    ax = axes[0, 0]
    ax.plot(history["auc_epoch"], history["train_auc1"], label="train", marker="o")
    ax.plot(history["auc_epoch"], history["test_auc1"], label="test", marker="o")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Task 1 AUC")
    ax.set_xlabel("epoch"); ax.set_ylabel("AUC"); ax.legend()

    ax = axes[0, 1]
    ax.plot(history["auc_epoch"], history["train_auc2"], label="train", marker="o")
    ax.plot(history["auc_epoch"], history["test_auc2"], label="test", marker="o")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Task 2 AUC")
    ax.set_xlabel("epoch"); ax.set_ylabel("AUC"); ax.legend()

    ax = axes[0, 2]
    ax.plot(history["epoch"], history["total"], label="total", color="black")
    ax.set_title("Total loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()

    ax = axes[1, 0]
    ax.plot(history["epoch"], history["task1"], label="task1")
    ax.plot(history["epoch"], history["task2"], label="task2")
    ax.set_title("Task losses (BCE)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()

    ax = axes[1, 1]
    ax.plot(history["epoch"], history["orth1"], label="orth1")
    ax.plot(history["epoch"], history["orth2"], label="orth2")
    ax.set_title("Orthogonality losses")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()

    ax = axes[1, 2]
    ax.plot(history["epoch"], history["recon"], label="recon", color="firebrick")
    ax.set_title("Reconstruction loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Prediction
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, X, device):
    model.eval()
    x = torch.as_tensor(X, dtype=torch.float32).to(device)
    out = model(x)
    p1 = torch.sigmoid(out["logit1"]).cpu().numpy()
    p2 = torch.sigmoid(out["logit2"]).cpu().numpy()
    return p1, p2, out["e1"].cpu().numpy(), out["p1"].cpu().numpy(), out["p2"].cpu().numpy()



# %%
# 'eid' is a patient identifier, not a feature -- it was never standardized in
# preprocess_features (only DEEP_FEATURES is) and its raw scale (~1e6+) was
# swamping the reconstruction loss (and hence the shared gradient signal) by
# many orders of magnitude relative to the O(1) task losses.
train_eid = pd.concat([d1_coriPreProcess['eid'], d3_coriPreProcess['eid']], ignore_index=True).to_numpy()
train_X = pd.concat([d1_coriPreProcess[DEEP_FEATURES],
                     d3_coriPreProcess [DEEP_FEATURES]],
                    ignore_index=True).to_numpy()
train_Y1 = [1] * len(d1_coriPreProcess) + [0] * len(d3_coriPreProcess)
train_Y2 = d1_coriPreProcess['Y_mace'].tolist() + d3_coriPreProcess['Y_mace'].tolist()

train_df = pd.DataFrame(train_X)
train_df['Y_cancer'] = train_Y1
train_df['Y_mace'] = train_Y2

print("Train event Task 1(cancer) counts:", np.bincount(train_Y1))
print("Train event Task 2(mace) counts:", np.bincount(train_Y2))

test_eid = pd.concat([d2_coriPreProcess['eid'], d4_coriPreProcess['eid']], ignore_index=True).to_numpy()
test_X = pd.concat([d2_coriPreProcess[DEEP_FEATURES],
                    d4_coriPreProcess [DEEP_FEATURES]],
                   ignore_index=True).to_numpy()
test_Y1 = [1] * len(d2_coriPreProcess) + [0] * len(d4_coriPreProcess)
test_Y2 = d2_coriPreProcess['Y_mace'].tolist() + d4_coriPreProcess['Y_mace'].tolist()

test_df = pd.DataFrame(test_X)
test_df['Y_cancer'] = test_Y1
test_df['Y_mace'] = test_Y2

print("Test event Task 1(cancer) counts:", np.bincount(test_Y1))
print("Test event Task 2(mace) counts:", np.bincount(test_Y2))

# %%

# ---------------------------------------------------------------------------
# 7. Usage (swap in your real X_train/Y1_train/Y2_train/X_test/Y1_test/Y2_test)
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- replace this block with your real arrays ----
# X_train: (n_train, d) float array
# Y1_train, Y2_train: (n_train,) arrays of 0/1/np.nan (nan = missing label)
# X_test, Y1_test, Y2_test: same shapes for the held-out set
train_loader, test_loader = make_loaders(
    train_X, train_Y1, train_Y2, test_X, test_Y1, test_Y2, batch_size=512
)

model = DisentangledMultiTaskNet(
    in_dim=train_X.shape[1], shared_dim=64, private_dim=32, e2_dim=32, use_recon=True
)

# %%


model, history = train(
    model, train_loader, test_loader, device,
    epochs=50, lr=1e-4, lambda_orth=0.1, lambda_recon=0.1, diagnostics_every=2,
)

# %%
plot_training_history(history)
plt.savefig("disentangled_training_history.png", dpi=300)
plt.show()

p1, p2, e1, p1_emb, p2_emb = predict(model, test_X, device)
