from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.special import expit
from statsmodels.stats.mediation import Mediation
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm

A = 'A_cancer'
Y = 'Y_mace'
COVARIATE_FORMULA = 'age + female + height + C(center) + HTN + Diabetes'
N_REP = 500
SEED = 42


@dataclass(frozen=True)
class MediationConfig:
    source_file: Path = Path('./data/source_population_with_retinal_scores.csv')
    clinical_file: Path = Path('./data/final_df_HTN_DB_Status.csv')
    output_dir: Path = Path('./figures/mediation_results')
    covariate_formula: str = 'age + female + height + C(center) + HTN + Diabetes'
    exposure: str = A
    outcome: str = Y
    n_rep: int = 500
    seed: int = 42


def load_and_merge_data(config: MediationConfig) -> pd.DataFrame:
    scores = pd.read_csv(config.source_file)
    clinical = pd.read_csv(config.clinical_file, usecols=['eid', 'HTN', 'Diabetes'])
    data = scores.merge(clinical, on='eid', how='left', validate='m:1')
    missing = data[['HTN', 'Diabetes']].isna().sum()
    if missing.any():
        raise ValueError(f'Unexpected missing HTN/Diabetes after merge: {missing.to_dict()}')
    before = len(data)
    data = data.dropna(subset=['height']).copy()
    print(f'Dropped {before - len(data)} rows with missing height ({before} -> {len(data)})')
    for column in [config.exposure, config.outcome, 'female', 'HTN', 'Diabetes']:
        data[column] = data[column].astype(int)
    return data

def build_adjusted_mediators(df, mediator_cols=("M_CORI_z", "M_MMACE_z"),
                              covariate_formula=COVARIATE_FORMULA):
    """Residualize each mediator against clinical covariates; re-standardize the residual."""
    df = df.copy()
    diagnostics = {}
    for col in mediator_cols:
        adj_col = col.replace("_z", "") + "_adj"
        model = smf.ols(f"{col} ~ {covariate_formula}", data=df).fit()
        resid = model.resid
        df[adj_col] = (resid - resid.mean()) / resid.std(ddof=1)
        diagnostics[col] = {"r_squared": model.rsquared, "adj_col": adj_col}
    return df, diagnostics

def run_assumption_checks(df, A, M, Y, covariate_formula):
    rows = []

    m_a = smf.ols(f"{M} ~ {A} + {covariate_formula}", data=df).fit()
    rows.append(dict(check="A_to_M", term=A, estimate=m_a.params[A], se=m_a.bse[A],
                      pvalue=m_a.pvalues[A]))

    m_y_given_a = smf.logit(f"{Y} ~ {A} + {M} + {covariate_formula}", data=df).fit(disp=0)
    rows.append(dict(check="M_to_Y_given_A", term=M, estimate=m_y_given_a.params[M],
                      se=m_y_given_a.bse[M], pvalue=m_y_given_a.pvalues[M],
                      OR=np.exp(m_y_given_a.params[M])))

    total_effect = smf.logit(f"{Y} ~ {A} + {covariate_formula}", data=df).fit(disp=0)
    rows.append(dict(check="Total_effect_A_to_Y", term=A, estimate=total_effect.params[A],
                      se=total_effect.bse[A], pvalue=total_effect.pvalues[A],
                      OR=np.exp(total_effect.params[A])))

    interaction = smf.logit(f"{Y} ~ {A} * {M} + {covariate_formula}", data=df).fit(disp=0)
    inter_term = f"{A}:{M}"
    rows.append(dict(check="AxM_interaction", term=inter_term,
                      estimate=interaction.params.get(inter_term, np.nan),
                      se=interaction.bse.get(inter_term, np.nan),
                      pvalue=interaction.pvalues.get(inter_term, np.nan)))

    g0 = df.loc[df[A] == 0, M]
    g1 = df.loc[df[A] == 1, M]
    _, tp = stats.ttest_ind(g1, g0, equal_var=False)
    rows.append(dict(check="Mediator_balance_ttest", term=M, estimate=g1.mean() - g0.mean(),
                      se=np.nan, pvalue=tp,
                      note=(f"mean(A=0)={g0.mean():.3f}, mean(A=1)={g1.mean():.3f}, "
                            f"range(A=0)=({g0.min():.2f},{g0.max():.2f}), "
                            f"range(A=1)=({g1.min():.2f},{g1.max():.2f})")))

    return pd.DataFrame(rows)

def compute_vif(df, A, covariate_formula):
    X = patsy.dmatrix(f"{A} + {covariate_formula}", data=df, return_type="dataframe")
    cols = [c for c in X.columns if c != "Intercept"]
    vif_rows = []
    for col in tqdm(cols, desc="  VIF", leave=False):
        i = list(X.columns).index(col)
        vif_rows.append(dict(term=col, VIF=variance_inflation_factor(X.values, i)))
    return pd.DataFrame(vif_rows)

def run_single_mediator(df, A, M, Y, covariate_formula, n_rep=N_REP, seed=SEED):
    mediator_model = smf.ols(f"{M} ~ {A} + {covariate_formula}", data=df)
    outcome_model = smf.glm(f"{Y} ~ {A} + {M} + {covariate_formula}", data=df,
                             family=sm.families.Binomial())

    np.random.seed(seed)
    t0 = time.time()
    print(f"       [single-mediator:{M}] fitting base models + running statsmodels "
          f"Mediation (n_rep={n_rep})...", flush=True)
    med = Mediation(outcome_model, mediator_model, A, M).fit(method="parametric", n_rep=n_rep)
    print(f"       [single-mediator:{M}] done in {time.time() - t0:.1f}s", flush=True)
    summary = med.summary()
    summary.index.name = "quantity"
    return summary.reset_index()

def test_mediator_independence_assumption(df, A, M1, M2, covariate_formula):
    """Probe the joint-mediation DAG's key assumption -- Cancer -> {CORI, MMACE} -> MACE with
    NO edge between CORI and MMACE themselves (both driven by A and C only, independently
    affecting Y). Observational data can never prove the absence of a causal edge or an
    unmeasured shared cause between the two mediators, but a non-trivial residual association
    between them *after* conditioning on A and the clinical covariates is evidence the
    assumption may not hold; a small/non-significant residual association is at least
    consistent with it. This does not affect TE/NDE/NIE_joint from the joint analysis (those
    don't rely on the assumption) -- only the pure per-mediator NIE_M1/NIE_M2 split does."""
    rows = []

    raw_r, raw_p = stats.pearsonr(df[M1], df[M2])
    rows.append(dict(check="raw_correlation_M1_M2", term=f"{M1}~{M2}", estimate=raw_r,
                      pvalue=raw_p))

    resid1 = smf.ols(f"{M1} ~ {A} + {covariate_formula}", data=df).fit().resid
    resid2 = smf.ols(f"{M2} ~ {A} + {covariate_formula}", data=df).fit().resid
    partial_r, partial_p = stats.pearsonr(resid1, resid2)
    rows.append(dict(check="partial_correlation_M1_M2_given_A_C", term=f"{M1}~{M2}",
                      estimate=partial_r, pvalue=partial_p))

    m2_on_m1 = smf.ols(f"{M2} ~ {A} + {covariate_formula} + {M1}", data=df).fit()
    rows.append(dict(check="M1_predicts_M2_given_A_C", term=M1, estimate=m2_on_m1.params[M1],
                      pvalue=m2_on_m1.pvalues[M1]))

    m1_on_m2 = smf.ols(f"{M1} ~ {A} + {covariate_formula} + {M2}", data=df).fit()
    rows.append(dict(check="M2_predicts_M1_given_A_C", term=M2, estimate=m1_on_m2.params[M2],
                      pvalue=m1_on_m2.pvalues[M2]))

    if partial_p < 0.05:
        print(f"       [DAG check] WARNING: {M1} and {M2} remain significantly correlated "
              f"after conditioning on {A} + covariates (partial r={partial_r:.3f}, "
              f"p={partial_p:.2g}). This does not prove a causal edge or shared latent cause "
              f"between them, but the 'no relationship between mediators' assumption cannot be "
              f"ruled out as violated. TE/NDE/NIE_joint remain valid regardless; treat the pure "
              f"NIE_M1_pure/NIE_M2_pure split as approximate.", flush=True)
    else:
        print(f"       [DAG check] {M1}/{M2} partial correlation given {A} + covariates is not "
              f"significant (r={partial_r:.3f}, p={partial_p:.2g}) -- data does not contradict "
              f"the no-mediator-relationship assumption.", flush=True)

    return pd.DataFrame(rows)

def run_two_mediator(df, A, M1, M2, Y, covariate_formula, n_rep=N_REP, seed=SEED):
    """Quasi-Bayesian (Imai/Keele/Tingley-style) g-computation extended to two parallel
    mediators with no assumed causal order between them (see module docstring)."""
    rng = np.random.default_rng(seed)

    print(f"       [two-mediator:{M1}+{M2}] fitting base mediator/outcome models...",
          flush=True)
    t0 = time.time()
    m1_model = smf.ols(f"{M1} ~ {A} + {covariate_formula}", data=df).fit()
    m2_model = smf.ols(f"{M2} ~ {A} + {covariate_formula}", data=df).fit()
    y_model = smf.glm(f"{Y} ~ {A} + {M1} + {M2} + {covariate_formula}", data=df,
                       family=sm.families.Binomial()).fit()
    print(f"       [two-mediator:{M1}+{M2}] base models fit in {time.time() - t0:.1f}s, "
          f"starting {n_rep}-rep bootstrap...", flush=True)

    n = df.shape[0]
    assert m1_model.model.exog.shape[0] == n
    assert m2_model.model.exog.shape[0] == n
    assert y_model.model.exog.shape[0] == n

    resid1 = m1_model.resid.values
    resid2 = m2_model.resid.values
    sigma1, sigma2 = resid1.std(ddof=1), resid2.std(ddof=1)
    rho = np.corrcoef(resid1, resid2)[0, 1]
    cov_resid = np.array([[sigma1 ** 2, rho * sigma1 * sigma2],
                           [rho * sigma1 * sigma2, sigma2 ** 2]])

    X1, X1_names = m1_model.model.exog, m1_model.model.exog_names
    X2, X2_names = m2_model.model.exog, m2_model.model.exog_names
    Xy, Xy_names = y_model.model.exog, y_model.model.exog_names

    a_idx1, a_idx2, a_idxy = X1_names.index(A), X2_names.index(A), Xy_names.index(A)
    m1_idxy, m2_idxy = Xy_names.index(M1), Xy_names.index(M2)

    def with_col(X, idx, val):
        Xc = X.copy()
        Xc[:, idx] = val
        return Xc

    X1_a0, X1_a1 = with_col(X1, a_idx1, 0), with_col(X1, a_idx1, 1)
    X2_a0, X2_a1 = with_col(X2, a_idx2, 0), with_col(X2, a_idx2, 1)

    beta1, cov1 = m1_model.params.values, m1_model.cov_params().values
    beta2, cov2 = m2_model.params.values, m2_model.cov_params().values
    betay, covy = y_model.params.values, y_model.cov_params().values

    def predict_mean(Xc, a_value, m1_vals, m2_vals, by):
        Xc = Xc.copy()
        Xc[:, a_idxy] = a_value
        Xc[:, m1_idxy] = m1_vals
        Xc[:, m2_idxy] = m2_vals
        return expit(Xc @ by).mean()

    quantities = ["TE", "NDE", "NIE_joint", "NIE_M1_pure", "NIE_M2_pure", "mediated_interaction"]
    draws = {q: np.empty(n_rep) for q in quantities}

    t_boot = time.time()
    for r in tqdm(range(n_rep), desc=f"    two-mediator bootstrap ({M1}+{M2})", leave=False):
        b1 = rng.multivariate_normal(beta1, cov1)
        b2 = rng.multivariate_normal(beta2, cov2)
        by = rng.multivariate_normal(betay, covy)

        eps = rng.multivariate_normal([0.0, 0.0], cov_resid, size=n)

        M1_0 = X1_a0 @ b1 + eps[:, 0]
        M1_1 = X1_a1 @ b1 + eps[:, 0]
        M2_0 = X2_a0 @ b2 + eps[:, 1]
        M2_1 = X2_a1 @ b2 + eps[:, 1]

        Y_1_11 = predict_mean(Xy, 1, M1_1, M2_1, by)
        Y_1_00 = predict_mean(Xy, 1, M1_0, M2_0, by)
        Y_0_00 = predict_mean(Xy, 0, M1_0, M2_0, by)
        Y_1_10 = predict_mean(Xy, 1, M1_1, M2_0, by)
        Y_1_01 = predict_mean(Xy, 1, M1_0, M2_1, by)

        te = Y_1_11 - Y_0_00
        nde = Y_1_00 - Y_0_00
        nie_joint = Y_1_11 - Y_1_00
        nie_m1 = Y_1_10 - Y_1_00
        nie_m2 = Y_1_01 - Y_1_00

        draws["TE"][r] = te
        draws["NDE"][r] = nde
        draws["NIE_joint"][r] = nie_joint
        draws["NIE_M1_pure"][r] = nie_m1
        draws["NIE_M2_pure"][r] = nie_m2
        draws["mediated_interaction"][r] = nie_joint - nie_m1 - nie_m2

    print(f"       [two-mediator:{M1}+{M2}] bootstrap finished in "
          f"{time.time() - t_boot:.1f}s", flush=True)

    def bootstrap_pvalue(vals):
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            return np.nan
        p_ge = np.mean(vals >= 0)
        p_le = np.mean(vals <= 0)
        return min(2 * min(p_ge, p_le), 1.0)

    rows = []
    for q in quantities:
        vals = draws[q]
        rows.append(dict(quantity=q, estimate=vals.mean(),
                          ci_low=np.percentile(vals, 2.5), ci_high=np.percentile(vals, 97.5),
                          pvalue=bootstrap_pvalue(vals)))

    te_draws, nie_draws = draws["TE"], draws["NIE_joint"]
    with np.errstate(divide="ignore", invalid="ignore"):
        prop_draws = np.where(te_draws != 0, nie_draws / te_draws, np.nan)
    rows.append(dict(quantity="prop_mediated_joint", estimate=np.nanmean(prop_draws),
                      ci_low=np.nanpercentile(prop_draws, 2.5),
                      ci_high=np.nanpercentile(prop_draws, 97.5),
                      pvalue=bootstrap_pvalue(prop_draws)))

    return pd.DataFrame(rows)


def run_mediation_pipeline(config: MediationConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_and_merge_data(config)
    data, diagnostics = build_adjusted_mediators(data, covariate_formula=config.covariate_formula)
    pd.DataFrame(diagnostics).T.to_csv(config.output_dir / 'adjusted_mediator_diagnostics.csv')
    populations = {
        'full_sample': data,
        'held_out': data[data['analysis_role'].ne('Development')].copy(),
    }
    mediator_sets = {
        'unadjusted': ('M_CORI_z', 'M_MMACE_z'),
        'adjusted': ('M_CORI_adj', 'M_MMACE_adj'),
    }
    for population_name, population in populations.items():
        for mediator_version, (m1, m2) in mediator_sets.items():
            stem = f'{population_name}_{mediator_version}'
            checks = []
            for mediator in (m1, m2):
                check = run_assumption_checks(population, config.exposure, mediator, config.outcome, config.covariate_formula)
                check.insert(0, 'mediator', mediator)
                checks.append(check)
                result = run_single_mediator(population, config.exposure, mediator, config.outcome, config.covariate_formula, config.n_rep, config.seed)
                result.to_csv(config.output_dir / f'{stem}_{mediator}_single.csv', index=False)
            pd.concat(checks, ignore_index=True).to_csv(config.output_dir / f'{stem}_assumption_checks.csv', index=False)
            compute_vif(population, config.exposure, config.covariate_formula).to_csv(config.output_dir / f'{stem}_vif.csv', index=False)
            independence = test_mediator_independence_assumption(population, config.exposure, m1, m2, config.covariate_formula)
            independence.to_csv(config.output_dir / f'{stem}_mediator_independence.csv', index=False)
            joint = run_two_mediator(population, config.exposure, m1, m2, config.outcome, config.covariate_formula, n_rep=config.n_rep, seed=config.seed)
            joint.to_csv(config.output_dir / f'{stem}_joint.csv', index=False)
