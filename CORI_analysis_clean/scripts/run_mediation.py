from __future__ import annotations

import argparse
from pathlib import Path

from cori_analysis.mediation import MediationConfig, run_mediation_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description='Run CORI/MMACE mediation analyses.')
    parser.add_argument('--source-file', type=Path, default=Path('./data/source_population_with_retinal_scores.csv'))
    parser.add_argument('--clinical-file', type=Path, default=Path('./data/final_df_HTN_DB_Status.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('./figures/mediation_results'))
    parser.add_argument('--n-rep', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--adjust-mediator-for-cancer-status', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    formula = 'age + female + height + C(center) + HTN + Diabetes'
    output = args.output_dir
    if args.adjust_mediator_for_cancer_status:
        formula += ' + A_cancer'
        if output == Path('./figures/mediation_results'):
            output = Path('./figures/mediation_results_cancerAdjusted')
    run_mediation_pipeline(MediationConfig(
        source_file=args.source_file,
        clinical_file=args.clinical_file,
        output_dir=output,
        covariate_formula=formula,
        n_rep=args.n_rep,
        seed=args.seed,
    ))


if __name__ == '__main__':
    main()
