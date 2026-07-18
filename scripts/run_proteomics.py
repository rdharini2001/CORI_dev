from __future__ import annotations

import argparse
from pathlib import Path

from cori_analysis.proteomics import ProteomicsConfig, run_proteomics


def parse_args():
    parser = argparse.ArgumentParser(description='Run CORI/MMACE proteomics comparison.')
    parser.add_argument('--source-file', type=Path, default=Path('./data/source_population_with_retinal_scores.csv'))
    parser.add_argument('--clinical-file', type=Path, default=Path('./data/final_df_HTN_DB_Status.csv'))
    parser.add_argument('--proteomics-file', type=Path, default=Path('./data/proteomics_50k_instance_0_sdf.csv'))
    parser.add_argument('--protein-columns-file', type=Path, default=Path('./data/alz_proteomics_columns.txt'))
    parser.add_argument('--output-dir', type=Path, default=Path('./figures/proteomics'))
    return parser.parse_args()


def main():
    args = parse_args()
    run_proteomics(ProteomicsConfig(
        source_file=args.source_file,
        clinical_file=args.clinical_file,
        proteomics_file=args.proteomics_file,
        protein_columns_file=args.protein_columns_file,
        output_dir=args.output_dir,
    ))


if __name__ == '__main__':
    main()
