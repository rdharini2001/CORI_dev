from __future__ import annotations

import argparse
from pathlib import Path

from cori_analysis.config import AnalysisConfig, DataPaths
from cori_analysis.pipelines import run_full_analysis


def parse_args():
    parser = argparse.ArgumentParser(description='Run the modular CORI analysis pipeline.')
    parser.add_argument('--input-dir', type=Path, default=Path('./data'))
    parser.add_argument('--output-dir', type=Path, default=Path('./figures'))
    parser.add_argument('--handcrafted-dir', type=Path, default=Path('./data/handcrafted_features'))
    parser.add_argument('--skip-handcrafted', action='store_true')
    parser.add_argument('--rebuild-handcrafted-cache', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    paths = DataPaths(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        handcrafted_feature_dir=args.handcrafted_dir,
    )
    run_full_analysis(
        AnalysisConfig(paths=paths),
        include_handcrafted=not args.skip_handcrafted,
        rebuild_handcrafted_cache=args.rebuild_handcrafted_cache,
    )


if __name__ == '__main__':
    main()
