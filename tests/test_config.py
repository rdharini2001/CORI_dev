from pathlib import Path

from cori_analysis.config import AnalysisConfig, DataPaths


def test_config_resolves_only_explicit_paths(tmp_path: Path):
    config = AnalysisConfig(paths=DataPaths(input_dir=tmp_path / 'data', output_dir=tmp_path / 'out'))
    assert config.paths.master == tmp_path / 'data' / 'CORI_allcancer_8Jan_ALL_COLUMNS_with_retfound_features.csv'
    assert config.cori_bundle == tmp_path / 'out' / 'H1_CORI_LOCKED_MODEL_v13' / 'models' / 'CORI_locked_model_bundle.pkl'
