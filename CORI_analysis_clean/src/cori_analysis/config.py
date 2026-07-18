from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    project_dir: Path = Path('.')
    input_dir: Path = Path('./data')
    output_dir: Path = Path('./figures')
    master_csv: str = 'CORI_allcancer_8Jan_ALL_COLUMNS_with_retfound_features.csv'
    noncancer_csv: str = 'cohort2_noncancer_mace_with_retfound.csv'
    treatment_csv: str = 'risk_score_df_final_shared_22April_2026.csv'
    chemo_csv: str = 'chemo_status.csv'
    cardiac_mri_csv: str = 'cardiac_mri.csv'
    clinical_csv: str = 'final_df_HTN_DB_Status.csv'
    handcrafted_feature_dir: Path = Path('./data/handcrafted_features')
    source_population_csv: str = 'source_population_with_retinal_scores.csv'
    proteomics_csv: Path = Path('./data/proteomics_50k_instance_0_sdf.csv')
    proteomics_columns_txt: Path = Path('./data/alz_proteomics_columns.txt')

    def input_path(self, filename: str) -> Path:
        return self.input_dir / filename

    @property
    def master(self) -> Path:
        return self.input_path(self.master_csv)

    @property
    def noncancer(self) -> Path:
        return self.input_path(self.noncancer_csv)

    @property
    def treatment(self) -> Path:
        return self.input_path(self.treatment_csv)

    @property
    def chemo(self) -> Path:
        return self.input_path(self.chemo_csv)

    @property
    def cardiac_mri(self) -> Path:
        return self.input_path(self.cardiac_mri_csv)

    @property
    def clinical(self) -> Path:
        return self.input_path(self.clinical_csv)

    @property
    def source_population(self) -> Path:
        return self.input_path(self.source_population_csv)


@dataclass(frozen=True)
class ColumnSchema:
    id: str = 'eid'
    image_visit: str = 'image_visit'
    cancer_status: str = 'allcancer_event_status'
    cancer_status_fallback: str = 'AnyCancer_present'
    center: str = 'assessment_center_at_image_visit'
    age: str = 'age_at_image_visit'
    sex: str = 'sex'
    height: str = 'height'
    event: str = 'event'
    time: str = 'time'
    cori_score: str = 'CORI_score'
    mmace_score: str = 'MMACE_score'
    clinical_sex: str = 'sex'
    clinical_diabetes: str = 'Diabetes'
    clinical_htn: str = 'HTN'


@dataclass(frozen=True)
class ModelConfig:
    train_centers: tuple[str, ...] = ('Birmingham', 'Croydon')
    primary_horizon: int = 10
    variance_threshold: float = 1e-8
    max_feature_missing: float = 0.50
    univariate_cox_penalizer: float = 0.01
    candidate_feature_counts: tuple[int, ...] = (5, 10, 15, 20, 30, 50, 75, 100)
    horizon_columns: dict[int, tuple[str, str]] = field(default_factory=lambda: {
        3: ('MACE_in_allCancer_3yr_censored_time', 'MACE_in_allCancer_3yr_censored_status'),
        5: ('MACE_in_allCancer_5yr_censored_time', 'MACE_in_allCancer_5yr_censored_status'),
        10: ('MACE_in_allCancer_10yr_censored_time', 'MACE_in_allCancer_10yr_censored_status'),
    })


@dataclass(frozen=True)
class CMRConfig:
    exact_variables: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    primary_families: str | tuple[str, ...] = 'ALL_CURATED'
    min_nonmissing: int = 30
    min_unique_values: int = 5
    top_features_per_family: int = 8
    top_features_overall: int = 20


@dataclass(frozen=True)
class AnalysisConfig:
    paths: DataPaths = field(default_factory=DataPaths)
    columns: ColumnSchema = field(default_factory=ColumnSchema)
    model: ModelConfig = field(default_factory=ModelConfig)
    cmr: CMRConfig = field(default_factory=CMRConfig)

    @property
    def h1_dir(self) -> Path:
        return self.paths.output_dir / 'H1_CORI_LOCKED_MODEL_v13'

    @property
    def cori_bundle(self) -> Path:
        return self.h1_dir / 'models' / 'CORI_locked_model_bundle.pkl'

    @property
    def hcori_dir(self) -> Path:
        return self.h1_dir / 'handcrafted_HCORI'

    @property
    def hcori_bundle(self) -> Path:
        return self.hcori_dir / 'models' / 'HCORI_locked_handcrafted_model_bundle.pkl'

    @property
    def handcrafted_cache(self) -> Path:
        return self.hcori_dir / 'tables' / 'H1_handcrafted_subject_level_features_cached.csv'

    def validate_inputs(self, *, include_optional: bool = False) -> None:
        required = [self.paths.master, self.paths.noncancer, self.paths.clinical]
        if include_optional:
            required.extend([self.paths.treatment, self.paths.chemo, self.paths.cardiac_mri])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Missing configured input files:\n' + '\n'.join(missing))
