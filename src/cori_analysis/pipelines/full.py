from __future__ import annotations

from dataclasses import dataclass

from ..config import AnalysisConfig
from .common import SharedData, prepare_shared_data
from .h1 import H1Result, run_h1
from .h2 import H2Result, run_h2
from .h3 import H3Result, run_h3
from .h4 import H4Result, run_h4
from .handcrafted import HCORIResult, HMMACEResult, apply_hcori_to_cmr, run_hcori, run_hmmace


@dataclass
class FullAnalysisResult:
    shared: SharedData
    h1: H1Result
    h2: H2Result
    h3: H3Result
    h4: H4Result
    hcori: HCORIResult | None = None
    hmmace: HMMACEResult | None = None
    hcori_cmr: object | None = None


def run_full_analysis(
    config: AnalysisConfig,
    *,
    include_handcrafted: bool = True,
    rebuild_handcrafted_cache: bool = False,
) -> FullAnalysisResult:
    config.validate_inputs(include_optional=True)
    shared = prepare_shared_data(config)
    h1 = run_h1(shared, config)
    h3 = run_h3(shared, config)
    h2 = run_h2(shared, h1, config)
    h4 = run_h4(shared, config)
    hcori = hmmace = hcori_cmr = None
    if include_handcrafted:
        hcori = run_hcori(h1, config, rebuild_cache=rebuild_handcrafted_cache)
        hmmace = run_hmmace(h2, hcori, config)
        hcori_cmr = apply_hcori_to_cmr(h4, config)
    return FullAnalysisResult(shared, h1, h2, h3, h4, hcori, hmmace, hcori_cmr)
