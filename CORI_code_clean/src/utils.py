from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import path


try:
    from cori_pipeline_utils_v13 import safe_name, pformat, fmt_ci, clean_id, savefig
except Exception:
    pass


# ------------------------------------------------------------
# 0. Small fallback helpers
# ------------------------------------------------------------
def _safe_name(x):
    try:
        return safe_name(x)
    except Exception:
        return re.sub(r"[^A-Za-z0-9]+", "_", str(x)).strip("_")


def _pformat(p):
    try:
        return pformat(p)
    except Exception:
        if p is None or not np.isfinite(p):
            return "NA"
        return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_ci(x, lo, hi, digits=2):
    try:
        return fmt_ci(x, lo, hi, digits)
    except Exception:
        if not all(np.isfinite(v) for v in [x, lo, hi]):
            return "NA"
        return f"{x:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"


def _clean_id(s):
    try:
        return clean_id(s)
    except Exception:
        return pd.Series(s).astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def _savefig(fig, basepath):
    try:
        savefig(fig, basepath)
    except Exception:
        basepath = Path(basepath)
        basepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(basepath) + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(str(basepath) + ".pdf", bbox_inches="tight")
