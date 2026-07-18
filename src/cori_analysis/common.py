from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 2026
np.random.seed(RANDOM_SEED)

class QCLogger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"CORI QC LOG\nStarted: {datetime.now()}\n{'='*100}\n", encoding="utf-8")
    def log(self, msg: str):
        print(msg)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    def section(self, title: str):
        self.log("\n" + "="*100 + f"\n{title}\n" + "="*100)
    def df(self, name: str, df: pd.DataFrame, max_rows: int = 30):
        self.section(name)
        if df is None:
            self.log("<None>")
            return
        self.log(df.head(max_rows).to_string(index=False))
        self.log(f"[shape={df.shape}]")

def ensure_dirs(base: Path):
    base = Path(base)
    fig = base / "figures"
    tab = base / "tables"
    mod = base / "models"
    qc = base / "qc"
    for p in [base, fig, tab, mod, qc]:
        p.mkdir(parents=True, exist_ok=True)
    return base, fig, tab, mod, qc

def savefig(fig, basepath):
    basepath = Path(basepath)
    basepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(basepath)+".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(basepath)+".pdf", bbox_inches="tight")
    fig.savefig(str(basepath)+".svg", bbox_inches="tight")

    print("Images saved to:", str(basepath)+".{png,pdf,svg}")

def safe_name(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s)).strip("_")

def clean_id(x):
    return pd.Series(x).astype(str).str.replace(r"\.0$", "", regex=True).str.strip()

def load_csv(path, logger=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV: {path}")
    if logger: logger.log(f"Loading CSV: {path}")
    return pd.read_csv(path, low_memory=False)

def as_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def pformat(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def fmt_ci(x, lo, hi, digits=2):
    if not all(np.isfinite(v) for v in [x, lo, hi]):
        return "NA"
    return f"{x:.{digits}f} ({lo:.{digits}f}-{hi:.{digits}f})"
