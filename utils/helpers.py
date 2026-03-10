"""Utility functions shared across agents."""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ── Console helpers ────────────────────────────────────────────────────────────

def log_step(agent_name: str, message: str) -> None:
    """Print a formatted step banner."""
    border = "─" * 70
    print(f"\n{border}")
    print(f"  [{agent_name.upper()}]  {message}")
    print(f"{border}")


def section(title: str) -> None:
    print(f"\n{'━' * 60}")
    print(f"  {title}")
    print(f"{'━' * 60}")


# ── I/O helpers ────────────────────────────────────────────────────────────────

def save_dataframe(df: pd.DataFrame, path: str | Path, fmt: str = "csv") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
    elif fmt == "excel":
        df.to_excel(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    print(f"  ✔ Saved → {path}")


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    elif suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {suffix}")


def save_json(obj: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    print(f"  ✔ Saved → {path}")


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Formatting ─────────────────────────────────────────────────────────────────

def format_currency(value: float, symbol: str = "R$") -> str:
    return f"{symbol} {value:,.2f}"


def format_pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


# ── Path constants ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS_REPORTS = ROOT / "outputs" / "reports"
OUTPUTS_PLOTS = ROOT / "outputs" / "plots"
OUTPUTS_MODELS = ROOT / "outputs" / "models"

for _p in (DATA_RAW, DATA_PROCESSED, OUTPUTS_REPORTS, OUTPUTS_PLOTS, OUTPUTS_MODELS):
    _p.mkdir(parents=True, exist_ok=True)
