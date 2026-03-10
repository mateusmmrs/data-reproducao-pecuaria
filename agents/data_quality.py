"""
Data Quality Agent
──────────────────
Detects missing values, duplicates, outliers, and inconsistent
categories. Produces a quality report and a cleaned dataset.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from agents.base_agent import BaseAgent
from utils.helpers import save_dataframe, save_json, DATA_PROCESSED, section


class DataQualityAgent(BaseAgent):
    name = "DataQualityAgent"

    # Acceptable value ranges for reproductive variables
    RANGES: dict[str, tuple[float, float]] = {
        "ecc":                  (1.0, 5.0),
        "ordem_parto":          (1, 15),
        "dias_abertos":         (21, 500),
        "iep":                  (304, 800),
        "servicos_concepcao":   (1, 10),
        "producao_leite_kg":    (0, 80),
        "dias_pos_parto":       (0, 600),
    }

    def _description(self) -> str:
        return "Assessing data quality — missing values, duplicates, outliers, categories"

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = context["df_engineered"].copy()

        report: dict[str, Any] = {}

        section("1. Missing Values")
        report["missing"] = self._missing(df)

        section("2. Duplicate Records")
        report["duplicates"] = self._duplicates(df)
        df = df.drop_duplicates(subset=[c for c in ("id_animal", "data_inseminacao") if c in df.columns])

        section("3. Out-of-Range Values")
        report["out_of_range"] = self._out_of_range(df)

        section("4. Statistical Outliers (IQR)")
        report["outliers"] = self._outliers(df)

        section("5. Category Consistency")
        report["categories"] = self._categories(df)

        section("6. Cleaning & Imputation")
        df = self._clean(df)

        # Overall score
        total_cells = df.shape[0] * df.shape[1]
        issues = (
            sum(v["count"] for v in report["missing"].values())
            + report["duplicates"]["count"]
            + sum(v["count"] for v in report["out_of_range"].values())
        )
        report["quality_score"] = round(max(0, 1 - issues / total_cells) * 100, 2)
        print(f"\n  ✦ Overall quality score: {report['quality_score']:.1f} / 100")

        out = DATA_PROCESSED / "dataset_cleaned.csv"
        save_dataframe(df, out)
        save_json(report, DATA_PROCESSED / "quality_report.json")

        context["df_cleaned"] = df
        context["cleaned_path"] = out
        context["quality_report"] = report
        return context

    # ── private ───────────────────────────────────────────────────────────────

    def _missing(self, df: pd.DataFrame) -> dict:
        result = {}
        for col in df.columns:
            n = int(df[col].isna().sum())
            pct = round(n / len(df) * 100, 2)
            if n:
                result[col] = {"count": n, "pct": pct}
                print(f"  {col:30s} {n:5d} missing  ({pct:.1f}%)")
        if not result:
            print("  ✔ No missing values detected.")
        return result

    def _duplicates(self, df: pd.DataFrame) -> dict:
        dup_cols = [c for c in ("id_animal", "data_inseminacao") if c in df.columns]
        n_dup = int(df.duplicated(subset=dup_cols).sum()) if dup_cols else 0
        n_full = int(df.duplicated().sum())
        print(f"  Full-row duplicates   : {n_full}")
        print(f"  Key-column duplicates : {n_dup}  (cols: {dup_cols})")
        return {"count": n_full, "key_duplicates": n_dup}

    def _out_of_range(self, df: pd.DataFrame) -> dict:
        result = {}
        for col, (lo, hi) in self.RANGES.items():
            if col not in df.columns:
                continue
            mask = (df[col] < lo) | (df[col] > hi)
            n = int(mask.sum())
            if n:
                result[col] = {"count": n, "range": [lo, hi]}
                print(f"  {col:30s} {n:5d} out-of-range  [{lo} – {hi}]")
        if not result:
            print("  ✔ All numeric columns within expected ranges.")
        return result

    def _outliers(self, df: pd.DataFrame) -> dict:
        result = {}
        num_cols = df.select_dtypes("number").columns
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((series < lo) | (series > hi)).sum())
            if n:
                result[col] = {"count": n, "iqr_bounds": [round(lo, 2), round(hi, 2)]}
                print(f"  {col:30s} {n:5d} IQR outliers")
        if not result:
            print("  ✔ No IQR outliers detected.")
        return result

    def _categories(self, df: pd.DataFrame) -> dict:
        result = {}
        cat_cols = df.select_dtypes(["object", "category"]).columns
        for col in cat_cols:
            uniques = sorted(df[col].dropna().astype(str).str.strip().unique().tolist())
            result[col] = uniques
            print(f"  {col:30s} → {uniques[:10]}{'…' if len(uniques) > 10 else ''}")
        return result

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clamp out-of-range numerics
        for col, (lo, hi) in self.RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lo, hi)

        # Impute numeric missing values with median
        for col in df.select_dtypes("number").columns:
            if df[col].isna().any():
                median = df[col].median()
                df[col] = df[col].fillna(median)
                print(f"  Imputed '{col}' missing values with median ({median:.2f})")

        # Standardise string categories
        for col in df.select_dtypes("object").columns:
            df[col] = df[col].astype(str).str.strip().str.title()

        # Drop rows where target variable is missing
        if "prenhe" in df.columns:
            before = len(df)
            df = df.dropna(subset=["prenhe"])
            removed = before - len(df)
            if removed:
                print(f"  Removed {removed} rows with missing 'prenhe' (target variable)")

        print(f"\n  ✔ Clean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
        return df

    def _summarize(self, context: dict[str, Any]) -> None:
        score = context.get("quality_report", {}).get("quality_score", "N/A")
        print(f"\n  Quality report saved  |  Score: {score}/100")
