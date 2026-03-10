"""
Data Engineer Agent
───────────────────
Loads raw datasets, standardises column names, enforces dtypes,
engineers reproductive-performance features, and persists the
clean intermediate dataset.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.helpers import save_dataframe, DATA_RAW, DATA_PROCESSED


# Column name normalisation map  (raw → canonical)
COLUMN_MAP: dict[str, str] = {
    # Identifiers
    "id_animal": "id_animal",
    "id": "id_animal",
    "animal_id": "id_animal",
    "brinco": "id_animal",
    "numero": "id_animal",
    # Breed
    "raca": "raca",
    "breed": "raca",
    "raça": "raca",
    # Category
    "categoria": "categoria",
    "category": "categoria",
    "tipo": "categoria",
    # Parity
    "ordem_parto": "ordem_parto",
    "paridade": "ordem_parto",
    "parity": "ordem_parto",
    "numero_partos": "ordem_parto",
    # Body Condition Score
    "escore_condicao_corporal": "ecc",
    "ecc": "ecc",
    "bcs": "ecc",
    "body_condition_score": "ecc",
    # Calving date
    "data_parto": "data_parto",
    "calving_date": "data_parto",
    "data_nascimento_cria": "data_parto",
    # Insemination date
    "data_inseminacao": "data_inseminacao",
    "insemination_date": "data_inseminacao",
    "data_ia": "data_inseminacao",
    # Diagnosis date
    "data_diagnostico": "data_diagnostico",
    "diagnosis_date": "data_diagnostico",
    # Pregnancy result
    "resultado_diagnostico": "prenhe",
    "prenhe": "prenhe",
    "pregnant": "prenhe",
    "concepcao": "prenhe",
    "concebeu": "prenhe",
    # Services per conception
    "servicos_por_concepcao": "servicos_concepcao",
    "spc": "servicos_concepcao",
    "services_per_conception": "servicos_concepcao",
    # Bull / semen
    "touro": "touro",
    "bull": "touro",
    "semen": "touro",
    "reprodutor": "touro",
    # Season
    "estacao": "estacao",
    "season": "estacao",
    "estação": "estacao",
    # Year
    "ano": "ano",
    "year": "ano",
    # Milk production
    "producao_leite": "producao_leite_kg",
    "milk_production": "producao_leite_kg",
    "leite_kg": "producao_leite_kg",
    "leite": "producao_leite_kg",
    # Voluntary waiting period (days)
    "periodo_espera_voluntario": "pev",
    "pev": "pev",
    "vwp": "pev",
    # Days open
    "dias_abertos": "dias_abertos",
    "days_open": "dias_abertos",
    # Calving interval
    "intervalo_entre_partos": "iep",
    "iep": "iep",
    "calving_interval": "iep",
    # Farm / property
    "fazenda": "fazenda",
    "farm": "fazenda",
    "propriedade": "fazenda",
    # Technician
    "tecnico": "tecnico",
    "technician": "tecnico",
    "inseminador": "tecnico",
    # Municipality (IBGE code or name)
    "municipio": "municipio",
    "município": "municipio",
    "municipality": "municipio",
    "cod_municipio": "municipio",
    "codigo_municipio": "municipio",
    "ibge": "municipio",
    "cidade": "municipio",
    "city": "municipio",
}

POSITIVE_PREGNANCY = {"sim", "yes", "s", "1", "true", "prenhe", "positivo", "positive", "p"}


class DataEngineerAgent(BaseAgent):
    name = "DataEngineerAgent"

    def _description(self) -> str:
        return "Loading, standardising, and feature-engineering the dataset"

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        raw_path: Path | None = context.get("raw_path")

        if raw_path and Path(raw_path).exists():
            df = self._load(Path(raw_path))
        else:
            print("  ⚠  No dataset found — generating synthetic demo data.")
            df = self._generate_synthetic()

        df = self._normalise_columns(df)
        df = self._cast_types(df)
        df = self._engineer_features(df)

        out = DATA_PROCESSED / "dataset_engineered.csv"
        save_dataframe(df, out)

        context["df_engineered"] = df
        context["engineered_path"] = out
        return context

    # ── private ───────────────────────────────────────────────────────────────

    def _load(self, path: Path) -> pd.DataFrame:
        print(f"  Loading: {path}")
        suffix = path.suffix.lower()
        if suffix == ".csv":
            for enc in ("utf-8-sig", "latin-1", "cp1252"):
                try:
                    return pd.read_csv(path, encoding=enc)
                except UnicodeDecodeError:
                    continue
        elif suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif suffix == ".parquet":
            return pd.read_parquet(path)
        raise ValueError(f"Unsupported format: {suffix}")

    def _normalise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(r"[\s\-/\\]", "_", regex=True)
            .str.replace(r"[áàãâä]", "a", regex=True)
            .str.replace(r"[éèêë]", "e", regex=True)
            .str.replace(r"[íìîï]", "i", regex=True)
            .str.replace(r"[óòõôö]", "o", regex=True)
            .str.replace(r"[úùûü]", "u", regex=True)
            .str.replace(r"[ç]", "c", regex=True)
            .str.replace(r"[^a-z0-9_]", "", regex=True)
        )
        df = df.rename(columns={c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP})
        return df

    def _cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        date_cols = [c for c in ("data_parto", "data_inseminacao", "data_diagnostico") if c in df.columns]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")

        if "prenhe" in df.columns:
            df["prenhe"] = (
                df["prenhe"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(lambda x: 1 if x in POSITIVE_PREGNANCY else (0 if x not in ("nan", "none", "") else np.nan))
            )

        numeric = ["ecc", "ordem_parto", "servicos_concepcao", "producao_leite_kg", "pev", "dias_abertos", "iep"]
        for col in [c for c in numeric if c in df.columns]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Days open from calving to confirmed pregnancy
        if "data_inseminacao" in df.columns and "data_parto" in df.columns:
            df["dias_pos_parto"] = (df["data_inseminacao"] - df["data_parto"]).dt.days

        # Calving interval (if not already present)
        if "iep" not in df.columns and "dias_abertos" in df.columns:
            df["iep"] = df["dias_abertos"] + 283          # gestation ~283 d

        # Season from insemination date
        if "estacao" not in df.columns and "data_inseminacao" in df.columns:
            df["estacao"] = df["data_inseminacao"].dt.month.map(self._month_to_season)

        # BCS category
        if "ecc" in df.columns:
            df["ecc_categoria"] = pd.cut(
                df["ecc"],
                bins=[0, 2.49, 2.99, 3.49, 5.0],
                labels=["Baixo", "Regular", "Bom", "Excelente"],
            )

        # Parity group
        if "ordem_parto" in df.columns:
            df["grupo_paridade"] = pd.cut(
                df["ordem_parto"],
                bins=[0, 1, 3, 100],
                labels=["Primípara", "Plurípara (2-3)", "Plurípara (4+)"],
            )

        # Year from insemination
        if "ano" not in df.columns and "data_inseminacao" in df.columns:
            df["ano"] = df["data_inseminacao"].dt.year

        return df

    @staticmethod
    def _month_to_season(month: int) -> str:
        if month in (12, 1, 2):
            return "Verão"
        elif month in (3, 4, 5):
            return "Outono"
        elif month in (6, 7, 8):
            return "Inverno"
        return "Primavera"

    # ── synthetic data ─────────────────────────────────────────────────────────

    def _generate_synthetic(self, n: int = 1_200) -> pd.DataFrame:
        rng = np.random.default_rng(42)

        racas = rng.choice(["Nelore", "Angus", "Girolando", "Senepol", "Brangus"], n,
                           p=[0.40, 0.20, 0.20, 0.10, 0.10])
        ecc = rng.normal(2.9, 0.5, n).clip(1, 5).round(1)
        paridade = rng.integers(1, 8, n)
        estacoes = rng.choice(["Verão", "Outono", "Inverno", "Primavera"], n)
        touros = rng.choice([f"T{i:03d}" for i in range(1, 21)], n)
        tecnicos = rng.choice(["Ana", "Bruno", "Carla", "Diego", "Elisa"], n)
        fazendas = rng.choice(["Fazenda Esperança", "Sítio Bela Vista", "Rancho do Sul",
                               "Agropecuária Norte", "Estância Central"], n)

        # Pregnancy probability influenced by ECC and parity
        logit = (
            -0.5
            + 1.2 * (ecc - 2.5)
            - 0.05 * np.where(paridade > 4, paridade - 4, 0)
            + rng.normal(0, 0.3, n)
        )
        prob = 1 / (1 + np.exp(-logit))
        prenhe = (rng.uniform(0, 1, n) < prob).astype(int)

        base_date = pd.Timestamp("2022-01-01")
        ia_dates = [base_date + pd.Timedelta(days=int(d)) for d in rng.integers(0, 730, n)]
        parto_dates = [d - pd.Timedelta(days=int(rng.integers(30, 180))) for d in ia_dates]

        milk = np.where(
            np.isin(racas, ["Girolando"]),
            rng.normal(22, 4, n),
            rng.normal(6, 2, n),
        ).clip(0).round(1)

        dias_abertos = rng.normal(130, 40, n).clip(21, 365).round(0).astype(int)

        df = pd.DataFrame({
            "id_animal":        [f"A{i:05d}" for i in range(1, n + 1)],
            "fazenda":          fazendas,
            "raca":             racas,
            "ordem_parto":      paridade,
            "ecc":              ecc,
            "estacao":          estacoes,
            "touro":            touros,
            "tecnico":          tecnicos,
            "data_parto":       parto_dates,
            "data_inseminacao": ia_dates,
            "prenhe":           prenhe,
            "servicos_concepcao": rng.choice([1, 1, 1, 2, 2, 3, 4], n),
            "producao_leite_kg": milk,
            "dias_abertos":     dias_abertos,
        })
        return df

    def _summarize(self, context: dict[str, Any]) -> None:
        df = context.get("df_engineered")
        if df is not None:
            print(f"\n  Dataset shape : {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"  Columns       : {list(df.columns)}")
