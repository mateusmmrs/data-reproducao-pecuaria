"""
Exploratory Data Analysis Agent
────────────────────────────────
Computes summary statistics, group-level KPIs, correlations,
and variable distributions specific to animal reproduction.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.helpers import save_json, DATA_PROCESSED, section
from utils.ibge_client import IBGESIDRAClient


class EDAAgent(BaseAgent):
    name = "EDAAgent"

    def _description(self) -> str:
        return "Computing reproductive KPIs, correlations, and distributions"

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = context["df_cleaned"].copy()
        eda: dict[str, Any] = {}

        section("Global Summary Statistics")
        eda["summary"] = self._summary(df)

        section("Reproductive KPIs")
        eda["kpis"] = self._kpis(df)

        section("Conception Rate by Group")
        eda["group_rates"] = self._group_rates(df)

        section("Correlation Matrix (numeric)")
        eda["correlations"] = self._correlations(df)

        section("Distribution Statistics")
        eda["distributions"] = self._distributions(df)

        section("Municipal Analysis — IBGE SIDRA")
        eda["municipal"] = self._municipal_analysis(df)

        save_json(eda, DATA_PROCESSED / "eda_results.json")
        context["eda"] = eda
        return context

    # ── private ───────────────────────────────────────────────────────────────

    def _summary(self, df: pd.DataFrame) -> dict:
        desc = df.describe(include="all").round(3).to_dict()
        print(df.describe(include="number").round(2).to_string())
        return desc

    def _kpis(self, df: pd.DataFrame) -> dict:
        kpis: dict[str, Any] = {}

        # Conception / pregnancy rate
        if "prenhe" in df.columns:
            cr = df["prenhe"].mean()
            kpis["taxa_concepcao_global"] = round(float(cr), 4)
            print(f"  Taxa de concepção global         : {cr:.1%}")

        # Services per conception
        if "servicos_concepcao" in df.columns:
            spc = df["servicos_concepcao"].mean()
            kpis["servicos_por_concepcao"] = round(float(spc), 2)
            print(f"  Serviços por concepção (média)   : {spc:.2f}")

        # Average calving interval
        if "iep" in df.columns:
            iep = df["iep"].mean()
            kpis["iep_medio_dias"] = round(float(iep), 1)
            print(f"  Intervalo entre partos (média)   : {iep:.0f} dias")

        # Average days open
        if "dias_abertos" in df.columns:
            da = df["dias_abertos"].mean()
            kpis["dias_abertos_medio"] = round(float(da), 1)
            print(f"  Dias abertos (média)             : {da:.0f} dias")

        # Average BCS
        if "ecc" in df.columns:
            ecc = df["ecc"].mean()
            kpis["ecc_medio"] = round(float(ecc), 2)
            print(f"  ECC médio                        : {ecc:.2f}")

        # Milk production
        if "producao_leite_kg" in df.columns:
            milk = df["producao_leite_kg"].mean()
            kpis["producao_leite_media_kg"] = round(float(milk), 1)
            print(f"  Produção de leite (média)        : {milk:.1f} kg/dia")

        return kpis

    def _group_rates(self, df: pd.DataFrame) -> dict:
        group_results: dict[str, Any] = {}

        groupby_cols = [
            c for c in ("raca", "estacao", "grupo_paridade", "ecc_categoria",
                        "fazenda", "tecnico", "ano")
            if c in df.columns
        ]

        if "prenhe" not in df.columns:
            return group_results

        for col in groupby_cols:
            grp = (
                df.groupby(col, observed=True)["prenhe"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "taxa_concepcao", "count": "n"})
                .sort_values("taxa_concepcao", ascending=False)
                .round(4)
            )
            group_results[col] = grp.to_dict(orient="index")
            print(f"\n  Taxa de concepção por {col}:")
            for k, v in grp.iterrows():
                print(f"    {str(k):25s}  {v['taxa_concepcao']:.1%}  (n={v['n']})")

        return group_results

    def _correlations(self, df: pd.DataFrame) -> dict:
        num = df.select_dtypes("number")
        if num.shape[1] < 2:
            return {}
        corr = num.corr().round(3)
        print(corr.to_string())
        return corr.to_dict()

    def _distributions(self, df: pd.DataFrame) -> dict:
        result: dict[str, Any] = {}
        num_cols = df.select_dtypes("number").columns
        for col in num_cols:
            s = df[col].dropna()
            result[col] = {
                "mean":   round(float(s.mean()), 3),
                "median": round(float(s.median()), 3),
                "std":    round(float(s.std()), 3),
                "min":    round(float(s.min()), 3),
                "max":    round(float(s.max()), 3),
                "q25":    round(float(s.quantile(0.25)), 3),
                "q75":    round(float(s.quantile(0.75)), 3),
                "skew":   round(float(s.skew()), 3),
                "kurtosis": round(float(s.kurtosis()), 3),
            }
        return result

    def _municipal_analysis(self, df: pd.DataFrame) -> dict:
        """
        If the dataset contains a 'municipio' column (IBGE code or name),
        fetch regional herd data from IBGE SIDRA and compare farm performance
        against the municipal/state context.
        """
        result: dict[str, Any] = {}

        if "municipio" not in df.columns:
            print("  ℹ  Coluna 'municipio' não encontrada — análise municipal ignorada.")
            print("     Adicione 'municipio' (código IBGE) ao dataset para habilitar.")
            return result

        ibge = IBGESIDRAClient()
        municipios_dataset = df["municipio"].dropna().astype(str).unique().tolist()
        print(f"  Municípios no dataset : {len(municipios_dataset)}")

        # Fetch names for codes
        mun_names = ibge.fetch_municipios()

        # Fetch regional cattle herd
        rebanho = ibge.fetch_rebanho_bovino_municipal()
        leite_prod = ibge.fetch_producao_leite_municipal()

        municipal_rows = []
        for cod in municipios_dataset:
            # Try exact code match, then 6-digit prefix (IBGE uses 7-digit)
            rebanho_val = rebanho.get(cod) or rebanho.get(cod[:6])
            leite_val = leite_prod.get(cod) or leite_prod.get(cod[:6])
            nome = mun_names.get(cod) or mun_names.get(cod[:7]) or cod

            # Farm subset for this municipality
            mask = df["municipio"].astype(str) == cod
            farm_cr = df.loc[mask, "prenhe"].mean() if "prenhe" in df.columns else None
            n_animals = int(mask.sum())

            row: dict[str, Any] = {
                "municipio_cod": cod,
                "municipio_nome": nome,
                "n_animais_dataset": n_animals,
                "rebanho_bovino_municipio": rebanho_val,
                "producao_leite_mil_litros": leite_val,
                "taxa_concepcao_fazenda": round(float(farm_cr), 4) if farm_cr is not None else None,
            }
            municipal_rows.append(row)
            print(
                f"  {nome[:30]:30s}  n={n_animals:4d}  "
                f"rebanho={int(rebanho_val):,} cab." if rebanho_val
                else f"  {nome[:30]:30s}  n={n_animals:4d}  rebanho=N/D"
            )

        result["municipios"] = municipal_rows
        result["fonte"] = "IBGE SIDRA / PPM"
        result["data_source"] = "https://sidra.ibge.gov.br"
        return result

    def _summarize(self, context: dict[str, Any]) -> None:
        kpis = context.get("eda", {}).get("kpis", {})
        if kpis:
            print("\n  Key KPIs stored in context:")
            for k, v in kpis.items():
                print(f"    {k}: {v}")
