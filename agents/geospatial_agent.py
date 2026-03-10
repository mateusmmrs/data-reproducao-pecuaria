"""
Geospatial Agent
────────────────
Generates choropleth maps of Brazil coloured by reproductive KPIs
(pregnancy rate, days open, economic loss) aggregated by state.

Requires an 'estado' column in df_cleaned.  If absent, a realistic
synthetic distribution is assigned based on the Brazilian cattle census.

Graceful fallback: if the IBGE API or plotly fails, logs a warning and
stores an empty context["geospatial"] without crashing the pipeline.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.helpers import section, ROOT

logger = logging.getLogger("GeospatialAgent")

OUTPUTS_MAPS = ROOT / "outputs" / "maps"

# Brazilian cattle census distribution (top states)
ESTADO_CHOICES = ["MT", "MG", "GO", "MS", "PA", "BA", "RS", "PR", "SP", "RO"]
ESTADO_PROBS   = [0.14, 0.12, 0.11, 0.09, 0.08, 0.08, 0.07, 0.06, 0.05, 0.20]

# Farm -> state mapping for synthetic assignment
FARM_STATE_MAP: dict[str, str] = {
    "Fazenda Esperança":  "MT",
    "Sítio Bela Vista":   "MG",
    "Rancho do Sul":      "RS",
    "Agropecuária Norte": "PA",
    "Estância Central":   "GO",
}


class GeospatialAgent(BaseAgent):
    name = "GeospatialAgent"

    def _description(self) -> str:
        return "Generating geospatial choropleth maps by Brazilian state"

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        try:
            context = self._run_geospatial(context)
        except Exception as exc:
            logger.warning(
                "GeospatialAgent failed gracefully — skipping maps. Reason: %s", exc
            )
            context["geospatial"] = {}
        return context

    # ── private ───────────────────────────────────────────────────────────────

    def _run_geospatial(self, context: dict[str, Any]) -> dict[str, Any]:
        OUTPUTS_MAPS.mkdir(parents=True, exist_ok=True)

        df: pd.DataFrame = context["df_cleaned"].copy()
        economic: dict = context.get("economic", {})

        section("Geospatial — Assigning States")
        df = self._ensure_estado(df)

        section("Geospatial — Adding Economic Loss Column")
        df = self._add_economic_loss(df, economic)

        section("Geospatial — Aggregating by State")
        from src.geospatial_analysis import aggregate_by_state
        state_df = aggregate_by_state(df)
        print(f"  States with data: {len(state_df)}")

        saved_maps: list[str] = []

        # --- Map 1: Pregnancy rate ---
        section("Geospatial — Map 1: Taxa de Concepção por Estado")
        if "taxa_concepcao" in state_df.columns:
            path1 = OUTPUTS_MAPS / "mapa_01_taxa_concepcao_por_estado.png"
            self._make_map(
                state_df, "taxa_concepcao",
                "Taxa de Concepção por Estado (%)",
                "RdYlGn", path1,
            )
            saved_maps.append(str(path1))

        # --- Map 2: Avg days open ---
        section("Geospatial — Map 2: Dias Abertos por Estado")
        if "dias_abertos_medio" in state_df.columns:
            path2 = OUTPUTS_MAPS / "mapa_02_dias_abertos_por_estado.png"
            self._make_map(
                state_df, "dias_abertos_medio",
                "Média de Dias Abertos por Estado",
                "RdYlGn_r", path2,
            )
            saved_maps.append(str(path2))

        # --- Map 3: Economic loss ---
        section("Geospatial — Map 3: Perda Econômica por Estado")
        if "perda_economica_total" in state_df.columns:
            path3 = OUTPUTS_MAPS / "mapa_03_perda_economica_por_estado.png"
            self._make_map(
                state_df, "perda_economica_total",
                "Perda Econômica Estimada por Estado (R$)",
                "Reds", path3,
            )
            saved_maps.append(str(path3))

        context["geospatial"] = {
            "state_data": state_df.to_dict(orient="records"),
            "maps": saved_maps,
            "n_states": len(state_df),
        }
        return context

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ensure_estado(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add 'estado' column if missing, using farm mapping or random draw."""
        if "estado" in df.columns:
            return df

        rng = np.random.default_rng(42)

        if "fazenda" in df.columns:
            df["estado"] = df["fazenda"].map(FARM_STATE_MAP)
            # Fill unmapped farms with distribution-weighted random choice
            missing = df["estado"].isna()
            if missing.any():
                df.loc[missing, "estado"] = rng.choice(
                    ESTADO_CHOICES, size=int(missing.sum()), p=ESTADO_PROBS
                )
            print(f"  'estado' assigned from farm mapping + random fill.")
        else:
            df["estado"] = rng.choice(ESTADO_CHOICES, size=len(df), p=ESTADO_PROBS)
            print(f"  'estado' assigned via random distribution (no 'fazenda' column).")

        return df

    def _add_economic_loss(
        self, df: pd.DataFrame, economic: dict
    ) -> pd.DataFrame:
        """Compute per-animal economic loss from days-open data if available."""
        if "perda_economica" in df.columns:
            return df

        days_open_data = economic.get("days_open", {})
        custo_dia = float(days_open_data.get("custo_dia_aberto_brl", 6.0))
        alvo_dias = 110

        if "dias_abertos" in df.columns:
            excess = (df["dias_abertos"] - alvo_dias).clip(lower=0)
            df["perda_economica"] = excess * custo_dia
        else:
            df["perda_economica"] = 0.0

        return df

    def _make_map(
        self,
        state_df: pd.DataFrame,
        column: str,
        title: str,
        colorscale: str,
        output_path,
    ) -> None:
        """Try Plotly choropleth; fall back to matplotlib bar chart."""
        try:
            from src.geospatial_analysis import create_choropleth_map
            create_choropleth_map(state_df, column, title, colorscale, output_path)
            print(f"  ✔ Plotly choropleth saved → {output_path.name}")
        except Exception as exc:
            logger.warning("Plotly map failed (%s). Falling back to bar chart.", exc)
            self._fallback_bar(state_df, column, title, output_path)

    def _fallback_bar(
        self,
        state_df: pd.DataFrame,
        column: str,
        title: str,
        output_path,
    ) -> None:
        """Simple horizontal bar chart as fallback when choropleth is unavailable."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        data = state_df.sort_values(column, ascending=True).dropna(subset=[column])
        if data.empty:
            return

        fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.5)))
        ax.barh(data["estado"], data[column], color="#3498DB", edgecolor="white")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.set_xlabel(column.replace("_", " ").title())
        ax.set_ylabel("Estado (UF)")
        fig.tight_layout()
        try:
            output_path = __import__("pathlib").Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
            print(f"  ✔ Fallback bar chart saved → {output_path.name}")
        except Exception as e:
            logger.warning("Could not save fallback bar chart: %s", e)
        finally:
            plt.close(fig)

    def _summarize(self, context: dict[str, Any]) -> None:
        geo = context.get("geospatial", {})
        n = geo.get("n_states", 0)
        maps = geo.get("maps", [])
        print(f"\n  States analysed : {n}")
        print(f"  Maps saved      : {len(maps)}")
        for m in maps:
            print(f"    → {m}")
