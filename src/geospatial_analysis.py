"""
Geospatial Analysis Module
──────────────────────────
Functions for generating choropleth maps of Brazilian states based on
livestock reproductive metrics fetched from the IBGE GeoJSON API.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# UF code -> IBGE numeric code (string)
UF_TO_CODE: dict[str, str] = {
    "RO": "11", "AC": "12", "AM": "13", "RR": "14", "PA": "15",
    "AP": "16", "TO": "17", "MA": "21", "PI": "22", "CE": "23",
    "RN": "24", "PB": "25", "PE": "26", "AL": "27", "SE": "28",
    "BA": "29", "MG": "31", "ES": "32", "RJ": "33", "SP": "35",
    "PR": "41", "SC": "42", "RS": "43", "MS": "50", "MT": "51",
    "GO": "52", "DF": "53",
}

IBGE_GEOJSON_URL = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
    "?resolucao=2&formato=application/vnd.geo+json"
)


# ── GeoJSON fetching ───────────────────────────────────────────────────────────

def fetch_brazil_states_geojson() -> dict:
    """
    Fetch Brazilian state boundaries from the IBGE GeoJSON API.
    Returns a GeoJSON dict.  On failure raises an exception.
    """
    import requests

    logger.info("Fetching Brazil states GeoJSON from IBGE …")
    resp = requests.get(IBGE_GEOJSON_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Data aggregation ───────────────────────────────────────────────────────────

def aggregate_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate farm-level data by 'estado' (UF code) column.

    Returns a DataFrame with columns:
        estado, codarea, n_animais, taxa_concepcao, dias_abertos_medio,
        perda_economica_total
    """
    required = "estado"
    if required not in df.columns:
        raise ValueError("DataFrame must contain an 'estado' column with UF codes.")

    agg: dict[str, Any] = {"n_animais": (df.columns[0], "count")}
    named_aggs: dict[str, Any] = {"n_animais": pd.NamedAgg(column=df.columns[0], aggfunc="count")}

    if "prenhe" in df.columns:
        named_aggs["taxa_concepcao"] = pd.NamedAgg(column="prenhe", aggfunc="mean")
    if "dias_abertos" in df.columns:
        named_aggs["dias_abertos_medio"] = pd.NamedAgg(column="dias_abertos", aggfunc="mean")
    if "perda_economica" in df.columns:
        named_aggs["perda_economica_total"] = pd.NamedAgg(column="perda_economica", aggfunc="sum")

    state_df = df.groupby("estado").agg(**named_aggs).reset_index()
    state_df.rename(columns={"estado": "estado"}, inplace=True)

    # Add IBGE numeric code
    state_df["codarea"] = state_df["estado"].map(UF_TO_CODE)

    # Round floats
    for col in ("taxa_concepcao", "dias_abertos_medio"):
        if col in state_df.columns:
            state_df[col] = state_df[col].round(4)

    return state_df


# ── Plotly choropleth ──────────────────────────────────────────────────────────

def create_choropleth_map(
    state_df: pd.DataFrame,
    column: str,
    title: str,
    colorscale: str = "RdYlGn",
    output_path: Path | str | None = None,
) -> Any:
    """
    Create a Plotly choropleth map of Brazil coloured by `column`.
    Saves as HTML (always) and PNG (if kaleido is available).
    Returns the plotly Figure.
    """
    import plotly.express as px

    geojson = fetch_brazil_states_geojson()

    fig = px.choropleth(
        state_df,
        geojson=geojson,
        locations="codarea",
        featureidkey="properties.codarea",
        color=column,
        color_continuous_scale=colorscale,
        hover_name="estado",
        hover_data={"codarea": False, column: True},
        title=title,
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r": 0, "t": 50, "l": 0, "b": 0},
        coloraxis_colorbar=dict(title=column.replace("_", " ").title()),
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save HTML
        html_path = output_path.with_suffix(".html")
        fig.write_html(str(html_path))
        logger.info("Saved choropleth HTML → %s", html_path)

        # Save PNG (requires kaleido)
        try:
            fig.write_image(str(output_path))
            logger.info("Saved choropleth PNG → %s", output_path)
        except Exception as exc:
            logger.warning("Could not save PNG (kaleido issue?): %s", exc)

    return fig


# ── Static matplotlib/geopandas map ───────────────────────────────────────────

def create_static_map(
    gdf: Any,  # GeoDataFrame
    column: str,
    title: str,
    output_path: Path | str | None = None,
) -> Any:
    """
    Create a static choropleth map using geopandas + matplotlib.
    `gdf` must be a GeoDataFrame with a `column` to colour by.
    Returns the matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    gdf.plot(
        column=column,
        ax=ax,
        legend=True,
        cmap="RdYlGn",
        missing_kwds={"color": "#CCCCCC", "label": "Sem dados"},
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        logger.info("Saved static map → %s", output_path)

    return fig
