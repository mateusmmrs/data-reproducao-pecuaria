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

# GeoJSON with all 27 Brazilian states (properties: sigla, name, codigo_ibg)
# Source: click_that_hood / IBGE — verified working, 27 features
BRAZIL_STATES_GEOJSON_URL = (
    "https://raw.githubusercontent.com/codeforamerica/click_that_hood"
    "/master/public/data/brazil-states.geojson"
)


# ── GeoJSON fetching ───────────────────────────────────────────────────────────

def fetch_brazil_states_geojson() -> dict:
    """
    Fetch Brazilian state boundaries (27 states, one feature each).
    Properties per feature: sigla (UF), name, codigo_ibg (IBGE numeric code).
    Returns a GeoJSON FeatureCollection dict.
    """
    import requests

    logger.info("Fetching Brazil states GeoJSON …")
    resp = requests.get(BRAZIL_STATES_GEOJSON_URL, timeout=30)
    resp.raise_for_status()
    geojson = resp.json()
    n = len(geojson.get("features", []))
    logger.info("Loaded %d state features", n)
    return geojson


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


# ── Main map function (geopandas PNG + Plotly HTML) ────────────────────────────

def create_choropleth_map(
    state_df: pd.DataFrame,
    column: str,
    title: str,
    colorscale: str = "RdYlGn",
    output_path: Path | str | None = None,
) -> Any:
    """
    Create a choropleth map of Brazil coloured by `column`.

    Strategy:
      • PNG (static)  → geopandas + matplotlib  (reliable, publication-quality)
      • HTML (interactive) → Plotly Express      (hover tooltips)

    Returns the matplotlib Figure.
    """
    geojson = fetch_brazil_states_geojson()

    # Build GeoDataFrame from the downloaded GeoJSON
    import geopandas as gpd
    gdf = gpd.GeoDataFrame.from_features(geojson["features"])
    # properties.sigla → UF code; align column name with state_df
    gdf = gdf.rename(columns={"sigla": "estado", "name": "nome_estado"})

    # Merge reproductive data into the GeoDataFrame
    gdf = gdf.merge(state_df, on="estado", how="left")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # ── PNG via matplotlib/geopandas ──────────────────────────────────────
        fig = _geopandas_map(gdf, column, title, colorscale, output_path)
        logger.info("Saved static PNG → %s", output_path)

        # ── HTML via Plotly (interactive) ─────────────────────────────────────
        try:
            _plotly_html(state_df, geojson, column, title, colorscale,
                         output_path.with_suffix(".html"))
        except Exception as exc:
            logger.warning("Plotly HTML skipped: %s", exc)

        return fig

    return _geopandas_map(gdf, column, title, colorscale, output_path=None)


def _geopandas_map(
    gdf: Any,
    column: str,
    title: str,
    colorscale: str,
    output_path: "Path | None",
) -> Any:
    """Render a geopandas GeoDataFrame as a matplotlib choropleth PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib import cm

    # Map colorscale name → matplotlib cmap
    cmap_map = {
        "RdYlGn":   "RdYlGn",
        "RdYlGn_r": "RdYlGn_r",
        "Reds":     "Reds",
        "Blues":    "Blues",
        "YlOrRd":   "YlOrRd",
    }
    cmap = cmap_map.get(colorscale, "RdYlGn")

    fig, ax = plt.subplots(figsize=(13, 10), facecolor="white")

    # States with data
    has_data = gdf[column].notna()

    gdf[has_data].plot(
        column=column,
        ax=ax,
        cmap=cmap,
        legend=True,
        legend_kwds={
            "label": column.replace("_", " ").title(),
            "orientation": "vertical",
            "shrink": 0.6,
            "pad": 0.02,
        },
        edgecolor="#FFFFFF",
        linewidth=0.8,
        missing_kwds={"color": "#D0D0D0"},
    )
    # States without data (grey)
    gdf[~has_data].plot(ax=ax, color="#D0D0D0", edgecolor="#FFFFFF", linewidth=0.8)

    # State labels
    for _, row in gdf.iterrows():
        if row.geometry is None:
            continue
        centroid = row.geometry.centroid
        val = row.get(column)
        label = row.get("estado", "")
        if val is not None and not pd.isna(val):
            val_str = f"{val:.1%}" if val < 2 else f"{val:,.0f}"
            ax.annotate(
                f"{label}\n{val_str}",
                xy=(centroid.x, centroid.y),
                ha="center", va="center",
                fontsize=6.5, color="#1a1a1a",
                fontweight="bold",
            )
        else:
            ax.annotate(label, xy=(centroid.x, centroid.y),
                        ha="center", va="center", fontsize=6.5, color="#777")

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16, color="#1a3a5c")
    ax.axis("off")
    ax.set_facecolor("white")

    # Footer with source
    fig.text(0.01, 0.01,
             "Fonte: dados sintéticos calibrados Embrapa/CNA  |  Fronteiras: IBGE",
             fontsize=7.5, color="#888", style="italic")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)

    return fig


def _plotly_html(
    state_df: pd.DataFrame,
    geojson: dict,
    column: str,
    title: str,
    colorscale: str,
    html_path: "Path",
) -> None:
    """Save an interactive Plotly choropleth as HTML."""
    import plotly.express as px

    fig = px.choropleth(
        state_df,
        geojson=geojson,
        locations="estado",
        featureidkey="properties.sigla",
        color=column,
        color_continuous_scale=colorscale,
        hover_name="estado",
        title=title,
        scope="south america",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(
        margin={"r": 20, "t": 60, "l": 20, "b": 20},
        paper_bgcolor="white",
    )
    fig.write_html(str(html_path))
    logger.info("Saved interactive HTML → %s", html_path)


# ── Convenience alias ──────────────────────────────────────────────────────────

def create_static_map(
    gdf: Any,
    column: str,
    title: str,
    output_path: "Path | str | None" = None,
) -> Any:
    """Thin wrapper kept for backwards compatibility."""
    return _geopandas_map(gdf, column, title, "RdYlGn",
                          Path(output_path) if output_path else None)
