"""
IBGE SIDRA API Client
─────────────────────
Fetches livestock and agricultural data from IBGE's SIDRA service.
Official REST API v3: https://servicodados.ibge.gov.br/api/v3/

Key tables:
  3939 — Efetivo dos rebanhos (PPM) — bovine herd by municipality
    Variable 105: Bovinos (cabeças)
  74   — Produção de origem animal (PPM) — milk production
    Variable 106: Leite (mil litros)
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

SIDRA_BASE = "https://servicodados.ibge.gov.br/api/v3/agregados"
LOCALIDADES_BASE = "https://servicodados.ibge.gov.br/api/v1/localidades"
TIMEOUT = 12  # seconds


class IBGESIDRAClient:
    """Thin client around the IBGE SIDRA REST API v3."""

    def __init__(self, timeout: int = TIMEOUT):
        self.timeout = timeout

    # ── public ────────────────────────────────────────────────────────────────

    def fetch_rebanho_bovino_municipal(self, year: int | None = None) -> dict[str, float]:
        """
        Bovine herd head-count by municipality (table 3939, variable 105).
        Returns {cod_municipio: head_count}.
        PPM data typically lags 2 years; defaults to best available year.
        """
        year = year or _latest_ppm_year()
        url = f"{SIDRA_BASE}/3939/periodos/{year}/variaveis/105?localidades=N6[all]"
        raw = self._get(url, "rebanho bovino municipal")
        return _parse_sidra_series(raw, year) if raw else {}

    def fetch_producao_leite_municipal(self, year: int | None = None) -> dict[str, float]:
        """
        Milk production (thousand litres) by municipality (table 74, variable 106).
        Returns {cod_municipio: mil_litros}.
        """
        year = year or _latest_ppm_year()
        url = f"{SIDRA_BASE}/74/periodos/{year}/variaveis/106?localidades=N6[all]"
        raw = self._get(url, "produção de leite municipal")
        return _parse_sidra_series(raw, year) if raw else {}

    def fetch_municipios(self) -> dict[str, str]:
        """
        All Brazilian municipalities from the Localidades API.
        Returns {cod_municipio: nome}.  Faster and lighter than SIDRA.
        """
        url = f"{LOCALIDADES_BASE}/municipios"
        try:
            resp = self._get_raw(url, "lista de municípios")
            if resp:
                return {str(m["id"]): m["nome"] for m in resp}
        except Exception as exc:
            print(f"  ⚠  IBGE municipios parse error: {exc}")
        return {}

    def fetch_estados(self) -> dict[str, str]:
        """Returns {sigla_uf: nome_uf} for all states."""
        url = f"{LOCALIDADES_BASE}/estados"
        try:
            resp = self._get_raw(url, "estados")
            if resp:
                return {str(e["sigla"]): e["nome"] for e in resp}
        except Exception as exc:
            print(f"  ⚠  IBGE estados parse error: {exc}")
        return {}

    # ── private ───────────────────────────────────────────────────────────────

    def _get(self, url: str, label: str) -> list | None:
        if not _HAS_REQUESTS:
            print("  ⚠  'requests' not installed — IBGE SIDRA data unavailable.")
            return None
        try:
            resp = _requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            print(f"  ⚠  IBGE SIDRA [{label}] failed: {exc}")
            return None

    def _get_raw(self, url: str, label: str) -> list | None:
        return self._get(url, label)


# ── helpers ───────────────────────────────────────────────────────────────────

def _latest_ppm_year() -> int:
    """PPM data lags ~2 years; return best available estimate."""
    return date.today().year - 2


def _parse_sidra_series(raw: list, year: int) -> dict[str, float]:
    """
    Parse SIDRA v3 response into {municipality_code: numeric_value}.

    Response structure:
      list of blocks → resultados → series → localidade + serie{year: value}
    """
    result: dict[str, float] = {}
    year_str = str(year)
    _MISSING = {"...", "-", "X", "", "null"}
    try:
        for bloco in raw:
            for resultado in bloco.get("resultados", []):
                for serie in resultado.get("series", []):
                    cod = str(serie["localidade"]["id"])
                    val_str = serie.get("serie", {}).get(year_str, "")
                    if val_str and val_str not in _MISSING:
                        try:
                            result[cod] = float(
                                val_str.replace(".", "").replace(",", ".")
                            )
                        except ValueError:
                            pass
    except Exception as exc:
        print(f"  ⚠  SIDRA parse error: {exc}")
    return result
