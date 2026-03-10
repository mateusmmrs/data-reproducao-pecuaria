"""
Commodity Price Client — multi-source com fallback em cascata
──────────────────────────────────────────────────────────────
Estratégia de busca de preços (em ordem):
  1. BCB/SGS  (Banco Central do Brasil — API pública JSON) ← fonte primária
  2. IPEADATA REST API                                     ← fallback robusto
  3. None                                                  ← callers usam DEFAULT_PARAMS

Preços buscados:
  • Boi gordo — R$/@ (arroba, 15 kg)  — SGS série 4452 / IPEADATA AGRO12_PBGSP12
  • Leite     — R$/L                  — SGS série 4390 / IPEADATA AGRO12_PLESP12

APIs:
  BCB SGS  → https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados/ultimos/3?formato=json
  IPEADATA → http://ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{serie}')
"""
from __future__ import annotations

from typing import Optional

try:
    import requests as _requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

TIMEOUT = 10

# ── BCB / SGS ─────────────────────────────────────────────────────────────────
_BCB_BASE = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados/ultimos/3?formato=json"

# Séries BCB/SGS confirmadas:
#   4452 — Boi gordo (São Paulo) R$/arroba — mensal
#   4390 — Leite tipo C (São Paulo) R$/litro — mensal
_BCB_BOI_SERIES   = [4452, 28483]   # 28483 = boi gordo indicador B3
_BCB_LEITE_SERIES = [4390, 4391]    # 4391 = leite tipo B

# ── IPEADATA API ──────────────────────────────────────────────────────────────
_IPEA_BASE = "http://ipeadata.gov.br/api/odata4"
_IPEA_BOI_SERIES   = ["AGRO12_PBGSP12", "AGR12_PBGSP12", "PRECOS12_ICBBSG12"]
_IPEA_LEITE_SERIES = ["AGRO12_PLESP12", "AGR12_PLESP12"]


class CEPEAClient:
    """Busca preços de commodities bovinas via BCB/SGS → IPEADATA → None."""

    def __init__(self, timeout: int = TIMEOUT):
        self.timeout = timeout

    def fetch_boi_gordo_arroba(self) -> Optional[float]:
        """Retorna preço atual do boi gordo em R$/@. None se indisponível."""
        price = self._bcb_fetch(_BCB_BOI_SERIES, "boi gordo", expected_range=(100, 800))
        if price:
            return price
        return self._ipeadata_fetch(_IPEA_BOI_SERIES, "boi gordo", expected_range=(100, 800))

    def fetch_leite_litro(self) -> Optional[float]:
        """
        Retorna preço atual do leite em R$/L. None se indisponível.
        Converte automaticamente se valor estiver em R$/1000L.
        """
        raw = self._bcb_fetch(_BCB_LEITE_SERIES, "leite", expected_range=(0.5, 5_000))
        if raw is None:
            raw = self._ipeadata_fetch(_IPEA_LEITE_SERIES, "leite", expected_range=(0.5, 5_000))
        if raw and raw > 20:
            raw = raw / 1_000   # era R$/1000L → converte para R$/L
        return raw

    # ── BCB / SGS ──────────────────────────────────────────────────────────────

    def _bcb_fetch(
        self,
        series_list: list[int],
        label: str,
        expected_range: tuple[float, float],
    ) -> Optional[float]:
        if not _HAS_REQUESTS:
            return None
        for serie in series_list:
            url = _BCB_BASE.format(serie=serie)
            try:
                resp = _requests.get(url, timeout=self.timeout)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                # BCB retorna lista de {"data": "DD/MM/YYYY", "valor": "NNN.NN"}
                for entry in reversed(data):   # mais recente primeiro
                    val_str = entry.get("valor", "")
                    if val_str and val_str not in ("", "null", None):
                        try:
                            val = float(str(val_str).replace(",", "."))
                        except ValueError:
                            continue
                        lo, hi = expected_range
                        if lo <= val <= hi:
                            data_ref = entry.get("data", "")
                            print(f"  BCB/SGS  {label}: R$ {val:.2f}  [{data_ref}]  [série {serie}]")
                            return val
            except Exception as exc:
                print(f"  ⚠  BCB/SGS [{label}] série {serie}: {exc}")
        return None

    # ── IPEADATA API ──────────────────────────────────────────────────────────

    def _ipeadata_fetch(
        self,
        series_list: list[str],
        label: str,
        expected_range: tuple[float, float],
    ) -> Optional[float]:
        if not _HAS_REQUESTS:
            return None
        for series in series_list:
            url = (
                f"{_IPEA_BASE}/ValoresSerie(SERCODIGO='{series}')"
                f"?$top=3&$orderby=VALDATA desc&$select=VALDATA,VALVALOR"
            )
            try:
                resp = _requests.get(url, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json().get("value", [])
                for entry in data:
                    val = entry.get("VALVALOR")
                    if val is not None:
                        val = float(val)
                        lo, hi = expected_range
                        if lo <= val <= hi:
                            data_ref = entry.get("VALDATA", "")[:10]
                            print(f"  IPEADATA {label}: R$ {val:.2f}  [{data_ref}]  [série {series}]")
                            return val
            except Exception as exc:
                print(f"  ⚠  IPEADATA [{label}] série {series}: {exc}")
        return None
