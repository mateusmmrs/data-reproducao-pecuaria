"""
Economic Impact Agent
──────────────────────
Translates reproductive performance metrics into financial figures:
  • Cost of a day open (dairy and beef)
  • Loss from low conception rate
  • Cost per additional service
  • Potential gains from improved reproductive efficiency
  • ROI of reproductive management interventions
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from agents.base_agent import BaseAgent
from utils.helpers import save_json, format_currency, DATA_PROCESSED, section
from utils.cepea_client import CEPEAClient


# ── Default economic parameters (configurable via context["econ_params"]) ─────

DEFAULT_PARAMS: dict[str, float] = {
    # Dairy
    "custo_dia_aberto_leite":       8.00,   # R$/day open (loss of milk + delayed calving)
    "custo_descarte_vaca":        3_500.00, # R$ cost to replace a culled cow
    "preco_leite_litro":            2.50,   # R$/L
    "producao_leite_dia":           22.0,   # L/day
    # Beef
    "custo_dia_aberto_corte":       5.00,   # R$/day open (maintenance cost + delayed weaning)
    "valor_bezerro":              1_500.00, # R$ value of a weaned calf
    "peso_arroba_boi":             350.00,  # R$/@
    # Insemination
    "custo_ia":                    65.00,   # R$ per insemination (semen + labor)
    "custo_dose_semen":            45.00,   # R$ per dose
    "custo_tecnico_ia":            20.00,   # R$ per IA procedure
    # Benchmarks
    "iep_ideal_dias":              365.0,   # days — ideal calving interval
    "taxa_concepcao_alvo":          0.60,   # 60% target conception rate
    "dias_abertos_alvo":          110.0,    # target days open
}


class EconomicImpactAgent(BaseAgent):
    name = "EconomicImpactAgent"

    def _description(self) -> str:
        return "Estimating economic losses and gains from reproductive performance"

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = context["df_cleaned"].copy()
        kpis: dict = context.get("eda", {}).get("kpis", {})

        # Start with defaults, then layer context overrides, then live CEPEA prices
        params: dict = {**DEFAULT_PARAMS, **context.get("econ_params", {})}
        live_prices = self._fetch_live_prices(params)
        params.update(live_prices)

        eco: dict[str, Any] = {
            "params": params,
            "price_sources": {k: "CEPEA (live)" if k in live_prices else "default"
                              for k in params},
        }

        section("Days Open — Economic Loss")
        eco["days_open"] = self._days_open_loss(df, kpis, params)

        section("Conception Rate — Lost Calves / Revenue")
        eco["conception_rate"] = self._conception_rate_loss(df, kpis, params)

        section("Additional Services Cost")
        eco["services"] = self._services_cost(df, kpis, params)

        section("Calving Interval — Productivity Gap")
        eco["calving_interval"] = self._iep_gap(df, kpis, params)

        section("Scenario — Impact of Improving Conception Rate by 10 pp")
        eco["improvement_scenario"] = self._improvement_scenario(df, kpis, params)

        section("Summary")
        eco["total_annual_loss"] = self._print_summary(eco, len(df))

        save_json(eco, DATA_PROCESSED / "economic_impact.json")
        context["economic"] = eco
        return context

    # ── live price fetch ──────────────────────────────────────────────────────

    def _fetch_live_prices(self, params: dict) -> dict:
        """
        Try to fetch current commodity prices from CEPEA.
        Returns only the keys that were successfully retrieved.
        """
        section("Live Prices — CEPEA/ESALQ")
        client = CEPEAClient()
        live: dict = {}

        arroba = client.fetch_boi_gordo_arroba()
        if arroba:
            # Boi gordo arroba price → derive cost per day open (beef)
            # Rule of thumb: ~0.5% of animal value per month in maintenance
            # Average steer ~16 arrobas at slaughter
            animal_value = arroba * 16
            custo_dia = round(animal_value * 0.005 / 30, 2)
            live["peso_arroba_boi"] = round(arroba, 2)
            live["custo_dia_aberto_corte"] = custo_dia
            print(f"  Arroba boi gordo      : R$ {arroba:.2f}")
            print(f"  Custo dia aberto corte: R$ {custo_dia:.2f}  (derivado)")
        else:
            print(f"  Arroba boi gordo      : R$ {params['peso_arroba_boi']:.2f}  (padrão)")

        leite = client.fetch_leite_litro()
        if leite:
            live["preco_leite_litro"] = round(leite, 4)
            # Update dairy day-open cost: milk revenue lost per day
            producao_dia = params.get("producao_leite_dia", DEFAULT_PARAMS["producao_leite_dia"])
            custo_dia_leite = round(leite * producao_dia, 2)
            live["custo_dia_aberto_leite"] = custo_dia_leite
            print(f"  Preço leite           : R$ {leite:.4f}/L")
            print(f"  Custo dia aberto leite: R$ {custo_dia_leite:.2f}  (derivado)")
        else:
            print(f"  Preço leite           : R$ {params['preco_leite_litro']:.4f}/L  (padrão)")

        return live

    # ── calculations ──────────────────────────────────────────────────────────

    def _days_open_loss(self, df: pd.DataFrame, kpis: dict, p: dict) -> dict:
        actual_da = kpis.get("dias_abertos_medio", df["dias_abertos"].mean() if "dias_abertos" in df.columns else 130)
        target_da = p["dias_abertos_alvo"]
        excess_da = max(0, actual_da - target_da)

        n = len(df)
        # Use a weighted cost: assume 50% dairy / 50% beef if we cannot distinguish
        custo_medio_dia = (p["custo_dia_aberto_leite"] + p["custo_dia_aberto_corte"]) / 2
        loss_per_animal = excess_da * custo_medio_dia
        total_loss = loss_per_animal * n

        print(f"  Dias abertos médios   : {actual_da:.0f} d  (alvo: {target_da:.0f} d)")
        print(f"  Excesso médio         : {excess_da:.0f} d/animal")
        print(f"  Custo/dia             : {format_currency(custo_medio_dia)}")
        print(f"  Perda por animal      : {format_currency(loss_per_animal)}")
        print(f"  Perda total (rebanho) : {format_currency(total_loss)}")

        return {
            "actual_days_open": round(float(actual_da), 1),
            "target_days_open": target_da,
            "excess_days": round(float(excess_da), 1),
            "loss_per_animal_brl": round(float(loss_per_animal), 2),
            "total_loss_brl": round(float(total_loss), 2),
            "n_animals": n,
        }

    def _conception_rate_loss(self, df: pd.DataFrame, kpis: dict, p: dict) -> dict:
        actual_cr = kpis.get("taxa_concepcao_global", df["prenhe"].mean() if "prenhe" in df.columns else 0.55)
        target_cr = p["taxa_concepcao_alvo"]
        gap = max(0, target_cr - actual_cr)

        n = len(df)
        missed_pregnancies = gap * n
        # Each missed pregnancy = lost calf + extra days open
        loss_per_missed = p["valor_bezerro"] + p["dias_abertos_alvo"] * p["custo_dia_aberto_corte"]
        total_loss = missed_pregnancies * loss_per_missed

        print(f"  Taxa concepção atual  : {actual_cr:.1%}  (alvo: {target_cr:.0%})")
        print(f"  Gap                   : {gap:.1%}")
        print(f"  Prenhezes perdidas    : {missed_pregnancies:.0f}")
        print(f"  Perda/prenhez perdida : {format_currency(loss_per_missed)}")
        print(f"  Perda total           : {format_currency(total_loss)}")

        return {
            "actual_conception_rate": round(float(actual_cr), 4),
            "target_conception_rate": target_cr,
            "gap_pp": round(float(gap), 4),
            "missed_pregnancies": round(float(missed_pregnancies), 1),
            "loss_per_missed_brl": round(float(loss_per_missed), 2),
            "total_loss_brl": round(float(total_loss), 2),
        }

    def _services_cost(self, df: pd.DataFrame, kpis: dict, p: dict) -> dict:
        actual_spc = kpis.get("servicos_por_concepcao", df["servicos_concepcao"].mean() if "servicos_concepcao" in df.columns else 1.7)
        target_spc = 1.0
        excess_svc = max(0, actual_spc - target_spc)

        n = len(df)
        total_extra_doses = excess_svc * n
        cost_per_extra = p["custo_dose_semen"] + p["custo_tecnico_ia"]
        total_cost = total_extra_doses * cost_per_extra

        print(f"  SPC atual             : {actual_spc:.2f}  (ideal: {target_spc:.1f})")
        print(f"  Doses extras/animal   : {excess_svc:.2f}")
        print(f"  Custo por dose extra  : {format_currency(cost_per_extra)}")
        print(f"  Custo total doses ext.: {format_currency(total_cost)}")

        return {
            "actual_spc": round(float(actual_spc), 2),
            "target_spc": target_spc,
            "excess_services": round(float(excess_svc), 2),
            "total_extra_doses": round(float(total_extra_doses), 1),
            "cost_per_extra_brl": round(float(cost_per_extra), 2),
            "total_cost_brl": round(float(total_cost), 2),
        }

    def _iep_gap(self, df: pd.DataFrame, kpis: dict, p: dict) -> dict:
        actual_iep = kpis.get("iep_medio_dias", df["iep"].mean() if "iep" in df.columns else 420)
        ideal_iep = p["iep_ideal_dias"]
        excess = max(0, actual_iep - ideal_iep)

        # Each extra day in IEP = one fewer calf per lifetime / one more day of maintenance
        n = len(df)
        calves_lost_per_year = (excess / ideal_iep) * n
        value_lost = calves_lost_per_year * p["valor_bezerro"]

        print(f"  IEP médio             : {actual_iep:.0f} d  (ideal: {ideal_iep:.0f} d)")
        print(f"  Excesso               : {excess:.0f} d")
        print(f"  Crias perdidas (est.) : {calves_lost_per_year:.1f}/ano")
        print(f"  Valor perdido         : {format_currency(value_lost)}/ano")

        return {
            "actual_iep_days": round(float(actual_iep), 1),
            "ideal_iep_days": ideal_iep,
            "excess_days": round(float(excess), 1),
            "estimated_calves_lost_year": round(float(calves_lost_per_year), 1),
            "value_lost_brl": round(float(value_lost), 2),
        }

    def _improvement_scenario(self, df: pd.DataFrame, kpis: dict, p: dict) -> dict:
        actual_cr = kpis.get("taxa_concepcao_global", 0.55)
        improved_cr = min(1.0, actual_cr + 0.10)
        n = len(df)

        extra_pregnancies = (improved_cr - actual_cr) * n
        revenue_gain = extra_pregnancies * p["valor_bezerro"]
        days_saved = 30 * n  # rough: each extra pregnancy saves ~30 days open
        saved_days_cost = days_saved * (p["custo_dia_aberto_corte"] + p["custo_dia_aberto_leite"]) / 2
        total_gain = revenue_gain + saved_days_cost

        # Investment: reproductive management program (estimate)
        investment = n * (p["custo_ia"] + 50)  # IA + hormones + labour per head
        roi = (total_gain - investment) / investment * 100 if investment else 0

        print(f"  Cenário: concepção sobe de {actual_cr:.0%} → {improved_cr:.0%}")
        print(f"  Prenhezes adicionais  : {extra_pregnancies:.0f}")
        print(f"  Receita adicional     : {format_currency(revenue_gain)}")
        print(f"  Economia dias abertos : {format_currency(saved_days_cost)}")
        print(f"  Ganho total estimado  : {format_currency(total_gain)}")
        print(f"  Investimento estimado : {format_currency(investment)}")
        print(f"  ROI estimado          : {roi:.1f}%")

        return {
            "actual_cr": round(float(actual_cr), 4),
            "improved_cr": round(float(improved_cr), 4),
            "extra_pregnancies": round(float(extra_pregnancies), 1),
            "revenue_gain_brl": round(float(revenue_gain), 2),
            "savings_days_open_brl": round(float(saved_days_cost), 2),
            "total_gain_brl": round(float(total_gain), 2),
            "estimated_investment_brl": round(float(investment), 2),
            "roi_pct": round(float(roi), 2),
        }

    def _print_summary(self, eco: dict, n: int) -> float:
        losses = [
            eco.get("days_open", {}).get("total_loss_brl", 0),
            eco.get("conception_rate", {}).get("total_loss_brl", 0),
            eco.get("services", {}).get("total_cost_brl", 0),
            eco.get("calving_interval", {}).get("value_lost_brl", 0),
        ]
        total = sum(losses)
        per_animal = total / n if n else 0
        print(f"\n  ┌─────────────────────────────────────────────┐")
        print(f"  │  PERDA TOTAL ESTIMADA  : {format_currency(total):>18s}  │")
        print(f"  │  Por animal            : {format_currency(per_animal):>18s}  │")
        print(f"  │  Rebanho               : {n:>18,} animais  │")
        print(f"  └─────────────────────────────────────────────┘")
        return round(float(total), 2)

    def _summarize(self, context: dict[str, Any]) -> None:
        total = context.get("economic", {}).get("total_annual_loss", 0)
        print(f"\n  Total estimated annual loss: {format_currency(total)}")
