"""
IATF Economic Simulator
───────────────────────
Compare conventional AI (artificial insemination) vs IATF (timed AI with
synchronisation protocol) protocols for bovine reproduction.

Provides side-by-side economic scenarios to help practitioners decide
whether to invest in hormonal synchronisation protocols.
"""

from __future__ import annotations

from typing import Any


# ── Default parameters ─────────────────────────────────────────────────────────

CONVENTIONAL_DEFAULTS: dict[str, float] = {
    "conception_rate": 0.55,       # proportion of cows that conceive per service
    "services_per_conception": 1.8, # expected services until conception
    "dose_cost": 45.0,             # R$ per semen dose
    "labor_per_service": 20.0,     # R$ labour cost per AI procedure
    "detection_rate": 0.70,        # estrus detection rate (heat detection)
    "sync_cost": 0.0,              # no synchronisation protocol
}

IATF_DEFAULTS: dict[str, float] = {
    "conception_rate": 0.58,        # slightly higher with good IATF protocol
    "services_per_conception": 1.0, # all services are timed; 1 per cycle
    "dose_cost": 55.0,             # better-quality semen used
    "labor_per_service": 20.0,
    "detection_rate": 1.0,         # 100% submission rate — hallmark of IATF
    "protocol_hormone_cost": 85.0, # GnRH + PGF2α + GnRH kit cost
    "sync_cost": 85.0,
}

# National benchmark reference values (Embrapa / CNA / ABIEC 2023)
BENCHMARKS: dict[str, Any] = {
    "taxa_concepcao_meta_embrapa": 0.60,
    "taxa_concepcao_media_nacional": 0.55,
    "spc_ideal": 1.0,
    "spc_aceitavel": 1.5,
    "spc_alerta": 2.0,
    "dias_abertos_meta_cna": 110,
    "dias_abertos_media_nacional": 145,
    "iep_ideal_dias": 365,
    "iep_media_nacional_dias": 430,
    "ecc_ideal_ao_servico": 3.0,
    "custo_dia_aberto_corte_rs": 6.0,
    "custo_dia_aberto_leite_rs": 10.0,
    "valor_bezerro_referencia_rs": 1_800.0,
}


# ── Main simulator class ───────────────────────────────────────────────────────

class IATFSimulator:
    """
    Simulates economic outcomes for conventional vs IATF AI protocols.

    Parameters
    ----------
    herd_size : int
        Number of cows to be served in the breeding season.
    base_conception_rate : float
        Baseline conception rate for the herd (0–1).  Used as a starting
        point when no protocol-specific rate is provided.
    calf_value : float
        Market value of a weaned calf (R$).  Default: R$ 1,800.
    days_open_cost : float
        Daily cost of a cow being open / not pregnant (R$).  Default: R$ 6.
    """

    def __init__(
        self,
        herd_size: int = 500,
        base_conception_rate: float = 0.55,
        calf_value: float = 1_800.0,
        days_open_cost: float = 6.0,
    ) -> None:
        self.herd_size = herd_size
        self.base_conception_rate = base_conception_rate
        self.calf_value = calf_value
        self.days_open_cost = days_open_cost

    # ── Scenario simulation ────────────────────────────────────────────────────

    def simulate_conventional(
        self, params: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Simulate conventional (natural-heat detection) AI protocol.

        Returns a dict with economic summary.
        """
        p = {**CONVENTIONAL_DEFAULTS, **(params or {})}
        return self._compute_scenario(p, protocol_name="Convencional")

    def simulate_iatf(
        self, params: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Simulate IATF (timed AI with synchronisation) protocol.

        Returns a dict with economic summary.
        """
        p = {**IATF_DEFAULTS, **(params or {})}
        return self._compute_scenario(p, protocol_name="IATF")

    def compare(
        self,
        conventional_params: dict[str, float] | None = None,
        iatf_params: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Side-by-side comparison of conventional vs IATF protocols.
        """
        conv = self.simulate_conventional(conventional_params)
        iatf = self.simulate_iatf(iatf_params)

        diff_pregnancies = iatf["expected_pregnancies"] - conv["expected_pregnancies"]
        diff_cost_per_pregnancy = iatf["cost_per_pregnancy"] - conv["cost_per_pregnancy"]
        diff_net_revenue = iatf["net_revenue"] - conv["net_revenue"]
        diff_roi_pct = iatf["roi_pct"] - conv["roi_pct"]

        recommendation = self._recommendation(conv, iatf)

        return {
            "convencional": conv,
            "iatf": iatf,
            "diferenca": {
                "pregnancies_gained": round(diff_pregnancies, 1),
                "cost_per_pregnancy_diff_brl": round(diff_cost_per_pregnancy, 2),
                "net_revenue_diff_brl": round(diff_net_revenue, 2),
                "roi_pct_diff": round(diff_roi_pct, 1),
            },
            "recommendation": recommendation,
        }

    def roi_analysis(
        self,
        conventional_params: dict[str, float] | None = None,
        iatf_params: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """
        Detailed ROI breakdown for the IATF vs conventional decision.
        """
        comparison = self.compare(conventional_params, iatf_params)
        conv = comparison["convencional"]
        iatf = comparison["iatf"]

        incremental_investment = iatf["total_cost"] - conv["total_cost"]
        incremental_revenue = iatf["net_revenue"] - conv["net_revenue"]
        incremental_roi = (
            incremental_revenue / incremental_investment * 100
            if incremental_investment != 0
            else 0.0
        )
        payback_cycles = (
            incremental_investment / max(incremental_revenue, 0.01)
            if incremental_revenue > 0
            else float("inf")
        )

        return {
            "incremental_investment_brl": round(incremental_investment, 2),
            "incremental_revenue_brl": round(incremental_revenue, 2),
            "incremental_roi_pct": round(incremental_roi, 1),
            "payback_cycles": round(payback_cycles, 2),
            "iatf_total_cost": iatf["total_cost"],
            "conv_total_cost": conv["total_cost"],
            "iatf_pregnancies": iatf["expected_pregnancies"],
            "conv_pregnancies": conv["expected_pregnancies"],
            "comparison": comparison,
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _compute_scenario(
        self, params: dict[str, float], protocol_name: str
    ) -> dict[str, Any]:
        herd = self.herd_size
        cr = float(params.get("conception_rate", self.base_conception_rate))
        spc = float(params.get("services_per_conception", 1.8))
        dose_cost = float(params.get("dose_cost", 45.0))
        labor = float(params.get("labor_per_service", 20.0))
        sync_cost = float(params.get("sync_cost", 0.0))
        detection_rate = float(params.get("detection_rate", 0.70))

        # Effective submission rate adjusts expected pregnancies for conventional
        # (IATF has 100% submission via sync; conventional limited by heat detection)
        effective_cr = cr * detection_rate if protocol_name == "Convencional" else cr

        expected_pregnancies = herd * effective_cr
        cost_per_animal = (dose_cost + labor + sync_cost) * spc
        total_cost = cost_per_animal * herd
        cost_per_pregnancy = (
            total_cost / expected_pregnancies if expected_pregnancies > 0 else 0.0
        )
        revenue = expected_pregnancies * self.calf_value
        net_revenue = revenue - total_cost
        roi_pct = net_revenue / total_cost * 100 if total_cost > 0 else 0.0

        return {
            "protocol": protocol_name,
            "herd_size": herd,
            "conception_rate": round(cr, 4),
            "effective_conception_rate": round(effective_cr, 4),
            "expected_pregnancies": round(expected_pregnancies, 1),
            "cost_per_animal_brl": round(cost_per_animal, 2),
            "total_cost_brl": round(total_cost, 2),
            "total_cost": round(total_cost, 2),
            "cost_per_pregnancy": round(cost_per_pregnancy, 2),
            "revenue_brl": round(revenue, 2),
            "net_revenue": round(net_revenue, 2),
            "roi_pct": round(roi_pct, 1),
            "calf_value_used": self.calf_value,
        }

    @staticmethod
    def _recommendation(
        conv: dict[str, Any], iatf: dict[str, Any]
    ) -> dict[str, str]:
        if iatf["net_revenue"] > conv["net_revenue"]:
            level = "positive"
            text = (
                "IATF apresenta maior retorno líquido estimado. "
                f"Ganho de R$ {iatf['net_revenue'] - conv['net_revenue']:,.0f} "
                "sobre o protocolo convencional."
            )
        elif iatf["cost_per_pregnancy"] < conv["cost_per_pregnancy"]:
            level = "neutral"
            text = (
                "IATF tem custo por prenhez menor, mas diferença no retorno líquido "
                "é pequena. Avaliar manejo operacional."
            )
        else:
            level = "caution"
            text = (
                "Protocolo convencional apresenta melhor relação custo-benefício "
                "para este rebanho. Revisar taxa de detecção de cio antes de migrar para IATF."
            )
        return {"level": level, "text": text}


# ── Standalone scenarios function ─────────────────────────────────────────────

def simulate_scenarios(
    herd_size: int = 500,
    ecc_mean: float = 2.9,
    days_open: float = 130.0,
    calf_value: float = 1_800.0,
    days_open_cost: float = 6.0,
) -> dict[str, Any]:
    """
    Generate multiple comparative scenarios for use in the Streamlit dashboard.

    Adjusts conception rates based on herd ECC (body condition score) and
    average days open, producing realistic scenario tables.

    Parameters
    ----------
    herd_size : int
    ecc_mean : float    Body Condition Score mean (1–5)
    days_open : float   Average days open
    calf_value : float  R$ per weaned calf
    days_open_cost : float  R$ per cow per open day

    Returns
    -------
    dict with keys "scenarios", "comparison", "roi_analysis"
    """
    # ECC adjustment: each point above 2.5 adds ~5 pp to conception rate
    ecc_adj = (ecc_mean - 2.5) * 0.05
    # Days open adjustment: very high days open signals poor conception, subtract
    da_adj = max(0.0, (days_open - 130) / 130) * -0.05

    base_cr = max(0.25, min(0.85, 0.55 + ecc_adj + da_adj))
    iatf_cr = min(0.85, base_cr + 0.05)  # IATF typically adds ~5 pp

    sim = IATFSimulator(
        herd_size=herd_size,
        base_conception_rate=base_cr,
        calf_value=calf_value,
        days_open_cost=days_open_cost,
    )

    conv_params = {**CONVENTIONAL_DEFAULTS, "conception_rate": base_cr}
    iatf_params = {**IATF_DEFAULTS, "conception_rate": iatf_cr}

    comparison = sim.compare(conv_params, iatf_params)
    roi = sim.roi_analysis(conv_params, iatf_params)

    # Days-open economic loss
    excess_days = max(0.0, days_open - BENCHMARKS["dias_abertos_meta_cna"])
    days_open_loss = excess_days * days_open_cost * herd_size

    # Potential gain if conception improves by 10 pp
    improved_pregnancies = herd_size * min(0.85, base_cr + 0.10)
    current_pregnancies = herd_size * base_cr
    potential_gain = (improved_pregnancies - current_pregnancies) * calf_value

    # Cost per pregnancy summary
    cost_per_pregnancy_conv = comparison["convencional"]["cost_per_pregnancy"]
    cost_per_pregnancy_iatf = comparison["iatf"]["cost_per_pregnancy"]

    scenarios = {
        "current": {
            "label": "Situação Atual",
            "herd_size": herd_size,
            "conception_rate": round(base_cr, 4),
            "expected_pregnancies": round(current_pregnancies, 1),
            "days_open_loss_brl": round(days_open_loss, 2),
            "cost_per_pregnancy_brl": round(cost_per_pregnancy_conv, 2),
        },
        "improved_conventional": {
            "label": "Convencional Melhorado (+10 pp)",
            "herd_size": herd_size,
            "conception_rate": round(min(0.85, base_cr + 0.10), 4),
            "expected_pregnancies": round(improved_pregnancies, 1),
            "days_open_loss_brl": round(days_open_loss * 0.6, 2),
            "cost_per_pregnancy_brl": round(cost_per_pregnancy_conv, 2),
        },
        "iatf": {
            "label": "IATF (Sincronização)",
            "herd_size": herd_size,
            "conception_rate": round(iatf_cr, 4),
            "expected_pregnancies": round(herd_size * iatf_cr, 1),
            "days_open_loss_brl": round(days_open_loss * 0.5, 2),
            "cost_per_pregnancy_brl": round(cost_per_pregnancy_iatf, 2),
        },
    }

    return {
        "scenarios": scenarios,
        "comparison": comparison,
        "roi_analysis": roi,
        "base_conception_rate": round(base_cr, 4),
        "iatf_conception_rate": round(iatf_cr, 4),
        "days_open_loss_brl": round(days_open_loss, 2),
        "potential_gain_brl": round(potential_gain, 2),
        "benchmarks": BENCHMARKS,
    }
